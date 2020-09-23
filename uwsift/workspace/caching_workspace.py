import argparse
import logging
import os
import shutil
import sys
import unittest
from collections import OrderedDict
from datetime import datetime
from typing import Generator, Tuple, Dict, Optional
from uuid import UUID

import numpy as np
from sqlalchemy.orm.exc import NoResultFound

from uwsift.common import Info, Kind, Flags, State
from uwsift.queue import TASK_PROGRESS, TASK_DOING
from .importer import aImporter, SatpyImporter
from .metadatabase import Metadatabase, Content, Product, Resource
from .workspace import BaseWorkspace, frozendict, ActiveContent

LOG = logging.getLogger(__name__)

DEFAULT_WORKSPACE_SIZE = 256
MIN_WORKSPACE_SIZE = 8

IMPORT_CLASSES = [SatpyImporter]

# first instance is main singleton instance; don't preclude the possibility of importing from another workspace later on
TheWorkspace = None


class CachingWorkspace(BaseWorkspace):
    """Data management and cache object.

    CachingWorkspace is a singleton object which works with Datasets shall:

    - own a working directory full of recently used datasets
    - provide DatasetInfo dictionaries for shorthand use between application subsystems

        - datasetinfo dictionaries are ordinary python dictionaries containing [Info.UUID],
          projection metadata, LOD info

    - identify datasets primarily with a UUID object which tracks the dataset and
      its various representations through the system
    - unpack data in "packing crate" formats like NetCDF into memory-compatible flat files
    - efficiently create on-demand subsections and strides of raster data as numpy arrays
    - incrementally cache often-used subsections and strides ("image pyramid") using appropriate tools like gdal
    - notify subscribers of changes to datasets (Qt signal/slot pub-sub)
    - during idle, clean out unused/idle data content, given DatasetInfo contents provides enough metadata to recreate
    - interface to external data processing or loading plug-ins and notify application of new-dataset-in-workspace

    """

    def product_state(self, uuid: UUID) -> Flags:
        state = Flags(self._state[uuid])
        # add any derived information
        if uuid in self._available:
            state.add(State.ATTACHED)
        with self._inventory as s:
            ncontent = s.query(Content).filter_by(uuid=uuid).count()
        if ncontent > 0:
            state.add(State.CACHED)
        return state

    @property
    def _S(self):
        """
        use scoped_session registry of metadatabase to provide thread-local session object.
        ref http://docs.sqlalchemy.org/en/latest/orm/contextual.html
        Returns:
        """
        return self._inventory.SessionRegistry

    @property
    def metadatabase(self) -> Metadatabase:
        return self._inventory

    def __init__(self, directory_path=None, process_pool=None, max_size_gb=None, queue=None, initial_clear=False):
        super(CachingWorkspace, self, ).__init__(directory_path)
        self._queue = queue
        self._max_size_gb = max_size_gb if max_size_gb is not None else DEFAULT_WORKSPACE_SIZE
        if self._max_size_gb < MIN_WORKSPACE_SIZE:
            self._max_size_gb = MIN_WORKSPACE_SIZE
            LOG.warning('setting workspace size to %dGB' % self._max_size_gb)
        if directory_path is None:
            import tempfile
            self._tempdir = tempfile.TemporaryDirectory()
            directory_path = str(self._tempdir)
            LOG.info('using temporary directory {}'.format(directory_path))

        self._inventory_path = os.path.join(self.cwd, '_inventory.db')
        if initial_clear:
            self.clear_workspace_content()

        else:
            LOG.info("attaching pre-existing workspace at {}".format(directory_path))
            self._own_cwd = False
            self._init_inventory_existing_datasets()

    def _init_create_workspace(self):
        """
        initialize a previously empty workspace
        :return:
        """
        should_init = not os.path.exists(self._inventory_path)
        dn, fn = os.path.split(self._inventory_path)
        if not os.path.isdir(dn):
            raise EnvironmentError("workspace directory {} does not exist".format(dn))
        LOG.info('{} database at {}'.format('initializing' if should_init else 'attaching', self._inventory_path))
        self._inventory = Metadatabase('sqlite:///' + self._inventory_path, create_tables=should_init)
        if should_init:
            with self._inventory as s:
                assert (0 == s.query(Content).count())
        LOG.info('done with init')

    def clear_workspace_content(self):
        """Remove binary files from workspace and workspace database."""
        LOG.info("Clearing workspace contents...")
        try:
            os.remove(self._inventory_path)
        except FileNotFoundError:
            LOG.debug("No inventory DB file found to remove: {}".format(self._inventory_path))

        try:
            shutil.rmtree(self.cache_dir)
        except OSError:
            LOG.debug("No binary cache directory found to remove: {}".format(self.cache_dir))

    def _purge_missing_content(self):
        """
        remove Content entries that no longer correspond to files in the cache directory
        """
        LOG.debug("purging Content no longer available in the cache")
        to_purge = []
        with self._inventory as s:
            for c in s.query(Content).all():
                if not ActiveContent.can_attach(self.cache_dir, c):
                    LOG.warning("purging missing content {}".format(c.path))
                    to_purge.append(c)
            LOG.debug(
                "{} content entities no longer present in cache - will remove from database".format(len(to_purge)))
            for c in to_purge:
                try:
                    c.product.content.remove(c)
                except AttributeError:
                    # no_product
                    LOG.warning("orphaned content {}??, removing".format(c.path))
                s.delete(c)

    def _purge_inaccessible_resources(self):
        """
        remove Resources that are no longer accessible
        """
        LOG.debug("purging any resources that are no longer accessible")
        with self._inventory as s:
            resall = list(s.query(Resource).all())
            n_purged = 0
            for r in resall:
                if not r.exists():
                    LOG.info("resource {} no longer exists, purging from database")
                    # for p in r.product:
                    #     p.resource.remove(r)
                    n_purged += 1
                    s.delete(r)
        LOG.info("discarded metadata for {} orphaned resources".format(n_purged))

    def _purge_orphan_products(self):
        """
        remove products from database that have no cached Content, and no Resource we can re-import from
        """
        LOG.debug("purging Products no longer recoverable by re-importing from Resources, "
                  "and having no Content representation in cache")
        with self._inventory as s:
            n_purged = 0
            prodall = list(s.query(Product).all())  # SIFT/sift#180, avoid locking database too long
            for p in prodall:
                if len(p.content) == 0 and len(p.resource) == 0:
                    n_purged += 1
                    s.delete(p)
        LOG.info("discarded metadata for {} orphaned products".format(n_purged))

    def _migrate_metadata(self):
        """Replace legacy metadata uses with new uses."""
        # with self._inventory as s:
        #     for p in s.query(Product).all():
        #         pass

    def _bgnd_startup_purge(self):
        ntot = 5
        n = 1
        yield {TASK_DOING: "DB pruning cache entries".format(n, ntot), TASK_PROGRESS: float(n) / float(ntot)}
        self._purge_missing_content()
        n += 1
        yield {TASK_DOING: "DB pruning stale resources".format(n, ntot), TASK_PROGRESS: float(n) / float(ntot)}
        self._purge_inaccessible_resources()
        n += 1
        yield {TASK_DOING: "DB pruning orphan products".format(n, ntot), TASK_PROGRESS: float(n) / float(ntot)}
        self._purge_orphan_products()
        n += 1
        yield {TASK_DOING: "DB migrating metadata".format(n, ntot), TASK_PROGRESS: float(n) / float(ntot)}
        self._migrate_metadata()
        n += 1
        yield {TASK_DOING: "DB ready".format(n, ntot), TASK_PROGRESS: float(n) / float(ntot)}

    def _then_refresh_mdb_customers(self, *args, **kwargs):
        self.didUpdateProductsMetadata.emit(set())

    def _init_inventory_existing_datasets(self):
        """
        Do an inventory of an pre-existing workspace
        FIXME: go through and check that everything in the workspace makes sense
        FIXME: check workspace subdirectories for helper sockets and mmaps
        :return:
        """
        # attach the database, creating it if needed
        self._init_create_workspace()
        for _ in self._bgnd_startup_purge():
            # SIFT/sift#180 -- background thread of lengthy database operations can cause lock failure in pysqlite
            pass
        # self._queue.add("database cleanup", self._bgnd_startup_purge(), "database cleanup",
        #           interactive=False, and_then=self._then_refresh_mdb_customers)

    def _store_inventory(self):
        """
        write inventory dictionary to an inventory.pkl file in the cwd
        :return:
        """
        self._S.commit()

    #
    #  data array handling
    #

    def _remove_content_files_from_workspace(self, c: Content):
        total = 0
        for filename in [c.path, c.coverage_path, c.sparsity_path]:
            if not filename:
                continue
            pn = os.path.join(self.cache_dir, filename)
            if os.path.exists(pn):
                LOG.debug('removing {}'.format(pn))
                total += os.stat(pn).st_size
                try:
                    os.remove(pn)
                except FileNotFoundError:
                    LOG.warning("could not remove {} - file not found; continuing".format(pn))

        return total

    def _activate_content(self, c: Content) -> ActiveContent:
        self._available[c.id] = zult = ActiveContent(self.cache_dir, c)
        c.touch()
        c.product.touch()
        return zult

    def _cached_arrays_for_content(self, c: Content):
        """
        attach cached data indicated in Content, unless it's been attached already and is in _available
        touch the content and product in the database to appease the LRU gods
        :param c: metadatabase Content object for session attached to current thread
        :return: workspace_content_arrays
        """
        cache_entry = self._available.get(c.id)
        return cache_entry or self._activate_content(c)

    #
    # often-used queries
    #

    def _product_with_uuid(self, session, uuid: UUID) -> Product:
        return session.query(Product).filter_by(uuid_str=str(uuid)).first()

    def _product_overview_content(self, session, prod: Product = None, uuid: UUID = None,
                                  kind: Kind = Kind.IMAGE) -> Optional[Content]:
        if prod is None and uuid is not None:
            # Get Product object
            try:
                prod = session.query(Product).filter(Product.uuid_str == str(uuid)).one()
            except NoResultFound:
                LOG.error("No product with UUID {} found".format(uuid))
                return None
        contents = session.query(Content).filter(Content.product_id == prod.id).order_by(Content.lod).all()
        contents = [c for c in contents if c.info.get(Info.KIND, Kind.IMAGE) == kind]
        return None if 0 == len(contents) else contents[0]

    def _product_native_content(self, session, prod: Product = None, uuid: UUID = None,
                                kind: Kind = Kind.IMAGE) -> Optional[Content]:
        # NOTE: This assumes the last Content object is the best resolution
        #       but it is untested
        if prod is None and uuid is not None:
            # Get Product object
            try:
                prod = session.query(Product).filter(Product.uuid_str == str(uuid)).one()
            except NoResultFound:
                LOG.error("No product with UUID {} found".format(uuid))
                return None
        contents = session.query(Content).filter(Content.product_id == prod.id).order_by(Content.lod.desc()).all()
        contents = [c for c in contents if c.info.get(Info.KIND, Kind.IMAGE) == kind]
        return None if 0 == len(contents) else contents[-1]

    #
    # combining queries with data content
    #

    def _overview_content_for_uuid(self, uuid: UUID, kind: Kind = Kind.IMAGE) \
            -> np.memmap:
        # FUTURE: do a compound query for this to get the Content entry
        # prod = self._product_with_uuid(uuid)
        # assert(prod is not None)
        with self._inventory as s:
            ovc = self._product_overview_content(s, uuid=uuid)
            assert (ovc is not None)
            arrays = self._cached_arrays_for_content(ovc)
            return arrays.data

    def _native_content_for_uuid(self, uuid: UUID) -> np.memmap:
        # FUTURE: do a compound query for this to get the Content entry
        # prod = self._product_with_uuid(uuid)
        with self._inventory as s:
            nac = self._product_native_content(s, uuid=uuid)
            arrays = self._cached_arrays_for_content(nac)
            return arrays.data

    #
    # workspace file management
    #

    @property
    def _total_workspace_bytes(self):
        """
        total number of bytes in the workspace by brute force instead of metadata search
        :return:
        """
        total = 0
        for root, dirs, files in os.walk(self.cache_dir):
            sz = sum(os.path.getsize(os.path.join(root, name)) for name in files)
            total += sz
            LOG.debug('%d bytes in %s' % (sz, root))

        return total

    def _all_product_uuids(self) -> list:
        with self._inventory as s:
            return [q.uuid for q in s.query(Product).all()]

    # ----------------------------------------------------------------------
    def get_info(self, dsi_or_uuid, lod=None) -> Optional[frozendict]:
        """
        :param dsi_or_uuid: existing datasetinfo dictionary, or its UUID
        :param lod: desired level of detail to focus
        :return: metadata access with mapping semantics, to be treated as read-only
        """
        from collections import ChainMap
        # FUTURE deprecate this
        if isinstance(dsi_or_uuid, str):
            uuid = UUID(dsi_or_uuid)
        elif not isinstance(dsi_or_uuid, UUID):
            uuid = dsi_or_uuid[Info.UUID]
        else:
            uuid = dsi_or_uuid
        with self._inventory as s:
            # look up the product for that uuid
            prod = self._product_with_uuid(s, uuid)
            if not prod:  # then it hasn't had its metadata scraped
                LOG.error('no info available for UUID {}'.format(dsi_or_uuid))
                LOG.error("known products: {}".format(repr(self._all_product_uuids())))
                return None
            kind = prod.info[Info.KIND]
            native_content = self._product_native_content(s, prod=prod, kind=kind)

            if native_content is not None:
                # FUTURE: this is especially saddening; upgrade to finer grained query and/or deprecate .get_info
                # once upon a time...
                # our old model was that product == content and shares a UUID with the layer
                # if content is available, we want to provide native content metadata along with the product metadata
                # specifically a lot of client code assumes that resource == product == content and
                # that singular navigation (e.g. cell_size) is norm
                assert (native_content.info[Info.CELL_WIDTH] is not None)  # FIXME DEBUG
                return frozendict(ChainMap(native_content.info, prod.info))
            # mapping semantics for database fields, as well as key-value fields;
            # flatten to one namespace and read-only
            return frozendict(prod.info)

    def get_algebraic_namespace(self, uuid: UUID):
        if uuid is None:
            return {}, ""

        with self._inventory as s:
            prod = self._product_with_uuid(s, uuid)
            if prod is None:
                return {}, ""
            symbols = {x.key: x.value for x in prod.symbol}
            code = prod.expression
        return symbols, code

    def _check_cache(self, path: str):
        """
        FIXME: does not work if more than one product inside a path
        :param path: file we're checking
        :return: uuid, info, overview_content if the data is already available without import
        """
        with self._inventory as s:
            hits = s.query(Resource).filter_by(path=path).all()
            if not hits:
                return None
            if len(hits) >= 1:
                if len(hits) > 1:
                    LOG.warning('more than one Resource found suitable, there can be only one')
                resource = hits[0]
                hits = list(s.query(Content).filter(
                    Content.product_id == Product.id).filter(
                    Product.resource_id == resource.id).order_by(
                    Content.lod).all())
                if len(hits) >= 1:
                    content = hits[0]  # presumably this is closest to LOD_OVERVIEW
                    # if len(hits)>1:
                    #     LOG.warning('more than one Content found suitable, there can be only one')
                    cac = self._cached_arrays_for_content(content)
                    if not cac:
                        LOG.error('unable to attach content')
                        data = None
                    else:
                        data = cac.data
                    return content.product.uuid, content.product.info, data

    @property
    def product_names_available_in_cache(self) -> dict:
        """
        Returns: dictionary of {UUID: product name,...}
        typically used for add-from-cache dialog
        """
        # find non-overview non-auxiliary data files
        # FIXME: also need to include coverage and sparsity paths?? really?
        zult = {}
        with self._inventory as s:
            for c in s.query(Content).order_by(Content.atime.desc()).all():
                p = c.product
                if p.uuid not in zult:
                    zult[p.uuid] = p.info[Info.DISPLAY_NAME]
        return zult

    @property
    def uuids_in_cache(self):
        with self._inventory as s:
            contents_of_cache = s.query(Content).all()
            return list(sorted(set(c.product.uuid for c in contents_of_cache)))

    def recently_used_products(self, n=32) -> Dict[UUID, str]:
        with self._inventory as s:
            return OrderedDict((p.uuid, p.info[Info.DISPLAY_NAME])
                               for p in s.query(Product).order_by(Product.atime.desc()).limit(n).all())

    def _purge_content_for_resource(self, resource: Resource, session, defer_commit=False):
        """
        remove all resource contents from the database
        if the resource original path no longer exists, also purge resource and products from database
        :param resource: resource object we
        :return: number of bytes freed from the workspace
        """
        if session is not None:
            defer_commit = True
        S = session  # or self._S
        total = 0
        for prod in resource.product:
            for con in prod.content:
                total += self._remove_content_files_from_workspace(con)
                S.delete(con)

        if not resource.exists():  # then purge the resource and its products as well
            S.delete(resource)
        if not defer_commit:
            S.commit()
        return total

    def remove_all_workspace_content_for_resource_paths(self, paths: list):
        total = 0
        with self._inventory as s:
            for path in paths:
                rsr_hits = s.query(Resource).filter_by(path=path).all()
                for rsr in rsr_hits:
                    total += self._purge_content_for_resource(rsr, defer_commit=True)
        return total

    def purge_content_for_product_uuids(self, uuids: list, also_products=False):
        """
        given one or more product uuids, purge the Content from the cache
        Note: this does not purge any ActiveContent that may still be using the files, but the files will be gone
        Args:
            uuids:

        Returns:

        """
        total = 0
        for uuid in uuids:
            with self._inventory as s:
                prod = s.query(Product).filter_by(uuid_str=str(uuid)).first()
                conterminate = list(prod.content)
                for con in conterminate:
                    if con.id in self._available:
                        LOG.warning("will not purge active content!")
                        continue
                    total += self._remove_content_files_from_workspace(con)
                    prod.content.remove(con)
                    s.delete(con)
                if also_products:
                    s.delete(prod)
        return total

    def _clean_cache(self):
        """
        find stale content in the cache and get rid of it
        this routine should eventually be compatible with backgrounding on a thread
        possibly include a workspace setting for max workspace size in bytes?
        :return:
        """
        # get information on current cache contents
        with self._inventory as S:
            LOG.info("cleaning cache")
            total_size = self._total_workspace_bytes
            GB = 1024 ** 3
            LOG.info("total cache size is {}GB of max {}GB".format(total_size / GB, self._max_size_gb))
            max_size = self._max_size_gb * GB
            for res in S.query(Resource).order_by(Resource.atime).all():
                if total_size < max_size:
                    break
                total_size -= self._purge_content_for_resource(res, session=S)
                # remove all content for lowest atimes until

    def close(self):
        self._clean_cache()
        # self._S.commit()

    def bgnd_task_complete(self):
        """
        handle operations that should be done at the end of a threaded background task
        """
        pass
        # self._S.commit()
        # self._S.remove()

    def get_metadata(self, uuid_or_path):
        """
        return metadata dictionary for a given product or the product being offered by a resource path (see get_info)
        Args:
            uuid_or_path: product uuid, or path to the resource path it lives in

        Returns:
            metadata (Mapping), metadata for the product at this path;
            FUTURE note more than one product may be in a single file
        """
        if isinstance(uuid_or_path, UUID):
            return self.get_info(uuid_or_path)  # get product metadata
        else:
            with self._inventory as s:
                hits = list(s.query(Resource).filter_by(path=uuid_or_path).all())
                if not hits:
                    return None
                if len(hits) >= 1:
                    if len(hits) > 1:
                        raise EnvironmentError('more than one Resource fits this path')
                    resource = hits[0]
                    if len(resource.product) >= 1:
                        if len(resource.product) > 1:
                            LOG.warning('more than one Product in this Resource, this query should be deprecated')
                        prod = resource.product[0]
                        return prod.info

    def collect_product_metadata_for_paths(self, paths: list,
                                           **importer_kwargs) -> Generator[Tuple[int, frozendict], None, None]:
        """Start loading URI data into the workspace asynchronously.

        Args:
            paths (list): String paths to open and get metadata for
            **importer_kwargs: Keyword arguments to pass to the lower-level
                importer class.

        Returns: sequence of read-only info dictionaries

        """
        with self._inventory as import_session:
            # FUTURE: consider returning importers instead of products,
            # since we can then re-use them to import the content instead of having to regenerate
            # import_session = self._S
            importers = []
            num_products = 0
            remaining_paths = []
            if 'reader' in importer_kwargs:
                # skip importer guessing and go straight to satpy importer
                paths, remaining_paths = [], paths

            for source_path in paths:
                # LOG.info('collecting metadata for {}'.format(source_path))
                # FIXME: Check if importer only accepts one path at a time
                #        Maybe sort importers by single files versus multiple files and doing single files first?
                # FIXME: decide whether to update database if mtime of file is newer than mtime in database
                # Collect all the importers we are going to use and count
                # how many products each expects to return
                for imp in self._importers:
                    if imp.is_relevant(source_path=source_path):
                        hauler = imp(source_path,
                                     database_session=import_session,
                                     workspace_cwd=self.cache_dir,
                                     **importer_kwargs)
                        hauler.merge_resources()
                        importers.append(hauler)
                        num_products += hauler.num_products
                        break
                else:
                    remaining_paths.append(source_path)

            # Pass remaining paths to SatPy importer and see what happens
            if remaining_paths:
                if 'reader' not in importer_kwargs:
                    raise NotImplementedError("Reader discovery is not "
                                              "currently implemented in "
                                              "the satpy importer.")
                if 'scenes' in importer_kwargs:
                    # another component already created the satpy scenes, use those
                    scenes = importer_kwargs.pop('scenes')
                    scenes = scenes.items()
                else:
                    scenes = [(remaining_paths, None)]
                for paths, scene in scenes:
                    imp = SatpyImporter
                    these_kwargs = importer_kwargs.copy()
                    these_kwargs['scene'] = scene
                    hauler = imp(paths,
                                 database_session=import_session,
                                 workspace_cwd=self.cache_dir,
                                 **these_kwargs)
                    hauler.merge_resources()
                    importers.append(hauler)
                    num_products += hauler.num_products

            for hauler in importers:
                for prod in hauler.merge_products():
                    assert (prod is not None)
                    # merge the product into our database session, since it may belong to import_session
                    zult = frozendict(prod.info)  # self._S.merge(prod)
                    # LOG.debug('yielding product metadata for {}'.format(
                    #     zult.get(Info.DISPLAY_NAME, '?? unknown name ??')))
                    yield num_products, zult

    def import_product_content(self, uuid: UUID = None, prod: Product = None,
                               allow_cache=True,
                               **importer_kwargs) -> np.memmap:
        with self._inventory as S:
            # S = self._S
            if prod is None and uuid is not None:
                prod = self._product_with_uuid(S, uuid)

            self.set_product_state_flag(prod.uuid, State.ARRIVING)
            default_prod_kind = prod.info[Info.KIND]

            if len(prod.content):
                LOG.info('product already has content available, using that rather than re-importing')
                ovc = self._product_overview_content(S, uuid=uuid, kind=default_prod_kind)
                assert (ovc is not None)
                arrays = self._cached_arrays_for_content(ovc)
                return arrays.data

            truck = aImporter.from_product(prod, workspace_cwd=self.cache_dir, database_session=S, **importer_kwargs)
            metadata = prod.info
            name = metadata[Info.SHORT_NAME]

            # FIXME: for now, just iterate the incremental load.
            #  later we want to add this to TheQueue and update the UI as we get more data loaded
            gen = truck.begin_import_products(prod.id)
            nupd = 0
            for update in gen:
                nupd += 1
                # we're now incrementally reading the input file
                # data updates are coming back to us (eventually asynchronously)
                # Content is in the metadatabase and being updated + committed, including sparsity and coverage arrays
                if update.data is not None:
                    # data = update.data
                    LOG.info("{} {}: {:.01f}%".format(name, update.stage_desc, update.completion * 100.0))
            # self._data[uuid] = data = self._convert_to_memmap(str(uuid), data)
            LOG.debug('received {} updates during import'.format(nupd))
            uuid = prod.uuid
            self.clear_product_state_flag(prod.uuid, State.ARRIVING)
        # S.commit()
        # S.flush()

        # make an ActiveContent object from the Content, now that we've imported it
        ac = self._overview_content_for_uuid(uuid, kind=default_prod_kind)
        if ac is None:
            return None
        return ac.data

    # def _preferred_cache_path(self, uuid):
    #     filename = str(uuid)
    #     return self._ws_path(filename)
    #
    # def _convert_to_memmap(self, uuid, data:np.ndarray):
    #     if isinstance(data, np.memmap):
    #         return data
    #     # from tempfile import TemporaryFile
    #     # fp = TemporaryFile()
    #     pathname = self._preferred_cache_path(uuid)
    #     fp = open(pathname, 'wb+')
    #     mm = np.memmap(fp, dtype=data.dtype, shape=data.shape, mode='w+')
    #     mm[:] = data[:]
    #     return mm

    def _create_product_from_array(self, info: Info, data, namespace=None, codeblock=None) \
            -> Tuple[UUID, Optional[frozendict], np.memmap]:
        """
        update metadatabase to include Product and Content entries for this new dataset we've calculated
        this allows the calculated data to reside in the workspace
        then return the "official" versions consistent with workspace product/content database
        Args:
            info: mapping of key-value metadata for new product
            data: ndarray with content to store, typically 2D float32
            namespace: {variable: uuid, } for calculation of this data
            codeblock: text, code to run to recalculate this data within namespace

        Returns:
            uuid, info, data: uuid of the new product, its official read-only metadata, and cached content ndarray
        """
        if Info.UUID not in info:
            raise ValueError('currently require an Info.UUID be included in product')
        parms = dict(info)
        now = datetime.utcnow()
        parms.update(dict(
            atime=now,
            mtime=now,
        ))
        P = Product.from_info(parms, symbols=namespace, codeblock=codeblock)
        uuid = P.uuid
        # FUTURE: add expression and namespace information, which would require additional parameters
        ws_filename = '{}.image'.format(str(uuid))
        ws_path = os.path.join(self.cache_dir, ws_filename)
        with open(ws_path, 'wb+') as fp:
            mm = np.memmap(fp, dtype=data.dtype, shape=data.shape, mode='w+')
            mm[:] = data[:]

        parms.update(dict(
            lod=Content.LOD_OVERVIEW,
            path=ws_filename,
            dtype=str(data.dtype),
            proj4=info[Info.PROJ],
            resolution=min(info[Info.CELL_WIDTH], info[Info.CELL_HEIGHT])
        ))
        rcls = dict(zip(('rows', 'cols', 'levels'), data.shape))
        parms.update(rcls)
        LOG.debug("about to create Content with this: {}".format(repr(parms)))

        C = Content.from_info(parms, only_fields=True)
        P.content.append(C)
        # FUTURE: do we identify a Resource to go with this? Probably not

        # transaction with the metadatabase to add the product and content
        with self._inventory as S:
            S.add(P)
            S.add(C)

        # FIXME: Do I have to flush the session so the Product gets added for sure?

        # activate the content we just loaded into the workspace
        overview_data = self._overview_content_for_uuid(uuid)
        # prod = self._product_with_uuid(S, uuid)
        return uuid, self.get_info(uuid), overview_data

    def _bgnd_remove(self, uuid: UUID):
        from uwsift.queue import TASK_DOING, TASK_PROGRESS
        yield {TASK_DOING: 'purging memory', TASK_PROGRESS: 0.5}
        with self._inventory as s:
            self._deactivate_content_for_product(self._product_with_uuid(s, uuid))
        yield {TASK_DOING: 'purging memory', TASK_PROGRESS: 1.0}

    def get_content(self, dsi_or_uuid, lod=None, kind: Kind = Kind.IMAGE) \
            -> Optional[np.memmap]:
        """
        By default, get the best-available (closest to native) np.ndarray-compatible view of the full dataset
        :param dsi_or_uuid: existing datasetinfo dictionary, or its UUID
        :param lod: desired level of detail to focus  (0 for overview)
        :return:
        """
        if dsi_or_uuid is None:
            return None
        elif isinstance(dsi_or_uuid, UUID):
            uuid = dsi_or_uuid
        elif isinstance(dsi_or_uuid, str):
            uuid = UUID(dsi_or_uuid)
        else:
            uuid = dsi_or_uuid[Info.UUID]
        # prod = self._product_with_uuid(dsi_or_uuid)
        # TODO: this causes a locking exception when run in a secondary thread.
        #  Keeping background operations lightweight makes sense however, so just review this
        # prod.touch()
        with self._inventory as s:
            content = s.query(Content).filter(
                (Product.uuid_str == str(uuid)) & (Content.product_id == Product.id)).order_by(Content.lod.desc()).all()
            content = [x for x in content if x.info.get(Info.KIND, Kind.IMAGE) == kind]
            if len(content) != 1:
                LOG.warning("More than one matching Content object for '{}'".format(dsi_or_uuid))
            if not len(content) or not content[0]:
                raise AssertionError('no content in workspace for {}, must re-import'.format(uuid))
            content = content[0]
            # content.touch()
            # self._S.commit()  # flush any pending updates to workspace db file

            # FIXME: find the content for the requested LOD, then return its ActiveContent - or attach one
            # for now, just work with assumption of one product one content
            active_content = self._cached_arrays_for_content(content)
            return active_content.data

    # NOTE: when using this function in future, decide whether data flipping needs to be considered
    # def _create_position_to_index_transform(self, dsi_or_uuid):
    #     info = self.get_info(dsi_or_uuid)
    #     origin_x = info[Info.ORIGIN_X]
    #     origin_y = info[Info.ORIGIN_Y]
    #     cell_width = info[Info.CELL_WIDTH]
    #     cell_height = info[Info.CELL_HEIGHT]
    #
    #     def _transform(x, y, origin_x=origin_x, origin_y=origin_y, cell_width=cell_width, cell_height=cell_height):
    #         col = (x - info[Info.ORIGIN_X]) / info[Info.CELL_WIDTH]
    #         row = (y - info[Info.ORIGIN_Y]) / info[Info.CELL_HEIGHT]
    #         return col, row
    #
    #     return _transform



def main():
    parser = argparse.ArgumentParser(
        description="PURPOSE",
        epilog="",
        fromfile_prefix_chars='@')
    parser.add_argument('-v', '--verbose', dest='verbosity', action="count", default=0,
                        help='each occurrence increases verbosity 1 level through ERROR-WARNING-Info-DEBUG')
    # http://docs.python.org/2.7/library/argparse.html#nargs
    # parser.add_argument('--stuff', nargs='5', dest='my_stuff',
    #                    help="one or more random things")
    parser.add_argument('pos_args', nargs='*',
                        help="positional arguments don't have the '-' prefix")
    args = parser.parse_args()

    levels = [logging.ERROR, logging.WARN, logging.INFO, logging.DEBUG]
    logging.basicConfig(level=levels[min(3, args.verbosity)])

    if not args.pos_args:
        unittest.main()
        return 0

    for pn in args.pos_args:
        pass

    return 0


if __name__ == '__main__':
    sys.exit(main())
