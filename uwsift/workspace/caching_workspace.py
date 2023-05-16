import logging
import os
import shutil
from collections import OrderedDict
from datetime import datetime
from typing import Dict, Generator, Mapping, Optional, Tuple
from uuid import UUID

import numpy as np
from sqlalchemy.orm.exc import NoResultFound

from uwsift.common import Info, Kind, State
from uwsift.queue import TASK_DOING, TASK_PROGRESS

from .importer import SatpyImporter, aImporter
from .metadatabase import Content, ContentImage, Metadatabase, Product, Resource
from .workspace import ActiveContent, BaseWorkspace, frozendict

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

    def __init__(
        self,
        directory_path: str,
        process_pool=None,
        max_size_gb=DEFAULT_WORKSPACE_SIZE,
        queue=None,
        initial_clear=False,
    ):
        super(
            CachingWorkspace,
            self,
        ).__init__(directory_path, queue=queue)
        self._max_size_gb = max_size_gb  # maximum size in gigabytes of flat files we cache in the workspace
        if self._max_size_gb < MIN_WORKSPACE_SIZE:
            self._max_size_gb = MIN_WORKSPACE_SIZE
            LOG.warning("setting workspace size to %dGB" % self._max_size_gb)
        if directory_path is None:
            # a directory name was not given, we need a temporary directory
            import tempfile

            self._tempdir = tempfile.TemporaryDirectory()
            directory_path = str(self._tempdir)
            LOG.info("using temporary directory {}".format(directory_path))

        # filename to store and load inventory information (simple cache)
        self._inventory_path = os.path.join(self.cwd, "_inventory.db")
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
        LOG.info("{} database at {}".format("initializing" if should_init else "attaching", self._inventory_path))
        # metadatabase instance, sqlalchemy:
        self._inventory = Metadatabase("sqlite:///" + self._inventory_path, create_tables=should_init)
        if should_init:
            with self._inventory as s:
                assert 0 == s.query(Content).count()  # nosec B101
        LOG.info("done with init")

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
                "{} content entities no longer present in cache - will remove from database".format(len(to_purge))
            )
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
                    n_purged += 1
                    s.delete(r)
        LOG.info("discarded metadata for {} orphaned resources".format(n_purged))

    def _purge_orphan_products(self):
        """
        remove products from database that have no cached Content, and no Resource we can re-import from
        """
        LOG.debug(
            "purging Products no longer recoverable by re-importing from Resources, "
            "and having no Content representation in cache"
        )
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

    def _bgnd_startup_purge(self):
        ntot = 5
        n = 1
        yield {TASK_DOING: "DB pruning cache entries", TASK_PROGRESS: float(n) / float(ntot)}
        self._purge_missing_content()
        n += 1
        yield {TASK_DOING: "DB pruning stale resources", TASK_PROGRESS: float(n) / float(ntot)}
        self._purge_inaccessible_resources()
        n += 1
        yield {TASK_DOING: "DB pruning orphan products", TASK_PROGRESS: float(n) / float(ntot)}
        self._purge_orphan_products()
        n += 1
        yield {TASK_DOING: "DB migrating metadata", TASK_PROGRESS: float(n) / float(ntot)}
        self._migrate_metadata()
        n += 1
        yield {TASK_DOING: "DB ready", TASK_PROGRESS: float(n) / float(ntot)}

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
                LOG.debug("removing {}".format(pn))
                total += os.stat(pn).st_size
                try:
                    os.remove(pn)
                except FileNotFoundError:
                    LOG.warning("could not remove {} - file not found; continuing".format(pn))

        return total

    def _activate_content(self, c: Content) -> ActiveContent:
        self._available[c.id] = zult = ActiveContent(self.cache_dir, c, self.get_info(c.uuid))
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

    def _product_overview_content(
        self, session, prod: Optional[Product] = None, uuid: Optional[UUID] = None, kind: Kind = Kind.IMAGE
    ) -> Optional[Content]:
        if prod is None and uuid is not None:
            # Get Product object
            try:
                prod = session.query(Product).filter(Product.uuid_str == str(uuid)).one()
            except NoResultFound:
                LOG.error("No product with UUID {} found".format(uuid))
                return None
        assert prod  # nosec B101 # suppress mypy [union-attr]

        if kind == Kind.IMAGE:
            contents = (
                session.query(ContentImage).filter(ContentImage.product_id == prod.id).order_by(ContentImage.lod).all()
            )
        else:
            contents = session.query(Content).filter(Content.product_id == prod.id)
        contents = [c for c in contents if c.info.get(Info.KIND, Kind.IMAGE) == kind]
        return None if 0 == len(contents) else contents[0]

    def _product_native_content(
        self, session, prod: Optional[Product] = None, uuid: Optional[UUID] = None, kind: Kind = Kind.IMAGE
    ) -> Optional[Content]:
        # NOTE: This assumes the last Content object is the best resolution,
        #       but it is untested
        if prod is None and uuid is not None:
            # Get Product object
            try:
                prod = session.query(Product).filter(Product.uuid_str == str(uuid)).one()
            except NoResultFound:
                LOG.error("No product with UUID {} found".format(uuid))
                return None
        assert prod  # nosec B101 # suppress mypy [union-attr]

        if kind == Kind.IMAGE:
            contents = (
                session.query(ContentImage)
                .filter(ContentImage.product_id == prod.id)
                .order_by(ContentImage.lod.desc())
                .all()
            )
        else:
            contents = session.query(Content).filter(Content.product_id == prod.id)
        contents = [c for c in contents if c.info.get(Info.KIND, Kind.IMAGE) == kind]
        return None if 0 == len(contents) else contents[-1]

    #
    # combining queries with data content
    #

    def _overview_content_for_uuid(self, uuid: UUID, kind: Kind = Kind.IMAGE) -> np.memmap:
        # FUTURE: do a compound query for this to get the Content entry
        # prod = self._product_with_uuid(uuid)
        # assert(prod is not None)
        with self._inventory as s:
            ovc = self._product_overview_content(s, uuid=uuid, kind=kind)
            assert ovc is not None  # nosec B101
            arrays = self._cached_arrays_for_content(ovc)
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
        for root, _, files in os.walk(self.cache_dir):
            sz = sum(os.path.getsize(os.path.join(root, name)) for name in files)
            total += sz
            LOG.debug("%d bytes in %s" % (sz, root))

        return total

    def _all_product_uuids(self) -> list:
        with self._inventory as s:
            return [q.uuid for q in s.query(Product).all()]

    # ----------------------------------------------------------------------
    def get_info(self, info_or_uuid, lod=None) -> Optional[frozendict]:
        """
        :param info_or_uuid: existing datasetinfo dictionary, or its UUID
        :param lod: desired level of detail to focus
        :return: metadata access with mapping semantics, to be treated as read-only
        """
        from collections import ChainMap

        # FUTURE deprecate this
        if isinstance(info_or_uuid, str):
            uuid = UUID(info_or_uuid)
        elif not isinstance(info_or_uuid, UUID):
            uuid = info_or_uuid[Info.UUID]
        else:
            uuid = info_or_uuid
        with self._inventory as s:
            # look up the product for that uuid
            prod = self._product_with_uuid(s, uuid)
            if not prod:  # then it hasn't had its metadata scraped
                LOG.error("no info available for UUID {}".format(info_or_uuid))
                LOG.error("known products: {}".format(repr(self._all_product_uuids())))
                return None
            kind = prod.info[Info.KIND]
            native_content = self._product_native_content(s, prod=prod, kind=kind)

            if native_content is not None:
                # FUTURE: this is especially saddening; upgrade to finer grained query and/or deprecate .get_info
                # once upon a time...
                # our old model was that product == content and shares a UUID with the dataset
                # if content is available, we want to provide native content metadata along with the product metadata
                # specifically a lot of client code assumes that resource == product == content and
                # that singular navigation (e.g. cell_size) is norm
                # FIXME DEBUG <- since commit 3576ff0122bd534f83422ce19479d40b7dc9e5b0
                assert (  # nosec B101
                    kind in [Kind.LINES, Kind.POINTS] or native_content.info[Info.CELL_WIDTH] is not None
                )
                return frozendict(ChainMap(native_content.info, prod.info))
            # mapping semantics for database fields, as well as key-value fields;
            # flatten to one namespace and read-only
            return frozendict(prod.info)

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

    def recently_used_products(self, n=32) -> Dict[UUID, str]:
        with self._inventory as s:
            return OrderedDict(
                (p.uuid, p.info[Info.DISPLAY_NAME])
                for p in s.query(Product).order_by(Product.atime.desc()).limit(n).all()
            )

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
            GB = 1024**3
            LOG.info("total cache size is {}GB of max {}GB".format(total_size / GB, self._max_size_gb))
            max_size = self._max_size_gb * GB
            for res in S.query(Resource).order_by(Resource.atime).all():
                if total_size < max_size:
                    break
                total_size -= self._purge_content_for_resource(res, session=S)
                # remove all content for lowest atimes until

    def close(self):
        self._clean_cache()

    def bgnd_task_complete(self):
        """
        handle operations that should be done at the end of a threaded background task
        """
        pass

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
                        raise EnvironmentError("more than one Resource fits this path")
                    resource = hits[0]
                    if len(resource.product) >= 1:
                        if len(resource.product) > 1:
                            LOG.warning("more than one Product in this Resource, this query should be deprecated")
                        prod = resource.product[0]
                        return prod.info

    def collect_product_metadata_for_paths(
        self, paths: list, **importer_kwargs
    ) -> Generator[Tuple[int, frozendict], None, None]:
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
            importers = []
            num_products = 0
            if "reader" not in importer_kwargs:
                # If there is no reader in the importer_kwargs then the SatPy Import can't be used
                return None

            # Pass paths to SatPy importer and see what happens
            if paths:
                if "reader" not in importer_kwargs:
                    raise NotImplementedError(
                        "Reader discovery is not " "currently implemented in " "the satpy importer."
                    )
                if "scenes" in importer_kwargs:
                    # another component already created the satpy scenes, use those
                    scenes = importer_kwargs.pop("scenes")
                    scenes = scenes.items()
                else:
                    scenes = [(paths, None)]
                for paths, scene in scenes:
                    imp = SatpyImporter
                    these_kwargs = importer_kwargs.copy()
                    these_kwargs["scene"] = scene
                    hauler = imp(paths, database_session=import_session, workspace_cwd=self.cache_dir, **these_kwargs)
                    hauler.merge_resources()
                    importers.append(hauler)
                    num_products += hauler.num_products

            for hauler in importers:
                for prod in hauler.merge_products():
                    assert prod is not None  # nosec B101
                    # merge the product into our database session, since it may belong to import_session
                    zult = frozendict(prod.info)
                    yield num_products, zult

    def import_product_content(
        self,
        uuid: UUID,
        prod: Optional[Product] = None,
        allow_cache=True,
        merge_target_uuid: Optional[UUID] = None,
        **importer_kwargs,
    ) -> np.memmap:
        with self._inventory as S:
            if prod is None and uuid is not None:
                prod = self._product_with_uuid(S, uuid)

            self.set_product_state_flag(prod.uuid, State.ARRIVING)
            default_prod_kind = prod.info[Info.KIND]

            if len(prod.content):
                LOG.info("product already has content available, using that rather than re-importing")
                ovc = self._product_overview_content(S, uuid=uuid, kind=default_prod_kind)
                assert ovc is not None  # nosec B101
                arrays = self._cached_arrays_for_content(ovc)
                return arrays.data

            truck = aImporter.from_product(prod, workspace_cwd=self.cache_dir, database_session=S, **importer_kwargs)
            if not truck:
                # aImporter.from_product() didn't return an Importer instance
                # since all files represent data granules, which are already
                # loaded and merged into existing datasets.
                # Thus: nothing to do.
                return None
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
                    LOG.info("{} {}: {:.01f}%".format(name, update.stage_desc, update.completion * 100.0))
            LOG.debug("received {} updates during import".format(nupd))
            uuid = prod.uuid
            self._clear_product_state_flag(prod.uuid, State.ARRIVING)

        # make an ActiveContent object from the Content, now that we've imported it
        ac = self._overview_content_for_uuid(uuid, kind=default_prod_kind)
        if ac is None:
            return None
        return ac.data

    def _create_product_from_array(
        self, info: Mapping, data, namespace=None, codeblock=None
    ) -> Tuple[UUID, Optional[frozendict], np.memmap]:
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
            raise ValueError("currently require an Info.UUID be included in product")
        parms = dict(info)
        now = datetime.utcnow()
        parms.update(
            dict(
                atime=now,
                mtime=now,
            )
        )
        P = Product.from_info(parms, symbols=namespace, codeblock=codeblock)
        uuid = P.uuid
        # FUTURE: add expression and namespace information, which would require additional parameters
        ws_filename = "{}.image".format(str(uuid))
        ws_path = os.path.join(self.cache_dir, ws_filename)
        with open(ws_path, "wb+") as fp:
            mm = np.memmap(fp, dtype=data.dtype, shape=data.shape, mode="w+")
            mm[:] = data[:]

        parms.update(
            dict(
                lod=ContentImage.LOD_OVERVIEW,
                path=ws_filename,
                dtype=str(data.dtype),
                proj4=info[Info.PROJ],
                resolution=min(info[Info.CELL_WIDTH], info[Info.CELL_HEIGHT]),
            )
        )
        rcls = dict(zip(("rows", "cols", "levels"), data.shape))
        parms.update(rcls)
        LOG.debug("about to create Content with this: {}".format(repr(parms)))

        C = ContentImage.from_info(parms, only_fields=True)
        P.content.append(C)
        # FUTURE: do we identify a Resource to go with this? Probably not

        # transaction with the metadatabase to add the product and content
        with self._inventory as S:
            S.add(P)
            S.add(C)

        # FIXME: Do I have to flush the session so the Product gets added for sure?

        # activate the content we just loaded into the workspace
        overview_data = self._overview_content_for_uuid(uuid)
        return uuid, self.get_info(uuid), overview_data

    def _bgnd_remove(self, uuid: UUID):
        from uwsift.queue import TASK_DOING, TASK_PROGRESS

        yield {TASK_DOING: "purging memory", TASK_PROGRESS: 0.5}
        LOG.debug(f"Active Content before deletion: {list(self._available.keys())}")
        with self._inventory as s:
            self._deactivate_content_for_product(self._product_with_uuid(s, uuid))
        LOG.debug(f"Active Content after deletion: {list(self._available.keys())}")
        yield {TASK_DOING: "purging memory", TASK_PROGRESS: 1.0}

    def get_content(self, info_or_uuid, lod=None, kind: Kind = Kind.IMAGE) -> Optional[np.memmap]:
        """
        By default, get the best-available (closest to native) np.ndarray-compatible view of the full dataset
        :param info_or_uuid: existing datasetinfo dictionary, or its UUID
        :param lod: desired level of detail to focus  (0 for overview)
        :return:
        """
        if info_or_uuid is None:
            return None
        elif isinstance(info_or_uuid, UUID):
            uuid = info_or_uuid
        elif isinstance(info_or_uuid, str):
            uuid = UUID(info_or_uuid)
        else:
            uuid = info_or_uuid[Info.UUID]
        # TODO: this causes a locking exception when run in a secondary thread.
        #  Keeping background operations lightweight makes sense however, so just review this
        with self._inventory as s:
            if kind == Kind.IMAGE:
                content = (
                    s.query(ContentImage)
                    .filter((Product.uuid_str == str(uuid)) & (ContentImage.product_id == Product.id))
                    .order_by(ContentImage.lod.desc())
                    .all()
                )
            else:
                content = s.query(Content).filter((Product.uuid_str == str(uuid)) & (Content.product_id == Product.id))

            content = [x for x in content if x.info.get(Info.KIND, Kind.IMAGE) == kind]
            if len(content) != 1:
                LOG.warning("More than one matching Content object for '{}'".format(info_or_uuid))
            if not len(content) or not content[0]:
                raise AssertionError("no content in workspace for {}, must re-import".format(uuid))
            content = content[0]

            # FIXME: find the content for the requested LOD, then return its
            #  ActiveContent - or attach one
            #  for now, just work with assumption of one product one content
            active_content = self._cached_arrays_for_content(content)
            return active_content.data

    def _deactivate_content_for_product(self, p: Optional[Product]):
        if p is None:
            return
        for c in p.content:
            self._available.pop(c.id, None)

    def _get_active_content_by_uuid(self, uuid: UUID) -> Optional[ActiveContent]:
        with self._inventory as s:
            prod = self._product_with_uuid(s, uuid)
            if prod is None:
                return None
            content = s.query(Content).filter(Content.product_id == prod.id).one()

            return self._available.get(content.id)
