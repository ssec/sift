import logging
import os
from collections import ChainMap
from datetime import datetime
from typing import Dict, Generator, Optional, Tuple
from uuid import UUID

import numpy as np
import satpy.readers

from uwsift import config, CLEANUP_FILE_CACHE
from uwsift.common import Info, Kind, Flags, State
from .importer import aImporter, SatpyImporter
from .metadatabase import Metadatabase, Product, Content
from .workspace import BaseWorkspace, frozendict, ActiveContent

LOG = logging.getLogger(__name__)


class SimpleWorkspace(BaseWorkspace):
    """
    Data management object for monitoring use case.

    Unlike CachingWorkspace SimpleWorkspace has no database where the
    datasets are saved. So every dataset which is loaded is only available while
    the software is running.

    SimpleWorkspace shall work with Datasets.
    SimpleWorkspace have one dictionary for saving the Product objects and
    one dictionary for saving the Content objects for a specific UUID.
    """

    def __init__(self, directory_path: str=None):
        super(SimpleWorkspace, self).__init__(directory_path)

        self.products: dict = {}
        self.contents: dict = {}
        self._available: dict = {}

        self._remove_content_data_from_cache_dir_checked()

    def product_state(self, uuid: UUID) -> Flags:
        state: Flags = Flags(self._state[uuid])
        # add any derived information
        if uuid in self._available:
            state.add(State.ATTACHED)
        return state

    @property
    def _S(self):
        return None

    @property
    def metadatabase(self) -> Metadatabase:
        return None

    def clear_workspace_content(self):
        """Remove binary files from workspace and workspace database."""
        LOG.info("Clearing workspace contents...")
        self.contents = {}
        self.products = {}

    #
    #  data array handling
    #

    def _activate_content(self, c: Content) -> ActiveContent:
        self._available[c.uuid] = zult = ActiveContent(self.cache_dir, c)
        c.touch()
        c.product.touch()
        self._remove_content_data_from_cache_dir_checked(c)
        return zult

    def _cached_arrays_for_content(self, c: Content):
        """
        attach cached data indicated in Content, unless it's been attached already and is in _available
        touch the content and product in the database to appease the LRU gods
        :param c: metadatabase Content object for session attached to current thread
        :return: workspace_content_arrays
        """
        cache_entry = self._available.get(c.uuid)
        return cache_entry or self._activate_content(c)

    # FIXME: Use code from CachingWorkspace._remove_content_files_from_workspace?
    def _remove_content_data_from_cache_dir_checked(self,
                                                    c: Optional[Content] = None):
        if CLEANUP_FILE_CACHE:
            for file in os.listdir(self.cache_dir):
                if c is not None:
                    if file.startswith(str(c.uuid)):
                        os.remove(os.path.join(self.cache_dir, file))
                else:
                    os.remove(os.path.join(self.cache_dir, file))

    #
    # often-used queries
    #

    def _product_with_uuid(self, session, uuid: UUID) -> Optional[Product]:
        return self.products.get(uuid, None)

    def _product_overview_content(self, session, prod: Product = None, uuid: UUID = None,
                                  kind: Kind = Kind.IMAGE) -> Optional[Content]:
        return self.contents.get(uuid, None)

    def _product_native_content(self, session, prod: Product = None, uuid: UUID = None,
                                kind: Kind = Kind.IMAGE) -> Optional[Content]:
        return self.contents.get(uuid, None)

    #
    # combining queries with data content
    #

    def _overview_content_for_uuid(self, uuid: UUID, kind: Kind = Kind.IMAGE) \
            -> np.memmap:
        ovc = self._product_overview_content(None, uuid=uuid, kind=kind)
        assert (ovc is not None)
        arrays = self._cached_arrays_for_content(ovc)
        return arrays.data

    def _native_content_for_uuid(self, uuid: UUID) -> np.memmap:
        nac = self._product_native_content(None, uuid=uuid)
        arrays = self._cached_arrays_for_content(nac)
        return arrays.data

    #
    # workspace file management
    #

    @property
    def _total_workspace_bytes(self):
        return None

    def _all_product_uuids(self) -> list:
        return [self.products[p] for p in self.products]

    def get_info(self, dsi_or_uuid, lod=None) -> Optional[frozendict]:
        """
        :param dsi_or_uuid: existing datasetinfo dictionary, or its UUID
        :param lod: desired level of detail to focus
        :return: metadata access with mapping semantics, to be treated as read-only
        """
        # FUTURE deprecate this
        if isinstance(dsi_or_uuid, str):
            uuid = UUID(dsi_or_uuid)
        elif not isinstance(dsi_or_uuid, UUID):
            uuid = dsi_or_uuid[Info.UUID]
        else:
            uuid = dsi_or_uuid

        prod = self._product_with_uuid(None, uuid)
        if not prod:  # then it hasn't had its metadata scraped
            LOG.error('no info available for UUID {}'.format(dsi_or_uuid))
            LOG.error("known products: {}".format(
                repr(self._all_product_uuids())))
            return None
        kind = prod.info[Info.KIND]
        native_content = self._product_native_content(None, prod=prod,
                                                      uuid=uuid,
                                                      kind=kind)
        if native_content is not None:
            # FUTURE: this is especially saddening; upgrade to finer grained
            # query and/or deprecate .get_info
            # once upon a time...
            # our old model was that product == content and shares a UUID with
            # the layer if content is available, we want to provide native
            # content metadata along with the product metadata
            # specifically a lot of client code assumes that resource
            # == product == content and
            # that singular navigation (e.g. cell_size) is norm
            assert (kind in [Kind.VECTORS, Kind.POINTS] or
                    native_content.info[Info.CELL_WIDTH] is not None)  # FIXME DEBUG
            return frozendict(ChainMap(native_content.info, prod.info))
        # mapping semantics for database fields, as well as key-value fields;
        # flatten to one namespace and read-only
        return frozendict(prod.info)

    def get_algebraic_namespace(self, uuid: UUID):
        if uuid is None:
            return {}, ""
        prod = self._product_with_uuid(None, uuid)
        if prod is None:
            return {}, ""
        symbols = {x.key: x.value for x in prod.symbol}
        code = prod.expression
        return symbols, code

    @property
    def product_names_available_in_cache(self) -> dict:
        return None

    @property
    def uuids_in_cache(self):
        return None

    def recently_used_products(self, n=32) -> Dict[UUID, str]:
        pass

    def remove_all_workspace_content_for_resource_paths(self, paths: list):
        pass

    def purge_content_for_product_uuids(self, uuids: list, also_products=False):
        pass

    def close(self):
        pass

    def bgnd_task_complete(self):
        pass

    def get_metadata(self, uuid_or_path):
        if isinstance(uuid_or_path, UUID):
            return self.get_info(uuid_or_path)  # get product metadata
        else:
            return None

    def collect_product_metadata_for_paths(self, paths: list,
                                           **importer_kwargs) -> Generator[Tuple[int, frozendict], None, None]:
        """Start loading URI data into the workspace asynchronously.

        Args:
            paths (list): String paths to open and get metadata for
            **importer_kwargs: Keyword arguments to pass to the lower-level
                importer class.

        Returns: sequence of read-only info dictionaries

        """
        importers = []
        num_products = 0
        remaining_paths = []
        if 'reader' in importer_kwargs:
            # skip importer guessing and go straight to satpy importer
            paths, remaining_paths = [], paths

        for source_path in paths:
            # LOG.info('collecting metadata for {}'.format(source_path))
            # FIXME: Check if importer only accepts one path at a time
            #        Maybe sort importers by single files versus multiple
            #        files and doing single files first?
            # FIXME: decide whether to update database if mtime of file
            #  is newer than mtime in database
            # Collect all the importers we are going to use and count
            # how many products each expects to return
            for imp in self._importers:
                if imp.is_relevant(source_path=source_path):
                    hauler = imp(source_path,
                                 database_session=None,
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
                             database_session=None,
                             workspace_cwd=self.cache_dir,
                             **these_kwargs)
                hauler.merge_resources()
                importers.append(hauler)
                num_products += hauler.num_products

        for hauler in importers:
            for prod in hauler.merge_products():
                assert (prod is not None)
                # add to-be-imported filenames to check for possible merge targets but
                # do not include this filenames in the product info
                extended_prod_info = dict(prod.info)
                extended_prod_info['paths'] = hauler.filenames
                zult = frozendict(extended_prod_info)
                # merge the product into our database session, since it may
                # belong to import_session
                # self._S.merge(prod)
                self.products[prod.uuid] = prod
                # LOG.debug('yielding product metadata for {}'.format(
                #     zult.get(Info.DISPLAY_NAME, '?? unknown name ??')))
                yield num_products, zult

    def import_product_content(self, uuid: UUID = None, prod: Product = None,
                               allow_cache=True, merge_uuid: Optional[UUID] = None,
                               **importer_kwargs) -> np.memmap:
        if prod is None and uuid is not None:
            prod = self._product_with_uuid(None, uuid)

        if merge_uuid:
            merge_target = self._product_with_uuid(None, merge_uuid)
            importer_kwargs["merge_target"] = merge_target
            self.products.pop(uuid, None)
        else:
            importer_kwargs["merge_target"] = None

        self.set_product_state_flag(prod.uuid, State.ARRIVING)
        default_prod_kind = prod.info[Info.KIND]

        if merge_uuid and len(prod.content):
            LOG.info(
                'product already has content available, using that '
                'rather than re-importing')
            ovc = self._product_overview_content(None, uuid=uuid,
                                                 kind=default_prod_kind)
            assert (ovc is not None)
            arrays = self._cached_arrays_for_content(ovc)
            return arrays.data

        truck = aImporter.from_product(prod, workspace_cwd=self.cache_dir,
                                       database_session=None,
                                       **importer_kwargs)
        if not truck:
            # aImporter.from_product() didn't return an Importer instance
            # since all files represent data granules, which are already
            # loaded and merged into existing datasets.
            # Thus: nothing to do.
            return None
        metadata = prod.info
        name = metadata[Info.SHORT_NAME]

        gen = truck.begin_import_products(prod)
        nupd = 0
        for update in gen:
            nupd += 1
            # we're now incrementally reading the input file
            # data updates are coming back to us (eventually asynchronously)
            # Content is in the metadatabase and being updated + committed,
            # including sparsity and coverage arrays
            if update.data is not None:
                # data = update.data
                LOG.info("{} {}: {:.01f}%".format(name, update.stage_desc,
                                                  update.completion * 100.0))
            if update.content is not None:
                self.contents[update.uuid] = update.content
        # self._data[uuid] = data = self._convert_to_memmap(str(uuid), data)
        LOG.debug('received {} updates during import'.format(nupd))
        self.clear_product_state_flag(prod.uuid, State.ARRIVING)
        # S.commit()
        # S.flush()

        # make an ActiveContent object from the Content, now that we've imported it
        ac = self._overview_content_for_uuid(merge_uuid if merge_uuid else prod.uuid,
                                             kind=default_prod_kind)
        if ac is None:
            return None
        return ac.data

    def find_merge_target(self, uuid: UUID, paths, info) -> Optional[Product]:
        """
        Try to find an existing product where the to-be-imported files could
        be merged into.

        :param uuid: uuid of the product which is about to be imported and
                     might be merged with an existing product
        :param paths: the paths which should be imported or merged
        :param info: metadata for the to-be-imported product
        :return: the existing product to merge new content into or None if no
                 existing product is compatible
        """
        reader = info['reader']
        group_keys = config.get(f"data_reading.{reader}.group_keys", None)
        for existing_uuid, existing_prod in self.products.items():
            # exclude all products which are incomplete (products which are imported right now)
            # and products with different kind or parameter
            # TODO: when loading granules without resampling after granules of
            #  of same FAMILY have been loaded *with* resampling already,
            #  merging must be prevented.
            #  As an attempt to achieve this the SHAPE size comparison checks,
            #  whether they are compatible for array broadcasting, but that is
            #  only a necessary, but not sufficient condition.
            #  Idea: modify the FAMILY info by adding grid information?
            if not existing_prod.content \
                    or reader != existing_prod.info['reader'] \
                    or info[Info.SHAPE][0] > existing_prod.info[Info.SHAPE][0] \
                    or info[Info.SHAPE][1] != existing_prod.info[Info.SHAPE][1] \
                    or info[Info.FAMILY] != existing_prod.info[Info.FAMILY]:
                continue

            # if to-be-imported product seem to be compatible with an existing product check
            # if satpy would group together the to-be-imported files and the already loaded files in
            # the existing merge candidate
            all_files = set(existing_prod.content[0].source_files) if existing_prod.content[0] else set()
            all_files |= set(paths)
            grouped_files = satpy.readers.group_files(all_files, reader=reader, group_keys=group_keys)
            if len(grouped_files) == 1 \
                    and len(grouped_files[0]) == 1 \
                    and reader in grouped_files[0] \
                    and len(all_files) == len(grouped_files[0][reader]):
                return existing_prod
        return None

    def _create_product_from_array(self, info: Info, data, namespace=None, codeblock=None) \
            -> Tuple[UUID, Optional[frozendict], np.memmap]:
        """
        Puts created image array into resp. data structures within workspace and returns
        uuid, updated info, as well as the memmap of the created array.

        Side effects include:
            Write np.memmap to disk for later retrieval by workspace. Also updates metadata Product
            object by path to .image memmap file and adds created Content to workspace's `contents`.
            Finally, imports product into workspace.

        Args:
            info: mapping of key-value metadata for new product
            data: ndarray with content to store, typically 2D float32
            namespace: {variable: uuid, } for calculation of this data
            codeblock: text, code to run to recalculate this data within namespace

        Returns:
            uuid, info, data: uuid of the new product, its official read-only metadata, and cached
            content ndarray
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
        # Write memmap to disk, for later reference by workspace
        with open(ws_path, 'wb+') as fp:
            mm = np.memmap(fp, dtype=data.dtype, shape=data.shape, mode='w+')
            mm[:] = data[:]
        # Update metadata to contain path to cached memmap .image file
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

        self.contents[uuid] = C
        self.products[uuid] = P
        # activate the content we just loaded into the workspace
        overview_data = self._overview_content_for_uuid(uuid)

        return uuid, self.get_info(uuid), overview_data

    def _bgnd_remove(self, uuid: UUID):
        from uwsift.queue import TASK_DOING, TASK_PROGRESS
        yield {TASK_DOING: 'purging memory', TASK_PROGRESS: 0.5}
        self._deactivate_content_for_product(self._product_with_uuid(None,
                                                                     uuid))
        self.contents.pop(uuid, None)
        self.products.pop(uuid, None)
        yield {TASK_DOING: 'purging memory', TASK_PROGRESS: 1.0}

    def get_content(self, dsi_or_uuid, lod=None, kind: Kind = Kind.IMAGE) \
            -> Optional[np.memmap]:
        """
        By default, get the best-available (closest to native)
        np.ndarray-compatible view of the full dataset
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

        content = [self.contents[uuid] if uuid in self.contents else None]
        content = [x for x in content if
                   x.info.get(Info.KIND, Kind.IMAGE) == kind]

        if len(content) != 1:
            LOG.warning(
                "More than one matching Content object for '{}'".format(
                    dsi_or_uuid))
        if not len(content) or content[0] is None:
            raise AssertionError(
                'no content in workspace for {}, must re-import'.format(
                    uuid))
        content = content[0]
        # content.touch()
        # self._S.commit()  # flush any pending updates to workspace db file

        # FIXME: find the content for the requested LOD, then return its
        #  ActiveContent - or attach one
        # for now, just work with assumption of one product one content
        active_content = self._cached_arrays_for_content(content)
        return active_content.data

    def _deactivate_content_for_product(self, p: Product):
        if p is None:
            return
        for c in p.content:
            self._available.pop(c.uuid, None)
