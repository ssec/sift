#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Implement Workspace, a singleton object which manages large amounts of data and caches local content.

Workspace of Products

- retrieved from Resources and
- represented by multidimensional Content, each of which has data,
  coverage, and sparsity arrays in separate workspace flat files

Workspace responsibilities include:

- understanding projections and y, x, z coordinate systems
- subsecting data within slicing or geospatial boundaries
- caching useful arrays as secondary content
- performing minimized on-demand calculations, e.g. algebraic layers, in the background
- use Importers to bring content arrays into the workspace from external resources, also in the background
- maintain a metadatabase of what products have in-workspace content, and what products are available
  from external resources
- compose Collector, which keeps track of Products within Resources outside the workspace

FUTURE import sequence:

- trigger: user requests skim (metadata only) or import (metadata plus bring into document)
      of a file or directory system for each file selected
- phase 1: regex for file patterns identifies which importers are worth trying
- phase 2: background: importers open files, form metadatabase insert transaction,
      first importer to succeed wins (priority order). stop after this if just skimming
- phase 3: background: load of overview (lod=0), adding flat files to workspace and Content entry to metadatabase
- phase 3a: document and scenegraph show overview up on screen
- phase 4: background: load of one or more levels of detail, with max LOD currently being considered native
- phase 4a: document updates to show most useful LOD+stride content

:author: R.K.Garcia <rayg@ssec.wisc.edu>
:copyright: 2014-2017 by University of Wisconsin Regents, see AUTHORS for more details
:license: GPLv3, see LICENSE for more details

"""

import logging
import os
import shutil
from collections import defaultdict, OrderedDict
from collections.abc import Mapping as ReadOnlyMapping
from datetime import datetime, timedelta
from typing import Mapping, Generator, Tuple, Dict
from uuid import UUID, uuid1 as uuidgen

import numba as nb
import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal
from pyproj import Proj
from rasterio import Affine
from shapely.geometry.polygon import LinearRing
from sqlalchemy.orm.exc import NoResultFound

from uwsift.common import Info, Kind, Flags, State
from uwsift.model.shapes import content_within_shape
from uwsift.queue import TASK_PROGRESS, TASK_DOING
from .importer import aImporter, SatpyImporter, generate_guidebook_metadata
from .metadatabase import Metadatabase, Content, Product, Resource

LOG = logging.getLogger(__name__)

DEFAULT_WORKSPACE_SIZE = 256
MIN_WORKSPACE_SIZE = 8

IMPORT_CLASSES = [SatpyImporter]

# first instance is main singleton instance; don't preclude the possibility of importing from another workspace later on
TheWorkspace = None


class frozendict(ReadOnlyMapping):
    def __init__(self, source=None):
        self._D = dict(source) if source else {}

    def __getitem__(self, key):
        return self._D[key]

    def __iter__(self):
        for k in self._D.keys():
            yield k

    def __len__(self):
        return len(self._D)

    def __repr__(self):
        return "frozendict({" + ", ".join("{}: {}".format(repr(k), repr(v)) for (k, v) in self.items()) + "})"


@nb.jit(nogil=True)
def mask_from_coverage_sparsity_2d(mask: np.ndarray, coverage: np.ndarray, sparsity: np.ndarray):
    """
    update a numpy.ma style mask from coverage and sparsity arrays
    Args:
        mask (np.array, dtype=bool): mutable array to be updated as a numpy.masked_array mask
        coverage (np.array) : coverage array to stretch across the mask
        sparsity (np.array) : sparsity array to repeat across the mask
    """
    # FIXME: we should not be using this without slicing to the part of interest; else we can eat a whole lot of memory
    h, w = mask.shape
    cov_rpt_y, cov_rpt_x = h // coverage.shape[0], w // coverage.shape[1]
    spr_h, spr_w = sparsity.shape
    for y in range(h):
        for x in range(w):
            mask[y, x] |= bool(0 == coverage[y // cov_rpt_y, x // cov_rpt_x] * sparsity[y % spr_h, x % spr_w])


class ActiveContent(QObject):
    """
    ActiveContent composes numpy.memmap arrays with their corresponding Content metadata, and is owned by Workspace
    Purpose: consolidate common operations on content, while factoring in things like sparsity, coverage, y, x, z arrays
    Workspace instantiates ActiveContent from metadatabase Content entries
    """
    _cid = None  # Content.id database entry I belong to
    _wsd = None  # full path of workspace
    _rcl = None
    _y = None
    _x = None
    _z = None
    _data = None
    _mask = None
    _coverage = None
    _sparsity = None

    def __init__(self, workspace_cwd: str, C: Content):
        super(ActiveContent, self).__init__()
        self._cid = C.id
        self._wsd = workspace_cwd
        if workspace_cwd is None and C is None:
            LOG.warning('test initialization of ActiveContent')
            self._test_init()
        else:
            self._attach(C)

    def _test_init(self):
        data = np.ones((4, 12), dtype=np.float32)
        data = np.cumsum(data, axis=0)
        data = np.cumsum(data, axis=1)
        self._data = data
        self._sparsity = sp = np.zeros((2, 2), dtype=np.int8)
        sp[1, 1] = 1  # only 1/4 of dataset loaded
        self._coverage = co = np.zeros((4, 1), dtype=np.int8)
        co[2:4] = 1  # and of that, only the bottom half of the image

    @staticmethod
    def _rcls(rows: int, columns: int, levels: int):
        """
        :param rows: rows or None
        :param columns: columns or None
        :param levels: levels or None
        :return: condensed tuple(string with 'rcl', 'rc', 'rl', dimension tuple corresponding to string)
        """
        rcl_shape = tuple(
            (name, dimension) for (name, dimension) in zip('rcl', (rows, columns, levels)) if dimension)
        rcl = tuple(x[0] for x in rcl_shape)
        shape = tuple(x[1] for x in rcl_shape)
        return rcl, shape

    @classmethod
    def can_attach(cls, wsd: str, c: Content):
        """
        Is this content available in the workspace?
        Args:
            wsd: workspace realpath
            c: Content metadatabase entry

        Returns:
            bool
        """
        path = os.path.join(wsd, c.path)
        return os.access(path, os.R_OK) and (os.stat(path).st_size > 0)

    @property
    def data(self):
        """
        Returns: content data (np.ndarray)
        """
        # FIXME: apply sparsity, coverage, and missing value masks
        return self._data

    def _update_mask(self):
        """
        merge sparsity and coverage mask to a standard maskedarray mask
        :return:
        """
        # FIXME: beware the race conditions with this
        # FIXME: it would be better to lazy-eval the mask, assuming coverage and sparsity << data
        if self._mask is None:
            # self._mask = mask = np.zeros_like(self._data, dtype=bool)
            self._mask = mask = ~np.isfinite(self._data)
        else:
            mask = self._mask
            mask[:] = False
        # LOG.warning('mask_from_coverage_sparsity needs inclusion')
        # present = np.array([[1]], dtype=np.int8)
        # mask_from_coverage_sparsity_2d(mask, self._coverage or present, self._sparsity or present)

    def _attach(self, c: Content, mode='c'):
        """
        attach content arrays, for holding by workspace in _available
        :param c: Content entity from database
        :return: workspace_data_arrays instance
        """
        self._rcl, self._shape = rcl, shape = self._rcls(c.rows, c.cols, c.levels)

        def mm(path, *args, **kwargs):
            full_path = os.path.join(self._wsd, path)
            if not os.access(full_path, os.R_OK):
                LOG.warning("unable to find {}".format(full_path))
                return None
            return np.memmap(full_path, *args, **kwargs)

        self._data = mm(c.path, dtype=c.dtype or np.float32, mode=mode, shape=shape)  # potentially very very large
        self._y = mm(c.y_path, dtype=c.dtype or np.float32, mode=mode, shape=shape) if c.y_path else None
        self._x = mm(c.x_path, dtype=c.dtype or np.float32, mode=mode, shape=shape) if c.x_path else None
        self._z = mm(c.z_path, dtype=c.dtype or np.float32, mode=mode, shape=shape) if c.z_path else None

        _, cshape = self._rcls(c.coverage_cols, c.coverage_cols, c.coverage_levels)
        self._coverage = mm(c.coverage_path, dtype=np.int8, mode=mode, shape=cshape) if c.coverage_path else np.array(
            [1])
        _, sshape = self._rcls(c.coverage_cols, c.coverage_cols, c.coverage_levels)
        self._sparsity = mm(c.sparsity_path, dtype=np.int8, mode=mode, shape=sshape) if c.sparsity_path else np.array(
            [1])

        self._update_mask()


class Workspace(QObject):
    """Data management and cache object.

    Workspace is a singleton object which works with Datasets shall:

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

    cwd = None  # directory we work in
    _own_cwd = None  # whether or not we created the cwd - which is also whether or not we're allowed to destroy it
    _pool = None  # process pool that importers can use for background activities, if any
    # _importers = None  # list of importers to consult when asked to start an import
    _available: Mapping[int, ActiveContent] = None  # dictionary of {Content.id : ActiveContent object}
    _inventory: Metadatabase = None  # metadatabase instance, sqlalchemy
    _inventory_path = None  # filename to store and load inventory information (simple cache)
    _tempdir = None  # TemporaryDirectory, if it's needed (i.e. a directory name was not given)
    _max_size_gb = None  # maximum size in gigabytes of flat files we cache in the workspace
    _queue = None

    # signals
    # a dataset started importing; generated after overview level of detail is available
    # didStartImport = pyqtSignal(dict)
    # didMakeImportProgress = pyqtSignal(dict)
    didUpdateProductsMetadata = pyqtSignal(set)  # set of UUIDs with changes to their metadata
    # didFinishImport = pyqtSignal(dict)  # all loading activities for a dataset have completed
    # didDiscoverExternalDataset = pyqtSignal(dict)  # a new dataset was added to the workspace from an external agent
    didChangeProductState = pyqtSignal(UUID, Flags)  # a product changed state, e.g. an importer started working on it

    _state: Mapping[UUID, Flags] = None

    def set_product_state_flag(self, uuid: UUID, flag):
        """primarily used by Importers to signal work in progress
        """
        state = self._state[uuid]
        state.add(flag)
        self.didChangeProductState.emit(uuid, state)

    def clear_product_state_flag(self, uuid: UUID, flag):
        state = self._state[uuid]
        state.remove(flag)
        self.didChangeProductState.emit(uuid, state)

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

    @staticmethod
    def defaultWorkspace():
        """
        return the default (global) workspace
        Currently no plans to have more than one workspace, but secondaries may eventually have some advantage.
        :return: Workspace instance
        """
        return TheWorkspace

    def __init__(self, directory_path=None, process_pool=None, max_size_gb=None, queue=None, initial_clear=False):
        """
        Initialize a new or attach an existing workspace, creating any necessary bookkeeping.
        """
        super(Workspace, self).__init__()
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

        # HACK: handle old workspace command line flag
        if isinstance(directory_path, (list, tuple)):
            self.cache_dir = cache_path = os.path.abspath(directory_path[1])
            self.cwd = directory_path = os.path.abspath(directory_path[0])
        else:
            self.cwd = directory_path = os.path.abspath(directory_path)
            self.cache_dir = cache_path = os.path.join(self.cwd, 'data_cache')
        self._inventory_path = os.path.join(self.cwd, '_inventory.db')
        if initial_clear:
            self.clear_workspace_content()
        if not os.path.isdir(cache_path):
            LOG.info("creating new workspace cache at {}".format(cache_path))
            os.makedirs(cache_path)
        if not os.path.isdir(directory_path):
            LOG.info("creating new workspace at {}".format(directory_path))
            os.makedirs(directory_path)
            self._own_cwd = True
            self._init_create_workspace()
        else:
            LOG.info("attaching pre-existing workspace at {}".format(directory_path))
            self._own_cwd = False
            self._init_inventory_existing_datasets()

        self._available = {}
        self._importers = IMPORT_CLASSES.copy()
        self._state = defaultdict(Flags)
        global TheWorkspace  # singleton
        if TheWorkspace is None:
            TheWorkspace = self

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

    def _deactivate_content_for_product(self, p: Product):
        if p is None:
            return
        for c in p.content:
            self._available.pop(c.id, None)

    #
    # often-used queries
    #

    def _product_with_uuid(self, session, uuid) -> Product:
        return session.query(Product).filter_by(uuid_str=str(uuid)).first()

    def _product_overview_content(self, session, prod: Product = None, uuid: UUID = None,
                                  kind: Kind = Kind.IMAGE) -> Content:
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
                                kind: Kind = Kind.IMAGE) -> Content:
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

    def _overview_content_for_uuid(self, uuid, kind=Kind.IMAGE):
        # FUTURE: do a compound query for this to get the Content entry
        # prod = self._product_with_uuid(uuid)
        # assert(prod is not None)
        with self._inventory as s:
            ovc = self._product_overview_content(s, uuid=uuid)
            assert (ovc is not None)
            arrays = self._cached_arrays_for_content(ovc)
            return arrays.data

    def _native_content_for_uuid(self, uuid):
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
        for root, _, files in os.walk(self.cache_dir):
            sz = sum(os.path.getsize(os.path.join(root, name)) for name in files)
            total += sz
            LOG.debug('%d bytes in %s' % (sz, root))

        return total

    def _all_product_uuids(self):
        with self._inventory as s:
            return [q.uuid for q in s.query(Product).all()]

    # ----------------------------------------------------------------------
    def get_info(self, dsi_or_uuid, lod=None):
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

    def get_algebraic_namespace(self, uuid):
        if uuid is None:
            return {}, ""

        with self._inventory as s:
            prod = self._product_with_uuid(s, uuid)
            if prod is None:
                return {}, ""
            symbols = {x.key: x.value for x in prod.symbol}
            code = prod.expression
        return symbols, code

    def _check_cache(self, path):
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
    def product_names_available_in_cache(self):
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

    def remove_all_workspace_content_for_resource_paths(self, paths):
        total = 0
        with self._inventory as s:
            for path in paths:
                rsr_hits = s.query(Resource).filter_by(path=path).all()
                for rsr in rsr_hits:
                    total += self._purge_content_for_resource(rsr, defer_commit=True)
        return total

    def purge_content_for_product_uuids(self, uuids, also_products=False):
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

    def idle(self):
        """Called periodically when application is idle.

        Does a clean-up tasks and returns True if more needs to be done later.
        Time constrained to ~0.1s.
        :return: True/False, whether or not more clean-up needs to be scheduled.

        """
        return False

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

    def import_product_content(self, uuid=None, prod=None, allow_cache=True, **importer_kwargs):
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
        return ac.data

    def create_composite(self, symbols: dict, relation: dict):
        """
        create a layer composite in the workspace
        :param symbols: dictionary of logical-name to uuid
        :param relation: dictionary with information on how the relation is calculated (FUTURE)
        """
        raise NotImplementedError()

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

    @staticmethod
    def _merge_famcat_strings(md_list, key, suffix=None):
        zult = []
        splatter = [md[key].split(':') for md in md_list]
        for pieces in zip(*splatter):
            uniq = set(pieces)
            zult.append(','.join(sorted(uniq)))
        if suffix:
            zult.append(suffix)
        return ':'.join(zult)

    def _get_composite_metadata(self, info, md_list, composite_array):
        """Combine composite dependency metadata in a logical way.

        Args:
            info: initial metadata for the composite
            md_list: list of metadata dictionaries for each input
            composite_array: array representing the final data values of the
                             composite for valid min/max calculations

        Returns: dict of overall metadata (same as `info`)

        """
        if not all(x[Info.PROJ] == md_list[0][Info.PROJ] for x in md_list[1:]):
            raise ValueError("Algebraic inputs must all be the same projection")

        uuid = uuidgen()
        info[Info.UUID] = uuid
        for k in (Info.PLATFORM, Info.INSTRUMENT, Info.SCENE):
            if md_list[0].get(k) is None:
                continue
            if all(x.get(k) == md_list[0].get(k) for x in md_list[1:]):
                info.setdefault(k, md_list[0][k])
        info.setdefault(Info.KIND, Kind.COMPOSITE)
        info.setdefault(Info.SHORT_NAME, '<unknown>')
        info.setdefault(Info.DATASET_NAME, info[Info.SHORT_NAME])
        info.setdefault(Info.UNITS, '1')

        max_meta = max(md_list, key=lambda x: x[Info.SHAPE])
        for k in (Info.PROJ, Info.ORIGIN_X, Info.ORIGIN_Y, Info.CELL_WIDTH, Info.CELL_HEIGHT, Info.SHAPE):
            info[k] = max_meta[k]

        info[Info.VALID_RANGE] = (np.nanmin(composite_array), np.nanmax(composite_array))
        info[Info.CLIM] = (np.nanmin(composite_array), np.nanmax(composite_array))
        info[Info.OBS_TIME] = min([x[Info.OBS_TIME] for x in md_list])
        info[Info.SCHED_TIME] = min([x[Info.SCHED_TIME] for x in md_list])
        # get the overall observation time
        info[Info.OBS_DURATION] = max([
            x[Info.OBS_TIME] + x.get(Info.OBS_DURATION, timedelta(seconds=0)) for x in md_list]) - info[Info.OBS_TIME]

        # generate family and category names
        info[Info.FAMILY] = family = self._merge_famcat_strings(md_list, Info.FAMILY, suffix=info.get(Info.SHORT_NAME))
        info[Info.CATEGORY] = category = self._merge_famcat_strings(md_list, Info.CATEGORY)
        info[Info.SERIAL] = serial = self._merge_famcat_strings(md_list, Info.SERIAL)
        LOG.debug("algebraic product will be {}::{}::{}".format(family, category, serial))

        return info

    def create_algebraic_composite(self, operations, namespace, info=None):
        if not info:
            info = {}

        import ast
        try:
            ops_ast = ast.parse(operations, mode='exec')
            ops = compile(ast.parse(operations, mode='exec'), '<string>', 'exec')
            result_name = ops_ast.body[-1].targets[0].id
        except SyntaxError:
            raise ValueError("Invalid syntax or operations in algebraic layer")

        dep_metadata = {n: self.get_metadata(u) for n, u in namespace.items() if isinstance(u, UUID)}

        # Get every combination of the valid mins and maxes
        # See: https://stackoverflow.com/a/35608701/433202
        names = list(dep_metadata.keys())
        try:
            valid_combos = np.array(np.meshgrid(*tuple(dep_metadata[n][Info.VALID_RANGE] for n in names))).reshape(
                len(names), -1)
        except KeyError:
            badboys = [n for n in names if Info.VALID_RANGE not in dep_metadata[n]]
            LOG.error("missing VALID_RANGE for: {}".format(repr([dep_metadata[n][Info.DISPLAY_NAME] for n in badboys])))
            LOG.error("witness sample: {}".format(repr(dep_metadata[badboys[0]])))
            raise
        valids_namespace = {n: valid_combos[idx] for idx, n in enumerate(names)}
        content = {n: self.get_content(m[Info.UUID]) for n, m in dep_metadata.items()}

        # Get all content in the same shape
        max_shape = max(x[Info.SHAPE] for x in dep_metadata.values())
        for k, v in content.items():
            if v.shape != max_shape:
                f0 = int(max_shape[0] / v.shape[0])
                f1 = int(max_shape[1] / v.shape[1])
                v = np.ma.repeat(np.ma.repeat(v, f0, axis=0), f1, axis=1)
                content[k] = v

        # Run the code: code_object, no globals, copy of locals
        exec(ops, None, valids_namespace)
        if result_name not in valids_namespace:
            raise RuntimeError("Unable to retrieve result '{}' from code execution".format(result_name))

        exec(ops, None, content)
        if result_name not in content:
            raise RuntimeError("Unable to retrieve result '{}' from code execution".format(result_name))
        info = self._get_composite_metadata(info, list(dep_metadata.values()), valids_namespace[result_name])
        # update the shape
        # NOTE: This doesn't work if the code changes the shape of the array
        # Need to update geolocation information too
        # info[Info.SHAPE] = content[result_name].shape

        info = generate_guidebook_metadata(info)

        uuid, info, data = self._create_product_from_array(info, content[result_name],
                                                           namespace=namespace,
                                                           codeblock=operations)
        return uuid, info, data

    def _create_product_from_array(self, info, data, namespace=None, codeblock=None):
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

    def _bgnd_remove(self, uuid):
        from uwsift.queue import TASK_DOING, TASK_PROGRESS
        yield {TASK_DOING: 'purging memory', TASK_PROGRESS: 0.5}
        with self._inventory as s:
            self._deactivate_content_for_product(self._product_with_uuid(s, uuid))
        yield {TASK_DOING: 'purging memory', TASK_PROGRESS: 1.0}

    def remove(self, dsi):
        """Formally detach a dataset.

        Removing its content from the workspace fully by the time that idle() has nothing more to do.

        :param dsi: datasetinfo dictionary or UUID of a dataset
        :return: True if successfully deleted, False if not found
        """
        uuid = dsi if isinstance(dsi, UUID) else dsi[Info.UUID]

        if self._queue is not None:
            self._queue.add(str(uuid), self._bgnd_remove(uuid), 'Purge dataset')
        else:
            # iterate over generator
            list(self._bgnd_remove(uuid))
        return True

    def get_content(self, dsi_or_uuid, lod=None, kind=Kind.IMAGE):
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

    def _create_position_to_index_transform(self, dsi_or_uuid):
        info = self.get_info(dsi_or_uuid)
        origin_x = info[Info.ORIGIN_X]
        origin_y = info[Info.ORIGIN_Y]
        cell_width = info[Info.CELL_WIDTH]
        cell_height = info[Info.CELL_HEIGHT]

        def _transform(x, y, origin_x=origin_x, origin_y=origin_y, cell_width=cell_width, cell_height=cell_height):
            col = (x - info[Info.ORIGIN_X]) / info[Info.CELL_WIDTH]
            row = (y - info[Info.ORIGIN_Y]) / info[Info.CELL_HEIGHT]
            return col, row

        return _transform

    def _create_layer_affine(self, dsi_or_uuid):
        info = self.get_info(dsi_or_uuid)
        affine = Affine(
            info[Info.CELL_WIDTH],
            0.0,
            info[Info.ORIGIN_X],
            0.0,
            info[Info.CELL_HEIGHT],
            info[Info.ORIGIN_Y],
        )
        return affine

    def _position_to_index(self, dsi_or_uuid, xy_pos):
        info = self.get_info(dsi_or_uuid)
        if info is None:
            return None, None
        # Assume `xy_pos` is lon/lat value
        if '+proj=latlong' in info[Info.PROJ]:
            x, y = xy_pos[:2]
        else:
            x, y = Proj(info[Info.PROJ])(*xy_pos)
        col = (x - info[Info.ORIGIN_X]) / info[Info.CELL_WIDTH]
        row = (y - info[Info.ORIGIN_Y]) / info[Info.CELL_HEIGHT]

        return np.int64(np.floor(row)), np.int64(np.floor(col))

    def layer_proj(self, dsi_or_uuid):
        """Project lon/lat probe points to image X/Y"""
        info = self.get_info(dsi_or_uuid)
        return Proj(info[Info.PROJ])

    def _project_points(self, p, points):
        points = np.array(points)
        points[:, 0], points[:, 1] = p(points[:, 0], points[:, 1])
        return points

    def get_content_point(self, dsi_or_uuid, xy_pos):
        row, col = self._position_to_index(dsi_or_uuid, xy_pos)
        if row is None or col is None:
            return None
        data = self.get_content(dsi_or_uuid)
        if not ((0 <= col < data.shape[1]) and (0 <= row < data.shape[0])):
            raise ValueError("X/Y position is outside of image with UUID: %s", dsi_or_uuid)
        return data[row, col]

    def get_content_polygon(self, dsi_or_uuid, points):
        data = self.get_content(dsi_or_uuid)
        trans = self._create_layer_affine(dsi_or_uuid)
        p = self.layer_proj(dsi_or_uuid)
        points = self._project_points(p, points)
        _, data = content_within_shape(data, trans, LinearRing(points))
        return data

    def highest_resolution_uuid(self, *uuids):
        return min([self.get_info(uuid) for uuid in uuids], key=lambda i: i[Info.CELL_WIDTH])[Info.UUID]

    def lowest_resolution_uuid(self, *uuids):
        return max([self.get_info(uuid) for uuid in uuids], key=lambda i: i[Info.CELL_WIDTH])[Info.UUID]

    def get_coordinate_mask_polygon(self, dsi_or_uuid, points):
        data = self.get_content(dsi_or_uuid)
        trans = self._create_layer_affine(dsi_or_uuid)
        p = self.layer_proj(dsi_or_uuid)
        points = self._project_points(p, points)
        index_mask, data = content_within_shape(data, trans, LinearRing(points))
        coords_mask = (index_mask[0] * trans.e + trans.f, index_mask[1] * trans.a + trans.c)
        # coords_mask is (Y, X) corresponding to (rows, cols) like numpy
        coords_mask = p(coords_mask[1], coords_mask[0], inverse=True)[::-1]
        return coords_mask, data

    def get_content_coordinate_mask(self, uuid, coords_mask):
        data = self.get_content(uuid)
        trans = self._create_layer_affine(uuid)
        p = self.layer_proj(uuid)
        # coords_mask is (Y, X) like a numpy array
        coords_mask = p(coords_mask[1], coords_mask[0])[::-1]
        index_mask = (
            np.round((coords_mask[0] - trans.f) / trans.e).astype(np.uint),
            np.round((coords_mask[1] - trans.c) / trans.a).astype(np.uint),
        )
        return data[index_mask]

    def get_pyresample_area(self, uuid, y_slice=None, x_slice=None):
        """Create a pyresample compatible AreaDefinition for this layer."""
        # WARNING: Untested!
        from pyresample.geometry import AreaDefinition
        from pyresample.utils import proj4_str_to_dict
        info = self.get_info(uuid)
        if y_slice is None:
            y_slice = slice(None, None, None)
        if x_slice is None:
            x_slice = slice(None, None, None)
        if y_slice.step not in [1, None] or x_slice.step not in [1, None]:
            raise ValueError("Slice steps other than 1 are not supported")
        rows, cols = info[Info.SHAPE]
        x_start = x_slice.start or 0
        y_start = y_slice.start or 0
        num_cols = (x_slice.stop or cols) - x_start
        num_rows = (y_slice.stop or rows) - y_start
        half_x = info[Info.CELL_WIDTH] / 2.
        half_y = info[Info.CELL_HEIGHT] / 2.
        min_x = info[Info.ORIGIN_X] - half_x + x_start * info[Info.CELL_WIDTH]
        max_y = info[Info.ORIGIN_Y] + half_y - y_start * info[Info.CELL_HEIGHT]
        min_y = max_y - num_rows * info[Info.CELL_HEIGHT]
        max_x = min_x + num_cols * info[Info.CELL_WIDTH]
        return AreaDefinition(
            'layer area',
            'layer area',
            'layer area',
            proj4_str_to_dict(info[Info.PROJ]),
            cols, rows,
            (min_x, min_y, max_x, max_y),
        )

    def __getitem__(self, datasetinfo_or_uuid):
        """
        return science content proxy capable of generating a numpy array when sliced
        :param datasetinfo_or_uuid: metadata or key for the dataset
        :return: sliceable object returning numpy arrays
        """
        pass
