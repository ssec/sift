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
from abc import abstractmethod
from collections import defaultdict
from collections.abc import Mapping as ReadOnlyMapping
from datetime import timedelta
from typing import Mapping, Generator, Tuple, Dict, Optional
from uuid import UUID, uuid1 as uuidgen

import numba as nb
import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal
from pyproj import Proj
from pyresample.geometry import AreaDefinition
from pyresample.utils import proj4_str_to_dict
from rasterio import Affine
from shapely.geometry.polygon import LinearRing

from uwsift.common import Info, Kind, Flags
from uwsift.model.shapes import content_within_shape
from .importer import SatpyImporter, generate_guidebook_metadata
from .metadatabase import Metadatabase, Product, Content, \
    ContentImage, ContentUnstructuredPoints

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
    def _rcls(rows: Optional[int], columns: Optional[int], levels: Optional[int]) \
            -> Tuple[tuple, tuple]:
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

    def _attach(self, c: Content, mode='c'):
        """
        attach content arrays, for holding by workspace in _available
        :param c: Content entity from database
        :return: workspace_data_arrays instance
        """
        if isinstance(c, ContentImage):
            rcl, shape = self._rcls(c.rows, c.cols, c.levels)
        elif isinstance(c, ContentUnstructuredPoints):
            rcl, shape = self._rcls(c.n_points, c.n_dimensions, None)
        else:
            raise NotImplementedError

        self._rcl, self._shape = rcl, shape

        def mm(path, *args, **kwargs):
            full_path = os.path.join(self._wsd, path)
            if not os.access(full_path, os.R_OK):
                LOG.warning("unable to find {}".format(full_path))
                return None
            return np.memmap(full_path, *args, **kwargs)

        self._data = mm(c.path, dtype=c.dtype or np.float32, mode=mode, shape=shape)  # potentially very very large

        if isinstance(c, ContentImage):
            self._y = mm(c.y_path, dtype=c.dtype or np.float32, mode=mode, shape=shape) if c.y_path else None
            self._x = mm(c.x_path, dtype=c.dtype or np.float32, mode=mode, shape=shape) if c.x_path else None
            self._z = mm(c.z_path, dtype=c.dtype or np.float32, mode=mode, shape=shape) if c.z_path else None

            if c.coverage_path:
                _, cshape = self._rcls(c.coverage_cols, c.coverage_cols, c.coverage_levels)
                self._coverage = mm(c.coverage_path, dtype=np.int8, mode=mode, shape=cshape)
            else:
                self._coverage = np.array([1])

            if c.sparsity_path:
                _, sshape = self._rcls(c.coverage_cols, c.coverage_cols, c.coverage_levels)
                self._sparsity = mm(c.sparsity_path, dtype=np.int8, mode=mode, shape=sshape)
            else:
                self._sparsity = np.array([1])


class BaseWorkspace(QObject):
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

    @abstractmethod
    def product_state(self, uuid: UUID) -> Flags:
        pass

    @property
    @abstractmethod
    def _S(self):
        pass

    @property
    @abstractmethod
    def metadatabase(self) -> Metadatabase:
        pass

    @staticmethod
    def default_workspace():
        """
        return the default (global) workspace
        Currently no plans to have more than one workspace, but secondaries may eventually have some advantage.
        :return: Workspace instance
        """
        return TheWorkspace

    def __init__(self, directory_path: str = None):
        """
        Initialize a new or attach an existing workspace, creating any necessary bookkeeping.
        """
        super(BaseWorkspace, self).__init__()

        # HACK: handle old workspace command line flag
        if isinstance(directory_path, (list, tuple)):
            self.cache_dir = cache_path = os.path.abspath(directory_path[1])
            self.cwd = directory_path = os.path.abspath(directory_path[0])
        else:
            self.cwd = directory_path = os.path.abspath(directory_path)
            self.cache_dir = cache_path = os.path.join(self.cwd, 'data_cache')

        self._available = {}
        self._importers = IMPORT_CLASSES.copy()
        self._state = defaultdict(Flags)
        global TheWorkspace  # singleton
        if TheWorkspace is None:
            TheWorkspace = self

        if not os.path.isdir(cache_path):
            LOG.info("creating new workspace cache at {}".format(cache_path))
            os.makedirs(cache_path)
        if not os.path.isdir(directory_path):
            LOG.info("creating new workspace at {}".format(directory_path))
            os.makedirs(directory_path)
            self._own_cwd = True
            self._init_create_workspace()

    @abstractmethod
    def clear_workspace_content(self):
        """Remove binary files from workspace and workspace database."""
        pass

    #
    #  data array handling
    #

    @abstractmethod
    def _activate_content(self, c: Content) -> ActiveContent:
        pass

    @abstractmethod
    def _cached_arrays_for_content(self, c: Content):
        """
        attach cached data indicated in Content, unless it's been attached already and is in _available
        touch the content and product in the database to appease the LRU gods
        :param c: metadatabase Content object for session attached to current thread
        :return: workspace_content_arrays
        """
        pass
    @abstractmethod
    def _deactivate_content_for_product(self, p: Product):
        pass

    #
    # often-used queries
    #

    @abstractmethod
    def _product_with_uuid(self, session, uuid: UUID) -> Product:
        pass

    @abstractmethod
    def _product_overview_content(self, session, prod: Product = None, uuid: UUID = None,
                                  kind: Kind = Kind.IMAGE) -> Optional[Content]:
        pass

    @abstractmethod
    def _product_native_content(self, session, prod: Product = None, uuid: UUID = None,
                                kind: Kind = Kind.IMAGE) -> Optional[Content]:
        pass

    #
    # combining queries with data content
    #

    @abstractmethod
    def _overview_content_for_uuid(self, uuid: UUID, kind: Kind = Kind.IMAGE) \
            -> np.memmap:
        pass

    @abstractmethod
    def _native_content_for_uuid(self, uuid: UUID) -> np.memmap:
        pass

    #
    # workspace file management
    #

    @property
    @abstractmethod
    def _total_workspace_bytes(self):
        pass

    @abstractmethod
    def _all_product_uuids(self) -> list:
        pass

    @abstractmethod
    def get_info(self, dsi_or_uuid, lod=None) -> Optional[frozendict]:
        """
        :param dsi_or_uuid: existing datasetinfo dictionary, or its UUID
        :param lod: desired level of detail to focus
        :return: metadata access with mapping semantics, to be treated as read-only
        """
        pass

    @abstractmethod
    def get_algebraic_namespace(self, uuid: UUID):
        pass

    @property
    @abstractmethod
    def product_names_available_in_cache(self) -> dict:
        """
        Returns: dictionary of {UUID: product name,...}
        typically used for add-from-cache dialog
        """
        pass

    @property
    @abstractmethod
    def uuids_in_cache(self):
        pass

    @abstractmethod
    def recently_used_products(self, n=32) -> Dict[UUID, str]:
        pass

    @abstractmethod
    def remove_all_workspace_content_for_resource_paths(self, paths: list):
        pass

    @abstractmethod
    def purge_content_for_product_uuids(self, uuids: list, also_products=False):
        """
        given one or more product uuids, purge the Content from the cache
        Note: this does not purge any ActiveContent that may still be using the files, but the files will be gone
        Args:
            uuids:

        Returns:

        """
        pass

    @abstractmethod
    def close(self):
        pass

    def bgnd_task_complete(self):
        """
        handle operations that should be done at the end of a threaded background task
        """
        pass

    def idle(self):
        """Called periodically when application is idle.

        Does a clean-up tasks and returns True if more needs to be done later.
        Time constrained to ~0.1s.
        :return: True/False, whether or not more clean-up needs to be scheduled.

        """
        return False

    @abstractmethod
    def get_metadata(self, uuid_or_path):
        """
        return metadata dictionary for a given product or the product being offered by a resource path (see get_info)
        Args:
            uuid_or_path: product uuid, or path to the resource path it lives in

        Returns:
            metadata (Mapping), metadata for the product at this path;
            FUTURE note more than one product may be in a single file
        """
        pass

    @abstractmethod
    def collect_product_metadata_for_paths(self, paths: list,
                                           **importer_kwargs) -> Generator[Tuple[int, frozendict], None, None]:
        """Start loading URI data into the workspace asynchronously.

        Args:
            paths (list): String paths to open and get metadata for
            **importer_kwargs: Keyword arguments to pass to the lower-level
                importer class.

        Returns: sequence of read-only info dictionaries

        """
        pass

    @abstractmethod
    def import_product_content(self, uuid: UUID = None, prod: Product = None,
                               allow_cache=True, merge_target_uuid: Optional[UUID] = None,
                               **importer_kwargs) -> np.memmap:
        pass

    def create_composite(self, symbols: dict, relation: dict):
        """
        create a layer composite in the workspace
        :param symbols: dictionary of logical-name to uuid
        :param relation: dictionary with information on how the relation is calculated (FUTURE)
        """
        raise NotImplementedError()

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
        for k in (Info.PROJ, Info.ORIGIN_X, Info.ORIGIN_Y, Info.CELL_WIDTH,
                  Info.CELL_HEIGHT, Info.SHAPE,
                  Info.GRID_ORIGIN,
                  Info.GRID_FIRST_INDEX_Y,
                  Info.GRID_FIRST_INDEX_X,
                  ):
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

    @abstractmethod
    def _create_product_from_array(self, info: Info, data, namespace=None, codeblock=None) \
            -> Tuple[UUID, Optional[frozendict], np.memmap]:
        pass

    @abstractmethod
    def _bgnd_remove(self, uuid: UUID):
        pass

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

    @abstractmethod
    def get_content(self, dsi_or_uuid, lod=None, kind: Kind = Kind.IMAGE) \
            -> Optional[np.memmap]:
        pass

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

    def _position_to_data_index(self, dsi_or_uuid, xy_pos) -> Tuple[int, int]:
        """Calculate the sift-internal data index from lon/lat values"""
        info = self.get_info(dsi_or_uuid)
        if info is None:
            return None, None
        # Assume `xy_pos` is lon/lat value
        if '+proj=latlong' in info[Info.PROJ]:
            x, y = xy_pos[:2]
        else:
            x, y = Proj(info[Info.PROJ])(*xy_pos)

        column = np.int64(np.floor(
            (x - info[Info.ORIGIN_X]) / info[Info.CELL_WIDTH])
        )
        row = np.int64(np.floor(
            (y - info[Info.ORIGIN_Y]) / info[Info.CELL_HEIGHT])
        )
        return row, column

    def position_to_grid_index(self, dsi_or_uuid, xy_pos) -> Tuple[int, int]:
        """Calculate the satellite grid index from lon/lat values"""
        info = self.get_info(dsi_or_uuid)
        if info is None:
            return None, None

        row, column = self._position_to_data_index(dsi_or_uuid, xy_pos)

        grid_origin = info[Info.GRID_ORIGIN]
        grid_first_index_of_rows = info[Info.GRID_FIRST_INDEX_Y]
        grid_first_index_of_columns = info[Info.GRID_FIRST_INDEX_X]

        rows, columns = info[Info.SHAPE]

        if grid_origin[0].upper() == "S":
            row = rows - 1 - row
        row += grid_first_index_of_rows

        if grid_origin[1].upper() == "E":
            column = columns - 1 - column
        column += grid_first_index_of_columns

        return row, column

    def layer_proj(self, dsi_or_uuid):
        """Project lon/lat probe points to image X/Y"""
        info = self.get_info(dsi_or_uuid)
        return Proj(info[Info.PROJ])

    def _project_points(self, p, points):
        points = np.array(points)
        points[:, 0], points[:, 1] = p(points[:, 0], points[:, 1])
        return points

    def get_content_point(self, dsi_or_uuid, xy_pos):
        row, col = self._position_to_data_index(dsi_or_uuid, xy_pos)
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

    def get_content_coordinate_mask(self, uuid: UUID, coords_mask):
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

    def get_pyresample_area(self, uuid: UUID, y_slice=None, x_slice=None):
        """Create a pyresample compatible AreaDefinition for this layer."""
        # WARNING: Untested!
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

    def find_merge_target(self, uuid: UUID, info) -> Optional[Product]:
        pass

    def get_points_arrays(self, uuid: UUID) \
            -> Tuple[Optional[np.array], Optional[np.array]]:
        """
        Get the DataArrays from a ``POINTS`` product. The first ``DataArray``
        contains the positions of the points. The second array represents the
        attribute.

        :param uuid: UUID of the layer
        :return: Tuple of a position array and maybe an attribute array
        """
        content = self.get_content(uuid, kind=Kind.POINTS)
        if content is None:
            return None, None

        if not (content.ndim == 2 and content.shape[1] in (2, 3)):
            # Try to accept data which is not actually a list of points but may
            # be a list of tuples of points by shaving off everything but the
            # first item of each entry.
            # See vispy.MarkersVisual.set_data() regarding the check criterion.
            return np.hsplit(content, np.array([2]))[0] # TODO when is this called?
        elif content.ndim == 2 and content.shape[1] == 3:
            return np.hsplit(content, [2])
        return content, None

    def get_lines_arrays(self, uuid: UUID) -> Tuple[Optional[np.array], Optional[np.array]]:
        """
        Get the DataArrays from a ``LINES`` product. The first ``DataArray``
        contains positions for the tip and base of the lines. The second array
        represents the attribute.

        :param uuid: UUID of the layer
        :return: Tuple of a lines array and maybe an attribute array
        """
        content = self.get_content(uuid, kind=Kind.LINES)
        if content is None:
            return None, None

        if content.shape[1] > 4:
            content, _ = np.hsplit(content, [4])
        return content, None
