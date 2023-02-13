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
- performing minimized on-demand calculations, e.g. datasets for algebraic layers, in the background
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
from typing import Dict, Generator, Mapping, Optional, Tuple
from uuid import UUID
from uuid import uuid1 as uuidgen

import numpy as np
import xarray
from pyproj import Proj
from PyQt5.QtCore import QObject, pyqtSignal
from rasterio import Affine
from shapely.geometry.polygon import LinearRing

from uwsift.common import FALLBACK_RANGE, Flags, Info, Instrument, Kind, Platform
from uwsift.model.shapes import content_within_shape

from ..util.common import is_same_proj
from .importer import SatpyImporter, generate_guidebook_metadata
from .metadatabase import (
    Content,
    ContentImage,
    ContentMultiChannelImage,
    ContentUnstructuredPoints,
    Product,
)
from .statistics import dataset_statistical_analysis

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


class ActiveContent(QObject):
    """
    ActiveContent composes numpy.memmap arrays with their corresponding Content metadata, and is owned by Workspace
    Purpose: consolidate common operations on content, while factoring in things like sparsity, coverage, y, x, z arrays
    Workspace instantiates ActiveContent from metadatabase Content entries
    """

    def __init__(self, workspace_cwd: str, C: Content, info):
        super(ActiveContent, self).__init__()
        self._cid = C.id  # Content.id database entry I belong to
        self._wsd = workspace_cwd  # full path of workspace
        if workspace_cwd is None and C is None:
            LOG.warning("test initialization of ActiveContent")
            self._test_init()
        else:
            self._attach(C)  # initializes self._data

        # Needed for the calculation of the correct statistics
        # we need a dict not a frozendict so convert it everytime to a dict
        attrs = dict(info)

        # exclude multichannel images from statistics calculation:
        if info.get(Info.KIND) != Kind.MC_IMAGE:
            data_array = xarray.DataArray(self._data, attrs=attrs)
            self.statistics = dataset_statistical_analysis(data_array)
        else:
            self.statistics = {}

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
    def _rcls(rows: Optional[int], columns: Optional[int], levels: Optional[int]) -> Tuple[tuple, tuple]:
        """
        :param rows: rows or None
        :param columns: columns or None
        :param levels: levels or None
        :return: condensed tuple(string with 'rcl', 'rc', 'rl', dimension tuple corresponding to string)
        """
        rcl_shape = tuple((name, dimension) for (name, dimension) in zip("rcl", (rows, columns, levels)) if dimension)
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

    def _attach(self, c: Content, mode="c"):
        """
        attach content arrays, for holding by workspace in _available
        :param c: Content entity from database
        :return: workspace_data_arrays instance
        """
        if isinstance(c, ContentMultiChannelImage):
            rcl, shape = self._rcls(c.rows, c.cols, c.bands)
        elif isinstance(c, ContentImage):
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

        self._data = mm(c.path, dtype=c.dtype or np.float32, mode=mode, shape=shape)  # potentially very, very large

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

    # signals
    # a dataset started importing; generated after overview level of detail is available
    # didStartImport = pyqtSignal(dict)
    # didMakeImportProgress = pyqtSignal(dict)
    didUpdateProductsMetadata = pyqtSignal(set)  # set of UUIDs with changes to their metadata
    # didFinishImport = pyqtSignal(dict)  # all loading activities for a dataset have completed
    # didDiscoverExternalDataset = pyqtSignal(dict)  # a new dataset was added to the workspace from an external agent
    didChangeProductState = pyqtSignal(UUID, Flags)  # a product changed state, e.g. an importer started working on it

    def set_product_state_flag(self, uuid: UUID, flag):
        """primarily used by Importers to signal work in progress"""
        state = self._state[uuid]
        state.add(flag)
        self.didChangeProductState.emit(uuid, state)

    def _clear_product_state_flag(self, uuid: UUID, flag):
        state = self._state[uuid]
        state.remove(flag)
        self.didChangeProductState.emit(uuid, state)

    @property
    @abstractmethod
    def _S(self):
        pass

    def __init__(self, directory_path: str, queue=None):
        """
        Initialize a new or attach an existing workspace, creating any necessary bookkeeping.
        """
        super(BaseWorkspace, self).__init__()

        self._queue = queue
        self.cache_dir = ""
        self.cwd = ""  # directory we work in
        self._own_cwd = (
            False  # whether or not we created the cwd - which is also whether or not we're allowed to destroy it
        )

        # HACK: handle old workspace command line flag
        if isinstance(directory_path, (list, tuple)):
            self.cache_dir = os.path.abspath(directory_path[1])
            self.cwd = os.path.abspath(directory_path[0])
        else:
            self.cwd = os.path.abspath(directory_path)
            self.cache_dir = os.path.join(self.cwd, "data_cache")

        self._available: Dict[int, ActiveContent] = {}  # dictionary of {Content.id : ActiveContent object}
        self._importers = IMPORT_CLASSES.copy()
        self._state: defaultdict = defaultdict(Flags)
        global TheWorkspace  # singleton
        if TheWorkspace is None:
            TheWorkspace = self

        if not os.path.isdir(self.cache_dir):
            LOG.info("creating new workspace cache at {}".format(self.cache_dir))
            os.makedirs(self.cache_dir)
        if not os.path.isdir(self.cwd):
            LOG.info("creating new workspace at {}".format(self.cwd))
            os.makedirs(self.cwd)
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
    def _deactivate_content_for_product(self, p: Optional[Product]):
        pass

    #
    # often-used queries
    #

    @abstractmethod
    def _product_with_uuid(self, session, uuid: UUID) -> Optional[Product]:
        pass

    @abstractmethod
    def _product_overview_content(
        self, session, prod: Optional[Product] = None, uuid: Optional[UUID] = None, kind: Kind = Kind.IMAGE
    ) -> Optional[Content]:
        pass

    @abstractmethod
    def _product_native_content(
        self, session, prod: Optional[Product] = None, uuid: Optional[UUID] = None, kind: Kind = Kind.IMAGE
    ) -> Optional[Content]:
        pass

    #
    # combining queries with data content
    #

    @abstractmethod
    def _overview_content_for_uuid(self, uuid: UUID, kind: Kind = Kind.IMAGE) -> np.memmap:
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
    def get_info(self, info_or_uuid, lod=None) -> Optional[frozendict]:
        """
        :param info_or_uuid: existing datasetinfo dictionary, or its UUID
        :param lod: desired level of detail to focus
        :return: metadata access with mapping semantics, to be treated as read-only
        """
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
        pass

    @abstractmethod
    def import_product_content(
        self,
        uuid: UUID,
        prod: Optional[Product] = None,
        allow_cache=True,
        merge_target_uuid: Optional[UUID] = None,
        **importer_kwargs,
    ) -> np.memmap:
        pass

    @staticmethod
    def _merge_famcat_strings(md_list, key, suffix=None):
        zult = []
        splatter = [md[key].split(":") for md in md_list]
        for pieces in zip(*splatter):
            uniq = set(pieces)
            zult.append(",".join(sorted(uniq)))
        if suffix:
            zult.append(suffix)
        return ":".join(zult)

    def _get_composite_metadata(self, info, md_list, composite_array):
        """Combine composite dependency metadata in a logical way.

        Args:
            info: initial metadata for the composite
            md_list: list of metadata dictionaries for each input
            composite_array: array representing the final data values of the
                             composite for valid min/max calculations

        Returns: dict of overall metadata (same as `info`)

        """
        if not all(is_same_proj(x[Info.PROJ], md_list[0][Info.PROJ]) for x in md_list[1:]):
            raise ValueError("Algebraic inputs must all be the same projection.")

        uuid = uuidgen()
        info[Info.UUID] = uuid

        mixed_info = {
            Info.PLATFORM: Platform.MIXED,
            Info.INSTRUMENT: Instrument.MIXED,
            Info.SCENE: None,
        }
        for k in mixed_info.keys():
            if md_list[0].get(k) is None:
                continue
            if all(x.get(k) == md_list[0].get(k) for x in md_list[1:]):
                info.setdefault(k, md_list[0][k])
            else:
                info.setdefault(k, mixed_info[k])

        info.setdefault(Info.KIND, Kind.COMPOSITE)
        info.setdefault(Info.SHORT_NAME, "<unknown>")
        info.setdefault(Info.DATASET_NAME, info[Info.SHORT_NAME])
        info.setdefault(Info.UNITS, "1")

        max_meta = max(md_list, key=lambda x: x[Info.SHAPE])
        for k in (
            Info.PROJ,
            Info.ORIGIN_X,
            Info.ORIGIN_Y,
            Info.CELL_WIDTH,
            Info.CELL_HEIGHT,
            Info.SHAPE,
            Info.GRID_ORIGIN,
            Info.GRID_FIRST_INDEX_Y,
            Info.GRID_FIRST_INDEX_X,
        ):
            info[k] = max_meta[k]

        info[Info.VALID_RANGE] = (np.nanmin(composite_array), np.nanmax(composite_array))
        info[Info.OBS_TIME] = min([x[Info.OBS_TIME] for x in md_list])
        info[Info.SCHED_TIME] = min([x[Info.SCHED_TIME] for x in md_list])
        # get the overall observation time
        info[Info.OBS_DURATION] = (
            max([x[Info.OBS_TIME] + x.get(Info.OBS_DURATION, timedelta(seconds=0)) for x in md_list])
            - info[Info.OBS_TIME]
        )

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
            ops_ast = ast.parse(operations, mode="exec")
            ops = compile(ast.parse(operations, mode="exec"), "<string>", "exec")
            result_name = ops_ast.body[-1].targets[0].id
        except SyntaxError:
            raise ValueError("Invalid syntax or operations in algebraic layer recipe")

        dep_metadata = {n: self.get_metadata(u) for n, u in namespace.items() if isinstance(u, UUID)}

        # Get every combination of the valid mins and maxes
        # See: https://stackoverflow.com/a/35608701/433202
        names = list(dep_metadata.keys())
        try:
            valid_combos = np.array(
                np.meshgrid(*tuple(self.get_range_for_dataset_no_fail(dep_metadata[n]) for n in names))
            ).reshape(len(names), -1)
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
        exec(ops, None, valids_namespace)  # nosec B102
        if result_name not in valids_namespace:
            raise RuntimeError("Unable to retrieve result '{}' from code execution".format(result_name))

        exec(ops, None, content)  # nosec B102
        if result_name not in content:
            raise RuntimeError("Unable to retrieve result '{}' from code execution".format(result_name))
        info = self._get_composite_metadata(info, list(dep_metadata.values()), valids_namespace[result_name])
        # update the shape
        # NOTE: This doesn't work if the code changes the shape of the array
        # Need to update geolocation information too
        # info[Info.SHAPE] = content[result_name].shape

        info = generate_guidebook_metadata(info)

        uuid, info, data = self._create_product_from_array(
            info, content[result_name], namespace=namespace, codeblock=operations
        )
        return uuid, info, data

    def get_range_for_dataset_no_fail(self, info: dict) -> tuple:
        """Return always a range.
        If possible, it is the valid range from the metadata, otherwise the actual range of the data given by the
        minimum and maximum data values, and if that doesn't work either, the FALLBACK_RANGE"""
        if Info.VALID_RANGE in info:
            return info[Info.VALID_RANGE]

        actual_range = self.get_min_max_value_for_dataset_by_uuid(info[Info.UUID])

        if actual_range:
            return actual_range

        return FALLBACK_RANGE

    @abstractmethod
    def _create_product_from_array(
        self, info: Mapping, data, namespace=None, codeblock=None
    ) -> Tuple[UUID, Optional[frozendict], np.memmap]:
        pass

    @abstractmethod
    def _bgnd_remove(self, uuid: UUID):
        pass

    def remove(self, info_or_uuid):
        """Formally detach a dataset.

        Removing its content from the workspace fully by the time that idle() has nothing more to do.

        :param info_or_uuid: datasetinfo dictionary or UUID of a dataset
        :return: True if successfully deleted, False if not found
        """
        uuid = info_or_uuid if isinstance(info_or_uuid, UUID) else info_or_uuid[Info.UUID]

        if self._queue is not None:
            self._queue.add(str(uuid), self._bgnd_remove(uuid), "Purge dataset")
        else:
            # iterate over generator
            list(self._bgnd_remove(uuid))
        return True

    @abstractmethod
    def get_content(self, info_or_uuid, lod=None, kind: Kind = Kind.IMAGE) -> Optional[np.memmap]:
        pass

    def _create_dataset_affine(self, info_or_uuid):
        info = self.get_info(info_or_uuid)
        affine = Affine(
            info[Info.CELL_WIDTH],
            0.0,
            info[Info.ORIGIN_X],
            0.0,
            info[Info.CELL_HEIGHT],
            info[Info.ORIGIN_Y],
        )
        return affine

    def _position_to_data_index(self, info_or_uuid, xy_pos) -> Tuple[Optional[int], Optional[int]]:
        """Calculate the sift-internal data index from lon/lat values"""
        info = self.get_info(info_or_uuid)
        if info is None:
            return None, None
        # Assume `xy_pos` is lon/lat value
        if "+proj=latlong" in info[Info.PROJ]:
            x, y = xy_pos[:2]
        else:
            x, y = Proj(info[Info.PROJ])(*xy_pos)

        column = np.int64(np.floor((x - info[Info.ORIGIN_X]) / info[Info.CELL_WIDTH]))
        row = np.int64(np.floor((y - info[Info.ORIGIN_Y]) / info[Info.CELL_HEIGHT]))
        return row, column

    def position_to_grid_index(self, info_or_uuid, xy_pos) -> Tuple[Optional[int], Optional[int]]:
        """Calculate the satellite grid index from lon/lat values"""
        info = self.get_info(info_or_uuid)
        if info is None:
            return None, None

        row, column = self._position_to_data_index(info_or_uuid, xy_pos)

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

    def dataset_proj(self, info_or_uuid):
        """Project lon/lat probe points to image X/Y"""
        info = self.get_info(info_or_uuid)
        return Proj(info[Info.PROJ])

    def _project_points(self, p, points):
        points = np.array(points)
        points[:, 0], points[:, 1] = p(points[:, 0], points[:, 1])
        return points

    def get_content_point(self, info_or_uuid, xy_pos):
        row, col = self._position_to_data_index(info_or_uuid, xy_pos)
        if row is None or col is None:
            return None
        data = self.get_content(info_or_uuid)
        if not ((0 <= col < data.shape[1]) and (0 <= row < data.shape[0])):
            raise ValueError("X/Y position is outside of image with UUID: %s", info_or_uuid)
        return data[row, col]

    def get_content_polygon(self, info_or_uuid, points):
        data = self.get_content(info_or_uuid)
        trans = self._create_dataset_affine(info_or_uuid)
        p = self.dataset_proj(info_or_uuid)
        points = self._project_points(p, points)
        _, data = content_within_shape(data, trans, LinearRing(points))
        return data

    def lowest_resolution_uuid(self, *uuids):
        return max([self.get_info(uuid) for uuid in uuids], key=lambda i: i[Info.CELL_WIDTH])[Info.UUID]

    def get_coordinate_mask_polygon(self, info_or_uuid, points):
        data = self.get_content(info_or_uuid)
        trans = self._create_dataset_affine(info_or_uuid)
        p = self.dataset_proj(info_or_uuid)
        points = self._project_points(p, points)
        index_mask, data = content_within_shape(data, trans, LinearRing(points))
        coords_mask = (index_mask[0] * trans.e + trans.f, index_mask[1] * trans.a + trans.c)
        # coords_mask is (Y, X) corresponding to (rows, cols) like numpy
        coords_mask = p(coords_mask[1], coords_mask[0], inverse=True)[::-1]
        return coords_mask, data

    def get_content_coordinate_mask(self, uuid: UUID, coords_mask):
        data = self.get_content(uuid)
        assert data is not None  # nosec B101 # suppress mypy [index]
        trans = self._create_dataset_affine(uuid)
        p = self.dataset_proj(uuid)
        # coords_mask is (Y, X) like a numpy array
        coords_mask = p(coords_mask[1], coords_mask[0])[::-1]
        index_mask = (
            np.round((coords_mask[0] - trans.f) / trans.e).astype(np.uint),
            np.round((coords_mask[1] - trans.c) / trans.a).astype(np.uint),
        )
        return data[index_mask]

    def __getitem__(self, datasetinfo_or_uuid):
        """
        return science content proxy capable of generating a numpy array when sliced
        :param datasetinfo_or_uuid: metadata or key for the dataset
        :return: sliceable object returning numpy arrays
        """
        pass

    def find_merge_target(self, uuid: UUID, paths, info) -> Optional[Product]:
        pass

    def get_points_arrays(self, uuid: UUID) -> Tuple[Optional[np.array], Optional[np.array]]:
        """
        Get the DataArrays from a ``POINTS`` product. The first ``DataArray``
        contains the positions of the points. The second array represents the
        attribute.

        :param uuid: UUID of the dataset
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
            return np.hsplit(content, np.array([2]))[0]  # TODO when is this called?
        elif content.ndim == 2 and content.shape[1] == 3:
            return np.hsplit(content, [2])
        return content, None

    def get_lines_arrays(self, uuid: UUID) -> Tuple[Optional[np.array], Optional[np.array]]:
        """
        Get the DataArrays from a ``LINES`` product. The first ``DataArray``
        contains positions for the tip and base of the lines. The second array
        represents the attribute.

        :param uuid: UUID of the dataset
        :return: Tuple of a lines array and maybe an attribute array
        """
        content = self.get_content(uuid, kind=Kind.LINES)
        if content is None:
            return None, None

        if content.shape[1] > 4:
            content, _ = np.hsplit(content, [4])
        return content, None

    @abstractmethod
    def _get_active_content_by_uuid(self, uuid: UUID) -> Optional[ActiveContent]:
        pass

    def get_statistics_for_dataset_by_uuid(self, uuid: UUID) -> dict:
        ac = self._get_active_content_by_uuid(uuid)
        if ac:
            stats = ac.statistics
        else:
            stats = {}
        return stats

    def get_min_max_value_for_dataset_by_uuid(self, uuid: UUID):
        """Return the minimum and maximum value of a dataset given by its UUID.

        Falls back to calculate these values if the minimum and maximum are not stored.
        The UUID must identify an existing dataset.
        """
        assert uuid is not None  # nosec B101
        ac = self._get_active_content_by_uuid(uuid)
        assert ac is not None  # nosec B101
        stats = ac.statistics

        if not stats:
            LOG.debug("Could not determine 'min/max' values: dataset has no computed statistics.")
            return None, None

        stats_values = stats.get("stats")

        if isinstance(stats_values, dict):
            min_ranges = stats_values.get("min")
            max_ranges = stats_values.get("max")
        else:
            # TODO: The following is a workaround for a missing concept for color mapping of categorial data and
            #  should be revised!
            # We seem to have categorial data (a dataset with "flag_{values,meanings,masks}") where the values
            # stored are numbers but have no numerical meaning, only that of an identifier.
            # Currently, for technical reasons, we need to be able to get a value range (i.e. a kind of min/max
            # values) even for such a dataset, otherwise no colormap could be applied automatically.
            # So, we trick the statistics module to compute everything as if the data was normal data:
            # To achieve this we simply don't provide the xarr.attrs which the statistics module uses to distinguish
            # categorial from normal data:
            dataarray = xarray.DataArray(ac.data)
            stats = dataset_statistical_analysis(dataarray)
            min_ranges = stats.get("stats").get("min")
            max_ranges = stats.get("stats").get("max")

        if not min_ranges or not max_ranges:  # Note: bool([0]) == True!
            LOG.error("Could not determine 'min/max' values: dataset statistics are invalid.")
            return None, None

        return min_ranges[0], max_ranges[0]
