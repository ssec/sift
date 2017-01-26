#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
workspace.py
~~~~~~~~~~~~

PURPOSE
Implement Workspace, a singleton object which manages large amounts of data
- background loading, up to and including reprojection
- providing memory-compatible, stride-able arrays
- accepting data from external sources written in arbitrary languages

REFERENCES


REQUIRES


:author: R.K.Garcia <rayg@ssec.wisc.edu>
:copyright: 2014 by University of Wisconsin Regents, see AUTHORS for more details
:license: GPLv3, see LICENSE for more details
"""
__author__ = 'rayg'
__docformat__ = 'reStructuredText'

import os, sys, re
import logging, unittest, argparse
import gdal, osr
import numpy as np
import shutil
import netCDF4 as nc4
from collections import namedtuple
from pickle import dump, load, HIGHEST_PROTOCOL
from uuid import UUID, uuid1 as uuidgen
from sift.common import KIND, INFO
from PyQt4.QtCore import QObject, pyqtSignal
from sift.model.shapes import content_within_shape
from sift.workspace.goesr_pug import PugL1bTools
from shapely.geometry.polygon import LinearRing
from rasterio import Affine
from pyproj import Proj

LOG = logging.getLogger(__name__)

DEFAULT_WORKSPACE_SIZE = 256
MIN_WORKSPACE_SIZE = 8

import_progress = namedtuple('import_progress', ['uuid', 'stages', 'current_stage', 'completion', 'stage_desc', 'dataset_info', 'data'])
# stages:int, number of stages this import requires
# current_stage:int, 0..stages-1 , which stage we're on
# completion:float, 0..1 how far we are along on this stage
# stage_desc:tuple(str), brief description of each of the stages we'll be doing


# first instance is main singleton instance; don't preclude the possibility of importing from another workspace later on
TheWorkspace = None


class WorkspaceImporter(object):
    """
    Instances of this class are typically singletons owned by Workspace.
    They're used to perform background activity for importing large input files.
    """
    def __init__(self, **kwargs):
        super(WorkspaceImporter, self).__init__()
    
    def is_relevant(self, source_path=None, source_uri=None):
        """
        return True if this importer is capable of reading this URI.
        """
        return False

    def __call__(self, dest_workspace, dest_wd, dest_uuid, source_path=None, source_uri=None, cache_path=None, **kwargs):
        """
        Yield a series of import_status tuples updating status of the import.
        Typically this is going to run on TheQueue when possible.
        :param dest_cwd: destination directory to place flat files into, may be anywhere inside workspace.cwd
        :param dest_uuid: uuid key to use in reference to this dataset at all LODs - may/not be used in file naming, but should be included in datasetinfo
        :param source_uri: uri to load from
        :param source_path: path to load from (alternative to source_uri)
        :param cache_path: preferred cache path to place data into
        :return: sequence of import_progress, the first and last of which must include data,
                 inbetween updates typically will release data when stages complete and have None for dataset_info and data fields
        """
        raise NotImplementedError('subclass must implement')


class GeoTiffImporter(WorkspaceImporter):
    """
    GeoTIFF data importer
    """
    def __init__(self, **kwargs):
        super(GeoTiffImporter, self).__init__()

    def is_relevant(self, source_path=None, source_uri=None):
        source = source_path or source_uri
        return True if (source.lower().endswith('.tif') or source.lower().endswith('.tiff')) else False

    def __call__(self, dest_workspace, dest_wd, dest_uuid, source_path=None, source_uri=None, cache_path=None, **kwargs):
        # yield successive levels of detail as we load
        if source_uri is not None:
            raise NotImplementedError("GeoTiffImporter cannot read from URIs yet")
        d = {}
        nc = nc4Dataset(source_path)

        # FIXME: consider yielding status at this point so our progress bar starts moving

        ox, cw, _, oy, _, ch = gtiff.GetGeoTransform()
        d[INFO.PROJ] = self._proj4_string_from_nc(nc)
        d[INFO.UUID] = dest_uuid
        d[INFO.KIND] = KIND.IMAGE
        d[INFO.ORIGIN_X] = ox
        d[INFO.ORIGIN_Y] = oy
        d[INFO.CELL_WIDTH] = cw
        d[INFO.CELL_HEIGHT] = ch
        # FUTURE: Should the Workspace normalize all input data or should the Image Layer handle any projection?
        srs = osr.SpatialReference()
        srs.ImportFromWkt(gtiff.GetProjection())
        d[INFO.PROJ] = srs.ExportToProj4().strip()  # remove extra whitespace

        d[INFO.NAME] = os.path.split(source_path)[-1]
        d[INFO.PATHNAME] = source_path

        # FIXME: read this into a numpy.memmap backed by disk in the workspace
        band = gtiff.GetRasterBand(1)  # FUTURE may be an assumption
        shape = rows, cols = band.YSize, band.XSize
        blockw, blockh = band.GetBlockSize()  # non-blocked files will report [band.XSize,1]

        # Fix PROJ4 string if it needs an "+over" parameter
        p = Proj(d[INFO.PROJ])
        lon_l, lat_u = p(ox, oy, inverse=True)
        lon_r, lat_b = p(ox + cw * cols, oy + ch * rows, inverse=True)
        if "+over" not in d[INFO.PROJ] and lon_r < lon_l:
            LOG.debug("Add '+over' to geotiff PROJ.4 because it seems to cross the anti-meridian")
            d[INFO.PROJ] += " +over"

        bandtype = gdal.GetDataTypeName(band.DataType)
        if bandtype.lower()!='float32':
            LOG.warning('attempting to read geotiff files with non-float32 content')
        # Full resolution shape
        # d["shape"] = self.get_dataset_data(item, time_step).shape
        d[INFO.SHAPE] = shape

        # shovel that data into the memmap incrementally
        # http://geoinformaticstutorial.blogspot.com/2012/09/reading-raster-data-with-python-and-gdal.html
        fp = open(cache_path, 'wb+')
        img_data = np.memmap(fp, dtype=np.float32, shape=shape, mode='w+')
        # load at an increment that matches the file's tile size if possible
        IDEAL_INCREMENT = 512.0
        increment = min(blockh * int(np.ceil(IDEAL_INCREMENT/blockh)), 2048)
        # FUTURE: consider explicit block loads using band.ReadBlock(x,y) once
        irow = 0
        while irow < rows:
            nrows = min(increment, rows-irow)
            row_data = band.ReadAsArray(0, irow, cols, nrows)
            img_data[irow:irow+nrows,:] = np.require(row_data, dtype=np.float32)
            irow += increment
            status = import_progress(uuid=dest_uuid,
                                       stages=1,
                                       current_stage=0,
                                       completion=float(irow)/float(rows),
                                       stage_desc="importing geotiff",
                                       dataset_info=d,
                                       data=img_data)
            yield status

        # img_data = gtiff.GetRasterBand(1).ReadAsArray()
        # img_data = np.require(img_data, dtype=np.float32, requirements=['C'])  # FIXME: is this necessary/correct?
        # normally we would place a numpy.memmap in the workspace with the content of the geotiff raster band/s here

        # single stage import with all the data for this simple case
        zult = import_progress(uuid=dest_uuid,
                               stages=1,
                               current_stage=0,
                               completion=1.0,
                               stage_desc="done loading geotiff",
                               dataset_info=d,
                               data=img_data)
        yield zult
        # further yields would logically add levels of detail with their own sampling values
        # FIXME: provide example of multiple LOD loading and how datasetinfo dictionary/dictionaries look in that case
        # note that once the coarse data is yielded, we may be operating in another thread - think about that for now?




class GoesRPUGImporter(WorkspaceImporter):
    """
    Import from PUG format GOES-16 netCDF4 files
    """
    def __init__(self, **kwargs):
        super(GoesRPUGImporter, self).__init__()

    def is_relevant(self, source_path=None, source_uri=None):
        source = source_path or source_uri
        return True if (source.lower().endswith('.nc') or source.lower().endswith('.nc4')) else False

    def __call__(self, dest_workspace, dest_wd, dest_uuid, source_path=None, source_uri=None, cache_path=None, **kwargs):
        # yield successive levels of detail as we load
        if source_uri is not None:
            raise NotImplementedError("GoesRPUGImporter cannot read from URIs yet")

        #
        # step 1: get any additional metadata and an overview tile
        #

        d = {}
        nc = nc4.Dataset(source_path)
        pug = PugL1bTools(nc)


        d[INFO.UUID] = dest_uuid
        d[INFO.NAME] = os.path.split(source_path)[-1]
        d[INFO.PATHNAME] = source_path
        d[INFO.KIND] = KIND.IMAGE
        d[INFO.PROJ] = pug.proj4_string

        # FUTURE: consider yielding status at this point so our progress bar starts moving

        # FIXME: what are these values?
        d[INFO.ORIGIN_X] = ox
        d[INFO.ORIGIN_Y] = oy
        d[INFO.CELL_WIDTH] = cw
        d[INFO.CELL_HEIGHT] = ch

        bandtype = np.float32
        shape = rows, cols = pug.shape
        d[INFO.SHAPE] = shape
        d[INFO.DISPLAY_TIME] = pug.display_time

        bt_or_refl, image, units = pug.convert_from_nc(nc)  # FIXME expensive
        # overview_image = fixme  # FIXME, we need a properly navigated overview image here

        # we got some metadata, let's yield progress
        # yield    import_progress(uuid=dest_uuid,
        #                          stages=1,
        #                          current_stage=0,
        #                          completion=1.0/3.0,
        #                          stage_desc="calculating imagery",
        #                          dataset_info=d,
        #                          data=image)

        #
        # step 2: read and convert the image data
        #   - in chunks if it's a huge image so we can show progress and/or cancel
        #   - push the data into a workspace memmap
        #   - record the content information in the workspace metadatabase
        #

        # shovel that data into the memmap incrementally
        # FUTURE as we're doing so, also update coverage array (showing what sections of data are loaded)
        # FUTURE and for some cases the sparsity array, if the data is interleaved (N/A for NetCDF imagery)

        fp = open(cache_path, 'wb+')
        img_data = np.memmap(fp, dtype=np.float32, shape=shape, mode='w+')
        img_data[:] = np.ma.fix_invalid(image, copy=False, fill_value=np.NAN)  # FIXME: expensive

        # FUTURE: workspace content can be int16 with conversion coefficients applied on the fly, after feature-matrix-model goes in

        # FIXME: for test purpose just brute-force the whole thing into place instead of doing incremental partial-coverage loads
        kind, data, unit = pug.convert_from_nc(nc)   # hopefully only a few seconds... but not good in long term to leave this

        yield import_progress(uuid=dest_uuid,
                             stages=1,
                             current_stage=0,
                             completion=2.0/3.0,
                             stage_desc="GOES PUG data add to workspace",
                             dataset_info=d,
                             data=img_data)

        # # even worse, copying data over to memmap-on-disk
        # img_data[:] = data

        # yield import_progress(uuid=dest_uuid,
        #                      stages=1,
        #                      current_stage=0,
        #                      completion=1.0,
        #                      stage_desc="done importing GOESR pug",
        #                      dataset_info=d,
        #                      data=img_data)


class Workspace(QObject):
    """
    Workspace is a singleton object which works with Datasets shall:
    - own a working directory full of recently used datasets
    - provide DatasetInfo dictionaries for shorthand use between application subsystems
    -- datasetinfo dictionaries are ordinary python dictionaries containing [INFO.UUID], projection metadata, LOD info
    - identify datasets primarily with a UUID object which tracks the dataset and its various representations through the system
    - unpack data in "packing crate" formats like NetCDF into memory-compatible flat files
    - efficiently create on-demand subsections and strides of raster data as numpy arrays
    - incrementally cache often-used subsections and strides ("image pyramid") using appropriate tools like gdal
    - notify subscribers of changes to datasets (Qt signal/slot pub-sub)
    - during idle, clean out unused/idle data content, given DatasetInfo contents provides enough metadata to recreate
    - interface to external data processing or loading plug-ins and notify application of new-dataset-in-workspace

    FIXME: deal with non-basic datasets (composites)
    """
    cwd = None  # directory we work in
    _own_cwd = None  # whether or not we created the cwd - which is also whether or not we're allowed to destroy it
    _pool = None  # process pool that importers can use for background activities, if any
    _importers = None  # list of importers to consult when asked to start an import
    _info = None
    _data = None
    _inventory = None  # dictionary of data
    _composite_inventory = None  # dicionary of composite datasets: { uuid: (symbols, relation, info) }
    _inventory_path = None  # filename to store and load inventory information (simple cache)
    _tempdir = None  # TemporaryDirectory, if it's needed (i.e. a directory name was not given)
    _max_size_gb = None  # maximum size in gigabytes of flat files we cache in the workspace
    _queue = None

    # signals
    didStartImport = pyqtSignal(dict)  # a dataset started importing; generated after overview level of detail is available
    didMakeImportProgress = pyqtSignal(dict)
    didUpdateDataset = pyqtSignal(dict)  # partial completion of a dataset import, new datasetinfo dict is released
    didFinishImport = pyqtSignal(dict)  # all loading activities for a dataset have completed
    didDiscoverExternalDataset = pyqtSignal(dict)  # a new dataset was added to the workspace from an external agent

    IMPORT_CLASSES = [ GeoTiffImporter, GoesRPUGImporter ]

    @staticmethod
    def defaultWorkspace():
        """
        return the default (global) workspace
        Currently no plans to have more than one workspace, but secondaries may eventually have some advantage.
        :return: Workspace instance
        """
        return TheWorkspace

    def __init__(self, directory_path=None, process_pool=None, max_size_gb=None, queue=None):
        """
        Initialize a new or attach an existing workspace, creating any necessary bookkeeping.
        """
        super(Workspace, self).__init__()

        self._max_size_gb = max_size_gb if max_size_gb is not None else DEFAULT_WORKSPACE_SIZE
        if self._max_size_gb < MIN_WORKSPACE_SIZE:
            self._max_size_gb = MIN_WORKSPACE_SIZE
            LOG.warning('setting workspace size to %dGB' % self._max_size_gb)
        if directory_path is None:
            import tempfile
            self._tempdir = tempfile.TemporaryDirectory()
            directory_path = str(self._tempdir)
            LOG.info('using temporary directory {}'.format(directory_path))
        self.cwd = directory_path = os.path.abspath(directory_path)
        self._inventory_path = os.path.join(self.cwd, 'inventory.pkl')
        self._composite_inventory = {}
        if not os.path.isdir(directory_path):
            os.makedirs(directory_path)
            self._own_cwd = True
            self._init_create_workspace()
        else:
            self._own_cwd = False
            self._init_inventory_existing_datasets()
        self._data = {}
        self._info = {}
        self._importers = [x() for x in self.IMPORT_CLASSES]
        global TheWorkspace  # singleton
        if TheWorkspace is None:
            TheWorkspace = self

    def _init_create_workspace(self):
        """
        initialize a previously empty workspace
        :return:
        """
        self._inventory = {}
        self._store_inventory()

    def _init_inventory_existing_datasets(self):
        """
        Do an inventory of an pre-existing workspace
        FIXME: go through and check that everything in the workspace makes sense
        FIXME: check workspace subdirectories for helper sockets and mmaps
        :return:
        """
        if os.path.exists(self._inventory_path):
            with open(self._inventory_path, 'rb') as fob:
                self._inventory = load(fob)
        else:
            self._init_create_workspace()

    @staticmethod
    def _key_for_path(path):
        if not os.path.exists(path):
            return None
        s = os.stat(path)
        return (os.path.realpath(path), s.st_mtime, s.st_size)

    def _store_inventory(self):
        """
        write inventory dictionary to an inventory.pkl file in the cwd
        :return:
        """
        atomic = self._inventory_path + '-tmp'
        with open(atomic, 'wb') as fob:
            dump(self._inventory, fob, HIGHEST_PROTOCOL)
        shutil.move(atomic, self._inventory_path)

    def _check_cache(self, path):
        """
        :param path: file we're checking
        :return: uuid, info, overview_content if the data is already available without import
        """
        key = self._key_for_path(path)
        nfo = self._inventory.get(key, None)
        if nfo is None:
            return None
        uuid, info, data_info = nfo
        dfilename, dtype, shape = data_info
        dpath = os.path.join(self.cwd, dfilename)
        if not os.path.exists(dpath):
            del self._inventory[key]
            self._store_inventory()
            return None
        data = np.memmap(dpath, dtype=dtype, mode='c', shape=shape)
        return uuid, info, data

    @property
    def paths_in_cache(self):
        return [x[0] for x in self._inventory.keys()]

    def _update_cache(self, path, uuid, info, data):
        """
        add or update the cache and put it to disk
        :param path: path to get key from
        :param uuid: uuid the data's been assigned
        :param info: dataset info dictionary
        :param data: numpy.memmap backed by a file in the workspace
        :return:
        """
        key = self._key_for_path(path)
        data_info = (os.path.split(data.filename)[1], data.dtype, data.shape)
        nfo = (uuid, info, data_info)
        self._inventory[key] = nfo
        self._store_inventory()

    def remove_paths_from_cache(self, paths):
        keys = [self._key_for_path(path) for path in paths]
        for key in keys:
            entry = self._inventory.get(key, None)
            if not entry:
                continue
            uuid,info,data_info = entry
            filename, dtype, shape = data_info
            path = os.path.join(self.cwd, filename)
            try:
                os.remove(path)
                LOG.info('removed {} = {} from cache'.format(key[0], path))
            except:
                # this could happen if the file is open in windows
                LOG.error('unable to remove {} from cache'.format(path))
                continue
            del self._inventory[key]
        self._store_inventory()

    def _inventory_check(self):
        "return revised_inventory_dict, [(access-time, size, path, key), ...], total_size"
        cache = []
        inv = dict(self._inventory)
        total_size = 0
        for key,nfo in self._inventory.items():
            uuid,info,data_info = nfo
            filename, dtype, shape = data_info
            path = os.path.join(self.cwd, filename)
            if os.path.exists(path):
                st = os.stat(path)
                cache.append((st.st_atime, st.st_size, path, key))
                total_size += st.st_size
            else:
                LOG.info('removing stale {}'.format(key))
                del inv[key]
        # sort by atime
        cache.sort()
        return inv, cache, total_size

    def recently_used_cache_paths(self, n=32):
        inv, cache, total_size = self._inventory_check()
        self._inventory = inv
        self._store_inventory()
        # get from most recently used end of list
        return [q[3][0] for q in cache[-n:]]

    def _clean_cache(self):
        """
        find stale content in the cache and get rid of it
        this routine should eventually be compatible with backgrounding on a thread
        possibly include a workspace setting for max workspace size in bytes?
        :return:
        """
        # get information on current cache contents
        LOG.info("cleaning cache")
        inv, cache, total_size = self._inventory_check()
        GB = 1024**3
        LOG.info("total cache size is {}GB of max {}GB".format(total_size/GB, self._max_size_gb))
        max_size = self._max_size_gb * GB
        while total_size > max_size:
            _, size, path, key = cache.pop(0)
            LOG.debug('{} GB in {}'.format(path, size/GB))
            try:
                os.remove(path)
                LOG.info('removed {} for {}GB'.format(path, size/GB))
            except:
                # this could happen if the file is open in windows
                LOG.error('unable to remove {} from cache'.format(path))
                continue
            del inv[key]
            total_size -= size
        self._inventory = inv
        self._store_inventory()
        # FUTURE: check for orphan files in the cache
        return

    def close(self):
        self._clean_cache()

    def idle(self):
        """
        Called periodically when application is idle. Does a clean-up tasks and returns True if more needs to be done later.
        Time constrained to ~0.1s.
        :return: True/False, whether or not more clean-up needs to be scheduled.
        """
        return False

    def import_image(self, source_path=None, source_uri=None, allow_cache=True):
        """
        Start loading URI data into the workspace asynchronously.

        :param source_path:
        :return:
        """
        if source_uri is not None and source_path is None:
            raise NotImplementedError('URI load not yet supported')

        nfo = None
        if allow_cache and source_path is not None:
            nfo = self._check_cache(source_path)
            if nfo is not None:
                uuid, info, data = nfo
                self._info[uuid] = info
                self._data[uuid] = data
                return nfo

        gen = None
        uuid = uuidgen()
        for imp in self._importers:
            if imp.is_relevant(source_path=source_path):
                gen = imp(self, self.cwd, uuid, source_path=source_path, cache_path=self._preferred_cache_path(uuid))
                break
        if gen is None:
            raise IOError("unable to import {}".format(source_path))

        # FIXME: for now, just iterate the incremental load. later we want to add this to TheQueue and update the UI as we get more data loaded
        for update in gen:
            if update.data is not None:
                info = self._info[uuid] = update.dataset_info
                data = self._data[uuid] = update.data
                LOG.info("{} {}: {:.01f}%".format(update.dataset_info[INFO.NAME], update.stage_desc, update.completion*100.0))
        # copy the data into an anonymous memmap
        self._data[uuid] = data = self._convert_to_memmap(str(uuid), data)
        if allow_cache:
            self._update_cache(source_path, uuid, info, data)
        # TODO: schedule cache cleaning in background after a series of imports completes
        return uuid, info, data

    def create_composite(self, symbols:dict, relation:dict):
        """
        create a layer composite in the workspace
        :param symbols: dictionary of logical-name to uuid
        :param relation: dictionary with information on how the relation is calculated (FUTURE)
        """


    def _preferred_cache_path(self, uuid):
        filename = str(uuid)
        return os.path.join(self.cwd, filename)

    def _convert_to_memmap(self, uuid, data:np.ndarray):
        if isinstance(data, np.memmap):
            return data
        # from tempfile import TemporaryFile
        # fp = TemporaryFile()
        pathname = self._preferred_cache_path(uuid)
        fp = open(pathname, 'wb+')
        mm = np.memmap(fp, dtype=data.dtype, shape=data.shape, mode='w+')
        mm[:] = data[:]
        return mm

    def _bgnd_remove(self, uuid):
        from sift.queue import TASK_DOING, TASK_PROGRESS
        yield {TASK_DOING: 'purging memory', TASK_PROGRESS: 0.5}
        if uuid in self._info:
            del self._info[uuid]
            zult = True
        if uuid in self._data:
            del self._data[uuid]
            zult = True
        yield {TASK_DOING: 'purging memory', TASK_PROGRESS: 1.0}

    def remove(self, dsi):
        """
        Formally detach a dataset, removing its content from the workspace fully by the time that idle() has nothing more to do.
        :param dsi: datasetinfo dictionary or UUID of a dataset
        :return: True if successfully deleted, False if not found
        """
        if isinstance(dsi, dict):
            name = dsi[INFO.NAME]
        else:
            name = 'dataset'
        uuid = dsi if isinstance(dsi, UUID) else dsi[INFO.UUID]
        zult = False

        if self._queue is not None:
            self._queue.add(str(uuid), self._bgnd_remove(uuid), 'Purge dataset')
        else:
            for blank in self._bgnd_remove(uuid):
                pass
        return True

    def get_info(self, dsi_or_uuid, lod=None):
        """
        :param dsi_or_uuid: existing datasetinfo dictionary, or its UUID
        :param lod: desired level of detail to focus
        :return:
        """
        if isinstance(dsi_or_uuid, str):
            dsi_or_uuid = UUID(dsi_or_uuid)
        return self._info.get(dsi_or_uuid, None)

    def get_content(self, dsi_or_uuid, lod=None):
        """
        :param dsi_or_uuid: existing datasetinfo dictionary, or its UUID
        :param lod: desired level of detail to focus
        :return:
        """
        if isinstance(dsi_or_uuid, str):
            dsi_or_uuid = UUID(dsi_or_uuid)
        return self._data.get(dsi_or_uuid, None)

    def _create_position_to_index_transform(self, dsi_or_uuid):
        info = self.get_info(dsi_or_uuid)
        origin_x = info[INFO.ORIGIN_X]
        origin_y = info[INFO.ORIGIN_Y]
        cell_width = info[INFO.CELL_WIDTH]
        cell_height = info[INFO.CELL_HEIGHT]
        def _transform(x, y, origin_x=origin_x, origin_y=origin_y, cell_width=cell_width, cell_height=cell_height):
            col = (x - info[INFO.ORIGIN_X]) / info[INFO.CELL_WIDTH]
            row = (y - info[INFO.ORIGIN_Y]) / info[INFO.CELL_HEIGHT]
            return col, row
        return _transform

    def _create_layer_affine(self, dsi_or_uuid):
        info = self.get_info(dsi_or_uuid)
        affine = Affine(
            info[INFO.CELL_WIDTH],
            0.0,
            info[INFO.ORIGIN_X],
            0.0,
            info[INFO.CELL_HEIGHT],
            info[INFO.ORIGIN_Y],
        )
        return affine

    def _position_to_index(self, dsi_or_uuid, xy_pos):
        info = self.get_info(dsi_or_uuid)
        if info is None:
            return None, None
        # Assume `xy_pos` is lon/lat value
        x, y = Proj(info[INFO.PROJ])(*xy_pos)
        col = (x - info[INFO.ORIGIN_X]) / info[INFO.CELL_WIDTH]
        row = (y - info[INFO.ORIGIN_Y]) / info[INFO.CELL_HEIGHT]
        return np.round(row), np.round(col)

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
        _, data = content_within_shape(data, trans, LinearRing(points))
        return data

    def highest_resolution_uuid(self, *uuids):
        return min([self.get_info(uuid) for uuid in uuids], key=lambda i: i[INFO.CELL_WIDTH])[INFO.UUID]

    def lowest_resolution_uuid(self, *uuids):
        return max([self.get_info(uuid) for uuid in uuids], key=lambda i: i[INFO.CELL_WIDTH])[INFO.UUID]

    def get_coordinate_mask_polygon(self, dsi_or_uuid, points):
        data = self.get_content(dsi_or_uuid)
        trans = self._create_layer_affine(dsi_or_uuid)
        index_mask, data = content_within_shape(data, trans, LinearRing(points))
        coords_mask = (index_mask[0] * trans.e + trans.f, index_mask[1] * trans.a + trans.c)
        return coords_mask, data

    def get_content_coordinate_mask(self, uuid, coords_mask):
        data = self.get_content(uuid)
        trans = self._create_layer_affine(uuid)
        index_mask = (
            ((coords_mask[0] - trans.f) / trans.e).astype(np.uint),
            ((coords_mask[1] - trans.c) / trans.a).astype(np.uint),
        )
        return data[index_mask]

    def __getitem__(self, datasetinfo_or_uuid):
        """
        return science content proxy capable of generating a numpy array when sliced
        :param datasetinfo_or_uuid: metadata or key for the dataset
        :return: sliceable object returning numpy arrays
        """
        pass

    def asProbeDataSource(self, **kwargs):
        """
        Produce delegate used to match masks to data content.
        :param kwargs:
        :return: delegate object used by probe objects to access workspace content
        """
        return self  # FUTURE: revise this once we have more of an interface specification

    def asLayerDataSource(self, uuid=None, **kwargs):
        """
        produce layer data source delegate to be handed to a LayerRep
        :param kwargs:
        :return:
        """
        return self  # FUTURE: revise this once we have more of an interface specification


def main():
    parser = argparse.ArgumentParser(
        description="PURPOSE",
        epilog="",
        fromfile_prefix_chars='@')
    parser.add_argument('-v', '--verbose', dest='verbosity', action="count", default=0,
                        help='each occurrence increases verbosity 1 level through ERROR-WARNING-INFO-DEBUG')
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
