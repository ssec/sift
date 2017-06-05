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

NOTES
    FUTURE import sequence:
        trigger: user requests skim (metadata only) or import (metadata plus bring into document) of a file or directory system
         for each file selected
        phase 1: regex for file patterns identifies which importers are worth trying
        phase 2: background: importers open files, form metadatabase insert transaction, first importer to succeed wins (priority order).
        stop after this if just skimming
        phase 3: background: load of overview (lod=0), adding flat files to workspace and Content entry to metadatabase
        phase 3a: document and scenegraph show overview up on screen
        phase 4: background: load of one or more levels of detail, with max LOD currently being considered native
        phase 4a: document updates to show most useful LOD+stride content


:author: R.K.Garcia <rayg@ssec.wisc.edu>
:copyright: 2014 by University of Wisconsin Regents, see AUTHORS for more details
:license: GPLv3, see LICENSE for more details
"""
import argparse
import logging
import os
import sys
import unittest
from uuid import UUID, uuid1 as uuidgen

import numba as nb
import numpy as np
from PyQt4.QtCore import QObject, pyqtSignal
from pyproj import Proj
from rasterio import Affine
from shapely.geometry.polygon import LinearRing

from sift.common import INFO
from sift.model.shapes import content_within_shape
from sift.workspace.importer import GeoTiffImporter, GoesRPUGImporter
from .metadatabase import Metadatabase, Content, Product, Resource

LOG = logging.getLogger(__name__)

DEFAULT_WORKSPACE_SIZE = 256
MIN_WORKSPACE_SIZE = 8



# first instance is main singleton instance; don't preclude the possibility of importing from another workspace later on
TheWorkspace = None



# @ctx.contextmanager
# def cwd(newcwd):
#     "temporarily change directories"
#     oldcwd = os.getcwd()
#     os.chdir(newcwd)
#     yield
#     os.chdir(oldcwd)

@nb.jit(nopython=True, nogil=True)
def mask_from_coverage_sparsity_2d(mask: np.ndarray, coverage: np.ndarray, sparsity: np.ndarray):
    """
    update a numpy.ma style mask from coverage and sparsity arrays
    Args:
        mask (np.array, dtype=bool): mutable array to be updated as a numpy.masked_array mask
        coverage (np.array) : coverage array to stretch across the mask
        sparsity (np.array) : sparsity array to repeat across the mask
    """
    h,w = mask.shape
    cov_rpt_y, cov_rpt_x = h // coverage.shape[0], w // coverage.shape[1]
    spr_h, spr_w = sparsity.shape
    for y in range(h):
        for x in range(w):
            mask[y, x] |= bool(0 == coverage[y // cov_rpt_y, x // cov_rpt_x] * sparsity[y % spr_h, x % spr_w])


class ActiveContent(QObject):
    """
    ActiveContent merges numpy.memmap arrays with their corresponding Content metadata
    Purpose: make ActiveContent a drop-in replacement for numpy arrays
    Workspace instantiates ActiveContent from metadatabase Content entries
    """
    _C = None  # my metadata
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
        self._C = C
        self._wsd = workspace_cwd
        if workspace_cwd is None and C is None:
            LOG.warning('test initialization of ActiveContent')
            self._test_init()
        else:
            self._attach()

    def _test_init(self):
        data = np.ones((4, 12), dtype=np.float32)
        data = np.cumsum(data, axis=0)
        data = np.cumsum(data, axis=1)
        self._data = data
        self._sparsity = sp = np.zeros((2, 2), dtype=np.int8)
        sp[1,1] = 1  # only 1/4 of dataset loaded
        self._coverage = co = np.zeros((4, 1), dtype=np.int8)
        co[2:4] = 1  # and of that, only the bottom half of the image

    @staticmethod
    def _rcls(r:int, c:int, l:int):
        """
        :param r: rows or None
        :param c: columns or None
        :param l: levels or None
        :return: condensed tuple(string with 'rcl', 'rc', 'rl', dimension tuple corresponding to string)
        """
        rcl_shape = tuple(
            (name, dimension) for (name, dimension) in zip('rcl', (r, c, l)) if dimension)
        rcl = tuple(x[0] for x in rcl_shape)
        shape = tuple(x[1] for x in rcl_shape)
        return rcl, shape

    def _update_mask(self):
        """
        merge sparsity and coverage mask to a standard maskedarray mask
        :return:
        """
        # FIXME: beware the race conditions with this
        # FIXME: it would be better to lazy-eval the mask, assuming coverage and sparsity << data
        if self._mask is None:
            self._mask = mask = np.zeros_like(self._data, dtype=bool)
        else:
            mask = self._mask
            mask[:] = False
        mask_from_coverage_sparsity_2d(mask, self._coverage, self._sparsity)

    def _attach(self, mode='c'):
        """
        attach content arrays, for holding by workspace in _available
        :param c: Content entity from database
        :return: workspace_data_arrays instance
        """
        c = self._C
        self._rcl, self._shape = rcl, shape = self._rcl(c.rows, c.cols, c.levels)
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
        self._coverage = mm(c.coverage_path, dtype=np.int8, mode=mode, shape=cshape) if c.coverage_path else np.array([1])
        _, sshape = self._rcls(c.coverage_cols, c.coverage_cols, c.coverage_levels)
        self._sparsity = mm(c.sparsity_path, dtype=np.int8, mode=mode, shape=sshape) if c.sparsity_path else np.array([1])

        self._update_mask()


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
    _available = None  # dictionary of {Content.id : workspace_data_arrays}
    _inventory = None  # metadatabase instance, sqlalchemy
    _inventory_path = None  # filename to store and load inventory information (simple cache)
    _S = None  # MDB session
    _tempdir = None  # TemporaryDirectory, if it's needed (i.e. a directory name was not given)
    _max_size_gb = None  # maximum size in gigabytes of flat files we cache in the workspace
    _queue = None

    # signals
    didStartImport = pyqtSignal(dict)  # a dataset started importing; generated after overview level of detail is available
    didMakeImportProgress = pyqtSignal(dict)
    didUpdateDataset = pyqtSignal(dict)  # partial completion of a dataset import, new datasetinfo dict is released
    didFinishImport = pyqtSignal(dict)  # all loading activities for a dataset have completed
    didDiscoverExternalDataset = pyqtSignal(dict)  # a new dataset was added to the workspace from an external agent

    IMPORT_CLASSES = [GeoTiffImporter, GoesRPUGImporter]

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
        self._inventory_path = self._ws_path('_inventory.db')
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
        should_init = os.path.exists(self._inventory_path)
        self._inventory = md = Metadatabase('sqlite://' + self._inventory_path)
        if should_init:
            md.create_tables()
        self._S = md.session()

    def _init_inventory_existing_datasets(self):
        """
        Do an inventory of an pre-existing workspace
        FIXME: go through and check that everything in the workspace makes sense
        FIXME: check workspace subdirectories for helper sockets and mmaps
        :return:
        """
        # attach the database, creating it if needed
        return self._init_create_workspace()

    def _store_inventory(self):
        """
        write inventory dictionary to an inventory.pkl file in the cwd
        :return:
        """
        self._S.commit()

    #
    #  data array handling
    #

    def _ws_path(self, rel_path):
        return self._ws_path(rel_path)

    def _remove_content_files_from_workspace(self, c: Content ):
        total = 0
        for filename in [c.path, c.coverage_path, c.sparsity_path]:
            pn = self._ws_path(filename)
            if os.path.exists(pn):
                os.remove(pn)
                total += os.stat(pn).st_size
        return total

    def _cached_arrays_for_content(self, c:Content):
        """
        attach cached data indicated in Content, unless it's been attached already and is in _available
        :param c: metadatabase Content object
        :return: workspace_content_arrays
        """
        c.touch()
        cache_entry = self._available.get(c.id)
        if cache_entry is None:
            new_entry = self._attach_content(c)
            self._available[c.id] = cache_entry = new_entry
        return cache_entry


    #
    # often-used queries
    #

    def _product_with_uuid(self, uuid) -> Product:
        return self._S.query(Product).filter_by(uuid=uuid).first()

    def _content_ordered_by_lod(self, p:Product):
        """
        return content entries ordered by ascending LOD, i.e. lowest (overview) to highest (native)
        :param p: Product
        :return: tuple of Content entries
        """
        cs = tuple(self._S.query(Content).filter_by(product_id=p.id).order_by(Content.lod).all())
        return cs

    def _product_overview_content(self, p:Product) -> Content:
        return self._S.query(Content).filter_by(product_id=p.id).order_by(Content.lod).first()

    def _product_native_content(self, p:Product) -> Content:
        return self._S.query(Content).filter_by(product_id=p.id).order_by(Content.lod.desc()).first()

    #
    # combining queries with data content
    #

    def _overview_content_for_uuid(self, uuid):
        # FUTURE: do a compound query for this to get the Content entry
        prod = self._product_with_uuid(uuid)
        ovc = self._product_overview_content(prod)
        arrays = self._cached_arrays_for_content(ovc)
        return arrays.data

    def _native_content_for_uuid(self, uuid):
        # FUTURE: do a compound query for this to get the Content entry
        prod = self._product_with_uuid(uuid)
        ovc = self._product_native_content(prod)
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
        for root, dirs, files in os.walk(self.cwd):
            sz = sum(os.path.getsize(os.path.join(root, name)) for name in files)
            LOG.debug('%d bytes in %s' % (sz, root))

        return total

    # @staticmethod
    # def _key_for_path(path):
    #     if not os.path.exists(path):
    #         return None
    #     s = os.stat(path)
    #     return (os.path.realpath(path), s.st_mtime, s.st_size)

    # @lru_cache
    def _product_std_info(self, prod:Product):
        nat = self._product_native_content(prod)

        std_info_dict = {
            INFO.NAME: prod.short_name,
            INFO.PATH: prod.uri,
            INFO.UUID: prod.uuid,
            INFO.PROJ: nat.proj4,
            INFO.CELL_WIDTH: nat.cell_width,
            INFO.CELL_HEIGHT: nat.cell_height,
            INFO.ORIGIN_X: nat.origin_x,
            INFO.ORIGIN_Y: nat.origin_y,
        }
        return std_info_dict

    def get_info(self, dsi_or_uuid, lod=None):
        """
        :param dsi_or_uuid: existing datasetinfo dictionary, or its UUID
        :param lod: desired level of detail to focus
        :return:
        """
        from collections import ChainMap

        if isinstance(dsi_or_uuid, str):
            dsi_or_uuid = UUID(dsi_or_uuid)
        # look up the product for that uuid
        prod = self._product_with_uuid(dsi_or_uuid)
        prod.touch()
        self._S.commit()  # flush any pending updates to workspace db file
        if prod is None:
            return None

        return ChainMap(prod, self._product_std_info(prod))  # any updates go to prod k-v table


#----------------------------------------------------------------------


    def _check_cache(self, path):
        """
        FIXME: does not work if more than one product inside a path
        :param path: file we're checking
        :return: uuid, info, overview_content if the data is already available without import
        """
        hits = self._S.query(Resource).filter_by(path=path).all()
        if not hits:
            return None
        if len(hits)>=1:
            if len(hits) > 1:
                LOG.warning('more than one Resource found suitable, there can be only one')
            resource = hits[0]
            hits = self._S.query(Content).filter(
                Content.product_id == Product.id).filter(
                Product.resource_id == resource.id).order_by(
                Content.lod).all()
            if len(hits)>=1:
                content = hits[0]  # presumably this is closest to LOD_OVERVIEW
                # if len(hits)>1:
                #     LOG.warning('more than one Content found suitable, there can be only one')
                cac = self._cached_arrays_for_content(content)
                return content.product.uuid, content.product.info, cac.data

        # key = self._key_for_path(path)
        # nfo = self._inventory.get(key, None)
        # if nfo is None:
        #     return None
        # uuid, info, data_info = nfo
        # dfilename, dtype, shape = data_info
        # if dfilename is None:
        #     LOG.debug("No data loaded from {}".format(path))
        #     data = None
        # else:
        #     dpath = self._ws_path(dfilename)
        #     if not os.path.exists(dpath):
        #         del self._inventory[key]
        #         self._store_inventory()
        #         return None
        #     data = np.memmap(dpath, dtype=dtype, mode='c', shape=shape)
        # return uuid, info, data

    @property
    def paths_in_cache(self):
        # find non-overview non-auxiliary data files
        # FIXME: also need to include coverage and sparsity paths
        return [x.path for x in self._S.query(Content).filter(Content.lod>0).all()]

    @property
    def uuids_in_cache(self):
        prods = self._S.query(Product).all()
        return [p.uuid for p in prods]

    # def _update_cache(self, path, uuid, info, data):
    #     """
    #     add or update the cache and put it to disk
    #     :param path: path to get key from
    #     :param uuid: uuid the data's been assigned
    #     :param info: dataset info dictionary
    #     :param data: numpy.memmap backed by a file in the workspace
    #     :return:
    #     """
    #     key = self._key_for_path(path)
    #     prev_nfo = self._inventory.get(key, (uuid, None, (None, None, None)))
    #
    #     if info is None:
    #         info = prev_nfo[1]
    #
    #     if data is None:
    #         data_info = prev_nfo[2]
    #     else:
    #         data_info = (os.path.split(data.filename)[1], data.dtype, data.shape)
    #
    #     nfo = (uuid, info, data_info)
    #     self._inventory[key] = nfo
    #     self._store_inventory()


    # def _inventory_check(self):
    #     "return revised_inventory_dict, [(access-time, size, path, key), ...], total_size"
    #     cache = []
    #     inv = dict(self._inventory)
    #     total_size = 0
    #     for key,nfo in self._inventory.items():
    #         uuid,info,data_info = nfo
    #         filename, dtype, shape = data_info
    #         path = self._ws_path(filename)
    #         if os.path.exists(path):
    #             st = os.stat(path)
    #             cache.append((st.st_atime, st.st_size, path, key))
    #             total_size += st.st_size
    #         else:
    #             LOG.info('removing stale {}'.format(key))
    #             del inv[key]
    #     # sort by atime
    #     cache.sort()
    #     return inv, cache, total_size

    def recently_used_resource_paths(self, n=32):
        return [p.resource.uri for p in self._S.query(Product).order_by(Product.atime.desc()).limit(n).all()]
        # FIXME "replace this completely with product list"
        # inv, cache, total_size = self._inventory_check()
        # self._inventory = inv
        # self._store_inventory()
        # # get from most recently used end of list
        # return [q[3][0] for q in cache[-n:]]

    def _eject_resource_from_workspace(self, resource: Resource, defer_commit=False):
        """
        remove all resource contents from the database
        if the resource original path no longer exists, also purge resource and products from database
        :param resource: resource object we
        :return: number of bytes freed from the workspace
        """
        total = 0
        for prod in resource.products:
            for con in prod.contents:
                total += self._remove_content_files_from_workspace(con)
                self._S.delete(con)

        if not resource.exists():  # then purge the resource and its products as well
           self._S.delete(resource)
        if not defer_commit:
            self._S.commit()
        return total

    def remove_all_workspace_content_for_resource_paths(self, paths):
        total = 0
        for path in paths:
            rsr_hits = self._S.query(Resource).filter_by(path=path).all()
            for rsr in rsr_hits:
                total += self._eject_resource_from_workspace(rsr, defer_commit=True)
        self._S.commit()
        return total

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
        for res in self._S.query(Resource).order_by(Resource.atime).all():
            if total_size < max_size:
                break
            total_size -= self._eject_resource_from_workspace(res)
            # remove all content for lowest atimes until

    def close(self):
        self._S.commit()
        self._clean_cache()

    def idle(self):
        """
        Called periodically when application is idle. Does a clean-up tasks and returns True if more needs to be done later.
        Time constrained to ~0.1s.
        :return: True/False, whether or not more clean-up needs to be scheduled.
        """
        return False

    def get_metadata(self, uuid_or_path):
        if isinstance(uuid_or_path, UUID):
            return self._info[uuid_or_path]
        else:
            return self._inventory[self._key_for_path(uuid_or_path)][1]

    def import_image(self, source_path=None, source_uri=None, allow_cache=True):
        """
        Start loading URI data into the workspace asynchronously.

        :param source_path:
        :return:
        """
        if source_uri is not None and source_path is None:
            raise NotImplementedError('URI load not yet supported')

        if allow_cache and source_path is not None:
            nfo = self._check_cache(source_path)
            if nfo is not None:
                uuid, info, data = nfo
                self._info[uuid] = info
                self._data[uuid] = data
                return info

        uuid = uuidgen()

        # find the best importer
        for imp in self._importers:
            if imp.is_relevant(source_path=source_path):
                # gen = imp(self, self.cwd, uuid, source_path=source_path, cache_path=self._preferred_cache_path(uuid))
                break
        else:
            raise IOError("unable to import {}".format(source_path))

        # Collect and cache metadata
        # collect metadata before iterating because we need metadata as soon as possible
        info = self._info[uuid] = imp.get_metadata(uuid, source_path=source_path)
        # we haven't loaded the data yet (will do it asynchronously later)
        self._data[uuid] = None
        if allow_cache:
            self._update_cache(source_path, uuid, info, None)
        return info

    def import_image_data(self, uuid, allow_cache=True):
        metadata = self.get_metadata(uuid)
        name = metadata[INFO.DATASET_NAME]
        source_path = metadata[INFO.PATHNAME]

        for imp in self._importers:
            if imp.is_relevant(source_path=source_path):
                break
        else:
            raise IOError("unable to import {}".format(source_path))

        # FIXME: for now, just iterate the incremental load. later we want to add this to TheQueue and update the UI as we get more data loaded
        gen = imp(uuid, source_path=source_path, cache_path=self._preferred_cache_path(uuid))
        for update in gen:
            if update.data is not None:
                data = self._data[uuid] = update.data
                LOG.info("{} {}: {:.01f}%".format(name, update.stage_desc, update.completion*100.0))
        # copy the data into an anonymous memmap
        # FIXME: register a new Content entry with the metadatabase
        self._data[uuid] = data = self._convert_to_memmap(str(uuid), data)
        if allow_cache:
            self._update_cache(source_path, uuid, None, data)
        # TODO: schedule cache cleaning in background after a series of imports completes
        return data

    def create_composite(self, symbols:dict, relation:dict):
        """
        create a layer composite in the workspace
        :param symbols: dictionary of logical-name to uuid
        :param relation: dictionary with information on how the relation is calculated (FUTURE)
        """
        raise NotImplementedError()

    def _preferred_cache_path(self, uuid):
        filename = str(uuid)
        return self._ws_path(filename)

    def _convert_to_memmap(self, uuid, data:np.ndarray):
        if isinstance(data, np.memmap):
            return data
        # from tempfile import TemporaryFile
        # fp = TemporaryFile()
        pathname = self._preferred_cache_path(uuid)
        fp = open(pathname, 'wb+')
        mm = np.memmap(fp, dtype=data.dtype, shape=data.shape, mode='w+')
        # FIXME: register with content table
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
        :param lod: desired level of detail to focus  (0 for overview)
        :return:
        """
        if isinstance(dsi_or_uuid, str):
            dsi_or_uuid = UUID(dsi_or_uuid)
        prod = self._product_with_uuid(dsi_or_uuid)
        prod.touch()
        content = self._S.query(Content).filter_by(Content.product_id==prod.id).order_by(Content.lod.desc()).first()
        content.touch()
        self._S.commit()  # flush any pending updates to workspace db file
        return self._cached_arrays_for_content(content)

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
        return np.int64(np.round(row)), np.int64(np.round(col))

    def layer_proj(self, dsi_or_uuid):
        """Project lon/lat probe points to image X/Y"""
        info = self.get_info(dsi_or_uuid)
        return Proj(info[INFO.PROJ])

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
        return min([self.get_info(uuid) for uuid in uuids], key=lambda i: i[INFO.CELL_WIDTH])[INFO.UUID]

    def lowest_resolution_uuid(self, *uuids):
        return max([self.get_info(uuid) for uuid in uuids], key=lambda i: i[INFO.CELL_WIDTH])[INFO.UUID]

    def get_coordinate_mask_polygon(self, dsi_or_uuid, points):
        data = self.get_content(dsi_or_uuid)
        trans = self._create_layer_affine(dsi_or_uuid)
        p = self.layer_proj(dsi_or_uuid)
        points = self._project_points(p, points)
        index_mask, data = content_within_shape(data, trans, LinearRing(points))
        coords_mask = (index_mask[0] * trans.e + trans.f, index_mask[1] * trans.a + trans.c)
        coords_mask = p(*coords_mask, inverse=True)
        return coords_mask, data

    def get_content_coordinate_mask(self, uuid, coords_mask):
        data = self.get_content(uuid)
        trans = self._create_layer_affine(uuid)
        p = self.layer_proj(uuid)
        coords_mask = p(*coords_mask)
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
