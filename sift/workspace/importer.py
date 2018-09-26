#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PURPOSE

REFERENCES

REQUIRES

:author: R.K.Garcia <rkgarcia@wisc.edu>
:copyright: 2017 by University of Wisconsin Regents, see AUTHORS for more details
:license: GPLv3, see LICENSE for more details
"""
import os, sys
import logging, unittest
import re
from abc import ABC, abstractmethod, abstractclassmethod
from collections import namedtuple
from datetime import datetime, timedelta
from typing import Sequence, Iterable, Generator, Mapping, Tuple
import gdal
import osr
import asyncio
import numpy as np
from pyproj import Proj
from sqlalchemy.orm import Session

from sift.common import PLATFORM, INFO, INSTRUMENT, KIND
from sift.workspace.goesr_pug import PugFile
from sift.workspace.guidebook import ABI_AHI_Guidebook, Guidebook
from .metadatabase import Resource, Product, Content

LOG = logging.getLogger(__name__)

try:
    from satpy import Scene
    from satpy.dataset import DatasetID
except ImportError:
    LOG.warning("SatPy is not installed and will not be used for importing.")
    Scene = None
    DatasetID = None

try:
    from skimage.measure import find_contours
except ImportError:
    find_contours = None

DEFAULT_GTIFF_OBS_DURATION = timedelta(seconds=60)
DEFAULT_GUIDEBOOK = ABI_AHI_Guidebook

GUIDEBOOKS = {
    PLATFORM.GOES_16: ABI_AHI_Guidebook,
    PLATFORM.GOES_17: ABI_AHI_Guidebook,
    PLATFORM.HIMAWARI_8: ABI_AHI_Guidebook,
    PLATFORM.HIMAWARI_9: ABI_AHI_Guidebook,
}

import_progress = namedtuple('import_progress', ['uuid', 'stages', 'current_stage', 'completion', 'stage_desc', 'dataset_info', 'data'])
# stages:int, number of stages this import requires
# current_stage:int, 0..stages-1 , which stage we're on
# completion:float, 0..1 how far we are along on this stage
# stage_desc:tuple(str), brief description of each of the stages we'll be doing


def get_guidebook_class(layer_info) -> Guidebook:
    platform = layer_info.get(INFO.PLATFORM)
    return GUIDEBOOKS.get(platform, DEFAULT_GUIDEBOOK)()


def get_contour_increments(layer_info):
    standard_name = layer_info[INFO.STANDARD_NAME]
    units = layer_info[INFO.UNITS]
    increments = {
        'air_temperature': [5., 2.5, 1., 0.5, 0.1],
        'relative_humidity': [15., 10., 5., 2., 1.],
        'eastward_wind': [15., 10., 5., 2., 1.],
        'northward_wind': [15., 10., 5., 2., 1.],
        'geopotential_height': [100., 50., 25., 10., 5.],
    }

    unit_increments = {
        'K': [5., 2.5, 1., 0.5, 0.1],
        '%': [15., 10., 5., 2., 1.],
    }

    contour_increments = increments.get(standard_name)
    if contour_increments is None:
        contour_increments = unit_increments.get(units)

        # def _in_group(wg, grp):
        #     return round(wg / grp) * grp >= grp
        # contour_increments = [5., 2.5, 1., 0.5, 0.1]
        # vmin, vmax = layer_info[INFO.VALID_RANGE]
        # width_guess = vmax - vmin
        # if _in_group(width_guess, 10000.):
        #     contour_increments = [x * 100. for x in contour_increments]
        # elif _in_group(width_guess, 1000.):
        #     contour_increments = [x * 50. for x in contour_increments]
        # elif _in_group(width_guess, 100.):
        #     contour_increments = [x * 10. for x in contour_increments]
        # elif _in_group(width_guess, 10.):
        #     contour_increments = [x * 1. for x in contour_increments]
    if contour_increments is None:
        LOG.warning("Unknown contour data type ({}, {}), guessing at contour "
                    "levels...".format(standard_name, units))
        return [5000., 1000., 500., 200., 100.]

    LOG.debug("Contour increments for ({}, {}): {}".format(
        standard_name, units, contour_increments))
    return contour_increments


def get_contour_levels(vmin, vmax, increments):
    levels = []
    for idx, inc in enumerate(increments):
        vmin_round = np.ceil(vmin / inc) * inc
        vmax_round = np.ceil(vmax / inc) * inc
        inc_levels = np.arange(vmin_round, vmax_round, inc)
        # round to the highest increment or modulo operations will be wrong
        inc_levels = np.round(inc_levels / increments[-1]) * increments[-1]
        if idx > 0:
            mask = np.logical_or.reduce([
                np.isclose(inc_levels % i, 0) for i in increments[:idx]])
            inc_levels = inc_levels[~mask]
        levels.append(inc_levels)

    return levels


def generate_guidebook_metadata(layer_info) -> Mapping:
    guidebook = get_guidebook_class(layer_info)
    # also get info for this layer from the guidebook
    gbinfo = guidebook.collect_info(layer_info)
    layer_info.update(gbinfo)  # FUTURE: should guidebook be integrated into DocBasicLayer?

    # add as visible to the front of the current set, and invisible to the rest of the available sets
    layer_info[INFO.COLORMAP] = guidebook.default_colormap(layer_info)
    layer_info[INFO.CLIM] = guidebook.climits(layer_info)
    layer_info[INFO.VALID_RANGE] = guidebook.valid_range(layer_info)
    if INFO.DISPLAY_TIME not in layer_info:
        layer_info[INFO.DISPLAY_TIME] = guidebook._default_display_time(layer_info)
    if INFO.DISPLAY_NAME not in layer_info:
        layer_info[INFO.DISPLAY_NAME] = guidebook._default_display_name(layer_info)

    if 'level' in layer_info:
        # calculate contour_levels and zoom levels
        increments = get_contour_increments(layer_info)
        vmin, vmax = layer_info[INFO.VALID_RANGE]
        contour_levels = get_contour_levels(vmin, vmax, increments)
        layer_info['contour_levels'] = contour_levels

    return layer_info


class aImporter(ABC):
    """
    Abstract Importer class creates or amends Resource, Product, Content entries in the metadatabase used by Workspace
    aImporter instances are backgrounded by the Workspace to bring Content into the workspace
    """
    _S: Session = None   # dedicated sqlalchemy database session to use during this import instance; revert if necessary, commit as appropriate
    _cwd: str = None  # where content flat files should be imported to within the workspace, omit this from content path

    def __init__(self, workspace_cwd, database_session, **kwargs):
        super(aImporter, self).__init__()
        self._S = database_session
        self._cwd = workspace_cwd

    @classmethod
    def from_product(cls, prod: Product, workspace_cwd, database_session, **kwargs):
        # FIXME: deal with products that need more than one resource
        try:
            cls = prod.resource[0].format
        except IndexError:
            LOG.error('no resources in {} {}'.format(repr(type(prod)), repr(prod)))
            raise
        return cls(prod.resource[0].path, workspace_cwd=workspace_cwd, database_session=database_session, **kwargs)

    @classmethod
    @abstractmethod
    def is_relevant(cls, source_path=None, source_uri=None) -> bool:
        """
        return True if this importer is capable of reading this URI.
        """
        return False

    @abstractmethod
    def merge_resources(self) -> Iterable[Resource]:
        """
        Returns:
            sequence of Resources found at the source, typically one resource per file
        """
        return []

    @abstractmethod
    def merge_products(self) -> Iterable[Product]:
        """
        products available in the resource, adding any metadata entries for Products within the resource
        this may be run by the metadata collection agent, or by the workspace!
        Returns:
            sequence of Products that could be turned into Content in the workspace
        """
        return []

    @abstractmethod
    def begin_import_products(self, *product_ids) -> Generator[import_progress, None, None]:
        """
        background import of content from a series of products
        if none are provided, all products resulting from merge_products should be imported
        Args:
            *products: sequence of products to import

        Returns:
            generator which yields status tuples as the content is imported
        """
        # FUTURE: this should be async def coroutine
        return


class aSingleFileWithSingleProductImporter(aImporter):
    """
    simplification of importer that handles a single-file with a single product
    """
    source_path: str = None
    _resource: Resource = None

    def __init__(self, source_path, workspace_cwd, database_session, **kwargs):
        super(aSingleFileWithSingleProductImporter, self).__init__(workspace_cwd, database_session)
        self.source_path = source_path

    @property
    def num_products(self):
        return 1

    def merge_resources(self) -> Iterable[Resource]:
        """
        Returns:
            sequence of Resources found at the source, typically one resource per file
        """
        now = datetime.utcnow()
        if self._resource is not None:
            res = self._resource
        else:
            self._resource = res = self._S.query(Resource).filter(Resource.path == self.source_path).first()
        if res is None:
            LOG.debug('creating new Resource entry for {}'.format(self.source_path))
            self._resource = res = Resource(
                format=type(self),
                path=self.source_path,
                mtime=now,
                atime=now,
            )
            self._S.add(res)
        return [self._resource]

    @abstractmethod
    def product_metadata(self):
        """
        Returns:
            info dictionary for the single product available
        """
        return {}

    def merge_products(self) -> Iterable[Product]:
        """
        products available in the resource, adding any metadata entries for Products within the resource
        this may be run by the metadata collection agent, or by the workspace!
        Returns:
            sequence of Products that could be turned into Content in the workspace
        """
        now = datetime.utcnow()
        if self._resource is not None:
            res = self._resource
        else:
            self._resource = res = self._S.query(Resource).filter(Resource.path == self.source_path).first()
        if res is None:
            LOG.debug('no resources for {}'.format(self.source_path))
            return []

        if len(res.product):
            zult = list(res.product)
            # LOG.debug('pre-existing products {}'.format(repr(zult)))
            return zult

        # else probe the file and add product metadata, without importing content
        from uuid import uuid1
        uuid = uuid1()
        meta = self.product_metadata()
        meta[INFO.UUID] = uuid

        prod = Product(
            uuid_str = str(uuid),
            atime = now,
        )
        prod.resource.append(res)
        assert(INFO.OBS_TIME in meta)
        assert(INFO.OBS_DURATION in meta)
        prod.update(meta)  # sets fields like obs_duration and obs_time transparently
        assert(prod.info[INFO.OBS_TIME] is not None and prod.obs_time is not None)
        assert(prod.info[INFO.VALID_RANGE] is not None)
        LOG.debug('new product: {}'.format(repr(prod)))
        self._S.add(prod)
        self._S.commit()
        return [prod]


class GeoTiffImporter(aSingleFileWithSingleProductImporter):
    """
    GeoTIFF data importer
    """
    @classmethod
    def is_relevant(self, source_path=None, source_uri=None):
        source = source_path or source_uri
        return True if (source.lower().endswith('.tif') or source.lower().endswith('.tiff')) else False

    @staticmethod
    def _metadata_for_path(pathname):
        meta = {}
        if not pathname:
            return meta

        # Old but still necesary, get some information from the filename instead of the content
        m = re.match(r'HS_H(\d\d)_(\d{8})_(\d{4})_B(\d\d)_([A-Za-z0-9]+).*', os.path.split(pathname)[1])
        if m is not None:
            plat, yyyymmdd, hhmm, bb, scene = m.groups()
            when = datetime.strptime(yyyymmdd + hhmm, '%Y%m%d%H%M')
            plat = PLATFORM('Himawari-{}'.format(int(plat)))
            band = int(bb)
            #
            # # workaround to make old files work with new information
            # from sift.model.guidebook import AHI_HSF_Guidebook
            # if band in AHI_HSF_Guidebook.REFL_BANDS:
            #     standard_name = "toa_bidirectional_reflectance"
            # else:
            #     standard_name = "toa_brightness_temperature"

            meta.update({
                INFO.PLATFORM: plat,
                INFO.BAND: band,
                INFO.INSTRUMENT: INSTRUMENT.AHI,
                INFO.SCHED_TIME: when,
                INFO.OBS_TIME: when,
                INFO.OBS_DURATION: DEFAULT_GTIFF_OBS_DURATION,
                INFO.SCENE: scene,
            })
        return meta

    @staticmethod
    def _check_geotiff_metadata(gtiff):
        gtiff_meta = gtiff.GetMetadata()
        # Sanitize metadata from the file to use SIFT's Enums
        if "name" in gtiff_meta:
            gtiff_meta[INFO.DATASET_NAME] = gtiff_meta.pop("name")
        if "platform" in gtiff_meta:
            plat = gtiff_meta.pop("platform")
            try:
                gtiff_meta[INFO.PLATFORM] = PLATFORM(plat)
            except ValueError:
                gtiff_meta[INFO.PLATFORM] = PLATFORM.UNKNOWN
                LOG.warning("Unknown platform being loaded: {}".format(plat))
        if "instrument" in gtiff_meta or "sensor" in gtiff_meta:
            inst = gtiff_meta.pop("sensor", gtiff_meta.pop("instrument", None))
            try:
                gtiff_meta[INFO.INSTRUMENT] = INSTRUMENT(inst)
            except ValueError:
                gtiff_meta[INFO.INSTRUMENT] = INSTRUMENT.UNKNOWN
                LOG.warning("Unknown instrument being loaded: {}".format(inst))
        if "start_time" in gtiff_meta:
            start_time = datetime.strptime(gtiff_meta["start_time"], "%Y-%m-%dT%H:%M:%SZ")
            gtiff_meta[INFO.SCHED_TIME] = start_time
            gtiff_meta[INFO.OBS_TIME] = start_time
            if "end_time" in gtiff_meta:
                end_time = datetime.strptime(gtiff_meta["end_time"], "%Y-%m-%dT%H:%M:%SZ")
                gtiff_meta[INFO.OBS_DURATION] = end_time - start_time
        if "valid_min" in gtiff_meta:
            gtiff_meta["valid_min"] = float(gtiff_meta["valid_min"])
        if "valid_max" in gtiff_meta:
            gtiff_meta["valid_max"] = float(gtiff_meta["valid_max"])
        if "standard_name" in gtiff_meta:
            gtiff_meta[INFO.STANDARD_NAME] = gtiff_meta["standard_name"]
        if "flag_values" in gtiff_meta:
            gtiff_meta["flag_values"] = tuple(int(x) for x in gtiff_meta["flag_values"].split(','))
        if "flag_masks" in gtiff_meta:
            gtiff_meta["flag_masks"] = tuple(int(x) for x in gtiff_meta["flag_masks"].split(','))
        if "flag_meanings" in gtiff_meta:
            gtiff_meta["flag_meanings"] = gtiff_meta["flag_meanings"].split(' ')
        if "units" in gtiff_meta:
            gtiff_meta[INFO.UNITS] = gtiff_meta.pop('units')
        return gtiff_meta

    @staticmethod
    def get_metadata(source_path=None, source_uri=None, **kwargs):
        if source_uri is not None:
            raise NotImplementedError("GeoTiffImporter cannot read from URIs yet")
        d = GeoTiffImporter._metadata_for_path(source_path)
        gtiff = gdal.Open(source_path)

        ox, cw, _, oy, _, ch = gtiff.GetGeoTransform()
        d[INFO.KIND] = KIND.IMAGE

        # FUTURE: this is Content metadata and not Product metadata:
        d[INFO.ORIGIN_X] = ox
        d[INFO.ORIGIN_Y] = oy
        d[INFO.CELL_WIDTH] = cw
        d[INFO.CELL_HEIGHT] = ch
        # FUTURE: Should the Workspace normalize all input data or should the Image Layer handle any projection?
        srs = osr.SpatialReference()
        srs.ImportFromWkt(gtiff.GetProjection())
        d[INFO.PROJ] = srs.ExportToProj4().strip()  # remove extra whitespace

        # Workaround for previously supported files
        # give them some kind of name that means something
        if INFO.BAND in d:
            d[INFO.DATASET_NAME] = "B{:02d}".format(d[INFO.BAND])
        else:
            # for new files, use this as a basic default
            # FUTURE: Use Dataset name instead when we can read multi-dataset files
            d[INFO.DATASET_NAME] = os.path.split(source_path)[-1]

        d[INFO.PATHNAME] = source_path
        band = gtiff.GetRasterBand(1)
        d[INFO.SHAPE] = rows, cols = (band.YSize, band.XSize)

        # Fix PROJ4 string if it needs an "+over" parameter
        p = Proj(d[INFO.PROJ])
        lon_l, lat_u = p(ox, oy, inverse=True)
        lon_r, lat_b = p(ox + cw * cols, oy + ch * rows, inverse=True)
        if "+over" not in d[INFO.PROJ] and lon_r < lon_l:
            LOG.debug("Add '+over' to geotiff PROJ.4 because it seems to cross the anti-meridian")
            d[INFO.PROJ] += " +over"

        bandtype = gdal.GetDataTypeName(band.DataType)
        if bandtype.lower() != 'float32':
            LOG.warning('attempting to read geotiff files with non-float32 content')

        gtiff_meta = GeoTiffImporter._check_geotiff_metadata(gtiff)
        d.update(gtiff_meta)
        generate_guidebook_metadata(d)
        LOG.debug("GeoTIFF metadata for {}: {}".format(source_path, repr(d)))
        return d

    def product_metadata(self):
        return GeoTiffImporter.get_metadata(self.source_path)

    # @asyncio.coroutine
    def begin_import_products(self, *product_ids):  # FUTURE: allow product_ids to be uuids
        source_path = self.source_path
        if product_ids:
            products = [self._S.query(Product).filter_by(id=anid).one() for anid in product_ids]
        else:
            products = list(self._S.query(Resource, Product).filter(
                Resource.path==source_path).filter(
                Product.resource_id==Resource.id).all())
            assert(products)
        if len(products)>1:
            LOG.warning('only first product currently handled in geotiff loader')
        prod = products[0]

        if prod.content:
            LOG.info('content is already available, skipping import')
            return

        now = datetime.utcnow()

        # re-collect the metadata, which should be separated between Product vs Content metadata in the FUTURE
        # principally we're not allowed to store ORIGIN_ or CELL_ metadata in the Product
        info = GeoTiffImporter.get_metadata(source_path)

        # Additional metadata that we've learned by loading the data
        gtiff = gdal.Open(source_path)

        band = gtiff.GetRasterBand(1)  # FUTURE may be an assumption
        shape = rows, cols = band.YSize, band.XSize
        blockw, blockh = band.GetBlockSize()  # non-blocked files will report [band.XSize,1]

        data_filename = '{}.image'.format(prod.uuid)
        data_path = os.path.join(self._cwd, data_filename)

        coverage_filename = '{}.coverage'.format(prod.uuid)
        coverage_path = os.path.join(self._cwd, coverage_filename)
        # no sparsity map

        # shovel that data into the memmap incrementally
        # http://geoinformaticstutorial.blogspot.com/2012/09/reading-raster-data-with-python-and-gdal.html
        img_data = np.memmap(data_path, dtype=np.float32, shape=shape, mode='w+')

        # load at an increment that matches the file's tile size if possible
        IDEAL_INCREMENT = 512.0
        increment = min(blockh * int(np.ceil(IDEAL_INCREMENT/blockh)), 2048)

        # how many coverage states are we traversing during the load? for now let's go simple and have it be just image rows
        # coverage_rows = int((rows + increment - 1) / increment) if we had an even increment but it's not guaranteed
        cov_data = np.memmap(coverage_path, dtype=np.int8, shape=(rows,), mode='w+')
        cov_data[:] = 0  # should not be needed except maybe in Windows?

        # LOG.debug("keys in geotiff product: {}".format(repr(list(prod.info.keys()))))
        LOG.debug("cell size in geotiff product: {} x {}".format(prod.info[INFO.CELL_HEIGHT], prod.info[INFO.CELL_WIDTH]))

        # create and commit a Content entry pointing to where the content is in the workspace, even if coverage is empty
        c = Content(
            lod = 0,
            resolution = int(min(abs(info[INFO.CELL_WIDTH]), abs(info[INFO.CELL_HEIGHT]))),
            atime = now,
            mtime = now,

            # info about the data array memmap
            path = data_filename,
            rows = rows,
            cols = cols,
            levels = 0,
            dtype = 'float32',

            cell_width = info[INFO.CELL_WIDTH],
            cell_height = info[INFO.CELL_HEIGHT],
            origin_x = info[INFO.ORIGIN_X],
            origin_y = info[INFO.ORIGIN_Y],
            proj4 = info[INFO.PROJ],

            # info about the coverage array memmap, which in our case just tells what rows are ready
            coverage_rows = rows,
            coverage_cols = 1,
            coverage_path = coverage_filename
        )
        # c.info.update(prod.info) would just make everything leak together so let's not do it
        self._S.add(c)
        prod.content.append(c)
        self._S.commit()

        # FIXME: yield initial status to announce content is available, even if it's empty

        # now do the actual array filling from the geotiff file
        # FUTURE: consider explicit block loads using band.ReadBlock(x,y) once
        irow = 0
        while irow < rows:
            nrows = min(increment, rows-irow)
            row_data = band.ReadAsArray(0, irow, cols, nrows)
            img_data[irow:irow+nrows,:] = np.require(row_data, dtype=np.float32)
            cov_data[irow:irow+nrows] = 1
            irow += increment
            status = import_progress(uuid=prod.uuid,
                                       stages=1,
                                       current_stage=0,
                                       completion=float(irow)/float(rows),
                                       stage_desc="importing geotiff",
                                       dataset_info=None,
                                       data=img_data)
            yield status

        # img_data = gtiff.GetRasterBand(1).ReadAsArray()
        # img_data = np.require(img_data, dtype=np.float32, requirements=['C'])  # FIXME: is this necessary/correct?
        # normally we would place a numpy.memmap in the workspace with the content of the geotiff raster band/s here

        # single stage import with all the data for this simple case
        zult = import_progress(uuid=prod.uuid,
                               stages=1,
                               current_stage=0,
                               completion=1.0,
                               stage_desc="done loading geotiff",
                               dataset_info=None,
                               data=img_data)

        yield zult

        # Finally, update content mtime and atime
        c.atime = c.mtime = datetime.utcnow()
        # self._S.commit()



# map .platform_id in PUG format files to SIFT platform enum
PLATFORM_ID_TO_PLATFORM = {
    'G16': PLATFORM.GOES_16,
    'G17': PLATFORM.GOES_17,
    # hsd2nc export of AHI data as PUG format
    'Himawari-8': PLATFORM.HIMAWARI_8,
    'Himawari-9': PLATFORM.HIMAWARI_9,
    # axi2cmi export as PUG, more consistent with other uses
    'H8': PLATFORM.HIMAWARI_8,
    'H9': PLATFORM.HIMAWARI_9
}


class GoesRPUGImporter(aSingleFileWithSingleProductImporter):
    """
    Import from PUG format GOES-16 netCDF4 files
    """

    @staticmethod
    def _basic_pug_metadata(pug):
        return {
            INFO.PLATFORM: PLATFORM_ID_TO_PLATFORM[pug.platform_id],  # e.g. G16, H8
            INFO.BAND: pug.band,
            INFO.DATASET_NAME: 'B{:02d}'.format(pug.band),
            INFO.INSTRUMENT: INSTRUMENT.AHI if 'Himawari' in pug.instrument_type else INSTRUMENT.ABI,
            INFO.SCHED_TIME: pug.sched_time,
            INFO.OBS_TIME: pug.time_span[0],
            INFO.OBS_DURATION: pug.time_span[1] - pug.time_span[0],
            INFO.DISPLAY_TIME: pug.display_time,
            INFO.SCENE: pug.scene_id,
            INFO.DISPLAY_NAME: pug.display_name,
        }

    @classmethod
    def is_relevant(cls, source_path=None, source_uri=None):
        source = source_path or source_uri
        return True if (source.lower().endswith('.nc') or source.lower().endswith('.nc4')) else False

    @staticmethod
    def pug_factory(source_path):
        dn,fn = os.path.split(source_path)
        is_netcdf = (fn.lower().endswith('.nc') or fn.lower().endswith('.nc4'))
        if not is_netcdf:
            raise ValueError("PUG loader requires files ending in .nc or .nc4: {}".format(repr(source_path)))
        return PugFile.attach(source_path)
        # if 'L1b' in fn:
        #     LOG.debug('attaching {} as PUG L1b'.format(source_path))
        #     return PugL1bTools(source_path)
        # else:
        #     LOG.debug('attaching {} as PUG CMI'.format(source_path))
        #     return PugCmiTools(source_path)

    @staticmethod
    def get_metadata(source_path=None, source_uri=None, pug=None, **kwargs):
        # yield successive levels of detail as we load
        if source_uri is not None:
            raise NotImplementedError("GoesRPUGImporter cannot read from URIs yet")

        #
        # step 1: get any additional metadata and an overview tile
        #

        d = {}
        # nc = nc4.Dataset(source_path)
        pug = pug or GoesRPUGImporter.pug_factory(source_path)

        d.update(GoesRPUGImporter._basic_pug_metadata(pug))
        # d[INFO.DATASET_NAME] = os.path.split(source_path)[-1]
        d[INFO.PATHNAME] = source_path
        d[INFO.KIND] = KIND.IMAGE

        # FUTURE: this is Content metadata and not Product metadata:
        d[INFO.PROJ] = pug.proj4_string
        # get nadir-meter-ish projection coordinate vectors to be used by proj4
        y,x = pug.proj_y, pug.proj_x
        d[INFO.ORIGIN_X] = x[0]
        d[INFO.ORIGIN_Y] = y[0]

        # midyi, midxi = int(y.shape[0] / 2), int(x.shape[0] / 2)
        # PUG states radiance at index [0,0] extends between coordinates [0,0] to [1,1] on a quadrille
        # centers of pixels are therefore at +0.5, +0.5
        # for a (e.g.) H x W image this means [H/2,W/2] coordinates are image center
        # for now assume all scenes are even-dimensioned (e.g. 5424x5424)
        # given that coordinates are evenly spaced in angular -> nadir-meters space,
        # technically this should work with any two neighbor values
        # d[INFO.CELL_WIDTH] = x[midxi+1] - x[midxi]
        # d[INFO.CELL_HEIGHT] = y[midyi+1] - y[midyi]
        cell_size = pug.cell_size
        d[INFO.CELL_HEIGHT], d[INFO.CELL_WIDTH] = cell_size

        shape = pug.shape
        d[INFO.SHAPE] = shape
        generate_guidebook_metadata(d)

        d[INFO.FAMILY] = '{}:{}:{}:{}µm'.format(KIND.IMAGE.name, 'geo', d[INFO.STANDARD_NAME], d[INFO.CENTRAL_WAVELENGTH]) # kind:pointofreference:measurement:wavelength
        d[INFO.CATEGORY] = 'NOAA-PUG:{}:{}:{}'.format(d[INFO.PLATFORM].name, d[INFO.INSTRUMENT].name, d[INFO.SCENE])  # system:platform:instrument:target
        d[INFO.SERIAL] = pug.time_span[0].strftime("%Y%m%dT%H%M%S")
        LOG.debug(repr(d))
        return d

    # def __init__(self, source_path, workspace_cwd, database_session, **kwargs):
    #     super(GoesRPUGImporter, self).__init__(workspace_cwd, database_session)
    #     self.source_path = source_path

    def product_metadata(self):
        return GoesRPUGImporter.get_metadata(self.source_path)

    # @asyncio.coroutine
    def begin_import_products(self, *product_ids):
        source_path = self.source_path
        if product_ids:
            products = [self._S.query(Product).filter_by(id=anid).one() for anid in product_ids]
            assert(products)
        else:
            products = list(self._S.query(Resource, Product).filter(
                Resource.path == source_path).filter(
                Product.resource_id == Resource.id).all())
            assert (products)
        if len(products) > 1:
            LOG.warning('only first product currently handled in pug loader')
        prod = products[0]

        if prod.content:
            LOG.warning('content was already available, skipping import')
            return

        pug = GoesRPUGImporter.pug_factory(source_path)
        rows, cols = shape = pug.shape
        cell_height, cell_width = pug.cell_size
        origin_y, origin_x = pug.origin
        proj4 = pug.proj4_string

        now = datetime.utcnow()

        data_filename = '{}.image'.format(prod.uuid)
        data_path = os.path.join(self._cwd, data_filename)

        # coverage_filename = '{}.coverage'.format(prod.uuid)
        # coverage_path = os.path.join(self._cwd, coverage_filename)
        # no sparsity map

        # shovel that data into the memmap incrementally
        # http://geoinformaticstutorial.blogspot.com/2012/09/reading-raster-data-with-python-and-gdal.html
        img_data = np.memmap(data_path, dtype=np.float32, shape=shape, mode='w+')

        LOG.info('converting radiance to %s' % pug.bt_or_refl)
        image = pug.bt if 'bt'==pug.bt_or_refl else pug.refl
        # bt_or_refl, image, units = pug.convert_from_nc()  # FIXME expensive
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

        # FUTURE as we're doing so, also update coverage array (showing what sections of data are loaded)
        # FUTURE and for some cases the sparsity array, if the data is interleaved (N/A for NetCDF imagery)

        bandtype = np.float32
        img_data[:] = np.ma.fix_invalid(image, copy=False, fill_value=np.NAN)  # FIXME: expensive

        # create and commit a Content entry pointing to where the content is in the workspace, even if coverage is empty
        c = Content(
            lod = 0,
            resolution = int(min(abs(cell_width), abs(cell_height))),
            atime = now,
            mtime = now,

            # info about the data array memmap
            path = data_filename,
            rows = rows,
            cols = cols,
            proj4 = proj4,
            # levels = 0,
            dtype = 'float32',

            # info about the coverage array memmap, which in our case just tells what rows are ready
            # coverage_rows = rows,
            # coverage_cols = 1,
            # coverage_path = coverage_filename

            cell_width = cell_width,
            cell_height = cell_height,
            origin_x = origin_x,
            origin_y = origin_y,
        )
        # c.info.update(prod.info) would just make everything leak together so let's not do it
        self._S.add(c)
        prod.content.append(c)
        # prod.touch()
        self._S.commit()

        yield import_progress(uuid=prod.uuid,
                              stages=1,
                              current_stage=0,
                              completion=1.0,
                              stage_desc="GOES PUG data add to workspace",
                              dataset_info=None,
                              data=img_data)


class SatPyImporter(aImporter):
    """Generic SatPy importer"""

    def __init__(self, source_path, workspace_cwd, database_session, **kwargs):
        super(SatPyImporter, self).__init__(workspace_cwd, database_session)
        reader = kwargs.pop('reader', None)
        if reader is None:
            raise NotImplementedError("Can't automatically determine reader.")
        if not isinstance(source_path, (list, tuple)):
            source_path = [source_path]

        if len(source_path) > 1:
            raise NotImplementedError("SatPy importer does not currently "
                                      "support reading more than one file "
                                      "at a time.")
        if Scene is None:
            raise ImportError("SatPy is not available and can't be used as "
                              "an importer")
        self.filenames = list(source_path)
        self.reader = reader
        if 'scene' in kwargs:
            self.scn = kwargs['scene']
        else:
            self.scn = Scene(reader=self.reader,
                               filenames=self.filenames)
        self._resources = []
        # DatasetID filters
        self.product_filters = {}
        for k in ['resolution', 'calibration', 'level']:
            if k in kwargs:
                self.product_filters[k] = kwargs.pop(k)
        # NOTE: product_filters don't do anything if the dataset_ids aren't
        #       specified since we are using all available dataset ids
        self.dataset_ids = sorted(kwargs.get('dataset_ids', self.scn.available_dataset_ids()))

    @classmethod
    def from_product(cls, prod: Product, workspace_cwd, database_session, **kwargs):
        # this overrides the base class because we assume that the product
        # has kwargs that we want
        # NOTE: The kwargs are currently provided by the caller in
        # workspace.py so this isn't needed right now
        # FIXME: deal with products that need more than one resource
        try:
            cls = prod.resource[0].format
        except IndexError:
            LOG.error('no resources in {} {}'.format(repr(type(prod)), repr(prod)))
            raise
        kwargs.pop('reader', None)
        kwargs.pop('scenes', None)
        kwargs.pop('scene', None)
        kwargs['dataset_ids'] = [DatasetID.from_dict(prod.info)]
        return cls(prod.resource[0].path, workspace_cwd=workspace_cwd, database_session=database_session, **kwargs)

    @classmethod
    def is_relevant(cls, source_path=None, source_uri=None):
        # this importer should only be used if specifically requested
        return False

    def merge_resources(self):
        if len(self._resources) == len(self.filenames):
            return self._resources

        resources = self._S.query(Resource).filter(
            Resource.path.in_(self.filenames)).all()
        if len(resources) == len(self.filenames):
            self._resources = resources
            return self._resources

        now = datetime.utcnow()
        res_dict = {r.path: r for r in self._resources}
        for fn in self.filenames:
            if fn in res_dict:
                continue

            res = Resource(
                format=type(self),
                path=fn,
                mtime=now,
                atime=now,
            )
            self._S.add(res)
            res_dict[fn] = res

        self._resources = res_dict.values()
        return self._resources

    def merge_products(self):
        resources = self.merge_resources()
        if resources is None:
            LOG.debug('no resources for {}'.format(self.filenames))
            return

        # FIXME: This doesn't work for multiple files to one product
        resources = list(resources)
        if len(resources) > 1:
            raise NotImplementedError("Products created from more than one "
                                      "file are not currently supported.")
        res = resources[0]
        products = list(res.product)
        existing_ids = {DatasetID.from_dict(prod.info): prod for prod in products}
        existing_prods = {x: existing_ids[x] for x in self.dataset_ids if x in existing_ids}
        if products and len(existing_prods) == len(self.dataset_ids):
            products = existing_prods.values()
            LOG.debug('pre-existing products {}'.format(repr(products)))
            yield from products
            return

        from uuid import uuid1
        scn = self.load_all_datasets()
        
        for ds_id, ds in scn.datasets.items():
            # don't recreate a Product for one we already have
            if ds_id in existing_ids:
                yield existing_ids[ds_id]
                continue

            existing_ids.get(ds_id, None)
            meta = ds.attrs
            uuid = uuid1()
            meta[INFO.UUID] = uuid
            now = datetime.utcnow()
            prod = Product(
                uuid_str=str(uuid),
                atime=now,
            )
            prod.resource.append(res)
            
            assert(INFO.OBS_TIME in meta)
            assert(INFO.OBS_DURATION in meta)
            prod.update(meta)  # sets fields like obs_duration and obs_time transparently
            assert(prod.info[INFO.OBS_TIME] is not None and prod.obs_time is not None)
            assert(prod.info[INFO.VALID_RANGE] is not None)
            LOG.debug('new product: {}'.format(repr(prod)))
            self._S.add(prod)
            self._S.commit()
            yield prod

    @property
    def num_products(self) -> int:
        # WARNING: This could provide radiances and higher level products
        #          which SIFT probably shouldn't care about
        return len(self.dataset_ids)

    def load_all_datasets(self) -> Scene:
        self.scn.load(self.dataset_ids, **self.product_filters)
        # copy satpy metadata keys to SIFT keys
        for ds in self.scn:
            start_time = ds.attrs['start_time']
            end_time=ds.attrs['end_time']
            ds.attrs[INFO.OBS_TIME] = start_time[0]
            ds.attrs[INFO.SCHED_TIME] = start_time[0]
            duration = end_time[0]-start_time[0]
            
            if duration.total_seconds() == 0:
                duration = timedelta(minutes=60)
            ds.attrs[INFO.OBS_DURATION] = duration

            # Handle GRIB platform/instrument
            ds.attrs[INFO.KIND] = KIND.IMAGE if self.reader != 'grib' else \
                KIND.CONTOUR
            ds.attrs[INFO.INSTRUMENT] = ds.attrs.get('sensor')
            ds.attrs[INFO.PLATFORM] = ds.attrs.get('platform_name')
            if 'centreDescription' in ds.attrs and \
                    ds.attrs[INFO.INSTRUMENT] == 'unknown':
                description = ds.attrs['centreDescription']
                if ds.attrs.get(INFO.PLATFORM) is None:
                    ds.attrs[INFO.PLATFORM] = 'NWP'
                if 'NCEP' in description:
                    ds.attrs[INFO.INSTRUMENT] = 'GFS'
            if ds.attrs[INFO.INSTRUMENT] in ['GFS', 'unknown']:
                ds.attrs[INFO.INSTRUMENT] = INSTRUMENT.GFS
            if ds.attrs[INFO.PLATFORM] in ['NWP', 'unknown']:
                ds.attrs[INFO.PLATFORM] = PLATFORM.NWP
            ds.attrs.setdefault(INFO.STANDARD_NAME, ds.attrs.get('standard_name'))
            if 'wavelength' in ds.attrs:
                ds.attrs.setdefault(INFO.CENTRAL_WAVELENGTH,
                                    ds.attrs['wavelength'])

            # Resolve anything else needed by SIFT
            id_str = ":".join(str(v) for v in DatasetID.from_dict(ds.attrs))
            ds.attrs[INFO.DATASET_NAME] = id_str
            model_time = ds.attrs.get('model_time')
            if model_time is not None:
                ds.attrs[INFO.DATASET_NAME] += " " + model_time.isoformat()
            ds.attrs[INFO.BAND] = 0
            ds.attrs[INFO.SHORT_NAME] = ds.attrs['name']
            if ds.attrs.get('level') is not None:
                ds.attrs[INFO.SHORT_NAME] = "{} @ {}hPa".format(
                    ds.attrs['name'], ds.attrs['level'])
            ds.attrs[INFO.SHAPE] = ds.shape
            generate_guidebook_metadata(ds.attrs)

            # Generate FAMILY and CATEGORY
            if 'model_time' in ds.attrs:
                model_time = ds.attrs['model_time'].isoformat()
                cat_id = model_time + ':' + id_str
            else:
                model_time = None
                cat_id = id_str
            ds.attrs[INFO.SCENE] = hash(ds.attrs['area'])
            ds.attrs[INFO.FAMILY] = '{}:{}:{}'.format(
                ds.attrs[INFO.KIND].name, ds.attrs[INFO.STANDARD_NAME],
                ds.attrs[INFO.SHORT_NAME])
            ds.attrs[INFO.CATEGORY] = 'SatPy:{}:{}:{}:{}'.format(
                ds.attrs[INFO.PLATFORM], ds.attrs[INFO.INSTRUMENT],
                ds.attrs[INFO.SCENE], cat_id)  # system:platform:instrument:target
            # TODO: Include level or something else in addition to time?
            #       Probably need to include the model run time (prefix id_str?)

            start_str = ds.attrs['start_time'][0].isoformat()
            ds.attrs[INFO.SERIAL] = start_str if model_time is None else model_time + ":" + start_str

        return self.scn

    def _area_to_sift_attrs(self, area):
        """Area to sift keys"""
        from pyresample.geometry import AreaDefinition
        if not isinstance(area, AreaDefinition):
            raise NotImplementedError("Only AreaDefinition datasets can "
                                      "be loaded at this time.")

        half_pixel_x = abs(area.pixel_size_x) / 2.
        half_pixel_y = abs(area.pixel_size_y) / 2.

        return {
            INFO.PROJ: area.proj4_string,
            INFO.ORIGIN_X: area.area_extent[0] + half_pixel_x,
            INFO.ORIGIN_Y: area.area_extent[3] - half_pixel_y,
            INFO.CELL_HEIGHT: -abs(area.pixel_size_y),
            INFO.CELL_WIDTH: area.pixel_size_x,
        }

    def begin_import_products(self, *product_ids) -> Generator[import_progress, None, None]:
        import dask.array as da

        if product_ids:
            products = [self._S.query(Product).filter_by(id=anid).one() for anid in product_ids]
            assert products
        else:
            products = list(self._S.query(Resource, Product).filter(
                Resource.path.in_(self.filenames)).filter(
                Product.resource_id == Resource.id).all())
            assert products

        # FIXME: Don't recreate the importer every time we want to load data
        dataset_ids = [DatasetID.from_dict(prod.info) for prod in products]
        self.scn.load(dataset_ids)
        num_stages = len(products)
        for idx, (prod, ds_id) in enumerate(zip(products, dataset_ids)):
            dataset = self.scn[ds_id]
            shape = dataset.shape
            num_contents = 1 if prod.info[INFO.KIND] == KIND.IMAGE else 2

            if prod.content:
                LOG.warning('content was already available, skipping import')
                continue

            now = datetime.utcnow()
            area_info = self._area_to_sift_attrs(dataset.attrs['area'])
            cell_width = area_info[INFO.CELL_WIDTH]
            cell_height = area_info[INFO.CELL_HEIGHT]
            proj4 = area_info[INFO.PROJ]
            origin_x = area_info[INFO.ORIGIN_X]
            origin_y = area_info[INFO.ORIGIN_Y]
            data = dataset.data

            # Handle building contours for data from 0 to 360 longitude
            antimeridian = 179.999
            if '+proj=latlong' not in proj4:
                # the x coordinate for the antimeridian in this projection
                antimeridian = Proj(proj4)(antimeridian, 0)[0]
            am_index = int(np.ceil((antimeridian - origin_x) / cell_width))
            if prod.info[INFO.KIND] == KIND.CONTOUR and 0 < am_index < shape[1]:
                # if we have data from 0 to 360 longitude, we want -180 to 360
                data = da.concatenate((data[:, am_index:], data), axis=1)
                shape = data.shape
                origin_x -= am_index * cell_width

            # shovel that data into the memmap incrementally
            data_filename = '{}.image'.format(prod.uuid)
            data_path = os.path.join(self._cwd, data_filename)
            img_data = np.memmap(data_path, dtype=np.float32, shape=shape, mode='w+')
            da.store(data, img_data)

            c = Content(
                lod=0,
                resolution=int(min(abs(cell_width), abs(cell_height))),
                atime=now,
                mtime=now,

                # info about the data array memmap
                path=data_filename,
                rows=shape[0],
                cols=shape[1],
                proj4=proj4,
                # levels = 0,
                dtype='float32',

                # info about the coverage array memmap, which in our case just tells what rows are ready
                # coverage_rows = rows,
                # coverage_cols = 1,
                # coverage_path = coverage_filename

                cell_width=cell_width,
                cell_height=cell_height,
                origin_x=origin_x,
                origin_y=origin_y,
            )
            c.info[INFO.KIND] = KIND.IMAGE
            # c.info.update(prod.info) would just make everything leak together so let's not do it
            self._S.add(c)
            prod.content.append(c)
            # prod.touch()
            self._S.commit()

            yield import_progress(uuid=prod.uuid,
                                  stages=num_stages,
                                  current_stage=idx,
                                  completion=1. / num_contents,
                                  stage_desc="SatPy image data add to workspace",
                                  dataset_info=None,
                                  data=img_data)

            if num_contents == 1:
                continue

            if find_contours is None:
                raise RuntimeError("Can't create contours without 'skimage' "
                                   "package installed.")

            # XXX: Should/could 'lod' be used for different contour level data?
            levels = [x for y in prod.info['contour_levels'] for x in y]
            data_filename = '{}.contour'.format(prod.uuid)
            data_path = os.path.join(self._cwd, data_filename)
            try:
                contour_data = self._compute_contours(
                        img_data, prod.info[INFO.VALID_RANGE][0],
                        prod.info[INFO.VALID_RANGE][1], levels)
            except ValueError:
                LOG.warning("Could not compute contour levels for '{}'".format(prod.uuid))
                LOG.debug("Contour error: ", exc_info=True)
                continue

            contour_data[:, 0] *= cell_width
            contour_data[:, 0] += origin_x
            contour_data[:, 1] *= cell_height
            contour_data[:, 1] += origin_y
            contour_data.tofile(data_path)
            c = Content(
                lod=0,
                resolution=int(min(abs(cell_width), abs(cell_height))),
                atime=now,
                mtime=now,

                # info about the data array memmap
                path=data_filename,
                rows=contour_data.shape[0],  # number of vertices
                cols=contour_data.shape[1],  # col (x), row (y), "connect", num_points_for_level
                proj4=proj4,
                # levels = 0,
                dtype='float32',

                cell_width=cell_width,
                cell_height=cell_height,
                origin_x=origin_x,
                origin_y=origin_y,
            )
            c.info[INFO.KIND] = KIND.CONTOUR
            # c.info["level_index"] = level_indexes
            self._S.add(c)
            prod.content.append(c)
            self._S.commit()

            completion = 2. / num_contents
            yield import_progress(uuid=prod.uuid,
                                  stages=num_stages,
                                  current_stage=idx,
                                  completion=completion,
                                  stage_desc="SatPy contour data add to workspace",
                                  dataset_info=None,
                                  data=img_data)

    def _get_verts_and_connect(self, paths):
        """ retrieve vertices and connects from given paths-list
        """
        # THIS METHOD WAS COPIED FROM VISPY
        verts = np.vstack(paths)
        gaps = np.add.accumulate(np.array([len(x) for x in paths])) - 1
        connect = np.ones(gaps[-1], dtype=bool)
        connect[gaps[:-1]] = False
        return verts, connect

    def _compute_contours(self, img_data: np.ndarray,
                          vmin: float, vmax: float,
                          levels: list) -> np.ndarray:
        all_levels = []
        level_indexes = []
        for level in levels:
            if level < vmin or level > vmax:
                continue

            contours = find_contours(img_data, level,
                                     positive_orientation='high')
            if not contours:
                LOG.debug("No contours found for level: {}".format(level))
                continue
            v, c = self._get_verts_and_connect(contours)
            # swap row, column to column, row (x, y)
            v[:, [0, 1]] = v[:, [1, 0]]
            v += np.array([0.5, 0.5])
            v[:, 0] = np.where(np.isnan(v[:, 0]), 0, v[:, 0])

            # HACK: Store float vertices, boolean, and index arrays together in one float array
            this_level = np.empty((v.shape[0],), np.float32)
            this_level[:] = np.nan
            this_level[-1] = v.shape[0]
            level_indexes.append(v.shape[0])
            c = np.concatenate((c, [False])).astype(np.float32)
            # level_data = np.concatenate((v, c[:, None]), axis=1)
            level_data = np.concatenate((v, c[:, None], this_level[:, None]), axis=1)
            all_levels.append(level_data)

        if not all_levels:
            raise ValueError("No valid contour levels")
        return np.concatenate(all_levels).astype(np.float32)


PATH_TEST_DATA = os.environ.get('TEST_DATA', os.path.expanduser("~/Data/test_files/thing.dat"))

class tests(unittest.TestCase):
    def setUp(self):
        pass

    def test_something(self):
        pass


def _debug(type, value, tb):
    "enable with sys.excepthook = debug"
    if not sys.stdin.isatty():
        sys.__excepthook__(type, value, tb)
    else:
        import traceback, pdb
        traceback.print_exception(type, value, tb)
        # …then start the debugger in post-mortem mode.
        pdb.post_mortem(tb)  # more “modern”


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="PURPOSE",
        epilog="",
        fromfile_prefix_chars='@')
    parser.add_argument('-v', '--verbose', dest='verbosity', action="count", default=0,
                        help='each occurrence increases verbosity 1 level through ERROR-WARNING-INFO-DEBUG')
    parser.add_argument('-d', '--debug', dest='debug', action='store_true',
                        help="enable interactive PDB debugger on exception")
    parser.add_argument('inputs', nargs='*',
                        help="input files to process")
    args = parser.parse_args()

    levels = [logging.ERROR, logging.WARN, logging.INFO, logging.DEBUG]
    logging.basicConfig(level=levels[min(3, args.verbosity)])

    if args.debug:
        sys.excepthook = _debug

    if not args.inputs:
        unittest.main()
        return 0

    for pn in args.inputs:
        pass

    return 0


if __name__ == '__main__':
    sys.exit(main())
