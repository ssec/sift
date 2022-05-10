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
import logging
import os
import re
from abc import ABC, abstractmethod
from collections import namedtuple
from datetime import datetime, timedelta
from typing import (Callable, Dict, Generator, Iterable, List, Mapping,
                    Optional, Set, Tuple, Union)

import dask.array as da
import numpy as np
import satpy.resample
import satpy.readers.yaml_reader
import yaml
from pyproj import Proj
from pyresample.geometry import AreaDefinition, StackedAreaDefinition
from satpy.dataset import DatasetDict
from sqlalchemy.orm import Session
from xarray import DataArray

from uwsift import config, USE_INVENTORY_DB
from uwsift.common import Platform, Info, Instrument, Kind, INSTRUMENT_MAP, PLATFORM_MAP
from uwsift.model.area_definitions_manager import AreaDefinitionsManager
from uwsift.satpy_compat import DataID, get_id_value, get_id_items, id_from_attrs
from uwsift.util import USER_CACHE_DIR
from uwsift.util.common import get_reader_kwargs_dict
from uwsift.workspace.guidebook import ABI_AHI_Guidebook, Guidebook
from .metadatabase import Resource, Product, Content, ContentImage, ContentUnstructuredPoints
from .utils import metadata_utils

from satpy import Scene, available_readers

_SATPY_READERS = None  # cache: see `available_satpy_readers()` below
SATPY_READER_CACHE_FILE = os.path.join(USER_CACHE_DIR,
                                       'available_satpy_readers.yaml')

LOG = logging.getLogger(__name__)

satpy_version = None
#try:
#    from satpy import __version__ as satpy_version
#except ImportError as e:
#    # Satpy's internal way of defining its version breaks when it is not
#    # installed. This leads to the exception when referencing satpy from just
#    # a git clone without installing it or within a PyInstaller package (bug
#    # report pending).
#    LOG.warning("BUG: Cannot determine satpy version. Cached information must be ignored."
#                "(Reason:" + e + ")")

try:
    from skimage.measure import find_contours
except ImportError:
    find_contours = None

DEFAULT_GTIFF_OBS_DURATION = timedelta(seconds=60)
DEFAULT_GUIDEBOOK = ABI_AHI_Guidebook

GUIDEBOOKS = {
    Platform.GOES_16: ABI_AHI_Guidebook,
    Platform.GOES_17: ABI_AHI_Guidebook,
    Platform.GOES_18: ABI_AHI_Guidebook,
    Platform.GOES_19: ABI_AHI_Guidebook,
    Platform.HIMAWARI_8: ABI_AHI_Guidebook,
    Platform.HIMAWARI_9: ABI_AHI_Guidebook,
}

import_progress = namedtuple('import_progress',
                             ['uuid', 'stages', 'current_stage', 'completion', 'stage_desc', 'dataset_info', 'data', 'content'])
"""
# stages:int, number of stages this import requires
# current_stage:int, 0..stages-1 , which stage we're on
# completion:float, 0..1 how far we are along on this stage
# stage_desc:tuple(str), brief description of each of the stages we'll be doing
"""


def _load_satpy_readers_cache(force_refresh=None):
    """Get Satpy reader information from a cache file or Satpy itself."""
    if force_refresh is None:
        force_refresh = os.getenv("UWSIFT_SATPY_CACHE_REFRESH", "False").lower()
        force_refresh = force_refresh in [True, "true"]

    try:
        if force_refresh:
            raise RuntimeError("Forcing refresh of available Satpy readers list")
        if not satpy_version:
            raise RuntimeError("Satpy version cannot be determined, regenerating available readers...")
        with open(SATPY_READER_CACHE_FILE, 'r') as cfile:
            LOG.info("Loading cached available Satpy readers from {}".format(SATPY_READER_CACHE_FILE))
            cache_contents = yaml.load(cfile, yaml.SafeLoader)
        if cache_contents is None:
            raise RuntimeError("Cached reader list is empty, regenerating...")
        if cache_contents['satpy_version'] != satpy_version:
            raise RuntimeError("Satpy has different version, regenerating available readers...")
    except (FileNotFoundError, RuntimeError, KeyError) as cause:
        LOG.info("Updating list of available Satpy readers...")
        cause.__suppress_context__ = True
        readers = available_readers(as_dict=True)
        # sort list of readers just in case we depend on this in the future
        readers = sorted(readers, key=lambda x: x['name'])
        readers = list(_sanitize_reader_info_for_yaml(readers))
        cache_contents = {
            'satpy_version': satpy_version,
            'readers': readers,
        }
        _save_satpy_readers_cache(cache_contents)
    return cache_contents['readers']


def _sanitize_reader_info_for_yaml(readers):
    # filter out known python objects to simplify YAML serialization
    for reader_info in readers:
        reader_info.pop('reader')
        reader_info.pop('data_identification_keys', None)
        reader_info['config_files'] = list(reader_info['config_files'])
        yield reader_info


def _save_satpy_readers_cache(cache_contents):
    """Write reader cache information to a file on disk."""
    cfile_dir = os.path.dirname(SATPY_READER_CACHE_FILE)
    os.makedirs(cfile_dir, exist_ok=True)
    with open(SATPY_READER_CACHE_FILE, 'w') as cfile:
        LOG.info("Caching available Satpy readers to {}".format(SATPY_READER_CACHE_FILE))
        yaml.dump(cache_contents, cfile)


def available_satpy_readers(as_dict=False, force_cache_refresh=None):
    """Get a list of reader names or reader information."""
    global _SATPY_READERS
    if _SATPY_READERS is None or force_cache_refresh:
        _SATPY_READERS = _load_satpy_readers_cache(force_refresh=force_cache_refresh)

    if not as_dict:
        return [r['name'] for r in _SATPY_READERS]
    return _SATPY_READERS


def filter_dataset_ids(ids_to_filter: Iterable[DataID]) -> Generator[DataID, None, None]:
    """Generate only non-filtered DataIDs based on EXCLUDE_DATASETS global filters."""
    # skip certain DataIDs
    for ds_id in ids_to_filter:
        for filter_key, filtered_values in config.get('data_reading.exclude_datasets').items():
            if get_id_value(ds_id, filter_key) in filtered_values:
                break
        else:
            yield ds_id


def get_guidebook_class(layer_info) -> Guidebook:
    platform = layer_info.get(Info.PLATFORM)
    return GUIDEBOOKS.get(platform, DEFAULT_GUIDEBOOK)()


def get_contour_increments(layer_info):
    standard_name = layer_info[Info.STANDARD_NAME]
    units = layer_info[Info.UNITS]
    increments = {
        'air_temperature': [10., 5., 2.5, 1., 0.5],
        'brightness_temperature': [10., 5., 2.5, 1., 0.5],
        'toa_brightness_temperature': [10., 5., 2.5, 1., 0.5],
        'toa_bidirectional_reflectance': [.15, .10, .05, .02, .01],
        'relative_humidity': [15., 10., 5., 2., 1.],
        'eastward_wind': [15., 10., 5., 2., 1.],
        'northward_wind': [15., 10., 5., 2., 1.],
        'geopotential_height': [100., 50., 25., 10., 5.],
    }

    unit_increments = {
        'kelvin': [10., 5., 2.5, 1., 0.5],
        '1': [.15, .10, .05, .02, .01],
        '%': [15., 10., 5., 2., 1.],
        'kg m**-2': [20., 10., 5., 2., 1.],
        'm s**-1': [15., 10., 5., 2., 1.],
        'm**2 s**-1': [5000., 1000., 500., 200., 100.],
        'gpm': [6000., 2000., 1000., 500., 200.],
        'kg kg**-1': [16e-8, 8e-8, 4e-8, 2e-8, 1e-8],
        'Pa s**-1': [6., 3., 2., 1., 0.5],
        's**-1': [0.02, 0.01, 0.005, 0.002, 0.001],
        'Pa': [5000., 1000., 500., 200., 100.],
        'J kg**-1': [5000., 1000., 500., 200., 100.],
        '(0 - 1)': [0.5, 0.2, 0.1, 0.05, 0.05],
    }

    contour_increments = increments.get(standard_name)
    if contour_increments is None:
        contour_increments = unit_increments.get(units)

        # def _in_group(wg, grp):
        #     return round(wg / grp) * grp >= grp
        # contour_increments = [5., 2.5, 1., 0.5, 0.1]
        # vmin, vmax = layer_info[Info.VALID_RANGE]
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
    mult = 1 / increments[-1]
    for idx, inc in enumerate(increments):
        vmin_round = np.ceil(vmin / inc) * inc
        vmax_round = np.ceil(vmax / inc) * inc
        inc_levels = np.arange(vmin_round, vmax_round, inc)
        # round to the highest increment or modulo operations will be wrong
        inc_levels = np.round(inc_levels / increments[-1]) * increments[-1]
        if idx > 0:
            # don't use coarse contours in the finer contour levels
            # we multiple by 1 / increments[-1] to try to resolve precision
            # errors which can be a big issue for very small increments.
            mask = np.logical_or.reduce([np.isclose((inc_levels * mult) % (i * mult), 0) for i in increments[:idx]])
            inc_levels = inc_levels[~mask]
        levels.append(inc_levels)

    return levels


def generate_guidebook_metadata(layer_info) -> Mapping:
    guidebook = get_guidebook_class(layer_info)
    # also get info for this layer from the guidebook
    gbinfo = guidebook.collect_info(layer_info)
    layer_info.update(gbinfo)  # FUTURE: should guidebook be integrated into DocBasicLayer?

    # add as visible to the front of the current set, and invisible to the rest of the available sets
    layer_info[Info.COLORMAP] = metadata_utils.get_default_colormap(layer_info, guidebook)
    if layer_info[Info.KIND] == Kind.POINTS:
        layer_info[Info.STYLE] = metadata_utils.get_default_point_style_name(layer_info)
    layer_info[Info.CLIM] = guidebook.climits(layer_info)
    layer_info[Info.VALID_RANGE] = guidebook.valid_range(layer_info)
    if Info.DISPLAY_TIME not in layer_info:
        layer_info[Info.DISPLAY_TIME] = guidebook._default_display_time(layer_info)
    if Info.DISPLAY_NAME not in layer_info:
        layer_info[Info.DISPLAY_NAME] = guidebook._default_display_name(layer_info)

    if 'level' in layer_info:
        # calculate contour_levels and zoom levels
        increments = get_contour_increments(layer_info)
        vmin, vmax = layer_info[Info.VALID_RANGE]
        contour_levels = get_contour_levels(vmin, vmax, increments)
        layer_info['contour_levels'] = contour_levels

    return layer_info


def _get_requires(scn: Scene) -> Set[str]:
    """
    Get the required file types for this scene. E.g. Seviri Prolog/Epilog.
    :param scn: scene object to analyse
    :return: set of required file types
    """
    requires = set()
    for reader in scn._readers.values():
        for file_handlers in reader.file_handlers.values():
            for file_handler in file_handlers:
                requires |= set(file_handler.filetype_info.get('requires', ()))
    return requires


def _required_files_set(scn: Scene, requires) -> Set[str]:
    """
    Get set of files additional required to load data. E.g. Seviri Prolog/Epilog files.
    :param scn: scene object to analyse
    :param requires: the file types which are required
    :return: set of required files
    """
    required_files = set()
    if requires:
        for reader in scn._readers.values():
            for file_handlers in reader.file_handlers.values():
                for file_handler in file_handlers:
                    if file_handler.filetype_info['file_type'] in requires:
                        required_files.add(file_handler.filename)
    return required_files


def _wanted_paths(scn: Scene, ds_name: str) -> List[str]:
    """
    Get list of paths in scene which are actually required to load given dataset name.
    :param scn: scene object to analyse
    :param ds_name: dataset name
    :return: paths in scene which are required to load given dataset
    """
    wanted_paths = []
    for reader in scn._readers.values():
        for file_handlers in reader.file_handlers.values():
            for file_handler in file_handlers:
                if not ds_name or not hasattr(file_handler, 'channel_name') \
                        or file_handler.channel_name == ds_name:
                    wanted_paths.append(file_handler.filename)
    return wanted_paths


class aImporter(ABC):
    """
    Abstract Importer class creates or amends Resource, Product, Content entries in the metadatabase used by Workspace
    aImporter instances are backgrounded by the Workspace to bring Content into the workspace
    """
    # dedicated sqlalchemy database session to use during this import instance;
    # revert if necessary, commit as appropriate
    _S: Session = None
    # where content flat files should be imported to within the workspace, omit this from content path
    _cwd: str = None

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
        paths = [r.path for r in prod.resource]
        # HACK for Satpy importer
        if 'reader' in prod.info:
            kwargs.setdefault('reader', prod.info['reader'])

        if 'scenes' in kwargs:
            scn = kwargs['scenes'].get(tuple(paths), None)
            if scn and '_satpy_id' in prod.info:
                # filter out files not required to load dataset for given product
                # extraneous files interfere with the merging process
                paths = _wanted_paths(scn, prod.info['_satpy_id'].get('name'))

        merge_target = kwargs.get('merge_target')
        if merge_target:
            # filter out all segments which are already loaded in the target product
            new_filenames = []
            existing_content = merge_target.content[0]
            for fn in paths:
                if fn not in existing_content.source_files:
                    new_filenames.append(fn)
            if new_filenames != paths:
                required_files = set()
                if scn:
                    requires = _get_requires(scn)
                    required_files = _required_files_set(scn, requires)
                paths = new_filenames
                if not paths:
                    return None
                if set(paths) == required_files:
                    return None
                paths.extend(list(required_files))  # TODO

        # In order to "load" converted datasets, we have to reuse the existing
        # scene from the first instantiation of SatpyImporter instead of loading
        # the data again, of course: we need to 'rescue across' the converted
        # data from the first SatpyImporter instantiation to the second one
        # (which will be created in the very last statement of this method).
        # The correct scene can be identified based on the `paths` list, which
        # contains the input files for a single scene. The keyword argument
        # `scenes` is mapping of a tuple of input files to a Scene object.
        # We reuse the Scene only if the `paths` list and the input files for
        # the Scene are the same. Since only products of the kinds POINTS, LINES
        # and VECTORS need conversion (at this time), the mechanism is only
        # applied for these. For an unknown reason reusing the scene breaks for
        # FCI data, this Importer data reading magic is too confused.
        if prod.info[Info.KIND] in (Kind.POINTS, Kind.LINES, Kind.VECTORS):
            for scene_files, scene in kwargs["scenes"].items():
                if list(scene_files) == paths:
                    del kwargs["scenes"]
                    assert "scene" not in kwargs
                    kwargs["scene"] = scene
                    break

        return cls(paths, workspace_cwd=workspace_cwd, database_session=database_session, **kwargs)

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
        if isinstance(source_path, list) and len(source_path) == 1:
            # backwards compatibility - we now expect a list
            source_path = source_path[0]
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
        meta[Info.UUID] = uuid

        prod = Product(
            uuid_str=str(uuid),
            atime=now,
        )
        prod.resource.append(res)
        assert (Info.OBS_TIME in meta)
        assert (Info.OBS_DURATION in meta)
        prod.update(meta)  # sets fields like obs_duration and obs_time transparently
        assert (prod.info[Info.OBS_TIME] is not None and prod.obs_time is not None)
        assert (prod.info[Info.VALID_RANGE] is not None)
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
            plat = Platform('Himawari-{}'.format(int(plat)))
            band = int(bb)
            #
            # # workaround to make old files work with new information
            # from uwsift.model.guidebook import AHI_HSF_Guidebook
            # if band in AHI_HSF_Guidebook.REFL_BANDS:
            #     standard_name = "toa_bidirectional_reflectance"
            # else:
            #     standard_name = "toa_brightness_temperature"

            meta.update({
                Info.PLATFORM: plat,
                Info.BAND: band,
                Info.INSTRUMENT: Instrument.AHI,
                Info.SCHED_TIME: when,
                Info.OBS_TIME: when,
                Info.OBS_DURATION: DEFAULT_GTIFF_OBS_DURATION,
                Info.SCENE: scene,
            })
        return meta

    @staticmethod
    def _check_geotiff_metadata(gtiff):
        gtiff_meta = gtiff.GetMetadata()
        # Sanitize metadata from the file to use SIFT's Enums
        if "name" in gtiff_meta:
            gtiff_meta[Info.DATASET_NAME] = gtiff_meta.pop("name")
        if "platform" in gtiff_meta:
            plat = gtiff_meta.pop("platform")
            try:
                gtiff_meta[Info.PLATFORM] = Platform(plat)
            except ValueError:
                gtiff_meta[Info.PLATFORM] = Platform.UNKNOWN
                LOG.warning("Unknown platform being loaded: {}".format(plat))
        if "instrument" in gtiff_meta or "sensor" in gtiff_meta:
            inst = gtiff_meta.pop("sensor", gtiff_meta.pop("instrument", None))
            try:
                gtiff_meta[Info.INSTRUMENT] = Instrument(inst)
            except ValueError:
                gtiff_meta[Info.INSTRUMENT] = Instrument.UNKNOWN
                LOG.warning("Unknown instrument being loaded: {}".format(inst))
        if "start_time" in gtiff_meta:
            start_time = datetime.strptime(gtiff_meta["start_time"], "%Y-%m-%dT%H:%M:%SZ")
            gtiff_meta[Info.SCHED_TIME] = start_time
            gtiff_meta[Info.OBS_TIME] = start_time
            if "end_time" in gtiff_meta:
                end_time = datetime.strptime(gtiff_meta["end_time"], "%Y-%m-%dT%H:%M:%SZ")
                gtiff_meta[Info.OBS_DURATION] = end_time - start_time
        if "valid_min" in gtiff_meta:
            gtiff_meta["valid_min"] = float(gtiff_meta["valid_min"])
        if "valid_max" in gtiff_meta:
            gtiff_meta["valid_max"] = float(gtiff_meta["valid_max"])
        if "standard_name" in gtiff_meta:
            gtiff_meta[Info.STANDARD_NAME] = gtiff_meta["standard_name"]
        if "flag_values" in gtiff_meta:
            gtiff_meta["flag_values"] = tuple(int(x) for x in gtiff_meta["flag_values"].split(','))
        if "flag_masks" in gtiff_meta:
            gtiff_meta["flag_masks"] = tuple(int(x) for x in gtiff_meta["flag_masks"].split(','))
        if "flag_meanings" in gtiff_meta:
            gtiff_meta["flag_meanings"] = gtiff_meta["flag_meanings"].split(' ')
        if "units" in gtiff_meta:
            gtiff_meta[Info.UNITS] = gtiff_meta.pop('units')
        return gtiff_meta

    @staticmethod
    def get_metadata(source_path=None, source_uri=None, **kwargs):
        import gdal
        import osr
        if source_uri is not None:
            raise NotImplementedError("GeoTiffImporter cannot read from URIs yet")
        d = GeoTiffImporter._metadata_for_path(source_path)
        gtiff = gdal.Open(source_path)

        ox, cw, _, oy, _, ch = gtiff.GetGeoTransform()
        d[Info.KIND] = Kind.IMAGE

        # FUTURE: this is Content metadata and not Product metadata:
        d[Info.ORIGIN_X] = ox
        d[Info.ORIGIN_Y] = oy
        d[Info.CELL_WIDTH] = cw
        d[Info.CELL_HEIGHT] = ch
        # FUTURE: Should the Workspace normalize all input data or should the Image Layer handle any projection?
        srs = osr.SpatialReference()
        srs.ImportFromWkt(gtiff.GetProjection())
        d[Info.PROJ] = srs.ExportToProj4().strip()  # remove extra whitespace

        # Workaround for previously supported files
        # give them some kind of name that means something
        if Info.BAND in d:
            d[Info.DATASET_NAME] = "B{:02d}".format(d[Info.BAND])
        else:
            # for new files, use this as a basic default
            # FUTURE: Use Dataset name instead when we can read multi-dataset files
            d[Info.DATASET_NAME] = os.path.split(source_path)[-1]

        band = gtiff.GetRasterBand(1)
        d[Info.SHAPE] = rows, cols = (band.YSize, band.XSize)

        # Fix PROJ4 string if it needs an "+over" parameter
        p = Proj(d[Info.PROJ])
        lon_l, lat_u = p(ox, oy, inverse=True)
        lon_r, lat_b = p(ox + cw * cols, oy + ch * rows, inverse=True)
        if "+over" not in d[Info.PROJ] and lon_r < lon_l:
            LOG.debug("Add '+over' to geotiff PROJ.4 because it seems to cross the anti-meridian")
            d[Info.PROJ] += " +over"

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
        import gdal
        source_path = self.source_path
        if product_ids:
            products = [self._S.query(Product).filter_by(id=anid).one() for anid in product_ids]
        else:
            products = list(self._S.query(Resource, Product).filter(
                Resource.path == source_path).filter(
                Product.resource_id == Resource.id).all())
            assert (products)
        if len(products) > 1:
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
        increment = min(blockh * int(np.ceil(IDEAL_INCREMENT / blockh)), 2048)

        # how many coverage states are we traversing during the load?
        # for now let's go simple and have it be just image rows
        # coverage_rows = int((rows + increment - 1) / increment) if we had an even increment but it's not guaranteed
        cov_data = np.memmap(coverage_path, dtype=np.int8, shape=(rows,), mode='w+')
        cov_data[:] = 0  # should not be needed except maybe in Windows?

        # LOG.debug("keys in geotiff product: {}".format(repr(list(prod.info.keys()))))
        LOG.debug(
            "cell size in geotiff product: {} x {}".format(prod.info[Info.CELL_HEIGHT], prod.info[Info.CELL_WIDTH]))

        # create and commit a Content entry pointing to where the content is in the workspace, even if coverage is empty
        c = ContentImage(
            lod=0,
            resolution=int(min(abs(info[Info.CELL_WIDTH]), abs(info[Info.CELL_HEIGHT]))),
            atime=now,
            mtime=now,

            # info about the data array memmap
            path=data_filename,
            rows=rows,
            cols=cols,
            levels=0,
            dtype='float32',

            cell_width=info[Info.CELL_WIDTH],
            cell_height=info[Info.CELL_HEIGHT],
            origin_x=info[Info.ORIGIN_X],
            origin_y=info[Info.ORIGIN_Y],
            proj4=info[Info.PROJ],

            # info about the coverage array memmap, which in our case just tells what rows are ready
            coverage_rows=rows,
            coverage_cols=1,
            coverage_path=coverage_filename
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
            nrows = min(increment, rows - irow)
            row_data = band.ReadAsArray(0, irow, cols, nrows)
            img_data[irow:irow + nrows, :] = np.require(row_data, dtype=np.float32)
            cov_data[irow:irow + nrows] = 1
            irow += increment
            status = import_progress(uuid=prod.uuid,
                                     stages=1,
                                     current_stage=0,
                                     completion=float(irow) / float(rows),
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
    'G16': Platform.GOES_16,
    'G17': Platform.GOES_17,
    'G18': Platform.GOES_18,
    'G19': Platform.GOES_19,
    # hsd2nc export of AHI data as PUG format
    'Himawari-8': Platform.HIMAWARI_8,
    'Himawari-9': Platform.HIMAWARI_9,
    # axi2cmi export as PUG, more consistent with other uses
    'H8': Platform.HIMAWARI_8,
    'H9': Platform.HIMAWARI_9
}


class GoesRPUGImporter(aSingleFileWithSingleProductImporter):
    """
    Import from PUG format GOES-16 netCDF4 files
    """

    @staticmethod
    def _basic_pug_metadata(pug):
        return {
            Info.PLATFORM: PLATFORM_ID_TO_PLATFORM[pug.platform_id],  # e.g. G16, H8
            Info.BAND: pug.band,
            Info.DATASET_NAME: 'B{:02d}'.format(pug.band),
            Info.INSTRUMENT: Instrument.AHI if 'Himawari' in pug.instrument_type else Instrument.ABI,
            Info.SCHED_TIME: pug.sched_time,
            Info.OBS_TIME: pug.time_span[0],
            Info.OBS_DURATION: pug.time_span[1] - pug.time_span[0],
            Info.DISPLAY_TIME: pug.display_time,
            Info.SCENE: pug.scene_id,
            Info.DISPLAY_NAME: pug.display_name,
        }

    @classmethod
    def is_relevant(cls, source_path=None, source_uri=None):
        source = source_path or source_uri
        return True if (source.lower().endswith('.nc') or source.lower().endswith('.nc4')) else False

    @staticmethod
    def pug_factory(source_path):
        dn, fn = os.path.split(source_path)
        is_netcdf = (fn.lower().endswith('.nc') or fn.lower().endswith('.nc4'))
        if not is_netcdf:
            raise ValueError("PUG loader requires files ending in .nc or .nc4: {}".format(repr(source_path)))
        return PugFile.attach(source_path)  # noqa
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
        # d[Info.DATASET_NAME] = os.path.split(source_path)[-1]
        d[Info.KIND] = Kind.IMAGE

        # FUTURE: this is Content metadata and not Product metadata:
        d[Info.PROJ] = pug.proj4_string
        # get nadir-meter-ish projection coordinate vectors to be used by proj4
        y, x = pug.proj_y, pug.proj_x
        d[Info.ORIGIN_X] = x[0]
        d[Info.ORIGIN_Y] = y[0]

        # midyi, midxi = int(y.shape[0] / 2), int(x.shape[0] / 2)
        # PUG states radiance at index [0,0] extends between coordinates [0,0] to [1,1] on a quadrille
        # centers of pixels are therefore at +0.5, +0.5
        # for a (e.g.) H x W image this means [H/2,W/2] coordinates are image center
        # for now assume all scenes are even-dimensioned (e.g. 5424x5424)
        # given that coordinates are evenly spaced in angular -> nadir-meters space,
        # technically this should work with any two neighbor values
        # d[Info.CELL_WIDTH] = x[midxi+1] - x[midxi]
        # d[Info.CELL_HEIGHT] = y[midyi+1] - y[midyi]
        cell_size = pug.cell_size
        d[Info.CELL_HEIGHT], d[Info.CELL_WIDTH] = cell_size

        shape = pug.shape
        d[Info.SHAPE] = shape
        generate_guidebook_metadata(d)

        d[Info.FAMILY] = '{}:{}:{}:{:5.2f}µm'.format(Kind.IMAGE.name, 'geo', d[Info.STANDARD_NAME], d[
            Info.CENTRAL_WAVELENGTH])  # kind:pointofreference:measurement:wavelength
        d[Info.CATEGORY] = 'NOAA-PUG:{}:{}:{}'.format(d[Info.PLATFORM].name, d[Info.INSTRUMENT].name,
                                                      d[Info.SCENE])  # system:platform:instrument:target
        d[Info.SERIAL] = d[Info.SCHED_TIME].strftime("%Y%m%dT%H%M%S")
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
            assert (products)
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
        img_data = np.memmap(data_path, dtype=np.float32, shape=shape, mode='w+')

        LOG.info('converting radiance to %s' % pug.bt_or_refl)
        image = pug.bt if 'bt' == pug.bt_or_refl else pug.refl
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
        img_data[:] = np.ma.fix_invalid(image, copy=False, fill_value=np.NAN)  # FIXME: expensive

        # create and commit a Content entry pointing to where the content is in the workspace, even if coverage is empty
        c = ContentImage(
            lod=0,
            resolution=int(min(abs(cell_width), abs(cell_height))),
            atime=now,
            mtime=now,

            # info about the data array memmap
            path=data_filename,
            rows=rows,
            cols=cols,
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


class SatpyImporter(aImporter):
    """Generic SatPy importer"""

    def __init__(self, source_paths, workspace_cwd, database_session, **kwargs):
        super(SatpyImporter, self).__init__(workspace_cwd, database_session)
        reader = kwargs.pop('reader', None)
        if reader is None:
            raise NotImplementedError("Can't automatically determine reader.")
        if not isinstance(source_paths, (list, tuple)):
            source_paths = [source_paths]

        if Scene is None:
            raise ImportError("SatPy is not available and can't be used as "
                              "an importer")
        self.filenames = list(source_paths)
        self.reader = reader
        self.resampling_info = kwargs.get('resampling_info')
        self.scn = kwargs.get('scene')
        self.merge_target = kwargs.get('merge_target')
        if self.scn is None:
            reader_kwargs = get_reader_kwargs_dict([self.reader])
            self.scn = Scene(filenames={self.reader: self.filenames},
                             reader_kwargs=reader_kwargs)
        self._resources = []
        # DataID filters
        self.product_filters = {}
        for k in ['resolution', 'calibration', 'level']:
            if k in kwargs:
                self.product_filters[k] = kwargs.pop(k)
        # NOTE: product_filters don't do anything if the dataset_ids aren't
        #       specified since we are using all available dataset ids
        self.dataset_ids = kwargs.get('dataset_ids')
        if self.dataset_ids is None:
            self.dataset_ids = filter_dataset_ids(self.scn.available_dataset_ids())
        self.dataset_ids = sorted(self.dataset_ids)

        self.use_inventory_db = USE_INVENTORY_DB
        self.requires = _get_requires(self.scn)

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
        kwargs['dataset_ids'] = [prod.info['_satpy_id']]
        filenames = [r.path for r in prod.resource]
        return cls(filenames, workspace_cwd=workspace_cwd, database_session=database_session, **kwargs)

    @classmethod
    def is_relevant(cls, source_path=None, source_uri=None):
        # this importer should only be used if specifically requested
        return False

    def merge_resources(self):
        if len(self._resources) == len(self.filenames):
            return self._resources
        if (self.use_inventory_db):
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
            if self.use_inventory_db:
                self._S.add(res)
            res_dict[fn] = res

        self._resources = res_dict.values()
        return self._resources

    def merge_products(self) -> Iterable[Product]:
        resources = self.merge_resources()
        if resources is None:
            LOG.debug('no resources for {}'.format(self.filenames))
            return

        existing_ids = {}
        if self.use_inventory_db:
            resources = list(resources)
            for res in resources:
                products = list(res.product)
                existing_ids.update({prod.info['_satpy_id']: prod for prod in products})
            existing_prods = {x: existing_ids[x] for x in self.dataset_ids if x in existing_ids}
            if products and len(existing_prods) == len(self.dataset_ids):
                products = existing_prods.values()
                LOG.debug('pre-existing products {}'.format(repr(products)))
                yield from products
                return

        from uuid import uuid1
        self.load_all_datasets()
        revised_datasets = self._revise_all_datasets()
        for ds_id, ds in revised_datasets.items():
            # don't recreate a Product for one we already have
            if ds_id in existing_ids:
                yield existing_ids[ds_id]
                continue

            meta = ds.attrs
            uuid = uuid1()
            meta[Info.UUID] = uuid
            meta['_satpy_id'] = ds_id
            now = datetime.utcnow()
            prod = Product(
                uuid_str=str(uuid),
                atime=now,
            )
            prod.resource.extend(resources)

            assert (Info.OBS_TIME in meta)
            assert (Info.OBS_DURATION in meta)
            prod.update(meta)  # sets fields like obs_duration and obs_time transparently
            assert (prod.info[Info.OBS_TIME] is not None and prod.obs_time is not None)
            assert (prod.info[Info.VALID_RANGE] is not None)
            LOG.debug('new product: {}'.format(repr(prod)))
            if self.use_inventory_db:
                self._S.add(prod)
                self._S.commit()
            yield prod

    def _extract_segment_number(self):
        """
        Load the segment numbers loaded by this scene.
        :return: list of all the segment numbers of all segments loaded by the scene
        """
        segments = []
        for reader in self.scn._readers.values():
            for file_handlers in reader.file_handlers.values():
                for file_handler in file_handlers:
                    if file_handler.filetype_info['file_type'] in self.requires:
                        continue
                    seg = file_handler.filename_info['segment']
                    segments.append(seg)
        return segments

    def _extract_expected_segments(self) -> Optional[int]:
        """
        Load the number of expected segments for the products loaded by the given scene.
        :return: number of expected segments or None if not found or products in scene
        have different numbers of expected segments
        """
        expected_segments = None
        for reader in self.scn._readers.values():
            for file_handlers in reader.file_handlers.values():
                for file_handler in file_handlers:
                    if file_handler.filetype_info['file_type'] in self.requires:
                        continue
                    es = file_handler.filetype_info['expected_segments']
                    if expected_segments is None:
                        expected_segments = es
                    if expected_segments != es:
                        return None
        return expected_segments

    @property
    def num_products(self) -> int:
        # WARNING: This could provide radiances and higher level products
        #          which SIFT probably shouldn't care about
        return len(self.dataset_ids)

    @staticmethod
    def _get_platform_instrument(attrs: dict):
        """Convert SatPy platform_name/sensor to """
        attrs[Info.INSTRUMENT] = attrs.get('sensor')
        attrs[Info.PLATFORM] = attrs.get('platform_name') or attrs.get('platform_shortname')

        # Special handling of GRIB forecast data
        if 'centreDescription' in attrs and \
                attrs[Info.INSTRUMENT] == 'unknown':
            description = attrs['centreDescription']
            if attrs.get(Info.PLATFORM) is None:
                attrs[Info.PLATFORM] = 'NWP'
            if 'NCEP' in description:
                attrs[Info.INSTRUMENT] = 'GFS'
        if attrs[Info.INSTRUMENT] in ['GFS', 'unknown']:
            attrs[Info.INSTRUMENT] = Instrument.GFS
        if attrs[Info.PLATFORM] in ['NWP', 'unknown']:
            attrs[Info.PLATFORM] = Platform.NWP

        # FUTURE: Use standard string names for platform/instrument
        #         instead of an Enum. Otherwise, could use a reverse
        #         Enum lookup to match Enum values to Enum keys.
        # if we haven't figured out what these are then give up and say they are unknown
        if isinstance(attrs[Info.PLATFORM], str):
            plat_str = attrs[Info.PLATFORM].lower().replace('-', '')
            attrs[Info.PLATFORM] = PLATFORM_MAP.get(plat_str, attrs[Info.PLATFORM])
        if not attrs[Info.PLATFORM] or isinstance(attrs[Info.PLATFORM], str):
            attrs[Info.PLATFORM] = Platform.UNKNOWN

        if isinstance(attrs[Info.INSTRUMENT], str):
            inst_str = attrs[Info.INSTRUMENT].lower().replace('-', '')
            attrs[Info.INSTRUMENT] = INSTRUMENT_MAP.get(inst_str, attrs[Info.INSTRUMENT])
        if not attrs[Info.INSTRUMENT] or isinstance(attrs[Info.INSTRUMENT], str):
            attrs[Info.INSTRUMENT] = Instrument.UNKNOWN

    def load_all_datasets(self) -> None:
        self.scn.load(self.dataset_ids, pad_data=False,
                      upper_right_corner="NE", **self.product_filters)
        # copy satpy metadata keys to SIFT keys
        for ds in self.scn:
            self._set_name_metadata(ds.attrs)
            self._set_time_metadata(ds.attrs)
            self._set_kind_metadata(ds.attrs)
            self._set_wavelength_metadata(ds.attrs)
            self._set_shape_metadata(ds.attrs, ds.shape)
            self._set_scene_metadata(ds.attrs)
            self._set_family_metadata(ds.attrs)
            self._set_category_metadata(ds.attrs)
            self._set_serial_metadata(ds.attrs)
            ds.attrs.setdefault('reader', self.reader)

    @staticmethod
    def _set_name_metadata(attrs: dict, name: Optional[str] = None,
                           short_name: Optional[str] = None,
                           long_name: Optional[str] = None) -> None:
        if not name:
            name = attrs['name'] or attrs.get(Info.STANDARD_NAME)

        id_str = ":".join(str(v[1]) for v in get_id_items(id_from_attrs(attrs)))
        attrs[Info.DATASET_NAME] = id_str

        model_time = attrs.get('model_time')
        if model_time is not None:
            attrs[Info.DATASET_NAME] += " " + model_time.isoformat()

        level = attrs.get('level')
        if level is None:
            attrs[Info.SHORT_NAME] = short_name or name
        else:
            attrs[Info.SHORT_NAME] = f"{short_name or name} @ {level}hPa"

        attrs[Info.LONG_NAME] = long_name or name
        attrs[Info.STANDARD_NAME] = attrs.get('standard_name') or name

    @staticmethod
    def _set_time_metadata(attrs: dict) -> None:
        start_time = attrs['start_time']
        attrs[Info.OBS_TIME] = start_time
        attrs[Info.SCHED_TIME] = start_time
        duration = attrs.get('end_time', start_time) - start_time
        if duration.total_seconds() <= 0:
            duration = timedelta(minutes=60)
        attrs[Info.OBS_DURATION] = duration

    def _set_kind_metadata(self, attrs: dict) -> None:
        reader_kind = config.get(f"data_reading.{self.reader}.kind", None)
        if reader_kind:
            try:
                attrs[Info.KIND] = Kind[reader_kind]
            except KeyError:
                raise KeyError(f"Unknown data kind '{reader_kind}'"
                               f" configured for reader {self.reader}.")
        else:
            LOG.info(f"No data kind configured for reader '{self.reader}'."
                     f" Falling back to 'IMAGE'.")
            attrs[Info.KIND] = Kind.IMAGE

    def _set_wavelength_metadata(self, attrs: dict) -> None:
        self._get_platform_instrument(attrs)
        if 'wavelength' in attrs:
            attrs.setdefault(Info.CENTRAL_WAVELENGTH, attrs['wavelength'][1])

    def _set_shape_metadata(self, attrs: dict, shape) -> None:
        attrs[Info.SHAPE] = shape \
            if not self.resampling_info else self.resampling_info['shape']
        attrs[Info.UNITS] = attrs.get('units')
        if attrs[Info.UNITS] == 'unknown':
            LOG.warning("Layer units are unknown, using '1'")
            attrs[Info.UNITS] = 1
        generate_guidebook_metadata(attrs)

    def _set_scene_metadata(self, attrs: dict) -> None:
        attrs[Info.SCENE] = attrs.get('scene_id')
        if attrs[Info.SCENE] is None:
            self._compute_scene_hash(attrs)

    @staticmethod
    def _set_family_metadata(attrs: dict) -> None:
        if attrs.get(Info.CENTRAL_WAVELENGTH) is None:
            cw = ""
        else:
            cw = ":{:5.2f}µm".format(attrs[Info.CENTRAL_WAVELENGTH])
        attrs[Info.FAMILY] = '{}:{}:{}{}'.format(
            attrs[Info.KIND].name, attrs[Info.STANDARD_NAME],
            attrs[Info.SHORT_NAME], cw)

    @staticmethod
    def _set_category_metadata(attrs: dict) -> None:
        # system:platform:instrument:target
        attrs[Info.CATEGORY] = 'SatPy:{}:{}:{}'.format(
            attrs[Info.PLATFORM].name, attrs[Info.INSTRUMENT].name,
            attrs[Info.SCENE])

    @staticmethod
    def _set_serial_metadata(attrs: dict) -> None:
        # TODO: Include level or something else in addition to time?
        start_str = attrs['start_time'].isoformat()
        if 'model_time' in attrs:
            model_time = attrs['model_time'].isoformat()
            attrs[Info.SERIAL] = f"{model_time}:{start_str}"
        else:
            attrs[Info.SERIAL] = start_str

    @staticmethod
    def _get_area_extent(area: Union[AreaDefinition, StackedAreaDefinition]) \
            -> List:
        """
        Return the area extent of the AreaDefinition or of the nominal
        AreaDefinition of the StackedAreaDefinition. The nominal AreaDefinition
        is the one referenced via 'area_id' by all AreaDefinitions listed in the
        StackedAreaDefinition. If there are different 'area_id's in the
        StackedAreaDefinition this function fails throwing an exception.

        Use case: Get the extent of the full-disk to which a non-contiguous set
        of segments loaded from a segmented GEOS file format (e.g. HRIT) belongs.

        :param area: AreaDefinition ar StackedAreaDefinition
        :return: Extent of the nominal AreaDefinition.
        :raises ValueError: when a StackedAreaDefinition with incompatible
        AreaDefinitions is passed
        """
        if isinstance(area, StackedAreaDefinition):
            any_area_def = area.defs[0]
            for area_def in area.defs:
                if area_def.area_id != any_area_def.area_id:
                    raise ValueError(f"Different area_ids found in StackedAreaDefinition: {area}.")
            area = satpy.resample.get_area_def(any_area_def.area_id)
        return area.area_extent

    def _compute_scene_hash(self, attrs: dict):
        """ Compute a "good enough" hash and store it as
        SCENE information.

        The SCENE information is used at other locations to identify data that
        has the roughly the same extent and projection. That is a  pre-requisite
        to allow to derive algebraics and compositions from them.

        Unstructured data has no clear extent and not an intrinsic projection
        (data locations are in latitude / longitude), thus something like a
        SCENE (in the sense of describing a view on a section of the earth's
        surface) cannot clearly be determined for it.
        """
        try:
            area = attrs['area'] if not self.resampling_info \
                else AreaDefinitionsManager.area_def_by_id(
                self.resampling_info['area_id'])
            # round extents to nearest 100 meters
            extents = tuple(int(np.round(x / 100.0) * 100.0)
                            for x in self._get_area_extent(area))
            attrs[Info.SCENE] = \
                "{}-{}".format(str(extents), area.proj_str)
        except (KeyError, AttributeError):
            # Scattered data, this is not suitable to define a scene
            attrs[Info.SCENE] = None

    def _stack_data_arrays(self, datasets: List[DataArray], attrs: dict,
                           name_prefix: str = None, axis: int = 1) -> DataArray:
        """
        Merge multiple DataArrays into a single ``DataArray``. Use the
        ``attrs`` dict for the DataArray metadata. This method also copies
        the Satpy metadata fields into SIFT fields. The ``attrs`` dict won't
        be modified.

        :param datasets: List of DataArrays
        :param attrs: metadata for the resulting DataArray
        :param name_prefix: if given, a prefix for the name of the new DataArray
        :param axis: numpy axis index
        :return: stacked Dask array
        """
        # Workaround for a Dask bug: Convert all DataArrays to float32
        # before calling into dask, because an int16 DataArray will be
        # converted into a Series instead of a dask Array with newer
        # versions. This then causes a TypeError.
        meta = np.stack([da.utils.meta_from_array(ds) for ds in datasets], axis=axis)
        datasets = [ds.astype(meta.dtype) for ds in datasets]
        combined_data = da.stack(datasets, axis=axis)

        attrs = attrs.copy()
        ds_id = attrs["_satpy_id"]
        name = f"{name_prefix or ''}{attrs['name']}"
        attrs["_satpy_id"] = DataID(ds_id.id_keys, name=name)
        self._set_name_metadata(attrs, name)
        self._set_time_metadata(attrs)
        self._set_kind_metadata(attrs)
        self._set_wavelength_metadata(attrs)
        self._set_shape_metadata(attrs, combined_data.shape)

        guidebook = get_guidebook_class(attrs)
        attrs[Info.DISPLAY_NAME] = guidebook._default_display_name(attrs)

        self._set_scene_metadata(attrs)
        self._set_family_metadata(attrs)
        self._set_category_metadata(attrs)

        return DataArray(combined_data, attrs=attrs)

    def _parse_style_attr_config(self) -> Dict[str, List[str]]:
        """
        Extract the ``style_attributes`` section from the reader config.
        This function doesn't validate whether the style attributes or the
        product names exist.

        :return: mapping of style attributes to products
        """
        style_config = config.get(f"data_reading.{self.reader}.style_attributes", None)
        if not style_config:
            return {}

        style_attrs = {}
        for attr_name, product_names in style_config.items():
            if attr_name in style_attrs:
                LOG.warning(f"duplicate style attribute: {attr_name}")
                continue

            if not isinstance(product_names, list):
                # a single product is allowed
                product_names = [product_names]

            distinct_products = []
            for product_name in product_names:
                if product_name in distinct_products:
                    LOG.warning(f"duplicate product {product_name} for "
                                f"style attribute: {attr_name}")
                    continue
                if product_name:
                    distinct_products.append(product_name)

            style_attrs[attr_name] = distinct_products
        return style_attrs

    def _combine_points(self, datasets: DatasetDict, converter) \
            -> Dict[DataID, DataArray]:
        """
        Find convertible POINTS datasets in the ``DatasetDict``. The ``converter``
        function is then used to generate new DataArrays.

        The latitude and longitude for the points are extracted from the dataset
        metadata. If configured in the reader config, the dataset itself will be
        used for the ``fill`` style attribute.

        :param datasets: all loaded datasets and previously converted datasets
        :param converter: function to convert a list of DataArrays
        :return: mapping of converted DataID to new DataArray
        """
        style_attrs = self._parse_style_attr_config()
        allowed_style_attrs = ["fill"]

        converted_datasets = {}
        for style_attr, product_names in style_attrs.items():
            if style_attr not in allowed_style_attrs:
                LOG.error(f"unknown style attribute: {style_attr}")
                continue

            for product_name in product_names:
                try:
                    ds = datasets[product_name]
                except KeyError:
                    LOG.debug(f"product wasn't selected in ImportWizard: {product_name}")
                    continue

                kind = ds.attrs[Info.KIND]
                if kind != Kind.POINTS:
                    LOG.error(f"dataset {product_name} isn't of POINTS kind: {kind}")
                    continue

                try:
                    convertable_ds = [ds.area.lons, ds.area.lats, ds]
                except AttributeError:
                    # Some products (e.g. `latitude` and `longitude`) may not
                    # (for whatever reason) have an associated SwathDefinition.
                    # Without it, there is no geo-location information per data
                    # point, because it is taken from its fields 'lats', 'lons'.
                    # This cannot be healed, point data loading fails.
                    LOG.error(f"Dataset has no point coordinates (lats, lons):"
                              f" {product_name} (Most likely due to missing"
                              f" SwathDefinition)")
                    continue

                ds_id = DataID.from_dataarray(ds)
                converted_datasets[ds_id] = converter(convertable_ds, ds.attrs)

        # All unused Datasets aren't defined in the style attributes config.
        # If one of them has an area, use it to display uncolored points.
        for ds_id, ds in datasets.items():
            if ds_id in converted_datasets:
                continue
            if ds.attrs[Info.KIND] == Kind.POINTS:
                try:
                    convertable_ds = [ds.area.lons, ds.area.lats]
                except AttributeError:
                    LOG.error(f"Dataset has no point coordinates (lats, lons):"
                              f" {ds.attrs['name']} (Most likely due to missing"
                              f" SwathDefinition)")
                    continue
                converted_datasets[ds_id] = converter(convertable_ds, ds.attrs)

        return converted_datasets

    def _parse_coords_end_config(self) -> List[str]:
        """
        Parse the ``coordinates_end`` section of the reader config.

        :return: List of ``coords`` identifiers
        """
        coords_end = config.get(f"data_reading.{self.reader}.coordinates_end", None)
        if not coords_end:
            return []

        if len(coords_end) < 2 or len(coords_end) > 2:
            LOG.warning("expected 2 end coordinates for LINES")
        return coords_end

    def _combine_lines(self, datasets: DatasetDict, converter) \
            -> Dict[DataID, DataArray]:
        """
        Find convertible LINES datasets in the ``DatasetDict``. The ``converter``
        function is then used to generate new DataArrays.

        The positions for the tip and base of the lines are extracted from the
        dataset metadata. The dataset itself will be discarded.

        :param datasets: all loaded datasets and previously converted datasets
        :param converter: function to convert a list of DataArrays
        :return: mapping of converted DataID to new DataArray
        """
        coords_end = self._parse_coords_end_config()

        converted_datasets = {}
        for ds_id, ds in datasets.items():
            if ds.attrs[Info.KIND] != Kind.LINES:
                continue

            convertable_ds = []
            try:
                for coord in coords_end:
                    convertable_ds.append(ds.coords[coord])  # base
            except KeyError:
                LOG.error(f"dataset has no coordinates: {ds.attrs['name']}")
                continue
            if len(convertable_ds) < 2:
                LOG.error(f"LINES dataset needs 4 coordinates: {ds.attrs['name']}")
                continue

            convertable_ds.extend([ds.area.lons, ds.area.lats])  # tip
            converted_datasets[ds_id] = converter(convertable_ds, ds.attrs)
        return converted_datasets

    def _revise_all_datasets(self) -> DatasetDict:
        """
        Revise all datasets and convert the data representation of the POINTS
        and LINES records found to the data format VisPy needs to display them
        as such.

        The original datasets are not kept in the scene but replaced by the
        converted datasets.

        :return: ``DatasetDict`` with the loaded and converted datasets
        """
        loaded_datasets = DatasetDict()
        for ds_id in self.scn.keys():
            ds = self.scn[ds_id]
            loaded_datasets[ds_id] = ds

        # Note: If you do not want the converted data sets to overwrite the
        # original data sets in a future development of this software,
        # you can pass a suitable prefix to self._stack_data_arrays() (which
        # will be used to create a new name for the converted data set) by
        # defining the appropriate converters, for example, as follows:
        #   self._combine_points: lambda *args: self._stack_data_arrays(*args, "POINTS-"),
        converters = {
            self._combine_points: lambda *args: self._stack_data_arrays(*args),
            self._combine_lines: lambda *args: self._stack_data_arrays(*args),
        }

        for detector, converter in converters.items():
            converted_datasets = detector(loaded_datasets, converter)
            for old_ds_id, new_ds in converted_datasets.items():
                del loaded_datasets[old_ds_id]

                new_ds_id = DataID.from_dataarray(new_ds)
                loaded_datasets[new_ds_id] = new_ds

        for ds_id, ds in loaded_datasets.items():
            self.scn[ds_id] = ds
        return loaded_datasets

    @staticmethod
    def _area_to_sift_attrs(area):
        """Area to uwsift keys"""
        if not isinstance(area, AreaDefinition):
            raise NotImplementedError("Only AreaDefinition datasets can "
                                      "be loaded at this time.")

        return {
            Info.PROJ:         area.proj_str,
            Info.ORIGIN_X:     area.area_extent[0],  # == lower_(left_)x
            Info.ORIGIN_Y:     area.area_extent[3],  # == upper_(right_)y
            Info.CELL_WIDTH:   area.pixel_size_x,
            Info.CELL_HEIGHT: -area.pixel_size_y,
        }

    def _get_grid_info(self):
        grid_origin = \
            config.get(f"data_reading.{self.reader}.grid.origin",
                       "NW")
        grid_first_index_x = \
            config.get(f"data_reading.{self.reader}.grid.first_index_x",
                       0)
        grid_first_index_y = \
            config.get(f"data_reading.{self.reader}.grid.first_index_x",
                       0)

        return {
            Info.GRID_ORIGIN:  grid_origin,
            Info.GRID_FIRST_INDEX_X: grid_first_index_x,
            Info.GRID_FIRST_INDEX_Y: grid_first_index_y,
        }

    def begin_import_products(self, *product_ids) -> Generator[import_progress, None, None]:
        if self.use_inventory_db:
            if product_ids:
                products = [self._S.query(Product).filter_by(id=anid).one() for anid in product_ids]
                assert products
            else:
                products = list(self._S.query(Resource, Product).filter(
                    Resource.path.in_(self.filenames)).filter(
                    Product.resource_id == Resource.id).all())
                assert products
        else:
            products = product_ids

        merge_with_existing = self.merge_target is not None

        # FIXME: Don't recreate the importer every time we want to load data
        dataset_ids = [prod.info['_satpy_id'] for prod in products]
        self.scn.load(dataset_ids, pad_data=not merge_with_existing,
                      upper_right_corner="NE")

        if self.resampling_info:
            resampler: str = self.resampling_info['resampler']
            max_area = self.scn.max_area()
            if isinstance(max_area, AreaDefinition) and \
                    max_area.area_id == self.resampling_info['area_id']:
                LOG.info(f"Source and target area ID are identical:"
                         f" '{self.resampling_info['area_id']}'."
                         f" Skipping resampling.")
            else:
                area_name = max_area.area_id if hasattr(max_area, 'area_id') \
                    else max_area.name
                LOG.info(f"Resampling from area ID/name '{area_name}'"
                         f" to area ID '{self.resampling_info['area_id']}'"
                         f" with method '{resampler}'")
                # Use as many processes for resampling as the number of CPUs
                # the application can use.
                # See https://pyresample.readthedocs.io/en/latest/multi.html
                #  and https://docs.python.org/3/library/os.html#os.cpu_count
                nprocs = len(os.sched_getaffinity(0))

                target_area_def = AreaDefinitionsManager.area_def_by_id(
                    self.resampling_info['area_id'])

                # About the next strange line of code: keep a reference to the
                # original scene to work around an issue in the resampling
                # implementation for NetCDF data: otherwise the original data
                # would be garbage collected too early.
                self.scn_original = self.scn  # noqa - Do not simply remove
                self.scn = self.scn.resample(
                    target_area_def,
                    resampler=resampler,
                    nprocs=nprocs,
                    radius_of_influence=self.resampling_info['radius_of_influence'])

        num_stages = len(products)
        for idx, (prod, ds_id) in enumerate(zip(products, dataset_ids)):
            dataset = self.scn[ds_id]
            shape = dataset.shape
            # Since in the first SatpyImporter loading pass (see
            # load_all_datasets()) no padding is applied (pad_data=False), the
            # Info.SHAPE stored at that time may not be the same as it is now if
            # we load with padding now. In that case we must update the
            # prod.info[Info.SHAPE] with the actual shape, but let's do this in
            # any case, it doesn't hurt.
            prod.info[Info.SHAPE] = shape
            kind = prod.info[Info.KIND]
            # TODO (Alexander Rettig): Review this, best with David Hoese:
            #  The line (exactly speaking, code equivalent to it) was
            #  introduced in commit bba8d73f but looked very suspicious - it
            #  seems to assume, that the only kinds here could be IMAGE or
            #  CONTOUR:
            #     num_contents = 1 if kind == Kind.IMAGE else 2
            #  Assuming that num_contents == 2 is the right setting only for
            #  kind == Kind.CONTOUR, the following implementation is supposed to
            #  do better:
            num_contents = 2 if kind == Kind.CONTOUR else 1

            if prod.content:
                LOG.warning('content was already available, skipping import')
                continue

            now = datetime.utcnow()

            if prod.info[Info.KIND] in [Kind.LINES, Kind.POINTS]:
                if len(shape) == 1:
                    LOG.error(f"one dimensional dataset can't be loaded: {ds_id['name']}")
                    continue

                data_filename, data_memmap = \
                    self._create_data_memmap_file(dataset.data, dataset.dtype,
                                                  prod)
                content = ContentUnstructuredPoints(
                    atime=now,
                    mtime=now,

                    # info about the data array memmap
                    path=data_filename,
                    n_points=shape[0],
                    n_dimensions=shape[1],
                    dtype=dataset.dtype,
                )
                content.info[Info.KIND] = kind
                prod.content.append(content)
                self.add_content_to_cache(content)

                completion = 2. / num_contents
                yield import_progress(uuid=prod.uuid,
                                      stages=num_stages,
                                      current_stage=idx,
                                      completion=completion,
                                      stage_desc=f"SatPy {kind.name} data add to workspace",
                                      dataset_info=None,
                                      data=data_memmap,
                                      content=content)
                continue

            data = dataset.data
            uuid = prod.uuid

            if merge_with_existing:
                existing_product = self.merge_target
                segments = self._extract_segment_number()
                uuid = existing_product.uuid
                c = existing_product.content[0]
                img_data = c.img_data
                self.merge_data_into_memmap(data, img_data, segments)
            else:
                area_info = self._area_to_sift_attrs(dataset.attrs['area'])
                cell_width = area_info[Info.CELL_WIDTH]
                cell_height = area_info[Info.CELL_HEIGHT]
                proj4 = area_info[Info.PROJ]
                origin_x = area_info[Info.ORIGIN_X]
                origin_y = area_info[Info.ORIGIN_Y]

                grid_info = self._get_grid_info()
                grid_origin = grid_info[Info.GRID_ORIGIN]
                grid_first_index_x = grid_info[Info.GRID_FIRST_INDEX_X]
                grid_first_index_y = grid_info[Info.GRID_FIRST_INDEX_Y]

                # Handle building contours for data from 0 to 360 longitude
                # Handle building contours for data from 0 to 360 longitude
                # our map's antimeridian is 180
                antimeridian = 179.999
                # find out if there is data beyond 180 in this data
                if '+proj=latlong' not in proj4:
                    # the x coordinate for the antimeridian in this projection
                    am = Proj(proj4)(antimeridian, 0)[0]
                    if am >= 1e30:
                        am_index = -1
                    else:
                        am_index = int(np.ceil((am - origin_x) / cell_width))
                else:
                    am_index = int(np.ceil((antimeridian - origin_x) / cell_width))
                # if there is data beyond 180, let's copy it to the beginning of the
                # array so it shows up in the primary -180/0 portion of the SIFT map
                if prod.info[Info.KIND] == Kind.CONTOUR and 0 < am_index < shape[1]:
                    # Previous implementation:
                    # Prepend a copy of the last half of the data (180 to 360 -> -180 to 0)
                    # data = da.concatenate((data[:, am_index:], data), axis=1)
                    # adjust X origin to be -180
                    # origin_x -= (shape[1] - am_index) * cell_width
                    # The above no longer works with newer PROJ because +over is deprecated
                    #
                    # New implementation:
                    # Swap the 180 to 360 portion to    be -180 to 0
                    # if we have data from 0 to 360 longitude, we want -180 to 360
                    data = da.concatenate((data[:, am_index:], data[:, :am_index]), axis=1)
                    # remove the custom 180 prime meridian in the projection
                    proj4 = proj4.replace("+pm=180 ", "")
                    area_info[Info.PROJ] = proj4

                # For kind IMAGE the dtype must be float32 seemingly, see class
                # Column, comment for 'dtype' and the construction of c = Content
                # just below.
                # FIXME: It is dubious to enforce that type conversion to happen in
                #  _create_data_memmap_file, but otherwise IMAGES of pixel counts
                #  data (dtype = np.uint16) crash.
                data_filename, img_data = \
                    self._create_data_memmap_file(data, np.float32, prod)

                c = ContentImage(
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

                    grid_origin=grid_origin,
                    grid_first_index_x=grid_first_index_x,
                    grid_first_index_y=grid_first_index_y
                )
                c.info[Info.KIND] = Kind.IMAGE
                c.img_data = img_data
                c.source_files = set()  # FIXME(AR) is this correct?
                # c.info.update(prod.info) would just make everything leak together so let's not do it

                prod.content.append(c)
                self.add_content_to_cache(c)

            required_files = _required_files_set(self.scn, self.requires)
            # Note loaded source files but not required files. This is necessary
            # because the merger will try to not load already loaded files again
            # but might need to reload required files.
            c.source_files |= (set(self.filenames) - required_files)

            yield import_progress(uuid=uuid,
                                  stages=num_stages,
                                  current_stage=idx,
                                  completion=1. / num_contents,
                                  stage_desc="SatPy IMAGE data add to workspace",
                                  dataset_info=None,
                                  data=img_data,
                                  content=c)

            if num_contents == 1:
                continue

            if find_contours is None:
                raise RuntimeError("Can't create contours without 'skimage' "
                                   "package installed.")

            # XXX: Should/could 'lod' be used for different contour level data?
            levels = [x for y in prod.info['contour_levels'] for x in y]

            try:
                contour_data = self._compute_contours(
                    img_data, prod.info[Info.VALID_RANGE][0],
                    prod.info[Info.VALID_RANGE][1], levels)
            except ValueError:
                LOG.warning("Could not compute contour levels for '{}'".format(prod.uuid))
                LOG.debug("Contour error: ", exc_info=True)
                continue

            contour_data[:, 0] *= cell_width
            contour_data[:, 0] += origin_x
            contour_data[:, 1] *= cell_height
            contour_data[:, 1] += origin_y
            data_filename: str = self.create_contour_file_cache_data(contour_data,
                                                                prod)
            c = ContentImage(
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
            c.info[Info.KIND] = Kind.CONTOUR
            # c.info["level_index"] = level_indexes
            self.add_content_to_cache(c)

            completion = 2. / num_contents
            yield import_progress(uuid=uuid,
                                  stages=num_stages,
                                  current_stage=idx,
                                  completion=completion,
                                  stage_desc="SatPy CONTOUR data add to workspace",
                                  dataset_info=None,
                                  data=img_data,
                                  content=c)


    def get_fci_segment_height(self, segment_number: int, segment_width: int) -> int:
        try:
            fci_width_to_grid_type = {
                3712: '3km',
                5568: '2km',
                11136: '1km',
                22272: '500m'}
            seg_heights = self.scn._readers[self.reader].segment_heights
            file_type = list(seg_heights.keys())[0]
            grid_type = fci_width_to_grid_type[segment_width]
            return seg_heights[file_type][grid_type][segment_number-1]
        except AttributeError:
            LOG.warning("You must be using an old version of Satpy; Please "
                        "update Satpy (>v0.37) to make sure that the merging of"
                        " FCI chunks in existing datasets works correctly "
                        "for any input data. Using fallback  satpy.readers."
                        "yaml_reader._get_FCI_L1c_FDHSI_chunk_height to compute"
                        " chunk heights.")
            return satpy.readers.yaml_reader._get_FCI_L1c_FDHSI_chunk_height(
                segment_width, segment_number)

    def _calc_segment_heights(self, segments_data, segments_indices):
        def get_segment_height_calculator() -> Callable:
            if self.reader in ['fci_l1c_nc', 'fci_l1c_fdhsi']:
                return self.get_fci_segment_height
            else:
                return lambda x, y: segments_data.shape[0] // len(segments_indices)

        calc_segment_height = get_segment_height_calculator()
        expected_num_segments = self._extract_expected_segments()
        segment_heights = np.zeros((expected_num_segments,))
        for i in range(1, expected_num_segments + 1):
            segment_heights[i - 1] = calc_segment_height(i, segments_data.shape[1])
        return segment_heights.astype(int)

    def _determine_segments_to_image_mapping(self, segments_data, segments_indices)\
            -> Tuple[List, List]:

        segment_heights = self._calc_segment_heights(segments_data, segments_indices)

        segment_starts_stops = []
        image_starts_stops = []
        visited_segments = []
        for segment_idx, segment_num in enumerate(segments_indices):
            curr_segment_height_idx = segment_num - 1
            curr_segment_height = segment_heights[curr_segment_height_idx]

            image_stop_pos = segment_heights[curr_segment_height_idx:].sum()
            image_start_pos = image_stop_pos - curr_segment_height

            curr_segment_stop = segments_data.shape[0]
            if visited_segments:
                curr_segment_stop -= segment_heights[visited_segments].sum()
            curr_segment_start = curr_segment_stop - curr_segment_height

            visited_segments.append(curr_segment_height_idx)
            segment_starts_stops.append((curr_segment_start, curr_segment_stop))
            image_starts_stops.append((image_start_pos, image_stop_pos))
        return segment_starts_stops, image_starts_stops

    def merge_data_into_memmap(self, segments_data, image_data, segments_indices):
        """
        Merge new segments from data into img_data. The data is expected to
        contain the data for all segments with the last segment (highest segment
        number) as first in the data array.
        The segments list defines which chunk of data belongs to which segment.
        The list must be sorted ascending.
        Those requirements match the current Satpy behavior which loads segments
        in ascending order and produce an data array with the last segments data
        first.
        :param segments_data: the segments data as provided by Satpy importer
        :param image_data: the layer data where the segments are merged into
        :param segments_indices: list of segments whose data is to be merged
        Note: this is not the highest segment number in the current segments
        list but the highest segment number which can appear for the product.
        """

        segment_starts_stops, image_starts_stops = \
            self._determine_segments_to_image_mapping(segments_data,
                                                      segments_indices)

        for i in range(len(segment_starts_stops)):
            segment_start = segment_starts_stops[i][0]
            segment_stop = segment_starts_stops[i][1]
            image_start = image_starts_stops[i][0]
            image_stop = image_starts_stops[i][1]
            image_data[image_start:image_stop, :] = \
                segments_data[segment_start:segment_stop, :]

    def create_contour_file_cache_data(self, contour_data: np.ndarray,
                                       prod: Product) -> str:
        """
        Creating a file in the current work directory for saving data in
        '.contour' files
        """
        data_filename: str = '{}.contour'.format(prod.uuid)
        data_path: str = os.path.join(self._cwd, data_filename)
        contour_data.tofile(data_path)

        return data_filename

    def add_content_to_cache(self, c: Content) -> None:
        if self.use_inventory_db:
            self._S.add(c)
            self._S.commit()

    def _create_data_memmap_file(self, data: da.array, dtype, prod: Product) \
            -> Tuple[str, Optional[np.memmap]]:
        """
        Create *binary* file in the current working directory to cache `data`
        as numpy memmap. The filename extension of the file is derived from the
        kind of the given `prod`.

        'dtype' must be given explicitly because it is not necessarily
        ``dtype == data.dtype`` (caller decides).
        """
        # shovel that data into the memmap incrementally

        kind = prod.info[Info.KIND]
        data_filename = '{}.{}'.format(prod.uuid, kind.name.lower())
        data_path = os.path.join(self._cwd, data_filename)
        if data is None or data.size == 0:
            # For empty data memmap is not possible
            return data_filename, None

        data_memmap = np.memmap(data_path, dtype=dtype, shape=data.shape,
                                mode='w+')
        da.store(data, data_memmap)

        return data_filename, data_memmap

    def _get_verts_and_connect(self, paths):
        """ retrieve vertices and connects from given paths-list
        """
        # THIS METHOD WAS COPIED FROM VISPY
        verts = np.vstack(paths)
        gaps = np.add.accumulate(np.array([len(x) for x in paths])) - 1
        connect = np.ones(gaps[-1], dtype=bool)
        connect[gaps[:-1]] = False
        return verts, connect

    def _compute_contours(self, img_data: np.ndarray, vmin: float, vmax: float, levels: list) -> np.ndarray:
        all_levels = []
        empty_level = np.array([[0, 0, 0, 0]], dtype=np.float32)
        for level in levels:
            if level < vmin or level > vmax:
                all_levels.append(empty_level)
                continue

            contours = find_contours(img_data, level, positive_orientation='high')
            if not contours:
                LOG.debug("No contours found for level: {}".format(level))
                all_levels.append(empty_level)
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
            c = np.concatenate((c, [False])).astype(np.float32)
            # level_data = np.concatenate((v, c[:, None]), axis=1)
            level_data = np.concatenate((v, c[:, None], this_level[:, None]), axis=1)
            all_levels.append(level_data)

        if not all_levels:
            raise ValueError("No valid contour levels")
        return np.concatenate(all_levels).astype(np.float32)
