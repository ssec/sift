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
from abc import ABC, abstractmethod
from collections import namedtuple
from datetime import datetime, timedelta
from typing import (
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    Union,
)

import dask.array as da
import numpy as np
import satpy.readers.yaml_reader
import satpy.resample
import yaml
from pyresample.geometry import AreaDefinition, StackedAreaDefinition, SwathDefinition
from satpy import DataQuery, Scene, available_readers
from satpy.dataset import DatasetDict
from satpy.writers import get_enhanced_image
from xarray import DataArray

from uwsift import USE_INVENTORY_DB, config
from uwsift.common import INSTRUMENT_MAP, PLATFORM_MAP, Info, Instrument, Kind, Platform
from uwsift.model.area_definitions_manager import AreaDefinitionsManager
from uwsift.satpy_compat import DataID, get_id_items, get_id_value, id_from_attrs
from uwsift.util import USER_CACHE_DIR
from uwsift.util.common import get_reader_kwargs_dict
from uwsift.workspace.guidebook import ABI_AHI_Guidebook

from .metadatabase import (
    Content,
    ContentImage,
    ContentMultiChannelImage,
    ContentUnstructuredPoints,
    Product,
    Resource,
)
from .utils import metadata_utils

_SATPY_READERS = None  # cache: see `available_satpy_readers()` below
SATPY_READER_CACHE_FILE = os.path.join(USER_CACHE_DIR, "available_satpy_readers.yaml")

LOG = logging.getLogger(__name__)

satpy_version = None

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

import_progress = namedtuple(
    "import_progress",
    ["uuid", "stages", "current_stage", "completion", "stage_desc", "dataset_info", "data", "content"],
)
"""
# stages:int, number of stages this import requires
# current_stage:int, 0..stages-1 , which stage we're on
# completion:float, 0..1 how far we are along on this stage
# stage_desc:tuple(str), brief description of each of the stages we'll be doing
"""

try:
    import hdf5plugin  # noqa F401 # suppress "'hdf5plugin' imported but unused"
except ImportError:
    LOG.warning(
        "Your installation/environment doesn't include the package hdf5plugin. If you want to visualise "
        'compressed FCI L1c data, please add this package using "pip install hdf5plugin" or '
        '"conda install -c conda-forge hdf5plugin "..'
    )


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
        with open(SATPY_READER_CACHE_FILE, "r") as cfile:
            LOG.info("Loading cached available Satpy readers from {}".format(SATPY_READER_CACHE_FILE))
            cache_contents = yaml.load(cfile, yaml.SafeLoader)
        if cache_contents is None:
            raise RuntimeError("Cached reader list is empty, regenerating...")
        if cache_contents["satpy_version"] != satpy_version:
            raise RuntimeError("Satpy has different version, regenerating available readers...")
    except (FileNotFoundError, RuntimeError, KeyError) as cause:
        LOG.info("Updating list of available Satpy readers...")
        cause.__suppress_context__ = True
        readers = available_readers(as_dict=True)
        # sort list of readers just in case we depend on this in the future
        readers = sorted(readers, key=lambda x: x["name"])
        readers = list(_sanitize_reader_info_for_yaml(readers))
        cache_contents = {
            "satpy_version": satpy_version,
            "readers": readers,
        }
        _save_satpy_readers_cache(cache_contents)
    return cache_contents["readers"]


def _sanitize_reader_info_for_yaml(readers):
    # filter out known python objects to simplify YAML serialization
    for reader_info in readers:
        reader_info.pop("reader")
        reader_info.pop("data_identification_keys", None)
        reader_info["config_files"] = list(reader_info["config_files"])
        yield reader_info


def _save_satpy_readers_cache(cache_contents):
    """Write reader cache information to a file on disk."""
    cfile_dir = os.path.dirname(SATPY_READER_CACHE_FILE)
    os.makedirs(cfile_dir, exist_ok=True)
    with open(SATPY_READER_CACHE_FILE, "w") as cfile:
        LOG.info("Caching available Satpy readers to {}".format(SATPY_READER_CACHE_FILE))
        yaml.dump(cache_contents, cfile)


def available_satpy_readers(as_dict=False, force_cache_refresh=None):
    """Get a list of reader names or reader information."""
    global _SATPY_READERS
    if _SATPY_READERS is None or force_cache_refresh:
        _SATPY_READERS = _load_satpy_readers_cache(force_refresh=force_cache_refresh)

    if not as_dict:
        return [r["name"] for r in _SATPY_READERS]
    return _SATPY_READERS


def filter_dataset_ids(ids_to_filter: Iterable[DataID]) -> Generator[DataID, None, None]:
    """Generate only non-filtered DataIDs based on EXCLUDE_DATASETS global filters."""
    # skip certain DataIDs
    for ds_id in ids_to_filter:
        for filter_key, filtered_values in config.get("open_file_wizard.exclude_datasets").items():
            if filtered_values and get_id_value(ds_id, filter_key) in filtered_values:
                break
        else:
            yield ds_id


def get_guidebook_class(dataset_info) -> ABI_AHI_Guidebook:
    platform = dataset_info.get(Info.PLATFORM)
    return GUIDEBOOKS.get(platform, DEFAULT_GUIDEBOOK)()


def generate_guidebook_metadata(info) -> Mapping:
    guidebook = get_guidebook_class(info)
    # also get more values for this info from the guidebook
    gbinfo = guidebook.collect_info(info)
    info.update(gbinfo)  # FUTURE: should guidebook be integrated into suitable place?

    # add as visible to the front of the current set, and invisible to the rest of the available sets
    info[Info.COLORMAP] = metadata_utils.get_default_colormap(info, guidebook)
    if info[Info.KIND] == Kind.POINTS:
        info[Info.STYLE] = metadata_utils.get_default_point_style_name(info)
    valid_range = guidebook.valid_range(info)
    if valid_range is not None:
        info[Info.VALID_RANGE] = valid_range
    if Info.DISPLAY_TIME not in info:
        info[Info.DISPLAY_TIME] = guidebook._default_display_time(info)
    if Info.DISPLAY_NAME not in info:
        info[Info.DISPLAY_NAME] = guidebook._default_display_name(info)

    return info


def _get_types_of_required_aux_files(scn: Scene) -> Set[str]:
    """
    Get the file types of required auxiliary files in the given Scene.

    For example, the types 'HRIT_PRO', 'HRIT_EPI' for the prolog/epilog files of
    SEVIRI in HRIT format.

    :param scn: Scene object to analyse
    :return: set of required file types
    """
    required_aux_file_types = set()
    for reader in scn._readers.values():
        for file_handlers in reader.file_handlers.values():
            for file_handler in file_handlers:
                required_aux_file_types |= set(file_handler.filetype_info.get("requires", ()))
    return required_aux_file_types


def _get_paths_of_required_aux_files(scn: Scene, required_aux_file_types) -> Set[str]:
    """
    Get set of paths of auxiliary files required to load data in Scene.
    For example, prolog/epilog files for SEVIRI in HRIT format.

    :param scn: Scene object to analyse
    :param required_aux_file_types: the file types which are required
    :return: set of required files
    """
    required_aux_paths = set()
    if required_aux_file_types:
        for reader in scn._readers.values():
            for file_handlers in reader.file_handlers.values():
                for file_handler in file_handlers:
                    if file_handler.filetype_info["file_type"] in required_aux_file_types:
                        required_aux_paths.add(file_handler.filename)
    return required_aux_paths


def _get_paths_in_scene_contributing_to_ds(scn: Scene, ds_name: str) -> List[str]:
    """
    Get list of paths in scene which actually contain data for the dataset given
    by its name `ds_name`.
    :param scn: scene object to analyse
    :param ds_name: dataset name
    :return: paths in scene which are required to load given dataset
    """
    contributing_paths = []
    for reader in scn._readers.values():
        for file_handlers in reader.file_handlers.values():
            for file_handler in file_handlers:
                if not ds_name or not hasattr(file_handler, "channel_name") or file_handler.channel_name == ds_name:
                    contributing_paths.append(file_handler.filename)
    return contributing_paths


class aImporter(ABC):
    """
    Abstract Importer class creates or amends Resource, Product, Content entries in the metadatabase used by Workspace
    aImporter instances are backgrounded by the Workspace to bring Content into the workspace
    """

    def __init__(self, workspace_cwd, database_session, **kwargs):
        super(aImporter, self).__init__()
        # dedicated sqlalchemy database session to use during this import instance;
        # revert if necessary, commit as appropriate
        self._S = database_session
        # where content flat files should be imported to within the workspace, omit this from content path
        self._cwd = workspace_cwd

    @classmethod
    def from_product(cls, prod: Product, workspace_cwd, database_session, **kwargs):
        # FIXME: deal with products that need more than one resource
        try:
            cls = prod.resource[0].format
        except IndexError:
            LOG.error("no resources in {} {}".format(repr(type(prod)), repr(prod)))
            raise
        paths = [r.path for r in prod.resource]

        # Get the Satpy Scene for this set of paths
        scn = kwargs["scenes"].get(tuple(paths), None)

        # Only Satpy readers / SatpyImporter still supported, thus:
        # There must be such a Scene and the prod must be from Satpy, we can
        assert scn and "_satpy_id" in prod.info  # nosec B101

        merge_target = kwargs.get("merge_target")
        if merge_target:
            # For the merging process it is crucial that it only has
            # to deal with files which belong to the given Product `prod`,
            # extraneous files would interfere with the merging process.
            # Thus: Filter the files in `paths` and keep only those
            # that contribute to the Satpy dataset for `prod`.
            if "prerequisites" in prod.info:
                paths = cls._collect_prerequisites_paths(prod, scn)
            else:
                paths = _get_paths_in_scene_contributing_to_ds(scn, prod.info["_satpy_id"].get("name"))

            # Now we can analyse, which files must be loaded
            paths = cls._extract_paths_to_merge(merge_target, paths, scn)
            if not paths:
                return None

        # In order to "load" converted datasets, we have to reuse the existing
        # Scene 'scn' from the first instantiation of SatpyImporter instead of
        # loading the data again, of course: we need to 'rescue across' the
        # converted data from the first SatpyImporter instantiation to the
        # second one (which will be created in the very last statement of this
        # method).
        # Since only products of the kinds POINTS, LINES and VECTORS need
        # conversion (at this time), the mechanism is only applied for these.
        # For an unknown reason reusing the scene breaks for FCI data,
        #
        # this Importer data reading magic is too confused.
        if prod.info[Info.KIND] in (Kind.POINTS, Kind.LINES, Kind.VECTORS):
            del kwargs["scenes"]
            kwargs["scene"] = scn

        # TODO: ignore mypy error for now because in the future aImporter should be merged with SatpyImporter
        return cls(paths, workspace_cwd=workspace_cwd, database_session=database_session, **kwargs)  # type: ignore

    @classmethod
    def _extract_paths_to_merge(cls, merge_target, paths, scn):
        """
        Get a subset of 'paths', filtering out all files that are already loaded
        into the target product.

        I.e., collect only files from the given 'paths' list that are not yet
        loaded and, if necessary, auxiliary files required to load these.
        """
        new_paths = []
        existing_content = merge_target.content[0]
        for path in paths:
            if path not in existing_content.source_files:
                new_paths.append(path)
        if not new_paths:
            return None
        if new_paths != paths:
            required_aux_file_types = _get_types_of_required_aux_files(scn)
            required_aux_file_paths = _get_paths_of_required_aux_files(scn, required_aux_file_types)
            if set(new_paths) == required_aux_file_paths:
                return None
            return new_paths + list(required_aux_file_paths)  # TODO
        return new_paths

    @classmethod
    def _collect_prerequisites_paths(cls, prod, scn):
        # If the dataset has prerequisites - this is the case for
        # RGB composites provided by Satpy - the files that are
        # required by these must be collected.
        paths = []
        for prerequisite in prod.info.get("prerequisites"):
            ds_name = prerequisite.get("name") if isinstance(prerequisite, DataQuery) else prerequisite
            if ds_name not in scn.available_dataset_names():
                # This is a not supported case: Merging should be
                # done for an RGB composite of which prequisite
                # datasets are missing in the scene. This can
                # happen, when the composite depends on other
                # composites - their datasets have not been created
                # in the Scene at this stage, thus the Scene can't
                # be queried for the files required by these datasets.
                # We have to interrupt this loading process (which
                # runs in its own thread) by raising an exception.
                raise RuntimeError(
                    f"The Satpy RGB Composite type '{prod.info[Info.SHORT_NAME]}'"
                    f" does not work when merging of new data chunks"
                    f" into existing data is active."
                    f" Consider switching it off by configuring"
                    f" 'data_reading.merge_with_existing: False'"
                )
            paths += _get_paths_in_scene_contributing_to_ds(scn, ds_name)
        # remove possible duplicates in paths
        paths = list(dict.fromkeys(paths))
        return paths

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
        pass


def determine_dynamic_dataset_kind(attrs: dict, reader_name: str) -> str:
    """Determine kind of dataset dynamically based on dataset attributes.

    This currently supports only the distinction between IMAGE and POINTS kinds.
    It makes the assumption that if the dataset has a SwathDefinition, and is 1-D, it represents points.
    """
    if isinstance(attrs["area"], SwathDefinition) and len(attrs["area"].shape) == 1:
        data_kind = "POINTS"
    else:
        data_kind = "IMAGE"
    LOG.info(f"Selected dynamically kind {data_kind} for dataset {attrs['name']} from reader {reader_name}")
    return data_kind


def set_kind_metadata_from_reader_config(reader_name: str, reader_kind: str, attrs: dict) -> None:
    """Determine the dataset kind starting from the reader configuration."""
    data_kind = determine_dynamic_dataset_kind(attrs, reader_name) if reader_kind == "DYNAMIC" else reader_kind

    try:
        attrs[Info.KIND] = Kind[data_kind]
    except KeyError:
        raise KeyError(f"Unknown data kind '{data_kind}' used for reader {reader_name}.")


class SatpyImporter(aImporter):
    """Generic SatPy importer"""

    def __init__(self, source_paths, workspace_cwd, database_session, **kwargs):
        super(SatpyImporter, self).__init__(workspace_cwd, database_session)
        reader = kwargs.pop("reader", None)
        if reader is None:
            raise NotImplementedError("Can't automatically determine reader.")
        if not isinstance(source_paths, (list, tuple)):
            source_paths = [source_paths]

        if Scene is None:
            raise ImportError("SatPy is not available and can't be used as " "an importer")
        self.filenames = list(source_paths)
        self.reader = reader
        self.resampling_info = kwargs.get("resampling_info")
        self.scn = kwargs.get("scene")
        self.merge_target = kwargs.get("merge_target")
        if self.scn is None:
            reader_kwargs = get_reader_kwargs_dict([self.reader])
            self.scn = Scene(filenames={self.reader: self.filenames}, reader_kwargs=reader_kwargs)
        self._resources = []
        # DataID filters
        self.product_filters = {}
        for k in ["resolution", "calibration", "level"]:
            if k in kwargs:
                self.product_filters[k] = kwargs.pop(k)
        # NOTE: product_filters don't do anything if the dataset_ids aren't
        #       specified since we are using all available dataset ids
        self.dataset_ids = kwargs.get("dataset_ids")
        if self.dataset_ids is None:
            self.dataset_ids = filter_dataset_ids(self.scn.available_dataset_ids())
        self.dataset_ids = sorted(self.dataset_ids)

        self.use_inventory_db = USE_INVENTORY_DB
        self.required_aux_file_types = _get_types_of_required_aux_files(self.scn)

        # NOTE: needed for an issue with resampling of NetCDF data.
        self.scn_original = None  # noqa - Do not simply remove

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
            LOG.error("no resources in {} {}".format(repr(type(prod)), repr(prod)))
            raise
        kwargs.pop("reader", None)
        kwargs.pop("scenes", None)
        kwargs.pop("scene", None)
        kwargs["dataset_ids"] = [prod.info["_satpy_id"]]
        filenames = [r.path for r in prod.resource]
        return cls(filenames, workspace_cwd=workspace_cwd, database_session=database_session, **kwargs)

    @classmethod
    def is_relevant(cls, source_path=None, source_uri=None):
        # this importer should only be used if specifically requested
        return False

    def merge_resources(self):
        if len(self._resources) == len(self.filenames):
            return self._resources
        if self.use_inventory_db:
            resources = self._S.query(Resource).filter(Resource.path.in_(self.filenames)).all()
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
            LOG.debug("no resources for {}".format(self.filenames))
            return

        existing_ids = {}
        if self.use_inventory_db:
            resources = list(resources)
            for res in resources:
                products = list(res.product)
                existing_ids.update({prod.info["_satpy_id"]: prod for prod in products})
            existing_prods = {x: existing_ids[x] for x in self.dataset_ids if x in existing_ids}
            # FIXME: products may be unset in next line, its use is suspicious anyway
            if products and len(existing_prods) == len(self.dataset_ids):
                products = list(existing_prods.values())
                LOG.debug("pre-existing products {}".format(repr(products)))
                yield from products
                return

        from uuid import uuid1

        self._load_all_datasets()
        revised_datasets = self._revise_all_datasets()
        for ds_id, ds in revised_datasets.items():
            # don't recreate a Product for one we already have
            if ds_id in existing_ids:
                yield existing_ids[ds_id]
                continue

            meta = ds.attrs
            uuid = uuid1()
            meta[Info.UUID] = uuid
            meta["_satpy_id"] = ds_id
            now = datetime.utcnow()
            prod = Product(
                uuid_str=str(uuid),
                atime=now,
            )
            prod.resource.extend(resources)

            assert Info.OBS_TIME in meta  # nosec B101
            assert Info.OBS_DURATION in meta  # nosec B101
            prod.update(meta)  # sets fields like obs_duration and obs_time transparently
            assert prod.info[Info.OBS_TIME] is not None and prod.obs_time is not None  # nosec B101
            LOG.debug("new product: {}".format(repr(prod)))
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
        collected_segment_count = 0
        for reader in self.scn._readers.values():
            for file_handlers in reader.file_handlers.values():
                collected_segment_count += 1
                for file_handler in file_handlers:
                    if file_handler.filetype_info["file_type"] in self.required_aux_file_types:
                        collected_segment_count -= 1
                        continue
                    seg = file_handler.filename_info["segment"]
                    segments.append(seg)

        filtered_segments = []
        for segment in set(segments):
            if segments.count(segment) == collected_segment_count:
                filtered_segments.append(segment)

        return sorted(filtered_segments)

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
                    if file_handler.filetype_info["file_type"] in self.required_aux_file_types:
                        continue
                    es = file_handler.filetype_info["expected_segments"]
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
        """Convert SatPy platform_name/sensor to"""
        attrs[Info.INSTRUMENT] = attrs.get("sensor")
        attrs[Info.PLATFORM] = attrs.get("platform_name") or attrs.get("platform_shortname")

        # Special handling of GRIB forecast data
        if "centreDescription" in attrs and attrs[Info.INSTRUMENT] == "unknown":
            description = attrs["centreDescription"]
            if attrs.get(Info.PLATFORM) is None:
                attrs[Info.PLATFORM] = "NWP"
            if "NCEP" in description:
                attrs[Info.INSTRUMENT] = "GFS"
        if attrs[Info.INSTRUMENT] in ["GFS", "unknown"]:
            attrs[Info.INSTRUMENT] = Instrument.GFS
        if attrs[Info.PLATFORM] in ["NWP", "unknown"]:
            attrs[Info.PLATFORM] = Platform.NWP

        # FUTURE: Use standard string names for platform/instrument
        #         instead of an Enum. Otherwise, could use a reverse
        #         Enum lookup to match Enum values to Enum keys.
        # if we haven't figured out what these are then give up and say they are unknown
        if isinstance(attrs[Info.PLATFORM], str):
            plat_str = attrs[Info.PLATFORM].lower().replace("-", "")
            attrs[Info.PLATFORM] = PLATFORM_MAP.get(plat_str, attrs[Info.PLATFORM])
        if not attrs[Info.PLATFORM] or isinstance(attrs[Info.PLATFORM], str):
            attrs[Info.PLATFORM] = Platform.UNKNOWN

        if isinstance(attrs[Info.INSTRUMENT], str):
            inst_str = attrs[Info.INSTRUMENT].lower().replace("-", "")
            attrs[Info.INSTRUMENT] = INSTRUMENT_MAP.get(inst_str, attrs[Info.INSTRUMENT])
        if not attrs[Info.INSTRUMENT] or isinstance(attrs[Info.INSTRUMENT], str):
            attrs[Info.INSTRUMENT] = Instrument.UNKNOWN

    def _load_all_datasets(self) -> None:
        self.scn.load(self.dataset_ids, pad_data=False, upper_right_corner="NE", **self.product_filters)
        # copy satpy metadata keys to SIFT keys

        if self.scn.missing_datasets:
            # deactivating reduce_data, see https://github.com/pytroll/satpy/issues/2476
            self.scn = self.scn.resample(resampler="native", reduce_data=False)

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
            ds.attrs.setdefault("reader", self.reader)

    @staticmethod
    def _set_name_metadata(
        attrs: dict, name: Optional[str] = None, short_name: Optional[str] = None, long_name: Optional[str] = None
    ) -> None:
        if not name:
            name = attrs["name"] or attrs.get(Info.STANDARD_NAME)

        id_str = ":".join(str(v[1]) for v in get_id_items(id_from_attrs(attrs)))
        attrs[Info.DATASET_NAME] = id_str

        model_time = attrs.get("model_time")
        if model_time is not None:
            attrs[Info.DATASET_NAME] += " " + model_time.isoformat()

        level = attrs.get("level")
        if level is None:
            attrs[Info.SHORT_NAME] = short_name or name
        else:
            attrs[Info.SHORT_NAME] = f"{short_name or name} @ {level}hPa"

        attrs[Info.LONG_NAME] = long_name or name
        attrs[Info.STANDARD_NAME] = attrs.get("standard_name") or name

    @staticmethod
    def _determine_sched_time(attrs: dict):
        if "time_parameters" in attrs:
            return attrs["time_parameters"]["nominal_start_time"]
        if "nominal_start_time" in attrs:
            return attrs["nominal_start_time"]
        return attrs["start_time"]

    @staticmethod
    def _set_time_metadata(attrs: dict) -> None:
        # TODO Review Satpy start/end times ("[{nominal,observation]_}{start,end}_times") and adjust SIFT
        #  terminology regarding start/end times and durations accordingly.
        start_time = attrs["start_time"]
        attrs[Info.OBS_TIME] = start_time
        attrs[Info.SCHED_TIME] = SatpyImporter._determine_sched_time(attrs)
        duration = attrs.get("end_time", start_time) - start_time
        if duration.total_seconds() <= 0:
            duration = timedelta(minutes=60)
        attrs[Info.OBS_DURATION] = duration

    def _set_kind_metadata(self, attrs: dict) -> None:
        if "prerequisites" in attrs:
            attrs[Info.KIND] = Kind.MC_IMAGE
            return

        reader_kind = config.get(f"data_reading.{self.reader}.kind", None)

        if reader_kind is None:
            LOG.info(f"No data kind configured for reader '{self.reader}'. Falling back to 'IMAGE'.")
            attrs[Info.KIND] = Kind.IMAGE
        else:
            set_kind_metadata_from_reader_config(self.reader, reader_kind, attrs)

    def _set_wavelength_metadata(self, attrs: dict) -> None:
        self._get_platform_instrument(attrs)
        if "wavelength" in attrs and attrs.get("wavelength"):
            attrs.setdefault(Info.CENTRAL_WAVELENGTH, attrs["wavelength"][1])

    def _set_shape_metadata(self, attrs: dict, shape) -> None:
        if len(shape) == 3 and "prerequisites" in attrs.keys():
            shape = (shape[1], shape[2], shape[0])
        attrs[Info.SHAPE] = shape if not self.resampling_info else self.resampling_info["shape"]
        attrs[Info.UNITS] = attrs.get("units")
        if attrs[Info.UNITS] == "unknown":
            LOG.warning("Dataset units are unknown, using '1'")
            attrs[Info.UNITS] = 1
        generate_guidebook_metadata(attrs)

    def _set_scene_metadata(self, attrs: dict) -> None:
        attrs[Info.SCENE] = attrs.get("scene_id")
        if attrs[Info.SCENE] is None:
            self._compute_scene_hash(attrs)

    @staticmethod
    def _set_family_metadata(attrs: dict) -> None:
        if attrs.get(Info.CENTRAL_WAVELENGTH) is None:
            cw = ""
        else:
            cw = ":{:5.2f}µm".format(attrs[Info.CENTRAL_WAVELENGTH])
        attrs[Info.FAMILY] = "{}:{}:{}{}".format(
            attrs[Info.KIND].name, attrs[Info.STANDARD_NAME], attrs[Info.SHORT_NAME], cw
        )

    @staticmethod
    def _set_category_metadata(attrs: dict) -> None:
        # system:platform:instrument:target
        attrs[Info.CATEGORY] = "SatPy:{}:{}:{}".format(
            attrs[Info.PLATFORM].name, attrs[Info.INSTRUMENT].name, attrs[Info.SCENE]
        )

    @staticmethod
    def _set_serial_metadata(attrs: dict) -> None:
        # TODO: Include level or something else in addition to time?
        start_str = attrs["start_time"].isoformat()
        if "model_time" in attrs:
            model_time = attrs["model_time"].isoformat()
            attrs[Info.SERIAL] = f"{model_time}:{start_str}"
        else:
            attrs[Info.SERIAL] = start_str

    @staticmethod
    def _get_area_extent(area: Union[AreaDefinition, StackedAreaDefinition]) -> List:
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
        """Compute a "good enough" hash and store it as
        SCENE information.

        The SCENE information is used at other locations to identify data that
        has roughly the same extent and projection. That is a  pre-requisite
        to allow to derive algebraics and compositions from them.

        Unstructured data has no clear extent and not an intrinsic projection
        (data locations are in latitude / longitude), thus something like a
        SCENE (in the sense of describing a view on a section of the earth's
        surface) cannot clearly be determined for it.
        """
        try:
            area = (
                attrs["area"]
                if not self.resampling_info
                else AreaDefinitionsManager.area_def_by_id(self.resampling_info["area_id"])
            )
            # round extents to nearest 100 meters
            extents = tuple(int(np.round(x / 100.0) * 100.0) for x in self._get_area_extent(area))
            attrs[Info.SCENE] = "{}-{}".format(str(extents), area.proj_str)
        except (KeyError, AttributeError):
            # Scattered data, this is not suitable to define a scene
            attrs[Info.SCENE] = None

    def _stack_data_arrays(
        self, datasets: List[DataArray], attrs: dict, name_prefix: str = "", axis: int = 1
    ) -> DataArray:
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

    def _parse_style_attributes_config(self) -> Dict[str, Set[str]]:
        """
        Extract the ``style_attributes`` section from the reader config.
        This function doesn't validate whether the style attributes or the
        product names exist.

        The returned dictionary lists for each product, which dependent styles
        it is configured for.

        Maintenance: this swaps the mapping direction since the new one is more
        useful for querying which features (style attributes) should be styled depending
        on a given product

        :return: mapping of products to style attributes
        """
        style_attributes = config.get(f"data_reading.{self.reader}.style_attributes", None)
        if not style_attributes:
            return {}

        style_attributes_by_product = {}
        for style_attribute, product_names in style_attributes.items():
            for product_name in product_names:
                if product_name not in style_attributes_by_product:
                    style_attributes_by_product[product_name] = {style_attribute}
                else:
                    style_attributes_by_product[product_name].add(style_attribute)

        return style_attributes_by_product

    def _combine_points(self, datasets: DatasetDict, converter) -> Dict[DataID, DataArray]:
        # Currently only the "fill" can be colored by a mapping from product values to colors (using a colormap) but in
        # the future other features could be colored as well, e.g.  like "stroke".
        supported_color_by_style_attrs = {"fill"}
        style_attributes_by_product = self._parse_style_attributes_config()

        converted_datasets = {}
        for ds_id, ds in datasets.items():
            if ds.attrs[Info.KIND] != Kind.POINTS:
                continue

            ds_name = ds_id["name"]
            do_color_by_values = ds_name in style_attributes_by_product and bool(
                style_attributes_by_product[ds_name] & supported_color_by_style_attrs
            )
            try:
                # Only if we want to use the values of the data to color
                # markers we need the dataset's array itself, otherwise the
                # coordinates alone are sufficient.
                convertible_ds = (
                    [ds.area.lons, ds.area.lats, ds] if do_color_by_values else [ds.area.lons, ds.area.lats]
                )
            except AttributeError:
                # Some products (e.g. `latitude` and `longitude`) may not
                # (for whatever reason) have an associated SwathDefinition.
                # Without it, there is no geolocation information per data
                # point, because it is taken from its fields 'lats', 'lons'.
                # This cannot be healed, point data loading fails.
                LOG.error(
                    f"Dataset '{ds.attrs['name']}' of kind POINTS has no point"
                    f" coordinates (lats, lons) (Missing SwathDefinition)."
                )
                continue
            converted_datasets[ds_id] = converter(convertible_ds, ds.attrs)
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

    def _combine_lines(self, datasets: DatasetDict, converter) -> Dict[DataID, DataArray]:
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

            convertible_ds = []
            try:
                for coord in coords_end:
                    convertible_ds.append(ds.coords[coord])  # base
            except KeyError:
                LOG.error(f"dataset has no coordinates: {ds.attrs['name']}")
                continue
            if len(convertible_ds) < 2:
                LOG.error(f"LINES dataset needs 4 coordinates: {ds.attrs['name']}")
                continue

            convertible_ds.extend([ds.area.lons, ds.area.lats])  # tip
            converted_datasets[ds_id] = converter(convertible_ds, ds.attrs)
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
            raise NotImplementedError(
                "Only AreaDefinition datasets can be loaded at this time. "
                "If you're trying to read data that needs resampling, like LEO data, remember "
                "to select a resampling method in the Open File Wizard, and set "
                "'geometry_definition: SwathDefinition' in your reader config."
            )

        return {
            Info.PROJ: area.proj_str,
            Info.ORIGIN_X: area.area_extent[0],  # == lower_(left_)x
            Info.ORIGIN_Y: area.area_extent[3],  # == upper_(right_)y
            Info.CELL_WIDTH: area.pixel_size_x,
            Info.CELL_HEIGHT: -area.pixel_size_y,
        }

    def _get_grid_info(self):
        grid_origin = config.get(f"data_reading.{self.reader}.grid.origin", "NW")
        grid_first_index_x = config.get(f"data_reading.{self.reader}.grid.first_index_x", 0)
        grid_first_index_y = config.get(f"data_reading.{self.reader}.grid.first_index_y", 0)

        return {
            Info.GRID_ORIGIN: grid_origin,
            Info.GRID_FIRST_INDEX_X: grid_first_index_x,
            Info.GRID_FIRST_INDEX_Y: grid_first_index_y,
        }

    def begin_import_products(self, *product_ids) -> Generator[import_progress, None, None]:  # noqa: C901
        if self.use_inventory_db:
            products = self._get_products_from_inventory_db(product_ids)
        else:
            products = product_ids

        merge_with_existing = self.merge_target is not None

        # FIXME: Don't recreate the importer every time we want to load data
        dataset_ids = [prod.info["_satpy_id"] for prod in products]
        self.scn.load(dataset_ids, pad_data=not merge_with_existing, upper_right_corner="NE")

        # If there are datasets missing, resampling is needed to create them:
        if self.scn.missing_datasets:
            # About the next strange line of code: keep a reference to the
            # original scene to work around an issue in the resampling
            # implementation for NetCDF data: otherwise the original data
            # would be garbage collected too early.
            self.scn_original = self.scn  # noqa - Do not simply remove

            # deactivating reduce_data, see https://github.com/pytroll/satpy/issues/2476
            self.scn = self.scn.resample(resampler="native", reduce_data=False)

        if self.resampling_info:
            self._preprocess_products_with_resampling()

        num_stages = len(products)
        for idx, (prod, ds_id) in enumerate(zip(products, dataset_ids)):
            dataset = (
                self.scn[ds_id] if prod.info[Info.KIND] != Kind.MC_IMAGE else get_enhanced_image(self.scn[ds_id]).data
            )
            if prod.info[Info.KIND] == Kind.MC_IMAGE:
                # The dimension of the dataset is ('bands', 'y', 'x')
                # but for later the dimension ('y', 'x', 'bands') is needed
                dataset = dataset.transpose("y", "x", "bands")
            # Since in the first SatpyImporter loading pass (see
            # load_all_datasets()) no padding is applied (pad_data=False), the
            # Info.SHAPE stored at that time may not be the same as it is now if
            # we load with padding now. In that case we must update the
            # prod.info[Info.SHAPE] with the actual shape, but let's do this in
            # any case, it doesn't hurt.
            prod.info[Info.SHAPE] = dataset.shape
            kind = prod.info[Info.KIND]

            if prod.content:
                LOG.warning("content was already available, skipping import")
                continue

            now = datetime.utcnow()

            if kind in [Kind.LINES, Kind.POINTS]:
                if len(dataset.shape) == 1:
                    LOG.error(f"one dimensional dataset can't be loaded: {ds_id['name']}")
                    continue

                completion, content, data_memmap = self._create_unstructured_points_dataset_content(dataset, now, prod)
                yield import_progress(
                    uuid=prod.uuid,
                    stages=num_stages,
                    current_stage=idx,
                    completion=completion,
                    stage_desc=f"SatPy {kind.name} data add to workspace",
                    dataset_info=None,
                    data=data_memmap,
                    content=content,
                )
                continue

            uuid = prod.uuid

            if merge_with_existing:
                existing_product = self.merge_target
                segments = self._extract_segment_number()
                uuid = existing_product.uuid
                c = existing_product.content[0]
                img_data = c.img_data
                self.merge_data_into_memmap(dataset.data, img_data, segments)
            else:
                area_info = self._area_to_sift_attrs(dataset.attrs["area"])
                grid_info = self._get_grid_info()

                if kind == Kind.MC_IMAGE:
                    c, img_data = self._create_mc_image_dataset_content(dataset, now, prod, area_info, grid_info)
                else:
                    c, img_data = self._create_image_dataset_content(dataset, now, prod, area_info, grid_info)

                c.info[Info.KIND] = prod.info[Info.KIND]
                c.img_data = img_data
                c.source_files = set()  # TODO(AR): adds member to the ContentImage object 'c'
                # c.info.update(prod.info) would just make everything leak together so let's not do it
                prod.content.append(c)
                self._add_content_to_cache(c)

            required_files = _get_paths_of_required_aux_files(self.scn, self.required_aux_file_types)
            # Note loaded source files but not required files. This is necessary
            # because the merger will try to not load already loaded files again
            # but might need to reload required files.
            c.source_files |= set(self.filenames) - required_files

            yield import_progress(
                uuid=uuid,
                stages=num_stages,
                current_stage=idx,
                completion=1.0,
                stage_desc=f"SatPy {kind.name} data add to workspace",
                dataset_info=None,
                data=img_data,
                content=c,
            )

    def _get_products_from_inventory_db(self, product_ids):
        if product_ids:
            products = [self._S.query(Product).filter_by(id=anid).one() for anid in product_ids]
            assert products  # nosec B101
        else:
            products = list(
                self._S.query(Resource, Product)
                .filter(Resource.path.in_(self.filenames))
                .filter(Product.resource_id == Resource.id)
                .all()
            )
            assert products  # nosec B101
        return products

    def _create_mc_image_dataset_content(self, dataset, now, prod, area_info, grid_info):
        data_filename, img_data = self._create_data_memmap_file(dataset.data, dataset.data.dtype, prod)
        shape = prod.info[Info.SHAPE]
        c = ContentMultiChannelImage(
            lod=0,
            resolution=int(min(abs(area_info[Info.CELL_WIDTH]), abs(area_info[Info.CELL_HEIGHT]))),
            atime=now,
            mtime=now,
            # info about the data array memmap
            path=data_filename,
            rows=shape[0],
            cols=shape[1],
            bands=shape[2],
            proj4=area_info[Info.PROJ],
            dtype=str(dataset.dtype),
            cell_width=area_info[Info.CELL_WIDTH],
            cell_height=area_info[Info.CELL_HEIGHT],
            origin_x=area_info[Info.ORIGIN_X],
            origin_y=area_info[Info.ORIGIN_Y],
            grid_origin=grid_info[Info.GRID_ORIGIN],
            grid_first_index_x=grid_info[Info.GRID_FIRST_INDEX_X],
            grid_first_index_y=grid_info[Info.GRID_FIRST_INDEX_Y],
        )
        return c, img_data

    def _create_image_dataset_content(self, dataset, now, prod, area_info, grid_info):
        # For kind IMAGE the dtype must be float32 seemingly, see class
        # Column, comment for 'dtype' and the construction of c = Content
        # just below.
        # FIXME: It is dubious to enforce that type conversion to happen in
        #  _create_data_memmap_file, but otherwise IMAGES of pixel counts
        #  data (dtype = np.uint16) crash.
        data_filename, img_data = self._create_data_memmap_file(dataset.data, np.float32, prod)
        shape = prod.info[Info.SHAPE]
        c = ContentImage(
            lod=0,
            resolution=int(min(abs(area_info[Info.CELL_WIDTH]), abs(area_info[Info.CELL_HEIGHT]))),
            atime=now,
            mtime=now,
            # info about the data array memmap
            path=data_filename,
            rows=shape[0],
            cols=shape[1],
            proj4=area_info[Info.PROJ],
            dtype="float32",
            cell_width=area_info[Info.CELL_WIDTH],
            cell_height=area_info[Info.CELL_HEIGHT],
            origin_x=area_info[Info.ORIGIN_X],
            origin_y=area_info[Info.ORIGIN_Y],
            grid_origin=grid_info[Info.GRID_ORIGIN],
            grid_first_index_x=grid_info[Info.GRID_FIRST_INDEX_X],
            grid_first_index_y=grid_info[Info.GRID_FIRST_INDEX_Y],
        )
        return c, img_data

    def _create_unstructured_points_dataset_content(self, dataset, now, prod):
        data_filename, data_memmap = self._create_data_memmap_file(dataset.data, dataset.dtype, prod)
        shape = prod.info[Info.SHAPE]
        content = ContentUnstructuredPoints(
            atime=now,
            mtime=now,
            # info about the data array memmap
            path=data_filename,
            n_points=shape[0],
            n_dimensions=shape[1],
            dtype=dataset.dtype,
        )
        content.info[Info.KIND] = prod.info[Info.KIND]
        prod.content.append(content)
        self._add_content_to_cache(content)
        completion = 2.0
        return completion, content, data_memmap

    def _preprocess_products_with_resampling(self) -> None:
        resampler: str = self.resampling_info["resampler"]
        max_area = self.scn.finest_area()
        if (
            isinstance(max_area, AreaDefinition)
            and max_area.area_id == self.resampling_info["area_id"]
            and self.resampling_info["custom"] is False
        ):
            # resampling is not needed
            LOG.info(
                f"Source and target area ID are identical:"
                f" '{self.resampling_info['area_id']}'."
                f" Skipping resampling."
            )
        else:
            # resampling is needed
            area_name = max_area.area_id if hasattr(max_area, "area_id") else max_area.name
            target_area_def = AreaDefinitionsManager.area_def_by_id(self.resampling_info["area_id"])

            if self.resampling_info["custom"] is False:
                LOG.info(
                    f"Resampling from area ID/name '{area_name}'"
                    f" to area ID '{self.resampling_info['area_id']}'"
                    f" with method '{resampler}'"
                )
            else:
                # custom resolution
                resampler_shape = self.resampling_info["shape"]
                resampler_width = resampler_shape[0]
                resampler_height = resampler_shape[1]
                LOG.info(
                    f"Resampling area with the characteristics of '{area_name}'"
                    f" with method '{resampler}'"
                    f" with CUSTOM SHAPE: '{resampler_width}, {resampler_height}'"
                )

                AreaDefinitionsManager.prepare_area_def_for_resampling(
                    target_area_def, resampler_width, resampler_height
                )

            # About the next strange line of code: keep a reference to the
            # original scene to work around an issue in the resampling
            # implementation for NetCDF data: otherwise the original data
            # would be garbage collected too early.
            if not self.scn_original:
                self.scn_original = self.scn  # noqa - Do not simply remove

            # deactivating reduce_data, see https://github.com/pytroll/satpy/issues/2476
            reduce_data = False if resampler == "native" else True
            self.scn = self.scn.resample(
                target_area_def,
                resampler=resampler,
                radius_of_influence=self.resampling_info["radius_of_influence"],
                reduce_data=reduce_data,
            )

    def _get_fci_segment_height(self, segment_number: int, segment_width: int) -> int:
        try:
            seg_heights = self.scn._readers[self.reader].segment_heights
            # we can assume that we're reading only from one filetype (either FDHSI or HRFI)
            file_type = list(self.scn._readers[self.reader].file_handlers.keys())[0]
            return seg_heights(file_type, segment_width)[segment_number - 1]
        except AttributeError:
            LOG.exception(
                "You're probably using an old version of Satpy. Please update Satpy to be >=0.39.0 "
                "to use merge_with_existing=True with the FCI reader."
            )
            raise

        return -1

    def _calc_segment_heights(self, segments_data, segments_indices):
        def get_segment_height_calculator() -> Callable:
            if self.reader in ["fci_l1c_nc", "fci_l1c_fdhsi"]:
                return self._get_fci_segment_height
            else:
                return lambda x, y: segments_data.shape[0] // len(segments_indices)

        calc_segment_height = get_segment_height_calculator()
        expected_num_segments = self._extract_expected_segments()
        segment_heights = np.zeros((expected_num_segments,))
        for i in range(1, expected_num_segments + 1):
            segment_heights[i - 1] = calc_segment_height(i, segments_data.shape[1])
        return segment_heights.astype(int)

    def _determine_segments_to_image_mapping(self, segments_data, segments_indices) -> Tuple[List, List]:
        segment_heights = self._calc_segment_heights(segments_data, segments_indices)

        segment_starts_stops = []
        image_starts_stops = []
        visited_segments: List[int] = []
        for segment_num in segments_indices:
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
        contain the data for all segments with the last segment (that has the
        largest segment number) as first in the data array.
        The segments list defines which chunk of data belongs to which segment.
        The list must be sorted ascending.
        Those requirements match the current Satpy behavior which loads segments
        in ascending order and produce a data array with the last segments data
        first.
        :param segments_data: the segments data as provided by Satpy importer
        :param image_data: the dataset data where the segments are merged into
        :param segments_indices: list of segments whose data is to be merged
        Note: this is not the highest segment number in the current segments
        list but the highest segment number which can appear for the product.
        """

        segment_starts_stops, image_starts_stops = self._determine_segments_to_image_mapping(
            segments_data, segments_indices
        )

        for i in range(len(segment_starts_stops)):
            segment_start = segment_starts_stops[i][0]
            segment_stop = segment_starts_stops[i][1]
            image_start = image_starts_stops[i][0]
            image_stop = image_starts_stops[i][1]
            image_data[image_start:image_stop, :] = segments_data[segment_start:segment_stop, :]

    def _add_content_to_cache(self, c: Content) -> None:
        if self.use_inventory_db:
            self._S.add(c)
            self._S.commit()

    def _create_data_memmap_file(self, data: da.array, dtype, prod: Product) -> Tuple[str, Optional[np.memmap]]:
        """
        Create *binary* file in the current working directory to cache `data`
        as numpy memmap. The filename extension of the file is derived from the
        kind of the given `prod`.

        'dtype' must be given explicitly because it is not necessarily
        ``dtype == data.dtype`` (caller decides).
        """
        # shovel that data into the memmap incrementally

        kind = prod.info[Info.KIND]
        data_filename = "{}.{}".format(prod.uuid, kind.name.lower())
        data_path = os.path.join(self._cwd, data_filename)
        if data is None or data.size == 0:
            # For empty data memmap is not possible
            return data_filename, None

        data_memmap = np.memmap(data_path, dtype=dtype, shape=data.shape, mode="w+")
        da.store(data, data_memmap)

        return data_filename, data_memmap
