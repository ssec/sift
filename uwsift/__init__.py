#!/usr/bin/env python

import os
import sys
from importlib.machinery import PathFinder
from importlib.util import spec_from_file_location

from donfig import Config

from .util.default_paths import DOCUMENT_SETTINGS_DIR
from .version import __version__  # noqa

BASE_CONFIG_DIR = os.path.join(DOCUMENT_SETTINGS_DIR, "config")
READERS_CONFIG_DIR = os.path.join(BASE_CONFIG_DIR, "readers")
os.environ.setdefault("UWSIFT_CONFIG", BASE_CONFIG_DIR)

CONFIG_PATHS = [
    BASE_CONFIG_DIR,
    READERS_CONFIG_DIR,
    os.path.join(os.path.expanduser("~"), ".config", "uwsift"),
]

# EXPERIMENTAL: This functionality is experimental and may change in future
#     releases until it is an advertised feature.
DEFAULT_CONFIGURATION = {
    # settings for storing temporary or persistent data on the file system
    # this preset is for using a simple workspace without any caching
    "storage": {"use_inventory_db": False, "cleanup_file_cache": True},
    # related to any reading of data
    "data_reading": {
        # What readers to use when opening files
        # None => all readers
        # from environment variable: export UWSIFT_DATA_READING__READERS = "['abi_l1b', 'ami_l1b']"
        # 'readers': None,
        "readers": [
            "abi_l1b",
            "ahi_hrit",
            "ahi_hsd",
            "ami_l1b",
            "fci_l1c_fdhsi",
            "fci_l1_geoobs",
            "glm_l2",
            "grib",
            "li_l2",
            "seviri_l1b_hrit",
            "seviri_l1b_native",
            "seviri_l1b_nc",
            "seviri_l2_bufr",
        ],
        # Filters for what datasets not to include
        "exclude_datasets": {
            # 'calibration': ['radiance', 'counts'],
        },
        # Reader-specific reading configuration
        "seviri_l1b_hrit": {
            # If group_keys is not specified it defaults to Satpy's configuration
            "group_keys": ["start_time", "platform_shortname", "service"],
            # Offered patterns to filter files (trollsift syntax), by default first entry is used.
            "filter_patterns": [
                "{rate:1s}-000-{hrit_format:_<6s}-{platform_shortname:4s}_{service:_<7s}-"
                "{channel:_<6s}___-{segment:_<6s}___-{start_time:%Y%m%d%H%M}-{c:1s}_",
                "{rate:1s}-000-{hrit_format:_<6s}-{platform_shortname:4s}_{service:_<7s}-"
                "{channel:_<6s}___-{segment:_<6s}___-{start_time:%Y%m%d%H%M}-__",
            ],
        },
    },
    # Reader-specific mapping configuration
    "data_mapping": {
        "seviri_l1b_hrit": {
            "iScansNegatively": True,
            "jScansPositively": True,
        },
    },
    # specific to the open file wizard dialog
    "open_file_wizard": {
        "default_reader": "seviri_l1b_hrit",  # use 'None' for first in list
        "id_components": [
            "name",
            "wavelength",
            "resolution",
            "calibration",
            "level",
        ],
    },
    # display options
    "display": {"use_tiled_geolocated_images": True},
}

config = Config("uwsift", defaults=[DEFAULT_CONFIGURATION], paths=CONFIG_PATHS)


def overwrite_import(package_name: str, custom_import_path: str, *, verbose=True):
    if (custom_import_path is not None) and (not os.path.exists(custom_import_path)):
        raise FileNotFoundError(
            f"Package '{package_name}' " f"doesn't exist at given custom import path '{custom_import_path}'"
        )

    class CustomPathFinder(PathFinder):
        @classmethod
        def find_spec(cls, fullname: str, path=None, target=None):
            package_parts = fullname.split(".")
            if package_parts[0] == package_name and custom_import_path is not None:
                del package_parts[0]

                if len(package_parts) == 0:
                    # import the root of the package
                    package_import_path = os.path.join(custom_import_path, "__init__.py")
                    spec = spec_from_file_location(fullname, package_import_path)
                else:
                    # import the base directory of the subpackage
                    package_parts.pop()
                    package_import_path = os.path.join(custom_import_path, *package_parts)
                    spec = super().find_spec(fullname, [package_import_path], target)

                if verbose:
                    # setup for the logger happens in __main__, thus use print
                    print(f"Custom import path for package `{fullname}`: {package_import_path}")
                return spec
            return None

    # The CustomPathFinder must have high priority, therefore set it to the front of the list.
    # This is harmless, because we modify only the import path of the Satpy module.
    sys.meta_path.insert(0, CustomPathFinder)


satpy_import_path = config.get("satpy_import_path", None)
if satpy_import_path is not None:
    overwrite_import("satpy", satpy_import_path)

satpy_extra_readers_import_path = config.get("satpy_extra_readers_import_path", None)
if satpy_extra_readers_import_path is not None:
    # See https://gitlab.eumetsat.int/Meraner/fci_l1_geoobs_satpy_reader/-/blob/master/README.md
    sys.path.insert(0, satpy_extra_readers_import_path)
    os.environ["SATPY_CONFIG_PATH"] = satpy_extra_readers_import_path


USE_TILED_GEOLOCATED_IMAGES = config.get("display.use_tiled_geolocated_images")

USE_INVENTORY_DB = config.get("storage.use_inventory_db")
CLEANUP_FILE_CACHE = config.get("storage.cleanup_file_cache")

AUTO_UPDATE_MODE__ACTIVE = config.get("auto_update.active", False)
