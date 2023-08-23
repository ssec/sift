#!/usr/bin/env python

import os
import sys
import warnings
from importlib.machinery import PathFinder
from importlib.util import spec_from_file_location

import satpy
from donfig import Config

from uwsift.common import ImageDisplayMode
from uwsift.util import get_base_dir

from .util.default_paths import APPLICATION_NAME, USER_CONFIG_DIR
from .version import __version__  # noqa

SYSTEM_CONFIG_DIR = os.path.join(get_base_dir(), "etc", APPLICATION_NAME)

__CONFIG = "config"
__READERS = "readers"

SYSTEM_CONFIG_PATHS = [
    os.path.join(SYSTEM_CONFIG_DIR, __CONFIG),
    os.path.join(SYSTEM_CONFIG_DIR, __CONFIG, __READERS),
]

USER_CONFIG_PATHS = [
    os.path.join(USER_CONFIG_DIR, __CONFIG),
    os.path.join(USER_CONFIG_DIR, __CONFIG, __READERS),
]

# Configurations read later by Donfig Config() overwrite previous settings, thus user config comes last:
CONFIG_PATHS = SYSTEM_CONFIG_PATHS + USER_CONFIG_PATHS


def init_user_config_dirs(user_config_dirs: list):
    print(f"Creating user configuration directories:\n\t{user_config_dirs}", file=sys.stderr)
    for user_config_dir in user_config_dirs:
        try:
            os.makedirs(user_config_dir, mode=0o777, exist_ok=True)
        except (PermissionError, FileExistsError) as e:
            print(f"Failed to create '{user_config_dir}': {e}", file=sys.stderr)


# FIXME: Move this to GUI initialization (argument parsing)
init_user_config_dirs(USER_CONFIG_PATHS)

print(f"Reading configuration from:\n\t{CONFIG_PATHS}", file=sys.stderr)
deprecations = {"satpy_extra_readers_import_path": "satpy_extra_config_path"}
config = Config("uwsift", paths=CONFIG_PATHS, deprecations=deprecations)


def overwrite_import(package_name: str, custom_import_path: str, *, verbose=True):
    if (custom_import_path is not None) and (not os.path.exists(custom_import_path)):
        raise FileNotFoundError(
            f"Package '{package_name}' doesn't exist at given custom import path '{custom_import_path}'"
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
                    print(f"Custom import path for package `{fullname}`: {package_import_path}", file=sys.stderr)
                return spec
            return None

    # The CustomPathFinder must have high priority, therefore set it to the front of the list.
    # This is harmless, because we modify only the import path of the Satpy module.
    sys.meta_path.insert(0, CustomPathFinder)


# Add additional satpy configuration paths
satpy_config_path_yml = config.get("satpy_extra_config_path", None)
if satpy_config_path_yml is not None:
    satpy.config.set(config_path=satpy.config.get("config_path") + [satpy_config_path_yml])
    sys.path.insert(0, satpy_config_path_yml)


def _map_str_to_image_display_mode(image_display_mode_str: str) -> ImageDisplayMode:
    for idm in ImageDisplayMode:
        if image_display_mode_str.lower() == idm:
            return idm
    if image_display_mode_str:  # empty is OK, but typos should not go unnoticed.
        warnings.warn(
            f"Unknown image display mode '{image_display_mode_str}', falling back to the default.",
            UserWarning,
            stacklevel=2,
        )
    return ImageDisplayMode.SIMPLE_GEOLOCATED  # this is the default


# FIXME: These defeat the purpose of a dynamic config object and prevents command-line overrides.
#    These can be converted to functions or direct `config.get(X)` usage.
IMAGE_DISPLAY_MODE = _map_str_to_image_display_mode(config.get("display.image_mode", ""))
print(f"Image Display Mode: {IMAGE_DISPLAY_MODE}", file=sys.stderr)

USE_INVENTORY_DB = config.get("storage.use_inventory_db")
CLEANUP_FILE_CACHE = config.get("storage.cleanup_file_cache")

AUTO_UPDATE_MODE__ACTIVE = config.get("auto_update.active", False)
