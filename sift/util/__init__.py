#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import logging
from sift.util.default_paths import (WORKSPACE_DB_DIR,
                                     DOCUMENT_SETTINGS_DIR,
                                     USER_DESKTOP_DIRECTORY)

LOG = logging.getLogger(__name__)
IS_FROZEN = getattr(sys, 'frozen', False)
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


def check_imageio_deps():
    if IS_FROZEN and not os.getenv('IMAGEIO_FFMPEG_EXE'):
        ffmpeg_exe = os.path.realpath(os.path.join(SCRIPT_DIR, '..', '..', 'ffmpeg'))
        LOG.debug("Setting ffmpeg location to %s", ffmpeg_exe)
        os.environ['IMAGEIO_FFMPEG_EXE'] = ffmpeg_exe


def check_grib_definition_dir():
    # patch GRIB API C library when frozen
    if IS_FROZEN and not os.getenv('GRIB_DEFINITION_PATH'):
        grib_path = os.path.realpath(os.path.join(SCRIPT_DIR, '..', '..', 'share', 'grib_api', 'definitions'))
        LOG.debug("Setting GRIB definition path to %s", grib_path)
        os.environ['GRIB_DEFINITION_PATH'] = grib_path


def get_package_data_dir():
    """Return location of the package 'data' directory.

    When frozen the data directory is placed in 'sift_data' of the root
    package directory.
    """
    if IS_FROZEN:
        return os.path.realpath(os.path.join(SCRIPT_DIR, "..", "..", "sift_data"))
    else:
        return os.path.realpath(os.path.join(SCRIPT_DIR, "..", "data"))


