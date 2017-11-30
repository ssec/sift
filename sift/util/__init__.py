#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
from sift.util.default_paths import (WORKSPACE_DB_DIR,
                                     DOCUMENT_SETTINGS_DIR,
                                     USER_DESKTOP_DIRECTORY)


def get_package_data_dir():
    """Return location of the package 'data' directory.

    When frozen the data directory is placed in 'sift_data' of the root
    package directory.
    """
    SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

    if getattr(sys, 'frozen', False):
        return os.path.realpath(os.path.join(SCRIPT_DIR, "..", "..", "sift_data"))
    else:
        return os.path.realpath(os.path.join(SCRIPT_DIR, "..", "data"))


