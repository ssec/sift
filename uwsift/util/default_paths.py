#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Utility functions and classes

Directory Constants:

 - WORKSPACE_DB_DIR: Default location to place the workspace metadata
   database(s)
 - WORKSPACE_CACHE_DIR: Default workspace cache directory
   Where any raster data may be cached.
 - DOCUMENT_SETTINGS_DIR: Default document user settings/profiles directory
 -
"""
import os
import sys

import appdirs

APPLICATION_AUTHOR = "CIMSS-SSEC"
APPLICATION_DIR = "SIFT"

USER_CACHE_DIR = appdirs.user_cache_dir(APPLICATION_DIR, APPLICATION_AUTHOR)
# Data and config are the same on everything except linux
USER_DATA_DIR = appdirs.user_data_dir(APPLICATION_DIR, APPLICATION_AUTHOR, roaming=True)
USER_CONFIG_DIR = appdirs.user_config_dir(APPLICATION_DIR, APPLICATION_AUTHOR, roaming=True)

WORKSPACE_DB_DIR = os.path.join(USER_CACHE_DIR, 'workspace')
DOCUMENT_SETTINGS_DIR = os.path.join(USER_CONFIG_DIR, 'settings')


# FUTURE: Is there Document data versus Document configuration?


def _desktop_directory():
    try:
        if sys.platform.startswith('win'):
            import appdirs
            # https://msdn.microsoft.com/en-us/library/windows/desktop/bb762494(v=vs.85).aspx
            return appdirs._get_win_folder('CSIDL_DESKTOPDIRECTORY')
        else:
            return os.path.join(os.path.expanduser('~'), 'Desktop')
    except (KeyError, ValueError):
        return os.getcwd()


USER_DESKTOP_DIRECTORY = _desktop_directory()
