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

