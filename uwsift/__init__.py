#!/usr/bin/env python

import os
from donfig import Config
from .version import __version__  # noqa
from .util.default_paths import DOCUMENT_SETTINGS_DIR

CONFIG_DIR = os.path.join(DOCUMENT_SETTINGS_DIR, 'config')
os.environ.setdefault('UWSIFT_CONFIG', CONFIG_DIR)
CONFIG_PATHS = [
    CONFIG_DIR,
    os.path.join(os.path.expanduser('~'), '.config', 'uwsift'),
]

# EXPERIMENTAL: This functionality is experimental and may change in future
#     releases until it is an advertised feature.
DEFAULT_CONFIGURATION = {
    # related to any reading of data
    'data_reading': {
        # What readers to use when opening files
        # None => all readers
        # from environment variable: export UWSIFT_DATA_READING__READERS = "['abi_l1b', 'ami_l1b']"
        # 'readers': None,
        'readers': [
            'abi_l1b',
            'ahi_hrit',
            'ahi_hsd',
            'ami_l1b',
            'fci_l1c_fdhsi',
            'glm_l2',
            'grib',
            'li_l2',
            'seviri_l1b_hrit',
            'seviri_l1b_native',
            'seviri_l1b_nc',
            'seviri_l2_bufr'],
        # Filters for what datasets not to include
        'exclude_datasets': {
            'calibration': ['radiance', 'counts'],
        }
    },
    # specific to the open file wizard dialog
    'open_file_wizard': {
        'default_reader': None,  # first in list
        'id_components': [
            'name',
            'wavelength',
            'resolution',
            'calibration',
            'level',
        ],
    }
}

config = Config('uwsift', defaults=[DEFAULT_CONFIGURATION], paths=CONFIG_PATHS)
