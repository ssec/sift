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

DEFAULT_CONFIGURATION = {
    # related to any reading of data
    'data_reading': {
        'readers': None,  # None=all readers
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