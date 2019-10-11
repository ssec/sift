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
    'open_file_wizard': {
        'readers': None,  # all readers
        'id_components': [
            'name',
            'wavelength',
            'calibration',
            'level',
        ],
        'exclude_datasets': {
            'calibration': ['radiance', 'counts'],
        }
    }
}

config = Config('uwsift', defaults=[DEFAULT_CONFIGURATION], paths=CONFIG_PATHS)