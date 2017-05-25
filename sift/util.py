#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Utility functions and classes
"""
import os
import sys


def get_data_dir():
    """Return location of the package 'data' directory.
    
    When frozen the data directory is placed in 'sift_data' of the root
    package directory.
    """
    SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

    if getattr(sys, 'frozen', False):
        return os.path.realpath(os.path.join(SCRIPT_DIR, "..", "sift_data"))
    else:
        return os.path.realpath(os.path.join(SCRIPT_DIR, "data"))
