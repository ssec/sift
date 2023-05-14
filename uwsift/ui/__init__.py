#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
.py
~~~

PURPOSE


REFERENCES


REQUIRES


:author: R.K.Garcia <rayg@ssec.wisc.edu>
:copyright: 2014 by University of Wisconsin Regents, see AUTHORS for more details
:license: GPLv3, see LICENSE for more details
"""
__author__ = "rayg"
__docformat__ = "reStructuredText"

import argparse
import logging
import sys
import unittest
from pathlib import Path

# Expose path to qml files.
QML_PATH = Path(__file__).parent.absolute()

LOG = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="PURPOSE", epilog="", fromfile_prefix_chars="@")
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbosity",
        action="count",
        default=0,
        help="each occurrence increases verbosity 1 level through ERROR-WARNING-Info-DEBUG",
    )
    # http://docs.python.org/2.7/library/argparse.html#nargs
    # parser.add_argument('--stuff', nargs='5', dest='my_stuff',
    #                    help="one or more random things")
    parser.add_argument("pos_args", nargs="*", help="positional arguments don't have the '-' prefix")
    args = parser.parse_args()

    levels = [logging.ERROR, logging.WARN, logging.INFO, logging.DEBUG]
    logging.basicConfig(level=levels[min(3, args.verbosity)])

    if not args.pos_args:
        unittest.main()
        return 0

    for pn in args.pos_args:
        pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
