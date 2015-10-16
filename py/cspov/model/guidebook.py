#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
guidebook.py
~~~~~~~~~~~~

PURPOSE
This module is the "scientific expert knowledge" that is consulted.



:author: R.K.Garcia <rayg@ssec.wisc.edu>
:copyright: 2014 by University of Wisconsin Regents, see AUTHORS for more details
:license: GPLv3, see LICENSE for more details
"""
__author__ = 'rayg'
__docformat__ = 'reStructuredText'

import os, sys, re
import logging, unittest, argparse
from enum import Enum
from cspov.common import INFO, KIND

LOG = logging.getLogger(__name__)

class INSTRUMENT(Enum):
    UNKNOWN = 0
    AHI = 1
    ABI = 2
    AMI = 3

GUIDEBOOKS = {}


class Guidebook(object):
    """
    guidebook which knows about AHI, ABI, AMI bands, timing, file naming conventions
    """
    @staticmethod
    def is_relevant(pathname):
        return False

    @staticmethod
    def for_info(info=None, path=None):
        """
        given an info dictionary, figure out which
        :param info:
        :return:
        """
        if info and not path:
            path = info[INFO.PATHNAME]


class AHI_HSF_Guidebook(Guidebook):
    @staticmethod
    def is_relevant(pathname):
        return True if re.match(r'HS_H\d\d_\d{8}_\d{4}_B\d\d.*', os.path.split(pathname)[1]) else False





def main():
    parser = argparse.ArgumentParser(
        description="PURPOSE",
        epilog="",
        fromfile_prefix_chars='@')
    parser.add_argument('-v', '--verbose', dest='verbosity', action="count", default=0,
                        help='each occurrence increases verbosity 1 level through ERROR-WARNING-INFO-DEBUG')
    # http://docs.python.org/2.7/library/argparse.html#nargs
    # parser.add_argument('--stuff', nargs='5', dest='my_stuff',
    #                    help="one or more random things")
    parser.add_argument('pos_args', nargs='*',
                        help="positional arguments don't have the '-' prefix")
    args = parser.parse_args()

    levels = [logging.ERROR, logging.WARN, logging.INFO, logging.DEBUG]
    logging.basicConfig(level=levels[min(3, args.verbosity)])

    if not args.pos_args:
        unittest.main()
        return 0

    for pn in args.pos_args:
        pass

    return 0


if __name__ == '__main__':
    sys.exit(main())
