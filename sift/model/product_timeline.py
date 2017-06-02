#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PURPOSE

REFERENCES

REQUIRES

:author: R.K.Garcia <rkgarcia@wisc.edu>
:copyright: 2017 by University of Wisconsin Regents, see AUTHORS for more details
:license: GPLv3, see LICENSE for more details
"""
import os, sys
import logging, unittest
from PyQt4.QtCore import QObject
from collections import namedtuple, MutableSequence, OrderedDict
from enum import Enum

LOG = logging.getLogger(__name__)

PATH_TEST_DATA = os.environ.get('TEST_DATA', os.path.expanduser("~/Data/test_files/thing.dat"))

class State(Enum):
    UNKNOWN = "unknown"
    AVAILABLE = "available: can be loaded from native format"
    LOADING = "loading: read in progress"
    READY = "ready: cached in workspace"
    ACTIVE = "active: cached in GPU"

class Frame(QObject):
    _state = State.UNKNOWN
    _uuid = None

    @property
    def uuid(self):
        return self._uuid

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, new_state):
        self._state = new_state



class Product(QObject):
    """
    zero or more Layers in a nonintersecting time sequence
    a Product is a sequence of 0 or more Frames
    """
    def seq(self, filter_set={State.ACTIVE}):
        """
        yield a sequence of frames that are currently
        Returns:

        """




class Stack(MutableSequence, QObject):
    """
    A stack of products ordered in lowest z-order to highest z-order
    """



class tests(unittest.TestCase):
    def setUp(self):
        pass

    def test_something(self):
        pass


def _debug(type, value, tb):
    "enable with sys.excepthook = debug"
    if not sys.stdin.isatty():
        sys.__excepthook__(type, value, tb)
    else:
        import traceback, pdb
        traceback.print_exception(type, value, tb)
        # …then start the debugger in post-mortem mode.
        pdb.post_mortem(tb)  # more “modern”


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="PURPOSE",
        epilog="",
        fromfile_prefix_chars='@')
    parser.add_argument('-v', '--verbose', dest='verbosity', action="count", default=0,
                        help='each occurrence increases verbosity 1 level through ERROR-WARNING-INFO-DEBUG')
    parser.add_argument('-d', '--debug', dest='debug', action='store_true',
                        help="enable interactive PDB debugger on exception")
    parser.add_argument('inputs', nargs='*',
                        help="input files to process")
    args = parser.parse_args()

    levels = [logging.ERROR, logging.WARN, logging.INFO, logging.DEBUG]
    logging.basicConfig(level=levels[min(3, args.verbosity)])

    if args.debug:
        sys.excepthook = _debug

    if not args.inputs:
        unittest.main()
        return 0

    for pn in args.inputs:
        pass

    return 0


if __name__ == '__main__':
    sys.exit(main())
