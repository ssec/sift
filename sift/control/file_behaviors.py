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
__author__ = 'rayg'
__docformat__ = 'reStructuredText'

import argparse
import logging
import sys
import unittest

from PyQt5.QtCore import QObject

# from PyQt4.QtGui import QAction


FORMAT_GUIDEBOOK = {}


class UserAddsFileToDoc(QObject):
    """
    Manage an open dialog and queue up adding files to the document
    """
    _main = None
    _open_dialog = None
    _task_queue = None

    def __init__(self, main_window, open_dialog, task_queue):
        super(UserAddsFileToDoc, self).__init__()
        self._main = main_window
        self._task_queue = task_queue
        self._open_dialog = open_dialog

    def open_files(self):
        """
        activate dialog
        do any previewing appropriate
        when ready, add file to document
        load coarse representation immediately
        queue up background refinement
        :return:
        """
        pass

    def __call__(self, files_to_open):
        """
        given information from dialog box,
        for each file
            identify file type using guidebook

        :param files_to_open:
        :return:
        """
        pass


LOG = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="PURPOSE",
        epilog="",
        fromfile_prefix_chars='@')
    parser.add_argument('-v', '--verbose', dest='verbosity', action="count", default=0,
                        help='each occurrence increases verbosity 1 level through ERROR-WARNING-Info-DEBUG')
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
