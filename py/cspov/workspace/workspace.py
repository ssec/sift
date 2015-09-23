#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
.py
~~~

PURPOSE
Implement Workspace, a singleton object which manages large amounts of data
- background loading, up to and including reprojection
- providing memory-compatible, stride-able arrays
- accepting data from external sources written in arbitrary languages

REFERENCES


REQUIRES


:author: R.K.Garcia <rayg@ssec.wisc.edu>
:copyright: 2014 by University of Wisconsin Regents, see AUTHORS for more details
:license: GPLv3, see LICENSE for more details
"""
__author__ = 'rayg'
__docformat__ = 'reStructuredText'

import os, sys
import logging, unittest, argparse
from PyQt4.QtCore import QObject, pyqtSignal


LOG = logging.getLogger(__name__)

class Workspace(QObject):
    """
    Workspace is a singleton object which works with Datasets shall:
    - own a working directory full of recently used datasets
    - provide DatasetInfo dictionaries for shorthand use between application subsystems
    - unpack data in "packing crate" formats like NetCDF into memory-compatible flat files
    - efficiently create on-demand subsections and strides of raster data as numpy arrays
    - incrementally cache often-used subsections and strides ("image pyramid") using appropriate tools like gdal
    - notify subscribers of changes to datasets (Qt signal/slot pub-sub)
    - during idle, clean out unused/idle data content, given DatasetInfo contents provides enough metadata to recreate
    - interface to external data processing or loading plug-ins and notify application of new-dataset-in-workspace
    """
    cwd = None  # directory we work in
    _own_cwd = None  # whether or not we created the cwd - which is also whether or not we're allowed to destroy it

    # signals
    didStartImport = pyqtSignal(dict)  # a dataset started importing; generated after overview level of detail is available
    didImportLevelOfDetail = pyqtSignal(dict)  # partial completion of a dataset import
    didFinishImport = pyqtSignal(dict)  # all loading activities for a dataset have completed
    didDiscoverExternalDataset = pyqtSignal(dict)  # a new dataset was added to the workspace from an external agent


    def __init__(self, directory_path=None):
        """
        Initialize a new or attach an existing workspace, creating any necessary bookkeeping.
        """
        self.cwd = directory_path = os.path.abspath(directory_path)
        if not os.path.isdir(directory_path):
            os.makedirs(directory_path)
            self._own_cwd = True
        else:
            self._own_cwd = False


    def idle(self):
        """
        Called periodically when application is idle. Does a clean-up tasks and returns True if more needs to be done later.
        Time constrained to ~0.1s.
        :return: True/False, whether or not more clean-up needs to be scheduled.
        """
        return False

    def import_uri(self, uri):
        """
        Start loading URI data into the workspace asynchronously.
        When enough of the data is available to produce and overview,
        return a DatasetInfo dictionary which can be used by client as a token to grab data.
        :param uri:
        :return:
        """

    def import_file(self, pathname):
        """
        Start loading URI data into the workspace asynchronously.

        :param pathname:
        :return:
        """
        return self.attach_uri('file://' + pathname)

    def expel(self, dsi):
        """
        Formally detach a dataset, removing its content from the workspace fully by the time that idle() has nothing more to do.
        :param dsi: datasetinfo dictionary
        :return: None
        """

    def __getitem__(self, datasetinfo):
        """
        return a dataset or dataset proxy capable of generating a numpy array when sliced
        :param datasetinfo: metadata on the dataset
        :return: sliceable object returning numpy arrays
        """



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
