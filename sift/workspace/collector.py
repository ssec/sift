#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PURPOSE
Collector is a zookeeper of products, which populates and revises the workspace metadatabase
 Collector uses Hunters to find individual formats/conventions/products
 Products live in Resources (typically files)
 Collector skims files without reading data
 Collector populates the metadatabase with information about available products
 More than one Product may be in a Resource

 Collector also knows which Importer can bring Content from the Resource into the Workspace

REFERENCES

REQUIRES

:author: R.K.Garcia <rkgarcia@wisc.edu>
:copyright: 2017 by University of Wisconsin Regents, see AUTHORS for more details
:license: GPLv3, see LICENSE for more details
"""
import os, sys
import logging, unittest
from abc import ABC, abstractmethod, abstractproperty
from typing import Union, Set, List, Iterable
from .metadatabase import Product, Metadatabase, Resource
from sqlalchemy.orm import Session

LOG = logging.getLogger(__name__)

class aHunter(ABC):
    """
    Hunter scans one or locations for metadata
    """
    _DB: Metadatabase = None

    def __init__(self, db: Metadatabase):
        self._DB = db

    @abstractmethod
    def hunt_dir(self, S: Session, dir_path:str) -> int:
        return 0

    @abstractmethod
    def hunt_file(self, S: Session, file_path:str) -> int:
        return 0

    def hunt_seq(self, S: Session, seq: Iterable[str]) -> int:
        found = 0
        for path in seq:
            if os.path.isfile(path):
                found += self.hunt_file(S, path)
            elif os.path.isdir(path):
                found += self.hunt_dir(S, path)
            else:
                raise ValueError('Unknown content: {}'.format(path))
        return found

    def hunt(self, directory_glob_path: Union[str, Iterable[str]], recurse_levels: int = 0) -> int:
        """
        scan a file or a directory for metadata, and update the metadatabase
        Args:
            directory_or_file_path:
            recurse_levels:

        Returns:
            count (int): number of products found
        """
        S = self._DB.session()
        if isinstance(directory_glob_path, str):
            if os.path.isfile(directory_glob_path):
                found = self.hunt_file(S, directory_glob_path)
            elif os.path.isdir(directory_glob_path):
                found = self.hunt_dir(S, directory_glob_path)
        else:
            found = self.hunt_seq(S, directory_glob_path)
        if found:
            S.commit()





PATH_TEST_DATA = os.environ.get('TEST_DATA', os.path.expanduser("~/Data/test_files/thing.dat"))

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
