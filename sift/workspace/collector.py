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
from .goesr_pug import PugL1bTools

LOG = logging.getLogger(__name__)

class aHunter(ABC):
    """
    Hunter scans one or locations for metadata
    """
    _DB: Metadatabase = None

    def __init__(self, db: Metadatabase):
        self._DB = db

    def product_for(self, realpath:str, fullname:str):
        """
        check database to see if a given path + product name
        Args:
            realpath:
            fullname:

        Returns:

        """

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


class GoesRHunter(aHunter):

    def get_metadata(self, dest_uuid, source_path=None, source_uri=None, cache_path=None, **kwargs):

        d = {}
        # nc = nc4.Dataset(source_path)
        pug = PugL1bTools(source_path)

        d.update(self._metadata_for_abi_path(pug))
        d[INFO.UUID] = dest_uuid
        d[INFO.DATASET_NAME] = os.path.split(source_path)[-1]
        d[INFO.PATHNAME] = source_path
        d[INFO.KIND] = KIND.IMAGE

        d[INFO.PROJ] = pug.proj4_string
        # get nadir-meter-ish projection coordinate vectors to be used by proj4
        y,x = pug.proj_y, pug.proj_x
        d[INFO.ORIGIN_X] = x[0]
        d[INFO.ORIGIN_Y] = y[0]

        midyi, midxi = int(y.shape[0] / 2), int(x.shape[0] / 2)
        # PUG states radiance at index [0,0] extends between coordinates [0,0] to [1,1] on a quadrille
        # centers of pixels are therefore at +0.5, +0.5
        # for a (e.g.) H x W image this means [H/2,W/2] coordinates are image center
        # for now assume all scenes are even-dimensioned (e.g. 5424x5424)
        # given that coordinates are evenly spaced in angular -> nadir-meters space,
        # technically this should work with any two neighbor values
        d[INFO.CELL_WIDTH] = x[midxi+1] - x[midxi]
        d[INFO.CELL_HEIGHT] = y[midyi+1] - y[midyi]

        shape = pug.shape
        d[INFO.SHAPE] = shape
        generate_guidebook_metadata(d)
        LOG.debug(repr(d))
        return d




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
