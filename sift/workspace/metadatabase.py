#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
metadatabase.py
===============

PURPOSE
Manage SQLAlchemy database of information used by DataMatrix
Used by DataMatrix, which is in turn used by Workspace


REFERENCES


REQUIRES
SQLAlchemy with SQLite

:author: R.K.Garcia <rayg@ssec.wisc.edu>
:copyright: 2016 by University of Wisconsin Regents, see AUTHORS for more details
:license: GPLv3, see LICENSE for more details
"""
__author__ = 'rayg'
__docformat__ = 'reStructuredText'

import os, sys
import logging, unittest, argparse

from sqlalchemy import Column, Integer, String, UnicodeText, Unicode, ForeignKey, DateTime, Interval, PickleType, PrimaryKeyConstraint
from sqlalchemy.orm import Session, relationship
from sqlalchemy.ext.declarative import declarative_base


LOG = logging.getLogger(__name__)


# =================
# Database Entities

Base = declarative_base()


class File(Base):
    """
    held metadata regarding a file that we can access and import data into the workspace from
    """
    __tablename__ = 'files'
    # identity information
    id = Column(Integer, primary_key=True)

    # primary handler
    format = Column(PickleType)  # class or callable which can pull this data into workspace from storage

    filename = Column(Unicode)  # basename, no path separators
    dirname = Column(Unicode)  # directory, no trailing separator
    mtime = Column(DateTime)  # last observed mtime of the file, for change checking
    atime = Column(DateTime)  # last time this file was accessed by application

    products = relationship("StoredProduct", backref="file")

    @property
    def path(self):
        return os.path.join(self.dirname, self.filename)

    @property
    def is_orphan(self):
        return not os.path.isfile(self.path)

    # def touch(self, session=None):
    #     ismine, session = (False, session) if session is not None else
    #     self.atime = datetime.utcnow()


class StoredProduct(Base):
    """
    Primary entity being tracked in metadatabase
    One or more StoredProduct are held in a single File
    A StoredProduct has zero or more CachedData representations, potentially at different projections
    A StoredProduct has zero or more KeyValue pairs with additional metadata
    A File's format allows data to be imported to the workspace
    A StoredProduct's kind determines how its cached data is transformed to different representations for display
    """
    __tablename__ = 'products'

    # identity information
    id = Column(Integer, primary_key=True)
    file_id = Column(Integer, ForeignKey('files.id'))

    # primary handler
    kind = Column(PickleType)  # class or callable which can perform transformations on this data in workspace

    # cached metadata provided by the file format handler
    platform = Column(String)  # platform or satellite name e.g. "GOES-16", "Himawari-8"
    identifier = Column(String)  # product identifier eg "B01", "B02"
    # times
    timeline = Column(DateTime)  # normalized instantaneous scheduled observation time e.g. 20170122T2310
    start = Column(DateTime, nullable=True)  # actual observation time start
    duration = Column(Interval, nullable=True)  # actual observation duration
    # geometry
    resolution = Column(Integer, nullable=True)  # meters max resolution, e.g. 500, 1000, 2000, 4000
    proj4 = Column(String, nullable=True)  # native projection of the data in its original form, if one exists
    # descriptive
    units = Column(Unicode, nullable=True)  # udunits compliant units, e.g. 'K'
    label = Column(Unicode, nullable=True)  # "AHI Refl B11"
    description = Column(UnicodeText, nullable=True)

    # link to workspace cache
    uuid = Column(PickleType, nullable=True)  # UUID object representing this data in SIFT, or None if not in cache
    cache = relationship("CachedData", backref="product")

    # link to key-value further information
    info = relationship("KeyValue", backref="product")


class KeyValue(Base):
    """
    key-value pairs associated with a product
    """
    __tablename__ = 'keyvalues'
    id = Column(Integer, primary_key=True)
    product_id = Column(Integer, ForeignKey('products.id'))
    key = Column(String)
    value = Column(PickleType)


class CachedData(Base):
    """
    represent flattened products in cache
     a given product may have several CachedData for different projections
    """
    __tablename__ = 'workspace_content'
    id = Column(Integer, primary_key=True)
    product_id = Column(Integer, ForeignKey('products.id'))

    # memory mappable read-only data files loaded from native format
    # data_type = Column(String)  # almost always float32; can be int16 in the future
    data_rows, data_cols, data_levels = Column(Integer), Column(Integer), Column(Integer, nullable=True)
    data_path = Column(String, unique=True)  # relative to workspace, binary array of data
    proj4 = Column(String, nullable=True)  # proj4 projection string for the data in this array, if one exists; else assume y=lat/x=lon
    y_path = Column(String, nullable=True)  # if needed, y location cache path relative to workspace
    x_path = Column(String, nullable=True)  # if needed, x location cache path relative to workspace
    z_path = Column(String, nullable=True)  # if needed, x location cache path relative to workspace

    # sparsity and coverage arrays are int8
    sparsity_rows, sparsity_cols = Column(Integer, nullable=True), Column(Integer, nullable=True)
    sparsity_path = Column(String, unique=True, nullable=True)  # availability array being broadcast across data, 0==unavailable, default 1s
    coverage_rows, coverage_cols = Column(Integer, nullable=True), Column(Integer, nullable=True)
    coverage_path = Column(String, unique=True, nullable=True)  # availability array being stretched across data, 0==unavailable, default 1s
    metadata_path = Column(String, unique=True, nullable=True)  # json metadata path, optional

    mtime = Column(DateTime)  # last observed mtime of the file, for change checking
    atime = Column(DateTime)  # last time this file was accessed by application

    @property
    def uuid(self):
        return self.product.uuid


# ============================
# support and testing routines

class tests(unittest.TestCase):
    data_file = os.environ.get('TEST_DATA', os.path.expanduser("~/Data/test_files/thing.dat"))

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
    parser = argparse.ArgumentParser(
        description="PURPOSE",
        epilog="",
        fromfile_prefix_chars='@')
    parser.add_argument('-v', '--verbose', dest='verbosity', action="count", default=0,
                        help='each occurrence increases verbosity 1 level through ERROR-WARNING-INFO-DEBUG')
    parser.add_argument('-d', '--debug', dest='debug', action='store_true',
                        help="enable interactive PDB debugger on exception")
    # http://docs.python.org/2.7/library/argparse.html#nargs
    # parser.add_argument('--stuff', nargs='5', dest='my_stuff',
    #                    help="one or more random things")
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
