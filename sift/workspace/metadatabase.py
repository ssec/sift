#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
metadatabase.py
===============

PURPOSE
Manage SQLAlchemy database of information used to manage workspace
Used by DataMatrix as well


OVERVIEW

The workspace caches content, which represents products, the native form of which resides in a file

File : a file somewhere in the filesystem
 |_ StoredProduct* : product stored in a file
     |_ Content* : workspace cache content corresponding to a product
     |   |_ ContentKeyValue* : additional information on content
     |_ ProductKeyValue* : additional information on product


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

from sqlalchemy import Column, Integer, String, UnicodeText, Unicode, ForeignKey, DateTime, Interval, PickleType, create_engine
from sqlalchemy.orm import Session, relationship, sessionmaker
from sqlalchemy.ext.declarative import declarative_base


LOG = logging.getLogger(__name__)


# =================
# Database Entities

Base = declarative_base()


class Source(Base):
    """
    held metadata regarding a file that we can access and import data into the workspace from
    """
    __tablename__ = 'sources'
    # identity information
    id = Column(Integer, primary_key=True)

    # primary handler
    format = Column(PickleType)  # class or callable which can pull this data into workspace from storage

    # {scheme}://{path}/{name}?{query}, default is just an absolute path in filesystem
    scheme = Column(Unicode, nullable=True)  # uri scheme for the content (the part left of ://), assume file:// by default
    path = Column(Unicode)  # directory or path component not including name
    name = Column(Unicode)  # the name of the file or resource, incl suffix but not query
    query = Column(Unicode, nullable=True)  # query portion of a URI or URL, e.g. 'interval=1m&stride=2'
    mtime = Column(DateTime)  # last observed mtime of the file, for change checking
    atime = Column(DateTime)  # last time this file was accessed by application

    products = relationship("Product", backref="source")

    # def touch(self, session=None):
    #     ismine, session = (False, session) if session is not None else
    #     self.atime = datetime.utcnow()


class Product(Base):
    """
    Primary entity being tracked in metadatabase
    One or more StoredProduct are held in a single File
    A StoredProduct has zero or more Content representations, potentially at different projections
    A StoredProduct has zero or more ProductKeyValue pairs with additional metadata
    A File's format allows data to be imported to the workspace
    A StoredProduct's kind determines how its cached data is transformed to different representations for display
    """
    __tablename__ = 'products'

    # identity information
    id = Column(Integer, primary_key=True)
    file_id = Column(Integer, ForeignKey('sources.id'))

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
    cache = relationship("Content", backref="product")

    # link to key-value further information
    info = relationship("ProductKeyValue", backref="product")


class ProductKeyValue(Base):
    """
    key-value pairs associated with a product
    """
    __tablename__ = 'product_key_values'
    id = Column(Integer, primary_key=True)
    product_id = Column(Integer, ForeignKey('products.id'))
    key = Column(String)
    value = Column(PickleType)


class Content(Base):
    """
    represent flattened product data files in cache (i.e. cache content)
    typically memory-map ready data (np.memmap)
    basic correspondence to projection/geolocation information may accompany
    images will typically have rows>0 cols>0 levels=None (implied levels=1)
    profiles may have rows>0 cols=None (implied cols=1) levels>0
    a given product may have several Content for different projections
    """
    __tablename__ = 'contents'
    id = Column(Integer, primary_key=True)
    product_id = Column(Integer, ForeignKey('products.id'))

    # memory mappable read-only data files loaded from native format
    data_rows, data_cols, data_levels = Column(Integer), Column(Integer, nullable=True), Column(Integer, nullable=True)
    data_type = Column(String, nullable=True)  # default float32; can be int16 in the future for scaled integer images for instance; should be a numpy type name
    data_coeffs = Column(PickleType, nullable=True)  # numpy array with polynomial coefficients for transforming native data to natural units (e.g. for scaled integers), c[0] + c[1]*x + c[2]*x**2 ...
    data_path = Column(String, unique=True)  # relative to workspace, binary array of data
    y_path = Column(String, nullable=True)  # if needed, y location cache path relative to workspace
    x_path = Column(String, nullable=True)  # if needed, x location cache path relative to workspace
    z_path = Column(String, nullable=True)  # if needed, x location cache path relative to workspace
    proj4 = Column(String, nullable=True)  # proj4 projection string for the data in this array, if one exists; else assume y=lat/x=lon

    # sparsity and coverage arrays are int8, must be factors of data_rows and data_cols
    sparsity_rows, sparsity_cols = Column(Integer, nullable=True), Column(Integer, nullable=True)
    sparsity_path = Column(String, unique=True, nullable=True)  # availability array being _broadcast_ across data, 0==unavailable, if not provided assume 1
    coverage_rows, coverage_cols = Column(Integer, nullable=True), Column(Integer, nullable=True)
    coverage_path = Column(String, unique=True, nullable=True)  # availability array being _stretched_ across data, 0==unavailable, if not provided assume 1
    metadata_path = Column(String, unique=True, nullable=True)  # json metadata path, optional

    mtime = Column(DateTime)  # last observed mtime of the file, for change checking
    atime = Column(DateTime)  # last time this product was accessed by application

    # link to key-value further information; primarily a hedge in case specific information has to be squirreled away for later consideration for main content table
    info = relationship("ContentKeyValue", backref="content")

    @property
    def uuid(self):
        return self.product.uuid


class ContentKeyValue(Base):
    """
    key-value pairs associated with a product
    """
    __tablename__ = 'content_key_values'
    id = Column(Integer, primary_key=True)
    product_id = Column(Integer, ForeignKey('contents.id'))
    key = Column(String)
    value = Column(PickleType)


# singleton instance
_MDB = None


class Metadatabase(object):
    """
    singleton interface to application metadatabase
    """
    engine = None
    connection = None
    session_factory = None

    def __init__(self, uri=None, **kwargs):
        global _MDB
        if _MDB is not None:
            raise AssertionError('Metadatabase is a singleton and already exists')
        if uri:
            self.connect(uri, **kwargs)

    @staticmethod
    def instance(*args):
        global _MDB
        if _MDB is None:
            _MDB = Metadatabase(*args)
        return _MDB

    def connect(self, uri, **kwargs):
        assert(self.engine is None)
        assert(self.connection is None)
        self.engine = create_engine(uri, **kwargs)
        self.connection = self.engine.connect()

    def create_tables(self):
        Base.metadata.create_all(self.engine)

    def session(self):
        if self.session_factory is None:
            self.session_factory = sessionmaker(bind=self.engine)
        return self.session_factory()



# ============================
# support and testing routines

class tests(unittest.TestCase):
    # data_file = os.environ.get('TEST_DATA', os.path.expanduser("~/Data/test_files/thing.dat"))
    mdb = None

    def setUp(self):
        pass

    def test_insert(self):
        from datetime import datetime
        mdb = Metadatabase.instance('sqlite://')
        mdb.create_tables()
        s = mdb.session()
        when = datetime.utcnow()
        f = Source(uri=u'file://foo.bar', mtime=when, atime=when, format=None)
        s.add(f)
        s.commit()

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
