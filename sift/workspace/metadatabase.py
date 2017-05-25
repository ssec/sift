#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
metadatabase.py
===============

PURPOSE
SQLAlchemy database tables of metadata used by Workspace to manage its local cache.
Incidentally used by manage adjacency matrix (DataMatrix).


OVERVIEW

The workspace caches content, which represents products, the native form of which resides in a file

File : a file somewhere in the filesystem
 |_ Product* : product stored in a file
     |_ Content* : workspace cache content corresponding to a product
     |   |_ ContentKeyValue* : additional information on content
     |_ ProductKeyValue* : additional information on product

A typical baseline product will have two content: and overview (lod==0) and a native resolution (lod>0)


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
from collections import MutableMapping

from sqlalchemy import Column, Integer, String, UnicodeText, Unicode, ForeignKey, DateTime, Interval, PickleType, Float, create_engine
from sqlalchemy.orm import Session, relationship, sessionmaker, backref
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm.collections import attribute_mapped_collection
from sqlalchemy.ext.associationproxy import association_proxy

LOG = logging.getLogger(__name__)

#
# ref   http://docs.sqlalchemy.org/en/latest/_modules/examples/vertical/dictlike.html
#

class ProxiedDictMixin(object):
    """Adds obj[key] access to a mapped class.

    This class basically proxies dictionary access to an attribute
    called ``_proxied``.  The class which inherits this class
    should have an attribute called ``_proxied`` which points to a dictionary.

    """

    def __len__(self):
        return len(self._proxied)

    def __iter__(self):
        return iter(self._proxied)

    def __getitem__(self, key):
        return self._proxied[key]

    def __contains__(self, key):
        return key in self._proxied

    def __setitem__(self, key, value):
        self._proxied[key] = value

    def __delitem__(self, key):
        del self._proxied[key]




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

    products = relationship("Product", backref=backref("source", cascade="all"))

    @property
    def uri(self):
        return os.path.join(self.path, self.name) if not self.scheme else "{}://{}/{}".format(self.scheme, self.path, self.name)

    # def touch(self, session=None):
    #     ismine, session = (False, session) if session is not None else
    #     self.atime = datetime.utcnow()


class Product(ProxiedDictMixin, Base):
    """
    Primary entity being tracked in metadatabase
    One or more StoredProduct are held in a single File
    A StoredProduct has zero or more Content representations, potentially at different projections
    A StoredProduct has zero or more ProductKeyValue pairs with additional metadata
    A File's format allows data to be imported to the workspace
    A StoredProduct's kind determines how its cached data is transformed to different representations for display
    additional information is stored in a key-value table addressable as product[key:str]
    """
    __tablename__ = 'products'

    # identity information
    id = Column(Integer, primary_key=True)
    source_id = Column(Integer, ForeignKey(Source.id))
    # relationship: .source
    uuid = Column(String, nullable=False, unique=True)  # UUID representing this data in SIFT, or None if not in cache

    # primary handler
    kind = Column(PickleType)  # class or callable which can perform transformations on this data in workspace

    # cached metadata provided by the file format handler
    platform = Column(String)  # platform or satellite name e.g. "GOES-16", "Himawari-8"
    identifier = Column(String)  # product identifier eg "B01", "B02"

    # times
    timeline = Column(DateTime)  # normalized instantaneous scheduled observation time e.g. 20170122T2310
    start = Column(DateTime, nullable=True)  # actual observation time start
    duration = Column(Interval, nullable=True)  # actual observation duration

    # native resolution information - see Content for projection details at different LODs
    resolution = Column(Integer, nullable=True)  # meters max resolution, e.g. 500, 1000, 2000, 4000

    # descriptive
    units = Column(Unicode, nullable=True)  # udunits compliant units, e.g. 'K'
    label = Column(Unicode, nullable=True)  # "AHI Refl B11"
    description = Column(UnicodeText, nullable=True)

    # link to workspace cache files representing this data, not lod=0 is overview
    contents = relationship("Content", backref=backref("product", cascade="all"))

    # link to key-value further information
    # this provides dictionary style access to key-value pairs
    info = relationship("ProductKeyValue", collection_class=attribute_mapped_collection('key'))
    _proxied = association_proxy("info", "value",
                                 creator=lambda key, value: ProductKeyValue(key=key, value=value))


class ProductKeyValue(Base):
    """
    key-value pairs associated with a product
    """
    __tablename__ = 'product_key_values'
    product_id = Column(ForeignKey(Product.id), primary_key=True)
    key = Column(String, primary_key=True)
    # relationship: .product
    value = Column(PickleType)


class Content(ProxiedDictMixin, Base):
    """
    represent flattened product data files in cache (i.e. cache content)
    typically memory-map ready data (np.memmap)
    basic correspondence to projection/geolocation information may accompany
    images will typically have rows>0 cols>0 levels=None (implied levels=1)
    profiles may have rows>0 cols=None (implied cols=1) levels>0
    a given product may have several Content for different projections
    additional information is stored in a key-value table addressable as content[key:str]
    """
    __tablename__ = 'contents'
    id = Column(Integer, primary_key=True)
    product_id = Column(Integer, ForeignKey(Product.id))

    # handle overview versus detailed data
    lod = Column(Integer)  # power of 2 level of detail; 0 for coarse-resolution overview
    resolution = Column(Integer)  # maximum resolution in meters for this representation of the dataset

    # time accounting, used to check if data needs to be re-imported to workspace, or whether data is LRU and can be removed from a crowded workspace
    mtime = Column(DateTime)  # last observed mtime of the original source of this data, for change checking
    atime = Column(DateTime)  # last time this product was accessed by application

    # actual data content
    # NaNs are used to signify missing data; NaNs can include integer category fields in significand; please ref IEEE 754
    path = Column(String, unique=True)  # relative to workspace, binary array of data
    rows, cols, levels = Column(Integer), Column(Integer, nullable=True), Column(Integer, nullable=True)
    dtype = Column(String, nullable=True)  # default float32; can be int16 in the future for scaled integer images for instance; should be a numpy type name
    coeffs = Column(String, nullable=True)  # json for numpy array with polynomial coefficients for transforming native data to natural units (e.g. for scaled integers), c[0] + c[1]*x + c[2]*x**2 ...
    values = Column(String, nullable=True)  # json for optional dict {int:string} lookup table for NaN flag fields (when dtype is float32 or float64) or integer values (when dtype is an int8/16/32/64)

    # projection information for this representation of the data
    proj4 = Column(String, nullable=True)  # proj4 projection string for the data in this array, if one exists; else assume y=lat/x=lon
    cell_width, cell_height, origin_x, origin_y = Column(Float, nullable=True), Column(Float, nullable=True), Column(Float, nullable=True), Column(Float, nullable=True)  # FIXME DJH should explain these

    # sparsity and coverage, int8 arrays if needed to show incremental availability of the data
    # dimensionality is always a reduction factor of rows/cols/levels
    # coverage is stretched across the data array
    #   e.g. for loading data sectioned or blocked across multiple files
    # sparsity is broadcast over the data array
    #   e.g. for incrementally loading sparse data into a dense array
    # a zero value indicates data is not available, nonzero signifies availability
    coverage_rows, coverage_cols, coverage_levels = Column(Integer, nullable=True), Column(Integer, nullable=True), Column(Integer, nullable=True)
    coverage_path = Column(String, nullable=True)
    sparsity_rows, sparsity_cols, sparsity_levels = Column(Integer, nullable=True), Column(Integer, nullable=True), Column(Integer, nullable=True)
    sparsity_path = Column(String, nullable=True)

    # navigation information, if required
    xyz_dtype = Column(String, nullable=True)  # dtype of x,y,z arrays, default float32
    y_path = Column(String, nullable=True)  # if needed, y location cache path relative to workspace
    x_path = Column(String, nullable=True)  # if needed, x location cache path relative to workspace
    z_path = Column(String, nullable=True)  # if needed, x location cache path relative to workspace

    # link to key-value further information; primarily a hedge in case specific information has to be squirreled away for later consideration for main content table
    # this provides dictionary style access to key-value pairs
    info = relationship("ContentKeyValue", collection_class=attribute_mapped_collection('key'))
    _proxied = association_proxy("info", "value",
                                 creator=lambda key, value: ContentKeyValue(key=key, value=value))

    @property
    def uuid(self):
        return self.product.uuid

    @property
    def is_overview(self):
        return self.lod==0

    def __str__(self):
        product = "%s:%s.%s" % (self.product.source.name or '?', self.product.platform or '?', self.product.identifier or '?')
        isoverview = ' overview' if self.is_overview else ''
        dtype = self.dtype or 'float32'
        xyzcs = ' '.join(
            q for (q,p) in zip('XYZCS', (self.x_path, self.y_path, self.z_path, self.coverage_path, self.sparsity_path)) if p
        )
        return "<product {product} content{isoverview} with path={path} dtype={dtype} {xyzcs}>".format(
            product=product, isoverview=isoverview, path=self.path, dtype=dtype, xyzcs=xyzcs)


class ContentKeyValue(Base):
    """
    key-value pairs associated with a product
    """
    __tablename__ = 'content_key_values'
    product_id = Column(ForeignKey(Content.id), primary_key=True)
    key = Column(String, primary_key=True)
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


# # ============================
# # mapping wrappers
#
# class ProductInfoAsWritableMappingAdapter(MutableMapping):
#     """
#     database Product.info dictionary adapter
#     """
#     def __init__(self, session, product, warn_on_write=True):
#         self.S = session
#         self.prod = product
#         self.wow = warn_on_write
#
#     def __contains__(self, item):
#         items = self.S.query(ProductKeyValue).filter_by(product_id=self.prod.id, key=item).all()
#         return len(items)>0
#
#     def __getitem__(self, item:str):
#         kvs = self.S.query(ProductKeyValue).filter_by(product_id=self.prod.id, key=item).all()
#         if not kvs:
#             raise KeyError("product does not have value for key {}".format(item))
#         if len(kvs)>1:
#             raise AssertionError('more than one value for %s' % item)
#
#     def __setitem__(self, key, value):
#         if self.wow:
#             LOG.warning('attempting to write to Product info dictionary in workspace??')
#         kvs = self.S.query(ProductKeyValue).filter_by(product_id=self.prod.id, key=key).all()
#         if not kvs:
#             kv = ProductKeyValue(key=key, value=value)
#             self.S.add(kv)
#             self.product.info.append(kv)
#             self.S.commit()
#         if len(kvs)>1:
#             raise AssertionError('more than one value for {}'.format(key))
#         kvs[0].value = value
#         self.S.commit()


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
        from uuid import uuid1
        uu = uuid1()
        when = datetime.utcnow()
        f = Source(path='', name='foo.bar', mtime=when, atime=when, format=None)
        p = Product(uuid=str(uu), source=f, platform='TEST', identifier='B00')
        p['test_key'] = u'test_value'
        p['turkey'] = u'cobbler'
        s.add(f)
        s.add(p)
        s.commit()
        q = s.query(Product).filter_by(source=f).first()
        self.assertEqual(q['test_key'], u'test_value')
        self.assertEqual(q['turkey'], u'cobbler')

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
