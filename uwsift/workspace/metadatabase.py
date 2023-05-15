#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
metadatabase.py
===============

PURPOSE
SQLAlchemy database tables of metadata used by CachingWorkspace to manage its local cache.


OVERVIEW

::

    Resource : a file containing products, somewhere in the filesystem,
     |         or a resource on a remote system we can access (openDAP etc)
     |_ Product* : product stored in a resource
         |_ Content* : workspace cache content corresponding to a product,
         |   |         may be one of many available views (e.g. projections)
         |   |_ ContentKeyValue* : additional information on content
         |_ ProductKeyValue* : additional information on product
         |_ SymbolKeyValue* : if product is derived from other products,
                              symbol table for that expression is in this kv table

A typical baseline product will have two content: and overview (lod==0) and a native resolution (lod>0)


REQUIRES
SQLAlchemy with SQLite

:author: R.K.Garcia <rayg@ssec.wisc.edu>
:copyright: 2016 by University of Wisconsin Regents, see AUTHORS for more details
:license: GPLv3, see LICENSE for more details
"""

import logging
import os
from collections import defaultdict
from collections.abc import MutableMapping
from datetime import datetime
from typing import Optional
from uuid import UUID

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    Interval,
    PickleType,
    String,
    Table,
    Unicode,
    create_engine,
)
from sqlalchemy.ext.associationproxy import association_proxy
from sqlalchemy.orm import Session, backref, relationship, scoped_session, sessionmaker
from sqlalchemy.orm.collections import attribute_mapped_collection

try:
    from sqlalchemy.orm import declarative_base, declared_attr
except ImportError:
    # sqlalchemy <2.0
    from sqlalchemy.ext import declarative_base, declared_attr

from uwsift.common import FCS_SEP, Info

LOG = logging.getLogger(__name__)

# =================
# Database Entities

Base = declarative_base()

# resources can have multiple products in them
# products may require multiple resources (e.g. separate GEO; tiled imagery)
PRODUCTS_FROM_RESOURCES_TABLE_NAME = "product_resource_assoc_v1"
ProductsFromResources = Table(
    PRODUCTS_FROM_RESOURCES_TABLE_NAME,
    Base.metadata,
    Column("product_id", Integer, ForeignKey("products_v1.id")),
    Column("resource_id", Integer, ForeignKey("resources_v1.id")),
)


class Resource(Base):  # type: ignore
    """
    held metadata regarding a file that we can access and import data into the workspace from
    resources are external to the workspace, but the workspace can keep track of them in its database
    """

    __tablename__ = "resources_v1"
    # identity information
    id = Column(Integer, primary_key=True)

    # primary handler
    format = Column(PickleType)  # classname, class or callable which can pull this data into workspace from storage

    # {scheme}://{path}/{name}?{query}, default is just an absolute path in filesystem
    scheme = Column(
        Unicode, nullable=True
    )  # uri scheme for the content (the part left of ://), assume file:// by default
    path = Column(Unicode)  # '/' separated real path
    query = Column(Unicode, nullable=True)  # query portion of a URI or URL, e.g. 'interval=1m&stride=2'

    mtime = Column(DateTime)  # last observed mtime of the file, for change checking
    atime = Column(DateTime)  # last time this file was accessed by application

    product = relationship("Product", secondary=ProductsFromResources, backref="resource")

    @property
    def uri(self):
        return (
            self.path
            if (not self.scheme or self.scheme == "file")
            else "{}://{}/{}{}".format(self.scheme, self.path, self.name, "" if not self.query else "?" + self.query)
        )

    def touch(self, when=None):
        self.atime = datetime.utcnow() if not when else when

    def exists(self):
        if self.scheme not in {None, "file"}:
            return True  # FUTURE: alternate tests for still-exists-ness
        if os.path.exists(self.path):
            return True
        LOG.info("path {} no longer exists".format(self.path))
        return False


class ChainRecordWithDict(MutableMapping):
    """
    allow Product database entries and key-value table to act as a coherent dictionary
    """

    def __init__(self, obj, field_keys, more):
        self._obj, self._field_keys, self._more = obj, field_keys, more

    def keys(self):
        return set(self._more.keys()) | set(self._field_keys.keys())

    def items(self):
        for k in self.keys():
            yield k, self[k]

    def values(self):
        for k in self.keys():
            yield self[k]

    def __len__(self):
        return len(self.keys())

    def __iter__(self):
        yield from self.keys()

    def __contains__(self, item):
        return item in self.keys()

    def __getitem__(self, key):
        fieldname = self._field_keys.get(key)
        if fieldname is not None:
            assert isinstance(fieldname, str)  # nosec B101
            return getattr(self._obj, fieldname)
        return self._more[key]

    def __repr__(self):
        return "<ChainRecordWithDict {}>".format(repr(tuple(self.keys())))

    def __setitem__(self, key, value):
        fieldname = self._field_keys.get(key)  # Info -> fieldname
        if fieldname is not None:
            LOG.debug("assigning database field {}".format(fieldname))
            setattr(self._obj, fieldname, value)
        else:
            self._more[key] = value

    def __delitem__(self, key):
        if key in self._field_keys:
            raise KeyError("cannot remove key {}".format(key))
        del self._more[key]


class Product(Base):  # type: ignore
    """
    Primary entity being tracked in metadatabase
    One or more StoredProduct are held in a single File
    A StoredProduct has zero or more Content representations, potentially at different projections
    A StoredProduct has zero or more ProductKeyValue pairs with additional metadata
    A File's format allows data to be imported to the workspace
    A StoredProduct's kind determines how its cached data is transformed to different representations for display
    additional information is stored in a key-value table addressable as product[key:str]
    """

    __tablename__ = "products_v1"

    # identity information
    id = Column(Integer, primary_key=True)
    resource_id = Column(Integer, ForeignKey(Resource.id))
    # relationship: .resource
    uuid_str = Column(
        String, nullable=False, unique=True
    )  # UUID representing this data in SIFT, or None if not in cache

    @property
    def uuid(self):
        return UUID(self.uuid_str)

    @uuid.setter
    def uuid(self, uu):
        self.uuid_str = str(uu)

    # primary handler
    atime = Column(DateTime, nullable=False)  # last time this file was accessed by application

    # cached metadata provided by the file format handler
    # product identifier eg "B01", "B02"  # resource + shortname should be sufficient to identify the data
    name = Column(String, nullable=False)

    # presentation is consistent within a family
    # family::category determines track, which should represent a product sequence in time
    # family::category::serial is effectively a product unique identifier
    # colon-separated family identifier, typically kind:pointofreference:measurement:wavelength, e.g. image:geo:refl:11µ
    #  with <> used for generated content.
    family = Column(Unicode, nullable=False)
    # colon-separated processing-system, platform, instrument and scene name,
    # typically system:platform:instrument:target e.g. NOAA-PUG:GOES-16:ABI:CONUS
    category = Column(Unicode, nullable=False)
    # serial number within family and category; typically time-related; use ISO8601 times please
    serial = Column(Unicode, nullable=False)

    @property
    def track(self):
        """track is family::category."""
        return self.family + FCS_SEP + self.category

    @track.setter
    def track(self, new_track: str):
        fam, ctg = new_track.split("::")
        self.family, self.category = fam, ctg

    @property
    def ident(self):
        return self.family + FCS_SEP + self.category + FCS_SEP + self.serial

    @ident.setter
    def ident(self, new_ident: str):
        fam, ctg, ser = new_ident.split("::")
        self.family, self.category, self.serial = fam, ctg, ser

    # times
    obs_time = Column(DateTime, nullable=False)  # actual observation time start
    obs_duration = Column(Interval, nullable=False)  # duration of the observation

    # link to workspace cache files representing this data
    content = relationship("Content", backref=backref("product"), cascade="all")

    # link to key-value further information
    # this provides dictionary style access to key-value pairs
    _key_values = relationship(
        "ProductKeyValue", collection_class=attribute_mapped_collection("key"), cascade="all, delete-orphan"
    )
    _kwinfo = association_proxy(
        "_key_values", "value", creator=lambda key, value: ProductKeyValue(key=key, value=value)
    )

    # derived / algebraic datasets have a symbol table and an expression
    # typically Content objects for algebraic datasets cache calculation output
    symbol = relationship("SymbolKeyValue", backref=backref("product"), cascade="all, delete-orphan")
    expression = Column(Unicode, nullable=True)

    _info = None  # database fields and key-value dictionary merged as one transparent mapping

    def __init__(self, *args, **kwargs):
        super(Product, self).__init__(*args, **kwargs)

    @classmethod
    def _separate_fields_and_keys(cls, mapping):
        fields = {}
        keyvalues = {}
        valset = set(cls.INFO_TO_FIELD.values())
        columns = set(cls.__table__.columns.keys())
        for k, v in mapping.items():
            f = cls.INFO_TO_FIELD.get(k)
            if f is not None:
                fields[f] = v
            elif k in valset:
                LOG.warning(
                    "key {} corresponds to a database field when standard key is available; "
                    "this code may not be intended".format(k)
                )
                fields[k] = v
            elif k in columns:
                fields[k] = v
            else:
                keyvalues[k] = v
        return fields, keyvalues

    @classmethod
    def from_info(cls, mapping, symbols=None, codeblock=None, only_fields=False):
        """
        create a Product using info Info dictionary items and arbitrary key-values
        :param mapping: dictionary of product metadata
        :return: Product object
        """
        fields, keyvalues = cls._separate_fields_and_keys(mapping)
        LOG.debug("fields to import: {}".format(repr(fields)))
        LOG.debug("key-value pairs to {}: {}".format("IGNORE" if only_fields else "import", repr(keyvalues)))
        try:
            p = cls(**fields)
        except AttributeError:
            LOG.error("unable to initialize Product from info: {}".format(repr(fields)))
            raise
        if not only_fields:
            p.info.update(keyvalues)
        if symbols:
            for k, v in symbols.items():
                p.symbol.append(SymbolKeyValue(key=k, value=v))
        if codeblock:
            p.expression = codeblock

        return p

    def __repr__(self):
        return "<Product '{}' @ {}~{} / {} keys>".format(
            self.name, self.obs_time, self.obs_time + self.obs_duration, len(self.info.keys())
        )

    @property
    def info(self):
        """
        :return: mapping merging Info-compatible database fields with key-value dictionary access pattern
        """
        if self._info is None:
            self._info = ChainRecordWithDict(self, self.INFO_TO_FIELD, self._kwinfo)
        return self._info

    def update(self, d, only_keyvalues=False, only_fields=False):
        """
        update metadata, optionally only permitting key-values to be updated instead of established database fields
        :param d: mapping of combined database fields and key-values (using Info keys where possible)
        :param only_keyvalues: true if only key-value attributes should be updated
        :return:
        """
        if only_keyvalues:
            _, keys = self._separate_fields_and_keys(d)
            self._kwinfo.update(keys)
        elif only_fields:
            fields, _ = self._separate_fields_and_keys(d)
            self.info.update(fields)
        else:
            self.info.update(d)

    @property
    def proj4(self):
        nat = self.content[-1] if len(self.content) else None
        return nat.proj4 if nat else None

    @proj4.setter
    def proj4(self, value):
        LOG.debug("DEPRECATED: setting proj4 on resource")

    @property
    def cell_height(self):
        if not self.content:
            return None
        nat = self.content[-1]
        if isinstance(nat, ContentImage):
            return nat.cell_height
        return None

    @cell_height.setter
    def cell_height(self, value):
        LOG.debug("DEPRECATED: setting cell_height on resource")

    @property
    def cell_width(self):
        if not self.content:
            return None
        nat = self.content[-1]
        if isinstance(nat, ContentImage):
            return nat.cell_width
        return None

    @cell_width.setter
    def cell_width(self, value):
        LOG.debug("DEPRECATED: setting cell_width on resource")

    @property
    def origin_x(self):
        if not self.content:
            return None
        nat = self.content[-1]
        if isinstance(nat, ContentImage):
            return nat.origin_x
        return None

    @origin_x.setter
    def origin_x(self, value):
        LOG.debug("DEPRECATED: setting origin_x on resource")

    @property
    def origin_y(self):
        if not self.content:
            return None
        nat = self.content[-1]
        if isinstance(nat, ContentImage):
            return nat.origin_y
        return None

    @origin_y.setter
    def origin_y(self, value):
        LOG.debug("DEPRECATED: setting origin_y on resource")

    @property
    def grid_origin(self):
        if not self.content:
            return None
        nat = self.content[-1]
        if isinstance(nat, ContentImage):
            return nat.grid_origin
        return None

    @grid_origin.setter
    def grid_origin(self, value):
        LOG.debug("DEPRECATED: setting grid_origin on resource")

    @property
    def grid_first_index_x(self):
        if not self.content:
            return None
        nat = self.content[-1]
        if isinstance(nat, ContentImage):
            return nat.grid_first_index_x
        return None

    @grid_first_index_x.setter
    def grid_first_index_x(self, value):
        LOG.debug("DEPRECATED: setting grid_first_index_x on resource")

    @property
    def grid_first_index_y(self):
        if not self.content:
            return None
        nat = self.content[-1]
        if isinstance(nat, ContentImage):
            return nat.grid_first_index_y
        return None

    @grid_first_index_y.setter
    def grid_first_index_y(self, value):
        LOG.debug("DEPRECATED: setting grid_first_index_y on resource")

    INFO_TO_FIELD = {
        Info.SHORT_NAME: "name",
        Info.UUID: "uuid",
        Info.PROJ: "proj4",
        Info.OBS_TIME: "obs_time",
        Info.OBS_DURATION: "obs_duration",
        Info.CELL_WIDTH: "cell_width",
        Info.CELL_HEIGHT: "cell_height",
        Info.ORIGIN_X: "origin_x",
        Info.ORIGIN_Y: "origin_y",
        Info.GRID_ORIGIN: "grid_origin",
        Info.GRID_FIRST_INDEX_X: "grid_first_index_x",
        Info.GRID_FIRST_INDEX_Y: "grid_first_index_y",
        Info.FAMILY: "family",
        Info.CATEGORY: "category",
        Info.SERIAL: "serial",
    }

    def touch(self, when=None):
        self.atime = when = when or datetime.utcnow()
        [x.touch(when) for x in self.resource]


class ProductKeyValue(Base):  # type: ignore
    """
    key-value pairs associated with a product
    """

    __tablename__ = "product_key_values_v1"
    product_id = Column(ForeignKey(Product.id), primary_key=True)
    key = Column(
        PickleType, primary_key=True
    )  # FUTURE: can this be a string? for now need pickling of Info/Platform Enum
    # relationship: .product
    value = Column(PickleType)


class SymbolKeyValue(Base):  # type: ignore
    """
    datasets of derived layers have a symbol table which becomes namespace used by expression
    """

    __tablename__ = "algebraic_symbol_key_values_v1"
    product_id = Column(ForeignKey(Product.id), primary_key=True)
    key = Column(Unicode, primary_key=True)
    # relationship: .product
    value = Column(PickleType, nullable=True)  # UUID object typically


class Content(Base):  # type: ignore
    """
    represent flattened product data files in cache (i.e. cache content)
    typically memory-map ready data (np.memmap)
    basic correspondence to projection/geolocation information may accompany
    images will typically have rows>0 cols>0 levels=None (implied levels=1)
    profiles may have rows>0 cols=None (implied cols=1) levels>0
    a given product may have several Content for different projections
    additional information is stored in a key-value table addressable as content[key:str]
    """

    __tablename__ = "content_v1"
    id = Column(Integer, primary_key=True)
    type = Column(String)
    __mapper_args__ = {
        "polymorphic_identity": "base",
        "polymorphic_on": type,
    }

    product_id = Column(Integer, ForeignKey(Product.id))

    # time accounting, used to check if data needs to be re-imported to workspace,
    # or whether data is LRU and can be removed from a crowded workspace
    mtime = Column(DateTime)  # last observed mtime of the original source of this data, for change checking
    atime = Column(DateTime)  # last time this product was accessed by application

    # actual data content
    # NaNs are used to signify missing data; NaNs can include integer category fields in significand;
    # please ref IEEE 754
    path = Column(String, unique=True)  # relative to workspace, binary array of data

    # TODO Number of attributes per point: n_attributes (e.g. lightning peak
    #  current) but then, what are the datatypes of the attributes?
    n_attributes = Column(Integer)  # always 1, reserverd for future extensions

    # TODO Currently all attributes must have the same datatype as the points,
    #  because there is no way yet to define them differently
    #  therefore only float32 is possible.
    #  lat, lon (from SwathDefinition) are always float, thus dtype must be
    #  float for now, only attributes could have a different dtype in the future
    # default float32; can be int16 in the future for scaled integer images for instance; should be a numpy type name
    dtype = Column(String, nullable=True)

    # json for numpy array with polynomial coefficients for transforming native data to natural units
    # (e.g. for scaled integers), c[0] + c[1]*x + c[2]*x**2 ...

    # json for optional dict {int:string} lookup table for NaN flag fields
    # (when dtype is float32 or float64) or integer values (when dtype is an int8/16/32/64)

    # projection information for this representation of the data
    # proj4 projection string for the data in this array, if one exists; else assume y=lat/x=lon
    proj4 = Column(String, nullable=True)  # TODO maybe basis

    # link to key-value further information; primarily a hedge in case specific information
    # has to be squirreled away for later consideration for main content table
    # this provides dictionary style access to key-value pairs
    _key_values = relationship(
        "ContentKeyValue", collection_class=attribute_mapped_collection("key"), cascade="all, delete-orphan"
    )
    _kwinfo = association_proxy(
        "_key_values", "value", creator=lambda key, value: ContentKeyValue(key=key, value=value)
    )

    INFO_TO_FIELD = {Info.PROJ: "proj4", Info.PATHNAME: "path"}

    _info = None  # database fields and key-value dictionary merged as one transparent mapping

    def __init__(self, *args, **kwargs):
        super(Content, self).__init__(*args, **kwargs)
        self._info = ChainRecordWithDict(self, self.INFO_TO_FIELD, self._kwinfo)

    @classmethod
    def _separate_fields_and_keys(cls, mapping):
        fields = {}
        keyvalues = {}
        valset = set(cls.INFO_TO_FIELD.values())
        columns = set(cls.__table__.columns.keys())
        for k, v in mapping.items():
            f = cls.INFO_TO_FIELD.get(k)
            if f is not None:
                fields[f] = v
            elif k in valset:
                LOG.warning(
                    "key {} corresponds to a database field when standard key is available; "
                    "this code may not be intended".format(k)
                )
                fields[k] = v
            elif k in columns:
                fields[k] = v
            else:
                keyvalues[k] = v
        return fields, keyvalues

    @classmethod
    def from_info(cls, mapping, only_fields=False):
        """
        create a Product using info Info dictionary items and arbitrary key-values
        :param mapping: dictionary of product metadata
        :return: Product object
        """
        fields, keyvalues = cls._separate_fields_and_keys(mapping)
        LOG.debug("fields to import: {}".format(repr(fields)))
        LOG.debug("key-value pairs to {}: {}".format("IGNORE" if only_fields else "import", repr(keyvalues)))
        try:
            p = cls(**fields)
        except AttributeError:
            LOG.error("unable to initialize Content from info: {}".format(repr(fields)))
            raise
        if not only_fields:
            p.info.update(keyvalues)
        return p

    @property
    def info(self):
        """
        :return: mapping merging Info-compatible database fields with key-value dictionary access pattern
        """
        if self._info is None:
            self._info = ChainRecordWithDict(self, self.INFO_TO_FIELD, self._kwinfo)
        return self._info

    def update(self, d, only_keyvalues=False, only_fields=False):
        """
        update metadata, optionally only permitting key-values to be updated instead of established database fields
        :param d: mapping of combined database fields and key-values (using Info keys where possible)
        :param only_keyvalues: true if only key-value attributes should be updated
        :return:
        """
        if only_keyvalues:
            _, keys = self._separate_fields_and_keys(d)
            self._kwinfo.update(keys)
        elif only_fields:
            fields, _ = self._separate_fields_and_keys(d)
            self.info.update(fields)
        else:
            self.info.update(d)

    @property
    def name(self):
        return self.product.name

    @property
    def uuid(self):
        return self.product.uuid

    def __str__(self):
        product = "%s:%s.%s" % (
            self.product.source.name or "?",
            self.product.platform or "?",
            self.product.identifier or "?",
        )
        dtype = self.dtype or "float32"
        xyzcs = " ".join(
            q
            for (q, p) in zip("XYZCS", (self.x_path, self.y_path, self.z_path, self.coverage_path, self.sparsity_path))
            if p
        )
        return "<{uuid} product {product} content with path={path} dtype={dtype} {xyzcs}>".format(
            uuid=self.uuid, product=product, path=self.path, dtype=dtype, xyzcs=xyzcs
        )

    def touch(self, when: Optional[datetime] = None) -> None:
        self.atime = when = when or datetime.utcnow()
        self.product.touch(when)


class ContentImage(Content):
    __mapper_args__ = {"polymorphic_identity": "image"}

    # handle overview versus detailed data
    lod = Column(Integer)  # power of 2 level of detail; 0 for coarse-resolution overview
    LOD_OVERVIEW = 0

    resolution = Column(Integer)  # maximum resolution in meters for this representation of the dataset

    rows = Column(Integer)
    cols = Column(Integer, nullable=True)
    levels = Column(Integer, nullable=True)

    cell_width = Column(Float, nullable=True)
    cell_height = Column(Float, nullable=True)
    origin_x = Column(Float, nullable=True)
    origin_y = Column(Float, nullable=True)

    # original grid layout information
    grid_origin = Column(String, nullable=True)
    grid_first_index_x = Column(Integer, nullable=True)
    grid_first_index_y = Column(Integer, nullable=True)

    # sparsity and coverage, int8 arrays if needed to show incremental availability of the data
    # dimensionality is always a reduction factor of rows/cols/levels
    # coverage is stretched across the data array
    #   e.g. for loading data sectioned or blocked across multiple files
    # sparsity is broadcast over the data array
    #   e.g. for incrementally loading sparse data into a dense array
    # a zero value indicates data is not available, nonzero signifies availability
    coverage_rows = Column(Integer, nullable=True)
    coverage_cols = Column(Integer, nullable=True)
    coverage_levels = Column(Integer, nullable=True)
    coverage_path = Column(String, nullable=True)
    sparsity_rows = Column(Integer, nullable=True)
    sparsity_cols = Column(Integer, nullable=True)
    sparsity_levels = Column(Integer, nullable=True)
    sparsity_path = Column(String, nullable=True)

    # navigation information, if required
    xyz_dtype = Column(String, nullable=True)  # dtype of x,y,z arrays, default float32
    y_path = Column(String, nullable=True)  # if needed, y location cache path relative to workspace
    x_path = Column(String, nullable=True)  # if needed, x location cache path relative to workspace
    z_path = Column(String, nullable=True)  # if needed, z location cache path relative to workspace

    INFO_TO_FIELD = {
        Info.PROJ: "proj4",
        Info.PATHNAME: "path",
        Info.CELL_HEIGHT: "cell_height",
        Info.CELL_WIDTH: "cell_width",
        Info.ORIGIN_X: "origin_x",
        Info.ORIGIN_Y: "origin_y",
        Info.GRID_ORIGIN: "grid_origin",
        Info.GRID_FIRST_INDEX_X: "grid_first_index_x",
        Info.GRID_FIRST_INDEX_Y: "grid_first_index_y",
    }

    def __init__(self, *args, **kwargs):
        super(ContentImage, self).__init__(*args, **kwargs)

    @property
    def is_overview(self):
        return self.lod == self.LOD_OVERVIEW

    @property
    def shape(self):
        return tuple(filter(lambda x: x, [self.rows, self.cols, self.levels]))


class ContentMultiChannelImage(ContentImage):
    __mapper_args__ = {"polymorphic_identity": "mc_image"}

    bands = Column(Integer, nullable=True)
    coverage_bands = Column(Integer, nullable=True)

    def __int__(self, *args, **kwargs):
        super(ContentMultiChannelImage, self).__int__(*args, **kwargs)

    @property
    def shape(self):
        return tuple(filter(lambda x: x, [self.rows, self.cols, self.bands]))


class ContentUnstructuredPoints(Content):
    __mapper_args__ = {"polymorphic_identity": "unstructured_points"}

    n_points = Column(Integer)

    @declared_attr
    def n_dimensions(cls):
        # Points may have 2 or 3 spatial dimensions
        return Content.__table__.c.get("n_dimensions", Column(Integer))


class ContentLines(Content):
    __mapper_args__ = {"polymorphic_identity": "lines"}

    n_lines = Column(Integer)

    @declared_attr
    def n_dimensions(cls):
        # Points have 4 spatial dimensions
        return Content.__table__.c.get("n_dimensions", Column(Integer))


class ContentKeyValue(Base):  # type: ignore
    """
    key-value pairs associated with a product
    """

    __tablename__ = "content_key_values_v1"
    product_id = Column(ForeignKey(Content.id), primary_key=True)
    key = Column(PickleType, primary_key=True)  # FUTURE: can this be a string?
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
    session_nesting = None

    def __init__(self, uri=None, **kwargs):
        self.session_nesting = defaultdict(int)
        global _MDB
        if _MDB is not None:
            raise AssertionError("Metadatabase is a singleton and already exists")
        self._MDB = self
        if uri:
            self._connect(uri, **kwargs)

    @staticmethod
    def instance(*args, **kwargs):
        global _MDB
        if _MDB is None:
            _MDB = Metadatabase(*args, **kwargs)
        return _MDB

    def _all_tables_present(self):
        from sqlalchemy.engine.reflection import Inspector

        inspector = Inspector.from_engine(self.engine)
        all_tables = set(inspector.get_table_names())
        zult = True
        for table_name in (
            Resource.__tablename__,
            Product.__tablename__,
            ProductKeyValue.__tablename__,
            SymbolKeyValue.__tablename__,
            Content.__tablename__,
            ContentKeyValue.__tablename__,
            PRODUCTS_FROM_RESOURCES_TABLE_NAME,
        ):
            present = table_name in all_tables
            LOG.debug("table {} {} present in database".format(table_name, "is" if present else "is not"))
            zult = False if not present else zult
        return zult

    def _connect(self, uri, create_tables=False, **kwargs):
        assert self.engine is None  # nosec B101
        assert self.connection is None  # nosec B101
        self.engine = create_engine(uri, **kwargs)
        LOG.info("attaching database at {}".format(uri))
        if create_tables or not self._all_tables_present():
            LOG.info("creating database tables")
            Base.metadata.create_all(self.engine)
        self.connection = self.engine.connect()
        # http://docs.sqlalchemy.org/en/latest/orm/contextual.html
        self.session_factory = sessionmaker(bind=self.engine)
        self.SessionRegistry = scoped_session(self.session_factory)  # thread-local session registry

    def session(self):
        return self.session_factory()

    def __enter__(self) -> Session:
        ses = self.SessionRegistry()
        assert self.session_nesting is not None  # nosec B101 # suppress mypy [index]
        self.session_nesting[id(ses)] += 1
        return ses

    def __exit__(self, exc_type, exc_val, exc_tb):
        # fetch the active session, typically zero or one active per thread
        s = self.SessionRegistry()
        self.session_nesting[id(s)] -= 1
        if self.session_nesting[id(s)] <= 0:
            if exc_val is not None:
                LOG.warning("an exception occurred, rolling back any metadatabase changes")
                s.rollback()
            else:
                if bool(s.dirty) or bool(s.new) or bool(s.deleted):
                    LOG.debug("committing metadatabase changes")
                    s.commit()
                else:
                    LOG.debug("closing clean database session without commit")
            s.close()
            del self.session_nesting[id(s)]
        else:  # we're in a nested context for this session
            if exc_val is not None:  # propagate the exception to the outermost session context
                LOG.warning("propagating database exception to outermost context")
                raise exc_val

    #
    # high-level functions
    #
