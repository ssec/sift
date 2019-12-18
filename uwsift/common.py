#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
.py
~~~

PURPOSE
Support calculations, namedtuples and constants used throughout the library and application.

REFERENCES


REQUIRES
numpy
numba

:author: R.K.Garcia <rayg@ssec.wisc.edu>
:copyright: 2015 by University of Wisconsin Regents, see AUTHORS for more details
:license: GPLv3, see LICENSE for more details
"""
from enum import Enum
from typing import MutableSequence, Tuple, Optional, Iterable, Any, NamedTuple
from uuid import UUID

import os
import sys
from datetime import datetime, timedelta
import logging
from pyproj import Proj

LOG = logging.getLogger(__name__)

# separator for family::category::serial representation of product identity
FCS_SEP = '::'
# standard N/A string used in FCS
NOT_AVAILABLE = FCS_NA = 'N/A'


def get_font_size(pref_size):
    """Get a font size that looks good on this platform.

    This is a HACK and can be replaced by PyQt5 font handling after migration.

    """
    # win = 7
    # osx = 12
    env_factor = os.getenv("SIFT_FONT_FACTOR", None)
    if env_factor is not None:
        factor = float(env_factor)
    elif sys.platform.startswith('win'):
        factor = 1.
    elif 'darwin' in sys.platform:
        factor = 1.714
    else:
        factor = 1.3

    return pref_size * factor


PREFERRED_SCREEN_TO_TEXTURE_RATIO = 1.0  # screenpx:texturepx that we want to keep, ideally, by striding

# http://home.online.no/~sigurdhu/WGS84_Eng.html

DEFAULT_TILE_HEIGHT = 512
DEFAULT_TILE_WIDTH = 512
DEFAULT_TEXTURE_HEIGHT = 2
DEFAULT_TEXTURE_WIDTH = 16
DEFAULT_ANIMATION_DELAY = 100.0  # milliseconds
# The values below are taken from the test geotiffs that are projected to the `DEFAULT_PROJECTION` below.
# These units are in meters in mercator projection space
DEFAULT_X_PIXEL_SIZE = 4891.969810251281160
DEFAULT_Y_PIXEL_SIZE = -7566.684931505724307
DEFAULT_ORIGIN_X = -20037508.342789247632027
DEFAULT_ORIGIN_Y = 15496570.739723727107048

DEFAULT_PROJECTION = "+proj=merc +datum=WGS84 +ellps=WGS84 +over"
DEFAULT_PROJ_OBJ = p = Proj(DEFAULT_PROJECTION)
C_EQ = p(180, 0)[0] - p(-180, 0)[0]
C_POL = p(0, 89.9)[1] - p(0, -89.9)[1]
MAX_EXCURSION_Y = C_POL / 2.0
MAX_EXCURSION_X = C_EQ / 2.0
# how many 'tessellation' tiles in one texture tile? 2 = 2 rows x 2 cols
TESS_LEVEL = 20
IMAGE_MESH_SIZE = 10
# smallest difference between two image extents (in canvas units)
# before the image is considered "out of view"
CANVAS_EXTENTS_EPSILON = 1e-4


# R_EQ = 6378.1370  # km
# R_POL = 6356.7523142  # km
# C_EQ = 40075.0170  # linear km
# C_POL = 40007.8630  # linear km

# MAX_EXCURSION_Y = C_POL/4.0
# MAX_EXCURSION_X = C_EQ/2.0


class Box(NamedTuple):
    bottom: float
    left: float
    top: float
    right: float


class Resolution(NamedTuple):
    """Pixel resolution (km per pixel)."""
    dy: float
    dx: float


class Point(NamedTuple):
    y: float
    x: float


class Coordinate(NamedTuple):
    deg_north: float
    deg_east: float


class ViewBox(NamedTuple):
    """Combination of Box + Resolution."""
    bottom: float
    left: float
    top: float
    right: float
    dy: float
    dx: float


class Span(NamedTuple):
    s: datetime  # start
    d: timedelta  # duration

    @property
    def e(self):
        return self.s + self.d

    @staticmethod
    def from_s_e(s: datetime, e: datetime):
        return Span(s, e - s) if (s is not None) and (e is not None) else None

    @property
    def is_instantaneous(self):
        return timedelta(seconds=0) == self.d


class Flags(set):
    """A set of enumerated Flags which may ultimately be represented as a bitfield, but observing set interface
    """
    pass


class State(Enum):
    """State for products in document."""
    UNKNOWN = 0
    POTENTIAL = 1  # product is available as a resource and could be imported or calculated
    ARRIVING = 2  # import or calculation in progress
    CACHED = 3  # stable in workspace cache
    ATTACHED = 5  # attached and can page in memory, as well as participating in document
    ONSCREEN = 6  # actually being displayed at this moment
    DANGLING = -1  # cached in workspace but no resource behind it
    # PURGE = -2  # scheduled for purging -- eventually


class Tool(Enum):
    """Names for cursor tools."""
    PAN_ZOOM = "pan_zoom"
    POINT_PROBE = "point_probe"
    REGION_PROBE = "region_probe"


class Kind(Enum):
    """Kind of entities we're working with."""
    UNKNOWN = 0
    IMAGE = 1
    OUTLINE = 2
    SHAPE = 3
    RGB = 4
    COMPOSITE = 1  # deprecated: use Kind.IMAGE instead
    CONTOUR = 6


class CompositeType(Enum):
    """Type of non-luminance image layers."""
    RGB = 1
    ARITHMETIC = 2


class Instrument(Enum):
    UNKNOWN = '???'
    AHI = 'AHI'
    ABI = 'ABI'
    AMI = 'AMI'
    GFS = 'GFS'
    NAM = 'NAM'
    SEVIRI = 'SEVIRI'
    LI = 'LI'
    GLM = 'GLM'


INSTRUMENT_MAP = {v.value.lower().replace('-', ''): v for v in Instrument}


class Platform(Enum):
    UNKNOWN = '???'
    HIMAWARI_8 = 'Himawari-8'
    HIMAWARI_9 = 'Himawari-9'
    GOES_16 = 'G16'
    GOES_17 = 'G17'
    GOES_18 = 'G18'
    NWP = 'NWP'
    MSG8 = 'Meteosat-8'
    MSG9 = 'Meteosat-9'
    MSG10 = 'Meteosat-10'
    MSG11 = 'Meteosat-11'
    GK2A = "GEO-KOMPSAT-2A"


PLATFORM_MAP = {v.value.lower().replace('-', ''): v for v in Platform}
PLATFORM_MAP['h8'] = Platform.HIMAWARI_8
PLATFORM_MAP['h9'] = Platform.HIMAWARI_9
PLATFORM_MAP['goes16'] = Platform.GOES_16
PLATFORM_MAP['goes17'] = Platform.GOES_17
PLATFORM_MAP['goes18'] = Platform.GOES_18


class Info(Enum):
    """
    Standard keys for info dictionaries
    Note: some fields correspond to database fields in workspace.metadatabase !
    """
    UNKNOWN = '???'
    # full path to the resource that the file came from
    # DEPRECATED since datasets may not have one-to-one pathname mapping
    PATHNAME = 'path'

    # CF content
    SHORT_NAME = 'short_name'  # CF short_name
    LONG_NAME = 'long_name'  # CF long_name
    STANDARD_NAME = 'standard_name'  # CF compliant standard_name (when possible)
    UNITS = 'units'  # CF compliant (udunits compliant) units string, original data units

    # SIFT bookkeeping
    DATASET_NAME = 'dataset_name'  # logical name of the file (possibly human assigned)
    KIND = 'kind'  # Kind enumeration on what kind of layer this makes
    UUID = 'uuid'  # UUID assigned on import, which follows the layer around the system

    # track determiner is family::category; presentation is determined by family
    # family::category::serial is a unique identifier equivalent to conventional make-model-serialnumber
    # string representing data family, typically instrument:measurement:wavelength but may vary by data content
    FAMILY = 'family'
    CATEGORY = 'category'  # string with platform:instrument:target typically but may vary by data content
    SERIAL = 'serial'  # serial number

    # projection information
    ORIGIN_X = 'origin_x'
    ORIGIN_Y = 'origin_y'
    CELL_WIDTH = 'cell_width'
    CELL_HEIGHT = 'cell_height'
    PROJ = 'proj4'

    # colormap amd data range
    CLIM = 'clim'  # (min,max) color map limits
    VALID_RANGE = 'valid_range'
    SHAPE = 'shape'  # (rows, columns) or (rows, columns, levels) data shape
    COLORMAP = 'colormap'  # name or UUID of a color map

    SCHED_TIME = 'timeline'  # scheduled time for observation
    OBS_TIME = 'obstime'  # actual time for observation
    OBS_DURATION = 'obsduration'  # time from start of observation to end of observation

    # instrument and scene information
    PLATFORM = 'platform'  # full standard name of spacecraft
    SCENE = 'scene'  # standard scene identifier string for instrument, e.g. FLDK
    INSTRUMENT = 'instrument'  # Instrument enumeration, or string with full standard name

    # human-friendly conventions
    DISPLAY_NAME = 'display_name'  # preferred name in the layer list
    DISPLAY_TIME = 'display_time'  # typically from guidebook, used for labeling animation frame
    UNIT_CONVERSION = 'unit_conversion'  # (preferred CF units, convert_func, format_func)
    # unit numeric conversion func: lambda x, inverse=False: convert-to-units
    # unit string format func: lambda x, numeric=True, units=True: formatted string
    CENTRAL_WAVELENGTH = 'nominal_wavelength'
    # only in family info dictionaries:
    DISPLAY_FAMILY = 'display_family'

    def __lt__(self, other):
        """
        when using pickletype in sqlalchemy tables, a comparator is needed for enumerations
        :param other:
        :return:
        """
        if isinstance(other, str):
            return self.value < other
        elif isinstance(other, type(self)):
            return self.value < other.value
        raise ValueError("cannot compare {} < {}".format(repr(self), repr(other)))

    def __gt__(self, other):
        """
        when using pickletype in sqlalchemy tables, a comparator is needed for enumerations
        :param other:
        :return:
        """
        if isinstance(other, str):
            return self.value > other
        elif isinstance(other, type(self)):
            return self.value > other.value
        raise ValueError("cannot compare {} > {}".format(repr(self), repr(other)))

    def __hash__(self):
        return hash(self.value)

    def __eq__(self, other):
        """
        when using pickletype in sqlalchemy tables, a comparator is needed for enumerations
        :param other:
        :return:
        """
        if isinstance(other, str):
            return self.value == other
        elif isinstance(other, type(self)):
            return self.value == other.value
        raise ValueError("cannot compare {} == {}".format(repr(self), repr(other)))


class Presentation(NamedTuple):
    """Presentation information for a layer.

    z_order comes from the layerset

    """
    uuid: UUID  # dataset in the document/workspace
    kind: Kind  # what kind of layer it is
    visible: bool  # whether it's visible or not
    a_order: int  # None for non-animating, 0..n-1 what order to draw in during animation
    colormap: object  # name or uuid: color map to use; name for default, uuid for user-specified
    climits: tuple  # valid min and valid max used for color mapping normalization
    gamma: float  # valid (0 to 5) for gamma correction (default should be 1.0)
    mixing: object  # mixing mode constant


class ZList(MutableSequence):
    """List indexed from high Z to low Z
    For z-ordered tracks, we want to have
    - contiguous Z order values from high to low (negative)
    - elements assigned negative stay negative, likewise with positive (negative Z implies inactive non-document track)
    - no Z value is repeated
    - insertions are correctly handled
    - append also works, by default arriving as most-negative z-order
    - assignment off either end gets snapped to the contiguous next value
    """
    _zmax: int = 0  # z-index of the first member of _content
    _content: list

    @property
    def min_max(self) -> Tuple[int, int]:
        return (self._zmax + 1 - len(self), self._zmax) if len(self) else (None, None)

    @property
    def top_z(self) -> Optional[int]:
        return self._zmax if len(self) else None

    @property
    def bottom_z(self) -> Optional[int]:
        return self._zmax + 1 - len(self) if len(self) else None

    def __contains__(self, z) -> bool:
        n, x = self.min_max
        return False if (n is None) or (x is None) or (z < n) or (z > x) else True

    def prepend(self, val):
        self._zmax += 1
        self._content.insert(0, val)

    def append(self, val, start_negative: bool = False, not_if_present: bool = False):
        if start_negative and 0 == len(self._content):
            self._zmax = -1
        if not_if_present and (val in self._content):
            return
        self._content.append(val)

    def items(self) -> Iterable[Tuple[int, Any]]:
        z = self._zmax
        for q, v in enumerate(self._content):
            yield z - q, v

    def index(self, val):
        ldex = self._content.index(val)
        z = self._zmax - ldex
        return z

    def keys(self):
        yield from range(self._zmax, self._zmax - len(self._content), -1)

    def values(self):
        yield from iter(self._content)

    def insert(self, z: int, val):
        """insert a value such that it lands at index z
        as needed:
        displace any content with z>=0 upward
        displace any content with z<0 downward
        """
        if len(self._content) == 0:
            self._zmax = -1 if (z < 0) else 0
            self._content.append(val)
        elif z not in self:
            if z >= 0:
                self._zmax += 1
                self._content.insert(0, val)
            else:
                self._content.append(val)
        else:
            adj = 1 if z >= 0 else 0
            ldex = max(0, self._zmax - z + adj)
            self._content.insert(ldex, val)
            self._zmax += adj

    def move(self, to_z: int, val):
        old_z = self.index(val)
        if old_z != to_z:
            del self[old_z]
            self.insert(to_z, val)

    def __init__(self, zmax: int = None, content: Iterable[Any] = None):
        super(ZList, self).__init__()
        if zmax is not None:
            self._zmax = zmax
        self._content = list(content) if content is not None else []

    def __len__(self) -> int:
        return len(self._content)

    def __setitem__(self, z, val):
        ldex = self._zmax - z
        if ldex < 0:
            self._content.insert(0, val)
            self._zmax += 1
        elif ldex >= len(self._content):
            self._content.append(val)
        else:
            self._content[ldex] = val

    def __getitem__(self, z) -> Any:
        ldex = self._zmax - z
        if ldex < 0 or ldex >= len(self._content):
            raise IndexError("Z={} not in ZList".format(z))
        return self._content[ldex]

    def __delitem__(self, z):
        if z not in self:
            raise IndexError("Z={} not in ZList".format(z))
        ldex = self._zmax - z
        if z >= 0:
            self._zmax -= 1
        del self._content[ldex]

    def merge_subst(self, new_values: Iterable[Tuple[int, Any]]):
        """batch merge of substitutions
        raises IndexError if any of them is outside current range
        """
        for z, q in new_values:
            ldex = self._zmax - z
            self._content[ldex] = q

    def __repr__(self) -> str:
        return 'ZList({}, {})'.format(self._zmax, repr(self._content))

    def __eq__(self, other) -> bool:
        return isinstance(other, ZList) and other._zmax == self._zmax and other._content == self._content

    def to_dict(self, inverse=False) -> dict:
        if not inverse:
            return dict(self.items())
        else:
            zult = dict((b, a) for (a, b) in self.items())
            if len(zult) != len(self._content):
                raise RuntimeWarning("ZList.to_dict inverse did not have fully unique keys")
            return zult
