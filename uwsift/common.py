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
import logging
from dataclasses import dataclass
from enum import Enum
from typing import List, NamedTuple, Optional, Union
from uuid import UUID

from pyproj import Proj

LOG = logging.getLogger(__name__)

# separator for family::category::serial representation of product identity
FCS_SEP = "::"
# standard N/A string used in FCS
NOT_AVAILABLE = FCS_NA = "N/A"

PREFERRED_SCREEN_TO_TEXTURE_RATIO = 1.0  # screenpx:texturepx that we want to keep, ideally, by striding

# http://home.online.no/~sigurdhu/WGS84_Eng.html

DEFAULT_TILE_HEIGHT = 512
DEFAULT_TILE_WIDTH = 512
DEFAULT_TEXTURE_HEIGHT = 2
DEFAULT_TEXTURE_WIDTH = 16
DEFAULT_ANIMATION_DELAY = 500.0  # milliseconds
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
TESS_LEVEL = 200
IMAGE_MESH_SIZE = 100
# smallest difference between two image extents (in canvas units)
# before the image is considered "out of view"
CANVAS_EXTENTS_EPSILON = 1e-4

# For simple geolocated image
# Size of a image quad subdivision grid cell in meters
DEFAULT_GRID_CELL_WIDTH = 96000
DEFAULT_GRID_CELL_HEIGHT = 96000

DEFAULT_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"

# R_EQ = 6378.1370  # km
# R_POL = 6356.7523142  # km
# C_EQ = 40075.0170  # linear km
# C_POL = 40007.8630  # linear km

# MAX_EXCURSION_Y = C_POL/4.0
# MAX_EXCURSION_X = C_EQ/2.0

N_A = "n/a"
NAN = str(float("nan"))

# LayerModel column display names
VISIBILITY = ""
SOURCE = "Satellite & Instrument"
NAME = "Name"
WAVELENGTH = "λ"
PROBE_VALUE = "Probe"
PROBE_UNIT = "Unit"

LAYER_TREE_VIEW_HEADER = [VISIBILITY, SOURCE, NAME, WAVELENGTH, PROBE_VALUE, PROBE_UNIT]

INVALID_COLOR_LIMITS = (float("inf"), float("-inf"))  # Yes, (+inf, -inf), this is the smallest imaginable range
FALLBACK_RANGE = (0.0, 255.0)


class ImageDisplayMode(str, Enum):
    SIMPLE_GEOLOCATED = "simple_geolocated"
    TILED_GEOLOCATED = "tiled_geolocated"
    PIXEL_MATRIX = "pixel_matrix"


# Calculate and provide LayerModel column indices from LAYER_TREE_VIEW_HEADER
class LayerModelColumns:
    VISIBILITY = LAYER_TREE_VIEW_HEADER.index(VISIBILITY)  # noqa
    SOURCE = LAYER_TREE_VIEW_HEADER.index(SOURCE)  # noqa
    NAME = LAYER_TREE_VIEW_HEADER.index(NAME)  # noqa
    WAVELENGTH = LAYER_TREE_VIEW_HEADER.index(WAVELENGTH)  # noqa
    PROBE_VALUE = LAYER_TREE_VIEW_HEADER.index(PROBE_VALUE)  # noqa
    PROBE_UNIT = LAYER_TREE_VIEW_HEADER.index(PROBE_UNIT)  # noqa


LATLON_GRID_DATASET_NAME = "Geo-Grid"  # noqa
BORDERS_DATASET_NAME = "Borders"
DEFAULT_GAMMA_VALUE = 1.0


class LayerVisibility(NamedTuple):
    """Combine the two parameters controlling the visibility of layers."""

    visible: bool
    opacity: float


class Box(NamedTuple):
    bottom: float
    left: float
    top: float
    right: float


class IndexBox(NamedTuple):
    """Box using integer index values as coordinates.

    This has a very similar purpose to :class:`Box`, but restricts coordinates
    to integer values.

    """

    bottom: int
    left: int
    top: int
    right: int


class Resolution(NamedTuple):
    """Pixel resolution (km per pixel)."""

    dy: float
    dx: float


class Point(NamedTuple):
    y: float
    x: float


class ViewBox(NamedTuple):
    """Combination of Box + Resolution."""

    bottom: float
    left: float
    top: float
    right: float
    dy: float
    dx: float


class Flags(set):
    """A set of enumerated Flags which may ultimately be represented as a bitfield, but observing set interface"""

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
    LINES = 7
    VECTORS = 8
    POINTS = 9
    MC_IMAGE = 10


class Instrument(Enum):
    UNKNOWN = "???"
    GENERATED = "Generated"  # for auxiliary data loaded/generated by SIFT
    MIXED = "Mixed"  # for data computed from data from different instruments
    AHI = "AHI"
    ABI = "ABI"
    AMI = "AMI"
    GFS = "GFS"
    NAM = "NAM"
    SEVIRI = "SEVIRI"
    LI = "LI"
    GLM = "GLM"
    FCI = "FCI"
    AVHRR3 = "AVHRR-3"


INSTRUMENT_MAP = {v.value.lower().replace("-", ""): v for v in Instrument}


class Platform(Enum):
    UNKNOWN = "???"
    SYSTEM = "System"  # for auxiliary data loaded/generated by SIFT
    MIXED = "Mixed"  # for data computed from data from different platforms
    HIMAWARI_8 = "Himawari-8"
    HIMAWARI_9 = "Himawari-9"
    GOES_16 = "G16"
    GOES_17 = "G17"
    GOES_18 = "G18"
    GOES_19 = "G19"
    NWP = "NWP"
    MSG1 = "Meteosat-8"
    MSG2 = "Meteosat-9"
    MSG3 = "Meteosat-10"
    MSG4 = "Meteosat-11"
    GK2A = "GEO-KOMPSAT-2A"
    MTGI1 = "MTG-I1"
    M01 = "Metop-B"
    M02 = "Metop-A"
    M03 = "Metop-C"


PLATFORM_MAP = {v.value.lower().replace("-", ""): v for v in Platform}
PLATFORM_MAP["h8"] = Platform.HIMAWARI_8
PLATFORM_MAP["h9"] = Platform.HIMAWARI_9
PLATFORM_MAP["goes16"] = Platform.GOES_16
PLATFORM_MAP["goes17"] = Platform.GOES_17
PLATFORM_MAP["goes18"] = Platform.GOES_18
PLATFORM_MAP["goes19"] = Platform.GOES_19
PLATFORM_MAP["met8"] = Platform.MSG1
PLATFORM_MAP["met9"] = Platform.MSG2
PLATFORM_MAP["met10"] = Platform.MSG3
PLATFORM_MAP["met11"] = Platform.MSG4


class Info(Enum):
    """
    Standard keys for info dictionaries
    Note: some fields correspond to database fields in workspace.metadatabase !
    """

    UNKNOWN = "???"
    # full path to the resource that the file came from
    # DEPRECATED since datasets may not have one-to-one pathname mapping
    PATHNAME = "path"

    # CF content
    SHORT_NAME = "short_name"  # CF short_name
    LONG_NAME = "long_name"  # CF long_name
    STANDARD_NAME = "standard_name"  # CF compliant standard_name (when possible)
    UNITS = "units"  # CF compliant (udunits compliant) units string, original data units

    # SIFT bookkeeping
    DATASET_NAME = "dataset_name"  # logical name of the file (possibly human assigned)
    KIND = "kind"  # Kind enumeration on what kind of layer/dataset this makes
    UUID = "uuid"  # UUID assigned on import, which follows the layer/dataset around the system

    # track determiner is family::category; presentation is determined by family
    # family::category::serial is a unique identifier equivalent to conventional make-model-serialnumber
    # string representing data family, typically instrument:measurement:wavelength but may vary by data content
    FAMILY = "family"
    CATEGORY = "category"  # string with platform:instrument:target typically but may vary by data content
    SERIAL = "serial"  # serial number

    # projection information
    ORIGIN_X = "origin_x"
    ORIGIN_Y = "origin_y"
    CELL_WIDTH = "cell_width"
    CELL_HEIGHT = "cell_height"
    PROJ = "proj4"

    # original data grid layout
    GRID_ORIGIN = "grid_layout_origin"
    GRID_FIRST_INDEX_X = "grid_first_index_x"
    GRID_FIRST_INDEX_Y = "grid_first_index_y"

    # colormap amd data range
    CLIM = "clim"  # (min,max) color map limits
    VALID_RANGE = "valid_range"
    SHAPE = "shape"  # (rows, columns) or (rows, columns, levels) data shape
    COLORMAP = "colormap"  # name or UUID of a color map

    # glyph (marker) or line styling
    STYLE = "style"  # name (or FUTURE: UUID?) of a SVG/HTML-like style

    SCHED_TIME = "timeline"  # scheduled time for observation
    OBS_TIME = "obstime"  # actual time for observation
    OBS_DURATION = "obsduration"  # time from start of observation to end of observation

    # instrument and scene information
    PLATFORM = "platform"  # full standard name of spacecraft
    SCENE = "scene"  # standard scene identifier string for instrument, e.g. FLDK
    INSTRUMENT = "instrument"  # Instrument enumeration, or string with full standard name

    # human-friendly conventions
    DISPLAY_NAME = "display_name"  # preferred name for displaying
    DISPLAY_TIME = "display_time"  # typically from guidebook, used for labeling animation frame
    UNIT_CONVERSION = "unit_conversion"  # (preferred CF units, convert_func, format_func)
    # unit numeric conversion func: lambda x, inverse=False: convert-to-units
    # unit string format func: lambda x, numeric=True, units=True: formatted string
    CENTRAL_WAVELENGTH = "nominal_wavelength"
    # only in family info dictionaries:
    DISPLAY_FAMILY = "display_family"

    # only used by algebraic datasets to save the operation kind
    ALGEBRAIC = "algebraic"

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


@dataclass
class Presentation:
    """Presentation information for a layer and dataset."""

    uuid: Optional[UUID]  # dataset in the layermodel/document/workspace, None if referring to a system layer
    kind: Kind  # what kind of layer/dataset it is
    visible: bool = True  # whether it's visible or not
    colormap: object = None  # name or uuid: color map to use; name for default, uuid for user-specified
    style: object = None  # name or uuid: SVG/HTML style to use; name for default, (FUTURE?) uuid for user-specified
    climits: Union[tuple, list] = INVALID_COLOR_LIMITS  # valid min and valid max used for color mapping normalization
    gamma: Union[float, List[float]] = 1.0  # valid (0 to 5) for gamma correction
    opacity: float = 1.0
