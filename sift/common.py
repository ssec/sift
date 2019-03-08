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

import numpy as np

__author__ = 'rayg'
__docformat__ = 'reStructuredText'

import os
import sys
from datetime import datetime, timedelta
import logging
from numba import jit, float64, int64, boolean, types as nb_types
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


INSTRUMENT_MAP = {v.value.lower().replace('-', ''): v for v in Instrument}


class Platform(Enum):
    UNKNOWN = '???'
    HIMAWARI_8 = 'Himawari-8'
    HIMAWARI_9 = 'Himawari-9'
    GOES_16 = 'G16'
    GOES_17 = 'G17'
    NWP = 'NWP'
    MSG8 = 'Meteosat-8'
    MSG9 = 'Meteosat-9'
    MSG10 = 'Meteosat-10'
    MSG11 = 'Meteosat-11'


PLATFORM_MAP = {v.value.lower().replace('-', ''): v for v in Platform}
PLATFORM_MAP['H8'] = Platform.HIMAWARI_8
PLATFORM_MAP['H9'] = Platform.HIMAWARI_8
PLATFORM_MAP['GOES-16'] = Platform.GOES_16
PLATFORM_MAP['GOES-17'] = Platform.GOES_17


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
    BAND = 'band'  # band number (multispectral instruments)
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


@jit(nb_types.UniTuple(int64, 2)(float64[:, :], float64[:, :]))
def get_reference_points(img_cmesh, img_vbox):
    """Get two image reference point indexes.

    This function will return the two nearest reference points to the
    center of the viewed canvas. The first argument `img_cmesh` is all
    valid image mesh points that were successfully transformed to the
    view projection. The second argument `img_vbox` is these same mesh
    points, but in the original image projection units.

    :param img_cmesh: (N, 2) array of valid points across the image space
    :param img_vbox: (N, 2) array of valid points across the image space
    :return: (reference array index 1, reference array index 2)
    :raises: ValueError if not enough valid points to create
             two reference points
    """
    # Sort points by nearest to further from the 0,0 center of the canvas
    # Uses a cheap Pythagorean theorem by summing X + Y
    near_points = np.sum(np.abs(img_cmesh), axis=1).argsort()
    ref_idx_1 = near_points[0]
    # pick a second reference point that isn't in the same row or column as the first
    near_points_2 = near_points[~np.isclose(img_vbox[near_points][:, 0], img_vbox[ref_idx_1][0]) &
                                ~np.isclose(img_vbox[near_points][:, 1], img_vbox[ref_idx_1][1])]
    if near_points_2.shape[0] == 0:
        raise ValueError("Could not determine reference points")

    return ref_idx_1, near_points_2[0]


@jit(nb_types.UniTuple(int64, 2)(float64[:, :], boolean[:, :]))
def get_reference_points_image(img_dist, valid_mask):
    """Get two image reference point indexes close to image center.

    This function will return the two nearest reference points to the
    center of the image. The argument `img_vbox` is an array of points
    across the image that can be successfully projected to the viewed
    projection.

    :param img_dist: (N, 2) array of distances from the image center
                     in image space for points that can be projected
                     to the viewed projection
    :return: (reference array index 1, reference array index 2)
    :raises: ValueError if not enough valid points to create
             two reference points
    """
    # Sort points by nearest to further from the 0,0 center of the canvas
    # Uses a cheap Pythagorean theorem by summing X + Y
    near_points = np.sum(np.abs(img_dist), axis=1)
    near_points[~valid_mask] = np.inf
    near_points = near_points.argsort()
    ref_idx_1 = near_points[0]
    if np.isinf(near_points[ref_idx_1]):
        raise ValueError("Could not determine reference points")
    # pick a second reference point that isn't in the same row or column as the first
    near_points_2 = near_points[~np.isclose(img_dist[near_points][:, 0], img_dist[ref_idx_1][0]) &
                                ~np.isclose(img_dist[near_points][:, 1], img_dist[ref_idx_1][1])]
    if near_points_2.shape[0] == 0:
        raise ValueError("Could not determine reference points")

    return ref_idx_1, near_points_2[0]


@jit(nb_types.UniTuple(float64, 2)(float64, float64, int64, float64), nopython=True)
def _calc_extent_component(canvas_point, image_point, num_pixels, meters_per_pixel):
    """Calculate """
    # Find the distance in image space between the closest
    # reference point and the center of the canvas view (0, 0)
    # divide canvas_point coordinate by 2 to get the ratio of that distance to the entire canvas view (-1 to 1)
    viewed_img_center_shift_x = (canvas_point / 2. * num_pixels * meters_per_pixel)
    # Find the theoretical center of the canvas in image space (X/Y)
    viewed_img_center_x = image_point - viewed_img_center_shift_x
    # Find the theoretical number of image units (meters) that
    # would cover an entire canvas in a perfect world
    half_canvas_width = num_pixels * meters_per_pixel / 2.
    # Calculate the theoretical bounding box if the image was
    # perfectly centered on the closest reference point
    # Clip the bounding box to the extents of the image
    left = viewed_img_center_x - half_canvas_width
    right = viewed_img_center_x + half_canvas_width
    return left, right


@jit(nb_types.UniTuple(float64, 2)(float64[:, :], float64[:, :], nb_types.UniTuple(int64, 2)), nopython=True)
def calc_pixel_size(canvas_point, image_point, canvas_size):
    # Calculate the number of image meters per display pixel
    # That is, use the ratio of the distance in canvas space
    # between two points to the distance of the canvas
    # (1 - (-1) = 2). Use this ratio to calculate number of
    # screen pixels between the two reference points. Then
    # determine how many image units cover that number of pixels.
    dx = abs((image_point[1, 0] - image_point[0, 0]) /
             (canvas_size[0] * (canvas_point[1, 0] - canvas_point[0, 0]) / 2.))
    dy = abs((image_point[1, 1] - image_point[0, 1]) /
             (canvas_size[1] * (canvas_point[1, 1] - canvas_point[0, 1]) / 2.))
    return dx, dy


@jit(nopython=True)
def clip(v, n, x):
    return max(min(v, x), n)


@jit(nb_types.NamedUniTuple(float64, 4, Box)(
    nb_types.NamedUniTuple(float64, 4, Box),
    nb_types.Array(float64, 1, 'C'),
    nb_types.Array(float64, 1, 'C'),
    nb_types.UniTuple(int64, 2),
    float64,
    float64
),
    nopython=True)
def calc_view_extents(image_extents_box: Box, canvas_point, image_point, canvas_size, dx, dy) -> Box:
    left, right = _calc_extent_component(canvas_point[0], image_point[0], canvas_size[0], dx)
    left = clip(left, image_extents_box.left, image_extents_box.right)
    right = clip(right, image_extents_box.left, image_extents_box.right)

    bot, top = _calc_extent_component(canvas_point[1], image_point[1], canvas_size[1], dy)
    bot = clip(bot, image_extents_box.bottom, image_extents_box.top)
    top = clip(top, image_extents_box.bottom, image_extents_box.top)

    if (right - left) < CANVAS_EXTENTS_EPSILON or (top - bot) < CANVAS_EXTENTS_EPSILON:
        # they are viewing essentially nothing or the image isn't in view
        raise ValueError("Image can't be currently viewed")

    return Box(left=left, right=right, bottom=bot, top=top)


@jit(nb_types.UniTuple(float64, 2)(
    nb_types.NamedUniTuple(int64, 2, Point),
    nb_types.NamedUniTuple(int64, 2, Point),
    nb_types.NamedUniTuple(int64, 2, Point)
),
    nopython=True)
def max_tiles_available(image_shape, tile_shape, stride):
    ath = (image_shape[0] / float(stride[0])) / tile_shape[0]
    atw = (image_shape[1] / float(stride[1])) / tile_shape[1]
    return ath, atw


# @jit(nb_types.NamedUniTuple(int64, 4, Box)(
#         nb_types.NamedUniTuple(float64, 2, Resolution),
#         nb_types.NamedUniTuple(float64, 2, Resolution),
#         nb_types.NamedUniTuple(float64, 2, Point),
#         nb_types.NamedUniTuple(int64, 2, Point),
#         nb_types.NamedUniTuple(int64, 2, Point),
#         nb_types.NamedUniTuple(float64, 6, ViewBox),
#         nb_types.NamedUniTuple(int64, 2, Point),
#         nb_types.NamedUniTuple(int64, 4, Box)
#     ),
#      nopython=True)
@jit(nopython=True)
def visible_tiles(pixel_rez: Resolution,
                  tile_size: Resolution,
                  image_center: Point,
                  image_shape: Point,
                  tile_shape: Point,
                  visible_geom: ViewBox, stride: Point, extra_tiles_box: Box):
    """
    given a visible world geometry and sampling, return (sampling-state, [Box-of-tiles-to-draw])
    sampling state is WELLSAMPLED/OVERSAMPLED/UNDERSAMPLED
    returned Box should be iterated per standard start:stop style
    tiles are specified as (iy,ix) integer pairs
    extra_box value says how many extra tiles to include around each edge
    """
    V = visible_geom
    X = extra_tiles_box  # FUTURE: extra_geom_box specifies in world coordinates instead of tile count
    Z = pixel_rez
    tile_size = Resolution(tile_size.dy * stride[0], tile_size.dx * stride[1])
    # should be the upper-left corner of the tile centered on the center of the image
    to = Point(image_center[0] + tile_size.dy / 2.,
               image_center[1] - tile_size.dx / 2.)  # tile origin

    # number of data pixels between view edge and originpoint
    pv = Box(
        bottom=(V.bottom - to.y) / -(Z.dy * stride[0]),
        top=(V.top - to.y) / -(Z.dy * stride[0]),
        left=(V.left - to.x) / (Z.dx * stride[1]),
        right=(V.right - to.x) / (Z.dx * stride[1])
    )

    th, tw = tile_shape
    # first tile we'll need is (tiy0, tix0)
    # floor to make sure we get the upper-left of the theoretical tile
    tiy0 = np.floor(pv.top / th)
    tix0 = np.floor(pv.left / tw)
    # number of tiles wide and high we'll absolutely need
    # add 0.5 and ceil to make sure we include all possible tiles
    # NOTE: output r and b values are exclusive, l and t are inclusive
    nth = np.ceil((pv.bottom - tiy0 * th) / th + 0.5)
    ntw = np.ceil((pv.right - tix0 * tw) / tw + 0.5)

    # now add the extras
    if X.bottom > 0:
        nth += int(X.bottom)
    if X.left > 0:
        tix0 -= int(X.left)
        ntw += int(X.left)
    if X.top > 0:
        tiy0 -= int(X.top)
        nth += int(X.top)
    if X.right > 0:
        ntw += int(X.right)

    # Total number of tiles in this image at this stride (could be fractional)
    ath, atw = max_tiles_available(image_shape, tile_shape, stride)
    # truncate to the available tiles
    hw = atw / 2.
    hh = ath / 2.
    # center tile is half pixel off because we want center of the center
    # tile to be at the center of the image
    if tix0 < -hw + 0.5:
        ntw += hw - 0.5 + tix0
        tix0 = -hw + 0.5
    if tiy0 < -hh + 0.5:
        nth += hh - 0.5 + tiy0
        tiy0 = -hh + 0.5
    # add 0.5 to include the "end of the tile" since the r and b are exclusive
    if tix0 + ntw > hw + 0.5:
        ntw = hw + 0.5 - tix0
    if tiy0 + nth > hh + 0.5:
        nth = hh + 0.5 - tiy0

    tilebox = Box(
        bottom=np.int64(np.ceil(tiy0 + nth)),
        left=np.int64(np.floor(tix0)),
        top=np.int64(np.floor(tiy0)),
        right=np.int64(np.ceil(tix0 + ntw)),
    )
    return tilebox


class TileCalculator(object):
    """Common calculations for geographic image tile groups in an array or file

    Tiles are identified by (iy,ix) zero-based indicators.

    """
    OVERSAMPLED = 'oversampled'
    UNDERSAMPLED = 'undersampled'
    WELLSAMPLED = 'wellsampled'

    name = None
    image_shape = None
    pixel_rez = None
    zero_point = None
    tile_shape = None
    # derived
    image_extents_box = None  # word coordinates that this image and its tiles corresponds to
    tiles_avail = None  # (ny,nx) available tile count for this image

    def __init__(self, name, image_shape, ul_origin, pixel_rez,
                 tile_shape=(DEFAULT_TILE_HEIGHT, DEFAULT_TILE_WIDTH),
                 texture_shape=(DEFAULT_TEXTURE_HEIGHT, DEFAULT_TEXTURE_WIDTH),
                 projection=DEFAULT_PROJECTION,
                 wrap_lon=False):
        """Initialize numbers used by multiple calculations.

        Args:
            name (str): the 'name' of the tile, typically the path of the file it represents
            image_shape (int, int): (height, width) in pixels
            ul_origin (float, float): (y, x) in world coords specifies upper-left coordinate of the image
            pixel_rez (float, float): (dy, dx) in world coords per pixel ascending from corner [0,0],
                as measured near zero_point
            tile_shape (int, int): the pixel dimensions (h:int, w:int) of the GPU tiling we want to use
            texture_shape (int, int): the size of the texture being used (h, w) in number of tiles

        Notes:

            - Tiling is aligned to pixels, not world
            - World coordinates are eqm such that 0,0 matches 0°N 0°E, going north/south +-90° and west/east +-180°
            - Data coordinates are pixels with b l or b r corner being 0,0

        """
        super(TileCalculator, self).__init__()
        self.name = name
        self.image_shape = Point(np.int64(image_shape[0]), np.int64(image_shape[1]))
        self.ul_origin = Point(*ul_origin)
        self.pixel_rez = Resolution(np.float64(pixel_rez[0]), np.float64(pixel_rez[1]))
        self.tile_shape = Point(np.int64(tile_shape[0]), np.int64(tile_shape[1]))
        # in units of tiles:
        self.texture_shape = texture_shape
        # in units of data elements (float32):
        self.texture_size = (self.texture_shape[0] * self.tile_shape[0], self.texture_shape[1] * self.tile_shape[1])
        self.image_tiles_avail = (self.image_shape[0] / self.tile_shape[0], self.image_shape[1] / self.tile_shape[1])
        self.wrap_lon = wrap_lon

        self.proj = Proj(projection)
        self.image_extents_box = e = Box(
            bottom=np.float64(self.ul_origin[0] - self.image_shape[0] * self.pixel_rez.dy),
            top=np.float64(self.ul_origin[0]),
            left=np.float64(self.ul_origin[1]),
            right=np.float64(self.ul_origin[1] + self.image_shape[1] * self.pixel_rez.dx),
        )
        # Array of points across the image space to be used as an estimate of image coverage
        # Used when checking if the image is viewable on the current canvas's projection
        self.image_mesh = np.meshgrid(np.linspace(e.left, e.right, IMAGE_MESH_SIZE),
                                      np.linspace(e.bottom, e.top, IMAGE_MESH_SIZE))
        self.image_mesh = np.column_stack((self.image_mesh[0].ravel(), self.image_mesh[1].ravel(),))
        self.image_center = Point(self.ul_origin.y - self.image_shape[0] / 2. * self.pixel_rez.dy,
                                  self.ul_origin.x + self.image_shape[1] / 2. * self.pixel_rez.dx)
        # size of tile in image projection
        self.tile_size = Resolution(self.pixel_rez.dy * self.tile_shape[0], self.pixel_rez.dx * self.tile_shape[1])
        self.overview_stride = self.calc_overview_stride()

    def visible_tiles(self, visible_geom, stride=Point(1, 1), extra_tiles_box=Box(0, 0, 0, 0)) -> Box:
        v = visible_geom
        e = extra_tiles_box
        return visible_tiles(
            Resolution(np.float64(self.pixel_rez[0]), np.float64(self.pixel_rez[1])),
            Resolution(np.float64(self.tile_size[0]), np.float64(self.tile_size[1])),
            Point(np.float64(self.image_center[0]), np.float64(self.image_center[1])),
            Point(np.int64(self.image_shape[0]), np.int64(self.image_shape[1])),
            Point(np.int64(self.tile_shape[0]), np.int64(self.tile_shape[1])),
            ViewBox(
                np.float64(v[0]), np.float64(v[1]),
                np.float64(v[2]), np.float64(v[3]),
                np.float64(v[4]), np.float64(v[5]),
            ),
            Point(np.int64(stride[0]), np.int64(stride[1])),
            Box(
                np.int64(e[0]), np.int64(e[1]),
                np.int64(e[2]), np.int64(e[3])
            ))

    @jit
    def calc_tile_slice(self, tiy, tix, stride):
        """Calculate the slice needed to get data.

        The returned slice assumes the original image data has already
        been reduced by the provided stride.

        Args:
            tiy (int): Tile Y index (down is positive)
            tix (int): Tile X index (right is positive)
            stride (tuple): (Original data Y-stride, Original data X-stride)

        """
        y_offset = int(self.image_shape[0] / 2. / stride[0] - self.tile_shape[0] / 2.)
        y_start = int(tiy * self.tile_shape[0] + y_offset)
        if y_start < 0:
            row_slice = slice(0, max(0, y_start + self.tile_shape[0]), 1)
        else:
            row_slice = slice(y_start, y_start + self.tile_shape[0], 1)

        x_offset = int(self.image_shape[1] / 2. / stride[1] - self.tile_shape[1] / 2.)
        x_start = int(tix * self.tile_shape[1] + x_offset)
        if x_start < 0:
            col_slice = slice(0, max(0, x_start + self.tile_shape[1]), 1)
        else:
            col_slice = slice(x_start, x_start + self.tile_shape[1], 1)
        return row_slice, col_slice

    @jit
    def calc_tile_fraction(self, tiy, tix, stride):
        """Calculate the fractional components of the specified tile

        Returns:
            (factor, offset): Two `Resolution` objects stating the relative size
                              of the tile compared to a whole tile and the
                              offset from the origin of a whole tile.
        """
        mt = max_tiles_available(self.image_shape, self.tile_shape, stride)

        if tix < -mt[1] / 2. + 0.5:
            # left edge tile
            offset_x = -mt[1] / 2. + 0.5 - tix
            factor_x = 1 - offset_x
        elif mt[1] / 2. + 0.5 - tix < 1:
            # right edge tile
            offset_x = 0.
            factor_x = mt[1] / 2. + 0.5 - tix
        else:
            # full tile
            offset_x = 0.
            factor_x = 1.

        if tiy < -mt[0] / 2. + 0.5:
            # left edge tile
            offset_y = -mt[0] / 2. + 0.5 - tiy
            factor_y = 1 - offset_y
        elif mt[0] / 2. + 0.5 - tiy < 1:
            # right edge tile
            offset_y = 0.
            factor_y = mt[0] / 2. + 0.5 - tiy
        else:
            # full tile
            offset_y = 0.
            factor_y = 1.

        factor_rez = Resolution(dy=factor_y, dx=factor_x)
        offset_rez = Resolution(dy=offset_y, dx=offset_x)
        return factor_rez, offset_rez

    @jit
    def calc_stride(self, visible, texture=None):
        """
        given world geometry and sampling as a ViewBox or Resolution tuple
        calculate a conservative stride value for rendering a set of tiles
        :param visible: ViewBox or Resolution with world pixels per screen pixel
        :param texture: ViewBox or Resolution with texture resolution as world pixels per screen pixel
        """
        # screen dy,dx in world distance per pixel
        # world distance per pixel for our data
        # compute texture pixels per screen pixels
        texture = texture or self.pixel_rez
        tsy = min(self.overview_stride[0].step,
                  max(1, np.ceil(visible.dy * PREFERRED_SCREEN_TO_TEXTURE_RATIO / texture.dy)))
        tsx = min(self.overview_stride[1].step,
                  max(1, np.ceil(visible.dx * PREFERRED_SCREEN_TO_TEXTURE_RATIO / texture.dx)))
        return Point(np.int64(tsy), np.int64(tsx))

    @jit
    def calc_overview_stride(self, image_shape=None):
        image_shape = image_shape or self.image_shape
        # FUTURE: Come up with a fancier way of doing overviews like averaging each strided section, if needed
        tsy = max(1, int(np.floor(image_shape[0] / self.tile_shape[0])))
        tsx = max(1, int(np.floor(image_shape[1] / self.tile_shape[1])))
        y_slice = slice(0, image_shape[0], tsy)
        x_slice = slice(0, image_shape[1], tsx)
        return y_slice, x_slice

    @jit
    def calc_vertex_coordinates(self, tiy, tix, stridey, stridex,
                                factor_rez, offset_rez, tessellation_level=1):
        quad = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0],
                         [0, 0, 0], [1, 1, 0], [0, 1, 0]],
                        dtype=np.float32)
        quads = np.tile(quad, (tessellation_level * tessellation_level, 1))
        tile_w = self.pixel_rez.dx * self.tile_shape[1] * stridex
        tile_h = self.pixel_rez.dy * self.tile_shape[0] * stridey
        origin_x = self.image_center[1] - tile_w / 2.
        origin_y = self.image_center[0] + tile_h / 2.
        for x_idx in range(tessellation_level):
            for y_idx in range(tessellation_level):
                start_idx = x_idx * tessellation_level + y_idx
                quads[start_idx * 6:(start_idx + 1) * 6, 0] *= tile_w * factor_rez.dx / tessellation_level
                quads[start_idx * 6:(start_idx + 1) * 6, 0] += origin_x + tile_w * (
                    tix + offset_rez.dx + factor_rez.dx * x_idx / tessellation_level)
                # Origin is upper-left so image goes down
                quads[start_idx * 6:(start_idx + 1) * 6, 1] *= -tile_h * factor_rez.dy / tessellation_level
                quads[start_idx * 6:(start_idx + 1) * 6, 1] += origin_y - tile_h * (
                    tiy + offset_rez.dy + factor_rez.dy * y_idx / tessellation_level)
        quads = quads.reshape(tessellation_level * tessellation_level * 6, 3)
        return quads[:, :2]

    @jit
    def calc_texture_coordinates(self, ttile_idx, factor_rez, offset_rez, tessellation_level=1):
        """Get texture coordinates for one tile as a quad.

        :param ttile_idx: int, texture 1D index that maps to some internal texture tile location
        """
        tiy = int(ttile_idx / self.texture_shape[1])
        tix = ttile_idx % self.texture_shape[1]
        # start with basic quad describing the entire texture
        quad = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0],
                         [0, 0, 0], [1, 1, 0], [0, 1, 0]],
                        dtype=np.float32)
        quads = np.tile(quad, (tessellation_level * tessellation_level, 1))
        # Now scale and translate the coordinates so they only apply to one tile in the texture
        one_tile_tex_width = 1.0 / self.texture_size[1] * self.tile_shape[1]
        one_tile_tex_height = 1.0 / self.texture_size[0] * self.tile_shape[0]
        for x_idx in range(tessellation_level):
            for y_idx in range(tessellation_level):
                start_idx = x_idx * tessellation_level + y_idx
                # offset for this tile isn't needed because the data should
                # have been inserted as close to the top-left of the texture
                # location as possible
                quads[start_idx * 6:(start_idx + 1) * 6, 0] *= one_tile_tex_width * factor_rez.dx / tessellation_level
                quads[start_idx * 6:(start_idx + 1) * 6, 0] += one_tile_tex_width * (
                    tix + factor_rez.dx * x_idx / tessellation_level)
                quads[start_idx * 6:(start_idx + 1) * 6, 1] *= one_tile_tex_height * factor_rez.dy / tessellation_level
                quads[start_idx * 6:(start_idx + 1) * 6, 1] += one_tile_tex_height * (
                    tiy + factor_rez.dy * y_idx / tessellation_level)
        quads = quads.reshape(6 * tessellation_level * tessellation_level, 3)
        quads = np.ascontiguousarray(quads[:, :2])
        return quads

    def calc_view_extents(self, canvas_point, image_point, canvas_size, dx, dy):
        return calc_view_extents(self.image_extents_box, canvas_point, image_point, canvas_size, dx, dy)


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
