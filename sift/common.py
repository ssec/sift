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
from collections import namedtuple
import numpy as np
from enum import Enum

__author__ = 'rayg'
__docformat__ = 'reStructuredText'

import os, sys
import logging, unittest, argparse
from numba import jit, float64, int64, uint64, boolean, types as nb_types
from pyproj import Proj

LOG = logging.getLogger(__name__)

# if sys.platform.startswith("win"):
PREFERRED_SCREEN_TO_TEXTURE_RATIO = 1.0  # screenpx:texturepx that we want to keep, ideally, by striding
# else:
#     PREFERRED_SCREEN_TO_TEXTURE_RATIO = 0.5  # screenpx:texturepx that we want to keep, ideally, by striding

# http://home.online.no/~sigurdhu/WGS84_Eng.html

DEFAULT_TILE_HEIGHT = 512
DEFAULT_TILE_WIDTH = 512
DEFAULT_TEXTURE_HEIGHT=2
DEFAULT_TEXTURE_WIDTH=16
DEFAULT_ANIMATION_DELAY=100.0  # milliseconds
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
MAX_EXCURSION_Y = C_POL/2.0
MAX_EXCURSION_X = C_EQ/2.0
# how many 'tessellation' tiles in one texture tile? 2 = 2 rows x 2 cols
TESS_LEVEL = 20
IMAGE_MESH_SIZE = 10
# smallest difference between two image extents (in canvas units)
# before the image is considered "out of view"
CANVAS_EXTENTS_EPSILON = 1e-4

#R_EQ = 6378.1370  # km
#R_POL = 6356.7523142  # km
#C_EQ = 40075.0170  # linear km
#C_POL = 40007.8630  # linear km

# MAX_EXCURSION_Y = C_POL/4.0
# MAX_EXCURSION_X = C_EQ/2.0

box = namedtuple('box', ('b', 'l', 't', 'r'))  # bottom, left, top, right
rez = namedtuple('rez', ('dy', 'dx'))  # world km / pixel distance
pnt = namedtuple('pnt', ('y', 'x'))
geo = namedtuple('geo', ('n', 'e'))  # lat N, lon E
vue = namedtuple('vue', ('b', 'l', 't', 'r', 'dy', 'dx'))  # combination of box + rez

WORLD_EXTENT_BOX = box(b=-MAX_EXCURSION_Y, l=-MAX_EXCURSION_X, t=MAX_EXCURSION_Y, r=MAX_EXCURSION_X)


class TOOL(Enum):
    """Names for cursor tools.
    """
    PAN_ZOOM = "pan_zoom"
    POINT_PROBE = "point_probe"
    REGION_PROBE = "region_probe"


class KIND(Enum):
    """kind of entities we're working with
    """
    UNKNOWN = 0
    IMAGE = 1
    OUTLINE = 2
    SHAPE = 3
    RGB = 4
    COMPOSITE = 5


class COMPOSITE_TYPE(Enum):
    """Type of non-luminance image layers
    """
    RGB = 1
    ARITHMETIC = 2


class INSTRUMENT(Enum):
    UNKNOWN = '???'
    AHI = 'AHI'
    ABI = 'ABI'
    AMI = 'AMI'

    @classmethod
    def from_value(cls, value_str):
        """Convert external string to Enum"""
        for m in cls:
            if m.value == value_str:
                return m
        return INSTRUMENT.UNKNOWN


class PLATFORM(Enum):
    UNKNOWN = '???'
    HIMAWARI_8 = 'Himawari-8'
    HIMAWARI_9 = 'Himawari-9'
    GOES_16 = 'G16'
    GOES_17 = 'G17'

    @classmethod
    def from_value(cls, value_str):
        """Convert external string to Enum"""
        for m in cls:
            if m.value == value_str:
                return m
        return INSTRUMENT.UNKNOWN


class INFO(Enum):
    """
    Standard keys for info dictionaries
    Note: some fields correspond to database fields in workspace.metadatabase !
    """

    PATHNAME = 'path'  # full path to the data file
    DATASET_NAME = 'dataset_name'  # logical name of the file (possibly human assigned)
    SHORT_NAME = 'short_name'  # CF short_name
    LONG_NAME = 'long_name'  # CF long_name
    STANDARD_NAME = 'standard_name'  # CF compliant standard_name (when possible)
    UNITS = 'units'  # CF compliant (udunits compliant) units string, original data units
    KIND = 'kind'  # KIND enumeration on what kind of layer this makes
    UUID = 'uuid'  # UUID assigned on import, which follows the layer around the system
    ORIGIN_X = 'origin_x'
    ORIGIN_Y = 'origin_y'
    CELL_WIDTH = 'cell_width'
    CELL_HEIGHT = 'cell_height'
    PROJ = 'proj4'
    CLIM = 'clim'  # (min,max) color map limits
    SHAPE = 'shape' # (rows, columns) or (rows, columns, levels) data shape
    COLORMAP = 'colormap'  # name or UUID of a color map
    DISPLAY_TIME = 'display_time'  # typically from guidebook, used for labeling animation frame
    # Previously in the GUIDE Enum:
    PLATFORM = 'platform' # full standard name of spacecraft
    SCHED_TIME = 'timeline'  # scheduled time for observation
    OBS_TIME = 'obstime'  # actual time for observation
    OBS_DURATION = 'obsduration'  # time from start of observation to end of observation
    BAND = 'band'  # band number (multispectral instruments)
    SCENE = 'scene'  # standard scene identifier string for instrument, e.g. FLDK
    INSTRUMENT = 'instrument'  # INSTRUMENT enumeration, or string with full standard name
    DISPLAY_NAME = 'display_name'  # preferred name in the layer list
    UNIT_CONVERSION = 'unit_conversion'  # (preferred CF units, convert_func, format_func)
    # unit numeric conversion func: lambda x, inverse=False: convert-to-units
    # unit string format func: lambda x, numeric=True, units=True: formatted string
    CENTRAL_WAVELENGTH = 'nominal_wavelength'


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
    print(near_points.shape, valid_mask.shape)
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
    l = viewed_img_center_x - half_canvas_width
    r = viewed_img_center_x + half_canvas_width
    return l, r


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


@jit(nb_types.NamedUniTuple(float64, 4, box)(
        nb_types.NamedUniTuple(float64, 4, box),
        nb_types.Array(float64, 1, 'C'),
        nb_types.Array(float64, 1, 'C'),
        nb_types.UniTuple(int64, 2),
        float64,
        float64
    ),
    nopython=True)
def calc_view_extents(image_extents_box, canvas_point, image_point, canvas_size, dx, dy):
    l, r = _calc_extent_component(canvas_point[0], image_point[0], canvas_size[0], dx)
    l = clip(l, image_extents_box.l, image_extents_box.r)
    r = clip(r, image_extents_box.l, image_extents_box.r)

    b, t = _calc_extent_component(canvas_point[1], image_point[1], canvas_size[1], dy)
    b = clip(b, image_extents_box.b, image_extents_box.t)
    t = clip(t, image_extents_box.b, image_extents_box.t)

    if (r - l) < CANVAS_EXTENTS_EPSILON or (t - b) < CANVAS_EXTENTS_EPSILON:
        # they are viewing essentially nothing or the image isn't in view
        raise ValueError("Image can't be currently viewed")

    return box(l=l, r=r, b=b, t=t)


@jit(nb_types.UniTuple(float64, 2)(
        nb_types.NamedUniTuple(int64, 2, pnt),
        nb_types.NamedUniTuple(int64, 2, pnt),
        nb_types.NamedUniTuple(int64, 2, pnt)
    ),
     nopython=True)
def max_tiles_available(image_shape, tile_shape, stride):
    ath = (image_shape[0] / float(stride[0])) / tile_shape[0]
    atw = (image_shape[1] / float(stride[1])) / tile_shape[1]
    return ath, atw


# @jit(nb_types.NamedUniTuple(int64, 4, box)(
#         nb_types.NamedUniTuple(float64, 2, rez),
#         nb_types.NamedUniTuple(float64, 2, rez),
#         nb_types.NamedUniTuple(float64, 2, pnt),
#         nb_types.NamedUniTuple(int64, 2, pnt),
#         nb_types.NamedUniTuple(int64, 2, pnt),
#         nb_types.NamedUniTuple(float64, 6, vue),
#         nb_types.NamedUniTuple(int64, 2, pnt),
#         nb_types.NamedUniTuple(int64, 4, box)
#     ),
#      nopython=True)
@jit(nopython=True)
def visible_tiles(pixel_rez,
                  tile_size,
                  image_center,
                  image_shape,
                  tile_shape,
                  visible_geom, stride, extra_tiles_box):
    """
    given a visible world geometry and sampling, return (sampling-state, [box-of-tiles-to-draw])
    sampling state is WELLSAMPLED/OVERSAMPLED/UNDERSAMPLED
    returned box should be iterated per standard start:stop style
    tiles are specified as (iy,ix) integer pairs
    extra_box value says how many extra tiles to include around each edge
    """
    V = visible_geom
    X = extra_tiles_box  # FUTURE: extra_geom_box specifies in world coordinates instead of tile count
    Z = pixel_rez
    tile_size = rez(tile_size.dy * stride[0], tile_size.dx * stride[1])
    # should be the upper-left corner of the tile centered on the center of the image
    to = pnt(image_center[0] + tile_size.dy / 2.,
             image_center[1] - tile_size.dx / 2.)  # tile origin

    # number of data pixels between view edge and originpoint
    pv = box(
        b=(V.b - to.y) / -(Z.dy * stride[0]),
        t=(V.t - to.y) / -(Z.dy * stride[0]),
        l=(V.l - to.x) / (Z.dx * stride[1]),
        r=(V.r - to.x) / (Z.dx * stride[1])
    )

    th, tw = tile_shape
    # first tile we'll need is (tiy0, tix0)
    # floor to make sure we get the upper-left of the theoretical tile
    tiy0 = np.floor(pv.t / th)
    tix0 = np.floor(pv.l / tw)
    # number of tiles wide and high we'll absolutely need
    # add 0.5 and ceil to make sure we include all possible tiles
    # NOTE: output r and b values are exclusive, l and t are inclusive
    nth = np.ceil((pv.b - tiy0 * th) / th + 0.5)
    ntw = np.ceil((pv.r - tix0 * tw) / tw + 0.5)

    # now add the extras
    if X.b > 0:
        nth += int(X.b)
    if X.l > 0:
        tix0 -= int(X.l)
        ntw += int(X.l)
    if X.t > 0:
        tiy0 -= int(X.t)
        nth += int(X.t)
    if X.r > 0:
        ntw += int(X.r)

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

    tilebox = box(
        b=np.int64(np.ceil(tiy0 + nth)),
        l=np.int64(np.floor(tix0)),
        t=np.int64(np.floor(tiy0)),
        r=np.int64(np.ceil(tix0 + ntw)),
    )
    return tilebox


class TileCalculator(object):
    """
    common calculations for mercator tile groups in an array or file
    tiles are identified by (iy,ix) zero-based indicators
    """
    OVERSAMPLED='oversampled'
    UNDERSAMPLED='undersampled'
    WELLSAMPLED='wellsampled'

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
        """
        name: the 'name' of the tile, typically the path of the file it represents
        image_shape: (h:int,w:int) in pixels
        ul_origin: (y:float,x:float) in world coords specifies upper-left coordinate of the image
        pixel_rez: (dy:float,dx:float) in world coords per pixel ascending from corner [0,0], as measured near zero_point
        tile_shape: the pixel dimensions (h:int, w:int) of the GPU tiling we want to use
        texture_shape: the size of the texture being used (h:int, w:int) in number of tiles

        Tiling is aligned to pixels, not world
        World coordinates are eqm such that 0,0 matches 0°N 0°E, going north/south +-90° and west/east +-180°
        Data coordinates are pixels with b l or b r corner being 0,0
        """
        super(TileCalculator, self).__init__()
        self.name = name
        self.image_shape = pnt(np.int64(image_shape[0]), np.int64(image_shape[1]))
        self.ul_origin = pnt(*ul_origin)
        self.pixel_rez = rez(np.float64(pixel_rez[0]), np.float64(pixel_rez[1]))
        self.tile_shape = pnt(np.int64(tile_shape[0]), np.int64(tile_shape[1]))
        # in units of tiles:
        self.texture_shape = texture_shape
        # in units of data elements (float32):
        self.texture_size = (self.texture_shape[0] * self.tile_shape[0], self.texture_shape[1] * self.tile_shape[1])
        self.image_tiles_avail = (self.image_shape[0] / self.tile_shape[0], self.image_shape[1] / self.tile_shape[1])
        self.wrap_lon = wrap_lon

        self.proj = Proj(projection)
        self.image_extents_box = e = box(
            b=np.float64(self.ul_origin[0] - self.image_shape[0] * self.pixel_rez.dy),
            t=np.float64(self.ul_origin[0]),
            l=np.float64(self.ul_origin[1]),
            r=np.float64(self.ul_origin[1] + self.image_shape[1] * self.pixel_rez.dx),
        )
        # Array of points across the image space to be used as an estimate of image coverage
        # Used when checking if the image is viewable on the current canvas's projection
        self.image_mesh = np.meshgrid(np.linspace(e.l, e.r, IMAGE_MESH_SIZE), np.linspace(e.b, e.t, IMAGE_MESH_SIZE))
        self.image_mesh = np.column_stack((self.image_mesh[0].ravel(), self.image_mesh[1].ravel(),))
        self.image_center = pnt(self.ul_origin.y - self.image_shape[0] / 2. * self.pixel_rez.dy,
                                self.ul_origin.x + self.image_shape[1] / 2. * self.pixel_rez.dx)
        # size of tile in image projection
        self.tile_size = rez(self.pixel_rez.dy * self.tile_shape[0], self.pixel_rez.dx * self.tile_shape[1])
        self.overview_stride = self.calc_overview_stride()

    def visible_tiles(self, visible_geom, stride=pnt(1, 1), extra_tiles_box=box(0, 0, 0, 0)):
        # return visible_tiles(self.pixel_rez,
        #                      self.tile_size,
        #                      self.image_center,
        #                      self.image_shape,
        #                      self.tile_shape,
        #                      visible_geom,
        #                      stride,
        #                      extra_tiles_box)
        v = visible_geom
        e = extra_tiles_box
        return visible_tiles(
            rez(np.float64(self.pixel_rez[0]), np.float64(self.pixel_rez[1])),
            rez(np.float64(self.tile_size[0]), np.float64(self.tile_size[1])),
            pnt(np.float64(self.image_center[0]), np.float64(self.image_center[1])),
            pnt(np.int64(self.image_shape[0]), np.int64(self.image_shape[1])),
            pnt(np.int64(self.tile_shape[0]), np.int64(self.tile_shape[1])),
            vue(
                np.float64(v[0]), np.float64(v[1]),
                np.float64(v[2]), np.float64(v[3]),
                np.float64(v[4]), np.float64(v[5]),
                ),
            pnt(np.int64(stride[0]), np.int64(stride[1])),
            box(
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
            (factor, offset): Two `rez` objects stating the relative size
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

        factor_rez = rez(dy=factor_y, dx=factor_x)
        offset_rez = rez(dy=offset_y, dx=offset_x)
        return factor_rez, offset_rez

    @jit
    def calc_stride(self, visible, texture=None):
        """
        given world geometry and sampling as a vue or rez tuple
        calculate a conservative stride value for rendering a set of tiles
        :param visible: vue or rez with world pixels per screen pixel
        :param texture: vue or rez with texture resolution as world pixels per screen pixel
        """
        # screen dy,dx in world distance per pixel
        # world distance per pixel for our data
        # compute texture pixels per screen pixels
        texture = texture or self.pixel_rez
        tsy = min(self.overview_stride[0].step, max(1, np.ceil(visible.dy * PREFERRED_SCREEN_TO_TEXTURE_RATIO / texture.dy)))
        tsx = min(self.overview_stride[1].step, max(1, np.ceil(visible.dx * PREFERRED_SCREEN_TO_TEXTURE_RATIO / texture.dx)))
        return pnt(np.int64(tsy), np.int64(tsx))

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
                quads[start_idx * 6:(start_idx + 1) * 6, 0] += origin_x + tile_w * (tix + offset_rez.dx + factor_rez.dx * x_idx / tessellation_level)
                quads[start_idx * 6:(start_idx + 1) * 6, 1] *= -tile_h * factor_rez.dy / tessellation_level  # Origin is upper-left so image goes down
                quads[start_idx * 6:(start_idx + 1) * 6, 1] += origin_y - tile_h * (tiy + offset_rez.dy + factor_rez.dy * y_idx / tessellation_level)
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
                quads[start_idx * 6:(start_idx + 1) * 6, 0] += one_tile_tex_width * (tix + factor_rez.dx * x_idx / tessellation_level)
                quads[start_idx * 6:(start_idx + 1) * 6, 1] *= one_tile_tex_height * factor_rez.dy / tessellation_level
                quads[start_idx * 6:(start_idx + 1) * 6, 1] += one_tile_tex_height * (tiy + factor_rez.dy * y_idx / tessellation_level)
        quads = quads.reshape(6 * tessellation_level * tessellation_level, 3)
        quads = np.ascontiguousarray(quads[:, :2])
        return quads

    def calc_view_extents(self, canvas_point, image_point, canvas_size, dx, dy):
        return calc_view_extents(self.image_extents_box, canvas_point, image_point, canvas_size, dx, dy)


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
