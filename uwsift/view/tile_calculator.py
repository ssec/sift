#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Supports calculations used throughout the library and application."""

import numpy as np
from numba import float32, float64, int64, jit
from numba import types as nb_types
from numba.extending import overload
from pyproj import Proj

from uwsift.common import (
    CANVAS_EXTENTS_EPSILON,
    DEFAULT_PROJECTION,
    DEFAULT_TEXTURE_HEIGHT,
    DEFAULT_TEXTURE_WIDTH,
    DEFAULT_TILE_HEIGHT,
    DEFAULT_TILE_WIDTH,
    IMAGE_MESH_SIZE,
    PREFERRED_SCREEN_TO_TEXTURE_RATIO,
    Box,
    IndexBox,
    Point,
    Resolution,
)


@overload(np.isclose)
def isclose(a, b):
    """Implementation of numpy.isclose() since it is currently not supported by numba."""

    def isclose_impl(a, b):
        atol = 1e-8
        rtol = 1e-5
        res = np.abs(a - b) <= (atol + rtol * np.abs(b))
        return res

    return isclose_impl


@jit(nb_types.UniTuple(int64, 2)(float64[:, :], float64[:, :]), nopython=True, cache=True, nogil=True)
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
    near_points_2 = near_points[
        ~np.isclose(img_vbox[near_points][:, 0], img_vbox[ref_idx_1][0])
        & ~np.isclose(img_vbox[near_points][:, 1], img_vbox[ref_idx_1][1])
    ]
    if near_points_2.shape[0] == 0:
        raise ValueError("Could not determine reference points")

    return ref_idx_1, near_points_2[0]


@jit(
    nb_types.UniTuple(float64, 2)(float64[:, :], float64[:, :], nb_types.UniTuple(int64, 2)),
    nopython=True,
    cache=True,
    nogil=True,
)
def calc_pixel_size(canvas_point, image_point, canvas_size):
    # Calculate the number of image meters per display pixel
    # That is, use the ratio of the distance in canvas space
    # between two points to the distance of the canvas
    # (1 - (-1) = 2). Use this ratio to calculate number of
    # screen pixels between the two reference points. Then
    # determine how many image units cover that number of pixels.
    dx = abs(
        (image_point[1, 0] - image_point[0, 0]) / (canvas_size[0] * (canvas_point[1, 0] - canvas_point[0, 0]) / 2.0)
    )
    dy = abs(
        (image_point[1, 1] - image_point[0, 1]) / (canvas_size[1] * (canvas_point[1, 1] - canvas_point[0, 1]) / 2.0)
    )
    return dx, dy


@jit(nb_types.UniTuple(float64, 2)(float64, float64, int64, float64), nopython=True, cache=True, nogil=True)
def _calc_extent_component(canvas_point, image_point, num_pixels, meters_per_pixel):
    """Calculate"""
    # Find the distance in image space between the closest
    # reference point and the center of the canvas view (0, 0)
    # divide canvas_point coordinate by 2 to get the ratio of that distance to the entire canvas view (-1 to 1)
    viewed_img_center_shift_x = canvas_point / 2.0 * num_pixels * meters_per_pixel
    # Find the theoretical center of the canvas in image space (X/Y)
    viewed_img_center_x = image_point - viewed_img_center_shift_x
    # Find the theoretical number of image units (meters) that
    # would cover an entire canvas in a perfect world
    half_canvas_width = num_pixels * meters_per_pixel / 2.0
    # Calculate the theoretical bounding box if the image was
    # perfectly centered on the closest reference point
    # Clip the bounding box to the extents of the image
    left = viewed_img_center_x - half_canvas_width
    right = viewed_img_center_x + half_canvas_width
    return left, right


@jit(float64(float64, float64, float64), nopython=True, cache=True, nogil=True)
def clip(v, n, x):
    return max(min(v, x), n)


@jit(
    nb_types.NamedUniTuple(float64, 4, Box)(
        nb_types.NamedUniTuple(float64, 4, Box),
        nb_types.Array(float64, 1, "C"),
        nb_types.Array(float64, 1, "C"),
        nb_types.UniTuple(int64, 2),
        float64,
        float64,
    ),
    nopython=True,
    cache=True,
    nogil=True,
)
def calc_view_extents(image_extents_box: Box, canvas_point, image_point, canvas_size, dx, dy) -> Box:
    left, right = _calc_extent_component(canvas_point[0], image_point[0], canvas_size[0], dx)
    left = clip(left, image_extents_box.left, image_extents_box.right)
    right = clip(right, image_extents_box.left, image_extents_box.right)

    bot, top = _calc_extent_component(canvas_point[1], image_point[1], canvas_size[1], dy)
    bot = clip(bot, image_extents_box.bottom, image_extents_box.top)
    top = clip(top, image_extents_box.bottom, image_extents_box.top)

    if (right - left) < CANVAS_EXTENTS_EPSILON or (top - bot) < CANVAS_EXTENTS_EPSILON:
        raise ValueError("Image is outside of canvas or empty")

    return Box(left=left, right=right, bottom=bot, top=top)


@jit(nb_types.UniTuple(float64, 2)(int64, int64, int64, int64, int64, int64), nopython=True, cache=True, nogil=True)
def max_tiles_available(image_shape_y, image_shape_x, tile_shape_y, tile_shape_x, stride_y, stride_x):
    ath = (image_shape_y / float(stride_y)) / tile_shape_y
    atw = (image_shape_x / float(stride_x)) / tile_shape_x
    return ath, atw


@jit(
    nb_types.NamedUniTuple(int64, 4, IndexBox)(
        float64,
        float64,
        float64,
        float64,
        float64,
        float64,
        int64,
        int64,
        int64,
        int64,
        float64,
        float64,
        float64,
        float64,
        float64,
        float64,
        int64,
        int64,
        int64,
        int64,
        int64,
        int64,
    ),
    nopython=True,
    cache=True,
    nogil=True,
)
def visible_tiles(
    z_dy,
    z_dx,
    tile_size_dy,
    tile_size_dx,
    image_center_y,
    image_center_x,
    image_shape_y,
    image_shape_x,
    tile_shape_y,
    tile_shape_x,
    v_bottom,
    v_left,
    v_top,
    v_right,
    v_dy,
    v_dx,
    stride_y,
    stride_x,
    x_bottom,
    x_left,
    x_top,
    x_right,
):
    tile_size = Resolution(tile_size_dy * stride_y, tile_size_dx * stride_x)
    # should be the upper-left corner of the tile centered on the center of the image
    to = Point(image_center_y + tile_size.dy / 2.0, image_center_x - tile_size.dx / 2.0)  # tile origin

    # number of data pixels between view edge and originpoint
    pv = Box(
        bottom=(v_bottom - to.y) / -(z_dy * stride_y),
        top=(v_top - to.y) / -(z_dy * stride_y),
        left=(v_left - to.x) / (z_dx * stride_x),
        right=(v_right - to.x) / (z_dx * stride_x),
    )

    th = tile_shape_y
    tw = tile_shape_x
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
    if x_bottom > 0:
        nth += int(x_bottom)
    if x_left > 0:
        tix0 -= int(x_left)
        ntw += int(x_left)
    if x_top > 0:
        tiy0 -= int(x_top)
        nth += int(x_top)
    if x_right > 0:
        ntw += int(x_right)

    # Total number of tiles in this image at this stride (could be fractional)
    ath, atw = max_tiles_available(image_shape_y, image_shape_x, tile_shape_y, tile_shape_x, stride_y, stride_x)
    # truncate to the available tiles
    hw = atw / 2.0
    hh = ath / 2.0
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

    tilebox = IndexBox(
        bottom=np.int64(np.ceil(tiy0 + nth)),
        left=np.int64(np.floor(tix0)),
        top=np.int64(np.floor(tiy0)),
        right=np.int64(np.ceil(tix0 + ntw)),
    )
    return tilebox


@jit(
    nb_types.UniTuple(nb_types.Tuple([int64, int64, int64]), 2)(
        int64, int64, int64, int64, nb_types.NamedUniTuple(int64, 2, Point), nb_types.NamedUniTuple(int64, 2, Point)
    ),
    nopython=True,
    cache=True,
    nogil=True,
)
def calc_tile_slice(tiy, tix, stride_y, stride_x, image_shape, tile_shape):
    y_offset = int(image_shape[0] / 2.0 / stride_y - tile_shape[0] / 2.0)
    y_start = int(tiy * tile_shape[0] + y_offset)
    if y_start < 0:
        row_slice = (0, max(0, y_start + tile_shape[0]), 1)
    else:
        row_slice = (y_start, y_start + tile_shape[0], 1)

    x_offset = int(image_shape[1] / 2.0 / stride_x - tile_shape[1] / 2.0)
    x_start = int(tix * tile_shape[1] + x_offset)
    if x_start < 0:
        col_slice = (0, max(0, x_start + tile_shape[1]), 1)
    else:
        col_slice = (x_start, x_start + tile_shape[1], 1)
    return row_slice, col_slice


@jit(
    nb_types.UniTuple(nb_types.NamedUniTuple(float64, 2, Resolution), 2)(
        int64, int64, int64, int64, int64, int64, int64, int64
    ),
    nopython=True,
    cache=True,
    nogil=True,
)
def calc_tile_fraction(tiy, tix, stride_y, stride_x, image_y, image_x, tile_y, tile_x):
    mt = max_tiles_available(image_y, image_x, tile_y, tile_x, stride_y, stride_x)

    if tix < -mt[1] / 2.0 + 0.5:
        # left edge tile
        offset_x = -mt[1] / 2.0 + 0.5 - tix
        factor_x = 1 - offset_x
    elif mt[1] / 2.0 + 0.5 - tix < 1:
        # right edge tile
        offset_x = 0.0
        factor_x = mt[1] / 2.0 + 0.5 - tix
    else:
        # full tile
        offset_x = 0.0
        factor_x = 1.0

    if tiy < -mt[0] / 2.0 + 0.5:
        # left edge tile
        offset_y = -mt[0] / 2.0 + 0.5 - tiy
        factor_y = 1 - offset_y
    elif mt[0] / 2.0 + 0.5 - tiy < 1:
        # right edge tile
        offset_y = 0.0
        factor_y = mt[0] / 2.0 + 0.5 - tiy
    else:
        # full tile
        offset_y = 0.0
        factor_y = 1.0

    factor_rez = Resolution(dy=factor_y, dx=factor_x)
    offset_rez = Resolution(dy=offset_y, dx=offset_x)
    return factor_rez, offset_rez


@jit(
    nb_types.NamedUniTuple(int64, 2, Point)(float64, float64, float64, float64, int64, int64),
    nopython=True,
    cache=True,
    nogil=True,
)
def calc_stride(v_dx, v_dy, t_dx, t_dy, overview_stride_y, overview_stride_x):
    # screen dy,dx in world distance per pixel
    # world distance per pixel for our data
    # compute texture pixels per screen pixels
    tsy = min(overview_stride_y, max(1, np.ceil(v_dy * PREFERRED_SCREEN_TO_TEXTURE_RATIO / t_dy)))
    tsx = min(overview_stride_x, max(1, np.ceil(v_dx * PREFERRED_SCREEN_TO_TEXTURE_RATIO / t_dx)))

    return Point(np.int64(tsy), np.int64(tsx))


@jit(
    nb_types.UniTuple(int64, 2)(int64, int64, nb_types.NamedUniTuple(int64, 2, Point)),
    nopython=True,
    cache=True,
    nogil=True,
)
def calc_overview_stride(image_shape_y, image_shape_x, tile_shape):
    tsy = max(1, int(np.floor(image_shape_y / tile_shape[0])))
    tsx = max(1, int(np.floor(image_shape_x / tile_shape[1])))
    return tsy, tsx


@jit(
    float32[:, :](
        int64,
        int64,
        int64,
        int64,
        float64,
        float64,
        float64,
        float64,
        int64,
        float64,
        float64,
        int64,
        int64,
        float64,
        float64,
        float32[:, :],
    ),
    nopython=True,
    cache=True,
    nogil=True,
)
def calc_vertex_coordinates(
    tiy,
    tix,
    stridey,
    stridex,
    factor_rez_dy,
    factor_rez_dx,
    offset_rez_dy,
    offset_rez_dx,
    tessellation_level,
    p_dx,
    p_dy,
    tile_shape_y,
    tile_shape_x,
    image_center_y,
    image_center_x,
    quads,
):
    tile_w = p_dx * tile_shape_x * stridex
    tile_h = p_dy * tile_shape_y * stridey
    origin_x = image_center_x - tile_w / 2.0
    origin_y = image_center_y + tile_h / 2.0
    for x_idx in range(tessellation_level):
        for y_idx in range(tessellation_level):
            start_idx = x_idx * tessellation_level + y_idx
            quads[start_idx * 6 : (start_idx + 1) * 6, 0] *= tile_w * factor_rez_dx / tessellation_level
            quads[start_idx * 6 : (start_idx + 1) * 6, 0] += origin_x + tile_w * (
                tix + offset_rez_dx + factor_rez_dx * x_idx / tessellation_level
            )
            # Origin is upper-left so image goes dow,n
            quads[start_idx * 6 : (start_idx + 1) * 6, 1] *= -tile_h * factor_rez_dy / tessellation_level
            quads[start_idx * 6 : (start_idx + 1) * 6, 1] += origin_y - tile_h * (
                tiy + offset_rez_dy + factor_rez_dy * y_idx / tessellation_level
            )
    return quads


@jit(
    float32[:, :](int64, int64, float64, float64, int64, int64, int64, int64, int64, float32[:, :]),
    nopython=True,
    cache=True,
    nogil=True,
)
def calc_texture_coordinates(
    tiy,
    tix,
    factor_rez_dy,
    factor_rez_dx,
    tessellation_level,
    texture_size_y,
    texture_size_x,
    tile_shape_y,
    tile_shape_x,
    quads,
):
    # Now scale and translate the coordinates so they only apply to one tile in the texture
    one_tile_tex_width = 1.0 / texture_size_x * tile_shape_x
    one_tile_tex_height = 1.0 / texture_size_y * tile_shape_y
    for x_idx in range(tessellation_level):
        for y_idx in range(tessellation_level):
            start_idx = x_idx * tessellation_level + y_idx
            # offset for this tile isn't needed because the data should
            # have been inserted as close to the top-left of the texture
            # location as possible
            quads[start_idx * 6 : (start_idx + 1) * 6, 0] *= one_tile_tex_width * factor_rez_dx / tessellation_level
            quads[start_idx * 6 : (start_idx + 1) * 6, 0] += one_tile_tex_width * (
                tix + factor_rez_dx * x_idx / tessellation_level
            )
            quads[start_idx * 6 : (start_idx + 1) * 6, 1] *= one_tile_tex_height * factor_rez_dy / tessellation_level
            quads[start_idx * 6 : (start_idx + 1) * 6, 1] += one_tile_tex_height * (
                tiy + factor_rez_dy * y_idx / tessellation_level
            )
    return quads


class TileCalculator(object):
    """Common calculations for geographic image tile groups in an array or file

    Tiles are identified by (iy,ix) zero-based indicators.

    """

    OVERSAMPLED = "oversampled"
    UNDERSAMPLED = "undersampled"
    WELLSAMPLED = "wellsampled"

    def __init__(
        self,
        name,
        image_shape,
        ul_origin,
        pixel_rez,
        tile_shape=(DEFAULT_TILE_HEIGHT, DEFAULT_TILE_WIDTH),
        texture_shape=(DEFAULT_TEXTURE_HEIGHT, DEFAULT_TEXTURE_WIDTH),
        projection=DEFAULT_PROJECTION,
        wrap_lon=False,
    ):
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
        # (ny,nx) available tile count for this image:
        self.image_tiles_avail = (self.image_shape[0] / self.tile_shape[0], self.image_shape[1] / self.tile_shape[1])
        self.wrap_lon = wrap_lon

        self.proj = Proj(projection)
        # word coordinates that this image and its tiles corresponds to
        self.image_extents_box = e = Box(
            bottom=np.float64(self.ul_origin[0] - self.image_shape[0] * self.pixel_rez.dy),
            top=np.float64(self.ul_origin[0]),
            left=np.float64(self.ul_origin[1]),
            right=np.float64(self.ul_origin[1] + self.image_shape[1] * self.pixel_rez.dx),
        )
        # Array of points across the image space to be used as an estimate of image coverage
        # Used when checking if the image is viewable on the current canvas's projection
        self.image_mesh = np.meshgrid(
            np.linspace(e.left, e.right, IMAGE_MESH_SIZE), np.linspace(e.bottom, e.top, IMAGE_MESH_SIZE)
        )
        self.image_mesh = np.column_stack(
            (
                self.image_mesh[0].ravel(),
                self.image_mesh[1].ravel(),
            )
        )
        self.image_center = Point(
            self.ul_origin.y - self.image_shape[0] / 2.0 * self.pixel_rez.dy,
            self.ul_origin.x + self.image_shape[1] / 2.0 * self.pixel_rez.dx,
        )
        # size of tile in image projection
        self.tile_size = Resolution(self.pixel_rez.dy * self.tile_shape[0], self.pixel_rez.dx * self.tile_shape[1])
        # maximum stride that we shouldn't lower resolution beyond
        self.overview_stride = self._calc_overview_stride()

    def visible_tiles(self, visible_geom, stride=None, extra_tiles_box=None) -> IndexBox:
        """Get box of tile indexes for visible tiles that should be drawn.

        Box indexes should be iterated with typical Python start:stop style
        (inclusive start index, exclusive stop index). Tiles are expected to
        be indexed (iy, ix) integer pairs. The ``extra_tiles_box`` value
        specifies how many extra tiles to include around each edge.

        """
        if stride is None:
            stride = Point(1, 1)
        if extra_tiles_box is None:
            extra_tiles_box = Box(0, 0, 0, 0)
        v = visible_geom
        e = extra_tiles_box
        return visible_tiles(
            float(self.pixel_rez[0]),
            float(self.pixel_rez[1]),
            float(self.tile_size[0]),
            float(self.tile_size[1]),
            float(self.image_center[0]),
            float(self.image_center[1]),
            int(self.image_shape[0]),
            int(self.image_shape[1]),
            int(self.tile_shape[0]),
            int(self.tile_shape[1]),
            float(v[0]),
            float(v[1]),
            float(v[2]),
            float(v[3]),
            float(v[4]),
            float(v[5]),
            int(stride[0]),
            int(stride[1]),
            int(e[0]),
            int(e[1]),
            int(e[2]),
            int(e[3]),
        )

    def calc_tile_slice(self, tiy, tix, stride):
        """Calculate the slice needed to get data.

        The returned slice assumes the original image data has already
        been reduced by the provided stride.

        Args:
            tiy (int): Tile Y index (down is positive)
            tix (int): Tile X index (right is positive)
            stride (tuple): (Original data Y-stride, Original data X-stride)

        """
        row_slice, col_slice = calc_tile_slice(tiy, tix, stride[0], stride[1], self.image_shape, self.tile_shape)
        return slice(*row_slice), slice(*col_slice)

    def calc_tile_fraction(self, tiy, tix, stride):
        """Calculate the fractional components of the specified tile

        Returns:
            (factor, offset): Two `Resolution` objects stating the relative size
                              of the tile compared to a whole tile and the
                              offset from the origin of a whole tile.
        """
        return calc_tile_fraction(
            tiy,
            tix,
            stride[0],
            stride[1],
            self.image_shape[0],
            self.image_shape[1],
            self.tile_shape[0],
            self.tile_shape[1],
        )

    def calc_stride(self, visible, texture=None):
        """
        given world geometry and sampling as a ViewBox or Resolution tuple
        calculate a conservative stride value for rendering a set of tiles
        :param visible: ViewBox or Resolution with world pixels per screen pixel
        :param texture: ViewBox or Resolution with texture resolution as world pixels per screen pixel
        """
        texture = texture or self.pixel_rez
        return calc_stride(
            visible.dx, visible.dy, texture.dx, texture.dy, self.overview_stride[0].step, self.overview_stride[1].step
        )

    def _calc_overview_stride(self, image_shape=None):
        image_shape = image_shape or self.image_shape
        # FUTURE: Come up with a fancier way of doing overviews like averaging each strided section, if needed
        tsy, tsx = calc_overview_stride(image_shape[0], image_shape[1], self.tile_shape)
        return slice(0, image_shape[0], tsy), slice(0, image_shape[1], tsx)

    def calc_vertex_coordinates(self, tiy, tix, stridey, stridex, factor_rez, offset_rez, tessellation_level=1):
        quad = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32)
        quads = np.tile(quad, (tessellation_level * tessellation_level, 1))
        quads = calc_vertex_coordinates(
            tiy,
            tix,
            stridey,
            stridex,
            factor_rez.dy,
            factor_rez.dx,
            offset_rez.dy,
            offset_rez.dx,
            tessellation_level,
            self.pixel_rez.dx,
            self.pixel_rez.dy,
            self.tile_shape.y,
            self.tile_shape.x,
            self.image_center.y,
            self.image_center.x,
            quads,
        )
        quads = quads.reshape(tessellation_level * tessellation_level * 6, 3)
        return quads[:, :2]

    def calc_texture_coordinates(self, ttile_idx, factor_rez, offset_rez, tessellation_level=1):
        """Get texture coordinates for one tile as a quad.

        :param ttile_idx: int, texture 1D index that maps to some internal texture tile location
        """
        tiy = int(ttile_idx / self.texture_shape[1])
        tix = ttile_idx % self.texture_shape[1]
        # start with basic quad describing the entire texture
        quad = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32)
        quads = np.tile(quad, (tessellation_level * tessellation_level, 1))
        quads = calc_texture_coordinates(
            tiy,
            tix,
            factor_rez.dy,
            factor_rez.dx,
            tessellation_level,
            self.texture_size[0],
            self.texture_size[1],
            self.tile_shape.y,
            self.tile_shape.x,
            quads,
        )
        quads = quads.reshape(6 * tessellation_level * tessellation_level, 3)
        quads = np.ascontiguousarray(quads[:, :2])
        return quads

    def calc_view_extents(self, canvas_point, image_point, canvas_size, dx, dy):
        return calc_view_extents(self.image_extents_box, canvas_point, image_point, canvas_size, dx, dy)
