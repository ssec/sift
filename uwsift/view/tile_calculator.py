import numpy as np
from numba import jit, float64, float32, int64, boolean, types as nb_types
from pyproj import Proj

from uwsift.common import (Resolution, Point, Box, ViewBox,
                           CANVAS_EXTENTS_EPSILON,
                           PREFERRED_SCREEN_TO_TEXTURE_RATIO,
                           IMAGE_MESH_SIZE,
                           DEFAULT_PROJECTION,
                           DEFAULT_TEXTURE_HEIGHT,
                           DEFAULT_TEXTURE_WIDTH,
                           DEFAULT_TILE_HEIGHT,
                           DEFAULT_TILE_WIDTH)


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


@jit(nopython=True)
def calc_tile_slice(tiy, tix, stride, image_shape, tile_shape):
    """Calculate the slice needed to get data.

    The returned slice assumes the original image data has already
    been reduced by the provided stride.

    Args:
        tiy (int): Tile Y index (down is positive)
        tix (int): Tile X index (right is positive)
        stride (tuple): (Original data Y-stride, Original data X-stride)

    """
    y_offset = int(image_shape[0] / 2. / stride[0] - tile_shape[0] / 2.)
    y_start = int(tiy * tile_shape[0] + y_offset)
    if y_start < 0:
        row_slice = (0, max(0, y_start + tile_shape[0]), 1)
    else:
        row_slice = (y_start, y_start + tile_shape[0], 1)

    x_offset = int(image_shape[1] / 2. / stride[1] - tile_shape[1] / 2.)
    x_start = int(tix * tile_shape[1] + x_offset)
    if x_start < 0:
        col_slice = (0, max(0, x_start + tile_shape[1]), 1)
    else:
        col_slice = (x_start, x_start + tile_shape[1], 1)
    return row_slice, col_slice


# @jit(nb_types.UniTuple(nb_types.NamedUniTuple(float64, 2, Resolution), 2)(
#     int64,
#     int64,
#     nb_types.NamedUniTuple(int64, 2, Point),
#     nb_types.NamedUniTuple(int64, 2, Point),
#     nb_types.NamedUniTuple(int64, 2, Point)
# ),
@jit(nopython=True)
def calc_tile_fraction(tiy, tix, stride, image_shape, tile_shape):
    """Calculate the fractional components of the specified tile

    Returns:
        (factor, offset): Two `Resolution` objects stating the relative size
                          of the tile compared to a whole tile and the
                          offset from the origin of a whole tile.
    """
    mt = max_tiles_available(image_shape, tile_shape, stride)

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


# @jit(nb_types.NamedUniTuple(int64, 2, Point)(
#     float64,
#     float64,
#     float64,
#     float64,
#     int64,
#     int64
# ),  nopython=True)
@jit(nopython=True)
def calc_stride(v_dx, v_dy, t_dx, t_dy, overview_stride_y, overview_stride_x):
    """
    given world geometry and sampling as a ViewBox or Resolution tuple
    calculate a conservative stride value for rendering a set of tiles
    :param visible: ViewBox or Resolution with world pixels per screen pixel
    :param texture: ViewBox or Resolution with texture resolution as world pixels per screen pixel
    """
    # screen dy,dx in world distance per pixel
    # world distance per pixel for our data
    # compute texture pixels per screen pixels
    tsy = min(overview_stride_y,
              max(1, np.ceil(v_dy * PREFERRED_SCREEN_TO_TEXTURE_RATIO / t_dy)))
    tsx = min(overview_stride_x,
              max(1, np.ceil(v_dx * PREFERRED_SCREEN_TO_TEXTURE_RATIO / t_dx)))

    return Point(np.int64(tsy), np.int64(tsx))


# @jit(nb_types.UniTuple(int64, 2)(
#     nb_types.NamedUniTuple(int64, 2, Point),
#     nb_types.NamedUniTuple(int64, 2, Point)
# ),  nopython=True)
@jit(nopython=True)
def calc_overview_stride(image_shape, tile_shape):
    # FUTURE: Come up with a fancier way of doing overviews like averaging each strided section, if needed
    tsy = max(1, int(np.floor(image_shape[0] / tile_shape[0])))
    tsx = max(1, int(np.floor(image_shape[1] / tile_shape[1])))
    return tsy, tsx


# @jit(float32[:,:](
#      int64,
#      int64,
#      int64,
#      int64,
#      nb_types.NamedUniTuple(float64, 2, Resolution),
#      nb_types.NamedUniTuple(float64, 2, Resolution),
#      int64,
#      float64,
#      float64,
#      nb_types.NamedUniTuple(int64, 2, Point),
#      nb_types.NamedUniTuple(float64, 2, Point),
#      float32[:,:]
# ), nopython=True)
@jit(nopython=True)
def calc_vertex_coordinates(tiy, tix, stridey, stridex, factor_rez, offset_rez, tessellation_level, p_dx, p_dy,
                            tile_shape, image_center, quads):
    tile_w = p_dx * tile_shape[1] * stridex
    tile_h = p_dy * tile_shape[0] * stridey
    origin_x = image_center[1] - tile_w / 2.
    origin_y = image_center[0] + tile_h / 2.
    for x_idx in range(tessellation_level):
        for y_idx in range(tessellation_level):
            start_idx = x_idx * tessellation_level + y_idx
            quads[start_idx * 6:(start_idx + 1) * 6, 0] *= tile_w * factor_rez.dx / tessellation_level
            quads[start_idx * 6:(start_idx + 1) * 6, 0] += origin_x + tile_w * (
                    tix + offset_rez.dx + factor_rez.dx * x_idx / tessellation_level)
            # Origin is upper-left so image goes dow,n
            quads[start_idx * 6:(start_idx + 1) * 6, 1] *= -tile_h * factor_rez.dy / tessellation_level
            quads[start_idx * 6:(start_idx + 1) * 6, 1] += origin_y - tile_h * (
                    tiy + offset_rez.dy + factor_rez.dy * y_idx / tessellation_level)
    return quads


@jit(nopython=True)
def calc_texture_coordinates(tiy, tix, factor_rez, tessellation_level, texture_size, tile_shape, quads):
    # Now scale and translate the coordinates so they only apply to one tile in the texture
    one_tile_tex_width = 1.0 / texture_size[1] * tile_shape[1]
    one_tile_tex_height = 1.0 / texture_size[0] * tile_shape[0]
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
    return quads


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

    def calc_tile_slice(self, tiy, tix, stride):
        """Calculate the slice needed to get data.

        The returned slice assumes the original image data has already
        been reduced by the provided stride.

        Args:
            tiy (int): Tile Y index (down is positive)
            tix (int): Tile X index (right is positive)
            stride (tuple): (Original data Y-stride, Original data X-stride)

        """
        row_slice, col_slice = calc_tile_slice(tiy, tix, stride, self.image_shape, self.tile_shape)
        return slice(*row_slice), slice(*col_slice)

    def calc_tile_fraction(self, tiy, tix, stride):
        return calc_tile_fraction(tiy, tix, stride, self.image_shape, self.tile_shape)

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
        return calc_stride(visible.dx, visible.dy, texture.dx, texture.dy, self.overview_stride[0].step,
                           self.overview_stride[1].step)

    def calc_overview_stride(self, image_shape=None):
        image_shape = image_shape or self.image_shape
        # FUTURE: Come up with a fancier way of doing overviews like averaging each strided section, if needed
        tsy, tsx = calc_overview_stride(image_shape, self.tile_shape)
        return slice(0, image_shape[0], tsy), slice(0, image_shape[1], tsx)

    def calc_vertex_coordinates(self, tiy, tix, stridey, stridex,
                                factor_rez, offset_rez, tessellation_level=1):
        quad = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0],
                         [0, 0, 0], [1, 1, 0], [0, 1, 0]],
                        dtype=np.float32)
        quads = np.tile(quad, (tessellation_level * tessellation_level, 1))
        quads = calc_vertex_coordinates(tiy, tix, stridey, stridex, factor_rez, offset_rez, tessellation_level,
                                        self.pixel_rez.dx, self.pixel_rez.dy, self.tile_shape, self.image_center, quads)

        quads = quads.reshape(tessellation_level * tessellation_level * 6, 3)
        return quads[:, :2]

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
        quads = calc_texture_coordinates(tiy, tix, factor_rez, tessellation_level, self.texture_size,
                                         self.tile_shape, quads)
        quads = quads.reshape(6 * tessellation_level * tessellation_level, 3)
        quads = np.ascontiguousarray(quads[:, :2])
        return quads

    def calc_view_extents(self, canvas_point, image_point, canvas_size, dx, dy):
        return calc_view_extents(self.image_extents_box, canvas_point, image_point, canvas_size, dx, dy)
