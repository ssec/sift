#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
visuals.py
~~~~~~~~~~~

PURPOSE
Layer representation - the "physical" realization of content to draw on the map.
A layer representation can have multiple levels of detail

A factory will convert URIs into LayerReps
LayerReps are managed by document, and handed off to the MapWidget as part of a LayerDrawingPlan

REFERENCES


REQUIRES


:author: R.K.Garcia <rayg@ssec.wisc.edu>
:copyright: 2014 by University of Wisconsin Regents, see AUTHORS for more details
:license: GPLv3, see LICENSE for more details
"""

import logging
from datetime import datetime

import numpy as np
import shapefile
from vispy.scene.visuals import create_visual_node
from vispy.visuals import LineVisual, ImageVisual, IsocurveVisual
# The below imports are needed because we subclassed the ImageVisual
from vispy.visuals.shaders import Function, FunctionChain
from vispy.gloo.texture import should_cast_to_f32


from uwsift.common import (
    DEFAULT_PROJECTION,
    DEFAULT_TILE_HEIGHT,
    DEFAULT_TILE_WIDTH,
    DEFAULT_TEXTURE_HEIGHT,
    DEFAULT_TEXTURE_WIDTH,
    TESS_LEVEL,
    Box, Point, Resolution, ViewBox,
)
from uwsift.view.texture_atlas import TextureAtlas2D, MultiChannelTextureAtlas2D, MultiChannelGPUScaledTexture2D
from uwsift.view.tile_calculator import TileCalculator, calc_pixel_size, get_reference_points

__author__ = 'rayg'
__docformat__ = 'reStructuredText'

LOG = logging.getLogger(__name__)
# if the absolute value of a vertex coordinate is beyond 'CANVAS_EPSILON'
# then we consider it invalid
# these values can get large when zoomed way in
CANVAS_EPSILON = 1e5


# CANVAS_EPSILON = 1e30


class ArrayProxy(object):
    def __init__(self, ndim, shape):
        self.ndim = ndim
        self.shape = shape


class TextureTileState(object):
    """Object to hold the state of the current tile texture.

    Terms:

    - itile: Image Tile, Tile in the image being shown. Coordinates are (Y, X)
    - lod: Level of Detail, Level of detail for the image tiles (1 highest - 5 lower)
    - ttile: Texture Tile, Tile in the actual GPU texture storage (0 to `num_tiles`)

    This class is meant to be used as a bookkeeper/consultant right before taking action
    on the Texture Atlas.
    """

    def __init__(self, num_tiles):
        self.num_tiles = num_tiles
        self.reset()

    def __getitem__(self, item):
        """Get the texture index associated with this image tile index.
        """
        return self.itile_cache[item]

    def __contains__(self, item):
        """Have we already added this image tile index (yidx, xidx).
        """
        return item in self.itile_cache

    def reset(self):
        self.itile_cache = {}
        self._rev_cache = {}
        # True if the data doesn't matter, False if data matters
        self.tile_free = [True] * self.num_tiles
        self.itile_age = []

    def next_available_tile(self):
        for idx, tile_free in enumerate(self.tile_free):
            if tile_free:
                return idx

        # We don't have any free tiles, remove the oldest one
        LOG.debug("Expiring image tile from texture atlas: %r", self.itile_age[0])
        ttile_idx = self.remove_tile(self.itile_age[0])
        return ttile_idx

    def refresh_age(self, itile_idx):
        """Update the age of an image tile so it is less likely to expire.
        """
        try:
            # Remove it from wherever it is
            self.itile_age.remove(itile_idx)
        except ValueError:
            # we haven't heard about this tile, that's ok, we'll add it to the list
            pass

        # Put it to the end as the "youngest" tile
        self.itile_age.append(itile_idx)

    def add_tile(self, itile_idx, expires=True):
        """Get texture index for new tile. If tile is already known return its current location.

        Note, this should be called even when the caller knows the tile exists to refresh the "age".
        """
        # Have we already added this tile, get the tile index
        if itile_idx in self:
            if expires:
                self.refresh_age(itile_idx)
            return self[itile_idx]

        ttile_idx = self.next_available_tile()

        self.itile_cache[itile_idx] = ttile_idx
        self._rev_cache[ttile_idx] = itile_idx
        self.tile_free[ttile_idx] = False
        if expires:
            self.refresh_age(itile_idx)
        return ttile_idx

    def remove_tile(self, itile_idx):
        ttile_idx = self.itile_cache.pop(itile_idx)
        self._rev_cache.pop(ttile_idx)
        self.tile_free[ttile_idx] = True
        self.itile_age.remove(itile_idx)
        return ttile_idx


class SIFTTiledGeolocatedMixin:
    def __init__(self, data, *area_params,
                 tile_shape=(DEFAULT_TILE_HEIGHT, DEFAULT_TILE_WIDTH),
                 texture_shape=(DEFAULT_TEXTURE_HEIGHT, DEFAULT_TEXTURE_WIDTH),
                 wrap_lon=False, projection=DEFAULT_PROJECTION,
                 **visual_kwargs):
        origin_x, origin_y, cell_width, cell_height = area_params
        if visual_kwargs.get("method", "subdivide") != "subdivide":
            raise ValueError("Only 'subdivide' drawing method is supported.")
        visual_kwargs["method"] = "subdivide"
        if "grid" in visual_kwargs:
            raise ValueError("The 'grid' keyword argument is not supported with the tiled mixin.")

        # visual nodes already have names, so be careful
        if not hasattr(self, "name"):
            self.name = visual_kwargs.pop("name", None)

        self._init_geo_parameters(
            origin_x,
            origin_y,
            cell_width,
            cell_height,
            projection,
            texture_shape,
            tile_shape,
            wrap_lon,
            visual_kwargs.get('shape'),
            data,
        )

        # Call the init of the Visual
        super().__init__(data, **visual_kwargs)

    def _init_geo_parameters(
            self,
            origin_x,
            origin_y,
            cell_width,
            cell_height,
            projection,
            texture_shape,
            tile_shape,
            wrap_lon,
            shape,
            data
    ):
        self._viewable_mesh_mask = None
        self._ref1 = None
        self._ref2 = None

        self.origin_x = origin_x
        self.origin_y = origin_y
        self.cell_width = cell_width
        self.cell_height = cell_height  # Note: cell_height is usually negative
        self.texture_shape = texture_shape
        self.tile_shape = tile_shape
        self.num_tex_tiles = self.texture_shape[0] * self.texture_shape[1]
        self._stride = (0, 0)
        self._latest_tile_box = None
        self.wrap_lon = wrap_lon
        self._tiles = {}
        assert (shape or data is not None), "`data` or `shape` must be provided"
        self.shape = shape or data.shape
        self.ndim = len(self.shape) or data.ndim

        # Where does this image lie in this lonely world
        self.calc = TileCalculator(
            self.name,
            self.shape,
            Point(x=self.origin_x, y=self.origin_y),
            Resolution(dy=abs(self.cell_height), dx=abs(self.cell_width)),
            self.tile_shape,
            self.texture_shape,
            wrap_lon=self.wrap_lon,
            projection=projection,
        )
        # What tiles have we used and can we use
        self.texture_state = TextureTileState(self.num_tex_tiles)

    def _normalize_data(self, data):
        if data is not None and data.dtype == np.float64:
            data = data.astype(np.float32)
        return data

    def _build_texture_tiles(self, data, stride, tile_box: Box):
        """Prepare and organize strided data in to individual tiles with associated information.
        """
        data = self._normalize_data(data)

        LOG.debug("Uploading texture data for %d tiles (%r)",
                  (tile_box.bottom - tile_box.top) * (tile_box.right - tile_box.left), tile_box)
        # Tiles start at upper-left so go from top to bottom
        tiles_info = []
        for tiy in range(tile_box.top, tile_box.bottom):
            for tix in range(tile_box.left, tile_box.right):
                already_in = (stride, tiy, tix) in self.texture_state
                # Update the age if already in there
                # Assume that texture_state does not change from the main thread if this is run in another
                tex_tile_idx = self.texture_state.add_tile((stride, tiy, tix))
                if already_in:
                    # FIXME: we should make a list/set of the tiles we need to add before this
                    continue

                # Assume we were given a total image worth of this stride
                y_slice, x_slice = self.calc.calc_tile_slice(tiy, tix, stride)
                tile_data = self._slice_texture_tile(data, y_slice, x_slice)
                tiles_info.append((stride, tiy, tix, tex_tile_idx, tile_data))

        return tiles_info

    def _slice_texture_tile(self, data, y_slice, x_slice):
        # force a copy of the data from the content array (provided by the workspace)
        # to a vispy-compatible contiguous float array
        # this can be a potentially time-expensive operation since content array is
        # often huge and always memory-mapped, so paging may occur
        # we don't want this paging deferred until we're back in the GUI thread pushing data to OpenGL!
        return np.array(data[y_slice, x_slice], dtype=np.float32)

    def _set_texture_tiles(self, tiles_info):
        for tile_info in tiles_info:
            stride, tiy, tix, tex_tile_idx, data = tile_info
            self._texture.set_tile_data(tex_tile_idx, data)

    def _build_vertex_tiles(self, preferred_stride, tile_box: Box):
        """Rebuild the vertex buffers used for rendering the image when using
        the subdivide method.

        SIFT Note: Copied from 0.5.0dev original ImageVisual class
        """
        total_num_tiles = (tile_box.bottom - tile_box.top) * (tile_box.right - tile_box.left)

        if total_num_tiles <= 0:
            # we aren't looking at this image
            # FIXME: What's the correct way to stop drawing here
            raise RuntimeError("View calculations determined a negative number of tiles are visible")
        elif total_num_tiles > self.num_tex_tiles:
            LOG.warning("Current view sees more tiles than can be held in the GPU")
            # We continue on, showing as many tiles as we can

        tex_coords = np.empty((6 * total_num_tiles * (TESS_LEVEL * TESS_LEVEL), 2), dtype=np.float32)
        vertices = np.empty((6 * total_num_tiles * (TESS_LEVEL * TESS_LEVEL), 2), dtype=np.float32)

        # What tile are we currently describing out of all the tiles being viewed
        used_tile_idx = -1
        LOG.debug("Building vertex data for %d tiles (%r)", total_num_tiles, tile_box)
        tl = TESS_LEVEL * TESS_LEVEL
        # Tiles start at upper-left so go from top to bottom
        for tiy in range(tile_box.top, tile_box.bottom):
            for tix in range(tile_box.left, tile_box.right):
                # Update the index here because we have multiple exit/continuation points
                used_tile_idx += 1

                # Check if the tile we want to draw is actually in the GPU
                # if not (atlas too small?) fill with zeros and keep going
                if (preferred_stride, tiy, tix) not in self.texture_state:
                    # THIS SHOULD NEVER HAPPEN IF TEXTURE BUILDING IS DONE CORRECTLY AND THE ATLAS IS BIG ENOUGH
                    tile_start = TESS_LEVEL * TESS_LEVEL * used_tile_idx * 6
                    tile_end = TESS_LEVEL * TESS_LEVEL * (used_tile_idx + 1) * 6
                    tex_coords[tile_start: tile_end, :] = 0
                    vertices[tile_start: tile_end, :] = 0
                    continue

                # we should have already loaded the texture data in to the GPU so get the index of that texture
                tex_tile_idx = self.texture_state[(preferred_stride, tiy, tix)]
                factor_rez, offset_rez = self.calc.calc_tile_fraction(tiy, tix, preferred_stride)
                tex_coords[tl * used_tile_idx * 6: tl * (used_tile_idx + 1) * 6, :] = \
                    self.calc.calc_texture_coordinates(tex_tile_idx, factor_rez, offset_rez,
                                                       tessellation_level=TESS_LEVEL)
                vertices[tl * used_tile_idx * 6: tl * (used_tile_idx + 1) * 6, :] = self.calc.calc_vertex_coordinates(
                    tiy, tix,
                    preferred_stride[0], preferred_stride[1],
                    factor_rez, offset_rez,
                    tessellation_level=TESS_LEVEL)

        return vertices, tex_coords

    def _set_vertex_tiles(self, vertices, tex_coords):
        self._subdiv_position.set_data(vertices.astype('float32'))
        self._subdiv_texcoord.set_data(tex_coords.astype('float32'))

    def determine_reference_points(self):
        # Image points transformed to canvas coordinates
        img_cmesh = self.transforms.get_transform().map(self.calc.image_mesh)
        # Mask any points that are really far off screen (can't be transformed)
        valid_mask = (np.abs(img_cmesh[:, 0]) < CANVAS_EPSILON) & (np.abs(img_cmesh[:, 1]) < CANVAS_EPSILON)
        # The image mesh projected to canvas coordinates (valid only)
        img_cmesh = img_cmesh[valid_mask]
        # The image mesh of only valid "viewable" projected coordinates
        img_vbox = self.calc.image_mesh[valid_mask]

        if not img_cmesh[:, 0].size or not img_cmesh[:, 1].size:
            self._viewable_mesh_mask = None
            self._ref1, self._ref2 = None, None
            return

        x_cmin, x_cmax = img_cmesh[:, 0].min(), img_cmesh[:, 0].max()
        y_cmin, y_cmax = img_cmesh[:, 1].min(), img_cmesh[:, 1].max()
        center_x = (x_cmax - x_cmin) / 2. + x_cmin
        center_y = (y_cmax - y_cmin) / 2. + y_cmin
        dist = img_cmesh.copy()
        dist[:, 0] = center_x - img_cmesh[:, 0]
        dist[:, 1] = center_y - img_cmesh[:, 1]
        self._viewable_mesh_mask = valid_mask
        self._ref1, self._ref2 = get_reference_points(dist, img_vbox)

    def get_view_box(self):
        """Calculate shown portion of image and image units per pixel

        This method utilizes a precomputed "mesh" of relatively evenly
        spaced points over the entire image space. This mesh is transformed
        to the canvas space (-1 to 1 user-viewed space) to figure out which
        portions of the image are currently being viewed and which portions
        can actually be projected on the viewed projection.

        While the result of the chosen method may not always be completely
        accurate, it should work for all possible viewing cases.
        """
        if self._viewable_mesh_mask is None or self.canvas.size[0] == 0 or self.canvas.size[1] == 0:
            raise ValueError("Image '%s' is not viewable in this projection" % (self.name,))

        # Image points transformed to canvas coordinates
        img_cmesh = self.transforms.get_transform().map(self.calc.image_mesh)
        # The image mesh projected to canvas coordinates (valid only)
        img_cmesh = img_cmesh[self._viewable_mesh_mask]
        # The image mesh of only valid "viewable" projected coordinates
        img_vbox = self.calc.image_mesh[self._viewable_mesh_mask]

        ref_idx_1, ref_idx_2 = get_reference_points(img_cmesh, img_vbox)
        dx, dy = calc_pixel_size(img_cmesh[(self._ref1, self._ref2), :],
                                 img_vbox[(self._ref1, self._ref2), :],
                                 self.canvas.size)
        view_extents = self.calc.calc_view_extents(img_cmesh[ref_idx_1], img_vbox[ref_idx_1], self.canvas.size, dx, dy)
        return ViewBox(*view_extents, dx=dx, dy=dy)

    def _get_stride(self, view_box):
        return self.calc.calc_stride(view_box)

    def assess(self):
        """Determine if a retile is needed.

        Tell workspace we will be needed
        """
        try:
            view_box = self.get_view_box()
            preferred_stride = self._get_stride(view_box)
            tile_box = self.calc.visible_tiles(view_box, stride=preferred_stride, extra_tiles_box=Box(1, 1, 1, 1))
        except ValueError as e:
            # If image is outside of canvas, then an exception will be raised
            LOG.warning("Could not determine viewable image area for '{}': {}".format(self.name, e))
            return False, self._stride, self._latest_tile_box

        num_tiles = (tile_box.bottom - tile_box.top) * (tile_box.right - tile_box.left)
        LOG.debug("Assessment: Prefer '%s' have '%s', was looking at %r, now looking at %r",
                  preferred_stride, self._stride, self._latest_tile_box, tile_box)

        # If we zoomed out or we panned
        need_retile = (num_tiles > 0) and (preferred_stride != self._stride or self._latest_tile_box != tile_box)

        return need_retile, preferred_stride, tile_box

    def retile(self, data, preferred_stride, tile_box):
        """Get data from workspace and retile/retexture as needed.
        """
        tiles_info = self._build_texture_tiles(data, preferred_stride, tile_box)
        vertices, tex_coords = self._build_vertex_tiles(preferred_stride, tile_box)
        return tiles_info, vertices, tex_coords

    def set_retiled(self, preferred_stride, tile_box, tiles_info, vertices, tex_coords):
        self._set_texture_tiles(tiles_info)
        self._set_vertex_tiles(vertices, tex_coords)

        # don't update here, the caller will do that
        # Store the most recent level of detail that we've done
        self._stride = preferred_stride
        self._latest_tile_box = tile_box


class TiledGeolocatedImageVisual(SIFTTiledGeolocatedMixin, ImageVisual):
    def __init__(self, data, origin_x, origin_y, cell_width, cell_height,
                 **image_kwargs):
        super().__init__(data, origin_x, origin_y, cell_width, cell_height, **image_kwargs)

    def _init_texture(self, data, texture_format):
        if self._interpolation == 'bilinear':
            texture_interpolation = 'linear'
        else:
            texture_interpolation = 'nearest'

        tex = TextureAtlas2D(self.texture_shape, tile_shape=self.tile_shape,
                             interpolation=texture_interpolation,
                             format="LUMINANCE", internalformat="R32F",
                             )
        return tex

    def set_data(self, image):
        """Set the data

        Parameters
        ----------
        image : array-like
            The image data.
        """
        if self._data is not None:
            raise NotImplementedError("This image subclass does not support the 'set_data' method.")
        # only do this on __init__
        super().set_data(image)

    def _build_texture(self):
        # _build_texture should not be used in this class, use the 2-step
        # process of '_build_texture_tiles' and '_set_texture_tiles'
        self._need_texture_upload = False

    def _build_vertex_data(self):
        # _build_vertex_data should not be used in this class, use the 2-step
        # process of '_build_vertex_tiles' and '_set_vertex_tiles'
        return


TiledGeolocatedImage = create_visual_node(TiledGeolocatedImageVisual)

_rgb_texture_lookup = """
    vec4 texture_lookup(vec2 texcoord) {
        if(texcoord.x < 0.0 || texcoord.x > 1.0 ||
        texcoord.y < 0.0 || texcoord.y > 1.0) {
            discard;
        }
        vec4 val;
        val.r = texture2D($texture_r, texcoord).r;
        val.g = texture2D($texture_g, texcoord).r;
        val.b = texture2D($texture_b, texcoord).r;
        val.a = 1.0;
        return val;
    }"""

_apply_clim = """
    vec4 apply_clim(vec4 color) {
        // If all the pixels are NaN make it completely transparent
        // http://stackoverflow.com/questions/11810158/how-to-deal-with-nan-or-inf-in-opengl-es-2-0-shaders
        if (
            !(color.r <= 0.0 || 0.0 <= color.r) &&
            !(color.g <= 0.0 || 0.0 <= color.g) &&
            !(color.b <= 0.0 || 0.0 <= color.b)) {
            color.a = 0;
        }
        
        // if color is NaN, set to minimum possible value
        color.r = !(color.r <= 0.0 || 0.0 <= color.r) ? min($clim_r.x, $clim_r.y) : color.r;
        color.g = !(color.g <= 0.0 || 0.0 <= color.g) ? min($clim_g.x, $clim_g.y) : color.g;
        color.b = !(color.b <= 0.0 || 0.0 <= color.b) ? min($clim_b.x, $clim_b.y) : color.b;
        // clamp data to minimum and maximum of clims
        color.r = clamp(color.r, min($clim_r.x, $clim_r.y), max($clim_r.x, $clim_r.y));
        color.g = clamp(color.g, min($clim_g.x, $clim_g.y), max($clim_g.x, $clim_g.y));
        color.b = clamp(color.b, min($clim_b.x, $clim_b.y), max($clim_b.x, $clim_b.y));
        // linearly scale data between clims
        color.r = (color.r - $clim_r.x) / ($clim_r.y - $clim_r.x);
        color.g = (color.g - $clim_g.x) / ($clim_g.y - $clim_g.x);
        color.b = (color.b - $clim_b.x) / ($clim_b.y - $clim_b.x);
        return max(color, 0);
    }
"""

_apply_gamma = """
    vec4 apply_gamma(vec4 color) {
        color.r = pow(color.r, $gamma_r);
        color.g = pow(color.g, $gamma_g);
        color.b = pow(color.b, $gamma_b);
        return color;
    }
"""

_null_color_transform = 'vec4 pass(vec4 color) { return color; }'


class SIFTMultiChannelTiledGeolocatedMixin(SIFTTiledGeolocatedMixin):
    def _normalize_data(self, data_arrays):
        if not isinstance(data_arrays, (list, tuple)):
            return super()._normalize_data(data_arrays)

        new_data = []
        for data in data_arrays:
            new_data.append(super()._normalize_data(data))
        return new_data

    def _init_geo_parameters(
            self,
            origin_x,
            origin_y,
            cell_width,
            cell_height,
            projection,
            texture_shape,
            tile_shape,
            wrap_lon,
            shape,
            data_arrays
    ):
        if shape is None:
            shape = self._compute_shape(shape, data_arrays)
        ndim = len(shape) or [x for x in data_arrays if x is not None][0].ndim
        data = ArrayProxy(ndim, shape)
        super()._init_geo_parameters(
            origin_x,
            origin_y,
            cell_width,
            cell_height,
            projection,
            texture_shape,
            tile_shape,
            wrap_lon,
            shape,
            data,
        )

        self.set_channels(
            data_arrays, shape=shape, cell_width=cell_width,
            cell_height=cell_height, origin_x=origin_x, origin_y=origin_y,
        )

    def set_channels(self, data_arrays, shape=None,
                     cell_width=None, cell_height=None,
                     origin_x=None, origin_y=None):
        assert (shape or data_arrays is not None), "`data` or `shape` must be provided"
        if cell_width is not None:
            self.cell_width = cell_width
        if cell_height:
            self.cell_height = cell_height  # Note: cell_height is usually negative
        if origin_x:
            self.origin_x = origin_x
        if origin_y:
            self.origin_y = origin_y
        self.shape = self._compute_shape(shape, data_arrays)
        assert None not in (self.cell_width, self.cell_height, self.origin_x, self.origin_y, self.shape)
        # how many of the higher resolution channel tiles (smaller geographic area) make
        # up a low resolution channel tile
        self._channel_factors = tuple(
            self.shape[0] / float(chn.shape[0]) if chn is not None else 1. for chn in data_arrays)
        self._lowest_factor = max(self._channel_factors)
        self._lowest_rez = Resolution(abs(self.cell_height * self._lowest_factor),
                                      abs(self.cell_width * self._lowest_factor))

        # Where does this image lie in this lonely world
        self.calc = TileCalculator(
            self.name,
            self.shape,
            Point(x=self.origin_x, y=self.origin_y),
            Resolution(dy=abs(self.cell_height), dx=abs(self.cell_width)),
            self.tile_shape,
            self.texture_shape,
            wrap_lon=self.wrap_lon
        )

        # Reset texture state, if we change things to know which texture
        # don't need to be updated then this can be removed/changed
        self.texture_state.reset()
        self._need_texture_upload = True
        self._need_vertex_update = True
        # Reset the tiling logic to force a retile
        # even though we might be looking at the exact same spot
        self._latest_tile_box = None

    @staticmethod
    def _compute_shape(shape, data_arrays):
        return shape or max(data.shape for data in data_arrays if data is not None)

    def _get_stride(self, view_box):
        s = self.calc.calc_stride(view_box, texture=self._lowest_rez)
        return Point(np.int64(s[0] * self._lowest_factor), np.int64(s[1] * self._lowest_factor))

    def _slice_texture_tile(self, data_arrays, y_slice, x_slice):
        new_data = []
        for data in data_arrays:
            if data is not None:
                # explicitly ask for the parent class of MultiBandTextureAtlas2D
                data = super()._slice_texture_tile(data, y_slice, x_slice)
            new_data.append(data)
        return new_data


class _NoColormap:
    """Placeholder colormap class to make MultiChannelImageVisual compatible with ImageVisual."""

    def texture_lut(self):
        """Get empty (None) colormap data."""
        return None


class MultiChannelImageVisual(ImageVisual):
    """Visual subclass displaying an image from three separate arrays.

    Note this Visual uses only GPU scaling, unlike the ImageVisual base
    class which allows for CPU or GPU scaling.

    Parameters
    ----------
    data : list
        A 3-element list of numpy arrays with 2 dimensons where the
        arrays are sorted by (R, G, B) order. These will be put together
        to make an RGB image. The list can contain ``None`` meaning there
        is no value for this channel currently, but it may be filled in
        later. In this case the underlying GPU storage is still allocated,
        but pre-filled with NaNs. Note that each channel may have different
        shapes.
    cmap : str | Colormap
        Unused by this Visual, but is still provided to the ImageVisual base
        class.
    clim : str | tuple | list | None
        Limits of each RGB data array. If provided as a string it must be
        "auto" and the limits will be computed on the fly. If a 2-element
        tuple then it will be considered the color limits for all channel
        arrays. If provided as a 3-element list of 2-element tuples then
        they represent the color limits of each channel array.
    gamma : float | list
        Gamma to use during colormap lookup.  Final value will be computed
        ``val**gamma`` for each RGB channel array. If provided as a float then
        it will be used for each channel. If provided as a 3-element tuple
        then each value is used for the separate channel arrays. Default is
        1.0 for each channel.
    **kwargs : dict
        Keyword arguments to pass to :class:`~vispy.visuals.ImageVisual`. Note
        that this Visual does not allow for ``texture_format`` to be specified
        and is hardcoded to ``r32f`` internal texture format.

    """

    def __init__(self, data_arrays, clim='auto', gamma=1.0, **kwargs):
        if kwargs.get("texture_format") is not None:
            raise ValueError("'texture_format' can't be specified with the "
                             "'MultiChannelImageVisual'.")
        kwargs["texture_format"] = "R32F"
        if kwargs.get("cmap") is not None:
            raise ValueError("'cmap' can't be specified with the"
                             "'MultiChannelImageVisual'.")
        kwargs["cmap"] = None
        self.num_channels = len(data_arrays)
        super().__init__(data_arrays, clim=clim, gamma=gamma, **kwargs)

    def _init_texture(self, data_arrays, texture_format):
        if self._interpolation == 'bilinear':
            texture_interpolation = 'linear'
        else:
            texture_interpolation = 'nearest'

        tex = MultiChannelGPUScaledTexture2D(
            data_arrays,
            internalformat=texture_format,
            format="LUMINANCE",
            interpolation=texture_interpolation,
        )
        return tex

    def _get_shapes(self, data_arrays):
        shapes = [x.shape for x in data_arrays if x is not None]
        if not shapes:
            raise ValueError("List of data arrays must contain at least one "
                             "numpy array.")
        return shapes

    def _get_min_shape(self, data_arrays):
        return min(self._get_shapes(data_arrays))

    def _get_max_shape(self, data_arrays):
        return max(self._get_shapes(data_arrays))

    @property
    def size(self):
        """Get size of the image (width, height)."""
        return self._get_max_shape(self._data)

    @property
    def clim(self):
        """Get color limits used when rendering the image (cmin, cmax)."""
        return self._texture.clim

    @clim.setter
    def clim(self, clims):
        if isinstance(clims, str) or len(clims) == 2:
            clims = [clims] * self.num_channels
        if self._texture.set_clim(clims):
            self._need_texture_upload = True
        self._update_colortransform_clim()
        self.update()

    def _update_colortransform_clim(self):
        if self._need_colortransform_update:
            # we are going to rebuild anyway so just do it later
            return
        try:
            norm_clims = self._texture.clim_normalized
        except RuntimeError:
            return
        else:
            clim_names = ('clim_r', 'clim_g', 'clim_b')
            # shortcut so we don't have to rebuild the whole color transform
            for clim_name, clim in zip(clim_names, norm_clims):
                # shortcut so we don't have to rebuild the whole color transform
                self.shared_program.frag['color_transform'][1][clim_name] = clim

    @property
    def gamma(self):
        """Get the gamma used when rendering the image."""
        return self._gamma

    @gamma.setter
    def gamma(self, value):
        """Set gamma used when rendering the image."""
        if not isinstance(value, (list, tuple)):
            value = [value] * self.num_channels
        if any(val <= 0 for val in value):
            raise ValueError("gamma must be > 0")
        self._gamma = tuple(float(x) for x in value)

        gamma_names = ('gamma_r', 'gamma_g', 'gamma_b')
        for gamma_name, gam in zip(gamma_names, self._gamma):
            # shortcut so we don't have to rebuild the color transform
            if not self._need_colortransform_update:
                self.shared_program.frag['color_transform'][2][gamma_name] = gam
        self.update()

    @ImageVisual.cmap.setter
    def cmap(self, cmap):
        if cmap is not None:
            raise ValueError("MultiChannelImageVisual does not support a colormap.")
        self._cmap = _NoColormap()

    def _build_interpolation(self):
        # assumes 'nearest' interpolation
        interpolation = self._interpolation
        if interpolation != 'nearest':
            raise NotImplementedError("MultiChannelImageVisual only supports 'nearest' interpolation.")
        texture_interpolation = 'nearest'

        self._data_lookup_fn = Function(_rgb_texture_lookup)
        self.shared_program.frag['get_data'] = self._data_lookup_fn
        if self._texture.interpolation != texture_interpolation:
            self._texture.interpolation = texture_interpolation
        self._data_lookup_fn['texture_r'] = self._texture.textures[0]
        self._data_lookup_fn['texture_g'] = self._texture.textures[1]
        self._data_lookup_fn['texture_b'] = self._texture.textures[2]

        self._need_interpolation_update = False

    def _build_color_transform(self):
        if self.num_channels != 3:
            raise NotImplementedError("MultiChannelimageVisuals only support 3 channels.")
        else:
            # RGB/A image data (no colormap)
            fclim = Function(_apply_clim)
            fgamma = Function(_apply_gamma)
            fun = FunctionChain(None, [Function(_null_color_transform), fclim, fgamma])
        fclim['clim_r'] = self._texture.textures[0].clim_normalized
        fclim['clim_g'] = self._texture.textures[1].clim_normalized
        fclim['clim_b'] = self._texture.textures[2].clim_normalized
        fgamma['gamma_r'] = self.gamma[0]
        fgamma['gamma_g'] = self.gamma[1]
        fgamma['gamma_b'] = self.gamma[2]
        return fun

    def set_data(self, data_arrays):
        """Set the data

        Parameters
        ----------
        image : array-like
            The image data.
        """
        if self._data is not None and any(self._shape_differs(x1, x2) for x1, x2 in zip(self._data, data_arrays)):
            self._need_vertex_update = True
        data_arrays = list(self._cast_arrays_if_needed(data_arrays))
        self._texture.check_data_format(data_arrays)
        self._data = data_arrays
        self._need_texture_upload = True

    @staticmethod
    def _cast_arrays_if_needed(data_arrays):
        for data in data_arrays:
            if data is not None and should_cast_to_f32(data.dtype):
                data = data.astype(np.float32)
            yield data

    @staticmethod
    def _shape_differs(arr1, arr2):
        none_change1 = arr1 is not None and arr2 is None
        none_change2 = arr1 is None and arr2 is not None
        shape_change = False
        if arr1 is not None and arr2 is not None:
            shape_change = arr1.shape[:2] != arr2.shape[:2]
        return none_change1 or none_change2 or shape_change

    def _build_texture(self):
        pre_clims = self._texture.clim
        pre_internalformat = self._texture.internalformat
        self._texture.scale_and_set_data(self._data)
        post_clims = self._texture.clim
        post_internalformat = self._texture.internalformat
        # color transform needs rebuilding if the internalformat was changed
        # new color limits need to be assigned if the normalized clims changed
        # otherwise, the original color transform should be fine
        # Note that this assumes that if clim changed, clim_normalized changed
        new_if = post_internalformat != pre_internalformat
        new_cl = post_clims != pre_clims
        if new_if or new_cl:
            self._need_colortransform_update = True
        self._need_texture_upload = False


MultiChannelImage = create_visual_node(MultiChannelImageVisual)


class RGBCompositeLayerVisual(SIFTMultiChannelTiledGeolocatedMixin,
                              TiledGeolocatedImageVisual,
                              MultiChannelImageVisual):
    def _init_texture(self, data_arrays, texture_format):
        if self._interpolation == 'bilinear':
            texture_interpolation = 'linear'
        else:
            texture_interpolation = 'nearest'

        tex_shapes = [self.texture_shape] * len(data_arrays)
        tex = MultiChannelTextureAtlas2D(
            tex_shapes, tile_shape=self.tile_shape,
            interpolation=texture_interpolation, format="LUMINANCE", internalformat="R32F"
        )
        return tex


RGBCompositeLayer = create_visual_node(RGBCompositeLayerVisual)


class ShapefileLinesVisual(LineVisual):
    def __init__(self, filepath, double=False, **kwargs):
        LOG.debug("Using border shapefile '%s'", filepath)
        self.sf = shapefile.Reader(filepath)

        LOG.info("Loading boundaries: %s", datetime.utcnow().isoformat(" "))
        # Prepare the arrays
        total_points = 0
        total_parts = 0
        for one_shape in self.sf.iterShapes():
            total_points += len(one_shape.points)
            total_parts += len(one_shape.parts)
        vertex_buffer = np.empty((total_points * 2 - total_parts * 2, 2), dtype=np.float32)
        prev_idx = 0
        for one_shape in self.sf.iterShapes():
            # end_idx = prev_idx + len(one_shape.points) * 2 - len(one_shape.parts) * 2
            # vertex_buffer[prev_idx:end_idx:2] = one_shape.points[:-1]
            # for part_idx in one_shape.parts:
            for part_start, part_end in zip(one_shape.parts, list(one_shape.parts[1:]) + [len(one_shape.points)]):
                end_idx = prev_idx + (part_end - part_start) * 2 - 2
                vertex_buffer[prev_idx:end_idx:2] = one_shape.points[part_start:part_end - 1]
                vertex_buffer[prev_idx + 1:end_idx:2] = one_shape.points[part_start + 1:part_end]
                prev_idx = end_idx

        # Clip lats to +/- 89.9 otherwise PROJ.4 on mercator projection will fail
        np.clip(vertex_buffer[:, 1], -89.9, 89.9, out=vertex_buffer[:, 1])
        # vertex_buffer[:, 0], vertex_buffer[:, 1] = self.proj(vertex_buffer[:, 0], vertex_buffer[:, 1])
        if double:
            LOG.debug("Adding 180 to 540 double of shapefile")
            orig_points = vertex_buffer.shape[0]
            vertex_buffer = np.concatenate((vertex_buffer, vertex_buffer), axis=0)
            # vertex_buffer[orig_points:, 0] += C_EQ
            vertex_buffer[orig_points:, 0] += 360

        kwargs.setdefault("color", (1.0, 1.0, 1.0, 1.0))
        kwargs.setdefault("width", 1)
        super().__init__(pos=vertex_buffer, connect="segments", **kwargs)
        LOG.info("Done loading boundaries: %s", datetime.utcnow().isoformat(" "))


ShapefileLines = create_visual_node(ShapefileLinesVisual)


class NEShapefileLinesVisual(ShapefileLinesVisual):
    """Layer class for handling shapefiles from Natural Earth.

    http://www.naturalearthdata.com/

    There should be no difference in the format of the file, but some
    assumptions can be made with data from Natural Earth about filenaming,
    data resolution, fields and other record information that is normally
    included in most Natural Earth files.
    """
    pass


NEShapefileLines = create_visual_node(NEShapefileLinesVisual)


class PrecomputedIsocurveVisual(IsocurveVisual):
    """IsocurveVisual that can use precomputed paths."""

    def __init__(self, verts, connects, level_indexes, levels, **kwargs):
        num_zoom_levels = len(levels)
        num_levels_per_zlevel = [len(x) for x in levels]
        self._zoom_level_indexes = [
            level_indexes[:sum(num_levels_per_zlevel[:z_level + 1])]
            for z_level in range(num_zoom_levels)]
        self._zoom_level_size = [sum(z_level_indexes) for z_level_indexes in self._zoom_level_indexes]

        self._all_verts = []
        self._all_connects = []
        self._all_levels = []
        self._zoom_level = -1
        for zoom_level in range(num_zoom_levels):
            end_idx = self._zoom_level_size[zoom_level]
            self._all_verts.append(verts[:end_idx])
            self._all_connects.append(connects[:end_idx])
            self._all_levels.append([x for y in levels[:zoom_level + 1] for x in y])

        super(PrecomputedIsocurveVisual, self).__init__(data=None, levels=levels, **kwargs)

        self._data = True
        self._level_min = 0
        self.zoom_level = kwargs.pop("zoom_level", 0)

    @property
    def zoom_level(self):
        return self._zoom_level

    @zoom_level.setter
    def zoom_level(self, val):
        if val == self._zoom_level:
            return
        self._zoom_level = val
        self._li = self._zoom_level_indexes[self._zoom_level]
        self._connect = self._all_connects[self._zoom_level]
        self._verts = self._all_verts[self._zoom_level]
        # this will trigger all of the recomputation
        self.levels = self._all_levels[self._zoom_level]

    @property
    def clim(self):
        return self._clim

    @clim.setter
    def clim(self, val):
        assert len(val) == 2
        self._clim = val
        self._need_level_update = True
        self._need_color_update = True
        self.update()

    def _compute_iso_line(self):
        """ compute LineVisual vertices, connects and color-index
        """
        return


PrecomputedIsocurve = create_visual_node(PrecomputedIsocurveVisual)
