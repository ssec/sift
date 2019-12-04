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
from vispy.ext.six import string_types
from vispy.gloo import VertexBuffer
from vispy.io.datasets import load_spatial_filters
from vispy.scene.visuals import create_visual_node
from vispy.visuals import LineVisual, ImageVisual, IsocurveVisual
# The below imports are needed because we subclassed the ImageVisual
from vispy.visuals.shaders import Function
from vispy.visuals.transforms import NullTransform


from uwsift.common import (
    DEFAULT_PROJECTION,
    DEFAULT_TILE_HEIGHT,
    DEFAULT_TILE_WIDTH,
    DEFAULT_TEXTURE_HEIGHT,
    DEFAULT_TEXTURE_WIDTH,
    TESS_LEVEL,
    Box, Point, Resolution, ViewBox,
)
from uwsift.view.texture_atlas import TextureAtlas2D, Texture2D
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


VERT_SHADER = """
uniform int method;  // 0=subdivide, 1=impostor
attribute vec2 a_position;
attribute vec2 a_texcoord;
varying vec2 v_texcoord;

void main() {
    v_texcoord = a_texcoord;
    gl_Position = $transform(vec4(a_position, 0., 1.));
}
"""

FRAG_SHADER = """
uniform vec2 image_size;
uniform int method;  // 0=subdivide, 1=impostor
uniform sampler2D u_texture;
varying vec2 v_texcoord;

vec4 map_local_to_tex(vec4 x) {
    // Cast ray from 3D viewport to surface of image
    // (if $transform does not affect z values, then this
    // can be optimized as simply $transform.map(x) )
    vec4 p1 = $transform(x);
    vec4 p2 = $transform(x + vec4(0, 0, 0.5, 0));
    p1 /= p1.w;
    p2 /= p2.w;
    vec4 d = p2 - p1;
    float f = p2.z / d.z;
    vec4 p3 = p2 - d * f;

    // finally map local to texture coords
    return vec4(p3.xy / image_size, 0, 1);
}


void main()
{
    vec2 texcoord;
    if( method == 0 ) {
        texcoord = v_texcoord;
    }
    else {
        // vertex shader ouptuts clip coordinates;
        // fragment shader maps to texture coordinates
        texcoord = map_local_to_tex(vec4(v_texcoord, 0, 1)).xy;
    }

    gl_FragColor = $color_transform($get_data(texcoord));
}
"""  # noqa

_null_color_transform = 'vec4 pass(vec4 color) { return color; }'
_c2l = 'float cmap(vec4 color) { return (color.r + color.g + color.b) / 3.; }'

_interpolation_template = """
    #include "misc/spatial-filters.frag"
    vec4 texture_lookup_filtered(vec2 texcoord) {
        if(texcoord.x < 0.0 || texcoord.x > 1.0 ||
        texcoord.y < 0.0 || texcoord.y > 1.0) {
            discard;
        }
        return %s($texture, $shape, texcoord);
    }"""

_texture_lookup = """
    vec4 texture_lookup(vec2 texcoord) {
        if(texcoord.x < 0.0 || texcoord.x > 1.0 ||
        texcoord.y < 0.0 || texcoord.y > 1.0) {
            discard;
        }
        vec4 val = texture2D($texture, texcoord);
        // http://stackoverflow.com/questions/11810158/how-to-deal-with-nan-or-inf-in-opengl-es-2-0-shaders
        if (!(val.r <= 0.0 || 0.0 <= val.r)) {
            discard;
        }

        if ($vmin < $vmax) {
            val.r = clamp(val.r, $vmin, $vmax);
        } else {
            val.r = clamp(val.r, $vmax, $vmin);
        }
        val.r = (val.r-$vmin)/($vmax-$vmin);
        val.r = pow(val.r, $gamma);
        val.g = val.r;
        val.b = val.r;

        return val;
    }"""


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


class TiledGeolocatedImageVisual(ImageVisual):
    def __init__(self, data, origin_x, origin_y, cell_width, cell_height,
                 shape=None,
                 tile_shape=(DEFAULT_TILE_HEIGHT, DEFAULT_TILE_WIDTH),
                 texture_shape=(DEFAULT_TEXTURE_HEIGHT, DEFAULT_TEXTURE_WIDTH),
                 wrap_lon=False, projection=DEFAULT_PROJECTION,
                 cmap='viridis', method='tiled', clim='auto', gamma=1.,
                 interpolation='nearest', **kwargs):
        if method != 'tiled':
            raise ValueError("Only 'tiled' method is currently supported")
        method = 'subdivide'
        grid = (1, 1)

        # visual nodes already have names, so be careful
        if not hasattr(self, "name"):
            self.name = kwargs.get("name", None)
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
        self._stride = (0, 0)  # Current stride is None when we are showing the overview
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

        # load 'float packed rgba8' interpolation kernel
        # to load float interpolation kernel use
        # `load_spatial_filters(packed=False)`
        kernel, self._interpolation_names = load_spatial_filters()

        self._kerneltex = Texture2D(kernel, interpolation='nearest')
        # The unpacking can be debugged by changing "spatial-filters.frag"
        # to have the "unpack" function just return the .r component. That
        # combined with using the below as the _kerneltex allows debugging
        # of the pipeline
        # self._kerneltex = Texture2D(kernel, interpolation='linear',
        #                             internalformat='r32f')

        # create interpolation shader functions for available
        # interpolations
        fun = [Function(_interpolation_template % n)
               for n in self._interpolation_names]
        self._interpolation_names = [n.lower()
                                     for n in self._interpolation_names]

        self._interpolation_fun = dict(zip(self._interpolation_names, fun))
        self._interpolation_names.sort()
        self._interpolation_names = tuple(self._interpolation_names)

        # overwrite "nearest" and "bilinear" spatial-filters
        # with  "hardware" interpolation _data_lookup_fn
        self._interpolation_fun['nearest'] = Function(_texture_lookup)
        self._interpolation_fun['bilinear'] = Function(_texture_lookup)

        if interpolation not in self._interpolation_names:
            raise ValueError("interpolation must be one of %s" %
                             ', '.join(self._interpolation_names))

        self._interpolation = interpolation

        # check texture interpolation
        if self._interpolation == 'bilinear':
            texture_interpolation = 'linear'
        else:
            texture_interpolation = 'nearest'

        self._method = method
        self._grid = grid
        self._need_texture_upload = True
        self._need_vertex_update = True
        self._need_colortransform_update = True
        self._need_interpolation_update = True
        self._texture = TextureAtlas2D(self.texture_shape, tile_shape=self.tile_shape,
                                       interpolation=texture_interpolation,
                                       format="LUMINANCE", internalformat="R32F",
                                       )
        self._subdiv_position = VertexBuffer()
        self._subdiv_texcoord = VertexBuffer()

        # impostor quad covers entire viewport
        vertices = np.array([[-1, -1], [1, -1], [1, 1],
                             [-1, -1], [1, 1], [-1, 1]],
                            dtype=np.float32)
        self._impostor_coords = VertexBuffer(vertices)
        self._null_tr = NullTransform()

        self._init_view(self)
        super(ImageVisual, self).__init__(vcode=VERT_SHADER, fcode=FRAG_SHADER)
        self.set_gl_state('translucent', cull_face=False)
        self._draw_mode = 'triangles'

        # define _data_lookup_fn as None, will be setup in
        # self._build_interpolation()
        self._data_lookup_fn = None

        self.gamma = gamma
        self.clim = clim if clim != 'auto' else (np.nanmin(data), np.nanmax(data))
        self._texture_LUT = None
        self.cmap = cmap

        self.overview_info = None
        self.init_overview(data)
        # self.transform = PROJ4Transform(projection, inverse=True)

        self.freeze()

    @property
    def gamma(self):
        return self._gamma

    @gamma.setter
    def gamma(self, gamma):
        self._gamma = gamma if gamma is not None else 1.
        self._need_texture_upload = True
        self.update()

    # @property
    # def clim(self):
    #     return (self._clim if isinstance(self._clim, string_types) else
    #             tuple(self._clim))
    #
    # @clim.setter
    # def clim(self, clim):
    #     if isinstance(clim, string_types):
    #         if clim != 'auto':
    #             raise ValueError('clim must be "auto" if a string')
    #     else:
    #         clim = np.array(clim, float)
    #         if clim.shape != (2,):
    #             raise ValueError('clim must have two elements')
    #     self._clim = clim
    #     # FIXME: Is this supposed to be assigned to something?:
    #     self._data_lookup_fn
    #     self._need_clim_update = True
    #     self.update()

    @property
    def size(self):
        # Added to shader program, but not used by subdivide/tiled method
        return self.shape[-2:][::-1]

    def init_overview(self, data):
        """Create and add a low resolution version of the data that is always
        shown behind the higher resolution image tiles.
        """
        # FUTURE: Actually use this data attribute. For now let the base
        #         think there is data (not None)
        self._data = ArrayProxy(self.ndim, self.shape)
        self.overview_info = nfo = {}
        y_slice, x_slice = self.calc.overview_stride
        nfo["data"] = data[y_slice, x_slice]
        # Update kwargs to reflect the new spatial resolution of the overview image
        nfo["cell_width"] = self.cell_width * x_slice.step
        nfo["cell_height"] = self.cell_height * y_slice.step
        # Tell the texture state that we are adding a tile that should never expire and should always exist
        nfo["texture_tile_index"] = ttile_idx = self.texture_state.add_tile((0, 0, 0), expires=False)
        self._texture.set_tile_data(ttile_idx, self._normalize_data(nfo["data"]))

        # Handle wrapping around the anti-meridian so there is a -180/180 continuous image
        num_tiles = 1 if not self.wrap_lon else 2
        tl = TESS_LEVEL * TESS_LEVEL
        nfo["texture_coordinates"] = np.empty((6 * num_tiles * tl, 2), dtype=np.float32)
        nfo["vertex_coordinates"] = np.empty((6 * num_tiles * tl, 2), dtype=np.float32)
        factor_rez, offset_rez = self.calc.calc_tile_fraction(0, 0,
                                                              Point(np.int64(y_slice.step), np.int64(x_slice.step)))
        nfo["texture_coordinates"][:6 * tl, :2] = self.calc.calc_texture_coordinates(ttile_idx, factor_rez, offset_rez,
                                                                                     tessellation_level=TESS_LEVEL)
        nfo["vertex_coordinates"][:6 * tl, :2] = self.calc.calc_vertex_coordinates(0, 0, y_slice.step, x_slice.step,
                                                                                   factor_rez, offset_rez,
                                                                                   tessellation_level=TESS_LEVEL)
        self._set_vertex_tiles(nfo["vertex_coordinates"], nfo["texture_coordinates"])

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
                # force a copy of the data from the content array (provided by the workspace)
                # to a vispy-compatible contiguous float array
                # this can be a potentially time-expensive operation since content array is
                # often huge and always memory-mapped, so paging may occur
                # we don't want this paging deferred until we're back in the GUI thread pushing data to OpenGL!
                tile_data = np.array(data[y_slice, x_slice], dtype=np.float32)
                tiles_info.append((stride, tiy, tix, tex_tile_idx, tile_data))

        return tiles_info

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
        total_overview_tiles = 0
        if self.overview_info is not None:
            # we should be providing an overview image
            total_overview_tiles = int(
                self.overview_info["vertex_coordinates"].shape[0] / 6 / (TESS_LEVEL * TESS_LEVEL))

        if total_num_tiles <= 0:
            # we aren't looking at this image
            # FIXME: What's the correct way to stop drawing here
            raise RuntimeError("View calculations determined a negative number of tiles are visible")
        elif total_num_tiles > self.num_tex_tiles - total_overview_tiles:
            LOG.warning("Current view sees more tiles than can be held in the GPU")
            # We continue on because there should be an overview image for any tiles that can't be drawn
        total_num_tiles += total_overview_tiles

        tex_coords = np.empty((6 * total_num_tiles * (TESS_LEVEL * TESS_LEVEL), 2), dtype=np.float32)
        vertices = np.empty((6 * total_num_tiles * (TESS_LEVEL * TESS_LEVEL), 2), dtype=np.float32)

        # What tile are we currently describing out of all the tiles being viewed
        used_tile_idx = -1
        # Set up the overview tile
        if self.overview_info is not None:
            # XXX: This completely depends on drawing order, putting it at the end seems to work
            tex_coords[-6 * total_overview_tiles * TESS_LEVEL * TESS_LEVEL:, :] = \
                self.overview_info["texture_coordinates"]
            vertices[-6 * total_overview_tiles * TESS_LEVEL * TESS_LEVEL:, :] = \
                self.overview_info["vertex_coordinates"]

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

        if not img_cmesh[:, 0].size or not img_cmesh[:, 0].size:
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
        except ValueError:
            LOG.error("Could not determine viewable image area for '{}'".format(self.name))
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

    def set_data(self, image):
        """Set the data

        Parameters
        ----------
        image : array-like
            The image data.
        """
        raise NotImplementedError("This image subclass does not support the 'set_data' method")

    def _build_texture(self):
        # _build_texture should not be used in this class, use the 2-step
        # process of '_build_texture_tiles' and '_set_texture_tiles'
        self._set_clim_vars()
        self._need_texture_upload = False

    def _build_vertex_data(self):
        # _build_vertex_data should not be used in this class, use the 2-step
        # process of '_build_vertex_tiles' and '_set_vertex_tiles'
        return

    def _set_clim_vars(self):
        self._data_lookup_fn["vmin"] = self._clim[0]
        self._data_lookup_fn["vmax"] = self._clim[1]
        self._data_lookup_fn["gamma"] = self._gamma
        # self._need_texture_upload = True


TiledGeolocatedImage = create_visual_node(TiledGeolocatedImageVisual)

_rgb_texture_lookup = """
    vec4 texture_lookup(vec2 texcoord) {
        if(texcoord.x < 0.0 || texcoord.x > 1.0 ||
        texcoord.y < 0.0 || texcoord.y > 1.0) {
            discard;
        }
        vec4 val = texture2D($texture, texcoord);
        // http://stackoverflow.com/questions/11810158/how-to-deal-with-nan-or-inf-in-opengl-es-2-0-shaders
        if (!(val.r <= 0.0 || 0.0 <= val.r)) {
            val.r = 0;
            val.g = 0;
            val.b = 0;
            val.a = 0;
            return val;
        }

        if ($vmin < $vmax) {
            val.r = clamp(val.r, $vmin, $vmax);
        } else {
            val.r = clamp(val.r, $vmax, $vmin);
        }
        val.r = (val.r-$vmin)/($vmax-$vmin);
        val.r = pow(val.r, $gamma);
        val.g = val.r;
        val.b = val.r;

        return val;
    }"""


class CompositeLayerVisual(TiledGeolocatedImageVisual):
    VERT_SHADER = None
    FRAG_SHADER = None

    def __init__(self, data_arrays, origin_x, origin_y, cell_width, cell_height,
                 shape=None,
                 tile_shape=(DEFAULT_TILE_HEIGHT, DEFAULT_TILE_WIDTH),
                 texture_shape=(DEFAULT_TEXTURE_HEIGHT, DEFAULT_TEXTURE_WIDTH),
                 wrap_lon=False,
                 cmap='viridis', method='tiled', clim='auto', gamma=None,
                 interpolation='nearest', **kwargs):
        # projection properties to be filled in later
        self.cell_width = None
        self.cell_height = None
        self.origin_x = None
        self.origin_y = None
        self.shape = None

        if method != 'tiled':
            raise ValueError("Only 'tiled' method is currently supported")
        method = 'subdivide'
        grid = (1, 1)

        # visual nodes already have names, so be careful
        if not hasattr(self, "name"):
            self.name = kwargs.get("name", None)
        self._viewable_mesh_mask = None
        self._ref1 = None
        self._ref2 = None

        self.texture_shape = texture_shape
        self.tile_shape = tile_shape
        self.num_tex_tiles = self.texture_shape[0] * self.texture_shape[1]
        self._stride = 0  # Current stride is None when we are showing the overview
        self._latest_tile_box = None
        self.wrap_lon = wrap_lon
        self._tiles = {}

        # What tiles have we used and can we use (each texture uses the same 'state')
        self.texture_state = TextureTileState(self.num_tex_tiles)

        self.set_channels(data_arrays, shape=shape,
                          cell_width=cell_width, cell_height=cell_height,
                          origin_x=origin_x, origin_y=origin_y)
        self.ndim = len(self.shape) or [x for x in data_arrays if x is not None][0].ndim
        self.num_channels = len(data_arrays)

        # load 'float packed rgba8' interpolation kernel
        # to load float interpolation kernel use
        # `load_spatial_filters(packed=False)`
        kernel, self._interpolation_names = load_spatial_filters()

        self._kerneltex = Texture2D(kernel, interpolation='nearest')
        # The unpacking can be debugged by changing "spatial-filters.frag"
        # to have the "unpack" function just return the .r component. That
        # combined with using the below as the _kerneltex allows debugging
        # of the pipeline
        # self._kerneltex = Texture2D(kernel, interpolation='linear',
        #                             internalformat='r32f')

        # create interpolation shader functions for available
        # interpolations
        fun = [Function(_interpolation_template % n)
               for n in self._interpolation_names]
        self._interpolation_names = [n.lower()
                                     for n in self._interpolation_names]

        self._interpolation_fun = dict(zip(self._interpolation_names, fun))
        self._interpolation_names.sort()
        self._interpolation_names = tuple(self._interpolation_names)

        # overwrite "nearest" and "bilinear" spatial-filters
        # with  "hardware" interpolation _data_lookup_fn
        self._interpolation_fun['nearest'] = Function(_texture_lookup)
        self._interpolation_fun['bilinear'] = Function(_texture_lookup)

        if interpolation not in self._interpolation_names:
            raise ValueError("interpolation must be one of %s" %
                             ', '.join(self._interpolation_names))

        self._interpolation = interpolation

        # check texture interpolation
        if self._interpolation == 'bilinear':
            texture_interpolation = 'linear'
        else:
            texture_interpolation = 'nearest'

        self._method = method
        self._grid = grid
        self._need_texture_upload = True
        self._need_vertex_update = True
        self._need_colortransform_update = False
        self._need_interpolation_update = True
        self._textures = [TextureAtlas2D(self.texture_shape, tile_shape=self.tile_shape,
                                         interpolation=texture_interpolation,
                                         format="LUMINANCE", internalformat="R32F",
                                         ) for i in range(self.num_channels)
                          ]
        self._subdiv_position = VertexBuffer()
        self._subdiv_texcoord = VertexBuffer()

        # impostor quad covers entire viewport
        vertices = np.array([[-1, -1], [1, -1], [1, 1],
                             [-1, -1], [1, 1], [-1, 1]],
                            dtype=np.float32)
        self._impostor_coords = VertexBuffer(vertices)
        self._null_tr = NullTransform()

        self._init_view(self)
        if self.VERT_SHADER is None or self.FRAG_SHADER is None:
            raise RuntimeError("No shader specified for this subclass")
        super(ImageVisual, self).__init__(vcode=self.VERT_SHADER, fcode=self.FRAG_SHADER)
        self.set_gl_state('translucent', cull_face=False)
        self._draw_mode = 'triangles'

        # define _data_lookup_fn as None, will be setup in
        # self._build_interpolation()
        self._data_lookup_fns = [Function(_rgb_texture_lookup) for i in range(self.num_channels)]

        if isinstance(clim, str):
            if clim != 'auto':
                raise ValueError("C-limits can only be 'auto' or 2 floats for each provided channel")
            clim = [clim] * self.num_channels
        if not isinstance(cmap, (tuple, list)):
            cmap = [cmap] * self.num_channels

        assert (len(clim) == self.num_channels)
        assert (len(cmap) == self.num_channels)
        _clim = []
        _cmap = []
        for idx in range(self.num_channels):
            cl = clim[idx]
            if cl == 'auto':
                _clim.append((np.nanmin(data_arrays[idx]), np.nanmax(data_arrays[idx])))
            elif cl is None:
                # Color limits don't matter (either empty channel array or other)
                _clim.append((0., 1.))
            elif isinstance(cl, tuple) and len(cl) == 2:
                _clim.append(cl)
            else:
                raise ValueError("C-limits must be a 2-element tuple or the string 'auto' for each channel provided")

            cm = cmap[idx]
            _cmap.append(cm)
        self.clim = _clim
        self._texture_LUT = None
        self.gamma = gamma if gamma is not None else (1.,) * self.num_channels
        # only set colormap if it isn't None
        # (useful when a subclass's shader doesn't expect a colormap)
        if _cmap[0] is not None:
            self.cmap = _cmap[0]

        self.overview_info = None
        self.init_overview(data_arrays)

        self.freeze()

    def set_channels(self, data_arrays, shape=None,
                     cell_width=None, cell_height=None,
                     origin_x=None, origin_y=None, **kwargs):
        assert (shape or data_arrays is not None), "`data` or `shape` must be provided"
        if cell_width is not None:
            self.cell_width = cell_width
        if cell_height:
            self.cell_height = cell_height  # Note: cell_height is usually negative
        if origin_x:
            self.origin_x = origin_x
        if origin_y:
            self.origin_y = origin_y
        self.shape = shape or max(data.shape for data in data_arrays if data is not None)
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

    def init_overview(self, data_arrays):
        """Create and add a low resolution version of the data that is always
        shown behind the higher resolution image tiles.
        """
        # FUTURE: Actually use this data attribute. For now let the base
        #         think there is data (not None)
        self._data = ArrayProxy(self.ndim, self.shape)
        self.overview_info = nfo = {}
        y_slice, x_slice = self.calc.overview_stride
        # Update kwargs to reflect the new spatial resolution of the overview image
        nfo["cell_width"] = self.cell_width * x_slice.step
        nfo["cell_height"] = self.cell_height * y_slice.step
        # Tell the texture state that we are adding a tile that should never expire and should always exist
        nfo["texture_tile_index"] = ttile_idx = self.texture_state.add_tile((0, 0, 0), expires=False)
        for idx, data in enumerate(data_arrays):
            if data is not None:
                _y_slice, _x_slice = self.calc.calc_overview_stride(image_shape=Point(data.shape[0], data.shape[1]))
                overview_data = data[_y_slice, _x_slice]
            else:
                overview_data = None
            self._textures[idx].set_tile_data(ttile_idx, self._normalize_data(overview_data))

        # Handle wrapping around the anti-meridian so there is a -180/180 continuous image
        num_tiles = 1 if not self.wrap_lon else 2
        tl = TESS_LEVEL * TESS_LEVEL
        nfo["texture_coordinates"] = np.empty((6 * num_tiles * tl, 2), dtype=np.float32)
        nfo["vertex_coordinates"] = np.empty((6 * num_tiles * tl, 2), dtype=np.float32)
        factor_rez, offset_rez = self.calc.calc_tile_fraction(
            0, 0, Point(np.int64(y_slice.step), np.int64(x_slice.step)))
        nfo["texture_coordinates"][:6 * tl, :2] = self.calc.calc_texture_coordinates(ttile_idx, factor_rez, offset_rez,
                                                                                     tessellation_level=TESS_LEVEL)
        nfo["vertex_coordinates"][:6 * tl, :2] = self.calc.calc_vertex_coordinates(0, 0, y_slice.step, x_slice.step,
                                                                                   factor_rez, offset_rez,
                                                                                   tessellation_level=TESS_LEVEL)
        self._set_vertex_tiles(nfo["vertex_coordinates"], nfo["texture_coordinates"])

    @property
    def gamma(self):
        return self._gamma

    @gamma.setter
    def gamma(self, gamma):
        assert isinstance(gamma, (tuple, list))
        assert len(gamma) == self.num_channels
        self._gamma = tuple(x if x is not None else 1. for x in gamma)
        self._need_texture_upload = True
        self.update()

    @property
    def clim(self):
        return (self._clim if isinstance(self._clim, string_types) else
                tuple(self._clim))

    @clim.setter
    def clim(self, clim):
        if isinstance(clim, string_types):
            if clim != 'auto':
                raise ValueError('clim must be "auto" if a string')
        else:
            # set clim to 0 and 1 for non-existent arrays
            clim = [c if c is not None else (0., 1.) for c in clim]
            clim = np.array(clim, float)
            if clim.shape != (self.num_channels, 2) and clim.shape != (2,):
                raise ValueError('clim must have either 2 elements or 6 (2 for each channel)')
            elif clim.shape == (2,):
                clim = np.array([clim, clim, clim], float)
        self._clim = clim
        self._need_texture_upload = True
        self.update()

    def _set_clim_vars(self):
        for idx, lookup_fn in enumerate(self._data_lookup_fns):
            lookup_fn["vmin"] = self._clim[idx, 0]
            lookup_fn["vmax"] = self._clim[idx, 1]
            lookup_fn["gamma"] = self._gamma[idx]

    def _build_interpolation(self):
        # assumes 'nearest' interpolation
        for idx, lookup_fn in enumerate(self._data_lookup_fns):
            self.shared_program.frag['get_data_%d' % (idx + 1,)] = lookup_fn
            lookup_fn['texture'] = self._textures[idx]
        self._need_interpolation_update = False

    def _build_texture_tiles(self, data, stride, tile_box):
        """Prepare and organize strided data in to individual tiles with associated information.
        """
        data = [self._normalize_data(d) for d in data]

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
                y_slice, x_slice = self.calc.calc_tile_slice(tiy, tix, tuple(stride))
                textures_data = []
                for chn_idx in range(self.num_channels):
                    # force a copy of the data from the content array (provided by the workspace)
                    # to a vispy-compatible contiguous float array
                    # this can be a potentially time-expensive operation since content array is often huge and
                    # always memory-mapped, so paging may occur
                    # we don't want this paging deferred until we're back in the GUI thread pushing data to OpenGL!
                    if data[chn_idx] is None:
                        # we need to fill the texture with NaNs instead of actual data
                        tile_data = None
                    else:
                        tile_data = np.array(data[chn_idx][y_slice, x_slice], dtype=np.float32)
                    textures_data.append(tile_data)
                tiles_info.append((stride, tiy, tix, tex_tile_idx, textures_data))

        return tiles_info

    def _set_texture_tiles(self, tiles_info):
        for tile_info in tiles_info:
            stride, tiy, tix, tex_tile_idx, data_arrays = tile_info
            for idx, data in enumerate(data_arrays):
                self._textures[idx].set_tile_data(tex_tile_idx, data)

    def _get_stride(self, view_box):
        s = self.calc.calc_stride(view_box, texture=self._lowest_rez)
        return Point(np.int64(s[0] * self._lowest_factor), np.int64(s[1] * self._lowest_factor))


CompositeLayer = create_visual_node(CompositeLayerVisual)

RGB_VERT_SHADER = """
uniform int method;  // 0=subdivide, 1=impostor
attribute vec2 a_position;
attribute vec2 a_texcoord;
varying vec2 v_texcoord;

void main() {
    v_texcoord = a_texcoord;
    gl_Position = $transform(vec4(a_position, 0., 1.));
}
"""

RGB_FRAG_SHADER = """
uniform vec2 image_size;
uniform int method;  // 0=subdivide, 1=impostor
uniform sampler2D u_texture;
varying vec2 v_texcoord;

vec4 map_local_to_tex(vec4 x) {
    // Cast ray from 3D viewport to surface of image
    // (if $transform does not affect z values, then this
    // can be optimized as simply $transform.map(x) )
    vec4 p1 = $transform(x);
    vec4 p2 = $transform(x + vec4(0, 0, 0.5, 0));
    p1 /= p1.w;
    p2 /= p2.w;
    vec4 d = p2 - p1;
    float f = p2.z / d.z;
    vec4 p3 = p2 - d * f;

    // finally map local to texture coords
    return vec4(p3.xy / image_size, 0, 1);
}


void main()
{
    vec2 texcoord;
    if( method == 0 ) {
        texcoord = v_texcoord;
    }
    else {
        // vertex shader ouptuts clip coordinates;
        // fragment shader maps to texture coordinates
        texcoord = map_local_to_tex(vec4(v_texcoord, 0, 1)).xy;
    }

    vec4 r_tmp, g_tmp, b_tmp;
    r_tmp = $get_data_1(texcoord);
    g_tmp = $get_data_2(texcoord);
    b_tmp = $get_data_3(texcoord);

    // Make the pixel transparent if all of the values are NaN/fill values
    if (r_tmp.a == 0 && g_tmp.a == 0 && b_tmp.a == 0) {
        gl_FragColor.a = 0;
    } else {
        gl_FragColor.a = 1;
    }
    gl_FragColor.r = r_tmp.r;
    gl_FragColor.g = g_tmp.r;
    gl_FragColor.b = b_tmp.r;
}
"""  # noqa


class RGBCompositeLayerVisual(CompositeLayerVisual):
    VERT_SHADER = RGB_VERT_SHADER
    FRAG_SHADER = RGB_FRAG_SHADER


RGBCompositeLayer = create_visual_node(RGBCompositeLayerVisual)


class ShapefileLinesVisual(LineVisual):
    def __init__(self, filepath, double=False, **kwargs):
        LOG.debug("Using border shapefile '%s'", filepath)
        self.sf = shapefile.Reader(filepath)

        LOG.info("Loading boundaries: %s", datetime.utcnow().isoformat(" "))
        # Prepare the arrays
        total_points = 0
        total_parts = 0
        for idx, one_shape in enumerate(self.sf.iterShapes()):
            total_points += len(one_shape.points)
            total_parts += len(one_shape.parts)
        vertex_buffer = np.empty((total_points * 2 - total_parts * 2, 2), dtype=np.float32)
        prev_idx = 0
        for idx, one_shape in enumerate(self.sf.iterShapes()):
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
