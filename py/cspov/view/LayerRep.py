#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LayerRep.py
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
import os
import sys
import logging
import unittest
import argparse

import shapefile

from vispy.scene.visuals import create_visual_node
from vispy.visuals import LineVisual, ImageVisual, CompoundVisual, Visual
from vispy.color import get_colormap
from vispy.ext.six import string_types
import numpy as np
from datetime import datetime
from pyproj import Proj

from cspov.common import (DEFAULT_X_PIXEL_SIZE,
                          DEFAULT_Y_PIXEL_SIZE,
                          DEFAULT_ORIGIN_X,
                          DEFAULT_ORIGIN_Y,
                          DEFAULT_PROJECTION,
                          DEFAULT_TILE_HEIGHT,
                          DEFAULT_TILE_WIDTH,
                          DEFAULT_TEXTURE_HEIGHT,
                          DEFAULT_TEXTURE_WIDTH,
                          WORLD_EXTENT_BOX,
                          C_EQ,
                          box, pnt, rez, vue,
                          MercatorTileCalc
                          )
from cspov.view.Program import TextureAtlas2D, Texture2D
# The below imports are needed because we subclassed the ImageVisual
from vispy.visuals.shaders import Function
from vispy.visuals.transforms import NullTransform
from vispy.gloo import VertexBuffer
from vispy.io.datasets import load_spatial_filters
from vispy.visuals.shaders import FunctionChain

__author__ = 'rayg'
__docformat__ = 'reStructuredText'

LOG = logging.getLogger(__name__)


class GeolocatedImageVisual(ImageVisual):
    """Visual class for image data that is geolocated and should be referenced that way when displayed.

    Note: VisPy separates the Visual from the Node used in the SceneGraph by dynamically creating
    the `vispy.scene.visuals.Image` class. Use the GeolocatedImage class with scenes.
    """
    def __init__(self,
                 origin_x=DEFAULT_ORIGIN_X, origin_y=DEFAULT_ORIGIN_Y,
                 cell_width=DEFAULT_X_PIXEL_SIZE, cell_height=DEFAULT_Y_PIXEL_SIZE,
                 double=False, **kwargs):
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.cell_width = cell_width
        self.cell_height = cell_height
        self.double = double
        super(GeolocatedImageVisual, self).__init__(**kwargs)

    @classmethod
    def from_geotiff(cls, geotiff_filepath, **kwargs):
        import gdal
        gtiff = gdal.Open(geotiff_filepath)
        ox, cw, _, oy, _, ch = gtiff.GetGeoTransform()
        img_data = gtiff.GetRasterBand(1).ReadAsArray()
        return cls(data=img_data, origin_x=ox, origin_y=oy, cell_width=cw, cell_height=ch, **kwargs)

    def _build_vertex_data(self):
        """Rebuild the vertex buffers used for rendering the image when using
        the subdivide method.

        CSPOV Note: Copied from 0.5.0dev original ImageVisual class
        """
        grid = self._grid
        w = 1.0 / grid[1]
        h = 1.0 / grid[0]

        quad = np.array([[0, 0, 0], [w, 0, 0], [w, h, 0],
                         [0, 0, 0], [w, h, 0], [0, h, 0]],
                        dtype=np.float32)
        quads = np.empty((grid[1], grid[0], 6, 3), dtype=np.float32)
        quads[:] = quad

        mgrid = np.mgrid[0.:grid[1], 0.:grid[0]].transpose(1, 2, 0)
        mgrid = mgrid[:, :, np.newaxis, :]
        mgrid[..., 0] *= w
        mgrid[..., 1] *= h

        quads[..., :2] += mgrid
        tex_coords = quads.reshape(grid[1]*grid[0]*6, 3)
        tex_coords = tex_coords[:, :2]
        vertices = tex_coords * self.size

        # FUTURE: This should probably be done as a transform
        # vertices = vertices.astype('float32')
        vertices[:, 0] *= self.cell_width
        vertices[:, 0] += self.origin_x
        vertices[:, 1] *= self.cell_height
        vertices[:, 1] += self.origin_y
        if self.double:
            orig_points = vertices.shape[0]
            vertices = np.concatenate((vertices, vertices), axis=0)
            tex_coords = np.concatenate((tex_coords, tex_coords), axis=0)
            vertices[orig_points:, 0] += C_EQ
        self._subdiv_position.set_data(np.require(vertices, dtype=np.float32, requirements=['C']))
        self._subdiv_texcoord.set_data(np.require(tex_coords, dtype=np.float32, requirements=['C']))


GeolocatedImage = create_visual_node(GeolocatedImageVisual)


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
                 wrap_lon=False,
                 cmap='viridis', method='tiled', clim='auto', interpolation='nearest', **kwargs):
        if method != 'tiled':
            raise ValueError("Only 'tiled' method is currently supported")
        method = 'subdivide'
        grid = (1, 1)

        # visual nodes already have names, so be careful
        if not hasattr(self, "name"):
            self.name = kwargs.get("name", None)
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.cell_width = cell_width
        self.cell_height = cell_height  # Note: cell_height is usually negative
        self.texture_shape = texture_shape
        self.tile_shape = tile_shape
        self.num_tex_tiles = self.texture_shape[0] * self.texture_shape[1]
        self._stride = 0  # Current stride is None when we are showing the overview
        self._latest_tile_box = None
        self.wrap_lon = wrap_lon
        self._tiles = {}
        assert (shape or data is not None), "`data` or `shape` must be provided"
        self.shape = shape or data.shape
        self.ndim = len(self.shape) or data.ndim

        # Where does this image lie in this lonely world
        self.calc = MercatorTileCalc(
            self.name,
            self.shape,
            pnt(x=self.origin_x, y=self.origin_y),
            rez(dy=abs(self.cell_height), dx=abs(self.cell_width)),
            self.tile_shape,
            self.texture_shape,
            wrap_lon=self.wrap_lon
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
        self._need_clim_update = True
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

        self.clim = clim if clim != 'auto' else (np.nanmin(data), np.nanmax(data))
        self.cmap = cmap

        self.overview_info = None
        self.init_overview(data)

        self.freeze()

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
            clim = np.array(clim, float)
            if clim.shape != (2,):
                raise ValueError('clim must have two elements')
        self._clim = clim
        # FIXME: Is this supposed to be assigned to something?:
        self._data_lookup_fn
        self._need_clim_update = True
        self.update()

    @property
    def size(self):
        # Added to shader program, but not used by subdivide/tiled method
        return self.shape[-2:][::-1]

    def init_overview(self, data):
        """Create and add a low resolution version of the data that is always
        shown behind the higher resolution image tiles.
        """
        self.overview_info = nfo = {}
        y_slice, x_slice = self.calc.overview_stride()
        # overview_stride = (int(image_shape[0]/tile_shape[0]), int(image_shape[1]/tile_shape[1]))
        nfo["data"] = data[y_slice, x_slice]
        # Update kwargs to reflect the new spatial resolution of the overview image
        nfo["cell_width"] = self.cell_width * x_slice.step
        nfo["cell_height"] = self.cell_height * y_slice.step
        # Tell the texture state that we are adding a tile that should never expire and should always exist
        nfo["texture_tile_index"] = ttile_idx = self.texture_state.add_tile((0, 0, 0), expires=False)
        self._texture.set_tile_data(ttile_idx, self._normalize_data(nfo["data"]))

        # Handle wrapping around the anti-meridian so there is a -180/180 continuous image
        num_tiles = 1 if not self.wrap_lon else 2
        nfo["texture_coordinates"] = np.empty((6 * num_tiles, 2), dtype=np.float32)
        nfo["vertex_coordinates"] = np.empty((6 * num_tiles, 2), dtype=np.float32)
        nfo["texture_coordinates"][:6, :2] = self.calc.calc_texture_coordinates(ttile_idx)
        nfo["vertex_coordinates"][:6, :2] = self.calc.calc_vertex_coordinates(0, 0, y_slice.step, x_slice.step)
        if self.wrap_lon:
            nfo["texture_coordinates"][6:12, :2] = nfo["texture_coordinates"][:6, :2]
            nfo["vertex_coordinates"][6:12, :2] = nfo["vertex_coordinates"][:6, :2]
            # increase the second set of X coordinates by the circumference of the earth
            nfo["vertex_coordinates"][6:12, 0] += nfo["cell_width"] * nfo["data"].shape[1]
        self._set_vertex_tiles(nfo["vertex_coordinates"], nfo["texture_coordinates"])

    def _normalize_data(self, data):
        if data.dtype == np.float64:
            data = data.astype(np.float32)

        return data

    def _build_texture_tiles(self, data, stride, tile_box):
        """Prepare and organize strided data in to individual tiles with associated information.
        """
        data = self._normalize_data(data)

        LOG.debug("Uploading texture data for %d tiles (%r)", (tile_box.b - tile_box.t) * (tile_box.r - tile_box.l), tile_box)
        max_tiles = self.calc.max_tiles_available(stride)

        # Tiles start at upper-left so go from top to bottom
        tiles_info = []
        for tiy in range(tile_box.t, tile_box.b):
            for tix in range(tile_box.l, tile_box.r):
                virt_tix = tix % max_tiles[1]
                already_in = (stride, tiy, virt_tix) in self.texture_state
                # Update the age if already in there
                # Assume that texture_state does not change from the main thread if this is run in another
                tex_tile_idx = self.texture_state.add_tile((stride, tiy, virt_tix))
                if already_in:
                    # FIXME: we should make a list/set of the tiles we need to add before this
                    continue

                # Assume we were given a total image worth of this stride
                y_start = tiy * self.tile_shape[0]
                y_end = y_start + self.tile_shape[0]
                x_start = virt_tix * self.tile_shape[1]
                x_end = x_start + self.tile_shape[1]
                # force a copy of the data from the content array (provided by the workspace) to a vispy-compatible contiguous float array
                # this can be a potentially time-expensive operation since content array is often huge and always memory-mapped, so paging may occur
                # we don't want this paging deferred until we're back in the GUI thread pushing data to OpenGL!
                tile_data = np.array(data[y_start: y_end, x_start: x_end], dtype=np.float32)
                tiles_info.append((stride, tiy, tix, tex_tile_idx, tile_data))

        return tiles_info

    def _set_texture_tiles(self, tiles_info):
        for tile_info in tiles_info:
            stride, tiy, tix, tex_tile_idx, data = tile_info
            self._texture.set_tile_data(tex_tile_idx, data)

    def _build_vertex_tiles(self, preferred_stride, tile_box):
        """Rebuild the vertex buffers used for rendering the image when using
        the subdivide method.

        CSPOV Note: Copied from 0.5.0dev original ImageVisual class
        """
        max_tiles = self.calc.max_tiles_available(preferred_stride)
        total_num_tiles = (tile_box.b - tile_box.t) * (tile_box.r - tile_box.l)

        total_overview_tiles = 0
        if self.overview_info is not None:
            # we should be providing an overview image
            total_overview_tiles = int(self.overview_info["vertex_coordinates"].shape[0] / 6)

        if total_num_tiles <= 0:
            # we aren't looking at this image
            # FIXME: What's the correct way to stop drawing here
            raise RuntimeError("View calculations determined a negative number of tiles are visible")
        elif total_num_tiles > self.num_tex_tiles - total_overview_tiles:
            LOG.warning("Current view sees more tiles than can be held in the GPU")
            # We continue on because there should be an overview image for any tiles that can't be drawn
        total_num_tiles += total_overview_tiles

        tex_coords = np.empty((6 * total_num_tiles, 2), dtype=np.float32)
        vertices = np.empty((6 * total_num_tiles, 2), dtype=np.float32)

        # What tile are we currently describing out of all the tiles being viewed
        used_tile_idx = -1
        # Set up the overview tile
        if self.overview_info is not None:
            # XXX: This completely depends on drawing order, putting it at the end seems to work
            tex_coords[-6 * total_overview_tiles:, :] = self.overview_info["texture_coordinates"]
            vertices[-6 * total_overview_tiles:, :] = self.overview_info["vertex_coordinates"]

        # preferred_stride = 1
        LOG.debug("Building vertex data for %d tiles (%r)", total_num_tiles, tile_box)
        # Tiles start at upper-left so go from top to bottom
        for tiy in range(tile_box.t, tile_box.b):
            for tix in range(tile_box.l, tile_box.r):
                # Update the index here because we have multiple exit/continuation points
                used_tile_idx += 1

                # Tiles on the "second" longitude wrapped image are the same tiles in the first/original images
                virt_tix = tix % max_tiles[1]

                # Check if the tile we want to draw is actually in the GPU, if not (atlas too small?) fill with zeros and keep going
                if (preferred_stride, tiy, virt_tix) not in self.texture_state:
                    # THIS SHOULD NEVER HAPPEN IF TEXTURE BUILDING IS DONE CORRECTLY AND THE ATLAS IS BIG ENOUGH
                    tex_coords[used_tile_idx*6: (used_tile_idx+1)*6, :] = 0
                    vertices[used_tile_idx*6: (used_tile_idx+1)*6, :] = 0
                    continue

                # we should have already loaded the texture data in to the GPU so get the index of that texture
                tex_tile_idx = self.texture_state[(preferred_stride, tiy, virt_tix)]
                tex_coords[used_tile_idx*6: (used_tile_idx+1)*6, :] = self.calc.calc_texture_coordinates(tex_tile_idx)
                vertices[used_tile_idx*6: (used_tile_idx+1)*6, :] = self.calc.calc_vertex_coordinates(tiy, tix, preferred_stride, preferred_stride)

        return vertices, tex_coords

    def _set_vertex_tiles(self, vertices, tex_coords):
        self._subdiv_position.set_data(vertices.astype('float32'))
        self._subdiv_texcoord.set_data(tex_coords.astype('float32'))

    def get_view_box(self):
        ll_corner, ur_corner = self.transforms.get_transform().imap([(-1, -1, 1), (1, 1, 1)])
        # How many tiles should be contained in this view?
        view_box = box(
            b=ll_corner[1],
            l=ll_corner[0],
            t=ur_corner[1],
            r=ur_corner[0]
        )
        view_box = vue(*view_box, dy=(view_box.t - view_box.b)/self.canvas.size[1], dx=(view_box.r - view_box.l)/self.canvas.size[0])
        return view_box

    def assess(self):
        """Determine if a retile is needed.

        Tell workspace we will be needed
        """
        view_box = self.get_view_box()
        preferred_stride = self.calc.calc_stride(view_box)
        _, tile_box = self.calc.visible_tiles(view_box, stride=preferred_stride, extra_tiles_box=box(1, 1, 1, 1))
        num_tiles = (tile_box.b - tile_box.t) * (tile_box.r - tile_box.l)
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
        raise NotImplementedError("_build_texture is not implemented in this subclass, use the 2-step process of '_build_texture_tiles' and '_set_texture_tiles'")

    def _build_vertex_data(self, transforms):
        raise NotImplementedError("_build_vertex_data is not implemented in this subclass, use the 2-step process of '_build_vertex_tiles' and '_set_vertex_tiles'")

    def _build_color_transform(self):
        if self.ndim == 2 or self.shape[2] == 1:
            fun = FunctionChain(None, [Function(_c2l),
                                       Function(self._cmap.glsl_map)])
        else:
            fun = Function(_null_color_transform)
        self.shared_program.frag['color_transform'] = fun
        self._need_colortransform_update = False

    def _set_clim_vars(self):
        self._data_lookup_fn["vmin"] = self._clim[0]
        self._data_lookup_fn["vmax"] = self._clim[1]
        self._need_clim_update = False

    def _prepare_draw(self, view):
        if self._need_interpolation_update:
            self._build_interpolation()

        # if self._need_texture_upload:
        #     self._build_texture()

        if self._need_clim_update:
            self._set_clim_vars()

        if self._need_colortransform_update:
            self._build_color_transform()

        # if self._need_vertex_update:
        #     self._build_vertex_data()

        if view._need_method_update:
            self._update_method(view)

TiledGeolocatedImage = create_visual_node(TiledGeolocatedImageVisual)


class CompositeLayerVisual(TiledGeolocatedImageVisual):
    VERT_SHADER = None
    FRAG_SHADER = None

    def __init__(self, data_arrays, origin_x, origin_y, cell_width, cell_height,
                 shape=None,
                 tile_shape=(DEFAULT_TILE_HEIGHT, DEFAULT_TILE_WIDTH),
                 texture_shape=(DEFAULT_TEXTURE_HEIGHT, DEFAULT_TEXTURE_WIDTH),
                 wrap_lon=False,
                 cmap='viridis', method='tiled', clim='auto', interpolation='nearest', **kwargs):
        if method != 'tiled':
            raise ValueError("Only 'tiled' method is currently supported")
        method = 'subdivide'
        grid = (1, 1)

        # visual nodes already have names, so be careful
        if not hasattr(self, "name"):
            self.name = kwargs.get("name", None)
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.cell_width = cell_width
        self.cell_height = cell_height  # Note: cell_height is usually negative
        self.texture_shape = texture_shape
        self.tile_shape = tile_shape
        self.num_tex_tiles = self.texture_shape[0] * self.texture_shape[1]
        self._stride = 0  # Current stride is None when we are showing the overview
        self._latest_tile_box = None
        self.wrap_lon = wrap_lon
        self._tiles = {}

        # What tiles have we used and can we use (each texture uses the same 'state')
        self.texture_state = TextureTileState(self.num_tex_tiles)

        self.set_channels(data_arrays, shape=shape)
        self.ndim = len(self.shape) or data_arrays[0].ndim
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
        self._need_colortransform_update = True
        self._need_clim_update = True
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
        self._data_lookup_fns = [Function(_texture_lookup) for i in range(self.num_channels)]

        if isinstance(clim, str):
            if clim != 'auto':
                raise ValueError("C-limits can only be 'auto' or 2 floats for each provided channel")
            clim = [clim] * self.num_channels
        if not isinstance(cmap, (tuple, list)):
            cmap = [cmap] * self.num_channels

        assert(len(clim) == self.num_channels)
        assert(len(cmap) == self.num_channels)
        _clim = []
        _cmap = []
        for idx in range(self.num_channels):
            cl = clim[idx]
            if cl == 'auto':
                _clim.append((np.nanmin(data_arrays[idx]), np.nanmax(data_arrays[idx])))
            elif isinstance(cl, tuple) and len(cl) == 2:
                _clim.append(cl)
            else:
                raise ValueError("C-limits must be a 2-element tuple or the string 'auto' for each channel provided")

            cm = cmap[idx]
            _cmap.append(cm)
        self.clim = _clim
        self.cmap = _cmap[0]

        self.overview_info = None
        self.init_overview(data_arrays)

        self.freeze()

    def set_channels(self, data_arrays, shape=None):
        assert (shape or data_arrays is not None), "`data` or `shape` must be provided"
        self.shape = shape or max(data.shape for data in data_arrays)
        # how many of the higher resolution channel tiles (smaller geographic area) make
        # up a low resolution channel tile
        self._channel_factors = tuple(round(self.shape[0] / float(chn.shape[0])) for chn in data_arrays)
        self._lowest_factor = max(self._channel_factors)
        self._lowest_rez = rez(abs(self.cell_height * self._lowest_factor), abs(self.cell_width * self._lowest_factor))

        # Where does this image lie in this lonely world
        self.calc = MercatorTileCalc(
            self.name,
            self.shape,
            pnt(x=self.origin_x, y=self.origin_y),
            rez(dy=abs(self.cell_height), dx=abs(self.cell_width)),
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
        self.overview_info = nfo = {}
        y_slice, x_slice = self.calc.overview_stride()
        # Update kwargs to reflect the new spatial resolution of the overview image
        nfo["cell_width"] = self.cell_width * x_slice.step
        nfo["cell_height"] = self.cell_height * y_slice.step
        # Tell the texture state that we are adding a tile that should never expire and should always exist
        nfo["texture_tile_index"] = ttile_idx = self.texture_state.add_tile((0, 0, 0), expires=False)
        for idx, data in enumerate(data_arrays):
            _y_slice, _x_slice = self.calc.overview_stride(image_shape=data.shape)
            overview_data = data[_y_slice, _x_slice]
            self._textures[idx].set_tile_data(ttile_idx, self._normalize_data(overview_data))

        # Handle wrapping around the anti-meridian so there is a -180/180 continuous image
        num_tiles = 1 if not self.wrap_lon else 2
        nfo["texture_coordinates"] = np.empty((6 * num_tiles, 2), dtype=np.float32)
        nfo["vertex_coordinates"] = np.empty((6 * num_tiles, 2), dtype=np.float32)
        nfo["texture_coordinates"][:6, :2] = self.calc.calc_texture_coordinates(ttile_idx)
        nfo["vertex_coordinates"][:6, :2] = self.calc.calc_vertex_coordinates(0, 0, y_slice.step, x_slice.step)
        if self.wrap_lon:
            nfo["texture_coordinates"][6:12, :2] = nfo["texture_coordinates"][:6, :2]
            nfo["vertex_coordinates"][6:12, :2] = nfo["vertex_coordinates"][:6, :2]
            # increase the second set of X coordinates by the circumference of the earth
            nfo["vertex_coordinates"][6:12, 0] += nfo["cell_width"] * nfo["data"].shape[1]
        self._set_vertex_tiles(nfo["vertex_coordinates"], nfo["texture_coordinates"])

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
            clim = np.array(clim, float)
            if clim.shape != (self.num_channels, 2) and clim.shape != (2,):
                raise ValueError('clim must have either 2 elements or 6 (2 for each channel)')
            elif clim.shape == (2,):
                clim = np.array([clim, clim, clim], float)
        self._clim = clim
        self._need_clim_update = True
        self.update()

    def _set_clim_vars(self):
        for idx, lookup_fn in enumerate(self._data_lookup_fns):
            lookup_fn["vmin"] = self._clim[idx, 0]
            lookup_fn["vmax"] = self._clim[idx, 1]
            self._need_clim_update = False

    def _build_color_transform(self):
        if self.ndim == 2 or self.shape[2] == 1:
            fun = FunctionChain(None, [Function(_c2l),
                                       Function(self._cmap.glsl_map)])
        else:
            fun = Function(_null_color_transform)
        # self.shared_program.frag['color_transform'] = fun
        self._need_colortransform_update = False

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

        LOG.debug("Uploading texture data for %d tiles (%r)", (tile_box.b - tile_box.t) * (tile_box.r - tile_box.l), tile_box)
        max_tiles = self.calc.max_tiles_available(stride)

        # Tiles start at upper-left so go from top to bottom
        tiles_info = []
        for tiy in range(tile_box.t, tile_box.b):
            for tix in range(tile_box.l, tile_box.r):
                virt_tix = tix % max_tiles[1]
                already_in = (stride, tiy, virt_tix) in self.texture_state
                # Update the age if already in there
                # Assume that texture_state does not change from the main thread if this is run in another
                tex_tile_idx = self.texture_state.add_tile((stride, tiy, virt_tix))
                if already_in:
                    # FIXME: we should make a list/set of the tiles we need to add before this
                    continue

                # Assume we were given a total image worth of this stride
                y_start = tiy * self.tile_shape[0]
                y_end = y_start + self.tile_shape[0]
                x_start = virt_tix * self.tile_shape[1]
                x_end = x_start + self.tile_shape[1]
                textures_data = []
                for chn_idx in range(self.num_channels):
                    # force a copy of the data from the content array (provided by the workspace) to a vispy-compatible contiguous float array
                    # this can be a potentially time-expensive operation since content array is often huge and always memory-mapped, so paging may occur
                    # we don't want this paging deferred until we're back in the GUI thread pushing data to OpenGL!
                    tile_data = np.array(data[chn_idx][y_start: y_end, x_start: x_end], dtype=np.float32)
                    textures_data.append(tile_data)
                tiles_info.append((stride, tiy, tix, tex_tile_idx, textures_data))

        return tiles_info

    def _set_texture_tiles(self, tiles_info):
        for tile_info in tiles_info:
            stride, tiy, tix, tex_tile_idx, data_arrays = tile_info
            for idx, data in enumerate(data_arrays):
                self._textures[idx].set_tile_data(tex_tile_idx, data)

    def assess(self):
        """Determine if a retile is needed.

        Tell workspace we will be needed
        """
        view_box = self.get_view_box()
        preferred_stride = self.calc.calc_stride(view_box, texture=self._lowest_rez) * self._lowest_factor
        _, tile_box = self.calc.visible_tiles(view_box, stride=preferred_stride, extra_tiles_box=box(1, 1, 1, 1))
        num_tiles = (tile_box.b - tile_box.t) * (tile_box.r - tile_box.l)
        LOG.debug("Assessment: Prefer '%s' have '%s', was looking at %r, now looking at %r",
                  preferred_stride, self._stride, self._latest_tile_box, tile_box)
        # If we zoomed out or we panned
        need_retile = (num_tiles > 0) and (preferred_stride != self._stride or self._latest_tile_box != tile_box)
        return need_retile, preferred_stride, tile_box

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

    gl_FragColor.r = $get_data_1(texcoord).r;
    gl_FragColor.g = $get_data_2(texcoord).r;
    gl_FragColor.b = $get_data_3(texcoord).r;
    gl_FragColor.a = 1.0;
}
"""  # noqa


class RGBCompositeLayerVisual(CompositeLayerVisual):
    VERT_SHADER = RGB_VERT_SHADER
    FRAG_SHADER = RGB_FRAG_SHADER


RGB_VERT_SHADER_OLD = """
uniform int method;  // 0=subdivide, 1=impostor
attribute vec2 a_position;
attribute vec2 a_texcoord_1;
attribute vec2 a_texcoord_2;
attribute vec2 a_texcoord_3;
varying vec2 v_texcoord_1;
varying vec2 v_texcoord_2;
varying vec2 v_texcoord_3;

void main() {
    v_texcoord_1 = a_texcoord_1;
    v_texcoord_2 = a_texcoord_2;
    v_texcoord_3 = a_texcoord_3;
    gl_Position = $transform(vec4(a_position, 0., 1.));
}
"""

RGB_FRAG_SHADER_OLD = """
uniform vec2 image_size;
uniform int method;  // 0=subdivide, 1=impostor
uniform sampler2D u_texture_1;
uniform sampler2D u_texture_2;
uniform sampler2D u_texture_3;
varying vec2 v_texcoord_1;
varying vec2 v_texcoord_2;
varying vec2 v_texcoord_3;

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
    vec2 texcoord_1;
    vec2 texcoord_2;
    vec2 texcoord_3;
    if( method == 0 ) {
        texcoord_1 = v_texcoord_1;
        texcoord_2 = v_texcoord_2;
        texcoord_3 = v_texcoord_3;
    }
    else {
        // vertex shader ouptuts clip coordinates;
        // fragment shader maps to texture coordinates
        texcoord_1 = map_local_to_tex(vec4(v_texcoord_1, 0, 1)).xy;
        texcoord_2 = map_local_to_tex(vec4(v_texcoord_2, 0, 1)).xy;
        texcoord_3 = map_local_to_tex(vec4(v_texcoord_3, 0, 1)).xy;
    }

    gl_FragColor.r = $get_data_1(texcoord_1).r;
    gl_FragColor.g = $get_data_2(texcoord_2).r;
    gl_FragColor.b = $get_data_3(texcoord_3).r;
    gl_FragColor.a = 1.0;
}
"""  # noqa

class RGBCompositeLayerVisualOld(CompositeLayerVisual):
    def __init__(self, dep_r, dep_g, dep_b,
                 cmap='viridis', clim='auto', **kwargs):
        # visual nodes already have names, so be careful
        if not hasattr(self, "name"):
            self.name = kwargs.get("name", None)

        # assume all the dependencies have the same information
        assert(isinstance(dep_r, TiledGeolocatedImage))
        self.dep_r = dep_r
        assert(isinstance(dep_g, TiledGeolocatedImage))
        self.dep_g = dep_g
        assert(isinstance(dep_b, TiledGeolocatedImage))
        self.dep_b = dep_b

        self._need_texture_upload = True
        self._need_vertex_update = True
        self._need_colortransform_update = True
        self._need_clim_update = True
        self._need_interpolation_update = True
        self._need_method_update = True
        self._null_tr = NullTransform()
        # self._subdiv_position = VertexBuffer()
        # self._subdiv_texcoord = VertexBuffer()

        self._init_view(self)
        super(RGBCompositeLayerVisualOld, self).__init__(vcode=RGB_VERT_SHADER_OLD, fcode=RGB_FRAG_SHADER_OLD)
        self.set_gl_state('translucent', cull_face=False)
        self._draw_mode = 'triangles'

        self._method = None
        self.method = self.dep_r.method
        self.shape = self.dep_r.shape
        self.ndim = self.dep_r.ndim

        self._data_lookup_fn_r = Function(_texture_lookup)
        self._data_lookup_fn_g = Function(_texture_lookup)
        self._data_lookup_fn_b = Function(_texture_lookup)

        self.clim = (self.dep_r.clim, self.dep_g.clim, self.dep_b.clim)
        self.cmap = cmap
        self.freeze()

    def _init_view(self, view):
        # Store some extra variables per-view
        view._need_method_update = True
        view._method_used = None

    @property
    def cmap(self):
        return self._cmap

    @cmap.setter
    def cmap(self, cmap):
        self._cmap = get_colormap(cmap)
        self._need_colortransform_update = True
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
            clim = np.array(clim, float)
            if clim.shape != (3, 2) and clim.shape != (2,):
                raise ValueError('clim must have either 2 elements or 6 (2 for each color)')
            elif clim.shape == (2,):
                clim = np.array([clim, clim, clim], float)
        self._clim = clim
        self._need_clim_update = True
        self.update()

    @property
    def method(self):
        return self._method

    @method.setter
    def method(self, m):
        if self._method != m:
            self._method = m
            self._need_vertex_update = True
            self.update()

    @property
    def size(self):
        return self.shape[:2][::-1]

    def _update_method(self, view):
        """Decide which method to use for *view* and configure it accordingly.
        """
        method = self._method
        if method == 'auto':
            if view.transforms.get_transform().Linear:
                method = 'subdivide'
            else:
                method = 'impostor'
        view._method_used = method

        if method == 'subdivide':
            view.view_program['method'] = 0
            view.view_program['a_position'] = self.dep_r._subdiv_position
            view.view_program['a_texcoord_r'] = self.dep_r._subdiv_texcoord
            view.view_program['a_texcoord_g'] = self.dep_g._subdiv_texcoord
            view.view_program['a_texcoord_b'] = self.dep_b._subdiv_texcoord
        else:
            raise ValueError("Unknown image draw method '%s'" % method)

        self.shared_program['image_size'] = self.size
        view._need_method_update = False
        self._prepare_transforms(view)

    def _prepare_transforms(self, view):
        trs = view.transforms
        prg = view.view_program
        prg.vert['transform'] = trs.get_transform()
        prg.frag['transform'] = self._null_tr

    def _build_interpolation(self):
        # assumes 'nearest' interpolation
        self.shared_program.frag['get_data_r'] = self._data_lookup_fn_r
        self.shared_program.frag['get_data_g'] = self._data_lookup_fn_g
        self.shared_program.frag['get_data_b'] = self._data_lookup_fn_b
        self._data_lookup_fn_r['texture'] = self.dep_r._texture
        self._data_lookup_fn_g['texture'] = self.dep_g._texture
        self._data_lookup_fn_b['texture'] = self.dep_b._texture
        self._need_interpolation_update = False

    def _build_color_transform(self):
        if self.ndim == 2 or self.shape[2] == 1:
            fun = FunctionChain(None, [Function(_c2l),
                                       Function(self._cmap.glsl_map)])
        else:
            fun = Function(_null_color_transform)
        # self.shared_program.frag['color_transform'] = fun
        self._need_colortransform_update = False

    def _set_clim_vars(self):
        self._data_lookup_fn_r["vmin"] = self._clim[0, 0]
        self._data_lookup_fn_r["vmax"] = self._clim[0, 1]
        self._data_lookup_fn_g["vmin"] = self._clim[1, 0]
        self._data_lookup_fn_g["vmax"] = self._clim[1, 1]
        self._data_lookup_fn_b["vmin"] = self._clim[2, 0]
        self._data_lookup_fn_b["vmax"] = self._clim[2, 1]
        self._need_clim_update = False

    def _prepare_draw(self, view):
        if self._need_interpolation_update:
            self._build_interpolation()

        # if self._need_texture_upload:
        #     self._build_texture()

        if self._need_clim_update:
            self._set_clim_vars()

        if self._need_colortransform_update:
            self._build_color_transform()

        if view._need_method_update:
            self._update_method(view)


RGBCompositeLayer = create_visual_node(RGBCompositeLayerVisual)


class ShapefileLinesVisual(LineVisual):
    def __init__(self, filepath, projection=DEFAULT_PROJECTION, double=False, **kwargs):
        LOG.debug("Using border shapefile '%s'", filepath)
        self.sf = shapefile.Reader(filepath)
        # FUTURE: Proj stuff should be done in GLSL for better speeds and flexibility with swapping projection (may require something in addition to transform)
        self.proj = Proj(projection)

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
                vertex_buffer[prev_idx:end_idx:2] = one_shape.points[part_start:part_end-1]
                vertex_buffer[prev_idx + 1:end_idx:2] = one_shape.points[part_start+1:part_end]
                prev_idx = end_idx

        # Clip lats to +/- 89.9 otherwise PROJ.4 on mercator projection will fail
        np.clip(vertex_buffer[:, 1], -89.9, 89.9, out=vertex_buffer[:, 1])
        vertex_buffer[:, 0], vertex_buffer[:, 1] = self.proj(vertex_buffer[:, 0], vertex_buffer[:, 1])
        if double:
            LOG.debug("Adding 180 to 540 double of shapefile")
            orig_points = vertex_buffer.shape[0]
            vertex_buffer = np.concatenate((vertex_buffer, vertex_buffer), axis=0)
            vertex_buffer[orig_points:, 0] += C_EQ

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

