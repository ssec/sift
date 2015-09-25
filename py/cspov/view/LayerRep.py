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

import scipy.misc as spm
from PyQt4.QtCore import QObject, pyqtSignal
import shapefile

from vispy.scene.visuals import create_visual_node
from vispy.visuals import LineVisual, ImageVisual, CompoundVisual
from vispy.ext.six import string_types
import numpy as np
from datetime import datetime
from pyproj import Proj

# from cspov.common import pnt, rez, MAX_EXCURSION_Y, MAX_EXCURSION_X, MercatorTileCalc, WORLD_EXTENT_BOX, \
#     DEFAULT_TILE_HEIGHT, DEFAULT_TILE_WIDTH, vue
from cspov.common import (DEFAULT_X_PIXEL_SIZE,
                          DEFAULT_Y_PIXEL_SIZE,
                          DEFAULT_ORIGIN_X,
                          DEFAULT_ORIGIN_Y,
                          DEFAULT_PROJECTION,
                          WORLD_EXTENT_BOX,
                          C_EQ,
                          box, pnt, rez, vue,
                          MercatorTileCalc
                          )
from cspov.view.Program import GlooRGBImageTile

__author__ = 'rayg'
__docformat__ = 'reStructuredText'

LOG = logging.getLogger(__name__)


class LayerRep(QObject):
    """
    A Layer Representation on the View side of the fence
    - has one or more representations available to immediately draw
    - may want to schedule the rendering of other representations during idle time, to get ideal view
    - may have a backing science representation which is pure science data instead of pixel values or RGBA maps
    - typically will cache a "coarsest" single-tile representation for zoom-out events (preferred for fast=True paint calls)
    - can have probes attached which operate primarily on the science representation
    """
    propertyDidChange = pyqtSignal(dict)
    _z = 0.0
    _alpha = 1.0
    _name = 'unnnamed'

    def __init__(self):
        super(LayerRep, self).__init__()

    def get_z(self):
        return self._z

    def set_z(self, new_z):
        self._z = new_z
        self.propertyDidChange.emit({'z': new_z})

    z = property(get_z, set_z)

    def get_alpha(self):
        return self._alpha

    def set_alpha(self, new_alpha):
        self._alpha = new_alpha
        self.propertyDidChange.emit({'alpha': new_alpha})

    alpha = property(get_alpha, set_alpha)

    def get_name(self):
        return self._name

    def set_name(self, new_name):
        self._name = new_name
        self.propertyDidChange.emit({'name': new_name})

    name = property(get_name, set_name)

    def paint(self, geom, mvp, fast=False, **kwargs):
        """
        draw the most appropriate representation for this layer, given world geometry represented and projection matrix
        if a better representation could be rendered for later draws, return False and render() will be queued for later idle time
        fast flag requests that low-cost rendering be used
        """
        return True

    def render(self, geom, *more_geom):
        """
        cache a rendering (typically a draw-list with textures) that best handles the extents and sampling requested
        if more than one view is active, more geometry may be provided for other views
        return False if resources were too limited and a purge is needed among the layer stack
        :param geom: screen geometry as a vue tuple, world coordinates with d(world)/d(pixel) dy and dx
        """
        return True

    def purge(self, geom, *more_geom):
        """
        release any cached representations that we haven't used lately, leaving at most 1
        :return: True if any GL resources were released
        """
        return False

    def probe_point_xy(self, x, y):
        """
        return a value array for the requested point as specified in mercator-meters
        """
        raise NotImplementedError()

    def probe_point_geo(self, lat, lon):
        """
        """
        raise NotImplementedError()

    def probe_shape(self, geo_shape):
        """
        given a shapely description of an area, return a masked array of data
        """
        raise NotImplementedError()


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
        tex_coords = np.ascontiguousarray(tex_coords[:, :2])
        vertices = tex_coords * self.size

        # FUTURE: This should probably be done as a transform
        vertices = vertices.astype('float32')
        vertices[:, 0] *= self.cell_width
        vertices[:, 0] += self.origin_x
        vertices[:, 1] *= self.cell_height
        vertices[:, 1] += self.origin_y
        if self.double:
            orig_points = vertices.shape[0]
            vertices = np.concatenate((vertices, vertices), axis=0)
            tex_coords = np.concatenate((tex_coords, tex_coords), axis=0)
            vertices[orig_points:, 0] += C_EQ
        self._subdiv_position.set_data(vertices.astype('float32'))
        self._subdiv_texcoord.set_data(tex_coords.astype('float32'))


GeolocatedImage = create_visual_node(GeolocatedImageVisual)


class OldTiledGeolocatedImageVisual(CompoundVisual):
    """Geolocated image that is specially tiled to save on GPU memory usage.
    """
    def __init__(self, name, data=None, tile_shape=(512, 512), **kwargs):
        self.name = name
        self.tile_shape = tile_shape
        self.cell_width = kwargs["cell_width"]
        self.cell_height = kwargs["cell_height"]  # Note: cell_height is usually negative
        self.origin_x = kwargs["origin_x"]
        self.origin_y = kwargs["origin_y"]
        self._stride = None  # Current stride is None when we are showing the overview
        self._waiting_on_data = False
        self._tiles = {}

        assert ("shape" in kwargs or data is not None), "`data` or `shape` must be provided"
        image_shape = kwargs.get("shape", data.shape)

        # Where does this image lie in this lonely world
        self.calc = MercatorTileCalc(
            self.name,
            image_shape,
            pnt(x=self.origin_x, y=self.origin_y),
            rez(dy=abs(self.cell_height), dx=abs(self.cell_width)),
            tile_shape,
        )

        # FIXME: We should not be holding on to the entire dataset, this should come from the workspace
        self.full_data = data

        # Start by showing the overview, the actual draw cycle will do the proper requests for higher resolution data
        # FIXME: Only making one large image for now
        y_slice, x_slice = self.calc.overview_stride()
        # overview_stride = (int(image_shape[0]/tile_shape[0]), int(image_shape[1]/tile_shape[1]))
        self.overview_data = data[y_slice, x_slice]
        # Update kwargs to reflect the new spatial resolution of the overview image
        kwargs["data"] = self.overview_data
        kwargs["cell_width"] *= x_slice.step
        kwargs["cell_height"] *= y_slice.step
        # FIXME: Reminder, origin x/y also need to be included in these updates for each tile
        # Keep a pointer to the overview so we can remove it from our CompoundVisual list
        self.overview_visual = GeolocatedImageVisual(**kwargs)

        # Initialize and 'freeze' the Visual class, don't declare any child visuals yet
        super(TiledGeolocatedImageVisual, self).__init__(tuple())

        # Add the overview as the first image that gets displayed, this will get updated in the actual draw
        # We don't update the tiles here in case transforms aren't completely set yet
        # self.add_subvisual(self.overview_visual)

    def _generate_tiles(self, view_box):
        pass

    def _show_overview(self):
        if self._stride is None and self.overview_visual.visible:
            # we're already showing it
            return
        # TODO: Clear out old tiles, probably in another method
        # TODO: Set Z level of overview to be one lower than the actual tiles so
        # this way we can show the overview but keep the current tiles until new data is ready
        self.add_subvisual(self.overview_visual)
        self.overview_visual.visible = True
        self._stride = None

    def _hide_overview(self, new_stride):
        if self._stride is not None or not self.overview_visual.visible:
            # we aren't showing it
            return
        self.remove_subvisual(self.overview_visual)
        self.overview_visual.visible = False
        self._stride = new_stride

    def set_data(self, new_data):
        self._waiting_on_data = False

    def get_new_data(self, y_slice, x_slice):
        # FIXME: Remove this and replace with an actual request to the Workspace
        return self.full_data[y_slice, x_slice]

    def paint(self, view):
        """Quickly and cheaply draw what we have.
        """
        return True

    def _get_view_box(self):
        ll_corner, ur_corner = self.transforms.get_transform().imap([(-1, -1, 1), (1, 1, 1)])
        # How many tiles should be contained in this view?
        view_box = box(
            max(ll_corner[1], WORLD_EXTENT_BOX.b),
            max(ll_corner[0], WORLD_EXTENT_BOX.l),
            min(ur_corner[1], WORLD_EXTENT_BOX.t),
            min(ur_corner[0], WORLD_EXTENT_BOX.r)
        )
        view_box = vue(*view_box, dy=(view_box.t - view_box.b)/self.canvas.size[1], dx=(view_box.r - view_box.l)/self.canvas.size[0])
        return view_box

    def assess(self, view):
        """Determine if a retile is needed.

        Tell workspace we will be needed
        """
        view_box = self._get_view_box()
        preferred_stride = self.calc.calc_stride(view_box)
        # XXX: Do we request new data here? How do the slices get moved around?
        # self.request_new_data(y_slice, x_slice)
        return preferred_stride != self._stride

    def _generate_tiles(self, new_data):
        # Remove tiles that aren't needed
        # TODO: Stop removing all tiles
        for tile_obj in self._tiles.values():
            # FIXME: This redraws everytime by calling update
            self.remove_subvisual(tile_obj)

        # Add tiles that we need
        y_tiles = int(np.ceil(new_data.shape[0] / self.tile_shape[0]))
        x_tiles = int(np.ceil(new_data.shape[1] / self.tile_shape[1]))
        for tiy in range(y_tiles):
            for tix in range(x_tiles):
                LOG.debug('y:{0} x:{1}'.format(tiy, tix))
                # subim = self.calc.tile_pixels(tiy, tix, 1)
                t = GeolocatedImageVisual(
                    data=self.calc.tile_pixels(new_data, tiy, tix, 1),
                    origin_x=self.origin_x + self.tile_shape[1] * self.cell_width,
                    origin_y=self.origin_y + self.tile_shape[1] * self.cell_height,
                    cell_width=self.cell_width / self.tile_shape[1],
                    cell_height=self.cell_height / self.tile_shape[0],
                )
                self._tiles[(tiy, tix)] = t
                self.add_subvisual(t)

    def retile(self):
        """Get data from workspace and retile/retexture as needed.
        """
        view_box = self._get_view_box()
        preferred_stride = self.calc.calc_stride(view_box)
        _, tile_box = self.calc.visible_tiles(view_box)

        LOG.debug("Requesting new data from Workspace...")
        # FIXME: Request just the part needed, not the whole strided image
        # y_slice = slice(0, self.full_data.shape[0], preferred_stride)
        # x_slice = slice(0, self.full_data.shape[1], preferred_stride)
        top_idx = self.full_data.shape[0] - (self.tile_shape[0] * tile_box.t)
        bot_idx = self.full_data.shape[0] - (self.tile_shape[0] * tile_box.b)
        y_slice = slice(top_idx, bot_idx, preferred_stride)
        x_slice = slice(tile_box.l * self.tile_shape[1], tile_box.r * self.tile_shape[1], preferred_stride)
        new_data = self.get_new_data(y_slice, x_slice)
        print(new_data.shape, tile_box)
        self._generate_tiles(new_data)
        self._stride = preferred_stride

    def _prepare_draw(self, view):
        self.paint(view)

        if self.assess(view):
            # We need a rerender/retile
            self.retile()
        return super(TiledGeolocatedImageVisual, self)._prepare_draw(view)


OldTiledGeolocatedImage = create_visual_node(OldTiledGeolocatedImageVisual)


from cspov.view.Program import TextureAtlas2D, Texture2D
from vispy.visuals.image import ImageVisual, load_spatial_filters, \
    Function, _texture_lookup, VertexBuffer, _interpolation_template, \
    NullTransform, VERT_SHADER, FRAG_SHADER
from cspov.common import DEFAULT_TILE_HEIGHT, DEFAULT_TILE_WIDTH

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

// http://stackoverflow.com/questions/11810158/how-to-deal-with-nan-or-inf-in-opengl-es-2-0-shaders
bool isNan(float val)
{
  return (val <= 0.0 || 0.0 <= val) ? false : true;
  // all comparisons on NaN return false. Note this may not work on sloppy non-IEEE754 GPUs! (Crap.)
  // also note that isnan isn't part of GLSL standard in 2.0, but we get isnan and isinf later.
}

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
        // NaN is not properly translated in the texture so use 0 as the fill value
        if (val.r == 0.0) {
            discard;
        }
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
        self.itile_cache = {}
        self._rev_cache = {}
        # True if the data doesn't matter, False if data matters
        self.tile_free = [True] * num_tiles
        self.itile_age = []

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
        self.tile_tree[:] = True
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

    def add_tile(self, itile_idx):
        """Get texture index for new tile. If tile is already known return its current location.

        Note, this should be called even when the caller knows the tile exists to refresh the "age".
        """
        # Have we already added this tile, get the tile index
        if itile_idx in self:
            self.refresh_age(itile_idx)
            return self[itile_idx]

        ttile_idx = self.next_available_tile()

        self.itile_cache[itile_idx] = ttile_idx
        self._rev_cache[ttile_idx] = itile_idx
        self.tile_free[ttile_idx] = False
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
                 tile_shape=(DEFAULT_TILE_HEIGHT, DEFAULT_TILE_WIDTH), num_tiles=32,
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
        self.num_tiles = num_tiles
        self.tile_shape = (DEFAULT_TILE_HEIGHT, DEFAULT_TILE_WIDTH)
        self.tile_info = {}
        self.lod = 5
        self.tile_shape = tile_shape
        self._stride = 0 # Current stride is None when we are showing the overview
        self._latest_tile_box = None
        self._waiting_on_data = False
        self._tiles = {}

        assert ("shape" in kwargs or data is not None), "`data` or `shape` must be provided"
        image_shape = kwargs.get("shape", data.shape)

        # Where does this image lie in this lonely world
        self.calc = MercatorTileCalc(
            self.name,
            image_shape,
            pnt(x=self.origin_x, y=self.origin_y),
            rez(dy=abs(self.cell_height), dx=abs(self.cell_width)),
            tile_shape,
        )
        # What tiles have we used and can we use
        self.texture_state = TextureTileState(self.num_tiles)

        # Copied from original visual
        self._data = None

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
        self._texture = TextureAtlas2D(num_tiles=num_tiles, tile_shape=self.tile_shape,
                                       interpolation=texture_interpolation)
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

        self.clim = clim
        self.cmap = cmap
        if data is not None:
            self.set_data(data)
        self.freeze()

    # def _build_texture(self):
    #     super(TiledGeolocatedImageVisual, self)._build_texture()
    #     self._data = self._data[:512, :self._texture.shape[1]]
    def _build_texture(self):
        data = self._data
        if data.dtype == np.float64:
            data = data.astype(np.float32)

        if data.ndim == 2 or data.shape[2] == 1:
            # deal with clim on CPU b/c of texture depth limits :(
            # can eventually do this by simulating 32-bit float... maybe
            clim = self._clim
            if isinstance(clim, string_types) and clim == 'auto':
                clim = np.min(data), np.max(data)
            clim = np.asarray(clim, dtype=np.float32)
            data = data - clim[0]  # not inplace so we don't modify orig data
            if clim[1] - clim[0] > 0:
                data /= clim[1] - clim[0]
            else:
                data[:] = 1 if data[0, 0] != 0 else 0
            self._clim = np.array(clim)

        view_box = self._get_view_box()
        preferred_stride = self.calc.calc_stride(view_box)
        _, tile_box = self.calc.visible_tiles(view_box, stride=preferred_stride)
        LOG.debug("Uploading texture data for %d tiles (%r)", (tile_box.b - tile_box.t) * (tile_box.r - tile_box.l), tile_box)

        # Tiles start at upper-left so go from top to bottom
        for tiy in range(tile_box.t, tile_box.b):
            for tix in range(tile_box.l, tile_box.r):
                already_in = (preferred_stride, tiy, tix) in self.texture_state
                # Update the age if already in there
                tex_tile_idx = self.texture_state.add_tile((preferred_stride, tiy, tix))
                if already_in:
                    # FIXME: we should make a list/set of the tiles we need to add before this
                    continue

                LOG.debug("Adding image tile (stride: {:d}, y: {:d}, x: {:d}) to texture tile {:d}".format(preferred_stride, tiy, tix, tex_tile_idx))
                y_slice, x_slice = self.calc.tile_slices(tiy, tix, preferred_stride)
                self._texture.set_tile_data(
                    tex_tile_idx,
                    data[y_slice, x_slice]
                )
        self._need_texture_upload = False
        self._need_vertex_update = True

    def _build_vertex_data(self):
        """Rebuild the vertex buffers used for rendering the image when using
        the subdivide method.

        CSPOV Note: Copied from 0.5.0dev original ImageVisual class
        """
        view_box = self._get_view_box()
        preferred_stride = self.calc.calc_stride(view_box)
        _, tile_box = self.calc.visible_tiles(view_box, stride=preferred_stride)
        total_num_tiles = (tile_box.b - tile_box.t) * (tile_box.r - tile_box.l)
        tex_coords = np.empty((6 * total_num_tiles, 2), dtype=np.float32)
        vertices = np.empty((6 * total_num_tiles, 2), dtype=np.float32)

        if total_num_tiles > self.num_tiles:
            LOG.warning("Current view sees more tiles than can be held in the GPU")
            # We continue on because there should be an overview image for any tiles that can't be drawn

        # preferred_stride = 1
        LOG.debug("Building vertex data for %d tiles (%r)", total_num_tiles, tile_box)
        # What tile are we currently describing out of all the tiles being viewed
        used_tile_idx = -1
        # Tiles start at upper-left so go from top to bottom
        for tiy in range(tile_box.t, tile_box.b):
            for tix in range(tile_box.l, tile_box.r):
                # Update the index here because we have multiple exit/continuation points
                used_tile_idx += 1

                # Check if the tile we want to draw is actually in the GPU, if not (atlas too small?) fill with zeros and keep going
                if (preferred_stride, tiy, tix) not in self.texture_state:
                    # THIS SHOULD NEVER HAPPEN IF TEXTURE BUILDING IS DONE CORRECTLY AND THE ATLAS IS BIG ENOUGH
                    tex_coords[used_tile_idx*6: (used_tile_idx+1)*6, :] = 0
                    vertices[used_tile_idx*6: (used_tile_idx+1)*6, :] = 0
                    continue

                # we should have already loaded the texture data in to the GPU so get the index of that texture
                tex_tile_idx = self.texture_state[(preferred_stride, tiy, tix)]
                # print(tex_tile_idx, preferred_stride, tiy, tix)
                tex_coords[used_tile_idx*6: (used_tile_idx+1)*6, :] = self._texture.get_texture_coordinates(tex_tile_idx)
                vertices[used_tile_idx*6: (used_tile_idx+1)*6, :] = self.calc.calc_vertex_coordinates(tiy, tix, preferred_stride)

        self._subdiv_position.set_data(vertices.astype('float32'))
        self._subdiv_texcoord.set_data(tex_coords.astype('float32'))
        # We don't need to recreate the vertex data unless the texture changes
        self._need_vertex_update = False
        # Store the most recent level of detail that we've done
        self._stride = preferred_stride
        self._latest_tile_box = tile_box

    def paint_old(self, view):
        """Quickly and cheaply draw what we have.
        """
        return True

    def _get_view_box(self):
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

    def assess(self, view):
        """Determine if a retile is needed.

        Tell workspace we will be needed
        """
        view_box = self._get_view_box()
        preferred_stride = self.calc.calc_stride(view_box)
        _, tile_box = self.calc.visible_tiles(view_box, stride=preferred_stride)
        LOG.debug("Assessment: Prefer '%s' have '%s', was looking at %r, now looking at %r",
                  preferred_stride, self._stride, self._latest_tile_box, tile_box)
        # If we zoomed out or we panned
        return preferred_stride != self._stride or self._latest_tile_box != tile_box

    def retile(self):
        """Get data from workspace and retile/retexture as needed.
        """
        view_box = self._get_view_box()
        preferred_stride = self.calc.calc_stride(view_box)
        _, tile_box = self.calc.visible_tiles(view_box)

        LOG.debug("Requesting new data from Workspace...")
        # FIXME: Request just the part needed, not the whole strided image
        # y_slice = slice(0, self.full_data.shape[0], preferred_stride)
        # x_slice = slice(0, self.full_data.shape[1], preferred_stride)
        top_idx = self.full_data.shape[0] - (self.tile_shape[0] * tile_box.t)
        bot_idx = self.full_data.shape[0] - (self.tile_shape[0] * tile_box.b)
        y_slice = slice(top_idx, bot_idx, preferred_stride)
        x_slice = slice(tile_box.l * self.tile_shape[1], tile_box.r * self.tile_shape[1], preferred_stride)
        new_data = self.get_new_data(y_slice, x_slice)
        print(new_data.shape, tile_box)
        self._generate_tiles(new_data)
        self._stride = preferred_stride

    def _prepare_draw(self, view):
        # TODO: Update the texture if things have changed
        # self.paint(view)

        if self.assess(view):
            # We need a rerender/retile
            # self.retile()
            print("Reassessment needed!!!")
            self._need_texture_upload = True
            self._need_vertex_update = True
        return super(TiledGeolocatedImageVisual, self)._prepare_draw(view)

TiledGeolocatedImage = create_visual_node(TiledGeolocatedImageVisual)

# FIXME: replace this with a LayerRepFactory which tracks all the GPU resources that have been dedicated?
class OldTiledImageFile(LayerRep):
    """
    Tile an RGB or float32 image representing the full -180..180 longitude, -90..90 latitude
    """
    image = None
    shape = None
    calc = None
    tiles = None  # dictionary of {(y,x): GlooRgbTile, ...}
    _stride = 1
    _tile_kwargs = {}

    def set_z(self, z):
        super(TiledImageFile, self).set_z(z)
        for tile in self.tiles.values():
            tile.z = z

    def set_alpha(self, alpha):
        for tile in self.tiles.values():
            tile.alpha = alpha
        super(TiledImageFile, self).set_alpha(alpha)

    def __init__(self, filename=None, world_box=None, tile_shape=None, tile_class=GlooRGBImageTile, **kwargs):
        super(TiledImageFile, self).__init__()
        self._tile_class = tile_class
        self._tile_kwargs = dict(kwargs)  # FUTURE this is too cryptic, initially used to propagate range default
        self.image = spm.imread(filename or 'cspov/data/shadedrelief.jpg')  # FIXME package resource
        self.image = self.image[::-1]  # flip so 0,0 is bottom left instead of top left
        if filename is None:
            tile_shape = (1080,1080)  # FIXME make tile shape smarter
            self.name = 'shadedrelief'
        else:
            self.name = os.path.split(filename)[-1]
        tile_shape = tile_shape or (DEFAULT_TILE_HEIGHT,DEFAULT_TILE_WIDTH)
        self.world_box = world_box or WORLD_EXTENT_BOX
        self.shape = (h,w) = tuple(self.image.shape[:2])
        zero_point = pnt(float(h)/2, float(w)/2)
        pixel_rez = rez(MAX_EXCURSION_Y*2/float(h), MAX_EXCURSION_X*2/float(w))
        self.calc = MercatorTileCalc('bgnd', self.shape, zero_point, pixel_rez, tile_shape)
        self.tiles = {}
        # start with a very coarse representation that fits in minimal tiles
        tinyworld = vue(*WORLD_EXTENT_BOX,
                        dy=MAX_EXCURSION_Y*2/DEFAULT_TILE_HEIGHT,
                        dx=MAX_EXCURSION_X*2/DEFAULT_TILE_WIDTH)
        self._generate_tiles(self.calc.calc_stride(tinyworld))
        # self._generate_tiles()

    def paint(self, visible_geom, mvp, fast=False, **kwargs):
        """
        draw the most appropriate representation for this layer
        if a better representation could be rendered for later draws, return False and render() will be queued for later idle time
        fast flag requests that low-cost rendering be used
        """
        _, tilebox = self.calc.visible_tiles(visible_geom)
        # LOG.info(tilebox)
        # print(repr(tilebox))
        for iy in range(tilebox.b, tilebox.t):
            for ix in range(tilebox.l, tilebox.r):
                tile = self.tiles[(iy,ix)]
                # LOG.debug('draw tile {0!r:s}'.format(tile))
                # m,v,p = mvp
                tile.set_mvp(*mvp)
                tile.draw()
        preferred_stride = self.calc.calc_stride(visible_geom)
        return True if preferred_stride != self._stride else False

    def render(self, geom, *more_geom):
        "render at a suitable sampling for the screen geometry"
        stride = self.calc.calc_stride(geom)
        self._generate_tiles(stride)

    def _generate_tiles(self, stride=None):
        h,w = self.image.shape[:2]
        _, tilebox = self.calc.visible_tiles(WORLD_EXTENT_BOX)
        # LOG.info(tilebox)
        # for tiy in range(int((tilebox.b+tilebox.t)/2), tilebox.t):  DEBUG
        #     for tix in range(int((tilebox.l+tilebox.r)/2), tilebox.r):
        if stride is not None:
            self._stride = stride
        for tiy in range(tilebox.b, tilebox.t):
            for tix in range(tilebox.l, tilebox.r):
                tilegeom = self.calc.tile_world_box(tiy,tix)
                # if (tilegeom.r+tilegeom.l) < 0 or (tilegeom.b+tilegeom.t) < 0: continue ## DEBUG
                LOG.debug('y:{0} x:{1} geom:{2!r:s}'.format(tiy,tix,tilegeom))
                subim = self.calc.tile_pixels(self.image, tiy, tix, self._stride)
                self.tiles[(tiy,tix)] = t = self._tile_class(tilegeom, subim, **self._tile_kwargs)
                # t.set_mvp(model=self.model, view=self.view)


class ShapefileLinesVisual(CompoundVisual):
    def __init__(self, filepath, projection=DEFAULT_PROJECTION, double=False, **kwargs):
        LOG.debug("Using border shapefile '%s'", filepath)
        self.sf = shapefile.Reader(filepath)
        # FUTURE: Proj stuff should be done in GLSL for better speeds and flexibility with swapping projection (may require something in addition to transform)
        self.proj = Proj(projection)

        print("Loading boundaries: ", datetime.utcnow().isoformat(" "))
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

        kwargs.setdefault("color", (0.0, 0.0, 1.0, 1.0))
        kwargs.setdefault("width", 1)
        self._border_lines = LineVisual(vertex_buffer, connect="segments", **kwargs)
        print("Done loading boundaries: ", datetime.utcnow().isoformat(" "))

        super(ShapefileLinesVisual, self).__init__((self._border_lines,))


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

