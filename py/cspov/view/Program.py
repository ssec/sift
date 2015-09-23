#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
.py
~~~

PURPOSE
gloo.Program wrappers for different purposes such as tile drawing

REFERENCES


REQUIRES


:author: R.K.Garcia <rayg@ssec.wisc.edu>
:copyright: 2015 by University of Wisconsin Regents, see AUTHORS for more details
:license: GPLv3, see LICENSE for more details
"""

import numpy as np
import logging
from vispy import gloo
from vispy.geometry import create_plane
from vispy.io import load_crate

__author__ = 'rayg'
__docformat__ = 'reStructuredText'


LOG = logging.getLogger(__name__)


SIMPLE_VERT_SHADER = """
uniform   mat4 u_model;
uniform   mat4 u_view;
uniform   mat4 u_projection;
uniform  float u_z;

attribute vec3 a_position;
attribute vec2 a_texcoord;

varying vec2 v_texcoord;

void main()
{
    v_texcoord = a_texcoord;
    gl_Position = u_projection * u_view * u_model * vec4(a_position.xy,u_z,1.0); // preferred
    // debug: show where x is large by tapering
    //gl_Position = u_projection * u_view * u_model * vec4(a_position.x, a_position.y+a_position.x*0.01,u_z,1.0); // debug
    //gl_Position = u_projection * u_view * u_model * vec4(a_position,1.0);
    //gl_Position = u_projection * vec4(a_position,1.0);
}
"""

# a fragment shader for projecting simple loaded images, e.g. png/jpg/tiff
RGB_FRAG_SHADER = """
uniform sampler2D u_texture;
uniform float u_alpha;
varying vec2 v_texcoord;

void main()
{
    float ty = v_texcoord.y;
    float tx = v_texcoord.x;
    vec2 txcoord = vec2(tx, ty);
    //gl_FragColor = texture2D(u_texture, txcoord);
    gl_FragColor = vec4(texture2D(u_texture, txcoord).xyz, texture2D(u_texture, txcoord).w * u_alpha);
    //gl_FragColor = vec4(texture2D(u_texture, txcoord).xyz, u_alpha); // DEBUG
}
"""

class GlooTile(object):
    """
    A GL program which plots a planar tile with a texture map
    texture map is assumed oriented such that ascending pixel row is ascending coordinate from bottom left of screen
    (typically requires inverting the Y direction)
    """
    program = None
    _world_box = None
    faces = None
    _z = 0.0
    _alpha = 1.0

    def __init__(self, vertex_code, fragment_code):
        super(GlooTile, self).__init__()
        self.program = gloo.Program(vertex_code, fragment_code)

    def get_world_box(self):
        return self._world_box

    def set_world_box(self, world_box):
        # get the geometry queued
        vtnc, faces, outline = create_plane(width=world_box.r-world_box.l,
                                             height=world_box.t-world_box.b,
                                             direction='+z')

        verts = np.array([q[0] for q in vtnc])
        # translate the vertices
        verts[:,0] += (world_box.l + world_box.r)/2.0
        verts[:,1] += (world_box.b + world_box.t)/2.0
        texcoords = np.array([q[1] for q in vtnc])
        # normals = np.array([q[2] for q in vtnc])
        # colors = np.array([q[3] for q in vtnc])
        faces_buffer = gloo.IndexBuffer(faces.astype(np.uint16))
        # print("V:", verts, len(verts))
        # print("T:", texcoords, len(texcoords))
        # print("N:", normals, len(normals))
        # print("C:", colors, len(colors))
        # print("F:", faces, len(faces))
        # print("O:", outline, len(outline))
        # print("T:", texcoords, len(texcoords))
        self.program['a_position'] = gloo.VertexBuffer(verts)
        self.program['a_texcoord'] = gloo.VertexBuffer(texcoords)
        self.faces = faces_buffer
        self._world_box = world_box
        self.program['u_z'] = self._z
        self.program['u_alpha'] = self._alpha

    world_box = property(get_world_box, set_world_box)


    def get_data(self):
        raise NotImplementedError('subclass must implement')

    def set_data(self, image, image_box=None):
        raise NotImplementedError('subclass must implement')

    data = property(get_data, set_data)


    def get_z(self):
        return self._z

    def set_z(self, z):
        self.program['u_z'] = self._z = z
        LOG.debug('z set to {}'.format(z))

    z = property(get_z, set_z)

    def get_alpha(self, alpha):
        return self._alpha

    def set_alpha(self, alpha):
        self.program['u_alpha'] = self._alpha = alpha
        LOG.debug('alpha set to {}'.format(alpha))

    alpha = property(get_alpha, set_alpha)

    def auto_range(self, field):
        pass

    def get_range(self):
        return None

    def set_range(self, min_value, max_value):
        pass  # raise?

    range = property(get_range, set_range)


    def get_colormap(self):
        return None

    def set_colormap(self, colormap):
        """
        Color map may be updated interactively and should be pushed to the GPU for impending redraw
        :return:
        """
        return None  # raise?

    colormap = property(get_colormap, set_colormap)

    def set_mvp(self, model=None, view=None, projection=None):
        if model is not None:
            self.program['u_model'] = model
        if view is not None:
            self.program['u_view'] = view
        if projection is not None:
            self.program['u_projection'] = projection


    def draw(self):
        self.program.draw('triangles', self.faces)



class GlooRGBImageTile(GlooTile):
    """
    A GL program which plots a planar tile with a texture map
    texture map is assumed oriented such that ascending pixel row is ascending coordinate from bottom left of screen
    (typically requires inverting the Y direction)
    """
    program = None
    image = None
    _world_box = None
    faces = None

    def __init__(self, world_box=None, image=None, image_box=None, **kwargs):
        super(GlooRGBImageTile, self).__init__(SIMPLE_VERT_SHADER, RGB_FRAG_SHADER)
        if image is not None:
            self.set_data(image, image_box)
        if world_box is not None:
            self.set_world_box(world_box)
        if kwargs:
            LOG.info("ignoring additional arguments {0!r:s}".format(list(kwargs.keys())))


    def get_data(self):
        return self.image

    def set_data(self, image, image_box=None):
        if isinstance(image, str) and image == 'crate':
            image = load_crate()
        if image_box is not None:
            self.image = image = image[image_box.b:image_box.t, image_box.l:image_box.r]
        else:
            self.image = image
        # get the texture queued
        self.program['u_texture'] = gloo.Texture2D(image)



####
####  Colormapped float32 field tiles
####



# FIXME: u_z and u_alpha need to be present in all frag shaders for subclasses of GlooTile; is this good?
# FIXME: allow color map to include alpha instead of just RGB - and multiply through with u_alpha
# FIXME: auto_range right now only is taking into account the local tile, not the whole image
# modified from imshow_cuts.py
COLORMAP_FRAG_SHADER = """
uniform float vmin;
uniform float vmax;
uniform float cmap;
uniform float n_colormaps;
uniform float u_z;
uniform float u_alpha;

uniform sampler2D field;
uniform sampler2D colormaps;

varying vec2 v_texcoord;

// http://stackoverflow.com/questions/11810158/how-to-deal-with-nan-or-inf-in-opengl-es-2-0-shaders
bool isNan(float val)
{
  return (val <= 0.0 || 0.0 <= val) ? false : true;
  // all comparisons on NaN return false. Note this may not work on sloppy non-IEEE754 GPUs! (Crap.)
  // also note that isnan isn't part of GLSL standard in 2.0, but we get isnan and isinf later.
}

void main()
{
    float value = texture2D(field, v_texcoord).r;
    float index = (cmap+0.5) / n_colormaps;
    //if (isNan(value)) {
    //    gl_FragColor = vec4(0,0,0,0);
    //} else
    if( value < vmin ) {
        gl_FragColor = texture2D(colormaps, vec2(0.0,index));
    } else if( value > vmax ) {
        gl_FragColor = texture2D(colormaps, vec2(1.0,index));
    } else {
        value = (value-vmin)/(vmax-vmin);
        value = 1.0/512.0 + 510.0/512.0*value;
        // gl_FragColor = texture2D(colormaps, vec2(value,index));
        gl_FragColor = vec4(texture2D(colormaps, vec2(value,index)).xyz, u_alpha);
    }
}
"""

# FIXME: implement a colormap editing and storage subsystem as part of the document

# Colormaps
COLORMAP_COUNT = 16
COLORMAP_LEN = 512
COLORMAPS = np.ones((COLORMAP_COUNT, COLORMAP_LEN, 4)).astype(np.float32)
values = np.linspace(0, 1, COLORMAP_LEN)[1:-1]

# Hot colormap
DEFAULT_COLORMAP = COLORMAP_HOT = 0
COLORMAPS[0, 0] = 0, 0, 1, 1  # Low values  (< vmin)
COLORMAPS[0, -1] = 0, 1, 0, 1  # High values (> vmax)
COLORMAPS[0, 1:-1, 0] = np.interp(values, [0.00, 0.33, 0.66, 1.00],
                                          [0.00, 1.00, 1.00, 1.00])
COLORMAPS[0, 1:-1, 1] = np.interp(values, [0.00, 0.33, 0.66, 1.00],
                                          [0.00, 0.00, 1.00, 1.00])
COLORMAPS[0, 1:-1, 2] = np.interp(values, [0.00, 0.33, 0.66, 1.00],
                                          [0.00, 0.00, 0.00, 1.00])

# Grey colormap
COLORMAP_GREY = 1
COLORMAPS[1, 0] = 0, 0, 1, 1  # Low values (< vmin)
COLORMAPS[1, -1] = 0, 1, 0, 1  # High values (> vmax)
COLORMAPS[1, 1:-1, 0] = np.interp(values, [0.00, 1.00],
                                          [0.00, 1.00])
COLORMAPS[1, 1:-1, 1] = np.interp(values, [0.00, 1.00],
                                          [0.00, 1.00])
COLORMAPS[1, 1:-1, 2] = np.interp(values, [0.00, 1.00],
                                          [0.00, 1.00])
# Jet colormap
# ...
del values


class GlooColormapDataTile(GlooTile):
    """
    A Tile program which uses a RGBA color map shader to render a mercator-projected float32 science data field
    NaNs are automatically mapped to alpha=0
    FUTURE: allow simple enhancement expressions to be embedded in the shader code, e.g. sqrt or log enhancement
    """
    program = None
    image = None  # float32 2D array
    _colormaps = None  # colormap data to push into GPU
    faces = None
    _world_box = None
    _range = None

    def __init__(self, world_box=None, image=None, image_box=None, colormap=None, range=None, **kwargs):
        super(GlooColormapDataTile, self).__init__(SIMPLE_VERT_SHADER, COLORMAP_FRAG_SHADER)
        if image is not None:
            self.set_data(image, image_box)
        if world_box is not None:
            self.set_world_box(world_box)
        self.set_colormap(colormap or COLORMAPS)
        if range is not None:
            self.set_range(range)
        else:
            self.auto_range()
        if kwargs:
            LOG.info("ignoring additional arguments {0!r:s}".format(list(kwargs.keys())))
        self.program['cmap'] = COLORMAP_GREY  # alternate COLORMAP_HOT
        self.program['n_colormaps'] = COLORMAP_COUNT
        # FIXME: deal with multiple colormaps being available, as well as loading/editing color maps

    def get_data(self):
        return self.image

    def set_data(self, image, image_box=None):
        assert(len(image.shape)==2)  # we only accept flat fields of intensity
        image = np.require(image, dtype=np.float32)

        if image_box is not None:
            self.image = image = image[image_box.b:image_box.t, image_box.l:image_box.r]
        else:
            self.image = image
        # get the texture queued
        self.program['field'] = gloo.Texture2D(image)  # FIXME: review if we need to send additional parameters


    def auto_range(self, field=None):
        field = field or self.image
        range = np.nanmin(field), np.nanmax(field)
        LOG.debug('autorange {}'.format(repr(range)))
        # range = (0.0, 255.0)  # FIXME DEBUG
        self.set_range(range)

    def get_range(self):
        return self._range

    def set_range(self, new_range):
        LOG.debug('range {}'.format(repr(new_range)))
        vmin, vmax = new_range
        self.program['vmin'] = vmin
        self.program['vmax'] = vmax


    def get_colormap(self):
        return self._colormaps

    def set_colormap(self, colormap):
        """
        Color map may be updated interactively and should be pushed to the GPU for impending redraw
        :return:
        """
        self.program['colormaps'] = self._colormaps = colormap



from vispy.gloo import Texture2D
from cspov.common import DEFAULT_TILE_HEIGHT, DEFAULT_TILE_WIDTH
class TextureAtlas2D(Texture2D):
    """A 2D Texture Array structure implemented as a 2D Texture Atlas.

    External tile indexes are 1 to `num_tiles` while internal structure may
    be 2 dimensional.
    """
    def __init__(self, num_tiles, tile_shape=(DEFAULT_TILE_HEIGHT, DEFAULT_TILE_WIDTH),
                 format=None, resizable=True,
                 interpolation=None, wrapping=None,
                 internalformat=None, resizeable=None):
        self._atlas_shape = (int(num_tiles/5)+1, 5)
        shape = (self._atlas_shape[0] * tile_shape[0], self._atlas_shape[1] * tile_shape[1])
        self.tile_shape = tile_shape
        self.num_tiles = num_tiles
        self.tile_dirty = np.ones((self.num_tiles,), dtype=np.bool)
        super(TextureAtlas2D, self).__init__(None, format, resizable, interpolation,
                                             wrapping, shape, internalformat, resizeable)

    def iter_tile_index(self):
        return (idx for idx, dirty in enumerate(self.tile_dirty) if not dirty)

    def _tex_offset(self, idx):
        """Return the X, Y texture index offset for the 1D tile index.

        This class presents a 1D indexing scheme, but internally can hold multiple tiles in both X and Y direction.
        """
        row = int(idx / self._atlas_shape[1])
        col = idx % self._atlas_shape[1]
        print(row, col)
        return row * self.tile_shape[0], col * self.tile_shape[1]

    def set_tile_data(self, tile_idx, data, copy=False):
        """Write a single tile of data into the texture.
        """
        offset = self._tex_offset(tile_idx)
        tile_offset = (min(self.tile_shape[0], data.shape[0]),
                       min(self.tile_shape[1], data.shape[1]))
        if tile_offset[0] < self.tile_shape[0] or tile_offset[1] < self.tile_shape[1]:
            # FIXME: This should be handled by the caller to expand the array to be NaN filled and aligned
            # Assign a fill value, make sure to copy the data so that we don't overwrite the original
            data = data.copy()
            data[:] = 0.0
            data[:tile_offset[0], :tile_offset[1]] = data[:tile_offset[0], :tile_offset[1]]
        super(TextureAtlas2D, self).set_data(data, offset=offset, copy=copy)
        self.tile_dirty[tile_idx] = False

    def set_data(self, data, offset=None, copy=False):
        tile_idx = 0
        # for y_idx in [0]:# range(0, data.shape[0], self.tile_shape[0]):
        for y_idx in range(0, data.shape[0], self.tile_shape[0]):
            for x_idx in range(0, data.shape[1], self.tile_shape[1]):
                self.set_tile_data(tile_idx,
                                   data[y_idx: y_idx + self.tile_shape[0], x_idx: x_idx + self.tile_shape[1]],
                                   copy=copy)
                tile_idx += 1

                if tile_idx >= self.num_tiles:
                    break
                    raise ValueError("Too much data to fit in texture atlas with shape %r" % (data.shape,))
                # FIXME: Don't go further than the data we have
            if tile_idx >= self.num_tiles:
                break

    def get_texture_coordinates(self, tile_idx):
        """Get texture coordinates for one tile as a quad.
        """
        # grid = (1, 1)
        w = 1
        h = 1
        row = int(tile_idx / self._atlas_shape[1])
        col = tile_idx % self._atlas_shape[1]

        # start with basic quad describing the entire texture
        quad = np.array([[0, 0, 0], [w, 0, 0], [w, h, 0],
                         [0, 0, 0], [w, h, 0], [0, h, 0]],
                        dtype=np.float32)
        # Now scale and translate the coordinates so they only apply to one tile in the texture
        one_tile_tex_width = 1.0 / self._shape[1] * self.tile_shape[1]
        one_tile_tex_height = 1.0 / self._shape[0] * self.tile_shape[0]
        quad[:, 0] *= one_tile_tex_width
        quad[:, 0] += one_tile_tex_width * col
        quad[:, 1] *= one_tile_tex_height
        quad[:, 1] += one_tile_tex_height * row
        quad = quad.reshape(6, 3)
        quad = np.ascontiguousarray(quad[:, :2])
        return quad
