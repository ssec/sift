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


# FIXME: use uniform float _z

VERT_CODE = """
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

FRAG_CODE = """
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


    def set_mvp(self, model=None, view=None, projection=None):
        if model is not None:
            self.program['u_model'] = model
        if view is not None:
            self.program['u_view'] = view
        if projection is not None:
            self.program['u_projection'] = projection


    def draw(self):
        self.program.draw('triangles', self.faces)



class GlooRGBTile(GlooTile):
    """
    A GL program which plots a planar tile with a texture map
    texture map is assumed oriented such that ascending pixel row is ascending coordinate from bottom left of screen
    (typically requires inverting the Y direction)
    """
    program = None
    image = None
    _world_box = None
    faces = None
    _z = 0.0
    _alpha = 1.0

    def __init__(self, world_box=None, image=None, image_box=None):
        super(GlooRGBTile, self).__init__(VERT_CODE, FRAG_CODE)
        if image is not None:
            self.set_data(image, image_box)
        if world_box is not None:
            self.set_world_box(world_box)


    def get_data(self):
        return self.image

    def set_data(self, image, image_box=None):
        if isinstance(image, str) and image == 'crate':
            image = load_crate()
        if image_box is not None:
            self.image = image = image[image_box.b:image_box.t, image_box.l:image_box.r]
            print("clipping")
        else:
            self.image = image
        # get the texture queued
        self.program['u_texture'] = gloo.Texture2D(image)




