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
from vispy import gloo
from vispy.geometry import create_plane
from vispy.io import load_crate

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
varying vec2 v_texcoord;

void main()
{
    float ty = v_texcoord.y;
    float tx = v_texcoord.x;
    gl_FragColor = texture2D(u_texture, vec2(tx, ty));
}
"""


class GlooRGBTile(object):
    """
    A GL program which plots a planar tile with a texture map
    texture map is assumed oriented such that ascending pixel row is ascending coordinate from bottom left of screen
    (typically requires inverting the Y direction)
    """
    program = None
    image = None
    world_box = None
    faces = None
    z = 0.0

    def __init__(self, world_box=None, image=None, image_box=None):
        super(GlooRGBTile, self).__init__()
        self.program = gloo.Program(VERT_CODE, FRAG_CODE)
        if image is not None:
            self.set_texture(image, image_box)
        if world_box is not None:
            self.set_world(world_box)

    def set_world(self, world_box):
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
        self.world_box = world_box
        self.program['u_z'] = self.z

    def set_texture(self, image, image_box=None):
        if isinstance(image, str) and image == 'crate':
            image = load_crate()
        if image_box is not None:
            self.image = image = image[image_box.b:image_box.t, image_box.l:image_box.r]
            print("clipping")
        else:
            self.image = image
        # get the texture queued
        self.program['u_texture'] = gloo.Texture2D(image)


    def set_z(self, z):
        self.program['u_z'] = self.z = z


    def set_mvp(self, model=None, view=None, projection=None):
        if model is not None:
            self.program['u_model'] = model
        if view is not None:
            self.program['u_view'] = view
        if projection is not None:
            self.program['u_projection'] = projection


    def draw(self):
        assert(self.image is not None)
        assert(self.world_box is not None)
        self.program.draw('triangles', self.faces)