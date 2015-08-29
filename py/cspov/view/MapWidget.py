#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
.py
~~~

PURPOSE


REFERENCES


REQUIRES


:author: R.K.Garcia <rayg@ssec.wisc.edu>
:copyright: 2014 by University of Wisconsin Regents, see AUTHORS for more details
:license: GPLv3, see LICENSE for more details
"""
from OpenGL.GL import *
# from PyQt4.QtCore import *
# from PyQt4.QtOpenGL import QGLWidget, QGLFormat
from vispy import app, gloo
import numpy as np
from cspov.view.Layer import TestTileLayer, Layer
from cspov.common import MAX_EXCURSION_Y, MAX_EXCURSION_X, box
from vispy.util.transforms import perspective, translate, rotate, ortho
from vispy.io import read_mesh, load_data_file, load_crate

__author__ = 'rayg'
__docformat__ = 'reStructuredText'

import os, sys
import logging, unittest, argparse

LOG = logging.getLogger(__name__)


class MapWidgetActivity(object):
    """
    Major mouse activities represented as objects, to simplify main window control logic
    Right now this is crude and we'll eventually run out of road and have to rethink it
    For now it's a good leg up.
    Eventually we want something more like a Behavior wired in at the Designer level??

    The Map window has an activity which is the main thing it's doing
    return None for "I remain status quo"
    return False for "dismiss me"
    return a new activity for "send to this guy"
    """
    main = None   # main map widget we serve

    def __init__(self, main):
        super(MapWidgetActivity, self).__init__()
        self.main = main

    def layer_paint_parms(self):
        """
        return additional keyword parameters to be sent to layers when they're painting
        """
        return {}

    def mouseReleaseEvent(self, event):
        return None

    def mouseMoveEvent(self, event):
        return None

    def mousePressEvent(self, event):
        return None

    def wheelEvent(self, event):
        return None


class UserPanningMap(MapWidgetActivity):
    """
    user mouses down
    user drags map
        click and drag OR
        Mac: scroll surface option
    user mouses up
    """
    def mouseReleaseEvent(self, event):
        """
        user is done zooming
        go back to idle
        draw at higher resolution (not fast-draw)
        """
        print("done panning")
        return False  # we're done, dismiss us

    def mouseMoveEvent(self, event):
        """
        change the visible region
        invalidate the view
        """
        x, y = event.x(), event.y()
        pdx = self.lx - x
        # GL coordinates are reversed from screen coordinates
        pdy = y - self.ly
        self.main.panViewport(pdy=pdy, pdx=pdx)
        # self.main.updateGL()  # repaint() is faster if we need it
        self.lx, self.ly = x, y
        # print("pan dx={0} dy={1}".format(pdx,pdy))
        return None

    def mousePressEvent(self, event):
        """
        Idling probably sent this our way, use it to note where we're starting from
          Also optionally change cursors
        """
        print("mouse down, starting pan")
        self.lx, self.ly = event.x(), event.y()
        return None


class UserZoomingMap(MapWidgetActivity):
    """
    user starts zooming
        scroll wheel OR
        chording with keyboard and mouse
    user zooms inward
    user zooms outward
    user ends zooming
    """
    def wheelEvent(self, event):
        event.accept()
        pos = event.pos()
        delta = event.delta()
        self.main.zoomViewport(delta)
        return None

    def mouseMoveEvent(self, event):
        # FIXME: does it work to drop the zooming by way of a mouse move?
        return False

    def mousePressEvent(self, event):
        return False

    def mouseReleaseEvent(self, event):
        return False


class UserZoomingRegion(MapWidgetActivity):
    """
    user starts region selection
        click and drag with tool OR
        click and drag middle mouse button??
    user continues selecting box region
    user finishes selecting region
    """
    def mouseReleaseEvent(self, event):
        """
        user is done zooming
        go back to idle
        draw at higher resolution (not fast-draw)
        """
        return False

    def mouseMoveEvent(self, event):
        return None

    def mousePressEvent(self, event):
        return False  # we should never get this


class Idling(MapWidgetActivity):
    """
    This is the default behavior we do when nothing else is going on
    :param Behavior:
    :return:
    """

    def mouseMoveEvent(self, event):
        # FIXME: send world coordinates to cursor coordinate content probe
        # print("mousie") # yeah this works
        return None

    def mousePressEvent(self, event):
        """
        Drag with left mouse button to pan
        Drag with right mouse button to zoom (to point?)
        Drag with middle button for rectangle zoom? Or drag with modifier key to zoom?
        """
        return UserPanningMap(self.main)  # FIXME more routes to route

    def wheelEvent(self, event):
        return UserZoomingMap(self.main)


class Animating(MapWidgetActivity):
    """
    When we're doing an animation cycle
    :param Behavior:
    :return:
    """

VERT_CODE = """
uniform   mat4 u_model;
uniform   mat4 u_view;
uniform   mat4 u_projection;

attribute vec3 a_position;
attribute vec2 a_texcoord;

varying vec2 v_texcoord;

void main()
{
    v_texcoord = a_texcoord;
    gl_Position = u_projection * u_view * u_model * vec4(a_position,1.0);
    //gl_Position = vec4(a_position,1.0);
}
"""


FRAG_CODE = """
uniform sampler2D u_texture;
varying vec2 v_texcoord;

void main()
{
    float ty = v_texcoord.y;
    float tx = sin(ty*50.0)*0.01 + v_texcoord.x;
    gl_FragColor = texture2D(u_texture, vec2(tx, ty));
}
"""

# from imshow_cuts.py
image_vertex = """
attribute vec2 position;
attribute vec2 texcoord;

varying vec2 v_texcoord;
void main()
{
    gl_Position = vec4(position, 0.0, 1.0 );
    v_texcoord = texcoord;
}
"""

image_fragment = """
uniform float vmin;
uniform float vmax;
uniform float cmap;
uniform float n_colormaps;

uniform sampler2D image;
uniform sampler2D colormaps;

varying vec2 v_texcoord;
void main()
{
    float value = texture2D(image, v_texcoord).r;
    float index = (cmap+0.5) / n_colormaps;

    if( value < vmin ) {
        gl_FragColor = texture2D(colormaps, vec2(0.0,index));
    } else if( value > vmax ) {
        gl_FragColor = texture2D(colormaps, vec2(1.0,index));
    } else {
        value = (value-vmin)/(vmax-vmin);
        value = 1.0/512.0 + 510.0/512.0*value;
        gl_FragColor = texture2D(colormaps, vec2(value,index));
    }
}
"""

# Read cube data
# FIXME: remove
positions, faces, normals, texcoords = \
    read_mesh(load_data_file('orig/cube.obj'))
colors = np.random.uniform(0, 1, positions.shape).astype('float32')

faces_buffer = gloo.IndexBuffer(faces.astype(np.uint16))


#
# class TestSingleImageLayer(Layer):
#
#     def __init__(self, **kwargs):
#         super(TestImageLayer, self).__init__(**kwargs)
#         self.image = Program(image_vertex, image_fragment, 4)
#         self.image['position'] = (-1, -1), (-1, +1), (+1, -1), (+1, +1)
#         self.image['texcoord'] = (0, 0), (0, +1), (+1, 0), (+1, +1)
#         self.image['vmin'] = +0.0
#         self.image['vmax'] = +1.0
#         self.image['cmap'] = 0  # Colormap index to use
#         self.image['colormaps'] = colormaps
#         self.image['n_colormaps'] = colormaps.shape[0]
#         self.image['image'] = I.astype('float32')
#         self.image['image'].interpolation = 'linear'
#

class CspovMainMapWidget(app.Canvas):

    # signals
    # viewportDidChange = pyqtSignal(box)

    # members
    _activity_stack = None  # Behavior object stack which we push/pop for primary activity; activity[-1] is what we're currently doing
    layers = None  # layers we're currently displaying, last on top
    viewport = None  # box with world coordinates of what we're showing

    def __init__(self, **kwargs):
        # http://stackoverflow.com/questions/17167194/how-to-make-updategl-realtime-in-qt
        #
        super(CspovMainMapWidget, self).__init__(**kwargs)

        # self.layers = [TestLayer()]
        self.layers = [] # FIXME [TestTileLayer()]
        self._activity_stack = [Idling(self)]
        self.viewport = box(l=-MAX_EXCURSION_X/4, b=-MAX_EXCURSION_Y/1.5, r=MAX_EXCURSION_X/4, t=MAX_EXCURSION_Y/1.5)
        # self.viewportDidChange.connect(self.updateGL)
        # assert(self.updatesEnabled())
        # self.setUpdatesEnabled(True)
        # self.setAutoBufferSwap(True)
        # self.setMouseTracking(True)  # gives us mouseMoveEvent calls in Idling
        # self.setAutoBufferSwap(True)
        # assert(self.hasMouseTracking())

        self.program = gloo.Program(VERT_CODE, FRAG_CODE)
        # Set attributes
        self.program['a_position'] = gloo.VertexBuffer(positions)
        self.program['a_texcoord'] = gloo.VertexBuffer(texcoords)
        self.program['u_texture'] = gloo.Texture2D(load_crate())

        # Handle transformations
        self.init_transforms()
        self.apply_zoom()

        gloo.set_clear_color((0, 0, 0, 1))
        gloo.set_state(depth_test=True)

        self._timer = app.Timer('auto', connect=self.update_transforms)
        self._timer.start()

        self.show()

    @property
    def activity(self):
        return self._activity_stack[-1]

    def zoomViewport(self, pdz=None, wdz=None):
        if pdz is not None:
            s = self.size()
            ph, pw = float(s.height()), float(s.width())
            wh, ww = self.viewport.t - self.viewport.b, self.viewport.r - self.viewport.l
            wdy, wdx = float(pdz)/ph*wh, float(pdz)/pw*ww
        nvp = box(b=self.viewport.b+wdy, t=self.viewport.t-wdy, l=self.viewport.l+wdx, r=self.viewport.r-wdx)
        # print("pan viewport {0!r:s} => {1!r:s}".format(self.viewport, nvp))
        self.viewport = nvp
        # self.viewportDidChange.emit(nvp)
        self.update()

    def panViewport(self, pdy=None, pdx=None, wdy=None, wdx=None):
        """
        displace view by pixel or world coordinates
        does not queue screen update
        :param pdy: displacement in pixels, y:int
        :param pdx: displacement in pixels, x:int
        :param wdy: displacement in world y:float
        :param wdx: displacement in world x:float
        :return: new world viewport
        """
        # print(" viewport pan requested {0!r:s}".format((pdy,pdx,wdy,wdx)))
        if (pdy, pdx) is not (None, None):
            s = self.size()
            ph, pw = float(s.height()), float(s.width())
            wh, ww = self.viewport.t - self.viewport.b, self.viewport.r - self.viewport.l
            wdy, wdx = float(pdy)/ph*wh, float(pdx)/pw*ww
        elif (wdy, wdx) is (None, None):
            return self.viewport
        nvp = box(b=self.viewport.b+wdy, t=self.viewport.t+wdy, l=self.viewport.l+wdx, r=self.viewport.r+wdx)
        # print("pan viewport {0!r:s} => {1!r:s}".format(self.viewport, nvp))
        self.viewport = nvp
        # self.viewportDidChange.emit(nvp)
        self.update()
        return self.viewport

    #
    # GLOO
    #

    def init_transforms(self):
        self.theta = 0
        self.phi = 0
        self.view = translate((0, 0, -5))
        self.model = np.eye(4, dtype=np.float32)
        self.projection = np.eye(4, dtype=np.float32)

        self.program['u_model'] = self.model
        self.program['u_view'] = self.view

    def update_transforms(self, event):
        self.theta += .5
        self.phi += .5
        self.model = np.dot(rotate(self.theta, (0, 0, 1)),
                            rotate(self.phi, (0, 1, 0)))
        self.program['u_model'] = self.model
        self.update()

    def on_resize(self, event):
        self.apply_zoom()

    def apply_zoom(self, event=None):
        if event is not None:
            gloo.set_viewport(0, 0, *event.physical_size)
        else:
            gloo.set_viewport(0, 0, self.physical_size[0], self.physical_size[1])
        vp = self.viewport
        self.projection = perspective(45.0, self.size[0] /
                                      float(self.size[1]), 2.0, 10.0)
        # self.projection = ortho(
        #     vp.l, vp.r,
        #     vp.b, vp.t,
        #     -50, 50
        # )
        self.program['u_projection'] = self.projection

    def on_draw(self, event):
        gloo.clear()
        for layer in self.layers:
            layer.on_draw(event)
        self.program.draw('triangles', faces_buffer)

    # def on_compile(self):
    #     vert_code = str(self.vertEdit.toPlainText())
    #     frag_code = str(self.fragEdit.toPlainText())
    #     self.canvas.program.set_shaders(vert_code, frag_code)


    def key_press(self, key):
        print('down', repr(key))

    def key_release(self, key):
        print('up', repr(key))

    def mouse_release(self, event):
        event = event.native  # FIXME: stop using .native
        newact = True
        while newact is not None:
            newact = self.activity.mouseReleaseEvent(event)
            if newact is None:
                break
            if newact is False:
                self._activity_stack.pop()
                continue
            assert(isinstance(newact, MapWidgetActivity))
            self._activity_stack.append(newact)

    def mouse_move(self, event):
        event = event.native
        newact = True
        while newact is not None:
            newact = self.activity.mouseMoveEvent(event)
            if newact is None:
                return
            if newact is False:
                self._activity_stack.pop()
                continue
            assert(isinstance(newact, MapWidgetActivity))
            self._activity_stack.append(newact)

    def mouse_press(self, event):
        event = event.native
        newact = True
        while newact is not None:
            newact = self.activity.mousePressEvent(event)
            if newact is None:
                return
            if newact is False:
                self._activity_stack.pop()
                continue
            assert(isinstance(newact, MapWidgetActivity))
            self._activity_stack.append(newact)

    def mouse_wheel(self, event):
        event = event.native
        newact = True
        while newact is not None:
            newact = self.activity.wheelEvent(event)
            if newact is None:
                return
            if newact is False:
                self._activity_stack.pop()
                continue
            assert(isinstance(newact, MapWidgetActivity))
            self._activity_stack.append(newact)