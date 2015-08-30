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
# from PyQt4.QtCore import *
# from PyQt4.QtOpenGL import QGLWidget, QGLFormat
from vispy import app, gloo
import numpy as np
import scipy.misc as spm
from vispy.util.transforms import translate, rotate, ortho

from cspov.common import box
from cspov.view.Program import RGBATileProgram

__author__ = 'rayg'
__docformat__ = 'reStructuredText'

import logging

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
        # event.accept()
        # pos = event.pos()
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








class CspovMainMapWidget(app.Canvas):

    # signals
    # viewportDidChange = pyqtSignal(box)

    # members
    _activity_stack = None  # Behavior object stack which we push/pop for primary activity; activity[-1] is what we're currently doing
    layers = None  # layers we're currently displaying, last on top
    viewport = None  # box with world coordinates of what we're showing

    _testtile = None

    def __init__(self, **kwargs):
        # http://stackoverflow.com/questions/17167194/how-to-make-updategl-realtime-in-qt
        #
        super(CspovMainMapWidget, self).__init__(**kwargs)

        # self.layers = [TestLayer()]
        self.layers = [] # FIXME [TestTileLayer()]
        self._activity_stack = [Idling(self)]

        aspect = self.size[1] / float(self.size[0])
        self.viewport = vp = box(l=-4, r=4, b=-4*aspect, t=4*aspect)
        # FIXME: use this viewport
        # self.viewport = box(l=-MAX_EXCURSION_X/4, b=-MAX_EXCURSION_Y/1.5, r=MAX_EXCURSION_X/4, t=MAX_EXCURSION_Y/1.5)

        # self.viewportDidChange.connect(self.updateGL)
        # assert(self.updatesEnabled())
        # self.setUpdatesEnabled(True)
        # self.setAutoBufferSwap(True)
        # self.setMouseTracking(True)  # gives us mouseMoveEvent calls in Idling
        # self.setAutoBufferSwap(True)
        # assert(self.hasMouseTracking())

        self._testtile = RGBATileProgram(world_box=box(l=-4.0, r=4.0, t=2.0, b=-2.0),
                                         image=spm.imread('cspov/data/shadedrelief.jpg'),
                                         image_box=box(b=3000, t=3512, l=3000, r=4024))

        # Handle transformations
        self.init_transforms()
        self.update_proj()

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
            pw, ph = self.size
            wh, ww = self.viewport.t - self.viewport.b, self.viewport.r - self.viewport.l
            wdy, wdx = float(pdz)/ph*wh, float(pdz)/pw*ww
        nvp = box(b=self.viewport.b+wdy, t=self.viewport.t-wdy, l=self.viewport.l+wdx, r=self.viewport.r-wdx)
        # print("pan viewport {0!r:s} => {1!r:s}".format(self.viewport, nvp))
        self.viewport = nvp
        self.update_proj()
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
            pw, ph = self.size
            # ph, pw = float(s.height()), float(s.width())
            wh, ww = self.viewport.t - self.viewport.b, self.viewport.r - self.viewport.l
            wdy, wdx = float(pdy)/ph*wh, float(pdx)/pw*ww
        elif (wdy, wdx) is (None, None):
            return self.viewport
        # print("pan {}y {}x".format(pdy, pdx))
        nvp = box(b=self.viewport.b+wdy, t=self.viewport.t+wdy, l=self.viewport.l+wdx, r=self.viewport.r+wdx)
        # print("pan viewport {0!r:s} => {1!r:s}".format(self.viewport, nvp))
        self.viewport = nvp
        self.update_proj()  # propagate projection matrix
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
        self._testtile.set_mvp(self.model, self.view, self.projection)

    def update_transforms(self, event):
        self.theta += .1
        self.phi += .1
        self.model = np.dot(rotate(self.theta, (0, 0, 1)),
                            rotate(self.phi, (0, 1, 0)))
        self._testtile.set_mvp(self.model)
        self.update()

    def on_resize(self, event):
        self.update_proj()

    def update_proj(self, event=None):
        if event is not None:
            gloo.set_viewport(0, 0, *event.physical_size)
        else:
            gloo.set_viewport(0, 0, self.physical_size[0], self.physical_size[1])
        vp = self.viewport
        # self.projection = perspective(45.0, self.size[0] /
        #                               float(self.size[1]), 2.0, 10.0)
        aspect = self.size[1] / float(self.size[0])
        #self.projection = ortho(-4, 4, -4*aspect, 4*aspect, -10, 10)
        self.projection = ortho(
            vp.l, vp.r,
            vp.b, vp.t,
            -10, 10
        )
        self._testtile.set_mvp(projection=self.projection)

    def on_draw(self, event):
        gloo.clear()
        for layer in self.layers:
            layer.on_draw(event)
        if self._testtile:
            self._testtile.draw()

    # def on_compile(self):
    #     vert_code = str(self.vertEdit.toPlainText())
    #     frag_code = str(self.fragEdit.toPlainText())
    #     self.canvas.program.set_shaders(vert_code, frag_code)


    def key_press(self, key):
        print('down', repr(key))

    def key_release(self, key):
        print('up', repr(key))

    def on_mouse_release(self, event):
        event = event.native  # FIXME: stop using .native, send the vispy event and refactor the Activities
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

    def on_mouse_move(self, event):
        # print("mouse_move")
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

    def on_mouse_press(self, event):
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

    def on_mouse_wheel(self, event):
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