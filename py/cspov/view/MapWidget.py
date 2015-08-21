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
from PyQt4.QtCore import *
from PyQt4.QtOpenGL import QGLWidget
from cspov.view.Layer import TestTileLayer
from cspov.common import MAX_EXCURSION_Y, MAX_EXCURSION_X, box

__author__ = 'rayg'
__docformat__ = 'reStructuredText'

import os, sys
import logging, unittest, argparse

LOG = logging.getLogger(__name__)


class MapWidgetActivity(QObject):
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
        print("pan dx={0} dy={1}".format(pdx,pdy))
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
        # FIXME

        return None


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


class CspovMainMapWidget(QGLWidget):

    # signals
    viewportDidChange = pyqtSignal(box)

    # members
    _activity_stack = None  # Behavior object stack which we push/pop for primary activity; activity[-1] is what we're currently doing
    layers = None  # layers we're currently displaying, last on top
    viewport = None  # box with world coordinates of what we're showing

    def __init__(self, parent=None):
        super(CspovMainMapWidget, self).__init__(parent)
        # self.layers = [TestLayer()]
        self.layers = [TestTileLayer()]
        self._activity_stack = [Idling(self)]
        self.viewport = box(l=-MAX_EXCURSION_X, b=-MAX_EXCURSION_Y, r=MAX_EXCURSION_X, t=MAX_EXCURSION_Y)
        self.viewportDidChange.connect(self.updateGL)
        # assert(self.updatesEnabled())
        # self.setUpdatesEnabled(True)
        # self.setAutoBufferSwap(True)
        self.setMouseTracking(True)  # gives us mouseMoveEvent calls in Idling
        # assert(self.hasMouseTracking())

    @property
    def activity(self):
        return self._activity_stack[-1]

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
        print(" viewport pan requested {0!r:s}".format((pdy,pdx,wdy,wdx)))
        if (pdy, pdx) is not (None, None):
            s = self.size()
            ph, pw = float(s.height()), float(s.width())
            wh, ww = self.viewport.t - self.viewport.b, self.viewport.r - self.viewport.l
            wdy, wdx = float(pdy)/ph*wh, float(pdx)/pw*ww
        elif (wdy, wdx) is (None, None):
            return self.viewport
        nvp = box(b=self.viewport.b+wdy, t=self.viewport.t+wdy, l=self.viewport.l+wdx, r=self.viewport.r+wdx)
        print("pan viewport {0!r:s} => {1!r:s}".format(self.viewport, nvp))
        self.viewport = nvp
        self.viewportDidChange.emit(nvp)
        return self.viewport

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT)
        glDisable(GL_CULL_FACE)
        for layer in self.layers:
            needs_rerender = layer.paint()
        # FIXME: schedule re-render for layers that are no longer optimal

    def resizeGL(self, w, h):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        # glOrtho(-50, 50, -50, 50, -50.0, 50.0)
        vp = self.viewport
        glOrtho(vp.l, vp.r,
                vp.b, vp.t,
                -50, 50)
        glViewport(0, 0, w, h)

    def initializeGL(self):
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)

        # print(glGetString(GL_VERSION))
        # print("GLSL {}".format(glGetIntegerv(GL_SHADING_LANGUAGE_VERSION)))

    def keyPressEvent(self, key):
        print(repr(key))
        self.updateGL()

    def mouseReleaseEvent(self, event):
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
        self.updateGL()  # FIXME DEBUG

    def mouseMoveEvent(self, event):
        newact = True
        self.updateGL()
        while newact is not None:
            newact = self.activity.mouseMoveEvent(event)
            if newact is None:
                return
            if newact is False:
                self._activity_stack.pop()
                continue
            assert(isinstance(newact, MapWidgetActivity))
            self._activity_stack.append(newact)

    def mousePressEvent(self, event):
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

    def wheelEvent(self, event):
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