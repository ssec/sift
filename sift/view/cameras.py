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
__docformat__ = 'reStructuredText'
__author__ = 'davidh'

import logging

import numpy as np
from vispy.geometry import Rect
from vispy.scene import PanZoomCamera, BaseCamera
from vispy.util.keys import SHIFT

LOG = logging.getLogger(__name__)


class PanZoomProbeCamera(PanZoomCamera):
    """Camera that maps mouse presses to events for probing.
    """

    # def viewbox_mouse_event(self, event):
    #     """
    #     The SubScene received a mouse event; update transform
    #     accordingly.
    #
    #     Parameters
    #     ----------
    #     event : instance of Event
    #         The event.
    #     """
    #     if event.handled or not self.interactive:
    #         return
    #
    #     # Scrolling
    #     BaseCamera.viewbox_mouse_event(self, event)
    #     event.handled = False

    def __init__(self, *args, **kwargs):
        self._pan_limits = kwargs.pop("pan_limits", None)
        self._zoom_limits = kwargs.pop("zoom_limits", (0.001, 0.001))
        super(PanZoomProbeCamera, self).__init__(*args, **kwargs)

    def _viewbox_set(self, viewbox):
        """ Friend method of viewbox to register itself.
        """
        self._viewbox = viewbox
        # Connect
        viewbox.events.mouse_press.connect(self.viewbox_mouse_event)
        viewbox.events.mouse_release.connect(self.viewbox_mouse_event)
        viewbox.events.mouse_move.connect(self.viewbox_mouse_event)
        viewbox.events.mouse_wheel.connect(self.viewbox_mouse_event)
        viewbox.events.resize.connect(self.viewbox_resize_event)
        # todo: also add key events! (and also on viewbox (they're missing)

    def _viewbox_unset(self, viewbox):
        """ Friend method of viewbox to unregister itself.
        """
        self._viewbox = None
        # Disconnect
        viewbox.events.mouse_press.disconnect(self.viewbox_mouse_event)
        viewbox.events.mouse_release.disconnect(self.viewbox_mouse_event)
        viewbox.events.mouse_move.disconnect(self.viewbox_mouse_event)
        viewbox.events.mouse_wheel.disconnect(self.viewbox_mouse_event)
        viewbox.events.resize.disconnect(self.viewbox_resize_event)

    @property
    def rect(self):
        """ The rectangular border of the ViewBox visible area, expressed in
        the coordinate system of the scene.

        Note that the rectangle can have negative width or height, in
        which case the corresponding dimension is flipped (this flipping
        is independent from the camera's ``flip`` property).
        """
        return self._rect

    @rect.setter
    def rect(self, args):
        if isinstance(args, tuple):
            rect = Rect(*args)
        else:
            rect = Rect(args)

        # THIS DOESN'T WORK. The `rect` object doesn't represent a fixed coordinate system
        # (0 and 1 have different meanings at different zoom levels)

        # Limit how far we can pan and zoom out
        # if self.parent.transform
        # if rect.left >= self._pan_limits[2]:
        #     print("To far right")
        #     return
        # elif rect.left <= self._pan_limits[0]:
        #     print("To far left")
        #     return
        # if self._pan_limits is not None:
        #     new_left = np.clip(rect.left, self._pan_limits[0] - self._zoom_limits[0], self._pan_limits[2])
        #     new_right = np.clip(rect.right, self._pan_limits[0], self._pan_limits[2] + self._zoom_limits[0])
        #     new_bottom = np.clip(rect.bottom, self._pan_limits[1] - self._zoom_limits[0], self._pan_limits[3])
        #     new_top = np.clip(rect.top, self._pan_limits[1], self._pan_limits[3] + self._zoom_limits[1])
        # else:
        #     new_left = rect.left
        #     new_right = rect.right
        #     new_bottom = rect.bottom
        #     new_top = rect.top
        #
        # # Limit how far we can zoom in
        # new_size_x = new_right - new_left
        # new_size_y = new_top - new_bottom
        # if self._zoom_limits is not None:
        #     new_size_x = max(new_size_x, self._zoom_limits[0])
        #     new_size_y = max(new_size_y, self._zoom_limits[1])
        # rect = Rect(new_left, new_bottom, new_size_x, new_size_y)

        if self._rect != rect:
            self._rect = rect
            self.view_changed()

    def viewbox_mouse_event(self, event):
        """
        The SubScene received a mouse event; update transform
        accordingly.

        Parameters
        ----------
        event : instance of Event
            The event.
        """
        if event.handled or not self.interactive:
            return

        # Scrolling
        BaseCamera.viewbox_mouse_event(self, event)
        # print("In camera event: ", event.button, event.buttons, event.mouse_event.modifiers)

        if event.type == 'mouse_wheel':
            center = self._scene_transform.imap(event.pos)
            self.zoom((1 + self.zoom_factor) ** (-event.delta[1] * 30), center)
            event.handled = True

        elif event.type == 'mouse_move':
            if event.press_event is None:
                return

            modifiers = event.mouse_event.modifiers
            p1 = event.mouse_event.press_event.pos
            p2 = event.mouse_event.pos

            if 1 in event.buttons and not modifiers:
                # Translate
                p1 = np.array(event.last_event.pos)[:2]
                p2 = np.array(event.pos)[:2]
                p1s = self._transform.imap(p1)
                p2s = self._transform.imap(p2)
                self.pan(p1s - p2s)
                event.handled = True
            elif 1 in event.buttons and modifiers == (SHIFT,):
                # Zoom
                p1c = np.array(event.last_event.pos)[:2]
                p2c = np.array(event.pos)[:2]
                scale = ((1 + self.zoom_factor) **
                         ((p1c - p2c) * np.array([1, -1])))
                center = self._transform.imap(event.press_event.pos[:2])
                self.zoom(scale, center)
                event.handled = True
            else:
                event.handled = False
        elif event.type == 'mouse_press':
            # accept the event if it is button 1 or 2.
            # This is required in order to receive future events
            # event.handled = event.button in [1, 2]
            event.handled = event.button == 1
        else:
            event.handled = False
