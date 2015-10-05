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

from vispy.scene import BaseCamera, STTransform, MatrixTransform
from vispy.geometry import Rect

LOG = logging.getLogger(__name__)


class ProbeCamera(BaseCamera):
    """Camera that maps mouse presses to events for probing.
    """
    _state_props = BaseCamera._state_props + ('rect', )

    def __init__(self, rect=(0, 0, 1, 1), aspect=None, **kwargs):
        super(ProbeCamera, self).__init__(**kwargs)

        self.transform = STTransform()

        # Set camera attributes
        self.aspect = aspect
        self._rect = None
        self.rect = rect

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
        event.handled = False

    @property
    def aspect(self):
        """ The ratio between the x and y dimension. E.g. to show a
        square image as square, the aspect should be 1. If None, the
        dimensions are scaled automatically, dependening on the
        available space. Otherwise the ratio between the dimensions
        is fixed.
        """
        return self._aspect

    @aspect.setter
    def aspect(self, value):
        if value is None:
            self._aspect = None
        else:
            self._aspect = float(value)
        self.view_changed()

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

        if self._rect != rect:
            self._rect = rect
            self.view_changed()

    @property
    def center(self):
        rect = self._rect
        return 0.5 * (rect.left+rect.right), 0.5 * (rect.top+rect.bottom), 0

    @center.setter
    def center(self, center):
        if not (isinstance(center, (tuple, list)) and len(center) in (2, 3)):
            raise ValueError('center must be a 2 or 3 element tuple')
        rect = self.rect or Rect(0, 0, 1, 1)
        # Get half-ranges
        x2 = 0.5 * (rect.right - rect.left)
        y2 = 0.5 * (rect.top - rect.bottom)
        # Apply new ranges
        rect.left = center[0] - x2
        rect.right = center[0] + x2
        rect.bottom = center[1] - y2
        rect.top = center[1] + y2
        #
        self.rect = rect

    def _set_range(self, init):
        if init and self._rect is not None:
            return
        # Convert limits to rect
        w = self._xlim[1] - self._xlim[0]
        h = self._ylim[1] - self._ylim[0]
        self.rect = self._xlim[0], self._ylim[0], w, h

    def zoom(self, factor, center=None):
        """ Zoom in (or out) at the given center

        Parameters
        ----------
        factor : float or tuple
            Fraction by which the scene should be zoomed (e.g. a factor of 2
            causes the scene to appear twice as large).
        center : tuple of 2-4 elements
            The center of the view. If not given or None, use the
            current center.
        """
        assert len(center) in (2, 3, 4)
        # Get scale factor, take scale ratio into account
        if np.isscalar(factor):
            scale = [factor, factor]
        else:
            if len(factor) != 2:
                raise TypeError("factor must be scalar or length-2 sequence.")
            scale = list(factor)
        if self.aspect is not None:
            scale[0] = scale[1]

        # Init some variables
        center = center if (center is not None) else self.center
        # Make a new object (copy), so that allocation will
        # trigger view_changed:
        rect = Rect(self.rect)
        # Get space from given center to edges
        left_space = center[0] - rect.left
        right_space = rect.right - center[0]
        bottom_space = center[1] - rect.bottom
        top_space = rect.top - center[1]
        # Scale these spaces
        rect.left = center[0] - left_space * scale[0]
        rect.right = center[0] + right_space * scale[0]
        rect.bottom = center[1] - bottom_space * scale[1]
        rect.top = center[1] + top_space * scale[1]
        self.rect = rect

    def _update_transform(self):
        rect = self.rect
        self._real_rect = Rect(rect)
        vbr = self._viewbox.rect.flipped(x=self.flip[0], y=(not self.flip[1]))
        d = self.depth_value

        # apply scale ratio constraint
        if self._aspect is not None:
            # Aspect ratio of the requested range
            requested_aspect = (rect.width / rect.height
                                if rect.height != 0 else 1)
            # Aspect ratio of the viewbox
            view_aspect = vbr.width / vbr.height if vbr.height != 0 else 1
            # View aspect ratio needed to obey the scale constraint
            constrained_aspect = abs(view_aspect / self._aspect)

            if requested_aspect > constrained_aspect:
                # view range needs to be taller than requested
                dy = 0.5 * (rect.width / constrained_aspect - rect.height)
                self._real_rect.top += dy
                self._real_rect.bottom -= dy
            else:
                # view range needs to be wider than requested
                dx = 0.5 * (rect.height * constrained_aspect - rect.width)
                self._real_rect.left -= dx
                self._real_rect.right += dx

        # Apply mapping between viewbox and cam view
        self.transform.set_mapping(self._real_rect, vbr, update=False)
        # Scale z, so that the clipping planes are between -alot and +alot
        self.transform.zoom((1, 1, 1/d))

        # We've now set self.transform, which represents our 2D
        # transform When up is +z this is all. In other cases,
        # self.transform is now set up correctly to allow pan/zoom, but
        # for the scene we need a different (3D) mapping. When there
        # is a minus in up, we simply look at the scene from the other
        # side (as if z was flipped).

        if self.up == '+z':
            thetransform = self.transform
        else:
            rr = self._real_rect
            tr = MatrixTransform()
            d = d if (self.up[0] == '+') else -d
            pp1 = [(vbr.left, vbr.bottom, 0), (vbr.left, vbr.top, 0),
                   (vbr.right, vbr.bottom, 0), (vbr.left, vbr.bottom, 1)]
            # Get Mapping
            if self.up[1] == 'z':
                pp2 = [(rr.left, rr.bottom, 0), (rr.left, rr.top, 0),
                       (rr.right, rr.bottom, 0), (rr.left, rr.bottom, d)]
            elif self.up[1] == 'y':
                pp2 = [(rr.left, 0, rr.bottom), (rr.left, 0, rr.top),
                       (rr.right, 0, rr.bottom), (rr.left, d, rr.bottom)]
            elif self.up[1] == 'x':
                pp2 = [(0, rr.left, rr.bottom), (0, rr.left, rr.top),
                       (0, rr.right, rr.bottom), (d, rr.left, rr.bottom)]
            # Apply
            tr.set_mapping(np.array(pp2), np.array(pp1))
            thetransform = tr

        # Set on viewbox
        self._set_scene_transform(thetransform)
