#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:author: R.K.Garcia <rkgarcia@wisc.edu>
:copyright: 2018 by University of Wisconsin Regents, see AUTHORS for more details
:license: GPLv3, see LICENSE for more details
"""

import os, sys
import logging, unittest
from collections import namedtuple
from datetime import datetime, timedelta
from enum import Enum
from typing import Tuple, Optional

from PyQt4.QtCore import QObject, QRectF

LOG = logging.getLogger(__name__)


DEFAULT_TIMELINE_HEIGHT = 48


class TimelineFrameState(Enum):
    """Displayed state of frames, corresponds to a color or style.
    Typically combines information from Workspace and Scenegraph, does not correspond to selection.
    """
    UNKNOWN = 0
    AVAILABLE = 1  # can be imported from native resources but has not been
    IMPORTED = 2  # imported into workspace
    READY = 3  # available as a memory map, but may not be fully resident in RAM or VRAM
    ACTIVE = 4  # both ready to go into VRAM (or already resident), and participating in the application scene graph, possibly as part of an animation
    VISIBLE = 5  # active, and currently on-screen for user to view


class TimelineTrackState(Enum):
    UNKNOWN = 0
    DISABLED = 1
    ENABLED = 2


ztdtup = namedtuple("ztdtup", ('z', 't', 'd'))  # z level > 0 ascending downward, start time: datetime, duration: timedelta



class TimelineCoordTransform(QObject):
    """
    configurable transformations between time and display space, typically a tool used by scene/timeline/frame
    typically one instance per scene!
    default is 1px=1s in X
    Z order starts at 0 (topmost timeline) with +1 being beneath 0, etc (lower numbers overlay higher; this is inverse from scenegraph)
    typically QTimelineItem will have a nonzero Z determining its y coordinate in the scene
    QTimelineItem contains zero or more QFrameItems
    QFrameItems coordinates are relative to their QTimelineItem parent and (for now) are z=0
        When we have expandable RGB timelines this will change
    """
    _time_base: datetime = None
    _time_unit: timedelta = None
    _track_height: float = None

    def __init__(self, time_base: datetime = None, time_unit: timedelta = timedelta(seconds=1), track_height=DEFAULT_TIMELINE_HEIGHT):
        """

        Args:
            time_base: "zero" value of time; defaults to now UTC as this is somewhat arbitrary
            time_unit: time width corresponding to delta-X=1.0, defaults to 1.0s
            track_height: pixels high to display individual ttracks
        """
        super(TimelineCoordTransform, self).__init__()
        self._time_base = time_base or datetime.utcnow()
        self._time_unit = time_unit
        self._track_height = track_height

    def calc_time_duration(self, scene_x: [float, None], scene_w: [float, None]) -> Tuple[Optional[datetime], Optional[timedelta]]:
        """
        calculate time and duration given scene X coordinate and width (from QRectF typically)
        Args:
            scene_x: x position relative to scene
            scene_w: width within scene

        Returns:
            (start time, time width)

        """
        return self._time_base + (self._time_unit * scene_x) if (scene_x is not None) else None, self._time_unit * scene_w

    def calc_scene_width(self, d: timedelta):
        return self._time_unit * d

    def calc_scene_rect(self, ztd: ztdtup=None, z:int=None, t: datetime=None, d: timedelta=None) -> [QRectF, None]:
        """
        calculate scene coordinates given time and Z
        Args:
            ztd: z,time,duration tuple
        Returns: QRectF relative to parent (typically either scene or timeline), None if no base or scale is known
        """
        if self._time_base is None or self._time_unit is None:
            LOG.warning("cannot calculate timeline rectangle without a base")
            return None
        if ztd is None:
            ztd = ztdtup(z, t, d)
        assert(ztd.z >= 0)
        atop = float(ztd.z) * self._track_height
        aheight = self._track_height
        aleft = (ztd.t - self._time_base) / self._time_unit
        awidth = ztd.d / self._time_unit
        return QRectF(aleft, atop, awidth, aheight)


