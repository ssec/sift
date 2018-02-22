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
import pickle as pkl
from typing import Tuple, Optional

from PyQt4.QtCore import QObject, QRectF, QByteArray, QPointF
from PyQt4.QtGui import QGraphicsSceneDragDropEvent

LOG = logging.getLogger(__name__)


class TimelineGraphicsConfig(object):
    track_height: float = 64.0
    track_corner_radius: float = 15.0
    track_left_pad: float = 128.0  # scene pixels space to left of first frame which we reserve for labels etc
    track_right_pad: float = 16.0  # scene pixels space to right of last frame we reserve for track closing etc
    track_title_pos: QPointF  = QPointF(3.0-128.0, -25.0)  # scene pixel offset relative to vertical centerline and start of first frame
    track_subtitle_pos: QPointF = QPointF(3.0-128.0, -13.0)
    frame_height: float = 56.0
    frame_corner_radius: float = 9.0
    frame_title_pos: QPointF = QPointF(2.0, 2.0)


# graphics constants in setting up items and painting
GFXC = TimelineGraphicsConfig()

# drag and drop mimetypes
MIMETYPE_TIMELINE_COLORMAP = 'application/sift.timeline.colormap'
MIMETYPE_TIMELINE_TRACK = 'application/sift.timeline.track'

mimed_track = namedtuple("mimed_track", ('uuid'))
mimed_colormap = namedtuple("mimed_colormap", ("pixmap"))

class TimelineFrameState(Enum):
    """Displayed state of frames, corresponds to a color or style.
    Typically combines information from Workspace and Scenegraph, does not correspond to selection.
    """
    UNKNOWN = 0
    AVAILABLE = 1  # can be imported from native resources but has not been
    ARRIVING = 2  # importer is bringing data into workspace
    CACHED = 3  # imported into workspace but not available as content in memory
    READY = 4  # available as a memory map, but may not be fully resident in RAM or VRAM
    ACTIVE = 5  # both ready to go into VRAM (or already resident), and participating in the application scene graph, possibly as part of an animation
    VISIBLE = 6  # active, and currently on-screen for user to view

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

    def __init__(self, time_base: datetime = None, time_unit: timedelta = timedelta(seconds=1), track_height=None):
        """

        Args:
            time_base: "zero" value of time; defaults to now UTC as this is somewhat arbitrary
            time_unit: time width corresponding to delta-X=1.0, defaults to 1.0s
            track_height: pixels high to display individual ttracks
        """
        super(TimelineCoordTransform, self).__init__()
        self._time_base = time_base or datetime.utcnow()
        self._time_unit = time_unit
        self._track_height = track_height or GFXC.track_height

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

    def calc_pixel_duration(self, d: timedelta):
        return d / self._time_unit

    def calc_pixel_x_pos(self, t: datetime, d: timedelta = None):
        x = (t - self._time_base) / self._time_unit
        if d is None:
            return x, None
        return x, d / self._time_unit

# utility functions allowing python objects (usually a dictionary in our case) to pass thru drag and drop serialization
def recv_mime(event: QGraphicsSceneDragDropEvent, mimetype: str):
    mime = event.mimeData()
    if not mime.hasFormat(mimetype):
        return None
    event.setAccepted(True)
    obj = pkl.loads(mime.data(mimetype).data())
    return obj

def send_mime(event: QGraphicsSceneDragDropEvent, mimetype: str, obj):
    qb = QByteArray(pkl.dumps(obj, protocol=pkl.HIGHEST_PROTOCOL))
    event.setData(mimetype, qb)
