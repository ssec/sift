#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:author: R.K.Garcia <rkgarcia@wisc.edu>
:copyright: 2018 by University of Wisconsin Regents, see AUTHORS for more details
:license: GPLv3, see LICENSE for more details
"""

import logging
import pickle as pkl
from datetime import datetime, timedelta
from enum import Enum
from typing import Tuple, Optional, NamedTuple, Any

from PyQt5.QtCore import QObject, QRectF, QByteArray, QPointF
from PyQt5.QtWidgets import QGraphicsSceneDragDropEvent

LOG = logging.getLogger(__name__)


# @dataclass
class GraphicsConfig(object):
    track_height: float = 64.0
    track_corner_radius1: float = 4.0
    track_corner_radius2: float = 4.0
    track_left_pad: float = 300.0  # scene pixels space to left of first frame which we reserve for labels etc
    track_right_pad: float = 16.0  # scene pixels space to right of last frame we reserve for track closing etc
    # scene pixel offset relative to vertical centerline and start of first frame
    track_title_pos: QPointF = QPointF(3.0 - 300.0, -25.0)
    track_subtitle_pos: QPointF = QPointF(3.0 - 300.0, -10.0)
    frame_height: float = 56.0
    frame_corner_radius: float = 6.0
    frame_title_pos: QPointF = QPointF(2.0, -13.0)
    frame_subtitle_pos: QPointF = QPointF(2.0, 2.0)


# graphics constants in setting up items and painting
GFXC = GraphicsConfig()

# drag and drop mimetypes
MIMETYPE_TIMELINE_COLORMAP = 'application/uwsift.timeline.colormap'
MIMETYPE_TIMELINE_TRACK = 'application/uwsift.timeline.track'


class mimed_track(NamedTuple):
    track: str


class mimed_colormap(NamedTuple):
    colormap: Any


class ztdtup(NamedTuple):
    z: int
    t: datetime
    d: timedelta


class VisualState(Enum):
    """Visual states of frames and tracks, corresponds to a color or style.
    They're collected in Flags() objects which behaves like a set()
    Derived from Document / Workspace / SceneGraph states.
    Typically combines information from Workspace and Scenegraph, does not correspond to selection.
    by default, display greyed-out potential frame
    """
    ERROR = 1  # error red
    WARNING = 2  # warning yellow

    DISABLED = 4  # greyed-out and muted, may not even display decoration; signifies user turned it off
    BUSY = 8  # when WorkspaceState.ARRIVING, show barberpole animation or at least diagonal hatch
    AVAILABLE = 16  # done arriving, now resides in cache but is not necessarily loaded
    READY = 32  # regular coloration - typically when WorkspaceState.CACHED
    HIGHLIGHT = 64  # frame is being shown on the screen at this moment in time

    LOCKED = 128  # indicate the frame is locked across is entire track
    RELEVANT = 256  # highlight that this (typically track) may be affected with the operation you have inflight
    SELECTED = 512  # track or frame is selected by user


class CoordTransform(QObject):
    """Configurable transformations between time and display space.

    Typically a tool used by scene/timeline/frame; typically one instance per scene!
    default is 1px=1s in X
    Z order starts at 0 (topmost timeline) with +1 being beneath 0, etc (lower numbers overlay higher;
    this is inverse from scenegraph)
    typically QTimelineItem will have a nonzero Z determining its y coordinate in the scene
    QTimelineItem contains zero or more QFrameItems
    QFrameItems coordinates are relative to their QTimelineItem parent and (for now) are z=0
    When we have expandable RGB timelines this will change

    """
    _time_base: datetime = None
    _time_unit: timedelta = None
    _track_height: float = None
    _max_z: int = 0  # the highest z value of the active tracks, may change. typically z>=0 for active tracks

    @property
    def max_z(self) -> int:
        return self._max_z

    @max_z.setter
    def max_z(self, mz: int):
        self._max_z = mz

    @property
    def settings(self) -> ztdtup:
        return ztdtup(self._max_z, self._time_base, self._time_unit)

    @settings.setter
    def settings(self, new_settings: ztdtup):
        self._max_z = new_settings.z
        self._time_base = new_settings.t
        self._time_unit = new_settings.d

    def __init__(self, time_base: datetime = None, time_unit: timedelta = None, track_height=None):
        """

        Args:
            time_base: "zero" value of time; defaults to now UTC as this is somewhat arbitrary
            time_unit: time width corresponding to delta-X=1.0, defaults to 1.0s
            track_height: pixels high to display individual ttracks
        """
        if time_unit is None:
            time_unit = timedelta(seconds=1)
        super(CoordTransform, self).__init__()
        self._time_base = time_base or datetime.utcnow()
        self._time_unit = time_unit
        self._track_height = track_height or GFXC.track_height

    def calc_time_duration(self, scene_x: Optional[float], scene_w: Optional[float]
                           ) -> Tuple[Optional[datetime], Optional[timedelta]]:
        """Calculate time and duration given scene X coordinate and width (from QRectF typically)

        Args:
            scene_x: x position relative to scene
            scene_w: width within scene

        Returns:
            (start time, time width)

        """
        return self._time_base + (self._time_unit * scene_x) if (
            scene_x is not None) else None, self._time_unit * scene_w

    def calc_scene_width(self, d: timedelta):
        return self._time_unit * d

    def calc_track_pixel_y_center(self, z: int):
        """normalize based on max_z such that max_z..-bignum becomes 0..bignum going downward on the screen
        """
        return self._track_height * (0.5 + float(self._max_z - z))

    def calc_scene_rect(self, ztd: ztdtup = None, z: int = None, t: datetime = None, d: timedelta = None
                        ) -> Optional[QRectF]:
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
        atop = float(self._max_z - ztd.z) * self._track_height
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
