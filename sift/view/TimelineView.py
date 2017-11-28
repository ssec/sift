#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# PURPOSE
Timeline View using QGraphicsView and its kin
Assume X coordinate corresponds to seconds, apply transforms as needed

# FUNCTION
- display a Scene of Timelines of Frames
- View is scrollable in time (screen X) and level (screen Y), compressible in time
- Timelines have logical Z order, with 0 being topmost and +n being bottom; top to bottom corresponds to screen position
- Timelines can be dragged and dropped to change their Z order, similar to layer list but including multiple Frames
- Frames represent individual visible dataset/image
- Frames may exist in the application only as metadata and can be in multiple states depending on user direction:
-

REFERENCES
http://pyqt.sourceforge.net/Docs/PyQt4/qgraphicsscene.html#details

REQUIRES

:author: R.K.Garcia <rkgarcia@wisc.edu>
:copyright: 2017 by University of Wisconsin Regents, see AUTHORS for more details
:license: GPLv3, see LICENSE for more details
"""
import os, sys
import logging, unittest
from uuid import UUID
from collections import namedtuple
from enum import Enum
from datetime import datetime, timedelta
from PyQt4.QtCore import QObject, QRectF
from PyQt4.QtGui import *

LOG = logging.getLogger(__name__)

# PATH_TEST_DATA = os.environ.get('TEST_DATA', os.path.expanduser("~/Data/test_files/thing.dat"))

class TimelineFrameState(Enum):
    UNKNOWN = 0
    AVAILABLE = 1  # can be imported from native resources but has not been
    IMPORTED = 2  # imported into workspace
    READY = 3  # available as a memory map, but may not be fully resident in RAM or VRAM
    ACTIVE = 4  # both ready to go into VRAM (or already resident), and participating in the application scene graph, possibly as part of an animation
    VISIBLE = 5  # active, and currently on-screen for user to view


DEFAULT_TIMELINE_HEIGHT = 48

# z level > 0 ascending downward, start time: datetime, duration: timedelta
ztdtup = namedtuple("ztdtup", ('z', 't', 'd'))

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
    _base: datetime = None
    _per_pixel: timedelta = None
    _timeline_height: float = None

    def __init__(self, time_base: datetime = None, pixel_time_width: timedelta = timedelta(seconds=1), timeline_height=DEFAULT_TIMELINE_HEIGHT):
        super(TimelineCoordTransform, self).__init__()
        self._base = time_base
        self._per_pixel = pixel_time_width
        self._timeline_height = timeline_height



    def calc_rect(self, ztd: ztdtup=None, z:int=None, t: datetime=None, d: timedelta=None) -> [QRectF, None]:
        """
        calculate scene coordinates given time and Z
        Args:
            ztd: z,time,duration tuple
        Returns: QRectF relative to parent (typically either scene or timeline), None if no base or scale is known
        """
        if self._base is None or self._per_pixel is None:
            LOG.warning("cannot calculate timeline rectangle without a base")
            return None
        if ztd is None:
            ztd = ztdtup(z, t, d)
        assert(ztd.z >= 0)
        atop = float(ztd.z) * self._timeline_height
        aheight = self._timeline_height
        aleft = (ztd.t - self._base) / self._per_pixel
        awidth = ztd.d / self._per_pixel
        return QRectF(aleft, atop, awidth, aheight)


class QFrameItem(QGraphicsRectItem):
    """
    QGraphicsView representation of a data frame, with a start and end time relative to the scene.
    Essentially a frame sprite
    """
    _state: TimelineFrameState = None
    # some frames are in-document, some are in-workspace, some are both
    _ws_uuid: UUID = None
    _doc_uuid: UUID = None
    _start: datetime = None
    _duration: timedelta = None
    _title: str = None
    _subtitle: str = None

    def __init__(self, timeline: QTimelineItem,
                 state: TimelineFrameState, start: datetime, duration: timedelta,
                 title: str, subtitle: str=None, thumb=None,
                 workspace_uuid=None, document_uuid=None):
        """
        create a frame representation and add it to a timeline
        Args:
            timeline: which timeline to add it to
            state: initial state
            start: frame start time
            duration: frame duration
            title: title of frame
            subtitle: subtitle (below title, optional)
            thumb: thumbnail image (via pillow), optional, may not be displayed if space is not available
            workspace_uuid: UUID of workspace representation (backed by content or simply metadata); may be None
            document_uuid: UUID of document representation (backed by composition/presentation of other content); may be None
        """
        super(QFrameItem, self).__init__()
        self.setParentItem(timeline)
        rect = timeline.calc_frame_rect_in_timeline(t=start, d=duration)
        if rect is not None:
            self.setRect(rect)
        # FIXME: set painter to make curved outline corners


class QTimelineItem(QGraphicsRectItem):
    """
    A Group of Frames corresponding to a timeline
    This allows drag and drop of timelines to be easier

    """
    # ref: http://doc.qt.io/qt-4.8/qgraphicsitemgroup.html
    _scene: QFramesInTimelinesScene = None
    _z: int = None
    _title: str = None
    _subtitle: str = None
    _icon: QImage = None   # e.g. whether it's algebraic or RGB
    _tooltip: str = None
    _color: QColor = None
    _selected: bool = False
    _colormap: [QGradient, QImage] = None
    _min: float = None
    _max: float = None

    def calc_frame_rect_in_timeline(self, ztd: ztdtup=None, z:int=0, t: datetime=None, d: timedelta=None):
        """
        calculate frame rect within scene
        Returns: QRectF
        """
        if ztd is None:
            ztd = ztdtup(z, t, d)
        return self._scene.calc_rect(z=ztd.z, t=ztd.t, d=ztd.d)

    def calc_timeline_rect_in_scene(self, ztd: ztdtup=None, z:int=0, t: datetime=None, d: timedelta=None):
        """

        """



class QFramesInTimelinesScene(QGraphicsScene):
    """
    QGraphicsScene collecting QTimelineItems collecting QFrameItems.
    includes a TimelineCoordTransform time-to-X coordinate transform used for generating screen coordinates.
    """
    _coords: TimelineCoordTransform = None

    def __init__(self, test_pattern=False):
        super(QFramesInTimelinesScene, self).__init__()
        self._coords = TimelineCoordTransform()

    @property
    def coords(self):
        return self._coords

    def calc_frame_rect(self, ztd: ztdtup=None, z:int=0, t: datetime=None, d: timedelta=None):
        raise NotImplementedError("FIXME") # FIXME



class tests(unittest.TestCase):
    def setUp(self):
        pass

    def test_something(self):
        pass


def _debug(type, value, tb):
    "enable with sys.excepthook = debug"
    if not sys.stdin.isatty():
        sys.__excepthook__(type, value, tb)
    else:
        import traceback, pdb
        traceback.print_exception(type, value, tb)
        # …then start the debugger in post-mortem mode.
        pdb.post_mortem(tb)  # more “modern”


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="PURPOSE",
        epilog="",
        fromfile_prefix_chars='@')
    parser.add_argument('-v', '--verbose', dest='verbosity', action="count", default=0,
                        help='each occurrence increases verbosity 1 level through ERROR-WARNING-INFO-DEBUG')
    parser.add_argument('-d', '--debug', dest='debug', action='store_true',
                        help="enable interactive PDB debugger on exception")
    parser.add_argument('inputs', nargs='*',
                        help="input files to process")
    args = parser.parse_args()

    levels = [logging.ERROR, logging.WARN, logging.INFO, logging.DEBUG]
    logging.basicConfig(level=levels[min(3, args.verbosity)])

    if args.debug:
        sys.excepthook = _debug

    if not args.inputs:
        unittest.main()
        return 0

    for pn in args.inputs:
        pass

    return 0


if __name__ == '__main__':
    sys.exit(main())
