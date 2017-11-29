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
from typing import Tuple, Optional
from enum import Enum
from datetime import datetime, timedelta
from PyQt4.QtCore import QObject, QRectF, Qt
from PyQt4.QtGui import *
from PyQt4.QtOpenGL import QGLWidget, QGLFormat, QGL

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


class QFramesInTracksScene(QGraphicsScene):
    """
    QGraphicsScene collecting QTimelineItems collecting QFrameItems.
    includes a TimelineCoordTransform time-to-X coordinate transform used for generating screen coordinates.
    """
    _coords: TimelineCoordTransform = None

    def __init__(self):
        super(QFramesInTracksScene, self).__init__()
        self._coords = TimelineCoordTransform()

    @property
    def coords(self):
        return self._coords

    @property
    def default_track_pen_brush(self) -> Tuple[Optional[QPen], Optional[QBrush]]:
        return None, None

    @property
    def default_frame_pen_brush(self) -> Tuple[Optional[QPen], Optional[QBrush]]:
        return None, None




class QTrackItem(QGraphicsRectItem):
    """
    A Group of Frames corresponding to a timeline
    This allows drag and drop of timelines to be easier

    """
    _scene: QFramesInTracksScene = None
    _z: int = None
    _title: str = None
    _subtitle: str = None
    _icon: QIcon = None   # e.g. whether it's algebraic or RGB
    _tooltip: str = None
    _color: QColor = None
    _selected: bool = False
    _colormap: [QGradient, QImage] = None
    _min: float = None
    _max: float = None
    _left_pad: timedelta = timedelta(hours=1)
    _right_pad: timedelta = timedelta(minutes=5)

    def __init__(self, scene: QFramesInTracksScene, z: int,
                 title: str, subtitle: str = None, icon: QIcon = None,
                 tooltip: str = None, color: QColor = None, selected: bool = False,
                 colormap: [QGradient, QImage] = None, min: float = None, max: float = None):
        super(QTrackItem, self).__init__()
        self._scene = scene
        self._z = z
        self._title = title
        self._subtitle = subtitle
        self._icon = icon
        self._tooltip = tooltip
        self._color = color
        self._selected = selected
        self._colormap = colormap
        self._min, self._max = min, max
        pen, brush = scene.default_track_pen_brush
        if pen:
            self.setPen(pen)
        if brush:
            self.setBrush(brush)
        scene.addItem(self)

    @property
    def default_frame_pen_brush(self):
        return self._scene.default_frame_pen_brush

    def update_time_range_from_children(self):
        extent = self.childrenBoundingRect()
        LOG.debug("track bounds left={} top={} width={} height={}".format(extent.left(), extent.top(), extent.width(), extent.height()))
        tx, dx = extent.left(), extent.width()
        t, d = self._scene.coords.calc_time_duration(tx, dx)
        rect = self._scene.coords.calc_scene_rect(z=self._z, t=t - self._left_pad, d=d + self._left_pad + self._right_pad)
        self.setRect(rect)


    def calc_frame_rect_in_track(self, ztd: ztdtup=None, z:int=0, t: datetime=None, d: timedelta=None):
        """
        calculate frame rect within scene
        Returns: QRectF
        """
        if ztd is None:
            ztd = ztdtup(z, t, d)
        return self._scene.coords.calc_scene_rect(ztd)


class QFrameItem(QGraphicsRectItem):
    """
    QGraphicsView representation of a data frame, with a start and end time relative to the scene.
    Essentially a frame sprite
    """
    _state: TimelineFrameState = None
    _track: QTrackItem = None
    # some frames are in-document, some are in-workspace, some are both
    _ws_uuid: UUID = None
    _doc_uuid: UUID = None
    _start: datetime = None
    _duration: timedelta = None
    _title: str = None
    _subtitle: str = None
    _thumb: QImage = None

    def __init__(self, track: QTrackItem,
                 state: TimelineFrameState, start: datetime, duration: timedelta,
                 title: str, subtitle: str=None, thumb=None,
                 workspace_uuid=None, document_uuid=None):
        """
        create a frame representation and add it to a timeline
        Args:
            track: which timeline to add it to
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
        rect = track.calc_frame_rect_in_track(t=start, d=duration)
        if rect is not None:
            self.setRect(rect)
        self._state = state
        self._start = start
        self._duration = duration
        self._title = title
        self._subtitle = subtitle
        self._thumb = thumb
        self._ws_uuid = workspace_uuid
        self._doc_uuid = document_uuid
        pen, brush = track.default_frame_pen_brush
        if pen:
            self.setPen(pen)
        if brush:
            self.setBrush(brush)
        self.setParentItem(track)


class TestWindow(QMainWindow):
    _scene = None
    _gfx = None

    def __init__(self, scene: QFramesInTracksScene, *args, **kwargs):
        super(TestWindow, self).__init__(*args, **kwargs)
        # self.windowTitleChanged.connect(self.onWindowTitleChange)
        self.setWindowTitle("timeline unit test")

        # toolbar = QToolBar("och")
        # toolbar.setIconSize(QSize(20,20))
        # self.addToolBar(toolbar)

        # button_action = QAction(QIcon("balance.png"), "ochtuse", self)
        # button_action.setStatusTip("och, just do something")
        # button_action.triggered.connect(self.onMyToolBarButtonClick)
        # button_action.setCheckable(True)
        # # button_action.setShortcut(QKeySequence("Ctrl+p"))
        # # button_action.setShortcut(QKeySequence(Qt.CTRL + Qt.Key_P))
        # button_action.setShortcut(QKeySequence.Print)
        # toolbar.addAction(button_action)
        # toolbar.addWidget(QLabel("OCH"))
        # toolbar.addWidget(QCheckBox())

        self.setStatusBar(QStatusBar(self))

        menu = self.menuBar()

        file_menu = menu.addMenu("&File")
        # file_menu.addAction(button_action)
        # file_menu.addSeparator()
        file_menu.addMenu("Do not push")
#        file_menu.addAction()

        self._scene = scene
        gfx = self._gfx = QGraphicsView(self)
        # label = QLabel("och!")
        # label.setAlignment(Qt.AlignCenter)

        # ref https://doc.qt.io/archives/qq/qq26-openglcanvas.html
        self.setCentralWidget(gfx)
        fmt = QGLFormat(QGL.SampleBuffers)
        wdgt = QGLWidget(fmt)
        assert(wdgt.isValid())
        gfx.setViewport(wdgt)
        gfx.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        scene.setSceneRect(QRectF(0,0, 800, 600))
        gfx.setScene(scene)

        # populate fills the scene with interesting stuff.
        # self.populate()

        # Make it bigger
        # self.setWindowState(Qt.WindowMaximized)

        # Well... it's going to have an animation, ok?

        # So, I set a timer to 1 second
        # self.animator=QTimer()

        # And when it triggers, it calls the animate method
        # self.animator.timeout.connect(self.animate)

        # And I animate it once manually.
        # self.animate()


def _test_populate(scene: QFramesInTracksScene):
    """
    test populate a scene
    Args:
        scene: scene to populate

    Returns: collection of items placed in scene, such that they can remain retained (else they drop out of scene)
    """
    from uuid import uuid1 as uuidgen
    once = datetime.utcnow()
    wuuid = uuidgen()
    min15 = timedelta(minutes=5)
    abitrack = QTrackItem(scene, 0, "G21 ABI B99 BT", "test track", None, "peremptorily cromulent", None)
    # scene.addItem(abitrack)  # done in init
    frame1 = QFrameItem(abitrack, TimelineFrameState.AVAILABLE, once+min15, min15, "frame1", "fulldiskimus", workspace_uuid=wuuid)
    abitrack.update_time_range_from_children()
    # scene.addItem(frame1)  # done in init
    blabla = QGraphicsTextItem('abcdcba')
    font = QFont('White Rabbit')
    blabla.setFont(font)
    blabla.setPos(140, 100)
    scene.addItem(blabla)
    return [abitrack, frame1, blabla]


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
    logging.basicConfig(level=logging.DEBUG)

    app = QApplication(sys.argv)

    scene = QFramesInTracksScene()
    stuff_not_to_lose = _test_populate(scene)
    window = TestWindow(scene)
    window.show()
    window.setFocus()

    app.exec_()

    return 0


if __name__ == '__main__':
    sys.exit(main())
