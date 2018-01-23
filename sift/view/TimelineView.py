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

# ACTIONS to support
- drag a track up or down the z order
- pop a context menu for a track or a frame
- tool tips for frames or tracks
- change display state of frame, represented by color (see TimelineFrameState)
- allow one or more tracks to be selected 
- allow one or more frames to be selected
- scroll left and right to follow playback animation in background
- display time axis with actual dates and times, including click-to-place
- scroll vertically (when more tracks than can be shown in View)
- display movable and live-updated time cursor (playhead), including highlighting borders of frames under time cursor
- signal playhead movement to external agencies
- jump playhead to arbitrary time, optionally using left-right arrow keys
    + when track/s or frames selected, jump to next/last frame transition within the selection
    + when no tracks selected, consider all available frames (may require document help)
- change horizontal seconds-per-pixel (generally done with an external slider or mouse alt-drag on time scale)
- permit dragging of colorbars between layers
- permit dragging of colorbars off an external palette
- allow selection of tracks and frames using metadata matching
- allow circulation of z-order using up/down arrow keys
- allow sorting of tracks based on metadata characteristics
- future: nested tracks, e.g. for RGB or Algebraic

# CONCEPTS and VOCABULARY with respect to SIFT
A timeline Frame represents a Product in the Workspace
A timeline Track in the timeline represents a time series of related Products
The Scene represents a combination of the Metadatabase and (to a lesser extent) the active Document
Stepping into wider application-wide scope:
Products may or may not have Content cached in the workspace
ActiveContent in the workspace is being used to feed the SceneGraph by the SceneGraphManager
The Workspace has a Metadatabase of Resource, Product and Content metadata
The Document holds user intent, including EngineRules for producing Product Content
The Engine performs operations on the Workspace and its Metadatabase to maintain implicit Product Content

# REFERENCES
http://pyqt.sourceforge.net/Docs/PyQt4/qgraphicsscene.html
http://doc.qt.io/qt-4.8/qgraphicsview.html
http://doc.qt.io/qt-4.8/qgraphicsitemgroup.html
http://doc.qt.io/qt-5/qtwidgets-graphicsview-dragdroprobot-example.html
http://pyqt.sourceforge.net/Docs/PyQt4/qgraphicsobject.html

:author: R.K.Garcia <rkgarcia@wisc.edu>
:copyright: 2017 by University of Wisconsin Regents, see AUTHORS for more details
:license: GPLv3, see LICENSE for more details
"""
import os, sys
import logging, unittest
from uuid import UUID
from collections import namedtuple, OrderedDict
from typing import Tuple, Optional, Mapping, List, Any, Callable, Set
from enum import Enum
from abc import ABC, abstractproperty, abstractmethod
from datetime import datetime, timedelta
from PyQt4.QtCore import QObject, QRectF, Qt, pyqtSignal
from PyQt4.QtGui import *
from PyQt4.QtOpenGL import QGLWidget, QGLFormat, QGL

LOG = logging.getLogger(__name__)

# PATH_TEST_DATA = os.environ.get('TEST_DATA', os.path.expanduser("~/Data/test_files/thing.dat"))

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


class QTrackItem(QGraphicsObject):
    """ A group of Frames corresponding to a timeline
    This allows drag and drop of timelines to be easier
    """
    _scene = None
    _uuid : UUID = None
    _z: int = None
    _title: str = None
    _subtitle: str = None
    _icon: QIcon = None   # e.g. whether it's algebraic or RGB
    _metadata: Mapping = None  # arbitrary key-value store for selecting by metadata; in our case this often includes item family for seleciton
    _tooltip: str = None
    _color: QColor = None
    _selected: bool = False
    _colormap: [QGradient, QImage] = None
    _min: float = None
    _max: float = None
    _left_pad: timedelta = timedelta(hours=1)
    _right_pad: timedelta = timedelta(minutes=5)
    _gi_title : QGraphicsTextItem = None
    _gi_subtitle : QGraphicsTextItem = None
    _gi_icon : QGraphicsPixmapItem = None
    _gi_colormap : QGraphicsPixmapItem = None

    def __init__(self, scene, uuid: UUID, z: int,
                 title: str, subtitle: str = None, icon: QIcon = None, metadata: dict = None,
                 tooltip: str = None, color: QColor = None, selected: bool = False,
                 colormap: [QGradient, QImage] = None, min: float = None, max: float = None):
        super(QTrackItem, self).__init__()
        self._scene = scene
        self._uuid = uuid
        self._z = z
        self._title = title
        self._subtitle = subtitle
        self._icon = icon
        self._metadata = metadata or {}
        self._tooltip = tooltip
        self._color = color
        self._selected = selected
        self._colormap = colormap
        self._min, self._max = min, max
        pen, brush = scene.default_track_pen_brush
        if pen:
            LOG.debug('setting pen')
            self.setPen(pen)
        if brush:
            LOG.debug('setting brush')
            self.setBrush(brush)
        self._add_decorations()
        scene.addItem(self)
        self.setAcceptDrops(True)

    @property
    def uuid(self):
        return self._uuid

    def _add_decorations(self):
        """Add decor sub-items to self
        title, subtitle, icon, colormap
        """
        scene = self._scene
        if self._title:
            self._gi_title = it = scene.addSimpleText(self._title)
            it.setParentItem(self)
        if self._subtitle:
            self._gi_subtitle = it = scene.addSimpleText(self._subtitle)
            it.setParentItem(self)

    # painting and boundaries
    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: QWidget=None):
        super(QFrameItem, self).paint(painter, option, widget)

    def boundingRect(self) -> QRectF:
        return super(QFrameItem, self).boundingRect()

    # handle drag and drop
    def dragEnterEvent(self, event: QGraphicsSceneDragDropEvent):
        event.setAccepted(False)

    def dragLeaveEvent(self, event: QGraphicsSceneDragDropEvent):
        event.setAccepted(False)

    def dropEvent(self, event: QGraphicsSceneDragDropEvent):
        event.setAccepted(False)

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


class QFrameItem(QGraphicsObject):
    """A Frame
    For SIFT use, this corresponds to a single Product or single composite of multiple Products (e.g. RGB composite)
    QGraphicsView representation of a data frame, with a start and end time relative to the scene.
    Essentially a frame sprite
    """
    _state: TimelineFrameState = None
    _track: QTrackItem = None
    _uuid: UUID = None
    _start: datetime = None
    _duration: timedelta = None
    _title: str = None
    _subtitle: str = None
    _thumb: QPixmap = None
    _metadata: Mapping = None

    def __init__(self, track: QTrackItem, uuid: UUID,
                 start: datetime, duration: timedelta, state: TimelineFrameState,
                 title: str, subtitle: str = None, thumb: QPixmap = None,
                 metadata: Mapping[str, Any] = None):
        """create a frame representation and add it to a timeline
        Args:
            track: which timeline to add it to
            state: initial state
            start: frame start time
            duration: frame duration
            title: title of frame
            subtitle: subtitle (below title, optional)
            thumb: thumbnail image (via pillow), optional, may not be displayed if space is not available
            uuid: UUID of workspace representation
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
        self._metadata = metadata
        self._uuid = uuid
        pen, brush = track.default_frame_pen_brush
        if pen:
            LOG.debug('setting pen')
            self.setPen(pen)
        if brush:
            LOG.debug('setting brush')
            self.setBrush(brush)
        self.setParentItem(track)
        self.setAcceptDrops(True)

    @property
    def uuid(self):
        return self._uuid

    # painting and boundaries
    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: QWidget=None):
        painter.begin(self)

        painter.end()
        # super(QFrameItem, self).paint(painter, option, widget)

    def boundingRect(self) -> QRectF:
        return super(QFrameItem, self).boundingRect()

    # handle drag and drop
    def dragEnterEvent(self, event: QGraphicsSceneDragDropEvent):
        event.setAccepted(False)

    def dragLeaveEvent(self, event: QGraphicsSceneDragDropEvent):
        event.setAccepted(False)

    def dropEvent(self, event: QGraphicsSceneDragDropEvent):
        event.setAccepted(False)


class QTimeRulerItem(QGraphicsRectItem):
    """A ruler object showing the time dimension, an instance of which is at the top, bottom, or both ends of the Scene"""

    def __init__(self):
        super(QTimeRulerItem, self).__init__()


class QFramesInTracksScene(QGraphicsScene):
    """
    QGraphicsScene collecting QTimelineItems collecting QFrameItems.
    includes a TimelineCoordTransform time-to-X coordinate transform used for generating screen coordinates.

    """
    # coordinate transform between track Z order and time to scene float x,y,w,h
    _coords: TimelineCoordTransform = None

    # rulers at top and/or bottom of scene
    _top_ruler_item: QTimeRulerItem = None
    _bottom_ruler_item: QTimeRulerItem = None
    _ruler_tick_interval: timedelta = None

    # cursor information
    _cursor_time: datetime = None
    _cursor_duration: timedelta = None

    # content representing document / workspace / scenegraph
    _track_items: Mapping[UUID, QTrackItem] = None  # retain QTrackItem objects lest they disappear; also bookkeeping
    _frame_items: Mapping[UUID, QFrameItem] = None  # likewise for QFrameItems

    # styling settings
    _track_pen_brush = None, None
    _frame_pen_brush = None, None

    # _frame_states_allowed = None  # allow filtering of frames and tracks for thinned views

    def __init__(self):
        super(QFramesInTracksScene, self).__init__()
        self._coords = TimelineCoordTransform()
        self._track_items = {}
        self._frame_items = {}
        pen = QPen()
        pen.setWidthF(4.0)
        pen.setColor(Qt.black)
        # pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        brush = QBrush()
        brush.setColor(Qt.blue)
        brush.setStyle(Qt.SolidPattern)
        self._frame_pen_brush = pen, brush
        pen = QPen(pen)
        pen.setWidthF(2.0)
        pen.setColor(Qt.blue)
        brush = QBrush(brush)
        brush.setColor(Qt.gray)
        self._track_pen_brush = pen, brush
        # by default, show extent
        # self.setSceneRect(None)

    @property
    def coords(self) -> TimelineCoordTransform:
        return self._coords

    #
    # drawing and arranging QGraphicsItems
    #

    @property
    def default_track_pen_brush(self) -> Tuple[Optional[QPen], Optional[QBrush]]:
        return self._track_pen_brush

    @property
    def default_frame_pen_brush(self) -> Tuple[Optional[QPen], Optional[QBrush]]:
        return self._frame_pen_brush

    def drawBackground(self, painter: QPainter, invalidated_region: QRectF):
        super(QFramesInTracksScene, self).drawBackground(painter, invalidated_region)

    def _update_rulers_to_extents(self, tick_interval: timedelta=None):
        """Revise ruler size and internal tick items and labels to match scene extents"""

    def _update_cursor_in_rulers(self):
        """Update ruler contents to show current cursor position and duration
        """

    def _align_tracks_to_scene_rect(self, rect: QRectF, immediate_refresh: bool = True):
        """Move track labels to left edge """
        # FUTURE: okay, how do we deal with more than one view?
        pass

    def sceneRectChanged(self, new_rect: QRectF):
        self._align_tracks_to_scene_rect(new_rect, False)
        super(QFramesInTracksScene, self).sceneRectChanged(new_rect)

    #
    #
    #

    def visible_tracks_frames(self, view: QGraphicsView = None) -> Mapping[UUID, List[UUID]]:
        """return OrderedDict with UUID keys for tracks and list values of frames, for tracks and frames visible on in view
        """
        raise NotImplementedError("NYI")  # FIXME

    def visible_time_range(self, view: QGraphicsView = None):
        """return visible time range for the view in question
        """
        raise NotImplementedError("NYI")  # FIXME

    #
    # internal mid-level update commands
    #

    def add_track_frames(self, track: QTrackItem, *frames):
        """Add track and optional frames to scene storage, to ensure they are retained
        Asserts that if frame or track for the given UUID is already known, that the same item is being used
        """
        assert(isinstance(track, QTrackItem))
        if track.uuid in self._track_items:
            assert(track is self._track_items[track.uuid])
        else:
            self._track_items[track.uuid] = track
        for frame in frames:
            if frame.uuid in self._frame_items:
                assert(frame is self._frame_items[frame.uuid])
            else:
                self._frame_items[frame.uuid] = frame

    def _del_track(self, track):
        raise NotImplementedError("NYI")  # FIXME

    def _change_frame_state(self, frame: UUID, new_state: TimelineFrameState):
        """Change the displayed state of a frame and queue a visual refresh
        """
        raise NotImplementedError("NYI")  # FIXME

    def _change_track_state(self, track: UUID, new_state: TimelineTrackState):
        """Change the displayed state of a track and queue a visual refresh
        """
        raise NotImplementedError("NYI")  # FIXME

    def _move_cursor(self, when: datetime, duration: timedelta = None, animating_over: timedelta = None):
        """Move visible cursor to a new time
        optionally animating the transition over a time interval starting from now"""
        raise NotImplementedError("NYI")  # FIXME

    # def select_tracks_by_metadata(self, key, value):
    #     raise NotImplementedError("NYI")
    #
    # def select_frames_by_metadata(self, key, value):
    #     raise NotImplementedError("NYI")

    #
    # high-level signals from scene to hook to other parts of application
    #
    didSelectTracksAndFrames = pyqtSignal(dict)  # dictionary of track UUID to set of frame UUIDs, may be empty
    didChangeTrackOrder = pyqtSignal(list)  # list of track UUIDs from top to bottom
    didMoveCursorToTime = pyqtSignal(datetime, timedelta)  # instantaneous or time-range cursor move occurred
    didRequestStateChangeForFrame = pyqtSignal(UUID,
                                               TimelineFrameState)  # user requests a state change for a given frame
    didCopyColormapBetweenTracks = pyqtSignal(UUID, UUID)  # from-track and to-track
    didChangeVisibleAreaForView = pyqtSignal(QGraphicsView,
                                             datetime,
                                             timedelta)  # note: multiple views can share one scene

    #
    # delegate functions to implement for document and workspace
    # these are typically called by view or scene control logic, e.g. to decide menu to display or progress of a drag operation
    # FUTURE: decide if we actually need a delegate ABC to compose, rather than subclass overrides
    # for now simpler is better and Scene is already delegate/model-like so let's not over-indirect
    #

    def get(self, uuid: UUID) -> [QTrackItem, QFrameItem, None]:
        z = self._track_items.get(uuid, None)
        if z is not None:
            return z
        z = self._frame_items.get(uuid, None)
        return z

    def may_rearrange_track_z_order(self, track_uuid_list: List[UUID]) -> Optional[Callable[[bool], None]]:
        """Determine whether tracks can be rearranged and provide a commit/abort function if so
        Optionally: reflect any such changes on other parts of the application

        Args:
            track_uuid_list: new track UUID arrangement in top to bottom order

        Returns: None if rearrange is not permitted,
        else a callable that can be used to commit the change in the document
        callable(True) commits change; callable(False) aborts change
        Only one callable is valid at a time
        """
        LOG.warning("using base class may_rearrange_track_z_order which does nothing")
        return lambda commit: None

    def tracks_in_same_family(self, track: UUID) -> Set[UUID]:
        """inform the view on which tracks are closely related to the given track
        typically this is used to stylistically highlight related tracks during a drag operation
        """
        LOG.warning("using base class tracks_in_same_family which does nothing")
        return set()

    def may_reassign_color_map(self, from_track: UUID, to_track: UUID) -> Optional[Callable[[bool], None]]:
        """User is dragging a color map around, determine if drop is permitted and provide a commit/abort function if so
        """
        LOG.warning("using base class may_reassign_color_map which does nothing")
        return lambda b: None

    def menu_for_track(self, track_uuid: UUID, frame_uuid: UUID = None) -> Optional[QMenu]:
        """Generate QMenu to use as context menu for a given track, optionally with frame if mouse was over that frame"""
        LOG.warning("using base class menu_for_track which does nothing")
        return None

    def update(self, changed_track_uuids: [Set, None] = None, changed_frame_uuids: [Set, None] = None) -> int:
        """Populate or update scene, returning number of items changed in scene
        Does not add new items for tracks and frames already present
        Parameters serve only as hints
        """
        LOG.warning("using base class update which does nothing")
        return 0


class QFramesInTracksView(QGraphicsView):
    """Adds Track- and Frame-specific actions and signals to QGraphicsView"""

    def __init__(self, *args, **kwargs):
        super(QFramesInTracksView, self).__init__(*args, **kwargs)



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
        gfx = self._gfx = QFramesInTracksView(self)
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

class TestScene(QFramesInTracksScene):
    _did_populate = False

    def __init__(self):
        super(TestScene, self).__init__()

    def update(self, changed_track_uuids: [Set, None] = None, changed_frame_uuids: [Set, None] = None) -> int:
        if self._did_populate:
            return 0
        self._test_populate()

    def _test_populate(self):
        from uuid import uuid1 as uuidgen
        once = datetime.utcnow()
        tuuid = uuidgen()
        fuuid = uuidgen()
        min15 = timedelta(minutes=5)
        abitrack = QTrackItem(self, tuuid, 0, "G21 ABI B99 BT", "test track", tooltip="peremptorily cromulent")
        # scene.addItem(abitrack)  # done in init
        frame1 = QFrameItem(abitrack, fuuid, once + min15, min15, TimelineFrameState.AVAILABLE, "frame1", "fulldiskimus")
        abitrack.update_time_range_from_children()
        # scene.addItem(frame1)  # done in init
        blabla = QGraphicsTextItem('abcdcba')
        font = QFont('White Rabbit')
        blabla.setFont(font)
        blabla.setPos(140, 100)
        self.addItem(blabla)
        self.content = [abitrack, frame1, blabla]


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
