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
http://doc.qt.io/archives/qt-4.8/examples-graphicsview.html
http://doc.qt.io/archives/qt-4.8/qt-graphicsview-diagramscene-example.html
http://pyqt.sourceforge.net/Docs/PyQt4/qgraphicsscene.html
http://doc.qt.io/qt-4.8/qgraphicsview.html
http://doc.qt.io/qt-4.8/qgraphicsitemgroup.html
http://pyqt.sourceforge.net/Docs/PyQt4/qgraphicsitem.html
http://doc.qt.io/qt-5/qtwidgets-graphicsview-dragdroprobot-example.html
http://pyqt.sourceforge.net/Docs/PyQt4/qgraphicsobject.html

:author: R.K.Garcia <rkgarcia@wisc.edu>
:copyright: 2017 by University of Wisconsin Regents, see AUTHORS for more details
:license: GPLv3, see LICENSE for more details
"""
import sys
import logging, unittest
from uuid import UUID
from typing import Mapping, Any
from datetime import datetime, timedelta
from PyQt4.QtCore import QRectF
from PyQt4.QtGui import *
from PyQt4.QtOpenGL import QGLWidget, QGLFormat, QGL

from sift.view.TimelineCommon import TimelineFrameState, ztdtup

LOG = logging.getLogger(__name__)

# PATH_TEST_DATA = os.environ.get('TEST_DATA', os.path.expanduser("~/Data/test_files/thing.dat"))

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
        # pen, brush = scene.default_track_pen_brush
        # if pen:
        #     LOG.debug('setting pen')
        #     self.setPen(pen)
        # if brush:
        #     LOG.debug('setting brush')
        #     self.setBrush(brush)
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
        super(QTrackItem, self).paint(painter, option, widget)

    def boundingRect(self) -> QRectF:
        return super(QTrackItem, self).boundingRect()

    # handle clicking
    def mousePressEvent(self, event: QGraphicsSceneMouseEvent):
        super(QTrackItem, self).mousePressEvent()

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent):
        super(QTrackItem, self).mouseReleaseEvent()

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


    def calc_frame_rect_in_track(self, ztd: ztdtup =None, z:int=0, t: datetime=None, d: timedelta=None):
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
        # pen, brush = track.default_frame_pen_brush
        # if pen:
        #     LOG.debug('setting pen')
        #     self.setPen(pen)
        # if brush:
        #     LOG.debug('setting brush')
        #     self.setBrush(brush)
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

    # handle clicking
    def mousePressEvent(self, event: QGraphicsSceneMouseEvent):
        super(QFrameItem, self).mousePressEvent()

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent):
        super(QFrameItem, self).mouseReleaseEvent()


class QTimeRulerItem(QGraphicsRectItem):
    """A ruler object showing the time dimension, an instance of which is at the top, bottom, or both ends of the Scene"""

    def __init__(self):
        super(QTimeRulerItem, self).__init__()


class QFramesInTracksView(QGraphicsView):
    """Adds Track- and Frame-specific actions and signals to QGraphicsView"""

    def __init__(self, *args, **kwargs):
        super(QFramesInTracksView, self).__init__(*args, **kwargs)



class TestWindow(QMainWindow):
    _scene = None
    _gfx = None

    def __init__(self, scene, *args, **kwargs):
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
    from sift.view.TimelineScene import TestScene
    logging.basicConfig(level=logging.DEBUG)

    app = QApplication(sys.argv)

    scene = TestScene()
    scene._test_populate()
    window = TestWindow(scene)
    window.show()
    window.setFocus()

    app.exec_()

    return 0


if __name__ == '__main__':
    sys.exit(main())
