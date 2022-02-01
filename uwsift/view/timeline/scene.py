#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PURPOSE

REFERENCES

REQUIRES

:author: R.K.Garcia <rkgarcia@wisc.edu>
:copyright: 2017 by University of Wisconsin Regents, see AUTHORS for more details
:license: GPLv3, see LICENSE for more details
"""
import logging
import sys
import unittest
from datetime import datetime, timedelta
from typing import Tuple, Optional, Mapping, List, Callable, Set, Iterable, Sequence, Any, Union
from uuid import UUID

from PyQt5.QtCore import QRectF, Qt, pyqtSignal
from PyQt5.QtGui import QPen, QBrush, QPainter
from PyQt5.QtWidgets import (QGraphicsScene, QGraphicsView,
                             QMenu, QMainWindow, QStatusBar, QApplication,
                             QGraphicsItem)
from PyQt5.QtOpenGL import QGLFormat, QGL, QGLWidget

from uwsift.common import Flags, Span
from .common import VisualState, CoordTransform
from .items import QTrackItem, QFrameItem, QTimeRulerItem

LOG = logging.getLogger(__name__)


class QFramesInTracksScene(QGraphicsScene):
    """
    QGraphicsScene collecting QTimelineItems collecting QFrameItems.
    includes a TimelineCoordTransform time-to-X coordinate transform used for generating screen coordinates.

    This is a "generic" Timeline scene which should be subclassed to a form that uses your specific model.
    """

    # whether to show tracks with zorder < 0, i.e. tracks not immediately part of document
    # many users will prefer to show only active tracks on the document,
    # only showing everything when they want to drag or otherwise activate potential tracks waiting in the wings
    _active_only: bool = False

    # coordinate transform between track Z order and time to scene float x,y,w,h
    _coords: CoordTransform = None

    # rulers at top and/or bottom of scene; QGraphicsItems have to be retained in python to avoid confusing Qt
    _top_ruler_item: QTimeRulerItem = None
    _bottom_ruler_item: QTimeRulerItem = None
    _ruler_tick_interval: timedelta = None

    # cursor information
    _cursor_time: datetime = None
    _cursor_duration: timedelta = None

    # content representing document / workspace / scenegraph
    _track_items: Mapping[str, QTrackItem] = None  # retain QTrackItem objects lest they disappear; also bookkeeping
    _frame_items: Mapping[UUID, QFrameItem] = None  # likewise for QFrameItems
    _decor_items: Mapping[Any, QGraphicsItem] = None  # decoration items

    # styling settings
    _track_pen_brush = None, None
    _frame_pen_brush = None, None

    # _frame_states_allowed = None  # allow filtering of frames and tracks for thinned views

    def __init__(self):
        super(QFramesInTracksScene, self).__init__()
        self._coords = CoordTransform()
        self._track_items = {}
        self._frame_items = {}
        pen = QPen()
        pen.setWidthF(1.25)
        pen.setColor(Qt.black)
        # pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        brush = QBrush()
        brush.setColor(Qt.blue)
        brush.setStyle(Qt.SolidPattern)
        self._frame_pen_brush = pen, brush
        pen = QPen(pen)
        pen.setWidthF(3.5)
        pen.setJoinStyle(Qt.RoundJoin)
        pen.setColor(Qt.blue)
        brush = QBrush(brush)
        brush.setColor(Qt.gray)
        self._track_pen_brush = pen, brush
        # by default, show extent
        # self.setSceneRect(None)

    def clear(self):
        LOG.debug("clearing out Timeline Scene")
        super(QFramesInTracksScene, self).clear()
        super(QFramesInTracksScene, self).update()
        self._track_items = {}
        self._frame_items = {}
        self._decor_items = {}

    @property
    def coords(self) -> CoordTransform:
        return self._coords

    def add_track(self, track: QTrackItem):
        """Called by QTrackItem constructor
        We need to maintain references to Q*Items, Qt will not do it for us
        """
        if track.track in self._track_items:
            LOG.error("track {} was already present in scene".format(track.track))
        self._track_items[track.track] = track
        self.addItem(track)
        self._verify_z_contiguity()

    def add_frame(self, frame: QFrameItem):
        """Called by QTrackItem at QFrameItem's constructor-time
        We need to maintain references to Q*Items, Qt will not do it for us
        """
        if frame.uuid in self._frame_items:
            LOG.error("frame {} was already present in scene".format(frame.uuid))
        self._frame_items[frame.uuid] = frame

    #
    # drawing and arranging QGraphicsItems
    #

    @property
    def default_track_pen_brush(self) -> Tuple[Optional[QPen], Optional[QBrush]]:
        return self._track_pen_brush

    @property
    def default_frame_pen_brush(self) -> Tuple[Optional[QPen], Optional[QBrush]]:
        return self._frame_pen_brush

    @property
    def timeline_span(self) -> Span:
        if hasattr(self, '_demo_time_span'):
            return self._demo_time_span
        t = datetime.utcnow()
        hh = timedelta(minutes=30)
        dts = self._demo_time_span = Span(t - hh, hh * 2)
        LOG.warning("using demo timespan on document")
        return dts

    @property
    def playback_span(self) -> Span:
        return self.timeline_span

    def drawBackground(self, painter: QPainter, invalidated_region: QRectF):
        super(QFramesInTracksScene, self).drawBackground(painter, invalidated_region)

    def _update_rulers_to_extents(self, tick_interval: timedelta = None):
        """Revise ruler size and internal tick items and labels to match scene extents"""

    def _update_cursor_in_rulers(self):
        """Update ruler contents to show current cursor position and duration
        """

    def _align_tracks_to_scene_rect(self, rect: QRectF, immediate_refresh: bool = True):
        """Move track labels to left edge """
        # FUTURE: okay, how do we deal with more than one view?
        pass

    def sceneRectChanged(self, new_rect: QRectF):
        """Update tracks when scene rect changes."""
        self._align_tracks_to_scene_rect(new_rect, False)
        super(QFramesInTracksScene, self).sceneRectChanged(new_rect)

    def visible_tracks_frames(self, view: QGraphicsView = None) -> Mapping[UUID, List[UUID]]:
        """return OrderedDict with UUID keys for tracks and list values of frames, for tracks and frames visible on in view
        """
        raise NotImplementedError("NYI")  # FIXME

    def visible_time_range(self, view: QGraphicsView = None):
        """return visible time range for the view in question
        """
        raise NotImplementedError("NYI")  # FIXME

    def center_view_on_frame(self, gv: QGraphicsView, frame_uuid: UUID):
        item = self._frame_items.get(frame_uuid)
        if item is not None:
            gv.centerOn(item)

    #
    # internal mid-level update commands
    #

    # def add_frames_to_track(self, track: [QTrackItem, str], paranoid=True, *frames):
    #     """Add track and optional frames to scene storage, to ensure they are retained
    #     Asserts that if frame or track for the given UUID is already known, that the same item is being used
    #     """
    #     if isinstance(track, str):
    #         track = self._track_items[track]
    #     if paranoid:
    #         assert(isinstance(track, QTrackItem))
    #         if track.track in self._track_items:
    #             assert(track is self._track_items[track.track])
    #         else:
    #             raise AssertionError("need to insert_track_with_zorder before adding frames")
    #             # self._track_items[track.uuid] = track
    #     for frame in frames:
    #         assert(isinstance(frame, QFrameItem))
    #         existing = self._frame_items.get(frame.uuid)
    #         if paranoid:
    #             assert((existing is None) or (existing is frame))
    #         if existing is None:
    #             self._frame_items[frame.uuid] = frame

    def _del_track(self, track):
        raise NotImplementedError("NYI")  # FIXME

    def _change_frame_state(self, frame: UUID, new_state: Flags):
        """Change the displayed state of a frame and queue a visual refresh
        """
        raise NotImplementedError("NYI")  # FIXME

    def _change_track_state(self, track: str, new_state: Flags):
        """Change the displayed state of a track and queue a visual refresh
        """
        raise NotImplementedError("NYI")  # FIXME

    def _move_cursor(self, when: datetime, duration: timedelta = None, animating_over: timedelta = None):
        """Move visible cursor to a new time
        optionally animating the transition over a time interval starting from now"""
        raise NotImplementedError("NYI")  # FIXME

    #
    # track management
    #

    # @property
    # def _current_tracks_order(self):
    #     L = dict((trk.z, trk) for trk in self._track_items.values())
    #     n = max(L.keys())
    #     if len(L) != n:
    #         raise AssertionError("inconsistency in z-order of tracks; expected {} but have {}".format(n, len(L)))
    #     for q in range(n):
    #         yield L[q]  # raises if we're missing a track
    #
    # @property
    # def _current_tracks_uuid_order(self):
    #     for trk in self._current_tracks_order:
    #         yield trk.uuid

    # FUTURE: look at handling z-order hijinks with a uwsift.common.ZList and removing .z from QTrackItem

    @property
    def _track_order(self) -> Mapping[int, QTrackItem]:
        return dict((q.z, q) for q in self._track_items.values())

    @property
    def _track_count(self) -> int:
        return len(self._track_items)

    @property
    def _track_max_z(self) -> int:
        return max(tuple(self._track_order.keys()) or (-1,))

    def _verify_z_contiguity(self):
        zord = self._track_order
        zx, zn = max(zord.keys()), min(zord.keys())
        ntracks = zx - zn + 1
        if ntracks != len(self._track_items) or ntracks != len(zord):
            readout = '\n'.join('{}: {}'.format(trk.z, trk.track) for trk in self._track_items.values())
            LOG.error("continuity check failed:\n" + readout)
            raise AssertionError("logic error: tracks are not contiguous in z-order")

    def set_tracks_zorder_by_name(self, order: Iterable[Tuple[str, int]]):
        changed = []
        for track, z in order:
            trk = self._track_items.get(track)
            if trk is None:
                raise ValueError("track {} not found in set_track_order offset {}".format(track, z))
            if trk.z == z:
                continue
            trk.z = z
            changed.append(trk)
        if changed:
            self.update()
        self._verify_z_contiguity()

    def tracks_zorder_by_name(self) -> Iterable[Tuple[str, int]]:
        for trk in self._track_items.values():
            yield trk.track, trk.z

    def propagate_max_z(self, new_max_z: int = None):
        if new_max_z is None:
            new_max_z = self._track_max_z
        LOG.debug('max_z propagating as {}'.format(new_max_z))
        self._coords.max_z = new_max_z

    def _shift_zorders_to_close(self, removing_at_z: int,
                                subject: QTrackItem = None,
                                zord: Mapping[int, QTrackItem] = None) -> List[QTrackItem]:
        changed = []
        if zord is None:
            zord = self._track_order
        if removing_at_z >= 0:
            # shift everything above downward
            downward = list(track for (z, track) in zord.items() if (z >= removing_at_z) and (track is not subject))
            for bub in downward:
                bub.z -= 1
                changed.append(bub)
        elif removing_at_z < 0:
            # shift everything below upward
            upward = list(track for (z, track) in zord.items() if (z < removing_at_z) and (track is not subject))
            # land above existing track
            for bub in upward:
                bub.z += 1
                changed.append(bub)
        return changed

    def _shift_zorders_to_open(self, inserting_above_z: int,
                               subject: QTrackItem = None,
                               zord: Mapping[int, QTrackItem] = None) -> Tuple[List[QTrackItem], int]:
        """Open up a z-order to move or insert a track.

        For zorder >=0 we bubble everything above upward and place new layer at trkz+1;
        for zorder < 0 we bubble everything downward.

        """
        changed = []
        if zord is None:
            zord = self._track_order
        if not zord:
            return [], inserting_above_z
        if inserting_above_z not in zord:
            if inserting_above_z >= 0:
                return [], max(zord.keys()) + 1
            else:
                return [], min(zord.keys()) - 1

        if inserting_above_z >= 0:
            upward = list(track for (z, track) in zord.items() if (z > inserting_above_z) and (track is not subject))
            # land above existing track, but we have to return a new z order above the existing track at desired z
            for bub in upward:
                bub.z += 2
                changed.append(bub)
            inserting_above_z += 1
        else:
            downward = list(track for (z, track) in zord.items() if (z <= inserting_above_z) and (track is not subject))
            for bub in downward:
                bub.z -= 1
                changed.append(bub)
        return changed, inserting_above_z

    # REMOVED since QTrackItem and QFrameItem cannot be created without a scene
    # def insert_track(self, trk: QTrackItem) -> Sequence[QTrackItem]:
    #     """Insert a track, if necessary displacing z orders such that old track at same z order is beneath
    #     return list of any QTrackItems that had to have their zorder changed
    #     """
    #     # zord = self._track_order
    #     # if trk.z not in zord:
    #     #     if (trk.track in set(zord.values())) and (trk is not self._track_items[trk.track]):
    #     #         raise AssertionError("remove track {} before replacing it with a new one".format(trk.track))
    #     #     if trk.z >= 0:  # nudge it to the contiguous top
    #     #         LOG.debug("placing {} at top of stack", trk.track)
    #     #         trk.z = self._track_max_z + 1
    #     #     else:
    #     #         LOG.debug("placing {} at bottom of stack", trk.track)
    #     #         trk.z = min(zord.keys()) - 1
    #     #     self._track_items[trk.track] = trk
    #     #     return [trk]
    #     assert(trk not in list(self._track_items.values()))
    #     changed, final_z = self._shift_zorders_to_open(trk.z)
    #     trk.z = final_z
    #     LOG.info("placing {} at zorder {}".format(trk.track, final_z))
    #     self._track_items[trk.track] = trk
    #     self.addItem(trk)
    #     trk.scene = self
    #     self._verify_z_contiguity()
    #     self.propagate_max_z()
    #     self.update()
    #     return changed

    def move_track(self, trk: QTrackItem, new_z: int) -> Sequence[QTrackItem]:
        """Change track z-order after it's been inserted
        return list of tracks that had their z-order shifted
        """
        assert (trk in self._track_items.values())
        assert (trk is self._track_items[trk.track])
        closing_changes = self._shift_zorders_to_close(trk.z, subject=trk)
        opening_changes, new_z = self._shift_zorders_to_open(new_z, subject=trk)
        trk.z = new_z
        all_changes = set(closing_changes) | set(opening_changes)
        LOG.info("shifting {} to zorder {} after moving z of {} tracks".format(trk.track, new_z, len(all_changes)))
        self._verify_z_contiguity()
        self.propagate_max_z()
        self.update()
        return [trk] + list(all_changes)

    # def select_tracks_by_metadata(self, key, value):
    #     raise NotImplementedError("NYI")
    #
    # def select_frames_by_metadata(self, key, value):
    #     raise NotImplementedError("NYI")

    #
    # high-level signals from scene to hook to other parts of application
    #
    didSelectTracksAndFrames = pyqtSignal(set, set)  # track names and frame uuids respectively
    didChangeTrackOrder = pyqtSignal(list)  # list of track UUIDs from top to bottom
    didMovePlayheadToTime = pyqtSignal(datetime, timedelta)  # instantaneous or time-range cursor move occurred
    # user requests activation/deactivation of tracks {trackname: bool} and frames {uuid: bool}
    didRequestActivation = pyqtSignal(dict, dict)
    didCopyPresentationBetweenTracks = pyqtSignal(str, str)  # from-track and to-track
    didChangePlaybackSpan = pyqtSignal(Span)  # overall playback Span had one or both of its ends moved
    didChangeVisibleAreaForView = pyqtSignal(QGraphicsView,
                                             datetime,
                                             timedelta)  # note: multiple views can share one scene

    # delegate functions to implement for document and workspace
    # these are typically called by view or scene control logic
    # e.g. to decide menu to display or progress of a drag operation
    # FUTURE: decide if we actually need a delegate ABC to compose, rather than subclass overrides
    # for now simpler is better and Scene is already delegate/model-like so let's not over-indirect
    #

    def iterate_track_info(self, start_z: int = 0, stop_z: Optional[int] = None):
        """Yield series of track information tuples which will be used to generate/update QTrackItems
        """

    def get(self, item: Union[UUID, str]) -> Union[QTrackItem, QFrameItem, None]:
        if isinstance(item, UUID):
            z = self._frame_items.get(item)
        elif isinstance(item, str):
            z = self._track_items.get(item)
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

    def tracks_in_same_family(self, track: str) -> Set[str]:
        """inform the view on which tracks are closely related to the given track
        typically this is used to stylistically highlight related tracks during a drag operation
        """
        LOG.warning("using base class tracks_in_same_family which does nothing")
        return set()

    def may_reassign_color_map(self, from_track: str, to_track: str) -> Optional[Callable[[bool], None]]:
        """Check if user colormap drag is permitted.

        User is dragging a color map around, determine if drop is permitted and provide a commit/abort function if so

        """
        LOG.warning("using base class may_reassign_color_map which does nothing")
        return lambda b: None

    def menu_for_track(self, track: str, frame: Optional[UUID] = None) -> Optional[
            Tuple[QMenu, Mapping[Any, Callable]]]:
        """Generate QMenu and action LUT to use as context menu for a given track.

        Optionally with frame if mouse was over that frame

        """
        LOG.warning("using base class menu_for_track which does nothing")
        return None

    def update(self, changed_tracks: Optional[Set[str]] = None, changed_frame_uuids: Optional[Set[UUID]] = None):
        """Populate or update scene, returning number of items changed in scene
        Does not add new items for tracks and frames already present
        Parameters serve only as hints
        """
        # super(QFramesInTracksScene, self).update()
        return 0


class TestScene(QFramesInTracksScene):
    _did_populate = False

    def __init__(self, *args, **kwargs):
        super(TestScene, self).__init__(*args, **kwargs)
        # assert(hasattr(self, '_track_order'))

    def update(self,
               changed_tracks: Optional[Set[str]] = None,
               changed_frame_uuids: Optional[Set[UUID]] = None) -> int:
        if self._did_populate:
            return 0
        self._test_populate()

    @property
    def timeline_span(self) -> Span:
        return self._span

    def _test_populate(self):
        from uuid import uuid1 as uuidgen
        once = datetime.utcnow()

        def minutes_td(minutes):
            return timedelta(minutes=minutes)
        # assert(hasattr(self, '_track_order'))
        self._span = Span(once - minutes_td(10), minutes_td(30))
        track0 = QTrackItem(self, self.coords, 'IMAGE:test::timeline:GOES-21:QBI:mars', 1,
                            "G21 QBI B99 BT", "test track", tooltip="peremptorily cromulent")
        # scene.addItem(abitrack)  # done in init
        frame01 = QFrameItem(track0, self.coords, uuidgen(), once + minutes_td(5), minutes_td(5),
                             Flags([VisualState.BUSY]), "abi1", "fulldiskimus")
        track1 = QTrackItem(self, self.coords, 'IMAGE:test::timeline:Himawari-11:AHI:mars', 0,
                            "H11 AHI B99 Rad", "second test track", tooltip="nominally cromulent")
        frame11 = QFrameItem(track1, self.coords, uuidgen(), once + minutes_td(6), minutes_td(1),
                             Flags([VisualState.READY]), "ahi1", "JP04")
        # self.insert_track(track0)
        # self.insert_track(track1)
        # assert(hasattr(self, '_propagate_max_z'))
        self.propagate_max_z()
        for track in [track0, track1]:
            track.update_pos_bounds()
            track.update_frame_positions()
        # scene.addItem(frame1)  # done in init
        # blabla = QGraphicsTextItem('abcdcba')
        # font = QFont('White Rabbit')
        # blabla.setFont(font)
        # blabla.setPos(140, 100)
        # self.addItem(blabla)
        # blabla = None
        self.content = [track0, frame01, track1, frame11]


class QFramesInTracksView(QGraphicsView):
    """Adds Track- and Frame-specific actions and signals to QGraphicsView"""

    def __init__(self, *args, **kwargs):
        super(QFramesInTracksView, self).__init__(*args, **kwargs)
        fmt = QGLFormat(QGL.SampleBuffers)
        wdgt = QGLWidget(fmt)
        assert (wdgt.isValid())
        self.setViewport(wdgt)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setRenderHints(QPainter.Antialiasing)


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

        scene.setSceneRect(QRectF(0, 0, 800, 600))
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
        self.show()
        self.raise_()


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
        import traceback
        import pdb  # noqa
        traceback.print_exception(type, value, tb)
        # …then start the debugger in post-mortem mode.
        pdb.post_mortem(tb)  # more “modern”


def main():
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
