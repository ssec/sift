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
from datetime import datetime, timedelta
from typing import Tuple, Optional, Mapping, List, Callable, Set
from uuid import UUID

from PyQt4.QtCore import QRectF, Qt, pyqtSignal
from PyQt4.QtGui import QGraphicsScene, QPen, QBrush, QPainter, QGraphicsView, QMenu, QGraphicsTextItem, QFont

from sift.view.TimelineCommon import TimelineFrameState, TimelineTrackState, TimelineCoordTransform
from sift.view.TimelineItems import QTimeRulerItem, QTrackItem, QFrameItem

LOG = logging.getLogger(__name__)


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


class QFramesInTracksView(QGraphicsView):
    """Adds Track- and Frame-specific actions and signals to QGraphicsView"""

    def __init__(self, *args, **kwargs):
        super(QFramesInTracksView, self).__init__(*args, **kwargs)