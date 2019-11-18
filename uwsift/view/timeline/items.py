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
http://pyqt.sourceforge.net/Docs/PyQt4/qpainter.html
https://stackoverflow.com/questions/4216139/python-object-in-qmimedata

:author: R.K.Garcia <rkgarcia@wisc.edu>
:copyright: 2017 by University of Wisconsin Regents, see AUTHORS for more details
:license: GPLv3, see LICENSE for more details
"""
import logging
from typing import Mapping, Any, Tuple
from uuid import UUID
from weakref import ref
import pickle as pkl
from datetime import datetime, timedelta

from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtGui import QIcon, QGradient, QImage, QPixmap, QPainter, QPen, QBrush
from PyQt5.QtWidgets import (QGraphicsItem, QGraphicsObject, QGraphicsTextItem,
                             QGraphicsPixmapItem, QStyleOptionGraphicsItem, QWidget,
                             QGraphicsSceneMouseEvent, QGraphicsSceneDragDropEvent,
                             QGraphicsSceneContextMenuEvent, QGraphicsRectItem)

from uwsift.common import Flags
from .common import (CoordTransform, GFXC, MIMETYPE_TIMELINE_COLORMAP, MIMETYPE_TIMELINE_TRACK,
                     mimed_colormap, mimed_track, recv_mime, VisualState)

LOG = logging.getLogger(__name__)


def _wtf_recursion():
    from inspect import getouterframes, currentframe
    level = len(getouterframes(currentframe()))
    LOG.debug("frame boundingRect depth {}".format(level))


def _perform_context_menu(event, scene, track, frame) -> bool:
    """Custom context menu handler shared by QTrackItem and QFrameItem
    return true if the menu fully activated and did something
    """
    qm = scene.menu_for_track(track, frame)
    if qm is not None:
        # pos: QPointF = event.pos()
        # wdgt = event.widget()
        menu, lut = qm
        if not lut or not menu:
            LOG.warning("scene menu generator returned nothing to do")
            return False
        event.accept()
        sel = menu.exec_(event.screenPos())  # wdgt.mapToGlobal(pos.toPoint()))
        todo = lut.get(sel)
        if callable(todo):
            todo(scene=scene, track=track, frame=frame)
            return True
        else:
            return False


class QTrackItem(QGraphicsObject):
    """ A group of Frames corresponding to a timeline
    This allows drag and drop of timelines to be easier
    """
    frames = None  # Iterable[QFrameItem], maintained privately between track and frame
    _scene = None  # weakref to scene
    _scale: CoordTransform = None
    _track: str = None  # tracks are family::category, e.g. IMAGE:geo:toa_reflectance:11µm::GOESR-PUG:GOES-16:ABI:CONUS
    _z: int = None  # our track number as displayed, 0 being highest on screen, with larger Z going downward
    _title: str = None
    _subtitle: str = None
    _icon: QIcon = None  # e.g. whether it's algebraic or RGB
    # arbitrary key-value store for selecting by metadata;
    # in our case this often includes item family for seleciton
    _metadata: Mapping = None
    _tooltip: str = None
    _state: Flags = None  # VisualState Flags determine how it's being presented
    _colormap: [QGradient, QImage, QPixmap] = None
    _min: float = None
    _max: float = None
    _dragging: bool = False  # whether or not a drag is in progress across this item
    # position in scene coordinates is determined by _z level and starting time of first frame, minus _left_pad
    # bounds of the track in scene coordinates, assuming 0,0 corresponds to vertical center of left edge of
    # frame representation
    _bounds: QRectF = QRectF()
    _gi_title: QGraphicsTextItem = None
    _gi_subtitle: QGraphicsTextItem = None
    _gi_icon: QGraphicsPixmapItem = None
    _gi_colormap: QGraphicsPixmapItem = None

    def __init__(self, scene, scale: CoordTransform, track: str, z: int,
                 title: str, subtitle: str = None, icon: QIcon = None, metadata: dict = None,
                 tooltip: str = None, state: Flags = None, colormap: [QGradient, QImage] = None,
                 min: float = None, max: float = None):
        """Create a track and connect it to its Scene
        """
        super(QTrackItem, self).__init__()
        self.frames = []
        self._scene = ref(scene) if scene else None
        self._scale = scale
        self._track = track
        self._z = z
        self._title = title
        self._subtitle = subtitle
        self._icon = icon
        self._metadata = metadata or {}
        self._tooltip = tooltip
        self._state = Flags(state or [])
        self._colormap = colormap
        self._min, self._max = min, max
        self.setToolTip(tooltip or "{}\n{}".format(title, subtitle))
        self.setFlag(QGraphicsItem.ItemClipsChildrenToShape, enabled=True)
        self.update_pos_bounds()
        self._update_decorations()
        scene.add_track(self)
        self.setAcceptDrops(True)

    def addFrame(self, frame):
        self._scene().add_frame(frame)
        self.frames.append(frame)

    # @property
    # def scene(self):
    #     return self._scene()
    #
    # @scene.setter
    # def scene(self, s):
    #     self._scene = ref(s)

    @property
    def track(self) -> str:
        return self._track

    @property
    def scene_(self):  # can't use just .scene due to Qt C++ scene()
        return self._scene()

    @property
    def z(self) -> int:
        return self._z

    @z.setter
    def z(self, new_z: int):
        self._z = new_z

    @property
    def default_frame_pen_brush(self):
        return self._scene().default_frame_pen_brush

    def _update_decorations(self):
        """Add decor sub-items to self
        title, subtitle, icon, colormap
        these are placed left of the local origin inside the _left_pad area
        """
        scene = self._scene()
        if self._title:
            self._gi_title = it = self._gi_title or scene.addSimpleText(self._title)
            it.setParentItem(self)
            it.setPos(GFXC.track_title_pos)
        if self._subtitle:
            self._gi_subtitle = it = self._gi_subtitle or scene.addSimpleText(self._subtitle)
            it.setParentItem(self)
            it.setPos(GFXC.track_subtitle_pos)
        # FUTURE: add draggable color-map pixmap

    # commands to cause item updates and then propagate back to the scene

    def set_colormap(self, cmap: mimed_colormap):
        """Inform scene that the user wants all tracks in our family to use this colormap
        """
        LOG.warning("set colormap from dragged colormap not yet implemented")

    def insert_track_before(self, track: mimed_track):
        """Inform scene that user wants a dragged scene moved to before us in the z-order"""
        LOG.warning("reorder tracks using dragged track not yet implemented")

    # painting and boundaries

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: QWidget = None):
        # LOG.debug("QTrackItem.paint")
        # _wtf_recursion()
        pen, brush = self._scene().default_track_pen_brush
        rect = self.boundingRect()
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(brush)
        painter.setPen(pen)

        # draw outer boundary
        painter.drawRoundedRect(rect, GFXC.track_corner_radius1, GFXC.track_corner_radius2, Qt.RelativeSize)

    def boundingRect(self) -> QRectF:
        #     if self._bounds is None:
        #         return self.update_pos_and_bounds()
        return self._bounds

    # click events / drag departures

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent):
        LOG.debug("QTrackItem mouse-down")
        return super(QTrackItem, self).mousePressEvent(event)

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent):
        LOG.debug("QTrackItem mouse-up")
        return super(QTrackItem, self).mouseReleaseEvent(event)

    def contextMenuEvent(self, event: QGraphicsSceneContextMenuEvent):
        # pos = self.mapToScene(event.pos())
        if not _perform_context_menu(event, self.scene_, self._track, None):
            return super(QTrackItem, self).contextMenuEvent(event)

    # handle drag and drop arrivals

    def dragEnterEvent(self, event: QGraphicsSceneDragDropEvent):
        self._dragging = True

        # test the content being dragged to see if it's compatible; if so, accept event
        mime = event.mimeData()
        if mime.hasFormat(MIMETYPE_TIMELINE_COLORMAP):
            event.setAccepted(True)
        elif mime.hasFormat(MIMETYPE_TIMELINE_TRACK):
            # FIXME: implement animated re-arrange of tracks
            event.setAccepted(True)
        else:
            event.setAccepted(False)

    def dragLeaveEvent(self, event: QGraphicsSceneDragDropEvent):
        self._dragging = False
        event.setAccepted(False)

    def dropEvent(self, event: QGraphicsSceneDragDropEvent):
        colormap = recv_mime(event, MIMETYPE_TIMELINE_COLORMAP)
        if colormap is not None:
            self.set_colormap(colormap)
            return
        new_track_before = recv_mime(event, MIMETYPE_TIMELINE_TRACK)
        if new_track_before is not None:
            self.insert_track_before(new_track_before)
        _ = pkl.loads(event.mimeData().data())
        event.setAccepted(False)

    # working with Frames as sub-items, updating and syncing position and extents

    def _iter_frame_children(self):
        return list(self.frames)
        # children = tuple(self.childItems())
        # # LOG.debug("{} children".format(len(children)))
        # for child in children:
        #     if isinstance(child, QFrameItem):
        #         yield child

    def _time_extent_of_frames(self):
        """start time and duration of the frames held by the track
        """
        s, e = None, None
        for child in self._iter_frame_children():
            # y relative to track is 0
            # calculate absolute x position in scene
            # assert (child.uuid != self.uuid)
            t, d = child.td
            s = t if (s is None) else min(t, s)
            e = (t + d) if (e is None) else max(e, t + d)
        if e is None:
            LOG.info("empty track cannot determine its horizontal extent")
            return None, None
        return s, e - s

    def update_pos_bounds(self):
        """Update position and bounds of the Track to reflect current TimelineCoordTransform, encapsulating frames owned
        Note that the local x=0.0 corresponds to the time of the first frame in the track
        This is also the center of rotation or animation "handle" (e.g. for track dragging)
        """
        # starting time and duration of the track, computed from frames owned
        t, d = self._time_extent_of_frames()
        if (t is None) or (d is None):
            LOG.debug("no frames contained, cannot adjust size or location of QTrackItem")
            return
        LOG.debug("time extent of frames in track '{}/{}' is {} +{}".format(self._title, self._subtitle, t, d))
        # scene y coordinate of upper left corner
        # convert track extent to scene coordinates using current transform
        frames_left, frames_width = self._scale.calc_pixel_x_pos(t, d)
        screen_track_center_y = self._scale.calc_track_pixel_y_center(self.z)
        # track_left, track_width = self._scale.calc_pixel_x_pos(t - GFXC.track_left_pad,
        #     d + GFXC.track_left_pad + GFXC.track_right_pad)
        # set track position, assuming we want origin coordinate of track item to be centered vertically within item
        # bounds relative to position in scene, left_pad space to left of local origin (x<0),
        # frames and right-pad at x>=0
        self.prepareGeometryChange()
        self._bounds = QRectF(-GFXC.track_left_pad, -GFXC.track_height / 2,
                              frames_width + GFXC.track_left_pad + GFXC.track_right_pad, GFXC.track_height)
        LOG.debug("new track bounds: {}".format(self._bounds))
        # origin is at the start of the first frame contained. padding extends into negative x
        LOG.debug("track centerline placed at {},{}".format(frames_left, screen_track_center_y))
        self.setPos(frames_left, screen_track_center_y)
        self._update_decorations()

    def update_frame_positions(self, *frames):
        """Update frames' origins relative to self after TimelineCoordTransform has changed scale
        """
        myx = self.pos().x()  # my x coordinate relative to scene
        frames = tuple(frames) or self._iter_frame_children()
        for frame in frames:
            # y relative to track is 0
            # calculate absolute x position in scene
            x, _ = self._scale.calc_pixel_x_pos(frame.t)
            frame.prepareGeometryChange()
            frame.setPos(x - myx, 0.0)


class QFrameItem(QGraphicsObject):
    """A Frame
    For SIFT use, this corresponds to a single Product or single composite of multiple Products (e.g. RGB composite)
    QGraphicsView representation of a data frame, with a start and end time relative to the scene.
    Essentially a frame sprite
    """
    _state: Flags = None
    _track = None  # weakref to track we belong to
    _scale: CoordTransform = None
    _uuid: UUID = None
    _start: datetime = None
    _duration: timedelta = None
    _title: str = None
    _subtitle: str = None
    _thumb: QPixmap = None
    _metadata: Mapping = None
    _bounds: QRectF = QRectF()
    # decorations
    _gi_title = None
    _gi_subtitle = None

    def __init__(self, track: QTrackItem, scale: CoordTransform, uuid: UUID,
                 start: datetime, duration: timedelta, state: Flags,
                 title: str, subtitle: str = None, thumb: QPixmap = None,
                 metadata: Mapping[str, Any] = None):
        """create a frame representation and add it to a timeline track within a scene
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
        self._track = ref(track)
        self._state = Flags(state)
        self._scale = scale
        self._start = start
        self._duration = duration
        self._title = title
        self._subtitle = subtitle
        self._thumb = thumb
        self._metadata = metadata
        self._uuid = uuid
        self.setToolTip("{}\n{}".format(title, subtitle))
        # self._pen, self._brush = track.default_frame_pen_brush
        # if pen:
        #     LOG.debug('setting pen')
        #     self.setPen(pen)
        # if brush:
        #     LOG.debug('setting brush')
        #     self.setBrush(brush)
        track.addFrame(self)  # adds to track, which in turn adds to scene
        self.setParentItem(track)
        self.update_bounds()
        track.update_pos_bounds()
        track.update_frame_positions()
        self.setFlag(QGraphicsItem.ItemClipsChildrenToShape, enabled=True)
        self._update_decorations()
        # self.setAcceptDrops(True)

    def _update_decorations(self):
        """Add decor sub-items to self
        title, subtitle, icon, colormap
        these are placed left of the local origin inside the _left_pad area
        """
        scene = self.scene_
        if self._title:
            self._gi_title = it = self._gi_title or scene.addSimpleText(self._title)
            it.setParentItem(self)
            it.setPos(GFXC.frame_title_pos)
        if self._subtitle:
            self._gi_subtitle = it = self._gi_subtitle or scene.addSimpleText(self._subtitle)
            it.setParentItem(self)
            it.setPos(GFXC.frame_subtitle_pos)
        # FUTURE: add draggable color-map pixmap

    @property
    def scene_(self):
        trk = self._track()
        return trk.scene_

    @property
    def track(self):
        trk = self._track()
        return trk.track

    @property
    def uuid(self):
        return self._uuid

    @property
    def t(self) -> datetime:
        """Start time of frame"""
        return self._start

    @property
    def d(self) -> timedelta:
        """Duration of frame"""
        return self._duration

    @property
    def td(self) -> Tuple[datetime, timedelta]:
        return self._start, self._duration

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, new_state: VisualState):
        if new_state != self._state:
            self._state = new_state

    # painting and boundaries

    @property
    def pen_brush(self) -> Tuple[QPen, QBrush]:
        """Pen and brush to use, based on VisualState
        """
        pen, brush = self._track().default_frame_pen_brush
        s = self._state
        if VisualState.READY in s:
            brush = QBrush(Qt.green, Qt.SolidPattern)
        return pen, brush

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: QWidget = None) -> None:
        # LOG.debug("QFrameItem.paint")
        pen, brush = self.pen_brush
        rect = self.boundingRect()
        painter.setBrush(brush)
        painter.setPen(pen)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.drawRoundedRect(rect, GFXC.frame_corner_radius, GFXC.frame_corner_radius, Qt.RelativeSize)
        # super(QFrameItem, self).paint(painter, option, widget)

    def boundingRect(self) -> QRectF:
        """return relative bounding rectangle, given position is set by Track parent as needed
        """
        # from inspect import getouterframes, currentframe
        # level = len(getouterframes(currentframe()))
        # LOG.debug("frame boundingRect depth {}".format(level))
        return self._bounds

    # internal recalculation / realignment

    def update_bounds(self):
        """set size and width based on current scaling
        position is controlled by the track, since we have to be track-relative
        """
        left = 0.0
        top = - GFXC.frame_height / 2
        height = GFXC.frame_height
        width = self._scale.calc_pixel_duration(self._duration)
        LOG.debug("width for {} is {} scene pixels".format(self._duration, width))
        old_bounds = self._bounds
        new_bounds = QRectF(left, top, width, height)
        if (old_bounds is None) or (new_bounds != old_bounds):
            self.prepareGeometryChange()
            self._bounds = new_bounds

    # # handle drag and drop
    # def dragEnterEvent(self, event: QGraphicsSceneDragDropEvent):
    #     event.setAccepted(False)
    #
    # def dragLeaveEvent(self, event: QGraphicsSceneDragDropEvent):
    #     event.setAccepted(False)
    #
    # def dropEvent(self, event: QGraphicsSceneDragDropEvent):
    #     event.setAccepted(False)

    # handle clicking
    def mousePressEvent(self, event: QGraphicsSceneMouseEvent):
        LOG.debug("QFrameItem mouse-down")
        return super(QFrameItem, self).mousePressEvent(event)

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent):
        LOG.debug("QFrameItem mouse-up")
        return super(QFrameItem, self).mouseReleaseEvent(event)

    def contextMenuEvent(self, event: QGraphicsSceneContextMenuEvent):
        LOG.debug("contextMenuEvent")
        trk = self._track()
        track = trk.track
        scene = trk.scene_
        if not _perform_context_menu(event, scene, track, self._uuid):
            return super(QFrameItem, self).contextMenuEvent(event)


class QTimeRulerItem(QGraphicsRectItem):
    """A ruler object showing the time dimension.

    An instance is at the top, bottom, or both ends of the Scene.

    """

    def __init__(self):
        super(QTimeRulerItem, self).__init__()
