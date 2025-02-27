#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Provide a SceneGraphManager to handle display of visuals.

As per http://api.vispy.org/en/latest/scene.html (abridged)

    - Vispy scene graph (SG) prerequisites:
        1. create SceneCanvas -> this object's scene property is top level node in SG:

           ```
           vispy_canvas = scene.SceneCanvas
           sg_root_node = vispy_canvas.scene
           ```
        2. create node instances (from vispy.scene.visuals)
        3. add node instances to scene by making them children of canvas scene, or
           of nodes already in the scene

"""
from __future__ import annotations

import logging
import os
from enum import Enum
from numbers import Number
from typing import TYPE_CHECKING, Optional
from uuid import UUID

import numpy as np
from PyQt5.QtCore import QObject, Qt, pyqtSignal
from PyQt5.QtGui import QCursor
from pyresample import AreaDefinition
from vispy import app, scene
from vispy.geometry import Rect
from vispy.gloo.util import _screenshot
from vispy.scene.visuals import Image, Line, Markers, Polygon
from vispy.util.keys import SHIFT
from vispy.visuals.transforms import MatrixTransform, STTransform

from uwsift import IMAGE_DISPLAY_MODE, config
from uwsift.common import (
    BORDERS_DATASET_NAME,
    DEFAULT_ANIMATION_DELAY,
    DEFAULT_GRID_CELL_HEIGHT,
    DEFAULT_GRID_CELL_WIDTH,
    DEFAULT_TILE_HEIGHT,
    DEFAULT_TILE_WIDTH,
    LATLON_GRID_DATASET_NAME,
    ImageDisplayMode,
    Info,
    Kind,
    Presentation,
    Tool,
)
from uwsift.model.area_definitions_manager import AreaDefinitionsManager
from uwsift.model.document import Document
from uwsift.model.layer_item import LayerItem
from uwsift.model.layer_model import LayerModel
from uwsift.model.product_dataset import ProductDataset
from uwsift.model.time_manager import TimeManager
from uwsift.queue import TASK_DOING, TASK_PROGRESS
from uwsift.util import get_package_data_dir
from uwsift.view.cameras import PanZoomProbeCamera
from uwsift.view.probes import DEFAULT_POINT_PROBE
from uwsift.view.transform import PROJ4Transform
from uwsift.view.visuals import (
    Lines,
    MultiChannelImage,
    NEShapefileLines,
    RGBCompositeImage,
    TiledGeolocatedImage,
)
from uwsift.workspace.utils.metadata_utils import (
    get_point_style_by_name,
    map_point_style_to_marker_kwargs,
)

if TYPE_CHECKING:
    import numpy.typing as npt


LOG = logging.getLogger(__name__)
DATA_DIR = get_package_data_dir()
DEFAULT_SHAPE_FILE = os.path.join(DATA_DIR, "ne_50m_admin_0_countries", "ne_50m_admin_0_countries.shp")
DEFAULT_STATES_SHAPE_FILE = os.path.join(
    DATA_DIR, "ne_50m_admin_1_states_provinces_lakes", "ne_50m_admin_1_states_provinces_lakes.shp"
)
DEFAULT_TEXTURE_SHAPE = (4, 16)


class CustomImage(Image):
    """Custom Image class to override the texture init.

    This class was made to prevent the texture's "internalFormat" from being set to None in the
    parent _init_texture method. Instead, "internalFormat" is explicitly set to r32f in the overridden method.
    This fixes issue #404.
    """

    def _init_texture(self, data, texture_format, **texture_kwargs):
        return super()._init_texture(data, texture_format, internalformat="r32f", **texture_kwargs)


class Markers2(Markers):
    pass


Markers = Markers2


class SIFTMainMapCanvas(scene.SceneCanvas):
    """High level map canvas node."""

    pass


class MainMap(scene.Node):
    """Scene node for holding all of the information for the main map area."""

    def __init__(self, *args, **kwargs):
        super(MainMap, self).__init__(*args, **kwargs)


class PendingPolygon(object):
    """Temporary information holder for Probe Polygons."""

    def __init__(self, point_parent):
        self.parent = point_parent
        self.markers = []
        self.canvas_points = []
        self.points = []
        self.radius = 10.0

    def _is_complete(self, canvas_pos):
        # XXX: Can't get "visuals_at" method of the SceneCanvas to work to find if the point is ready
        if len(self.points) < 3:
            return False
        p1 = self.canvas_points[0]
        r = self.radius
        if (p1[0] - r <= canvas_pos[0] <= p1[0] + r) and (p1[1] - r <= canvas_pos[1] <= p1[1] + r):
            return True

    def add_point(self, canvas_pos, xy_pos, z=100):
        if self._is_complete(canvas_pos):
            # Are you finishing the polygon by adding this point (same point as the first point...or near it)
            return True
        self.canvas_points.append(canvas_pos)
        self.points.append(xy_pos)
        if len(xy_pos) == 2:
            xy_pos = [xy_pos[0], xy_pos[1], z]
        point_visual = Markers(
            parent=self.parent,
            name="polygon_%02d" % (len(self.markers),),
            symbol="disc",
            pos=np.array([xy_pos]),
            face_color=np.array([0.0, 0.5, 0.5, 1.0]),
            edge_color=np.array([0.5, 1.0, 1.0, 1.0]),
            size=18.0,
            edge_width=3.0,
        )
        self.markers.append(point_visual)
        return False

    def reset(self):
        self.markers = []
        self.canvas_points = []
        self.points = []


class AnimationController(object):
    """Basic bookkeeping object for each layer set (A, B, C, D) from the UI."""

    def __init__(self):
        self._animation_speed = DEFAULT_ANIMATION_DELAY  # milliseconds
        self._animating = False

        self.time_manager = TimeManager(self._animation_speed)

        self._animation_timer = app.Timer(self._convert_ms_to_s(self._animation_speed))
        self._animation_timer.connect(self.time_manager.tick)

    @staticmethod
    def _convert_ms_to_s(time_ms: float) -> float:
        return time_ms / 1000.0

    @property
    def animation_speed(self):
        """speed in milliseconds"""
        return self._animation_speed

    @animation_speed.setter
    def animation_speed(self, milliseconds):
        if milliseconds <= 0:
            return
        self._animation_timer.stop()
        self._animation_speed = milliseconds
        self._animation_timer.interval = self._convert_ms_to_s(milliseconds)
        if self.animating:
            self._animation_timer.start()

    @property
    def animating(self):
        return self._animating

    @animating.setter
    def animating(self, animate):
        if animate == self._animating:
            # Don't update anything if nothing about the animation has changed
            return
        elif self._animating and not animate:
            # Stop animation
            self._animating = False
            self._animation_timer.stop()
        elif not self._animating and animate:
            # Start animation
            self._animating = True
            self._animation_timer.start()

    def toggle_animation(self, *args):
        self.animating = not self._animating
        return self.animating

    def jump(self, index):
        self.time_manager.jump(index)

    def connect_to_model(self, model: LayerModel):
        self.time_manager.connect_to_model(model)

    def get_frame_count(self):
        return self.time_manager.get_current_timebase_dataset_count()

    def get_current_frame_index(self):
        return self.time_manager.get_current_timebase_timeline_index()

    def get_current_frame_uuid(self):
        return self.time_manager.get_current_timebase_current_dataset_uuid()

    def get_frame_uuids(self):
        """
        Get a list of dataset uuids, one for each frame of the animation as the
        current timeline manager would play. The uuids are those of the current
        driving layer, therefore they are unique in the list.

        :return: list of dataset UUIDs
        """
        return self.time_manager.get_current_timebase_dataset_uuids()


class SceneGraphManager(QObject):
    """
    SceneGraphManager represents a document as a vispy scenegraph.
    When document changes, it updates to correspond.
    Handles animation by cycling visibility.
    Provides means of highlighting areas.
    Decides what sampling to bring data in from the workspace,
    in order to feed the display optimally.
    """

    # TODO(ar) REVIEW: distinction between class and member/instance
    #  variables seems random (see below)
    queue = None  # background jobs go here

    texture_shape = None

    datasets = None
    colormaps = None

    _current_tool = None
    _color_choices = None

    # FIXME: many more undocumented member variables

    didRetilingCalcs = pyqtSignal(object, object, object, object, object, object)
    newPointProbe = pyqtSignal(str, tuple)
    # REMARK: PyQT tends to fail if a signal with an argument of type 'list' is
    # passed an empty list or the 'None' object. By declaring the signal as
    # having an argument of type 'object' is avoided.
    newProbePolygon = pyqtSignal(object)

    def __init__(
        self,
        doc,
        workspace,
        queue,
        borders_shapefiles: Optional[list] = None,
        states_shapefile=None,
        parent=None,
        texture_shape=(4, 16),
        center=None,
    ):
        super(SceneGraphManager, self).__init__(parent)
        self.didRetilingCalcs.connect(self._set_retiled)

        # Parent should be the Qt widget that this GLCanvas belongs to
        self.document = doc  # Document object we work with
        self.workspace = workspace  # where we get data arrays from
        self.queue = queue
        self.borders_shapefiles = borders_shapefiles or [DEFAULT_SHAPE_FILE, DEFAULT_STATES_SHAPE_FILE]
        self.texture_shape = texture_shape
        self.polygon_probes: dict = {}
        self.point_probes: dict = {}

        self.layer_nodes: dict = {}  # {layer_uuid: layer_node}
        self.dataset_nodes: dict = {}  # {dataset_uuid: dataset_node}
        self.latlon_grid_node: Optional[Line] = None
        self.borders_nodes: list = []

        self.composite_element_dependencies: dict = {}  # {dataset_uuid:set-of-dependent-uuids}
        self.animation_controller = AnimationController()

        self._current_tool = None

        self._connect_doc_signals(self.document)

        # border and lat/lon grid color choices
        self._color_choices = [
            np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),  # white
            np.array([0.5, 0.5, 0.5, 1.0], dtype=np.float32),  # gray
            np.array([0.0, 1.0, 1.0, 1.0], dtype=np.float32),  # cyan
            np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),  # black
            np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),  # transparent
        ]
        self._latlon_grid_color_idx = 1
        self._borders_color_idx = 0

        self._setup_initial_canvas(center)
        self.pending_polygon = PendingPolygon(self.main_map)

    def get_screenshot_array(
        self, frame_range: None | tuple[int, int] = None
    ) -> list[tuple[str | UUID, npt.NDArray[np.uint8]]]:
        """Get numpy arrays representing the current canvas.

        Args:
            frame_range: Start and end frame indexes to get arrays for.
                Indexes are 0-based and both values (start and end) are
                *inclusive*. Specifying ``(1, 3)`` means you'll get arrays
                for frame at index 1 (the second frame), index 2, and index 3.
                If not specified or ``None`` the current frame's data is
                returned.

        """
        # Store current index to reset the view once we are done
        # Or use it as the frame to screenshot if no frame range is specified
        current_frame = self.animation_controller.get_current_frame_index()
        current_uuid = self.animation_controller.get_current_frame_uuid()
        if not current_frame and not current_uuid:
            # no data loaded
            self.main_canvas.on_draw(None)
            return [("", _screenshot())]
        if frame_range is None:
            # screenshot the current view
            s = e = current_frame
        else:
            s, e = frame_range

        images = []
        for i in range(s, e + 1):
            self.animation_controller.jump(i)
            self._update()
            self.main_canvas.on_draw(None)
            u = self.animation_controller.get_current_frame_uuid()
            images.append((u, _screenshot()))

        self.animation_controller.jump(current_frame)
        self._update()
        self.main_canvas.on_draw(None)
        return images

    def _setup_initial_canvas(self, center=None):
        self.main_canvas = SIFTMainMapCanvas(parent=self.parent())
        self.main_view = self.main_canvas.central_widget.add_view(name="MainView")

        # Camera Setup
        self.pz_camera = PanZoomProbeCamera(
            name=Tool.PAN_ZOOM.name, aspect=1, pan_limits=(-1.0, -1.0, 1.0, 1.0), zoom_limits=(0.0015, 0.0015)
        )
        self.main_view.camera = self.pz_camera
        self.main_view.camera.flip = (False, False, False)
        self.main_view.events.mouse_press.connect(self.on_mouse_press_point)
        self.main_view.events.mouse_press.connect(self.on_mouse_press_region)
        self.change_tool(Tool.PAN_ZOOM)

        z_level_transform = MatrixTransform()
        # near/far is backwards it seems:
        camera_z_scale = 1e-6
        z_level_transform.set_ortho(-1.0, 1.0, -1.0, 1.0, -100.0 * camera_z_scale, 100.0 * camera_z_scale)

        # Head node of all visualizations, needed mostly to scale Z level
        self.main_map_parent = scene.Node(name="HeadNode", parent=self.main_view.scene)
        self.main_map_parent.transform = z_level_transform

        # Head node of the map graph
        self.main_map = MainMap(name="MainMap", parent=self.main_map_parent)

        self._create_test_image()

        area_def = self.document.area_definition()
        self._set_projection(area_def)

    def _create_test_image(self):
        proj4_str = os.getenv("SIFT_DEBUG_IMAGE_PROJ", None)
        if proj4_str is None:
            return
        shape = (2000, 2000)
        fake_data = np.zeros(shape, np.float32) + 0.5
        fake_data[:5, :] = 1.0
        fake_data[-5:, :] = 1.0
        fake_data[:, :5] = 1.0
        fake_data[:, -5:] = 1.0
        cell_size = 1000
        origin_x = -shape[1] / 2.0 * cell_size
        origin_y = shape[0] / 2.0 * cell_size

        image = TiledGeolocatedImage(
            fake_data,
            origin_x,
            origin_y,
            cell_size,
            cell_size,
            name="Test Image",
            clim=(0.0, 1.0),
            gamma=1.0,
            interpolation="nearest",
            method="subdivide",
            cmap=self.document.find_colormap("grays"),
            double=False,
            texture_shape=DEFAULT_TEXTURE_SHAPE,
            wrap_lon=False,
            parent=self.main_map,
            projection=proj4_str,
        )
        image.transform = PROJ4Transform(proj4_str, inverse=True)
        image.transform *= STTransform(translate=(0, 0, -50.0))
        self._test_img = image

    def set_projection(self, area_display_name: str, center=None):
        area_def = AreaDefinitionsManager.area_def_by_name(area_display_name)
        assert area_def is not None  # nosec B101
        self._set_projection(area_def, center)

        for dataset_node in self.dataset_nodes.values():
            if hasattr(dataset_node, "determine_reference_points"):
                dataset_node.determine_reference_points()
        self.on_view_change(None)

    def _set_projection(self, area_def: AreaDefinition, center=None):
        self.main_map.transform = PROJ4Transform(area_def.proj_str)

        ll_xy = area_def.area_extent[:2]
        ur_xy = area_def.area_extent[2:]

        # FIXME: This method is called via setup_initial_canvas() before the
        #  system layer(s) and their nodes (here: self.latlon_grid_node) have
        #  been initialized.  When 'center' is not None in that case
        #  calculating 'mapped_center' will crash. Therefore as long as there
        #  is no solution for the next FIX-ME this must be prevented by
        #  revising the application setup process.  For the moment, we assume
        #  that no one wants to use 'center' already when the application is
        #  started and therefore we ...
        assert center is None or self.latlon_grid_node is not None  # nosec B101

        if center:
            # FIXME: We should be able to use the main_map object to do the
            #  transform but it doesn't work (waiting on vispy developers)
            # mapped_center = self.main_map.transforms\
            #    .get_transform(map_to="scene").map([center])[0][:2]
            assert self.latlon_grid_node is not None  # nosec B101 # suppress mypy [union-attr]
            mapped_center = self.latlon_grid_node.transforms.get_transform(map_to="scene").map([center])[0][:2]
            ll_xy += mapped_center
            ur_xy += mapped_center

        self.main_view.camera.rect = Rect(ll_xy, (ur_xy[0] - ll_xy[0], ur_xy[1] - ll_xy[1]))

    @staticmethod
    def _create_latlon_grid_points(resolution=5.0):
        """Create a series of line segments representing latitude and longitude lines.

        :param resolution: number of degrees between lines
        """
        lons = np.arange(-180.0, 180.0 + resolution, resolution, dtype=np.float32)
        lats = np.arange(-90.0, 90.0 + resolution, resolution, dtype=np.float32)

        # One long line of lawn mower pattern (lon lines, then lat lines)
        points = np.empty((lons.shape[0] * lats.shape[0] * 2, 2), np.float32)
        LOG.debug("Generating longitude lines...")
        for idx, lon_point in enumerate(lons):
            points[idx * lats.shape[0] : (idx + 1) * lats.shape[0], 0] = lon_point
            if idx % 2 == 0:
                points[idx * lats.shape[0] : (idx + 1) * lats.shape[0], 1] = lats
            else:
                points[idx * lats.shape[0] : (idx + 1) * lats.shape[0], 1] = lats[::-1]
        start_idx = lons.shape[0] * lats.shape[0]
        LOG.debug("Generating latitude lines...")
        for idx, lat_point in enumerate(lats[::-1]):
            points[start_idx + idx * lons.shape[0] : start_idx + (idx + 1) * lons.shape[0], 1] = lat_point
            if idx % 2 == 0:
                points[start_idx + idx * lons.shape[0] : start_idx + (idx + 1) * lons.shape[0], 0] = lons
            else:
                points[start_idx + idx * lons.shape[0] : start_idx + (idx + 1) * lons.shape[0], 0] = lons[::-1]

        # Repeat for "second" size of the earth (180 to 540)
        offset = 360  # box_x[lons.shape[0] - 1] - box_x[0]
        points2 = np.empty((points.shape[0] * 2, 2), dtype=np.float32)
        points2[: points.shape[0], :] = points
        points2[points.shape[0] :, :] = points
        points2[points.shape[0] :, 0] += offset

        return points2

    def on_mouse_press_point(self, event):
        """Handle mouse events that mean we are using the point probe."""
        if event.handled:
            return
        modifiers = event.mouse_event.modifiers
        if (event.button == 2 and not modifiers) or (self._current_tool == Tool.POINT_PROBE and event.button == 1):
            buffer_pos = event.sources[0].transforms.get_transform().map(event.pos)
            # FIXME: We should be able to use the main_map object to do the transform
            #  but it doesn't work (waiting on vispy developers)
            # map_pos = self.main_map.transforms.get_transform().imap(buffer_pos)
            map_pos = self.latlon_grid_node.transforms.get_transform().imap(buffer_pos)
            if np.any(np.abs(map_pos[:2]) > 1e25):
                LOG.error("Invalid point probe location")
                return
            self.newPointProbe.emit(DEFAULT_POINT_PROBE, tuple(map_pos[:2]))

    def on_mouse_press_region(self, event):
        """Handle mouse events that mean we are using the point probe."""
        if event.handled:
            return
        modifiers = event.mouse_event.modifiers
        if (event.button == 2 and modifiers == (SHIFT,)) or (
            self._current_tool == Tool.REGION_PROBE and event.button == 1
        ):
            buffer_pos = event.sources[0].transforms.get_transform().map(event.pos)
            # FIXME: We should be able to use the main_map object to do the transform
            #  but it doesn't work (waiting on vispy developers)
            # map_pos = self.main_map.transforms.get_transform().imap(buffer_pos)
            map_pos = self.latlon_grid_node.transforms.get_transform().imap(buffer_pos)
            if np.any(np.abs(map_pos[:2]) > 1e25):
                LOG.error("Invalid region probe location")
                return
            if self.pending_polygon.add_point(event.pos[:2], map_pos[:2], 60):
                points = self.pending_polygon.points + [self.pending_polygon.points[0]]
                self.clear_pending_polygon()
                self.newProbePolygon.emit(points)

    def clear_pending_polygon(self):
        for marker in self.pending_polygon.markers:
            # Remove the marker from the scene graph
            marker.parent = None
        # Reset the pending polygon object
        self.pending_polygon.reset()

    def remove_polygon(self, name=None):
        """Remove a polygon from the SGM or clear the pending polygon if it exists."""
        if name is None:
            LOG.debug("No polygon name specified to remove")
            return

        if name not in self.polygon_probes:
            LOG.warning("Tried to remove a nonexistent polgyon: %s", name)
            return

        self.polygon_probes[name].parent = None
        del self.polygon_probes[name]

    def has_pending_polygon(self):
        return len(self.pending_polygon.points) != 0

    def on_point_probe_set(self, probe_name, state, xy_pos, **kwargs):
        z = float(kwargs.get("z", 60))
        edge_color = kwargs.get("edge_color", np.array([1.0, 0.5, 0.5, 1.0]))
        face_color = kwargs.get("face_color", np.array([0.5, 0.0, 0.0, 1.0]))
        if len(xy_pos) == 2:
            xy_pos = [xy_pos[0], xy_pos[1], z]

        probe_kwargs = {
            "symbol": "disc",
            "pos": np.array([xy_pos]),
            "face_color": face_color,
            "edge_color": edge_color,
            "size": 18.0,
            "edge_width": 3.0,
        }

        if probe_name not in self.point_probes and xy_pos is None:
            raise ValueError("Probe '{}' does not exist".format(probe_name))
        elif probe_name not in self.point_probes:
            point_visual = Markers(parent=self.main_map, name=probe_name, **probe_kwargs)
            self.point_probes[probe_name] = point_visual
        else:
            point_visual = self.point_probes[probe_name]
            point_visual.set_data(**probe_kwargs)

        # set the Point visible or not
        point_visual.visible = state

    def on_new_polygon(self, probe_name, points, **kwargs):
        points = np.array(points, dtype=np.float32)  # convert list to NumPy array

        kwargs.setdefault("color", None)
        kwargs.setdefault("border_color", (1.0, 0.0, 1.0, 1.0))

        # marker default is 60, polygon default is 50 so markers can be put on top of polygons
        z = float(kwargs.get("z", 50))
        poly = Polygon(parent=self.main_map, pos=points, **kwargs)
        poly.order = 50  # set polygons to be drawn last (stops 'see through' polygons)
        poly.transform = STTransform(translate=(0, 0, z))
        if probe_name in self.polygon_probes:
            self.polygon_probes[probe_name].parent = None
        self.polygon_probes[probe_name] = poly

    def copy_polygon(self, old_name, new_name):
        self.on_new_polygon(new_name, self.polygon_probes[old_name].pos)

    def show_only_polygons(self, polygon_names_to_show):
        temp_set = set(polygon_names_to_show)
        for polygon_name in self.polygon_probes.keys():
            self.polygon_probes[polygon_name].visible = polygon_name in temp_set

    def _update(self):
        return self.main_canvas.update()

    def cycle_borders_color(self):
        self._borders_color_idx = (self._borders_color_idx + 1) % len(self._color_choices)
        if self._borders_color_idx + 1 == len(self._color_choices):
            for borders_node in self.borders_nodes:
                borders_node.visible = False
        else:
            for borders_node in self.borders_nodes:
                borders_node.set_data(color=self._color_choices[self._borders_color_idx])
                borders_node.visible = True

    def cycle_latlon_grid_color(self):
        self._latlon_grid_color_idx = (self._latlon_grid_color_idx + 1) % len(self._color_choices)
        if self._latlon_grid_color_idx + 1 == len(self._color_choices):
            self.latlon_grid_node.visible = False
        else:
            self.latlon_grid_node.set_data(color=self._color_choices[self._latlon_grid_color_idx])
            self.latlon_grid_node.visible = True

    def change_tool(self, name: Tool):
        prev_tool = self._current_tool
        if name == prev_tool:
            # it's the same tool
            return

        self._current_tool = name

        # Set the cursor
        if name == Tool.PAN_ZOOM:
            self.main_canvas.native.setCursor(QCursor(Qt.OpenHandCursor))
        elif name == Tool.POINT_PROBE:
            self.main_canvas.native.setCursor(QCursor(Qt.PointingHandCursor))
        elif name == Tool.REGION_PROBE:
            self.main_canvas.native.setCursor(QCursor(Qt.CrossCursor))

        # disconnect the previous signals (if needed)
        if prev_tool == Tool.PAN_ZOOM:
            self.main_view.events.mouse_press.disconnect(self.pz_camera.viewbox_mouse_event)
            self.main_view.events.mouse_release.disconnect(self.pz_camera.viewbox_mouse_event)
            self.main_view.events.mouse_move.disconnect(self.pz_camera.viewbox_mouse_event)

        # connect the new signals (if needed)
        if name == Tool.PAN_ZOOM:
            self.main_view.events.mouse_press.connect(self.pz_camera.viewbox_mouse_event)
            self.main_view.events.mouse_release.connect(self.pz_camera.viewbox_mouse_event)
            self.main_view.events.mouse_move.connect(self.pz_camera.viewbox_mouse_event)

        LOG.info("Changing tool to '%s'", name)

    def next_tool(self):
        tool_names = list(Tool)
        idx = tool_names.index(self._current_tool)
        idx = (idx + 1) % len(tool_names)
        self.change_tool(tool_names[idx])

    def _set_colormap(self, colormap, uuid=None):
        colormap = self.document.find_colormap(colormap)

        uuids = uuid
        if uuid is None:
            uuids = self.dataset_nodes.keys()
        elif not isinstance(uuid, (list, tuple)):
            uuids = [uuid]

        for uuid in uuids:
            dataset_node = self.dataset_nodes[uuid]
            if isinstance(dataset_node, TiledGeolocatedImage) or isinstance(dataset_node, Image):
                self.dataset_nodes[uuid].cmap = colormap
            else:
                self.dataset_nodes[uuid].color = colormap

    def _set_color_limits(self, clims, uuid=None):
        """Update the color limits for the specified UUID"""
        uuids = uuid
        if uuid is None:
            uuids = self.dataset_nodes.keys()
        elif not isinstance(uuid, (list, tuple)):
            uuids = [uuid]

        for uuid in uuids:
            dataset_node = self.dataset_nodes.get(uuid, None)
            if dataset_node is not None:
                self.dataset_nodes[uuid].clim = clims

    def _set_gamma(self, gamma, uuid):
        uuids = uuid
        if uuid is None:
            uuids = self.dataset_nodes.keys()
        elif not isinstance(uuid, (list, tuple)):
            uuids = [uuid]

        for uuid in uuids:
            dataset_node = self.dataset_nodes.get(uuid, None)
            if dataset_node is not None and hasattr(dataset_node, "gamma"):
                self.dataset_nodes[uuid].gamma = gamma

    def change_dataset_nodes_colormap(self, change_dict):
        for uuid, cmapid in change_dict.items():
            LOG.info("changing {} to colormap {}".format(uuid, cmapid))
            self._set_colormap(cmapid, uuid)

    def change_dataset_nodes_color_limits(self, change_dict):
        for uuid, clims in change_dict.items():
            LOG.debug("changing {} to color limits {}".format(uuid, clims))
            self._set_color_limits(clims, uuid)

    def change_dataset_nodes_gamma(self, change_dict):
        for uuid, gamma in change_dict.items():
            LOG.debug("changing {} to gamma {}".format(uuid, gamma))
            self._set_gamma(gamma, uuid)

    def change_layer_visible(self, layer_uuid: UUID, visible: bool):
        self.layer_nodes[layer_uuid].visible = visible

    def change_layer_opacity(self, layer_uuid: UUID, opacity: float):
        # According to
        #   https://vispy.org/api/vispy.scene.node.html#vispy.scene.node.Node.parent
        # this should be sufficient, but it seems to be not:
        #   self.layer_nodes[uuid].opacity = opacity
        # Thus opacity must be set for all layer node children:
        for child in self.layer_nodes[layer_uuid].children:
            child.opacity = opacity
        # TODO in case a dataset has its own Presentation simply overwriting
        #  the opacity of the 'child' node representing it is wrong:
        #  opacities have to be mixed then. This cannot be done here though
        self._update()

    def change_dataset_visible(self, dataset_uuid: UUID, visible: bool):
        self.dataset_nodes[dataset_uuid].visible = visible

    @staticmethod
    def _overwrite_with_test_pattern(data):
        """
        Fill given data with distinct test data.

        Fill the given data array with zeros except for some selected cells
        which are set to distinct values: 5 cells at each corner and 6 cells
        around the center which form asymmetrical patterns to make them clearly
        distinguishable.

        When the data is visualized with the color table 'Rainbow (IR Default)'
        these cells are colored as named in Enum RainbowValue.

        This function must only be called during development for calibration/
        validation purposes.
        """
        max_x, max_y = data.shape
        data[:, :] = 0

        class RainbowValue(Enum):
            BROWN = 320.0
            RED = 300.0
            LIGHT_GREEN = 280.0
            GREEN = 260.0
            LIGHT_BLUE = 240.0
            DARK_BLUE = 220.0
            PINK = 200.0

        center_x, center_y = max_x // 2, max_y // 2
        pixels = [
            {"x": center_x, "y": center_y, "color": RainbowValue.RED, "desc": "center"},
            {"x": center_x - 1, "y": center_y - 1, "color": RainbowValue.GREEN, "desc": "upper left"},
            {"x": center_x - 1, "y": center_y + 1, "color": RainbowValue.LIGHT_BLUE, "desc": "bottom left"},
            {"x": center_x - 1, "y": center_y + 2, "color": RainbowValue.LIGHT_GREEN, "desc": "below bottom left"},
            {"x": center_x + 1, "y": center_y - 1, "color": RainbowValue.DARK_BLUE, "desc": "upper right"},
            {"x": center_x + 1, "y": center_y + 1, "color": RainbowValue.PINK, "desc": "bottom right"},
            {"x": max_x - 1, "y": max_y - 1, "color": RainbowValue.RED, "desc": "bottom right corner"},
            {"x": max_x - 2, "y": max_y - 1, "color": RainbowValue.GREEN, "desc": "bottom right corner"},
            {"x": max_x - 3, "y": max_y - 1, "color": RainbowValue.LIGHT_BLUE, "desc": "bottom right corner"},
            {"x": max_x - 1, "y": max_y - 2, "color": RainbowValue.PINK, "desc": "bottom right corner"},
            {"x": max_x - 2, "y": max_y - 2, "color": RainbowValue.BROWN, "desc": "bottom right corner"},
            {"x": 0, "y": max_y - 1, "color": RainbowValue.PINK, "desc": "bottom left corner"},
            {"x": 1, "y": max_y - 1, "color": RainbowValue.LIGHT_BLUE, "desc": "bottom left corner"},
            {"x": 0, "y": max_y - 2, "color": RainbowValue.BROWN, "desc": "bottom left corner"},
            {"x": 0, "y": max_y - 3, "color": RainbowValue.RED, "desc": "bottom left corner"},
            {"x": 1, "y": max_y - 2, "color": RainbowValue.GREEN, "desc": "bottom left corner"},
            {"x": max_x - 1, "y": 0, "color": RainbowValue.LIGHT_BLUE, "desc": "upper right corner"},
            {"x": max_x - 2, "y": 0, "color": RainbowValue.LIGHT_GREEN, "desc": "upper right corner"},
            {"x": max_x - 3, "y": 0, "color": RainbowValue.RED, "desc": "upper right corner"},
            {"x": max_x - 1, "y": 1, "color": RainbowValue.BROWN, "desc": "upper right corner"},
            {"x": max_x - 2, "y": 1, "color": RainbowValue.PINK, "desc": "upper right corner"},
            {"x": 0, "y": 0, "color": RainbowValue.BROWN, "desc": "upper left corner"},
            {"x": 1, "y": 0, "color": RainbowValue.RED, "desc": "upper left corner"},
            {"x": 0, "y": 1, "color": RainbowValue.PINK, "desc": "upper left corner"},
            {"x": 0, "y": 2, "color": RainbowValue.DARK_BLUE, "desc": "upper left corner"},
            {"x": 1, "y": 1, "color": RainbowValue.LIGHT_GREEN, "desc": "upper left corner"},
        ]

        for pixel in pixels:
            data[pixel["y"], pixel["x"]] = pixel["color"].value

        return data

    def add_node_for_layer(self, layer: LayerItem):
        if IMAGE_DISPLAY_MODE == ImageDisplayMode.PIXEL_MATRIX and layer.kind in [
            Kind.IMAGE,
            Kind.COMPOSITE,
            Kind.RGB,
            Kind.MC_IMAGE,
        ]:
            # Circumvent all reprojecting transformations
            layer_node = scene.Node(parent=self.main_map_parent, name=str(layer.uuid))
        else:
            # Make child of the node with the reprojecting transform
            layer_node = scene.Node(parent=self.main_map, name=str(layer.uuid))

        z_transform = STTransform(translate=(0, 0, 0))
        layer_node.transform = z_transform

        self.layer_nodes[layer.uuid] = layer_node

    def add_node_for_system_generated_data(self, layer: LayerItem):
        layer_node = self.layer_nodes[layer.uuid]
        if layer.name == LATLON_GRID_DATASET_NAME:
            self._build_latlon_grid_node(layer_node)
        elif layer.name == BORDERS_DATASET_NAME:
            self._build_borders_nodes(layer_node)
        else:
            raise ValueError(f"Unsupported generated layer: {layer.name}")

    def _build_latlon_grid_node(self, layer_node):
        """Helper function for setting up the VisualNode for the system
        layer for latitude/longitude grid.

        :param layer_node: Scene graph node to be used as parent for the grid
                           node.
        """
        latlon_grid_resolution = get_configured_latlon_grid_resolution()
        latlon_grid_points = self._create_latlon_grid_points(resolution=latlon_grid_resolution)
        self.latlon_grid_node = Line(
            pos=latlon_grid_points,
            connect="strip",
            color=self._color_choices[self._latlon_grid_color_idx],
            parent=layer_node,
        )
        self.latlon_grid_node.set_gl_state("translucent")

    def _build_borders_nodes(self, layer_node):
        """Helper function for setting up the VisualNodes for the system
        layer for political borders.

        One node is generated for each file stored in the (currently) internal
        list of political borders shapefiles.

        :param layer_node: Scene graph node to be used as parent for the
                           borders node(s).
        """
        for shapefile in self.borders_shapefiles:
            node = NEShapefileLines(
                shapefile, double=True, color=self._color_choices[self._borders_color_idx], parent=layer_node
            )
            node.set_gl_state("translucent")
            self.borders_nodes.append(node)

    def apply_presentation_to_image_node(
        self, image: Image, presentation: Presentation, visible: Optional[bool] = None
    ):
        """
        Apply all relevant and set properties (not None) of the given
        presentation to the given image.

        Visibility can be explicitly overridden, because this is (at least for
        now) the only property where a dataset may deviate from the layer
        presentation; it depends on whether the dataset is active in the layer's
        timeline.

        :param image: the image node which should get the new presentation
        :param presentation: to apply, usually the presentation of the owning
               layer
        :param visible:
        """
        if visible is not None:
            image.visible = visible
        elif presentation.visible:
            image.visible = presentation.visible

        if presentation.colormap:
            image.cmap = self.document.find_colormap(presentation.colormap)
        if presentation.climits:
            image.clim = presentation.climits
        if presentation.gamma:
            image.gamma = presentation.gamma
        if presentation.opacity:
            image.opacity = presentation.opacity

    @staticmethod
    def _calc_subdivision_grid(dataset_info) -> tuple:
        grid_cell_width = float(config.get("display.grid_cell_width", DEFAULT_GRID_CELL_WIDTH))
        grid_cell_height = float(config.get("display.grid_cell_height", DEFAULT_GRID_CELL_HEIGHT))

        if "longlat" in dataset_info[Info.PROJ]:
            # The cell size unit is not metres but degrees, thus we do a rough unit conversion
            EARTH_CIRCUMFERENCE: float = 40075017.0  # metres
            pixel_width_metres = abs(dataset_info[Info.CELL_WIDTH]) * EARTH_CIRCUMFERENCE / 360.0
            pixel_height_metres = abs(dataset_info[Info.CELL_HEIGHT]) * EARTH_CIRCUMFERENCE / 360.0

        else:
            pixel_width_metres = abs(dataset_info[Info.CELL_WIDTH])
            pixel_height_metres = abs(dataset_info[Info.CELL_HEIGHT])

        pixels_per_grid_cell_x = round(grid_cell_width / pixel_width_metres)
        pixels_per_grid_cell_y = round(grid_cell_height / pixel_height_metres)

        num_grid_cells_x = dataset_info[Info.SHAPE][0] // pixels_per_grid_cell_x
        num_grid_cells_y = dataset_info[Info.SHAPE][1] // pixels_per_grid_cell_y

        actual_grid_cell_width = dataset_info[Info.SHAPE][0] * abs(dataset_info[Info.CELL_WIDTH]) / num_grid_cells_x
        actual_grid_cell_height = dataset_info[Info.SHAPE][1] * abs(dataset_info[Info.CELL_HEIGHT]) / num_grid_cells_y

        LOG.debug(
            f"Gridding to ({num_grid_cells_x} x {num_grid_cells_y}) cells"
            f" with cell size ({actual_grid_cell_width} m, {actual_grid_cell_height} m) "
        )

        return num_grid_cells_x, num_grid_cells_y

    def add_node_for_image_dataset(self, layer: LayerItem, product_dataset: ProductDataset):
        assert self.layer_nodes[layer.uuid] is not None  # nosec B101
        assert product_dataset.kind in [Kind.IMAGE, Kind.COMPOSITE]  # nosec B101

        image_data = self.workspace.get_content(product_dataset.uuid, kind=product_dataset.kind)

        if False:  # Set to True FOR TESTING ONLY DON'T REMOVE!
            self._overwrite_with_test_pattern(image_data)

        if IMAGE_DISPLAY_MODE == ImageDisplayMode.TILED_GEOLOCATED:
            image = TiledGeolocatedImage(
                image_data,
                product_dataset.info[Info.ORIGIN_X],
                product_dataset.info[Info.ORIGIN_Y],
                product_dataset.info[Info.CELL_WIDTH],
                product_dataset.info[Info.CELL_HEIGHT],
                name=str(product_dataset.uuid),
                interpolation="nearest",
                method="subdivide",
                double=False,
                texture_shape=DEFAULT_TEXTURE_SHAPE,
                wrap_lon=False,
                parent=self.layer_nodes[layer.uuid],
                projection=product_dataset.info[Info.PROJ],
            )
            image.transform = PROJ4Transform(product_dataset.info[Info.PROJ], inverse=True)
            image.determine_reference_points()
        elif IMAGE_DISPLAY_MODE == ImageDisplayMode.SIMPLE_GEOLOCATED:
            grid = self._calc_subdivision_grid(product_dataset.info)
            image = CustomImage(
                image_data,
                name=str(product_dataset.uuid),
                interpolation="nearest",
                method="subdivide",
                grid=grid,
                parent=self.layer_nodes[layer.uuid],
            )
            image.transform = PROJ4Transform(product_dataset.info[Info.PROJ], inverse=True) * STTransform(
                scale=(product_dataset.info[Info.CELL_WIDTH], product_dataset.info[Info.CELL_HEIGHT], 1),
                translate=(product_dataset.info[Info.ORIGIN_X], product_dataset.info[Info.ORIGIN_Y], 0),
            )
        else:
            image = CustomImage(
                image_data,
                name=str(product_dataset.uuid),
                interpolation="nearest",
                parent=self.layer_nodes[layer.uuid],
            )
            image.transform = STTransform(
                scale=(product_dataset.info[Info.CELL_WIDTH], product_dataset.info[Info.CELL_HEIGHT], 1),
                translate=(product_dataset.info[Info.ORIGIN_X], product_dataset.info[Info.ORIGIN_Y], 0),
            )
        self.dataset_nodes[product_dataset.uuid] = image
        # Make sure *all* applicable properties of the owning layer's current
        # presentation are applied to the new image node
        self.apply_presentation_to_image_node(image, layer.presentation)
        self.on_view_change(None)
        LOG.debug("Scene Graph after IMAGE dataset insertion:")
        LOG.debug(self.main_view.describe_tree(with_transform=True))

    def add_node_for_mc_image_dataset(self, layer: LayerItem, product_dataset: ProductDataset) -> None:
        """Create and add a new node for a multichannel images to the SceneGraphManager.

        Depending on the system configuration either a node with or without tiling is created from the product_dataset
        and inserted as a child of the layer's node in the scene graph.

        :param layer: LayerItem which owns the ProductDataset
        :param product_dataset: ProductDataset to create the multichannel image for
        """
        assert self.layer_nodes[layer.uuid] is not None  # nosec B101
        assert product_dataset.kind == Kind.MC_IMAGE  # nosec B101

        img_data = self.workspace.get_content(product_dataset.uuid, kind=product_dataset.kind)

        if IMAGE_DISPLAY_MODE == ImageDisplayMode.TILED_GEOLOCATED:
            image = TiledGeolocatedImage(
                img_data,
                product_dataset.info[Info.ORIGIN_X],
                product_dataset.info[Info.ORIGIN_Y],
                product_dataset.info[Info.CELL_WIDTH],
                product_dataset.info[Info.CELL_HEIGHT],
                name=str(product_dataset.uuid),
                interpolation="nearest",
                method="subdivide",
                double=False,
                #  TODO: (Inform David about the strange behavior)
                #   workaround (setting clims to (0.0, 1.0) because if no clim is set then default is auto and
                #   this is not corrcetly resolved before build tiles. But the real clim value is need before
                #   ImageVisual._build_color_transform() is executed.
                clim=(0.0, 1.0),
                texture_shape=DEFAULT_TEXTURE_SHAPE,
                tile_shape=(DEFAULT_TILE_HEIGHT, DEFAULT_TILE_WIDTH, img_data.shape[2]),
                wrap_lon=False,
                parent=self.layer_nodes[layer.uuid],
                projection=product_dataset.info[Info.PROJ],
            )
            image.transform = PROJ4Transform(product_dataset.info[Info.PROJ], inverse=True)
            image.determine_reference_points()
        elif IMAGE_DISPLAY_MODE == ImageDisplayMode.SIMPLE_GEOLOCATED:
            grid = self._calc_subdivision_grid(product_dataset.info)
            image = Image(
                img_data,
                name=str(product_dataset.uuid),
                interpolation="nearest",
                method="subdivide",
                grid=grid,
                parent=self.layer_nodes[layer.uuid],
            )
            image.transform = PROJ4Transform(product_dataset.info[Info.PROJ], inverse=True) * STTransform(
                scale=(product_dataset.info[Info.CELL_WIDTH], product_dataset.info[Info.CELL_HEIGHT], 1),
                translate=(product_dataset.info[Info.ORIGIN_X], product_dataset.info[Info.ORIGIN_Y], 0),
            )
        else:  # IMAGE_DISPLAY_MODE == ImageDisplayMode.PIXEL_MATRIX
            image = Image(
                img_data,
                name=str(product_dataset.uuid),
                interpolation="nearest",
                parent=self.layer_nodes[layer.uuid],
            )
            image.transform = STTransform(
                scale=(product_dataset.info[Info.CELL_WIDTH], product_dataset.info[Info.CELL_HEIGHT], 1),
                translate=(product_dataset.info[Info.ORIGIN_X], product_dataset.info[Info.ORIGIN_Y], 0),
            )
        self.dataset_nodes[product_dataset.uuid] = image
        self.on_view_change(None)
        LOG.debug("Scene Graph after MC IMAGE dataset insertion:")
        LOG.debug(self.main_view.describe_tree(with_transform=True))

    def add_node_for_composite_dataset(self, layer: LayerItem, product_dataset: ProductDataset):
        assert self.layer_nodes[layer.uuid] is not None  # nosec B101
        assert product_dataset.kind == Kind.RGB  # nosec B101
        assert product_dataset.input_datasets_uuids is not None  # nosec B101 # suppress mypy [union-attr]

        images_data = list(
            self.workspace.get_content(curr_input_uuid, Kind.IMAGE)
            for curr_input_uuid in product_dataset.input_datasets_uuids
        )

        if IMAGE_DISPLAY_MODE == ImageDisplayMode.TILED_GEOLOCATED:
            composite = RGBCompositeImage(
                images_data,
                product_dataset.info[Info.ORIGIN_X],
                product_dataset.info[Info.ORIGIN_Y],
                product_dataset.info[Info.CELL_WIDTH],
                product_dataset.info[Info.CELL_HEIGHT],
                name=str(product_dataset.uuid),
                clim=layer.presentation.climits,
                gamma=layer.presentation.gamma,
                interpolation="nearest",
                method="subdivide",
                cmap=None,
                double=False,
                texture_shape=DEFAULT_TEXTURE_SHAPE,
                wrap_lon=False,
                parent=self.layer_nodes[layer.uuid],
                projection=product_dataset.info[Info.PROJ],
            )
            composite.transform = PROJ4Transform(product_dataset.info[Info.PROJ], inverse=True)
            composite.determine_reference_points()
        elif IMAGE_DISPLAY_MODE == ImageDisplayMode.SIMPLE_GEOLOCATED:
            grid = self._calc_subdivision_grid(product_dataset.info)
            composite = MultiChannelImage(
                images_data,
                name=str(product_dataset.uuid),
                clim=layer.presentation.climits,
                gamma=layer.presentation.gamma,
                interpolation="nearest",
                method="subdivide",
                grid=grid,
                cmap=None,
                parent=self.layer_nodes[layer.uuid],
            )
            composite.transform = PROJ4Transform(product_dataset.info[Info.PROJ], inverse=True) * STTransform(
                scale=(product_dataset.info[Info.CELL_WIDTH], product_dataset.info[Info.CELL_HEIGHT], 1),
                translate=(product_dataset.info[Info.ORIGIN_X], product_dataset.info[Info.ORIGIN_Y], 0),
            )
        else:
            composite = MultiChannelImage(
                images_data,
                name=str(product_dataset.uuid),
                clim=layer.presentation.climits,
                gamma=layer.presentation.gamma,
                interpolation="nearest",
                cmap=None,
                parent=self.layer_nodes[layer.uuid],
            )
            composite.transform = STTransform(
                scale=(product_dataset.info[Info.CELL_WIDTH], product_dataset.info[Info.CELL_HEIGHT], 1),
                translate=(product_dataset.info[Info.ORIGIN_X], product_dataset.info[Info.ORIGIN_Y], 0),
            )
        self.composite_element_dependencies[product_dataset.uuid] = product_dataset.input_datasets_uuids
        self.dataset_nodes[product_dataset.uuid] = composite
        self.on_view_change(None)
        LOG.debug("Scene Graph after COMPOSITE dataset insertion:")
        LOG.debug(self.main_view.describe_tree(with_transform=True))

    def add_node_for_lines_dataset(self, layer: LayerItem, product_dataset: ProductDataset) -> scene.VisualNode:
        assert self.layer_nodes[layer.uuid] is not None  # nosec B101
        assert product_dataset.kind == Kind.LINES  # nosec B101

        content, _ = self.workspace.get_lines_arrays(product_dataset.uuid)
        if content is None:
            LOG.info(f"Dataset contains no lines: {product_dataset.uuid}")
            return

        lines = Lines(content, parent=self.layer_nodes[layer.uuid])
        lines.set_gl_state("translucent")
        lines.name = str(product_dataset.uuid)

        self.dataset_nodes[product_dataset.uuid] = lines
        self.on_view_change(None)
        LOG.debug("Scene Graph after LINES dataset insertion:")
        LOG.debug(self.main_view.describe_tree(with_transform=True))

    def add_node_for_points_dataset(self, layer: LayerItem, product_dataset: ProductDataset) -> scene.VisualNode:
        assert self.layer_nodes[layer.uuid] is not None  # nosec B101
        assert product_dataset.kind == Kind.POINTS  # nosec B101

        pos, values = self.workspace.get_points_arrays(product_dataset.uuid)
        if pos is None:
            LOG.info(f"dataset contains no points: {product_dataset.uuid}")
            return

        kwargs = map_point_style_to_marker_kwargs(get_point_style_by_name(layer.presentation.style))

        if values is not None:
            assert len(pos) == len(values)  # nosec B101
            # TODO use climits of the presentation instead of autoscaling?
            colormap = self.document.find_colormap(layer.presentation.colormap)
            kwargs["face_color"] = self.map_to_colors_autoscaled(colormap, values)

        points = Markers(pos=pos, parent=self.layer_nodes[layer.uuid], **kwargs)
        points.set_gl_state("translucent")  # makes no difference though
        points.name = str(product_dataset.uuid)

        self.dataset_nodes[product_dataset.uuid] = points
        self.on_view_change(None)
        LOG.debug("Scene Graph after POINTS dataset insertion:")
        LOG.debug(self.main_view.describe_tree(with_transform=True))

    def map_to_colors_autoscaled(self, colormap, values, m=2):
        """Get a list of colors by mapping each entry in values by the given colormap.

        The mapping range is adjusted automatically to m times the standard
        deviation from the mean. This ignores outliers in the calculation of
        the mapping range.

        Caution: this is an expensive operation and must not be called in tight
        loops.

        :param colormap: the colormap to apply
        :param values: the values to map to colors
        :param m: factor to stretch the standard deviation around the mean to define the mapping range
        :return: list of mapped colors in the same order as the input values
        """
        std_dev = np.std(values)
        mean = np.mean(values)
        min = mean - m * std_dev  # noqa: calm down PyCharm's spelling check, ...
        max = mean + m * std_dev  # noqa: ... 'min' and 'max' are fine!
        scaled_attr = np.interp(values, (min, max), (0, 1))
        colors = colormap.map(scaled_attr)
        return colors

    def change_node_for_composite_dataset(self, layer: LayerItem, product_dataset: ProductDataset):
        if layer.kind == Kind.RGB:
            if product_dataset.uuid in self.dataset_nodes:
                # RGB selection has changed, rebuild the dataset
                LOG.debug(
                    f"Changing existing composite dataset to"
                    f" Scene Graph Manager with UUID:"
                    f" {product_dataset.uuid}"
                )

                assert product_dataset.input_datasets_uuids is not None  # nosec B101 # suppress mypy [union-attr]
                images_data = list(
                    self.workspace.get_content(curr_input_id, Kind.IMAGE)
                    for curr_input_id in product_dataset.input_datasets_uuids
                )

                self.composite_element_dependencies[product_dataset.uuid] = product_dataset.input_datasets_uuids

                composite = self.dataset_nodes[product_dataset.uuid]
                if isinstance(composite, RGBCompositeImage):
                    composite.set_channels(
                        images_data,
                        cell_width=product_dataset.info[Info.CELL_WIDTH],
                        cell_height=product_dataset.info[Info.CELL_HEIGHT],
                        origin_x=product_dataset.info[Info.ORIGIN_X],
                        origin_y=product_dataset.info[Info.ORIGIN_Y],
                    )
                elif isinstance(composite, MultiChannelImage):
                    composite.set_data(images_data)
                composite.clim = layer.presentation.climits
                composite.gamma = layer.presentation.gamma
                self.on_view_change(None)
                if isinstance(composite, RGBCompositeImage):
                    composite.determine_reference_points()
                self._update()
            else:
                self.add_node_for_composite_dataset(layer, product_dataset)
        else:
            raise ValueError("Unknown or unimplemented composite type")

    def update_basic_dataset(self, uuid: UUID, kind: Kind):
        """
        Push the data (content) of a basic dataset again to the associated scene
        graph node.

        This method shall be called whenever the data of a basic dataset changes.
        :param uuid: identifier of the dataset
        :param kind: kind of the dataset / data content.
        """
        try:
            dataset_node = self.dataset_nodes[uuid]
            dataset_content = self.workspace.get_content(uuid, kind=kind)
            dataset_node.set_data(dataset_content)
        except NotImplementedError:
            if isinstance(dataset_node, TiledGeolocatedImage):
                LOG.debug(
                    f"Updating data for UUID {uuid} on its associated"
                    f" scenegraph TiledGeolocatedImage node is not"
                    f" possible, hopefully the data was modified in-place"
                    f" (e.g. when merging new granules)."
                )
                # TODO: How to detect the case that the data was not changed in
                #  place but a new reference was given? In this case, we must
                #  re-raise the NotImplementedError exception (as in the 'else'
                #  path)
                # TODO: TiledGeolocatedImage does not provide a way to tell it
                #  that it should drop all retiled data and start from scratch.
            else:
                # This is an unforeseen case: at the moment this method
                # should only be called when merging data segments into existing
                # image(!) data, looks like it was called for a node of another
                # type not having set_data() too.
                raise
        except KeyError:
            LOG.fatal(f"Node for dataset with the uuid '{uuid}' does not exist in the scene graph. This is a BUG!")
            raise

        self.on_view_change(None)
        self._update()

    def update_layers_z(self, uuids: list):
        if self.layer_nodes:
            # Rendering order must be set analogous to z order
            # (higher z values -> further away), render back to front
            # https://vispy.org/faq.html#how-to-achieve-transparency-with-2d-objects
            z_counter = 0
            for z_level, uuid in enumerate(uuids):
                layer_node = self.layer_nodes[uuid]
                layer_node.transform.translate = (0, 0, 0 - z_level)
                layer_node.order = z_counter
                z_counter -= 1
            self._update()

    def purge_dataset(self, uuid_removed: UUID):
        """
        Dataset has been purged from document (no longer used anywhere) - flush it all out
        :param uuid_removed: UUID of the dataset that is to be removed
        :return:
        """
        if uuid_removed in self.dataset_nodes:
            dataset = self.dataset_nodes[uuid_removed]
            dataset.parent = None
            del self.dataset_nodes[uuid_removed]
            LOG.info(f"dataset {uuid_removed} purge from Scene Graph")
        else:
            LOG.debug(f"dataset {uuid_removed} already purged from Scene Graph")
        LOG.debug("Scene Graph after dataset deletion:")
        LOG.debug(self.main_view.describe_tree(with_transform=True))

    def remove_layer_node(self, uuid_removed: UUID):
        """
        Layer will be removed, but before it can be removed correctly the scene graph node has to be removed.
        :param uuid_removed: UUID of the layer that will be removed
        """
        if uuid_removed in self.layer_nodes:
            layer = self.layer_nodes[uuid_removed]
            layer.parent = None
            del self.layer_nodes[uuid_removed]
            LOG.info(f"layer {uuid_removed} removed from Scene Graph")
        else:
            LOG.debug(f"Layer {uuid_removed} already removed from Scene Graph")
        LOG.debug("Scene Graph after layer deletion:")
        LOG.debug(self.main_view.describe_tree(with_transform=True))

    def _connect_doc_signals(self, document: Document):
        document.didUpdateBasicDataset.connect(self.update_basic_dataset)  # new data integrated in existing layer

    def _set_dataset_visible(self, uuid: UUID, visible: Optional[bool] = None):
        dataset_node = self.dataset_nodes.get(uuid, None)
        if dataset_node is None:
            return
        dataset_node.visible = not dataset_node.visible if visible is None else visible

    def on_view_change(self, scheduler):
        """Simple event handler for when we need to reassess image datasets."""
        # Stop the timer so it doesn't continuously call this slot
        if scheduler:
            scheduler.stop()

        def _assess(uuid, child):
            need_retile, preferred_stride, tile_box = child.assess()
            if need_retile:
                self._start_retiling_task(uuid, preferred_stride, tile_box)

        current_datasets_uuids = self.dataset_nodes.keys()

        def _assess_if_active(uuid):
            dataset_node = self.dataset_nodes.get(uuid, None)
            if dataset_node is not None and hasattr(dataset_node, "assess"):
                _assess(uuid, dataset_node)

        # update all available datasets nodes
        for uuid in current_datasets_uuids:
            _assess_if_active(uuid)

    def _start_retiling_task(self, uuid, preferred_stride, tile_box):
        LOG.debug("Scheduling retile for child with UUID: %s", uuid)
        self.queue.add(
            str(uuid) + "_retile",
            self._retile_child(uuid, preferred_stride, tile_box),
            "Retile calculations for image dataset" + str(uuid),
            interactive=True,
        )

    def _retile_child(self, uuid, preferred_stride, tile_box):
        LOG.debug("Retiling child with UUID: '%s'", uuid)
        yield {TASK_DOING: "Re-tiling", TASK_PROGRESS: 0.0}
        if uuid not in self.composite_element_dependencies:
            kind = self.document[uuid].get(Info.KIND)
            child = self.dataset_nodes[uuid]
            data = self.workspace.get_content(uuid, lod=preferred_stride, kind=kind)
            yield {TASK_DOING: "Re-tiling", TASK_PROGRESS: 0.5}
            # FIXME: Use LOD instead of stride and provide the lod to the workspace
            data = data[:: preferred_stride[0], :: preferred_stride[1]]
            tiles_info, vertices, tex_coords = child.retile(data, preferred_stride, tile_box)
            yield {TASK_DOING: "Re-tiling", TASK_PROGRESS: 1.0}
            self.didRetilingCalcs.emit(uuid, preferred_stride, tile_box, tiles_info, vertices, tex_coords)
        else:
            child = self.dataset_nodes[uuid]
            data = [
                self.workspace.get_content(d_uuid, lod=preferred_stride)
                for d_uuid in self.composite_element_dependencies[uuid]
            ]
            yield {TASK_DOING: "Re-tiling", TASK_PROGRESS: 0.5}
            # FIXME: Use LOD instead of stride and provide the lod to the workspace
            data = [
                d[:: int(preferred_stride[0] / factor), :: int(preferred_stride[1] / factor)] if d is not None else None
                for factor, d in zip(child._channel_factors, data)
            ]
            tiles_info, vertices, tex_coords = child.retile(data, preferred_stride, tile_box)
            yield {TASK_DOING: "Re-tiling", TASK_PROGRESS: 1.0}
            self.didRetilingCalcs.emit(uuid, preferred_stride, tile_box, tiles_info, vertices, tex_coords)
        self.workspace.bgnd_task_complete()  # FUTURE: consider a threading context manager for this??

    def _set_retiled(self, uuid, preferred_stride, tile_box, tiles_info, vertices, tex_coords):
        """Slot to take data from background thread and apply it to the dataset living in the image dataset."""
        child = self.dataset_nodes.get(uuid, None)
        if child is None:
            LOG.warning("unable to find uuid %s in dataset_nodes" % uuid)
            return
        child.set_retiled(preferred_stride, tile_box, tiles_info, vertices, tex_coords)
        child.update()


# TODO move these defaults to common config defaults location
LATLON_GRID_RESOLUTION_MIN: float = 0.1
LATLON_GRID_RESOLUTION_DEFAULT: float = 5.0
LATLON_GRID_RESOLUTION_MAX: float = 10.0


def get_configured_latlon_grid_resolution() -> float:
    resolution: float = config.get("latlon_grid.resolution", LATLON_GRID_RESOLUTION_DEFAULT)

    if not isinstance(resolution, Number):
        LOG.warning(
            f"Invalid configuration for lat/lon grid resolution"
            f" (='{resolution}') found."
            f" Using the default {LATLON_GRID_RESOLUTION_DEFAULT}°."
        )
        return LATLON_GRID_RESOLUTION_DEFAULT

    if resolution > LATLON_GRID_RESOLUTION_MAX:
        LOG.warning(
            f"Configured lat/lon grid resolution {resolution}°"
            f" is greater than allowed maximum."
            f" Using the maximum {LATLON_GRID_RESOLUTION_MAX}°."
        )
        return LATLON_GRID_RESOLUTION_MAX

    if resolution < LATLON_GRID_RESOLUTION_MIN:
        LOG.warning(
            f"Configured lat/lon grid resolution {resolution}°"
            f" is less than allowed minimum."
            f" Using the minimum {LATLON_GRID_RESOLUTION_MIN}°."
        )
        return LATLON_GRID_RESOLUTION_MIN

    return resolution
