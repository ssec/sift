#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LayerRep.py
~~~~~~~~~~~

PURPOSE
Layer representation - the "physical" realization of content to draw on the map.
A layer representation can have multiple levels of detail

A factory will convert URIs into LayerReps
LayerReps are managed by document, and handed off to the MapWidget as part of a LayerDrawingPlan

REFERENCES


REQUIRES


:author: R.K.Garcia <rayg@ssec.wisc.edu>
:copyright: 2014 by University of Wisconsin Regents, see AUTHORS for more details
:license: GPLv3, see LICENSE for more details
"""
__docformat__ = 'reStructuredText'
__author__ = 'davidh'

import logging
import os
from uuid import UUID

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal, Qt
from PyQt5.QtGui import QCursor
from vispy import app
from vispy import scene
from vispy.geometry import Rect
from vispy.scene.visuals import Markers, Polygon, Compound, Line
from vispy.util.keys import SHIFT
from vispy.visuals import LineVisual
from vispy.visuals.transforms import STTransform, MatrixTransform, ChainTransform
from vispy.gloo.util import _screenshot

from uwsift.common import DEFAULT_ANIMATION_DELAY, Info, Kind, Tool, Presentation
from uwsift.model.document import DocLayerStack, DocBasicLayer
from uwsift.queue import TASK_DOING, TASK_PROGRESS
from uwsift.util import get_package_data_dir
from uwsift.view.cameras import PanZoomProbeCamera
from uwsift.view.probes import DEFAULT_POINT_PROBE
from uwsift.view.transform import PROJ4Transform
from uwsift.view.visuals import (NEShapefileLines, TiledGeolocatedImage, RGBCompositeLayer, PrecomputedIsocurve)

LOG = logging.getLogger(__name__)
DATA_DIR = get_package_data_dir()
DEFAULT_SHAPE_FILE = os.path.join(DATA_DIR, 'ne_50m_admin_0_countries', 'ne_50m_admin_0_countries.shp')
DEFAULT_STATES_SHAPE_FILE = os.path.join(DATA_DIR, 'ne_50m_admin_1_states_provinces_lakes',
                                         'ne_50m_admin_1_states_provinces_lakes.shp')
DEFAULT_TEXTURE_SHAPE = (4, 16)


class Markers2(Markers):
    def _set_clipper(self, node, clipper):
        return


Markers = Markers2


class FakeMarker(Compound):
    # FIXME: Temporary workaround because markers don't work on the target Windows laptops
    def __init__(self, pos=None, parent=None, symbol=None, **kwargs):
        self.line_one = None
        self.line_two = None
        self.symbol = symbol
        point = pos[0]

        kwargs["connect"] = "segments"
        width = 5
        pos1, pos2 = self._get_positions(point)
        if self.line_one is None:
            self.line_one = LineVisual(pos=pos1, width=width, **kwargs)
            self.line_two = LineVisual(pos=pos2, width=width, **kwargs)

        # For some reason we can't add the subvisuals later, so we'll live with redundant logic
        super().__init__((self.line_one, self.line_two), parent=parent)

        # self.set_point(point, **kwargs)

    def _get_positions(self, point):
        margin = 0.5
        if self.symbol == 'x':
            pos1 = np.array([[point[0] - margin, point[1] - margin * 2, point[2]],
                             [point[0] + margin, point[1] + margin * 2, point[2]]])
            pos2 = np.array([[point[0] - margin, point[1] + margin * 2, point[2]],
                             [point[0] + margin, point[1] - margin * 2, point[2]]])
        else:
            pos1 = np.array([[point[0] - margin, point[1], point[2]], [point[0] + margin, point[1], point[2]]])
            pos2 = np.array([[point[0], point[1] - margin * 2, point[2]], [point[0], point[1] + margin * 2, point[2]]])
        return pos1, pos2

    def set_point(self, point, **kwargs):
        kwargs["connect"] = "segments"
        pos1, pos2 = self._get_positions(point)
        self.line_one.set_data(pos=pos1)
        self.line_two.set_data(pos=pos2)


class SIFTMainMapCanvas(scene.SceneCanvas):
    """High level map canvas node."""
    pass


class MainMap(scene.Node):
    """Scene node for holding all of the information for the main map area."""

    def __init__(self, *args, **kwargs):
        super(MainMap, self).__init__(*args, **kwargs)


class PendingPolygon(object):
    """Temporary information holder for Probe Polygons.
    """

    def __init__(self, point_parent):
        self.parent = point_parent
        self.markers = []
        self.canvas_points = []
        self.points = []
        self.radius = 10.0

    def is_complete(self, canvas_pos):
        # XXX: Can't get "visuals_at" method of the SceneCanvas to work to find if the point is ready
        if len(self.points) < 3:
            return False
        p1 = self.canvas_points[0]
        r = self.radius
        if (p1[0] - r <= canvas_pos[0] <= p1[0] + r) and (p1[1] - r <= canvas_pos[1] <= p1[1] + r):
            return True

    def add_point(self, canvas_pos, xy_pos, z=100):
        if self.is_complete(canvas_pos):
            # Are you finishing the polygon by adding this point (same point as the first point...or near it)
            return True
        self.canvas_points.append(canvas_pos)
        self.points.append(xy_pos)
        if len(xy_pos) == 2:
            xy_pos = [xy_pos[0], xy_pos[1], z]
        point_visual = Markers(parent=self.parent,
                               name='polygon_%02d' % (len(self.markers),),
                               symbol="disc", pos=np.array([xy_pos]),
                               face_color=np.array([0., 0.5, 0.5, 1.]),
                               edge_color=np.array([.5, 1.0, 1.0, 1.]),
                               size=18.,
                               edge_width=3.,
                               )
        self.markers.append(point_visual)
        return False

    def reset(self):
        self.markers = []
        self.canvas_points = []
        self.points = []


class LayerSet(object):
    """Basic bookkeeping object for each layer set (A, B, C, D) from the UI.

    Each LayerSet has its own:
     - Per layer visiblity
     - Animation loop and frame order
     - Layer Order
    """

    def __init__(self, parent, layers=None, layer_order=None, frame_order=None, frame_change_cb=None):
        if layers is None and (layer_order is not None or frame_order is not None):
            raise ValueError("'layers' required when 'layer_order' or 'frame_order' is specified")

        self.parent = parent
        self._layers = {}
        self._layer_order = []  # display (z) order, top to bottom
        self._frame_order = []  # animation order, first to last
        self._animating = False
        self._frame_number = 0
        self._frame_change_cb = frame_change_cb
        self._animation_speed = DEFAULT_ANIMATION_DELAY  # milliseconds
        self._animation_timer = app.Timer(self._animation_speed / 1000.0, connect=self.next_frame)

        if layers is not None:
            self.set_layers(layers)

            if layer_order is None:
                layer_order = [x.name for x in layers.keys()]
            self.set_layer_order(layer_order)

            if frame_order is None:
                frame_order = [x.name for x in layers.keys()]
            self.frame_order = frame_order

    @property
    def current_frame(self):
        return self._frame_number

    @property
    def max_frame(self):
        return len(self._frame_order)

    @property
    def animation_speed(self):
        """speed in milliseconds
        """
        return self._animation_speed

    @animation_speed.setter
    def animation_speed(self, milliseconds):
        if milliseconds <= 0:
            return
        self._animation_timer.stop()
        self._animation_speed = milliseconds
        self._animation_timer.interval = milliseconds / 1000.0
        if self._frame_order:
            self._animating = True
            self._animation_timer.start()
        if self._frame_change_cb is not None and self._frame_order:
            uuid = self._frame_order[self._frame_number]
            self._frame_change_cb((self._frame_number, len(self._frame_order), self._animating, uuid))

    def set_layers(self, layers):
        # FIXME clear the existing layers
        for layer in layers:
            self.add_layer(layer)

    def add_layer(self, layer):
        LOG.debug('add layer {}'.format(layer))
        uuid = UUID(layer.name)  # we backitty-forth this because
        self._layers[uuid] = layer
        self._layer_order.insert(0, uuid)
        self.update_layers_z()
        # self._frame_order.append(uuid)

    def set_layer_order(self, layer_order):
        for o in layer_order:
            # Layer names are UUIDs
            if o not in self._layers and o is not None:
                LOG.error('set_layer_order cannot deal with unknown layer {}'.format(o))
                return
        self._layer_order = list(layer_order)
        self.update_layers_z()

    @property
    def frame_order(self):
        return self._frame_order

    @frame_order.setter
    def frame_order(self, frame_order):
        for o in frame_order:
            if o not in self._layers:
                LOG.error('set_frame_order cannot deal with unknown layer {}'.format(o))
                return
        self._frame_order = frame_order
        # FIXME: ticket #92: this is not a good idea
        self._frame_number = 0
        # LOG.debug('accepted new frame order of length {}'.format(len(frame_order)))
        # if self._frame_change_cb is not None and self._frame_order:
        #     uuid = self._frame_order[self._frame_number]
        #     self._frame_change_cb((self._frame_number, len(self._frame_order), self._animating, uuid))

    def update_layers_z(self):
        for z_level, uuid in enumerate(self._layer_order):
            transform = self._layers[uuid].transform
            if isinstance(transform, ChainTransform):
                # assume ChainTransform where the last transform is STTransform for Z level
                transform = transform.transforms[-1]
            transform.translate = (0, 0, 0 - int(z_level))
            self._layers[uuid].order = len(self._layer_order) - int(z_level)
        # Need to tell the scene to recalculate the drawing order (HACK, but it works)
        # FIXME: This should probably be accomplished by overriding the right method from the Node or Visual class
        self.parent.main_canvas._update_scenegraph(None)

    def top_layer_uuid(self):
        for layer_uuid in self._layer_order:
            if self._layers[layer_uuid].visible:
                return layer_uuid
        # None of the image layers are visible
        return None

    @property
    def animating(self):
        return self._animating

    @animating.setter
    def animating(self, animate):
        if animate == self._animating:
            # Don't update anything if nothing about the animation has changed
            return
        elif self._animating and not animate:
            # We are currently, but don't want to be
            self._animating = False
            self._animation_timer.stop()
        elif not self._animating and animate and self._frame_order:
            # We are not currently, but want to be
            self._animating = True
            self._animation_timer.start()
            # TODO: Add a proper AnimationEvent to self.events
        if self._frame_change_cb is not None and self._frame_order:
            uuid = self._frame_order[self._frame_number]
            self._frame_change_cb((self._frame_number, len(self._frame_order), self._animating, uuid))

    def toggle_animation(self, *args):
        self.animating = not self._animating
        return self.animating

    def _set_visible_node(self, node):
        """Set all nodes to invisible except for the `event.added` node.
        """
        for child in self._layers.values():
            with child.events.blocker():
                if child is node.added:
                    child.visible = True
                else:
                    child.visible = False

    def _set_visible_child(self, frame_number):
        for idx, uuid in enumerate(self._frame_order):
            child = self._layers[uuid]
            # not sure if this is actually doing anything
            with child.events.blocker():
                if idx == frame_number:
                    child.visible = True
                else:
                    child.visible = False

    def next_frame(self, event=None, frame_number=None):
        """
        skip to the frame (from 0) or increment one frame and update
        typically this is run by self._animation_timer
        :param frame_number: optional frame to go to, from 0
        :return:
        """
        lfo = len(self._frame_order)
        frame = self._frame_number
        if frame_number is None:
            frame = self._frame_number + 1
        elif isinstance(frame_number, int):
            if frame_number == -1:
                frame = self._frame_number + (lfo - 1)
            else:
                frame = frame_number
        if lfo > 0:
            frame %= lfo
        else:
            frame = 0
        self._set_visible_child(frame)
        self._frame_number = frame
        self.parent.update()
        if self._frame_change_cb is not None and lfo:
            uuid = self._frame_order[self._frame_number]
            self._frame_change_cb((self._frame_number, lfo, self._animating, uuid))


class ContourGroupNode(scene.Node):
    """VisPy scene graph node managing multiple visuals.

    This Node handles view changes and representing different "zoom" levels
    in the data that is provided to its child widgets.

    """

    @staticmethod
    def visible_first(children):
        invisible_children = []
        for c in children:
            if c.visible:
                yield c
            else:
                invisible_children.append(c)
        for c in invisible_children:
            yield c

    def on_view_change(self):
        zoom_level = None
        for child in self.visible_first(self.children):
            if isinstance(child, PrecomputedIsocurve):
                if zoom_level is None:
                    zoom_level = self._assess_contour(child)
                # child handles an unchanged zoom_level
                child.zoom_level = zoom_level
            else:
                raise NotImplementedError("Don't know how to assess "
                                          "non-contour layer")

    def _assess_contour(self, child):
        """Calculate shown portion of image and image units per pixel

        This method utilizes a precomputed "mesh" of relatively evenly
        spaced points over the entire image space. This mesh is transformed
        to the canvas space (-1 to 1 user-viewed space) to figure out which
        portions of the image are currently being viewed and which portions
        can actually be projected on the viewed projection.

        While the result of the chosen method may not always be completely
        accurate, it should work for all possible viewing cases.
        """
        # in contour coordinate space, the extents of the canvas
        canvas_extents = child.transforms.get_transform().imap([
            [-1., -1.],
            [0., 0.],
            [1., 1.],
            [-1., 1.],
            [1., -1.]
        ])[:, :2]
        canvas_size = self.canvas.size
        # valid projection coordinates
        canvas_extents = canvas_extents[(canvas_extents[:, 0] <= 1e30) & (canvas_extents[:, 1] <= 1e30), :]
        if not canvas_extents.size:
            LOG.warning("Can't determine current view box, using lowest contour resolution")
            zoom_level = 0
        else:
            min_x = canvas_extents[:, 0].min()
            max_x = canvas_extents[:, 0].max()
            min_y = canvas_extents[:, 1].min()
            max_y = canvas_extents[:, 1].max()
            pixel_ratio = max((max_x - min_x) / canvas_size[0], (max_y - min_y) / canvas_size[1])

            if pixel_ratio > 10000:
                zoom_level = 0
            elif pixel_ratio > 5000:
                zoom_level = 1
            elif pixel_ratio > 3000:
                zoom_level = 2
            elif pixel_ratio > 1000:
                zoom_level = 3
            else:
                zoom_level = 4

        return zoom_level


class SceneGraphManager(QObject):
    """
    SceneGraphManager represents a document as a vispy scenegraph.
    When document changes, it updates to correspond.
    Handles animation by cycling visibility.
    Provides means of highlighting areas.
    Decides what sampling to bring data in from the workspace,
    in order to feed the display optimally.
    """

    document = None  # Document object we work with
    workspace = None  # where we get data arrays from
    queue = None  # background jobs go here

    border_shapefile = None  # background political map
    texture_shape = None
    polygon_probes = None
    point_probes = None

    image_elements = None  # {layer_uuid:element}
    composite_element_dependencies = None  # {layer_uuid:set-of-dependent-uuids}
    datasets = None
    colormaps = None
    layer_set = None

    _current_tool = None
    _color_choices = None

    # FIXME: many more undocumented member variables

    didRetilingCalcs = pyqtSignal(object, object, object, object, object, object)
    didChangeFrame = pyqtSignal(tuple)
    didChangeLayerVisibility = pyqtSignal(dict)  # similar to document didChangeLayerVisibility
    newPointProbe = pyqtSignal(str, tuple)
    newProbePolygon = pyqtSignal(object, object)

    def __init__(self, doc, workspace, queue,
                 border_shapefile=None, states_shapefile=None,
                 parent=None, texture_shape=(4, 16), center=None):
        super(SceneGraphManager, self).__init__(parent)
        self.didRetilingCalcs.connect(self._set_retiled)

        # Parent should be the Qt widget that this GLCanvas belongs to
        self.document = doc
        self.workspace = workspace
        self.queue = queue
        self.border_shapefile = border_shapefile or DEFAULT_SHAPE_FILE
        self.conus_states_shapefile = states_shapefile or DEFAULT_STATES_SHAPE_FILE
        self.texture_shape = texture_shape
        self.polygon_probes = {}
        self.point_probes = {}

        self.image_elements = {}
        self.composite_element_dependencies = {}
        self.layer_set = LayerSet(self, frame_change_cb=self.frame_changed)
        self._current_tool = None

        self._connect_doc_signals(self.document)

        # border and lat/lon grid color choices
        self._color_choices = [
            np.array([1., 1., 1., 1.], dtype=np.float32),  # white
            np.array([.5, .5, .5, 1.], dtype=np.float32),  # gray
            np.array([0., 1., 1., 1.], dtype=np.float32),  # cyan
            np.array([0., 0., 0., 1.], dtype=np.float32),  # black
            np.array([0., 0., 0., 0.], dtype=np.float32),  # transparent
        ]

        self.setup_initial_canvas(center)
        self.pending_polygon = PendingPolygon(self.main_map)

    def get_screenshot_array(self, frame_range=None):
        """Get numpy arrays representing the current canvas."""
        if frame_range is None:
            self.main_canvas.on_draw(None)
            return [(self.layer_set.top_layer_uuid(), _screenshot())]
        s, e = frame_range

        # reset the view once we are done
        c = self.layer_set.current_frame
        images = []
        for i in range(s, e + 1):
            self.set_frame_number(i)
            self.update()
            self.main_canvas.on_draw(None)
            u = self.layer_set.frame_order[i] if self.layer_set.frame_order else None
            images.append((u, _screenshot()))
        self.set_frame_number(c)
        self.update()
        self.main_canvas.on_draw(None)
        return images

    def frame_changed(self, frame_info):
        """Callback which emits information on current animation frame as a signal (see LayerSet.next_frame)

        Args:
            frame_info (tuple): to be relayed in the signal.
                Typically (frame_index:int, total_frames:int, animating:bool, frame_id:UUID)

        """
        # LOG.debug('emitting didChangeFrame')
        self.didChangeFrame.emit(frame_info)
        is_animating = frame_info[2]
        if not is_animating:
            # emit a signal equivalent to document's didChangeLayerVisibility,
            # except that visibility is being changed by animation interactions
            # only do this when we're not animating, however
            # watch out for signal loops!
            uuids = self.layer_set.frame_order
            # note that all the layers in the layer_order but the current one are now invisible
            vis = dict((u, u == frame_info[3]) for u in uuids)
            self.didChangeLayerVisibility.emit(vis)

    def setup_initial_canvas(self, center=None):
        self.main_canvas = SIFTMainMapCanvas(parent=self.parent())
        self.main_view = self.main_canvas.central_widget.add_view()

        # Camera Setup
        self.pz_camera = PanZoomProbeCamera(name=Tool.PAN_ZOOM.name, aspect=1, pan_limits=(-1., -1., 1., 1.),
                                            zoom_limits=(0.0015, 0.0015))
        self.main_view.camera = self.pz_camera
        self.main_view.camera.flip = (False, False, False)
        self.main_view.events.mouse_press.connect(self.on_mouse_press_point)
        self.main_view.events.mouse_press.connect(self.on_mouse_press_region)
        self.change_tool(Tool.PAN_ZOOM)

        z_level_transform = MatrixTransform()
        # near/far is backwards it seems:
        camera_z_scale = 1e-6
        z_level_transform.set_ortho(-1., 1., -1., 1., -100.0 * camera_z_scale, 100.0 * camera_z_scale)

        # Head node of all visualizations, needed mostly to scale Z level
        self.main_map_parent = scene.Node(name="HeadNode", parent=self.main_view.scene)
        self.main_map_parent.transform = z_level_transform

        # Head node of the map graph
        proj_info = self.document.projection_info()
        self.main_map = MainMap(name="MainMap", parent=self.main_map_parent)
        self.main_map.transform = PROJ4Transform(proj_info['proj4_str'])
        self.proxy_nodes = {}

        self._borders_color_idx = 0
        self.borders = NEShapefileLines(self.border_shapefile, double=True,
                                        color=self._color_choices[self._borders_color_idx], parent=self.main_map)
        self.borders.transform = STTransform(translate=(0, 0, 40))
        self.conus_states = NEShapefileLines(self.conus_states_shapefile, double=True,
                                             color=self._color_choices[self._borders_color_idx], parent=self.main_map)
        self.conus_states.transform = STTransform(translate=(0, 0, 45))

        self._latlon_grid_color_idx = 1
        self.latlon_grid = self._init_latlon_grid_layer(color=self._color_choices[self._latlon_grid_color_idx])
        self.latlon_grid.transform = STTransform(translate=(0, 0, 45))

        self.create_test_image()

        # Make the camera center on Guam
        # center = (144.8, 13.5)
        center = center or proj_info["default_center"]
        width = proj_info["default_width"] / 2.
        height = proj_info["default_height"] / 2.
        ll_xy = self.borders.transforms.get_transform(map_to="scene").map(
            [(center[0] - width, center[1] - height)])[0][:2]
        ur_xy = self.borders.transforms.get_transform(map_to="scene").map(
            [(center[0] + width, center[1] + height)])[0][:2]
        self.main_view.camera.rect = Rect(ll_xy, (ur_xy[0] - ll_xy[0], ur_xy[1] - ll_xy[1]))

    def create_test_image(self):
        proj4_str = os.getenv("SIFT_DEBUG_IMAGE_PROJ", None)
        if proj4_str is None:
            return
        shape = (2000, 2000)
        fake_data = np.zeros(shape, np.float32) + 0.5
        fake_data[:5, :] = 1.
        fake_data[-5:, :] = 1.
        fake_data[:, :5] = 1.
        fake_data[:, -5:] = 1.
        cell_size = 1000
        origin_x = -shape[1] / 2. * cell_size
        origin_y = shape[0] / 2. * cell_size

        image = TiledGeolocatedImage(
            fake_data,
            origin_x,
            origin_y,
            cell_size,
            cell_size,
            name="Test Image",
            clim=(0., 1.),
            gamma=1.,
            interpolation='nearest',
            method='tiled',
            cmap=self.document.find_colormap('grays'),
            double=False,
            texture_shape=DEFAULT_TEXTURE_SHAPE,
            wrap_lon=False,
            parent=self.main_map,
            projection=proj4_str,
        )
        image.transform = PROJ4Transform(proj4_str, inverse=True)
        image.transform *= STTransform(translate=(0, 0, -50.0))
        self._test_img = image

    def set_projection(self, projection_name, proj_info, center=None):
        self.main_map.transform = PROJ4Transform(proj_info['proj4_str'])
        center = center or proj_info["default_center"]
        width = proj_info["default_width"] / 2.
        height = proj_info["default_height"] / 2.
        ll_xy = self.borders.transforms.get_transform(map_to="scene").map(
            [(center[0] - width, center[1] - height)])[0][:2]
        ur_xy = self.borders.transforms.get_transform(map_to="scene").map(
            [(center[0] + width, center[1] + height)])[0][:2]
        self.main_view.camera.rect = Rect(ll_xy, (ur_xy[0] - ll_xy[0], ur_xy[1] - ll_xy[1]))
        for img in self.image_elements.values():
            if hasattr(img, 'determine_reference_points'):
                img.determine_reference_points()
        self.on_view_change(None)

    def _init_latlon_grid_layer(self, color=None, resolution=5.):
        """Create a series of line segments representing latitude and longitude lines.

        :param resolution: number of degrees between lines
        """
        lons = np.arange(-180., 180. + resolution, resolution, dtype=np.float32)
        lats = np.arange(-90., 90. + resolution, resolution, dtype=np.float32)

        # One long line of lawn mower pattern (lon lines, then lat lines)
        points = np.empty((lons.shape[0] * lats.shape[0] * 2, 2), np.float32)
        LOG.debug("Generating longitude lines...")
        for idx, lon_point in enumerate(lons):
            points[idx * lats.shape[0]:(idx + 1) * lats.shape[0], 0] = lon_point
            if idx % 2 == 0:
                points[idx * lats.shape[0]:(idx + 1) * lats.shape[0], 1] = lats
            else:
                points[idx * lats.shape[0]:(idx + 1) * lats.shape[0], 1] = lats[::-1]
        start_idx = lons.shape[0] * lats.shape[0]
        LOG.debug("Generating latitude lines...")
        for idx, lat_point in enumerate(lats[::-1]):
            points[start_idx + idx * lons.shape[0]:start_idx + (idx + 1) * lons.shape[0], 1] = lat_point
            if idx % 2 == 0:
                points[start_idx + idx * lons.shape[0]:start_idx + (idx + 1) * lons.shape[0], 0] = lons
            else:
                points[start_idx + idx * lons.shape[0]:start_idx + (idx + 1) * lons.shape[0], 0] = lons[::-1]

        # Repeat for "second" size of the earth (180 to 540)
        offset = 360  # box_x[lons.shape[0] - 1] - box_x[0]
        points2 = np.empty((points.shape[0] * 2, 2), dtype=np.float32)
        points2[:points.shape[0], :] = points
        points2[points.shape[0]:, :] = points
        points2[points.shape[0]:, 0] += offset

        # return Line(pos=points2, connect="segments", color=color, parent=self.main_map)
        return Line(pos=points2, connect="strip", color=color, parent=self.main_map)
        # return Line(pos=points, connect="strip", color=color, parent=self.main_map)

    def on_mouse_press_point(self, event):
        """Handle mouse events that mean we are using the point probe.
        """
        if event.handled:
            return
        modifiers = event.mouse_event.modifiers
        if (event.button == 2 and not modifiers) or (self._current_tool == Tool.POINT_PROBE and event.button == 1):
            buffer_pos = event.sources[0].transforms.get_transform().map(event.pos)
            # FIXME: We should be able to use the main_map object to do the transform
            #  but it doesn't work (waiting on vispy developers)
            # map_pos = self.main_map.transforms.get_transform().imap(buffer_pos)
            map_pos = self.borders.transforms.get_transform().imap(buffer_pos)
            if np.any(np.abs(map_pos[:2]) > 1e25):
                LOG.error("Invalid point probe location")
                return
            self.newPointProbe.emit(DEFAULT_POINT_PROBE, tuple(map_pos[:2]))

    def on_mouse_press_region(self, event):
        """Handle mouse events that mean we are using the point probe.
        """
        if event.handled:
            return
        modifiers = event.mouse_event.modifiers
        if (event.button == 2 and modifiers == (SHIFT,)) or (
                self._current_tool == Tool.REGION_PROBE and event.button == 1):
            buffer_pos = event.sources[0].transforms.get_transform().map(event.pos)
            map_pos = self.borders.transforms.get_transform().imap(buffer_pos)
            if np.any(np.abs(map_pos[:2]) > 1e25):
                LOG.error("Invalid region probe location")
                return
            if self.pending_polygon.add_point(event.pos[:2], map_pos[:2], 60):
                points = self.pending_polygon.points + [self.pending_polygon.points[0]]
                self.clear_pending_polygon()
                self.newProbePolygon.emit(self.layer_set.top_layer_uuid(), points)

    def clear_pending_polygon(self):
        for marker in self.pending_polygon.markers:
            # Remove the marker from the scene graph
            marker.parent = None
        # Reset the pending polygon object
        self.pending_polygon.reset()

    def remove_polygon(self, name=None):
        """Remove a polygon from the SGM or clear the pending polygon if it exists.
        """
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
        edge_color = kwargs.get("edge_color", np.array([1.0, 0.5, 0.5, 1.]))
        face_color = kwargs.get("face_color", np.array([0.5, 0., 0., 1.]))
        if len(xy_pos) == 2:
            xy_pos = [xy_pos[0], xy_pos[1], z]

        probe_kwargs = {
            'symbol': 'disc',
            'pos': np.array([xy_pos]),
            'face_color': face_color,
            'edge_color': edge_color,
            'size': 18.,
            'edge_width': 3.,
        }

        if probe_name not in self.point_probes and xy_pos is None:
            raise ValueError("Probe '{}' does not exist".format(probe_name))
        elif probe_name not in self.point_probes:
            # point_visual = FakeMarker(parent=self.main_map, symbol="x", pos=np.array([xy_pos]), color=color)
            point_visual = Markers(parent=self.main_map, name=probe_name, **probe_kwargs)
            self.point_probes[probe_name] = point_visual
        else:
            point_visual = self.point_probes[probe_name]
            # point_visual.set_point(xy_pos)
            point_visual.set_data(**probe_kwargs)

        # set the Point visible or not
        point_visual.visible = state

    def on_new_polygon(self, probe_name, points, **kwargs):
        kwargs.setdefault("color", (1.0, 0.0, 1.0, 0.5))
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

    def update(self):
        return self.main_canvas.update()

    def cycle_borders_color(self):
        self._borders_color_idx = (self._borders_color_idx + 1) % len(self._color_choices)
        if self._borders_color_idx + 1 == len(self._color_choices):
            self.borders.visible = False
            self.conus_states.visible = False
        else:
            self.borders.set_data(color=self._color_choices[self._borders_color_idx])
            self.borders.visible = True
            self.conus_states.set_data(color=self._color_choices[self._borders_color_idx])
            self.conus_states.visible = True

    def cycle_grid_color(self):
        self._latlon_grid_color_idx = (self._latlon_grid_color_idx + 1) % len(self._color_choices)
        if self._latlon_grid_color_idx + 1 == len(self._color_choices):
            self.latlon_grid.visible = False
        else:
            self.latlon_grid.set_data(color=self._color_choices[self._latlon_grid_color_idx])
            self.latlon_grid.visible = True

    def change_tool(self, name):
        prev_tool = self._current_tool
        if name == prev_tool:
            # it's the same tool
            return

        self._current_tool = name

        # Set the cursor
        if name == Tool.PAN_ZOOM:
            # self.main_canvas.native.setCursor(QCursor(QPixmap("py/uwsift/ui/cursors/noun_275_cc.png")))
            # self.main_canvas.native.setCursor(QCursor(Qt.SizeAllCursor))
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

    def set_colormap(self, colormap, uuid=None):
        colormap = self.document.find_colormap(colormap)

        uuids = uuid
        if uuid is None:
            uuids = self.image_elements.keys()
        elif not isinstance(uuid, (list, tuple)):
            uuids = [uuid]

        for uuid in uuids:
            layer = self.image_elements[uuid]
            if isinstance(layer, TiledGeolocatedImage):
                self.image_elements[uuid].cmap = colormap
            else:
                self.image_elements[uuid].color = colormap

    def set_color_limits(self, clims, uuid=None):
        """Update the color limits for the specified UUID
        """
        uuids = uuid
        if uuid is None:
            uuids = self.image_elements.keys()
        elif not isinstance(uuid, (list, tuple)):
            uuids = [uuid]

        for uuid in uuids:
            element = self.image_elements.get(uuid, None)
            if element is not None:
                self.image_elements[uuid].clim = clims

    def set_gamma(self, gamma, uuid):
        uuids = uuid
        if uuid is None:
            uuids = self.image_elements.keys()
        elif not isinstance(uuid, (list, tuple)):
            uuids = [uuid]

        for uuid in uuids:
            element = self.image_elements.get(uuid, None)
            if element is not None and hasattr(element, 'gamma'):
                self.image_elements[uuid].gamma = gamma

    def change_layers_colormap(self, change_dict):
        for uuid, cmapid in change_dict.items():
            LOG.info('changing {} to colormap {}'.format(uuid, cmapid))
            self.set_colormap(cmapid, uuid)

    def change_layers_color_limits(self, change_dict):
        for uuid, clims in change_dict.items():
            LOG.info('changing {} to color limits {}'.format(uuid, clims))
            self.set_color_limits(clims, uuid)

    def change_layers_gamma(self, change_dict):
        for uuid, gamma in change_dict.items():
            LOG.info('changing {} to gamma {}'.format(uuid, gamma))
            self.set_gamma(gamma, uuid)

    def change_layers_image_kind(self, change_dict):
        for uuid, new_pz in change_dict.items():
            LOG.info('changing {} to kind {}'.format(uuid, new_pz.kind.name))
            self.add_basic_layer(None, uuid, new_pz)

    def add_contour_layer(self, layer: DocBasicLayer, p: Presentation, overview_content: np.ndarray):
        verts = overview_content[:, :2]
        connects = overview_content[:, 2].astype(np.bool)
        level_indexes = overview_content[:, 3]
        level_indexes = level_indexes[~np.isnan(level_indexes)].astype(np.int)
        levels = layer["contour_levels"]
        cmap = self.document.find_colormap(p.colormap)

        proj4_str = layer[Info.PROJ]
        parent = self.proxy_nodes.get(proj4_str)
        if parent is None:
            parent = ContourGroupNode(parent=self.main_map)
            parent.transform = PROJ4Transform(layer[Info.PROJ], inverse=True)
            self.proxy_nodes[proj4_str] = parent

        contour_visual = PrecomputedIsocurve(verts, connects, level_indexes,
                                             levels=levels, color_lev=cmap,
                                             clim=p.climits,
                                             parent=parent,
                                             name=str(layer[Info.UUID]))
        contour_visual.transform *= STTransform(translate=(0, 0, -50.0))
        self.image_elements[layer[Info.UUID]] = contour_visual
        self.layer_set.add_layer(contour_visual)
        self.on_view_change(None)

    def add_basic_layer(self, new_order: tuple, uuid: UUID, p: Presentation):
        layer = self.document[uuid]
        # create a new layer in the imagelist
        if not layer.is_valid:
            LOG.warning('unable to add an invalid layer, will try again later when layer changes')
            return
        if layer[Info.UUID] in self.image_elements:
            image = self.image_elements[layer[Info.UUID]]
            if p.kind == Kind.CONTOUR and isinstance(image, PrecomputedIsocurve):
                LOG.warning("Contour layer already exists in scene")
                return
            if p.kind == Kind.IMAGE and isinstance(image, TiledGeolocatedImage):
                LOG.warning("Image layer already exists in scene")
                return
            # we already have an image layer for it and it isn't what we want
            # remove the existing image object and create the proper type now
            image.parent = None
            del self.image_elements[layer[Info.UUID]]

        overview_content = self.workspace.get_content(layer.uuid, kind=p.kind)
        if p.kind == Kind.CONTOUR:
            return self.add_contour_layer(layer, p, overview_content)

        image = TiledGeolocatedImage(
            overview_content,
            layer[Info.ORIGIN_X],
            layer[Info.ORIGIN_Y],
            layer[Info.CELL_WIDTH],
            layer[Info.CELL_HEIGHT],
            name=str(uuid),
            clim=p.climits,
            gamma=p.gamma,
            interpolation='nearest',
            method='tiled',
            cmap=self.document.find_colormap(p.colormap),
            double=False,
            texture_shape=DEFAULT_TEXTURE_SHAPE,
            wrap_lon=False,
            parent=self.main_map,
            projection=layer[Info.PROJ],
        )
        image.transform = PROJ4Transform(layer[Info.PROJ], inverse=True)
        image.transform *= STTransform(translate=(0, 0, -50.0))
        self.image_elements[uuid] = image
        self.layer_set.add_layer(image)
        image.determine_reference_points()
        self.on_view_change(None)

    def add_composite_layer(self, new_order: tuple, uuid: UUID, p: Presentation):
        layer = self.document[uuid]
        LOG.debug("SceneGraphManager.add_composite_layer %s" % repr(layer))
        if not layer.is_valid:
            LOG.info('unable to add an invalid layer, will try again later when layer changes')
            return
        if p.kind == Kind.RGB:
            dep_uuids = r, g, b = [c.uuid if c is not None else None for c in [layer.r, layer.g, layer.b]]
            overview_content = list(self.workspace.get_content(cuuid, kind=Kind.IMAGE) for cuuid in dep_uuids)
            uuid = layer.uuid
            LOG.debug("Adding composite layer to Scene Graph Manager with UUID: %s", uuid)
            self.image_elements[uuid] = element = RGBCompositeLayer(
                overview_content,
                layer[Info.ORIGIN_X],
                layer[Info.ORIGIN_Y],
                layer[Info.CELL_WIDTH],
                layer[Info.CELL_HEIGHT],
                name=str(uuid),
                clim=p.climits,
                gamma=p.gamma,
                interpolation='nearest',
                method='tiled',
                cmap=None,
                double=False,
                texture_shape=DEFAULT_TEXTURE_SHAPE,
                wrap_lon=False,
                parent=self.main_map,
                projection=layer[Info.PROJ],
            )
            element.transform = PROJ4Transform(layer[Info.PROJ], inverse=True)
            element.transform *= STTransform(translate=(0, 0, -50.0))
            self.composite_element_dependencies[uuid] = dep_uuids
            self.layer_set.add_layer(element)
            if new_order:
                self.layer_set.set_layer_order(new_order)
            self.on_view_change(None)
            element.determine_reference_points()
            self.update()
            return True
        elif p.kind in [Kind.COMPOSITE, Kind.IMAGE]:
            # algebraic layer
            return self.add_basic_layer(new_order, uuid, p)

    def change_composite_layers(self, new_order: tuple, uuid_list: list, presentations: list):
        for uuid, presentation in zip(uuid_list, presentations):
            self.change_composite_layer(None, uuid, presentation)
        # set the order after we've updated and created all the new layers
        if new_order:
            self.layer_set.set_layer_order(new_order)

    def change_composite_layer(self, new_order: tuple, uuid: UUID, presentation: Presentation):
        layer = self.document[uuid]
        if presentation.kind == Kind.RGB:
            if layer.uuid in self.image_elements:
                if layer.is_valid:
                    # RGB selection has changed, rebuild the layer
                    LOG.debug("Changing existing composite layer to Scene Graph Manager with UUID: %s", layer.uuid)
                    dep_uuids = r, g, b = [c.uuid if c is not None else None for c in [layer.r, layer.g, layer.b]]
                    overview_content = list(self.workspace.get_content(cuuid) for cuuid in dep_uuids)
                    self.composite_element_dependencies[layer.uuid] = dep_uuids
                    elem = self.image_elements[layer.uuid]
                    elem.set_channels(overview_content,
                                      cell_width=layer[Info.CELL_WIDTH],
                                      cell_height=layer[Info.CELL_HEIGHT],
                                      origin_x=layer[Info.ORIGIN_X],
                                      origin_y=layer[Info.ORIGIN_Y])
                    elem.init_overview(overview_content)
                    elem.clim = presentation.climits
                    elem.gamma = presentation.gamma
                    self.on_view_change(None)
                    elem.determine_reference_points()
                else:
                    # layer is no longer valid and has to be removed
                    LOG.debug("Purging composite ")
                    self.purge_layer(layer.uuid)
                self.update()
            else:
                if layer.is_valid:
                    # Add this now valid layer
                    self.add_composite_layer(new_order, layer.uuid, presentation)
                else:
                    LOG.info('unable to add an invalid layer, will try again later when layer changes')
                    return
        else:
            raise ValueError("Unknown or unimplemented composite type")

    def remove_layer(self, new_order: tuple, uuids_removed: tuple, row: int, count: int):
        """
        remove (disable) a layer, though this may be temporary due to a move.
        wait for purge to truly flush out this puppy
        :param new_order:
        :param uuid_removed:
        :return:
        """
        for uuid_removed in uuids_removed:
            self.set_layer_visible(uuid_removed, False)
        # XXX: Used to rebuild_all instead of just update, is that actually needed?
        # self.rebuild_all()

    def _remove_layer(self, *args, **kwargs):
        self.remove_layer(*args, **kwargs)
        # when removing the layer is the only operation being performed then update when we are done
        self.update()

    def purge_layer(self, uuid_removed: UUID):
        """
        Layer has been purged from document (no longer used anywhere) - flush it all out
        :param uuid_removed: UUID of the layer that is to be removed
        :return:
        """
        self.set_layer_visible(uuid_removed, False)
        if uuid_removed in self.image_elements:
            image_layer = self.image_elements[uuid_removed]
            image_layer.parent = None
            del self.image_elements[uuid_removed]
            LOG.info("layer {} purge from scenegraphmanager".format(uuid_removed))
        else:
            LOG.debug("Layer {} already purged from Scene Graph".format(uuid_removed))

    def _purge_layer(self, *args, **kwargs):
        res = self.purge_layer(*args, **kwargs)
        # when purging the layer is the only operation being performed then update when we are done
        self.update()
        return res

    def change_layers_visibility(self, layers_changed: dict):
        for uuid, visible in layers_changed.items():
            self.set_layer_visible(uuid, visible)

    def rebuild_new_layer_set(self, new_set_number: int, new_prez_order: DocLayerStack, new_anim_order: list):
        self.rebuild_all()
        # raise NotImplementedError("layer set change not implemented in SceneGraphManager")

    def _connect_doc_signals(self, document):
        document.didReorderLayers.connect(self._rebuild_layer_order)  # current layer set changed z/anim order
        document.didAddBasicLayer.connect(self.add_basic_layer)  # layer added to one or more layer sets
        document.didAddCompositeLayer.connect(
            self.add_composite_layer)  # layer derived from other layers (either basic or composite themselves)
        document.didRemoveLayers.connect(self._remove_layer)  # layer removed from current layer set
        document.willPurgeLayer.connect(self._purge_layer)  # layer removed from document
        document.didSwitchLayerSet.connect(self.rebuild_new_layer_set)
        document.didChangeColormap.connect(self.change_layers_colormap)
        document.didChangeLayerVisibility.connect(self.change_layers_visibility)
        document.didReorderAnimation.connect(self._rebuild_frame_order)
        document.didChangeComposition.connect(self.change_composite_layer)
        document.didChangeCompositions.connect(self.change_composite_layers)
        document.didChangeColorLimits.connect(self.change_layers_color_limits)
        document.didChangeGamma.connect(self.change_layers_gamma)
        document.didChangeImageKind.connect(self.change_layers_image_kind)

    def set_frame_number(self, frame_number=None):
        self.layer_set.next_frame(None, frame_number)

    def set_layer_visible(self, uuid, visible=None):
        image = self.image_elements.get(uuid, None)
        if image is None:
            return
        image.visible = not image.visible if visible is None else visible

    def rebuild_layer_order(self, new_layer_index_order, *args, **kwargs):
        """
        layer order has changed; shift layers around.
        an empty list is sent if the whole layer order has been changed
        :param change:
        :return:
        """
        # TODO this is the lazy implementation, eventually just change z order on affected layers
        self.layer_set.set_layer_order(self.document.current_layer_uuid_order)

    def _rebuild_layer_order(self, *args, **kwargs):
        res = self.rebuild_layer_order(*args, **kwargs)
        self.update()
        return res

    def rebuild_frame_order(self, uuid_list: list, *args, **kwargs):
        LOG.debug('setting SGM new frame order to {0!r:s}'.format(uuid_list))
        self.layer_set.frame_order = uuid_list

    def _rebuild_frame_order(self, *args, **kwargs):
        res = self.rebuild_frame_order(*args, **kwargs)
        # when purging the layer is the only operation being performed then update when we are done
        self.update()
        return res

    def rebuild_presentation(self, presentation_info: dict):
        # refresh our presentation info
        # presentation_info = self.document.current_layer_set
        for uuid, layer_prez in presentation_info.items():
            self.set_colormap(layer_prez.colormap, uuid=uuid)
            self.set_color_limits(layer_prez.climits, uuid=uuid)
            self.set_layer_visible(uuid, visible=layer_prez.visible)
            # FUTURE, if additional information is added to the presentation tuple, you must also update it here

    def rebuild_all(self, *args, **kwargs):
        """
        resynchronize the scenegraph to the document content
        This includes creating elements for any newly-valid layers,
        removing elements for no-longer-valid layers, and
        making the display order, visibility, and animation order match the document
        """
        # get the list of layers which are valid, and either visible or in the animation order
        doc_layers = list(self.document.active_layer_order)
        presentation_info = tuple(p for (p, l) in doc_layers)
        active_layers = tuple(l for (p, l) in doc_layers)
        active_uuids = set(x.uuid for x in active_layers)
        active_lookup = dict((x.uuid, x) for x in active_layers)
        prez_lookup = dict((x.uuid, x) for x in presentation_info)

        uuids_w_elements = set(self.image_elements.keys())
        # get set of valid layers not having elements and invalid layers having elements
        inconsistent_uuids = uuids_w_elements ^ active_uuids

        # current_uuid_order = self.document.current_layer_uuid_order
        current_uuid_order = tuple(p.uuid for p in presentation_info)

        remove_elements = []
        for uuid in inconsistent_uuids:
            if uuid in active_lookup and active_lookup[uuid].is_valid:
                layer = active_lookup[uuid]
                # create elements for layers which have transitioned to a valid state
                LOG.debug('creating deferred element for layer %s' % layer.uuid)
                if layer.kind in [Kind.COMPOSITE, Kind.RGB]:
                    # create an invisible element with the RGB
                    self.change_composite_layer(current_uuid_order, layer, prez_lookup[uuid])
                else:
                    # FIXME this was previously a NotImplementedError
                    LOG.warning('unable to create deferred scenegraph element for %s' % repr(layer))
            else:
                # remove elements for layers which are no longer valid
                remove_elements.append(uuid)

        # get info on the new order
        self.layer_set.set_layer_order(current_uuid_order)
        self.layer_set.frame_order = self.document.current_animation_order
        self.rebuild_presentation(prez_lookup)

        for elem in remove_elements:
            self.purge_layer(elem)

        self.update()

    def on_view_change(self, scheduler):
        """Simple event handler for when we need to reassess image layers.
        """
        # Stop the timer so it doesn't continuously call this slot
        if scheduler:
            scheduler.stop()

        def _assess(uuid, child):
            need_retile, preferred_stride, tile_box = child.assess()
            if need_retile:
                self.start_retiling_task(uuid, preferred_stride, tile_box)

        current_visible_layers = [p.uuid for (p, l) in self.document.active_layer_order if p.visible]
        current_invisible_layers = set(self.image_elements.keys()) - set(current_visible_layers)

        def _assess_if_active(uuid):
            element = self.image_elements.get(uuid, None)
            if element is not None and hasattr(element, 'assess'):
                _assess(uuid, element)

        for uuid in current_visible_layers:
            _assess_if_active(uuid)
        # update contours
        for node in self.proxy_nodes.values():
            node.on_view_change()
        # update invisible layers
        for uuid in current_invisible_layers:
            _assess_if_active(uuid)

    def start_retiling_task(self, uuid, preferred_stride, tile_box):
        LOG.debug("Scheduling retile for child with UUID: %s", uuid)
        self.queue.add(str(uuid) + "_retile", self._retile_child(uuid, preferred_stride, tile_box),
                       'Retile calculations for image layer ' + str(uuid), interactive=True)

    def _retile_child(self, uuid, preferred_stride, tile_box):
        LOG.debug("Retiling child with UUID: '%s'", uuid)
        yield {TASK_DOING: 'Re-tiling', TASK_PROGRESS: 0.0}
        if uuid not in self.composite_element_dependencies:
            child = self.image_elements[uuid]
            data = self.workspace.get_content(uuid, lod=preferred_stride)
            yield {TASK_DOING: 'Re-tiling', TASK_PROGRESS: 0.5}
            # FIXME: Use LOD instead of stride and provide the lod to the workspace
            data = data[::preferred_stride[0], ::preferred_stride[1]]
            tiles_info, vertices, tex_coords = child.retile(data, preferred_stride, tile_box)
            yield {TASK_DOING: 'Re-tiling', TASK_PROGRESS: 1.0}
            self.didRetilingCalcs.emit(uuid, preferred_stride, tile_box, tiles_info, vertices, tex_coords)
        else:
            child = self.image_elements[uuid]
            data = [self.workspace.get_content(d_uuid, lod=preferred_stride) for d_uuid in
                    self.composite_element_dependencies[uuid]]
            yield {TASK_DOING: 'Re-tiling', TASK_PROGRESS: 0.5}
            # FIXME: Use LOD instead of stride and provide the lod to the workspace
            data = [
                d[::int(preferred_stride[0] / factor), ::int(preferred_stride[1] / factor)] if d is not None else None
                for factor, d in zip(child._channel_factors, data)]
            tiles_info, vertices, tex_coords = child.retile(data, preferred_stride, tile_box)
            yield {TASK_DOING: 'Re-tiling', TASK_PROGRESS: 1.0}
            self.didRetilingCalcs.emit(uuid, preferred_stride, tile_box, tiles_info, vertices, tex_coords)
        self.workspace.bgnd_task_complete()  # FUTURE: consider a threading context manager for this??

    def _set_retiled(self, uuid, preferred_stride, tile_box, tiles_info, vertices, tex_coords):
        """Slot to take data from background thread and apply it to the layer living in the image layer.
        """
        child = self.image_elements.get(uuid, None)
        if child is None:
            LOG.warning('unable to find uuid %s in image_elements' % uuid)
            return
        child.set_retiled(preferred_stride, tile_box, tiles_info, vertices, tex_coords)
        child.update()

    def on_layer_visible_toggle(self, visible):
        pass

    def on_layer_change(self, event):
        pass

    def on_data_loaded(self, event):
        pass
