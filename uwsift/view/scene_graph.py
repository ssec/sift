#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scene_graph.py
~~~~~~~~~~~~~~

PURPOSE
Provides a SceneGraphManager to handle display of visuals, in this case satellite imaging data,
latitude/longitude lines and coastlines.

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

REFERENCES
http://api.vispy.org/en/latest/scene.html

REQUIRES


:author: R.K.Garcia <rayg@ssec.wisc.edu>
:copyright: 2014 by University of Wisconsin Regents, see AUTHORS for more details
:license: GPLv3, see LICENSE for more details
"""
__docformat__ = 'reStructuredText'
__author__ = 'davidh'

import logging
import os
from enum import Enum
from numbers import Number
from typing import Dict, Optional
from uuid import UUID

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal, Qt
from PyQt5.QtGui import QCursor
from pyresample import AreaDefinition
from vispy import app
from vispy import scene
from vispy.geometry import Rect
from vispy.gloo.util import _screenshot
from vispy.scene.visuals import Image, Markers, Polygon, Compound, Line
from vispy.util.keys import SHIFT
from vispy.visuals import LineVisual
from vispy.visuals.transforms import STTransform, MatrixTransform, \
    ChainTransform

from uwsift import USE_TILED_GEOLOCATED_IMAGES
from uwsift import config
from uwsift.common import DEFAULT_ANIMATION_DELAY, Info, Kind, Tool, \
    Presentation, \
    LATLON_GRID_DATASET_NAME, BORDERS_DATASET_NAME
from uwsift.model.area_definitions_manager import AreaDefinitionsManager
from uwsift.model.document import DocLayerStack, DocBasicDataset, Document
from uwsift.model.layer_item import LayerItem
from uwsift.model.product_dataset import ProductDataset
from uwsift.model.time_manager import TimeManager
from uwsift.queue import TASK_DOING, TASK_PROGRESS
from uwsift.util import get_package_data_dir
from uwsift.view.cameras import PanZoomProbeCamera
from uwsift.view.probes import DEFAULT_POINT_PROBE
from uwsift.view.transform import PROJ4Transform
from uwsift.view.visuals import (NEShapefileLines, TiledGeolocatedImage,
                                 MultiChannelImage,
                                 RGBCompositeLayer,
                                 PrecomputedIsocurve, Lines)
from uwsift.workspace.utils.metadata_utils import (
    map_point_style_to_marker_kwargs, get_point_style_by_name)

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


class AnimationController(object):
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

        doc = parent.document
        self.time_manager = TimeManager(doc.data_layer_collection, self._animation_speed)

        self._animation_timer = app.Timer(self._animation_speed / 1000.0)
        self._animation_timer.connect(self.step)

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

    # FIXME(mk): this is never used anyway?
    def _set_visible_node(self, node):
        """Set all nodes to invisible except for the `event.added` node.
        """
        for child in self._layers.values():
            with child.events.blocker():
                if child is node.added:
                    child.visible = True
                else:
                    child.visible = False

    # FIXME(mk): not in use anymore
    def _set_visible_child(self, frame_number):
        for idx, uuid in enumerate(self._frame_order):
            child = self._layers[uuid]
            # not sure if this is actually doing anything
            with child.events.blocker():
                if idx == frame_number:
                    child.visible = True
                else:
                    child.visible = False

    # FIXME(mk): not in use anymore
    def _set_uuid_visibility(self, uuid: UUID, visible: bool) -> None:
        """
        Convenience function to set a uuid's visibility.
        :param uuid: UUID of layer whose visibility is to be set.
        :param visible: Flag indicating wether uuid is supposed to be visible or not.
        """
        child = self._layers[uuid]
        with child.events.blocker():
            if visible:
                child.visible = True
            else:
                child.visible = False

    def _set_visible_from_data_layers(self) -> None:
        """
        Set layer (original SIFT layer meant here) visibility per data_layer according to matched
        datetime.

        """
        for pfkey, data_layer in self.time_manager.collection.data_layers.items():
            # Set all uuids in the current data layer to invisible
            # If a time was matched turn that time's uuid visible
            data_layer_uuids = list(data_layer.timeline.values())
            self.parent.document.toggle_layer_visibility(data_layer_uuids, visible=False)
            layer_uuid = data_layer.t_matched_uuid()
            layer_uuid_idx = data_layer.get_index_of_uuid(layer_uuid)
            self._frame_number = layer_uuid_idx
            if layer_uuid is not None:
                self.parent.document.toggle_layer_visibility([layer_uuid])

    def jump(self, index):
        self.time_manager.jump(index)
        self._set_visible_from_data_layers()

    def step(self, event=None, backwards=False):
        # Slot that triggers the time manager's tick in the relevant direction
        # and subsequently makes the appropriate layers visible.
        self.time_manager.tick(backwards=backwards)
        self._set_visible_from_data_layers()

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
        # self._set_visible_child(frame)
        self._frame_number = frame
        self.parent.update()
        if self._frame_change_cb is not None and lfo:

            uuid = self._frame_order[self._frame_number]
            self._frame_change_cb((self._frame_number, lfo, self._animating, uuid))

    def update_time_manager_collection(self, coll):
        self.time_manager.collection.didUpdateCollection.emit()
        self._set_visible_from_data_layers()


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

    # TODO(ar) REVIEW: distinction between class and member/instance
    #  variables seems random (see below)
    document = None  # Document object we work with
    workspace = None  # where we get data arrays from
    queue = None  # background jobs go here

    borders_shapefiles = None  # political map overlay
    texture_shape = None
    polygon_probes = None
    point_probes = None

    layer_nodes = None  # {layer_uuid: layer_node}
    dataset_nodes = None  # {dataset_uuid: dataset_node}
    composite_element_dependencies = None  # {dataset_uuid:set-of-dependent-uuids}
    datasets = None
    colormaps = None
    animation_controller = None

    _current_tool = None
    _color_choices = None

    # FIXME: many more undocumented member variables

    didRetilingCalcs = pyqtSignal(object, object, object, object, object, object)
    didChangeFrame = pyqtSignal(tuple)
    didChangeLayerVisibility = pyqtSignal(dict)  # similar to document didChangeLayerVisibility
    newPointProbe = pyqtSignal(str, tuple)
    newProbePolygon = pyqtSignal(object, object)

    def __init__(self, doc, workspace, queue,
                 borders_shapefiles: list = None, states_shapefile=None,
                 parent=None, texture_shape=(4, 16), center=None):
        super(SceneGraphManager, self).__init__(parent)
        self.didRetilingCalcs.connect(self._set_retiled)

        # Parent should be the Qt widget that this GLCanvas belongs to
        self.document = doc
        self.workspace = workspace
        self.queue = queue
        self.borders_shapefiles = borders_shapefiles or \
            [DEFAULT_SHAPE_FILE, DEFAULT_STATES_SHAPE_FILE]
        self.texture_shape = texture_shape
        self.polygon_probes = {}
        self.point_probes = {}

        self.layer_nodes = {}
        self.dataset_nodes = {}
        self.latlon_grid_node = None  # noqa
        self.borders_nodes = []

        self.composite_element_dependencies = {}
        self.animation_controller = AnimationController(self, frame_change_cb=self.frame_changed)
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
        self._latlon_grid_color_idx = 1
        self._borders_color_idx = 0

        # TODO(ar) REVIEW: distinction between class and member/instance
        # variables seems random (see above)
        # These following three were initialized in self.setup_initial_canvas()
        # thus indirectly as instance/member variables.
        # Why aren't they class variables like 'document', 'workspace', ...?
        self.main_view = None
        self.main_canvas = None
        self.pz_camera = None

        self.setup_initial_canvas(center)
        self.pending_polygon = PendingPolygon(self.main_map)

    def get_screenshot_array(self, frame_range=None):
        """Get numpy arrays representing the current canvas."""
        if frame_range is None:
            self.main_canvas.on_draw(None)
            return [(self.animation_controller.top_layer_uuid(), _screenshot())]
        s, e = frame_range

        # reset the view once we are done
        c = self.animation_controller.current_frame
        images = []
        for i in range(s, e + 1):
            self.set_frame_number(i)
            self.update()
            self.main_canvas.on_draw(None)
            u = self.animation_controller.frame_order[i] if self.animation_controller.frame_order else None
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
            uuids = self.animation_controller.frame_order
            # note that all the layers in the layer_order but the current one are now invisible
            vis = dict((u, u == frame_info[3]) for u in uuids)
            self.didChangeLayerVisibility.emit(vis)

    def setup_initial_canvas(self, center=None):
        self.main_canvas = SIFTMainMapCanvas(parent=self.parent())
        self.main_view = self.main_canvas.central_widget.add_view(name="MainView")

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
        self.main_map = MainMap(name="MainMap", parent=self.main_map_parent)
        self.proxy_nodes = {}

        self.create_test_image()

        # Make the camera center on Guam
        # center = (144.8, 13.5)
        ##proj_info = self.document.projection_info()
        ##self._set_projection(proj_info)

        area_def = self.document.area_definition()
        self._set_projection(area_def)

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
            method='subdivide',
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

    def set_projection(self, area_display_name: str, center=None):
        area_def = AreaDefinitionsManager.area_def_by_name(area_display_name)
        assert area_def is not None
        self._set_projection(area_def, center)

        for dataset_node in self.dataset_nodes.values():
            if hasattr(dataset_node, 'determine_reference_points'):
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
        assert center is None or self.latlon_grid_node is not None

        if center:
            # FIXME: We should be able to use the main_map object to do the
            #  transform but it doesn't work (waiting on vispy developers)
            # mapped_center = self.main_map.transforms\
            #    .get_transform(map_to="scene").map([center])[0][:2]
            mapped_center = self.latlon_grid_node.transforms \
                .get_transform(map_to="scene").map([center])[0][:2]
            ll_xy += mapped_center
            ur_xy += mapped_center

        self.main_view.camera.rect = \
            Rect(ll_xy, (ur_xy[0] - ll_xy[0], ur_xy[1] - ll_xy[1]))


    @staticmethod
    def _create_latlon_grid_points(resolution=5.):
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

        return points2

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
            map_pos = self.latlon_grid_node.transforms\
                .get_transform().imap(buffer_pos)
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
            # FIXME: We should be able to use the main_map object to do the transform
            #  but it doesn't work (waiting on vispy developers)
            # map_pos = self.main_map.transforms.get_transform().imap(buffer_pos)
            map_pos = self.latlon_grid_node.transforms\
                .get_transform().imap(buffer_pos)
            if np.any(np.abs(map_pos[:2]) > 1e25):
                LOG.error("Invalid region probe location")
                return
            if self.pending_polygon.add_point(event.pos[:2], map_pos[:2], 60):
                points = self.pending_polygon.points + [self.pending_polygon.points[0]]
                self.clear_pending_polygon()
                self.newProbePolygon.emit(self.animation_controller.top_layer_uuid(), points)

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
        points = np.array(points, dtype=np.float32) # convert list to NumPy array

        # kwargs.setdefault("color", (1.0, 0.0, 1.0, 0.5))
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

    def update(self):
        return self.main_canvas.update()

    def cycle_borders_color(self):
        self._borders_color_idx = \
            (self._borders_color_idx + 1) % len(self._color_choices)
        if self._borders_color_idx + 1 == len(self._color_choices):
            for borders_node in self.borders_nodes:
                borders_node.visible = False
        else:
            for borders_node in self.borders_nodes:
                borders_node.set_data(
                    color=self._color_choices[self._borders_color_idx])
                borders_node.visible = True

    def cycle_latlon_grid_color(self):
        self._latlon_grid_color_idx = \
            (self._latlon_grid_color_idx + 1) % len(self._color_choices)
        if self._latlon_grid_color_idx + 1 == len(self._color_choices):
            self.latlon_grid_node.visible = False
        else:
            self.latlon_grid_node.set_data(
                color=self._color_choices[self._latlon_grid_color_idx])
            self.latlon_grid_node.visible = True

    def change_tool(self, name: Tool):
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
            uuids = self.dataset_nodes.keys()
        elif not isinstance(uuid, (list, tuple)):
            uuids = [uuid]

        for uuid in uuids:
            dataset_node = self.dataset_nodes[uuid]
            if (isinstance(dataset_node, TiledGeolocatedImage)
                    or isinstance(dataset_node, Image)):
                self.dataset_nodes[uuid].cmap = colormap
            else:
                self.dataset_nodes[uuid].color = colormap

    def set_color_limits(self, clims, uuid=None):
        """Update the color limits for the specified UUID
        """
        uuids = uuid
        if uuid is None:
            uuids = self.dataset_nodes.keys()
        elif not isinstance(uuid, (list, tuple)):
            uuids = [uuid]

        for uuid in uuids:
            dataset_node = self.dataset_nodes.get(uuid, None)
            if dataset_node is not None:
                self.dataset_nodes[uuid].clim = clims

    def set_gamma(self, gamma, uuid):
        uuids = uuid
        if uuid is None:
            uuids = self.dataset_nodes.keys()
        elif not isinstance(uuid, (list, tuple)):
            uuids = [uuid]

        for uuid in uuids:
            dataset_node = self.dataset_nodes.get(uuid, None)
            if dataset_node is not None and hasattr(dataset_node, 'gamma'):
                self.dataset_nodes[uuid].gamma = gamma

    def change_layers_colormap(self, change_dict):
        for uuid, cmapid in change_dict.items():
            LOG.info('changing {} to colormap {}'.format(uuid, cmapid))
            self.set_colormap(cmapid, uuid)

    def change_layers_color_limits(self, change_dict):
        for uuid, clims in change_dict.items():
            LOG.debug('changing {} to color limits {}'.format(uuid, clims))
            self.set_color_limits(clims, uuid)

    def change_layers_gamma(self, change_dict):
        for uuid, gamma in change_dict.items():
            LOG.debug('changing {} to gamma {}'.format(uuid, gamma))
            self.set_gamma(gamma, uuid)

    # This method may be revived again in case CONTOURS should be supported
    # (again)
    # def change_layers_image_kind(self, change_dict):
    #     for uuid, new_pz in change_dict.items():
    #         LOG.info('changing {} to kind {}'.format(uuid, new_pz.kind.name))
    #         self.add_basic_dataset(None, uuid, new_pz)

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
        self.update()

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
            BROWN = 320.
            RED = 300.
            LIGHT_GREEN = 280.
            GREEN = 260.
            LIGHT_BLUE = 240.
            DARK_BLUE = 220.
            PINK = 200.

        center_x, center_y = max_x // 2, max_y // 2
        pixels = [
            {"x": center_x,     "y": center_y,     "color": RainbowValue.RED,         "desc": "center"},
            {"x": center_x - 1, "y": center_y - 1, "color": RainbowValue.GREEN,       "desc": "upper left"},
            {"x": center_x - 1, "y": center_y + 1, "color": RainbowValue.LIGHT_BLUE,  "desc": "bottom left"},
            {"x": center_x - 1, "y": center_y + 2, "color": RainbowValue.LIGHT_GREEN, "desc": "below bottom left"},
            {"x": center_x + 1, "y": center_y - 1, "color": RainbowValue.DARK_BLUE,   "desc": "upper right"},
            {"x": center_x + 1, "y": center_y + 1, "color": RainbowValue.PINK,        "desc": "bottom right"},

            {"x": max_x - 1, "y": max_y - 1, "color": RainbowValue.RED,        "desc": "bottom right corner"},
            {"x": max_x - 2, "y": max_y - 1, "color": RainbowValue.GREEN,      "desc": "bottom right corner"},
            {"x": max_x - 3, "y": max_y - 1, "color": RainbowValue.LIGHT_BLUE, "desc": "bottom right corner"},
            {"x": max_x - 1, "y": max_y - 2, "color": RainbowValue.PINK,       "desc": "bottom right corner"},
            {"x": max_x - 2, "y": max_y - 2, "color": RainbowValue.BROWN,      "desc": "bottom right corner"},

            {"x": 0, "y": max_y - 1, "color": RainbowValue.PINK,       "desc": "bottom left corner"},
            {"x": 1, "y": max_y - 1, "color": RainbowValue.LIGHT_BLUE, "desc": "bottom left corner"},
            {"x": 0, "y": max_y - 2, "color": RainbowValue.BROWN,      "desc": "bottom left corner"},
            {"x": 0, "y": max_y - 3, "color": RainbowValue.RED,        "desc": "bottom left corner"},
            {"x": 1, "y": max_y - 2, "color": RainbowValue.GREEN,      "desc": "bottom left corner"},

            {"x": max_x - 1, "y": 0, "color": RainbowValue.LIGHT_BLUE,  "desc": "upper right corner"},
            {"x": max_x - 2, "y": 0, "color": RainbowValue.LIGHT_GREEN, "desc": "upper right corner"},
            {"x": max_x - 3, "y": 0, "color": RainbowValue.RED,         "desc": "upper right corner"},
            {"x": max_x - 1, "y": 1, "color": RainbowValue.BROWN,       "desc": "upper right corner"},
            {"x": max_x - 2, "y": 1, "color": RainbowValue.PINK,        "desc": "upper right corner"},

            {"x": 0, "y": 0, "color": RainbowValue.BROWN,       "desc": "upper left corner"},
            {"x": 1, "y": 0, "color": RainbowValue.RED,         "desc": "upper left corner"},
            {"x": 0, "y": 1, "color": RainbowValue.PINK,        "desc": "upper left corner"},
            {"x": 0, "y": 2, "color": RainbowValue.DARK_BLUE,   "desc": "upper left corner"},
            {"x": 1, "y": 1, "color": RainbowValue.LIGHT_GREEN, "desc": "upper left corner"},
        ]

        for pixel in pixels:
            data[pixel["y"], pixel["x"]] = pixel["color"].value
            print(f'{pixel["desc"]} ({pixel["color"].name}) -> x: {pixel["x"]} y: {pixel["y"]}')

        return data

    def add_contour_dataset(self, dataset: DocBasicDataset, p: Presentation, image_data: np.ndarray):
        verts = image_data[:, :2]
        connects = image_data[:, 2].astype(np.bool)
        level_indexes = image_data[:, 3]
        level_indexes = level_indexes[~np.isnan(level_indexes)].astype(np.int)
        levels = dataset["contour_levels"]
        cmap = self.document.find_colormap(p.colormap)

        proj4_str = dataset[Info.PROJ]
        parent = self.proxy_nodes.get(proj4_str)
        if parent is None:
            parent = ContourGroupNode(parent=self.main_map)
            parent.transform = PROJ4Transform(dataset[Info.PROJ], inverse=True)
            self.proxy_nodes[proj4_str] = parent

        contour_visual = PrecomputedIsocurve(verts, connects, level_indexes,
                                             levels=levels, color_lev=cmap,
                                             clim=p.climits,
                                             parent=parent,
                                             name=str(dataset[Info.UUID]))
        contour_visual.transform *= STTransform(translate=(0, 0, -50.0))
        self.dataset_nodes[dataset[Info.UUID]] = contour_visual
        self.animation_controller.add_layer(contour_visual)
        self.on_view_change(None)

    def add_node_for_layer(self, layer: LayerItem):
        if not USE_TILED_GEOLOCATED_IMAGES and layer.kind == Kind.IMAGE:
            layer_node = scene.Node(parent=self.main_map_parent,
                                    name=str(layer.uuid))
        else:
            layer_node = scene.Node(parent=self.main_map,
                                    name=str(layer.uuid))

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
        """ Helper function for setting up the VisualNode for the system
        layer for latitude/longitude grid.

        :param layer_node: Scene graph node to be used as parent for the grid
                           node.
        """
        latlon_grid_resolution = get_configured_latlon_grid_resolution()
        latlon_grid_points = \
            self._create_latlon_grid_points(resolution=latlon_grid_resolution)
        self.latlon_grid_node = Line(
            pos=latlon_grid_points, connect="strip",
            color=self._color_choices[self._latlon_grid_color_idx],
            parent=layer_node
        )

    def _build_borders_nodes(self, layer_node):
        """ Helper function for setting up the VisualNodes for the system
        layer for political borders.

        One node is generated for each file stored in the (currently) internal
        list of political borders shapefiles.

        :param layer_node: Scene graph node to be used as parent for the
                           borders node(s).
        """
        for shapefile in self.borders_shapefiles:
            node = NEShapefileLines(
                shapefile, double=True,
                color=self._color_choices[self._borders_color_idx],
                parent=layer_node
            )
            self.borders_nodes.append(node)

    def add_node_for_image_dataset(self, layer: LayerItem,
                                   product_dataset: ProductDataset):
        assert self.layer_nodes[layer.uuid] is not None
        assert product_dataset.kind == Kind.IMAGE

        image_data = self.workspace.get_content(product_dataset.uuid,
                                                kind=product_dataset.kind)

        if False:  # Set to True FOR TESTING ONLY
            self._overwrite_with_test_pattern(image_data)

        if USE_TILED_GEOLOCATED_IMAGES:
            image = TiledGeolocatedImage(
                image_data,
                product_dataset.info[Info.ORIGIN_X],
                product_dataset.info[Info.ORIGIN_Y],
                product_dataset.info[Info.CELL_WIDTH],
                product_dataset.info[Info.CELL_HEIGHT],
                name=str(product_dataset.uuid),
                clim=layer.presentation.climits,
                gamma=layer.presentation.gamma,
                interpolation='nearest',
                method='subdivide',
                cmap=self.document.find_colormap(layer.presentation.colormap),
                double=False,
                texture_shape=DEFAULT_TEXTURE_SHAPE,
                wrap_lon=False,
                parent=self.layer_nodes[layer.uuid],
                projection=product_dataset.info[Info.PROJ],
            )
            image.transform = PROJ4Transform(product_dataset.info[Info.PROJ],
                                             inverse=True)
            image.determine_reference_points()
        else:
            image = Image(
                image_data,
                name=str(product_dataset.uuid),
                clim=layer.presentation.climits,
                gamma=layer.presentation.gamma,
                interpolation='nearest',
                cmap=self.document.find_colormap(layer.presentation.colormap),
                parent=self.layer_nodes[layer.uuid],
            )
            image.transform = STTransform(
                scale=(product_dataset.info[Info.CELL_WIDTH],
                       product_dataset.info[Info.CELL_HEIGHT], 1),
                translate=(product_dataset.info[Info.ORIGIN_X],
                           product_dataset.info[Info.ORIGIN_Y], 0))
        self.dataset_nodes[product_dataset.uuid] = image
        self.on_view_change(None)
        LOG.debug("Scene Graph after IMAGE dataset insertion:")
        LOG.debug(self.main_view.describe_tree(with_transform=True))

    def add_composite_dataset(self, new_order: tuple, uuid: UUID, p: Presentation):
        layer = self.document[uuid]
        LOG.debug("SceneGraphManager.add_composite_layer %s" % repr(layer))
        if not layer.is_valid:
            LOG.info('unable to add an invalid composite layer, will try again later when layer changes')
            return
        if p.kind == Kind.RGB:
            dep_uuids = r, g, b = [c.uuid if c is not None else None for c in [layer.r, layer.g, layer.b]]
            image_data = list(self.workspace.get_content(cuuid, kind=Kind.IMAGE) for cuuid in dep_uuids)
            uuid = layer.uuid
            LOG.debug("Adding composite layer to Scene Graph Manager with UUID: %s", uuid)
            if USE_TILED_GEOLOCATED_IMAGES:
                self.dataset_nodes[uuid] = dataset_node = RGBCompositeLayer(
                    image_data,
                    layer[Info.ORIGIN_X],
                    layer[Info.ORIGIN_Y],
                    layer[Info.CELL_WIDTH],
                    layer[Info.CELL_HEIGHT],
                    name=str(uuid),
                    clim=p.climits,
                    gamma=p.gamma,
                    interpolation='nearest',
                    method='subdivide',
                    cmap=None,
                    double=False,
                    texture_shape=DEFAULT_TEXTURE_SHAPE,
                    wrap_lon=False,
                    parent=self.main_map,
                    projection=layer[Info.PROJ],
                )
                dataset_node.transform = PROJ4Transform(layer[Info.PROJ], inverse=True)
                dataset_node.transform *= STTransform(translate=(0, 0, -50.0))
                self.composite_element_dependencies[uuid] = dep_uuids
                self.animation_controller.add_layer(dataset_node)
                if new_order:
                    self.animation_controller.set_layer_order(new_order)
                self.on_view_change(None)
                dataset_node.determine_reference_points()
            else:
                self.dataset_nodes[uuid] = dataset_node = MultiChannelImage(
                    image_data,
                    name=str(uuid),
                    clim=p.climits,
                    gamma=p.gamma,
                    interpolation='nearest',
                    cmap=None,
                    parent=self.main_map_parent
                )
                # TODO: We must use a ChainTransform, because of the assumption made
                # in LayerSet.update_layers_z(): there it is assumed, that the
                # (last) transformation of the layer can be overwritten incautiously
                # since it would only carry z translation. So the question: why
                # doesn't LayerSet.add_layer() take care to guarantee this?
                # Anyhow, here we can't concatenate two STTransforms with *= because
                # they would end up in *one* transform whose 'translate' part would
                # be overwritten in LayerSet.update_layers_z(), i.e. any x,y-offset
                # would be reset to 0.0,0.0.
                calibration_transform = STTransform(
                    scale=(layer[Info.CELL_WIDTH], layer[Info.CELL_HEIGHT], 1),
                    translate=(layer[Info.ORIGIN_X], layer[Info.ORIGIN_Y], 0))
                z_transform = STTransform(translate=(0, 0, -50))
                dataset_node.transform = ChainTransform([calibration_transform,
                                                    z_transform])
                self.composite_element_dependencies[uuid] = dep_uuids
                self.animation_controller.add_layer(dataset_node)
                if new_order:
                    self.animation_controller.set_layer_order(new_order)
                self.on_view_change(None)

            self.update()
            return True
        elif p.kind in [Kind.COMPOSITE, Kind.IMAGE]:
            # algebraic layer
            return self.add_basic_dataset(new_order, uuid, p)

    def add_node_for_lines_dataset(self, layer: LayerItem,
                                   product_dataset: ProductDataset) \
            -> scene.VisualNode:
        assert self.layer_nodes[layer.uuid] is not None
        assert product_dataset.kind == Kind.LINES

        content, _ = self.workspace.get_lines_arrays(product_dataset.uuid)
        if content is None:
            LOG.info(f"Dataset contains no lines: {product_dataset.uuid}")
            return

        lines = Lines(content,
                      parent=self.layer_nodes[layer.uuid])
        lines.name = str(product_dataset.uuid)

        self.dataset_nodes[product_dataset.uuid] = lines
        self.on_view_change(None)
        LOG.debug("Scene Graph after LINES dataset insertion:")
        LOG.debug(self.main_view.describe_tree(with_transform=True))

    def add_node_for_points_dataset(self, layer: LayerItem,
                                    product_dataset: ProductDataset) \
            -> scene.VisualNode:
        assert self.layer_nodes[layer.uuid] is not None
        assert product_dataset.kind == Kind.POINTS

        pos, values = self.workspace.get_points_arrays(product_dataset.uuid)
        if pos is None:
            LOG.info(f"layer contains no points: {product_dataset.uuid}")
            return

        kwargs = map_point_style_to_marker_kwargs(
            get_point_style_by_name(layer.presentation.style))

        if values is not None:
            assert len(pos) == len(values)
            # TODO use climits of the presentation instead of autoscaling?
            colormap = self.document.find_colormap(layer.presentation.colormap)
            kwargs["face_color"] = \
                self.map_to_colors_autoscaled(colormap, values)

        points = Markers(pos=pos,
                         parent=self.layer_nodes[layer.uuid],
                         **kwargs)
        points.name = str(product_dataset.uuid)

        self.dataset_nodes[product_dataset.uuid] = points
        self.on_view_change(None)
        LOG.debug("Scene Graph after POINTS dataset insertion:")
        LOG.debug(self.main_view.describe_tree(with_transform=True))

    def map_to_colors_autoscaled(self, colormap, values, m=2):
        """
        Get a list of colors by mapping each entry in values by the given
        colormap.

        The mapping range is adjusted automatically to m times the standard
        deviation from the mean. This ignores outliers in the calculation of
        the mapping range.

        Caution: this is an expensive operation and must not be called in tight
        loops.

        :param colormap: the colormap to apply
        :param values: the values to map to colors
        :param m: factor to stretch the standard deviation around the mean to
        define the mapping range
        :return: list of mapped colors in the same order as the input values
        """
        std_dev = np.std(values)
        mean = np.mean(values)
        min = mean - m * std_dev  # noqa: calm down PyCharm's spelling check, ...
        max = mean + m * std_dev  # noqa: ... 'min' and 'max' are fine!
        scaled_attr = np.interp(values, (min, max), (0, 1))
        colors = colormap.map(scaled_attr)
        return colors

    def change_composite_datasets(self, new_order: tuple, uuid_list: list, presentations: list):
        for uuid, presentation in zip(uuid_list, presentations):
            self.change_composite_dataset(None, uuid, presentation)
        # set the order after we've updated and created all the new layers
        if new_order:
            self.animation_controller.set_layer_order(new_order)

    def change_composite_dataset(self, new_order: tuple, uuid: UUID, presentation: Presentation):
        layer = self.document[uuid]
        if presentation.kind == Kind.RGB:
            if layer.uuid in self.dataset_nodes:
                if layer.is_valid:
                    # RGB selection has changed, rebuild the layer
                    LOG.debug("Changing existing composite layer to Scene Graph Manager with UUID: %s", layer.uuid)
                    dep_uuids = r, g, b = [c.uuid if c is not None else None for c in [layer.r, layer.g, layer.b]]
                    image_arrays = list(self.workspace.get_content(cuuid) for cuuid in dep_uuids)
                    self.composite_element_dependencies[layer.uuid] = dep_uuids
                    elem = self.dataset_nodes[layer.uuid]
                    if isinstance(elem, RGBCompositeLayer):
                        elem.set_channels(image_arrays,
                                          cell_width=layer[Info.CELL_WIDTH],
                                          cell_height=layer[Info.CELL_HEIGHT],
                                          origin_x=layer[Info.ORIGIN_X],
                                          origin_y=layer[Info.ORIGIN_Y])
                    elif isinstance(elem, MultiChannelImage):
                        elem.set_data(image_arrays)
                    elem.clim = presentation.climits
                    elem.gamma = presentation.gamma
                    self.on_view_change(None)
                    if isinstance(elem, RGBCompositeLayer):
                        elem.determine_reference_points()
                else:
                    # layer is no longer valid and has to be removed
                    LOG.debug("Purging composite ")
                    self.purge_dataset(layer.uuid)
                self.update()
            else:
                if layer.is_valid:
                    # Add this now valid layer
                    self.add_composite_dataset(new_order, layer.uuid, presentation)
                else:
                    LOG.info('unable to add an changed invalid composite layer, will try again later when layer changes')
                    return
        else:
            raise ValueError("Unknown or unimplemented composite type")

    def update_basic_dataset(self, uuid: UUID, kind: Kind):
        """
        Push the data (content) of a basic layer again to the associated scene
        graph node.

        This method shall be called whenever the data of a basic layer changes.
        :param uuid: identifier of the layer
        :param kind: kind of the layer / data content.
        """
        layer = self.document[uuid]
        dataset_node = self.dataset_nodes[layer[Info.UUID]]
        dataset_content = self.workspace.get_content(layer[Info.UUID], kind=kind)
        try:
            dataset_node.set_data(dataset_content)
        except NotImplementedError:
            if isinstance(dataset_node, TiledGeolocatedImage):
                LOG.debug(f"Updating data for UUID {uuid} on its associated"
                          f" scenegraph TiledGeolocatedImage node is not"
                          f" possible, hopefully the data was modified in-place"
                          f" (e.g. when merging new granules).")
                # TODO: How to detect the case that the data was not changed in
                #  place but a new reference was given? In this case, we must
                #  re-raise the NotImplementedError exception (as in the 'else'
                #  path)
                # TODO: TiledGeolocatedImage does not provide a way to tell it
                #  that it should drop all retiled data and start from scratch.
            else:
                # This is a unforeseen case: at the moment this method
                # should only be called when merging data segments into existing
                # image(!) data, looks like it was called for a node of another
                # type not having set_data() too.
                raise

        self.on_view_change(None)

    def update_layers_z(self, uuids: list):
        if self.layer_nodes:
            for z_level, uuid in enumerate(uuids):
                layer_node = self.layer_nodes[uuid]
                layer_node.transform.translate = (0, 0, 0 - z_level)
            self.update()

    def remove_dataset(self, new_order: tuple, uuids_removed: tuple, row: int, count: int):
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

    def _remove_dataset(self, *args, **kwargs):
        self.remove_dataset(*args, **kwargs)
        # when removing the layer is the only operation being performed then update when we are done
        self.update()

    def purge_dataset(self, uuid_removed: UUID):
        """
        Layer has been purged from document (no longer used anywhere) - flush it all out
        :param uuid_removed: UUID of the layer that is to be removed
        :return:
        """
        self.set_layer_visible(uuid_removed, False)
        if uuid_removed in self.dataset_nodes:
            image_layer = self.dataset_nodes[uuid_removed]
            image_layer.parent = None
            del self.dataset_nodes[uuid_removed]
            LOG.info("layer {} purge from scenegraphmanager".format(uuid_removed))
        else:
            LOG.debug("Layer {} already purged from Scene Graph".format(uuid_removed))

    def _purge_dataset(self, *args, **kwargs):
        res = self.purge_dataset(*args, **kwargs)
        # when purging the layer is the only operation being performed then update when we are done
        self.update()
        return res

    def change_datasets_visibility(self, layers_changed: Dict[UUID, bool]):
        for uuid, visible in layers_changed.items():
            self.set_layer_visible(uuid, visible)

    def rebuild_new_layer_set(self, new_set_number: int, new_prez_order: DocLayerStack, new_anim_order: list):
        self.rebuild_all()
        # raise NotImplementedError("layer set change not implemented in SceneGraphManager")

    def _connect_doc_signals(self, document: Document):
        document.didReorderDatasets.connect(self._rebuild_dataset_order)  # current layer set changed z/anim order
        # REMOVE document.didAddBasicDataset.connect(self.add_basic_dataset)  # layer added to one or more layer sets
        document.didUpdateBasicDataset.connect(self.update_basic_dataset)  # new data integrated in existing layer
        document.didAddCompositeDataset.connect(
            self.add_composite_dataset)  # layer derived from other layers (either basic or composite themselves)
        # REMOVE document.didAddLinesDataset.connect(self.add_lines_dataset)
        # REMOVE document.didAddPointsDataset.connect(self.add_points_dataset)
        document.didRemoveDatasets.connect(self._remove_dataset)  # layer removed from current layer set
        document.willPurgeDataset.connect(self._purge_dataset)  # layer removed from document
        document.didSwitchLayerSet.connect(self.rebuild_new_layer_set)
        document.didChangeColormap.connect(self.change_layers_colormap)
        document.didChangeLayerVisibility.connect(self.change_datasets_visibility)
        document.didReorderAnimation.connect(self._rebuild_frame_order)
        document.didChangeComposition.connect(self.change_composite_dataset)
        document.didChangeCompositions.connect(self.change_composite_datasets)
        document.didChangeColorLimits.connect(self.change_layers_color_limits)
        document.didChangeGamma.connect(self.change_layers_gamma)
        # document.didChangeImageKind.connect(self.change_layers_image_kind)

    def set_frame_number(self, frame_number=None):
        self.animation_controller.next_frame(None, frame_number)

    def set_layer_visible(self, uuid: UUID, visible: Optional[bool] = None):
        dataset_node = self.dataset_nodes.get(uuid, None)
        if dataset_node is None:
            return
        dataset_node.visible = not dataset_node.visible if visible is None else visible

    def rebuild_dataset_order(self, new_layer_index_order, *args, **kwargs):
        """
        layer order has changed; shift layers around.
        an empty list is sent if the whole layer order has been changed
        :param change:
        :return:
        """
        # TODO this is the lazy implementation, eventually just change z order on affected layers
        self.animation_controller.set_layer_order(self.document.current_layer_uuid_order)

    def _rebuild_dataset_order(self, *args, **kwargs):
        res = self.rebuild_dataset_order(*args, **kwargs)
        self.update()
        return res

    def rebuild_frame_order(self, uuid_list: list, *args, **kwargs):
        LOG.debug('setting SGM new frame order to {0!r:s}'.format(uuid_list))
        self.animation_controller.frame_order = uuid_list
        #self.layer_set.update_time_manager_collection(self.document.data_layer_collection)

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

        uuids_w_elements = set(self.dataset_nodes.keys())
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
                    self.change_composite_dataset(current_uuid_order, layer, prez_lookup[uuid])
                else:
                    # FIXME this was previously a NotImplementedError
                    LOG.warning('unable to create deferred scenegraph element for %s' % repr(layer))
            else:
                # remove elements for layers which are no longer valid
                remove_elements.append(uuid)

        # get info on the new order
        self.animation_controller.set_layer_order(current_uuid_order)
        self.animation_controller.frame_order = self.document.current_animation_order
        self.rebuild_presentation(prez_lookup)

        for elem in remove_elements:
            self.purge_dataset(elem)
        # This is triggered, when the layer set is updated?
        # import_product_content when data loaded?
        self.animation_controller.update_time_manager(self.document.data_layer_collection)

        # Triggers main canvas update
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

        current_visible_datasets_uuids = \
            [p.uuid for (p, l) in self.document.active_layer_order if p.visible]
        current_invisible_datasets_uuids = \
            set(self.dataset_nodes.keys()) - set(current_visible_datasets_uuids)

        def _assess_if_active(uuid):
            dataset_node = self.dataset_nodes.get(uuid, None)
            if dataset_node is not None and hasattr(dataset_node, 'assess'):
                _assess(uuid, dataset_node)

        for uuid in current_visible_datasets_uuids:
            _assess_if_active(uuid)
        # update contours
        for node in self.proxy_nodes.values():
            node.on_view_change()
        # update invisible datasets
        for uuid in current_invisible_datasets_uuids:
            _assess_if_active(uuid)

    def start_retiling_task(self, uuid, preferred_stride, tile_box):
        LOG.debug("Scheduling retile for child with UUID: %s", uuid)
        self.queue.add(str(uuid) + "_retile", self._retile_child(uuid, preferred_stride, tile_box),
                       'Retile calculations for image layer ' + str(uuid), interactive=True)

    def _retile_child(self, uuid, preferred_stride, tile_box):
        LOG.debug("Retiling child with UUID: '%s'", uuid)
        yield {TASK_DOING: 'Re-tiling', TASK_PROGRESS: 0.0}
        if uuid not in self.composite_element_dependencies:
            child = self.dataset_nodes[uuid]
            data = self.workspace.get_content(uuid, lod=preferred_stride)
            yield {TASK_DOING: 'Re-tiling', TASK_PROGRESS: 0.5}
            # FIXME: Use LOD instead of stride and provide the lod to the workspace
            data = data[::preferred_stride[0], ::preferred_stride[1]]
            tiles_info, vertices, tex_coords = child.retile(data, preferred_stride, tile_box)
            yield {TASK_DOING: 'Re-tiling', TASK_PROGRESS: 1.0}
            self.didRetilingCalcs.emit(uuid, preferred_stride, tile_box, tiles_info, vertices, tex_coords)
        else:
            child = self.dataset_nodes[uuid]
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
        child = self.dataset_nodes.get(uuid, None)
        if child is None:
            LOG.warning('unable to find uuid %s in dataset_nodes' % uuid)
            return
        child.set_retiled(preferred_stride, tile_box, tiles_info, vertices, tex_coords)
        child.update()

    def on_layer_visible_toggle(self, visible):
        pass

    def on_layer_change(self, event):
        pass

    def on_data_loaded(self, event):
        pass


# TODO move these defaults to common config defaults location
LATLON_GRID_RESOLUTION_MIN: float = 0.1
LATLON_GRID_RESOLUTION_DEFAULT: float = 5.0
LATLON_GRID_RESOLUTION_MAX: float = 10.0


def get_configured_latlon_grid_resolution() -> float:

    resolution: float = config.get("latlon_grid.resolution",
                                   LATLON_GRID_RESOLUTION_DEFAULT)

    if not isinstance(resolution, Number):
        LOG.warning(
            f"Invalid configuration for lat/lon grid resolution"
            f" (='{resolution}') found."
            f" Using the default {LATLON_GRID_RESOLUTION_DEFAULT}°.")
        return LATLON_GRID_RESOLUTION_DEFAULT

    if resolution > LATLON_GRID_RESOLUTION_MAX:
        LOG.warning(
            f"Configured lat/lon grid resolution {resolution}°"
            f" is greater than allowed maximum."
            f" Using the maximum {LATLON_GRID_RESOLUTION_MAX}°.")
        return LATLON_GRID_RESOLUTION_MAX

    if resolution < LATLON_GRID_RESOLUTION_MIN:
        LOG.warning(
            f"Configured lat/lon grid resolution {resolution}°"
            f" is less than allowed minimum."
            f" Using the minimum {LATLON_GRID_RESOLUTION_MIN}°.")
        return LATLON_GRID_RESOLUTION_MIN

    return resolution


