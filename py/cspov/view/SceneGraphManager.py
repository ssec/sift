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

from vispy import app
from vispy import scene
from vispy.util.event import Event
from vispy.visuals.transforms import STTransform, MatrixTransform
from vispy.visuals import MarkersVisual, marker_types, LineVisual
from vispy.scene.visuals import Markers, Polygon, Compound
from cspov.common import WORLD_EXTENT_BOX, DEFAULT_ANIMATION_DELAY, INFO, KIND
# from cspov.control.layer_list import LayerStackListViewModel
from cspov.view.LayerRep import NEShapefileLines, TiledGeolocatedImage
from cspov.view.MapWidget import CspovMainMapCanvas
from cspov.view.Cameras import ProbeCamera
from cspov.queue import TASK_DOING, TASK_PROGRESS

from PyQt4.QtCore import QObject, pyqtSignal
import numpy as np
from uuid import UUID


import os
import logging

LOG = logging.getLogger(__name__)
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
DEFAULT_SHAPE_FILE = os.path.join(SCRIPT_DIR, "..", "data", "ne_50m_admin_0_countries", "ne_50m_admin_0_countries.shp")
DEFAULT_TEXTURE_SHAPE = (4, 16)


class FakeMarker(Compound):
    # FIXME: Temporary workaround because markers don't work on the target Windows laptops
    def __init__(self, pos=None, parent=None, symbol=None, **kwargs):
        kwargs["connect"] = "segments"
        margin = 50000
        width = 5
        point = pos[0]
        pos1 = np.array([[point[0] - margin, point[1]], [point[0] + margin, point[1]]])
        pos2 = np.array([[point[0], point[1] - margin * 2], [point[0], point[1] + margin * 2]])
        print("Creating FakeMarker: ", pos1, pos2)
        self.line_one = LineVisual(pos=pos1, width=width, **kwargs)
        self.line_two = LineVisual(pos=pos2, width=width, **kwargs)
        super().__init__((self.line_one, self.line_two), parent=parent)


class MainMap(scene.Node):
    """Scene node for holding all of the information for the main map area
    """
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

    def add_point(self, canvas_pos, xy_pos):
        if self.is_complete(canvas_pos):
            # Are you finishing the polygon by adding this point (same point as the first point...or near it)
            return True
        self.canvas_points.append(canvas_pos)
        self.points.append(xy_pos)
        point_visual = FakeMarker(parent=self.parent, symbol="disc", pos=np.array([xy_pos[:2]]), color=np.array([0., 0.5, 0.5, 1.]))
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
        self._animation_timer = app.Timer(DEFAULT_ANIMATION_DELAY, connect=self.next_frame)

        if layers is not None:
            self.set_layers(layers)

            if layer_order is None:
                layer_order = [x.name for x in layers.keys()]
            self.set_layer_order(layer_order)

            if frame_order is None:
                frame_order = [x.name for x in layers.keys()]
            self.set_frame_order(frame_order)

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
            if o not in self._layers:
                LOG.error('set_layer_order cannot deal with unknown layer {}'.format(o))
                return
        self._layer_order = layer_order
        self.update_layers_z()

    def set_frame_order(self, frame_order):
        for o in frame_order:
            if o not in self._layers:
                LOG.error('set_frame_order cannot deal with unknown layer {}'.format(o))
                return
        self._frame_order = frame_order
        self._frame_number = 0
        if self._frame_change_cb is not None:
            self._frame_change_cb((self._frame_number, len(self._frame_order), self._animating))

    def update_layers_z(self):
        for z_level, uuid in enumerate(self._layer_order):
            self._layers[uuid].transform = STTransform(translate=(0, 0, 0-int(z_level)))

    # def set_layer_z(self, uuid, z_level):
    #     """
    #     :param uuid: layer to change
    #     :param z_level: -100..100, 100 being closest to the camera
    #     :return:
    #     """
    #     self._layers[uuid].transform = STTransform(translate=(0, 0, int(z_level)))
    #
    # def set_layers_z(self, layer_levels):
    #     """
    #     z_levels are -100..100, 100 being closest to the camera
    #     :param layer_levels: {uuid:level}
    #     :return:
    #     """
    #     for uuid, z_level in layer_levels.items():
    #         self._layers[uuid].transform = STTransform(translate=(0, 0, int(z_level)))

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
        print("Running animating {} with {} frames".format(animate, len(self._frame_order)))
        if self._animating and not animate:
            # We are currently, but don't want to be
            self._animating = False
            self._animation_timer.stop()
        elif not self._animating and animate:
            # We are not currently, but want to be
            self._animating = True
            self._animation_timer.start()
            # TODO: Add a proper AnimationEvent to self.events

    def toggle_animation(self, *args):
        self.animating = not self._animating
        if self._frame_change_cb is not None:
            self._frame_change_cb((self._frame_number, len(self._frame_order), self._animating))

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
            if frame_number==-1:
                frame = self._frame_number + (lfo - 1)
            else:
                frame = frame_number
        if lfo>0:
            frame %= lfo
        else:
            frame = 0
        self._set_visible_child(frame)
        self._frame_number = frame
        self.parent.update()
        if self._frame_change_cb is not None:
            self._frame_change_cb((self._frame_number, lfo, self._animating))


class SceneGraphManager(QObject):
    didRetilingCalcs = pyqtSignal(object, object, object, object, object, object)
    didChangeFrame = pyqtSignal(tuple)
    newProbePoint = pyqtSignal(object, object)
    newProbePolygon = pyqtSignal(object, object)

    def __init__(self, doc, workspace, queue, border_shapefile=None, glob_pattern=None, parent=None, texture_shape=(4, 16)):
        super(SceneGraphManager, self).__init__(parent)
        self.didRetilingCalcs.connect(self._set_retiled)

        # Parent should be the Qt widget that this GLCanvas belongs to
        self.document = doc
        self.workspace = workspace
        self.queue = queue
        self.border_shapefile = border_shapefile or DEFAULT_SHAPE_FILE
        self.glob_pattern = glob_pattern
        self.texture_shape = texture_shape
        self.polygons = []

        self.image_layers = {}
        self.datasets = {}
        self.colormaps = {}
        self.layer_set = LayerSet(self, frame_change_cb=self.frame_changed)

        self.set_document(self.document)

        self.setup_initial_canvas()
        self.pending_polygon = PendingPolygon(self.main_map)

    def frame_changed(self, frame_info):
        """
        callback which emits information on current animation frame as a signal
        (see LayerSet.next_frame)
        :param frame_info: tuple to be relayed in the signal, typically (frame_index:int, total_frames:int, animating:bool)
        """
        # LOG.debug('emitting didChangeFrame')
        self.didChangeFrame.emit(frame_info)

    def setup_initial_canvas(self):
        self.main_canvas = CspovMainMapCanvas(parent=self.parent())
        self.main_view = self.main_canvas.central_widget.add_view()

        # Camera Setup
        self.pz_camera = scene.PanZoomCamera(name="pz_camera", aspect=1)
        self.main_view.camera = self.pz_camera

        self.main_view.camera.flip = (False, False, False)
        # FIXME: these ranges just happen to look ok, but I'm not really sure the 'best' way to set these
        self.main_view.camera.set_range(x=(-10.0, 10.0), y=(-10.0, 10.0), margin=0)
        self.main_view.camera.zoom(0.1, (0, 0))

        # Point Probe Mode/Camera
        self.point_probe_camera = ProbeCamera(name="point_probe_camera", aspect=1)
        self.main_view.camera.link(self.point_probe_camera)

        # Polygon Probe Mode/Camera
        self.polygon_probe_camera = ProbeCamera(name="polygon_probe_camera", aspect=1)
        self.main_view.camera.link(self.polygon_probe_camera)

        self._cameras = dict((c.name, c) for c in [self.main_view.camera, self.point_probe_camera, self.polygon_probe_camera])
        self._camera_names = [self.pz_camera.name, self.point_probe_camera.name, self.polygon_probe_camera.name]

        self.main_view.events.mouse_press.connect(self.on_mouse_press, after=list(self.main_view.events.mouse_press.callbacks))

        # Head node of the map graph
        self.main_map = MainMap(name="MainMap", parent=self.main_view.scene)
        merc_ortho = MatrixTransform()
        # near/far is backwards it seems:
        camera_z_scale = 1e-6
        l, r, b, t = [getattr(WORLD_EXTENT_BOX, x) for x in ['l', 'r', 'b', 't']]
        merc_ortho.set_ortho(l, r, b, t, -100.0 * camera_z_scale, 100.0 * camera_z_scale)
        # self.main_map.transforms.visual_transform = merc_ortho
        self.main_map.transform = merc_ortho

        self.boundaries = NEShapefileLines(self.border_shapefile, double=True, parent=self.main_map)

    def on_mouse_press(self, event):
        if event.handled:
            return
        # What does this mouse press mean?
        if self.main_view.camera is self.point_probe_camera:
            buffer_pos = event.sources[0].transforms.get_transform().map(event.pos)
            # FIXME: We should be able to use the main_map object to do the transform...but it doesn't work (waiting on vispy developers)
            # map_pos = self.main_map.transforms.get_transform().imap(buffer_pos)
            map_pos = list(self.image_layers.values())[0].transforms.get_transform().imap(buffer_pos)
            # point_marker = Markers(parent=self.main_map, symbol="disc", pos=np.array([map_pos[:2]]))
            # self.points.append(point_marker)
            self.newProbePoint.emit(self.layer_set.top_layer_uuid(), map_pos[:2])
        elif self.main_view.camera is self.polygon_probe_camera:
            buffer_pos = event.sources[0].transforms.get_transform().map(event.pos)
            map_pos = list(self.image_layers.values())[0].transforms.get_transform().imap(buffer_pos)
            if self.pending_polygon.add_point(event.pos, map_pos):
                points = self.pending_polygon.points + [self.pending_polygon.points[0]]
                for marker in self.pending_polygon.markers:
                    # Remove the marker from the scene graph
                    marker.parent = None
                # Reset the pending polygon object
                self.pending_polygon.reset()
                self.newProbePolygon.emit(self.layer_set.top_layer_uuid(), points)
        else:
            print("I don't know how to handle this camera for a mouse press")

    def on_new_polygon(self, points, **kwargs):
        kwargs.setdefault("color", (1.0, 0.0, 1.0, 0.5))
        poly = Polygon(parent=self.main_map, pos=points, **kwargs)
        self.polygons.append(poly)

    def update(self):
        return self.main_canvas.update()

    def change_camera(self, idx_or_name):
        if isinstance(idx_or_name, str):
            camera = self._cameras[idx_or_name]
        else:
            camera = self._cameras[self._camera_names[idx_or_name]]

        print("Changing camera to ", camera)
        self.main_view.camera = camera

    def next_camera(self):
        idx = self._camera_names.index(self.main_view.camera.name)
        idx = (idx + 1) % len(self._camera_names)
        self.change_camera(idx)

    def set_colormap(self, colormap, uuid=None):
        if isinstance(colormap, str) and colormap in self.colormaps:
            colormap = self.colormaps[colormap]

        uuids = uuid
        if uuid is None:
            uuids = self.image_layers.keys()
        elif not isinstance(uuid, (list, tuple)):
            uuids = [uuid]

        for uuid in uuids:
            self.image_layers[uuid].cmap = colormap

    def add_colormap(self, name:str, colormap):
        self.colormaps[name] = colormap

    def change_layers_colormap(self, change_dict):
        for uuid,cmapid in change_dict.items():
            self.set_colormap(cmapid, uuid)

    def add_layer(self, new_order:list, ds_info:dict, overview_content:np.ndarray):
        uuid = ds_info[INFO.UUID]
        # create a new layer in the imagelist
        image = TiledGeolocatedImage(
            overview_content,
            ds_info[INFO.ORIGIN_X],
            ds_info[INFO.ORIGIN_Y],
            ds_info[INFO.CELL_WIDTH],
            ds_info[INFO.CELL_HEIGHT],
            name=str(uuid),
            clim=ds_info[INFO.CLIM],
            interpolation='nearest',
            method='tiled',
            cmap='grays',
            double=False,
            texture_shape=DEFAULT_TEXTURE_SHAPE,
            wrap_lon=True,
            parent=self.main_map,
        )
        image.transform *= STTransform(translate=(0, 0, -50.0))
        self.image_layers[uuid] = image
        self.datasets[uuid] = ds_info
        self.layer_set.add_layer(image)
        cao = self.document.current_animation_order
        self.layer_set.set_frame_order(cao)
        print('animation order has {} frames'.format(len(cao)))

    def remove_layer(self, new_order:list, uuid_removed:UUID):
        self.set_layer_visible(uuid_removed, False)
        raise NotImplementedError("layer removal from scenegraph not implemented")  # FIXME

    def change_layers_visibility(self, layers_changed:dict):
        for uuid,visible in layers_changed.items():
            self.set_layer_visible(uuid, visible)

    def rebuild_new_layer_set(self, new_set_number:int, new_prez_order:list, new_anim_order:list):
        self.rebuild_all()
        # raise NotImplementedError("layer set change not implemented in SceneGraphManager")

    def set_document(self, document):
        document.didReorderLayers.connect(self.rebuild_new_order)
        document.didAddLayer.connect(self.add_layer)
        document.didRemoveLayer.connect(self.remove_layer)
        document.didSwitchLayerSet.connect(self.rebuild_new_layer_set)
        document.didChangeColormap.connect(self.change_layers_colormap)
        document.didChangeLayerVisibility.connect(self.change_layers_visibility)

    def set_frame_number(self, frame_number=None):
        self.layer_set.next_frame(None, frame_number)

    def set_layer_visible(self, uuid, visible=None):
        image = self.image_layers[uuid]
        image.visible = not image.visible if visible is None else visible

    def rebuild_new_order(self, new_layer_index_order, *args, **kwargs):
        """
        layer order has changed; shift layers around.
        an empty list is sent if the whole layer order has been changed
        :param change:
        :return:
        """
        # TODO this is the lazy implementation, eventually just change z order on affected layers
        self.layer_set.set_layer_order(self.document.current_layer_order)
        print("New layer order: ", new_layer_index_order)
        self.update()

    def rebuild_all(self, *args, **kwargs):
        self.layer_set.set_layer_order(self.document.current_layer_order)
        self.layer_set.set_frame_order(self.document.current_animation_order)
        self.update()

    def on_view_change(self, scheduler, ws=None):
        """Simple event handler for when we need to reassess image layers.
        """
        # Stop the timer so it doesn't continuously call this slot
        scheduler.stop()

        for uuid, child in self.image_layers.items():
            need_retile, preferred_stride, tile_box = child.assess()
            if need_retile:
                self.start_retiling_task(uuid, preferred_stride, tile_box)

    def start_retiling_task(self, uuid, preferred_stride, tile_box):
        LOG.debug("Scheduling retile for child with UUID: %s", uuid)
        self.queue.add(str(uuid) + "_retile", self._retile_child(uuid, preferred_stride, tile_box), 'Retile calculations for image layer ' + str(uuid))

    def _retile_child(self, uuid, preferred_stride, tile_box):
        LOG.debug("Retiling child with UUID: '%s'", uuid)
        yield {TASK_DOING: 'Re-tiling', TASK_PROGRESS: 0.0}
        child = self.image_layers[uuid]
        data = self.workspace.get_content(uuid, lod=preferred_stride)
        yield {TASK_DOING: 'Re-tiling', TASK_PROGRESS: 0.5}
        # FIXME: Use LOD instead of stride and provide the lod to the workspace
        data = data[::preferred_stride, ::preferred_stride]
        tiles_info, vertices, tex_coords = child.retile(data, preferred_stride, tile_box)
        yield {TASK_DOING: 'Re-tiling', TASK_PROGRESS: 1.0}
        self.didRetilingCalcs.emit(uuid, preferred_stride, tile_box, tiles_info, vertices, tex_coords)

    def _set_retiled(self, uuid, preferred_stride, tile_box, tiles_info, vertices, tex_coords):
        """Slot to take data from background thread and apply it to the layer living in the image layer.
        """
        child = self.image_layers[uuid]
        child.set_retiled(preferred_stride, tile_box, tiles_info, vertices, tex_coords)
        child.update()

    def on_layer_visible_toggle(self, visible):
        pass

    def on_layer_change(self, event):
        pass

    def on_data_loaded(self, event):
        pass


