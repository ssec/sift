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
from cspov.common import WORLD_EXTENT_BOX
from cspov.control.layer_list import LayerStackListViewModel
from cspov.view.LayerRep import NEShapefileLines, TiledGeolocatedImage
from cspov.view.MapWidget import CspovMainMapCanvas

import os
import logging

LOG = logging.getLogger(__name__)
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
DEFAULT_SHAPE_FILE = os.path.join(SCRIPT_DIR, "..", "data", "ne_110m_admin_0_countries", "ne_110m_admin_0_countries.shp")
DEFAULT_TEXTURE_SHAPE = (4, 16)


class MainMap(scene.Node):
    """Scene node for holding all of the information for the main map area
    """
    def __init__(self, *args, **kwargs):
        super(MainMap, self).__init__(*args, **kwargs)


class RetileEvent(Event):
    pass


class LayerList(scene.Node):
    """SceneGraph container for multiple image layers.
    """
    def __init__(self, name=None, parent=None):
        super(LayerList, self).__init__(name=name, parent=parent)

    def _timeout_slot(self, scheduler, ws=None):
        """Simple event handler for when we need to reassess.
        """
        # Stop the timer so it doesn't continously call this slot
        scheduler.stop()
        # Keep track of any children being updated
        update = False
        for child in self.children:
            need_retile, view_box, preferred_stride, tile_box = child.assess(ws)
            if need_retile:
                update = True
                LOG.debug("Retiling child '%s'", child.name)
                child.retile(ws, view_box, preferred_stride, tile_box)

        if update:
            # XXX: Should we update after every child?
            # draw any changes that were made
            self.update()

    def set_document(self, document):
        document.docDidChangeLayerOrder.connect(self.rebuild_new_order)
        document.docDidChangeLayer.connect(self.rebuild_layer_changed)

    def rebuild_new_order(self, new_layer_index_order, *args, **kwargs):
        """
        layer order has changed; shift layers around
        :param change:
        :return:
        """
        pass

    def rebuild_layer_changed(self, change_dict, *args, **kwargs):
        """
        document layer changed, update that layer
        :param change_dict: dictionary of change information
        :return:
        """
        if change_dict['change']=='add':  # a layer was added
            # add visuals to scene
            ds_info = change_dict['info']
            overview_content = change_dict['content']

            # create a new layer in the imagelist
            image = TiledGeolocatedImage(
                overview_content,
                ds_info["origin_x"],
                ds_info["origin_y"],
                ds_info["cell_width"],
                ds_info["cell_height"],
                name=ds_info["name"],
                clim=ds_info["clim"],
                interpolation='nearest',
                method='tiled',
                cmap='grays',
                double=False,
                texture_shape=DEFAULT_TEXTURE_SHAPE,
                wrap_lon=True,
                parent=self,
            )
        else:
            pass  # FIXME: other events? remove?

    def rebuild_all(self, *args, **kwargs):
        pass


class AnimatedLayerList(LayerList):
    def __init__(self, *args, **kwargs):
        super(AnimatedLayerList, self).__init__(*args, **kwargs)
        self._animating = False
        self._frame_number = 0
        self._animation_timer = app.Timer(1.0/10.0, connect=self.next_frame)
        # Make the newest child as the only visible node
        self.events.children_change.connect(self._set_visible_node)

    @property
    def animating(self):
        return self._animating

    @animating.setter
    def animating(self, animate):
        print("Running animating ", animate)
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

    def _set_visible_node(self, node):
        """Set all nodes to invisible except for the `event.added` node.
        """
        for child in self._children:
            with child.events.blocker():
                if child is node.added:
                    child.visible = True
                else:
                    child.visible = False

    def _set_visible_child(self, frame_number):
        for idx, child in enumerate(self._children):
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
        frame = frame_number if isinstance(frame_number, int) else (self._frame_number + 1) % len(self.children)
        self._set_visible_child(frame)
        self._frame_number = frame
        self.update()


class SceneGraphManager(object):
    def __init__(self, doc, workspace, border_shapefile=None, glob_pattern=None, parent=None, texture_shape=(4, 16)):
        # Parent should be the Qt widget that this GLCanvas belongs to
        self.parent = parent
        self.document = doc
        self.workspace = workspace
        self.border_shapefile = border_shapefile or DEFAULT_SHAPE_FILE
        self.glob_pattern = glob_pattern
        self.texture_shape = texture_shape

        self.image_layers = {}
        self.datasets = {}
        # FIXME:
        self._image_data = {}

        self.setup_initial_canvas()

    def setup_initial_canvas(self):
        self.main_canvas = CspovMainMapCanvas(parent=self.parent)
        self.main_view = self.main_canvas.central_widget.add_view(scene.PanZoomCamera(aspect=1))
        # Camera Setup
        self.main_view.camera.flip = (0, 0, 0)
        # FIXME: these ranges just happen to look ok, but I'm not really sure the 'best' way to set these
        self.main_view.camera.set_range(x=(-10.0, 10.0), y=(-10.0, 10.0), margin=0)
        self.main_view.camera.zoom(0.1, (0, 0))

        # Head node of the map graph
        self.main_map = MainMap(name="MainMap", parent=self.main_view.scene)
        merc_ortho = MatrixTransform()
        # near/far is backwards it seems:
        camera_z_scale = 1e-6
        # merc_ortho.set_ortho(-180.0, 180.0, -90.0, 90.0, -100.0 * camera_z_scale, 100.0 * camera_z_scale)
        l, r, b, t = [getattr(WORLD_EXTENT_BOX, x) for x in ['l', 'r', 'b', 't']]
        merc_ortho.set_ortho(l, r, b, t, -100.0 * camera_z_scale, 100.0 * camera_z_scale)
        self.main_map.transform *= merc_ortho # ortho(l, r, b, t, -100.0 * camera_z_scale, 100.0 * camera_z_scale)

        # Head node of the image layer graph
        # FIXME: merge to the document delegate
        self.image_list = AnimatedLayerList(parent=self.main_map)
        self.image_list.set_document(self.document)
        # Put all the images to the -50.0 Z level
        # TODO: Make this part of whatever custom Image class we make
        self.image_list.transform *= STTransform(translate=(0, 0, -50.0))
        # FIXME: merge to the document delegate

        self.boundaries = NEShapefileLines(self.border_shapefile, double=True, parent=self.main_map)

    def update(self):
        return self.main_canvas.update()

    # def on_layer_add(self, ds_info, full_data):
    #     uuid = ds_info["uuid"]
    #     image = TiledGeolocatedImage(
    #         full_data,
    #         ds_info["origin_x"],
    #         ds_info["origin_y"],
    #         ds_info["cell_width"],
    #         ds_info["cell_height"],
    #         name=ds_info["name"],
    #         clim=ds_info["clim"],
    #         interpolation='nearest',
    #         method='tiled',
    #         cmap='grays',
    #         texture_shape=self.texture_shape,
    #         wrap_lon=True,
    #         parent=self.image_list,  # FIXME move into document tilestack
    #     )
    #     # FIXME: Have the Z level configurable
    #     image.transform *= STTransform(translate=(0, 0, -50.0))
    #
    #     self.image_layers[uuid] = image
    #     self.datasets[uuid] = ds_info
    #     self._image_data[uuid] = full_data

    def on_view_change(self, scheduler, ws=None):
        """Simple event handler for when we need to reassess image layers.
        """
        # Stop the timer so it doesn't continously call this slot
        scheduler.stop()

        update = False
        for child in self.image_layers.values():
            need_retile, view_box, preferred_stride, tile_box = child.assess(ws)
            if need_retile:
                # Check if the workspace has this stride
                update = True
                LOG.debug("Retiling child '%s'", child.name)
                child.retile(ws, view_box, preferred_stride, tile_box)

        if update:
            self.update()

    def on_layer_visible_toggle(self, visible):
        pass

    def on_layer_change(self, event):
        pass

    def on_data_loaded(self, event):
        pass


