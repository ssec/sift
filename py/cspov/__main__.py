#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
.py
~~~

PURPOSE


REFERENCES


REQUIRES


:author: R.K.Garcia <rayg@ssec.wisc.edu>
:copyright: 2014 by University of Wisconsin Regents, see AUTHORS for more details
:license: GPLv3, see LICENSE for more details
"""
__author__ = 'rayg'
__docformat__ = 'reStructuredText'


from vispy import app
from cspov.queue import TaskQueue, test_task, TASK_PROGRESS, TASK_DOING
from cspov.workspace import Workspace

try:
    app_object = app.use_app('pyqt4')
except Exception:
    app_object = app.use_app('pyside')
QtCore = app_object.backend_module.QtCore
QtGui = app_object.backend_module.QtGui

from cspov.control.layer_list import LayerStackTableModel
from cspov.view.MapWidget import CspovMainMapCanvas
from cspov.view.LayerRep import NEShapefileLines, TiledGeolocatedImage
from cspov.model import Document
from cspov.common import WORLD_EXTENT_BOX
from functools import partial

# this is generated with pyuic4 pov_main.ui >pov_main_ui.py
from cspov.ui.pov_main_ui import Ui_MainWindow

import os
import logging
from vispy import scene
from vispy.visuals.transforms.linear import STTransform, MatrixTransform
from vispy.util.event import Event

LOG = logging.getLogger(__name__)
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
DEFAULT_SHAPE_FILE = os.path.join(SCRIPT_DIR, "data", "ne_110m_admin_0_countries", "ne_110m_admin_0_countries.shp")
DEFAULT_TEXTURE_SHAPE = (4, 16)
PROGRESS_BAR_MAX = 1000


def test_layers_from_directory(ws, doc, layer_tiff_glob, range_txt=None):
    """
    TIFF_GLOB='/Users/keoni/Data/CSPOV/2015_07_14_195/00?0/HS*_B03_*merc.tif' VERBOSITY=3 python -m cspov
    :param model:
    :param view:
    :param layer_tiff_glob:
    :return:
    """
    from glob import glob
    range = None
    if range_txt:
        import re
        range = tuple(map(float, re.findall(r'[\.0-9]+', range_txt)))
    for tif in glob(layer_tiff_glob):
        # doc.addFullGlobMercatorColormappedFloatImageLayer(tif, range=range)
        # uuid, info, overview_data = ws.import_image(tif)
        uuid, info, overview_data = doc.open_file(tif)
        LOG.info('loaded uuid {} from {}'.format(uuid, tif))
        yield uuid, info, overview_data


def test_layers(ws, doc, glob_pattern=None):
    if glob_pattern:
        return test_layers_from_directory(ws, doc, glob_pattern, os.environ.get('RANGE', None))
    LOG.warning("No image glob pattern provided")
    return []


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


class AnimatedLayerList(LayerList):
    def __init__(self, *args, **kwargs):
        super(AnimatedLayerList, self).__init__(*args, **kwargs)
        self._animating = False
        self._frame_number = 0
        self._animation_timer = app.Timer(1.0/10.0, connect=self.next_frame)
        # Make the newest child as the only visible node
        self.events.children_change.connect(self._set_visible_node)

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


class Main(QtGui.QMainWindow):
    def _init_add_file_dialog(self):
        pass
        # self._b_adds_files = UserAddsFileToDoc(self, self.ui.)

    def update_progress_bar(self, status_info, *args, **kwargs):
        active = status_info[0]
        LOG.warning('{0!r:s}'.format(status_info))
        val = active[TASK_PROGRESS]
        self.ui.progressBar.setValue(int(val*PROGRESS_BAR_MAX))
        #LOG.warning('progress bar updated to {}'.format(val))

    def __init__(self, workspace_dir=None, glob_pattern=None, border_shapefile=None):
        super(Main, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        # refer to objectName'd entities as self.ui.objectName

        self.queue = TaskQueue()
        self.ui.progressBar.setRange(0, PROGRESS_BAR_MAX)
        self.queue.didMakeProgress.connect(self.update_progress_bar)

        # create document
        self.workspace = Workspace(workspace_dir)
        self.document = doc = Document(self.workspace)

        self.main_canvas = CspovMainMapCanvas(parent=self)
        self.ui.mainWidgets.addTab(self.main_canvas.native, 'Mercator')

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
        self.image_list.set_document(doc)
        # Put all the images to the -50.0 Z level
        # TODO: Make this part of whatever custom Image class we make
        self.image_list.transform *= STTransform(translate=(0, 0, -50.0))

        self.boundaries = NEShapefileLines(border_shapefile, double=True, parent=self.main_map)

        # Create Layers
        texture_shape = DEFAULT_TEXTURE_SHAPE
        # tex_tiles_per_image = 16 * 8
        for uuid, ds_info, full_data in test_layers(self.workspace, self.document, glob_pattern=glob_pattern):
            # this now fires off a document modification cascade resulting in a new layer going up
            pass
            # add visuals to scene
            # image = TiledGeolocatedImage(
            #     full_data,
            #     ds_info["origin_x"],
            #     ds_info["origin_y"],
            #     ds_info["cell_width"],
            #     ds_info["cell_height"],
            #     name=ds_info["name"],
            #     clim=ds_info["clim"],
            #     interpolation='nearest',
            #     method='tiled',
            #     cmap='grays',
            #     double=False,
            #     texture_shape=texture_shape,
            #     wrap_lon=True,
            #     parent=self.image_list,  # FIXME move into document tilestack
            # )

        # Interaction Setup
        self.setup_key_releases()
        self.scheduler = QtCore.QTimer(parent=self)
        self.scheduler.setInterval(500.0)
        self.scheduler.timeout.connect(partial(self.image_list._timeout_slot, self.scheduler))
        def start_wrapper(timer, event):
            """Simple wrapper around a timers start method so we can accept but ignore the event provided
            """
            timer.start()
        self.main_canvas.events.draw.connect(partial(start_wrapper, self.scheduler))


        # things to refresh the map window
        # doc.docDidChangeLayerOrder.connect(main_canvas.update)
        # doc.docDidChangeEnhancement.connect(main_canvas.update)
        # doc.docDidChangeLayer.connect(main_canvas.update)

        self.ui.mainWidgets.removeTab(0)
        self.ui.mainWidgets.removeTab(0)

        # convey action between document and layer list view
        self.behaviorLayersList = LayerStackTableModel([self.ui.layerSet1Table, self.ui.layerSet2Table, self.ui.layerSet3Table, self.ui.layerSet4Table], doc)

        self.queue.add('test', test_task(), 'test000')
        # self.ui.layers
        # print(self.main_view.describe_tree(with_transform=True))

    def setup_key_releases(self):
        def cb_factory(required_key, cb):
            def tmp_cb(key, cb=cb):
                if key.text == required_key:
                    return cb()
            return tmp_cb

        self.main_canvas.events.key_release.connect(cb_factory("a", self.image_list.toggle_animation))
        self.main_canvas.events.key_release.connect(cb_factory("n", self.image_list.next_frame))

    def updateLayerList(self):
        # self.ui.layers.add
        pass


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run CSPOV")
    parser.add_argument("-w", "--workspace", default='.',
                        help="Specify workspace base directory")
    parser.add_argument("--border-shapefile", default=DEFAULT_SHAPE_FILE,
                        help="Specify alternative coastline/border shapefile")
    parser.add_argument("--glob-pattern", default=os.environ.get("TIFF_GLOB", None),
                        help="Specify glob pattern for input images")
    parser.add_argument('-v', '--verbose', dest='verbosity', action="count", default=int(os.environ.get("VERBOSITY", 2)),
                        help='each occurrence increases verbosity 1 level through ERROR-WARNING-INFO-DEBUG (default INFO)')
    args = parser.parse_args()

    levels = [logging.ERROR, logging.WARN, logging.INFO, logging.DEBUG]
    level=levels[min(3, args.verbosity)]
    logging.basicConfig(level=level)
    # logging.getLogger('vispy').setLevel(level)

    app.create()
    # app = QApplication(sys.argv)
    window = Main(
        workspace_dir=args.workspace,
        glob_pattern=args.glob_pattern,
        border_shapefile=args.border_shapefile
    )
    window.show()
    print("running")
    # bring window to front
    window.raise_()
    app.run()

if __name__ == '__main__':
    import sys
    sys.exit(main())
