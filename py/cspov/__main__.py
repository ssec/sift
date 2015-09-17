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
try:
    app_object = app.use_app('pyqt4')
except Exception:
    app_object = app.use_app('pyside')
QtCore = app_object.backend_module.QtCore
QtGui = app_object.backend_module.QtGui

from cspov.view.MapWidget import CspovMainMapWidget, CspovMainMapCanvas
from cspov.view.LayerRep import NEShapefileLayer
from cspov.model import Document
from cspov.common import (DEFAULT_X_PIXEL_SIZE,
                          DEFAULT_Y_PIXEL_SIZE,
                          DEFAULT_ORIGIN_X,
                          DEFAULT_ORIGIN_Y,
                          WORLD_EXTENT_BOX,
                          )

# this is generated with pyuic4 pov_main.ui >pov_main_ui.py
from cspov.ui.pov_main_ui import Ui_MainWindow

import os
import logging
from vispy import scene, visuals
from vispy.io import imread
from vispy.visuals.transforms.linear import MatrixTransform, STTransform
import numpy as np

LOG = logging.getLogger(__name__)
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
DEFAULT_SHAPE_FILE = os.path.join(SCRIPT_DIR, "data", "ne_110m_admin_0_countries", "ne_110m_admin_0_countries.shp")


def test_merc_layers(doc, fn):
    # FIXME: pass in the Layers object rather than building it right here (test pattern style)
    raw_layers = []  # front to back
    LOG.info('loading {}'.format(fn))
    doc.addRGBImageLayer(fn)


def test_layers_from_directory(doc, layer_tiff_glob, range_txt=None):
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
        doc.addFullGlobMercatorColormappedFloatImageLayer(tif, range=range)


def test_layers(doc):
    if 'TIFF_GLOB' in os.environ:
        return test_layers_from_directory(doc, os.environ['TIFF_GLOB'], os.environ.get('RANGE',None))
    elif 'MERC' in os.environ:
        return test_merc_layers(doc, os.environ.get('MERC', None))
    return []


class MainMap(scene.Node):
    """Scene node for holding all of the information for the main map area
    """
    def __init__(self, *args, **kwargs):
        super(MainMap, self).__init__(*args, **kwargs)
        self.events.key_release.connect(self.on_key_release)

    def on_key_release(self, key):
        print("Key release")
        print(key)
        if key.text == "a":
            print("Got 'a'")


class LayerList(scene.Node):
    def __init__(self, name=None, parent=None):
        super(LayerList, self).__init__(name=name, parent=parent)
        # children are usually added by specifying a LayerList as their parent Node


class AnimatedLayerList(LayerList):
    def __init__(self, *args, **kwargs):
        super(AnimatedLayerList, self).__init__(*args, **kwargs)
        self._animating = False
        self._frame_number = 0
        self._animation_timer = app.Timer(1.0/10.0, connect=self.next_frame)

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


class ImageLayerVisual(visuals.ImageVisual):
    """CSPOV Image Layer

    Note: VisPy separates this with a ImageVisual and a dynamically created scene.visuals.Image.
    """
    def __init__(self, image_filepath, **kwargs):
        img_data = imread(image_filepath)
        super(ImageLayerVisual, self).__init__(img_data, **kwargs)

    def _build_vertex_data(self):
        """Rebuild the vertex buffers used for rendering the image when using
        the subdivide method.

        CSPOV Note: Copied from 0.5.0dev original ImageVisual class
        """
        grid = self._grid
        w = 1.0 / grid[1]
        h = 1.0 / grid[0]

        quad = np.array([[0, 0, 0], [w, 0, 0], [w, h, 0],
                         [0, 0, 0], [w, h, 0], [0, h, 0]],
                        dtype=np.float32)
        quads = np.empty((grid[1], grid[0], 6, 3), dtype=np.float32)
        quads[:] = quad

        mgrid = np.mgrid[0.:grid[1], 0.:grid[0]].transpose(1, 2, 0)
        mgrid = mgrid[:, :, np.newaxis, :]
        mgrid[..., 0] *= w
        mgrid[..., 1] *= h

        quads[..., :2] += mgrid
        tex_coords = quads.reshape(grid[1]*grid[0]*6, 3)
        tex_coords = np.ascontiguousarray(tex_coords[:, :2])
        vertices = tex_coords * self.size
        # vertices = tex_coords

        # TODO: Read the source to figure out how the vertices are actually used
        # FIXME: temporary hack to get the image geolocation (should really be determined by the geo info in the image file or metadata
        vertices = vertices.astype('float32')
        vertices[:, 0] *= DEFAULT_X_PIXEL_SIZE
        vertices[:, 0] += DEFAULT_ORIGIN_X
        vertices[:, 1] *= DEFAULT_Y_PIXEL_SIZE
        vertices[:, 1] += DEFAULT_ORIGIN_Y
        self._subdiv_position.set_data(vertices.astype('float32'))
        self._subdiv_texcoord.set_data(tex_coords.astype('float32'))

# XXX: Doing the subclassing myself causes some inheritance problems for some reason
ImageLayer = scene.visuals.create_visual_node(ImageLayerVisual)


class DatasetInfo(dict):
    pass


class Workspace(object):
    def __init__(self, base_dir):
        if not os.path.isdir(base_dir):
            raise IOError("Workspace '%s' does not exist" % (base_dir,))
        self.base_dir = os.path.realpath(base_dir)

    def get_dataset_info(self, item, time_step=None, resolution=None):
        if resolution is not None:
            raise NotImplementedError("Resolution can not be specified yet")

        # FIXME: Workspace structure
        # 'item_start_time' is a string representing the directory name for the string to get
        fn_pat = "HS_H08_20150714_{}_{}_FLDK_R20.merc.tif"
        item_path = os.path.join(self.base_dir, time_step, fn_pat.format(time_step, item))

        dataset_info = DatasetInfo()
        dataset_info["filepath"] = item_path
        return dataset_info


class Main(QtGui.QMainWindow):
    def _init_add_file_dialog(self):
        pass
        # self._b_adds_files = UserAddsFileToDoc(self, self.ui.)

    def __init__(self, workspace_dir=None, border_shapefile=None):
        super(Main, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        # refer to objectName'd entities as self.ui.objectName

        # create document
        self.document = doc = Document()
        self.workspace = Workspace(workspace_dir)

        self.main_canvas = CspovMainMapCanvas(parent=self)
        self.ui.mainWidgets.addTab(self.main_canvas.native, 'Mercator')
        self.main_view = self.main_canvas.central_widget.add_view()
        # Head node of the map graph
        self.main_map = MainMap(name="MainMap", parent=self.main_view.scene)
        merc_ortho = MatrixTransform()
        # near/far is backwards it seems:
        camera_z_scale = 1e-6
        # merc_ortho.set_ortho(-180.0, 180.0, -90.0, 90.0, -100.0 * camera_z_scale, 100.0 * camera_z_scale)
        l, r, b, t = [getattr(WORLD_EXTENT_BOX, x) for x in ['l', 'r', 'b', 't']]
        merc_ortho.set_ortho(l, r, b, t, -100.0 * camera_z_scale, 100.0 * camera_z_scale)
        self.main_map.transform *= merc_ortho
        # Head node of the image layer graph
        self.image_list = AnimatedLayerList(parent=self.main_map)
        # Put all the images to the -50.0 Z level
        # TODO: Make this part of whatever custom Image class we make
        self.image_list.transform *= STTransform(translate=(0, 0, -50.0))

        self.boundaries = NEShapefileLayer(border_shapefile, double=True, parent=self.main_map)

        # Create Layers
        for time_step in ["0330", "0340"]:
            ds_info = self.workspace.get_dataset_info("B02", time_step=time_step)
            image = ImageLayer(ds_info["filepath"], interpolation='nearest', method='subdivide', grid=(20, 20), parent=self.image_list)

        # Interaction Setup
        self.setup_key_releases()

        # Camera Setup
        self.main_view.camera = scene.PanZoomCamera(aspect=1)
        self.main_view.camera.flip = (0, 0, 0)
        # range limits are subject to zoom fraction (I think?)
        self.main_view.camera.set_range(x=(-10.0, 10.0), y=(-10.0, 10.0), margin=0)
        self.main_view.camera.zoom(0.1, (0, 0))

        # things to refresh the map window
        # doc.docDidChangeLayerOrder.connect(main_canvas.update)
        # doc.docDidChangeEnhancement.connect(main_canvas.update)
        # doc.docDidChangeLayer.connect(main_canvas.update)

        self.ui.mainWidgets.removeTab(0)
        self.ui.mainWidgets.removeTab(0)

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
        self.ui.layers.add


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run CSPOV")
    parser.add_argument("-w", "--workspace", default='.',
                        help="Specify workspace base directory")
    parser.add_argument("--border-shapefile", default=DEFAULT_SHAPE_FILE,
                        help="Specify alternative coastline/border shapefile")
    parser.add_argument('-v', '--verbose', dest='verbosity', action="count", default=0,
                        help='each occurrence increases verbosity 1 level through ERROR-WARNING-INFO-DEBUG (default INFO)')
    args = parser.parse_args()

    levels = [logging.ERROR, logging.WARN, logging.INFO, logging.DEBUG]
    logging.basicConfig(level=levels[min(3, args.verbosity)])

    app.create()
    # app = QApplication(sys.argv)
    window = Main(workspace_dir=args.workspace, border_shapefile=args.border_shapefile)
    window.show()
    print("running")
    # bring window to front
    window.raise_()
    app.run()

if __name__ == '__main__':
    import sys
    sys.exit(main())
