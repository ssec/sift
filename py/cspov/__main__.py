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
from .queue import TaskQueue, test_task, TASK_PROGRESS, TASK_DOING
from .workspace import Workspace

try:
    app_object = app.use_app('pyqt4')
except Exception:
    app_object = app.use_app('pyside')
QtCore = app_object.backend_module.QtCore
QtGui = app_object.backend_module.QtGui

from cspov.control.layer_list import LayerStackListViewModel
from cspov.view.MapWidget import CspovMainMapCanvas
from cspov.view.LayerRep import NEShapefileLines, TiledGeolocatedImage
from cspov.model import Document
from cspov.common import WORLD_EXTENT_BOX, DEFAULT_TILE_HEIGHT, DEFAULT_TILE_WIDTH

# this is generated with pyuic4 pov_main.ui >pov_main_ui.py
from cspov.ui.pov_main_ui import Ui_MainWindow

import os
import logging
import gdal
from vispy import scene
from vispy.visuals.transforms.linear import STTransform, MatrixTransform
# from vispy.util.transforms import translate, ortho

LOG = logging.getLogger(__name__)
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
DEFAULT_SHAPE_FILE = os.path.join(SCRIPT_DIR, "data", "ne_110m_admin_0_countries", "ne_110m_admin_0_countries.shp")



def test_merc_layers(ws, doc, fn):
    raise NotImplementedError('deprecated')
    # raw_layers = []  # front to back
    # LOG.info('loading {}'.format(fn))
    # doc.addRGBImageLayer(fn)


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
        uuid, info, overview_data = ws.import_image(tif)
        yield uuid, info, overview_data


def test_layers(ws, doc):
    if 'TIFF_GLOB' in os.environ:
        return test_layers_from_directory(ws, doc, os.environ['TIFF_GLOB'], os.environ.get('RANGE',None))
    elif 'MERC' in os.environ:
        return test_merc_layers(ws, doc, os.environ.get('MERC', None))
    return []


PROGRESS_BAR_MAX = 1000

class MainMap(scene.Node):
    """Scene node for holding all of the information for the main map area
    """
    def __init__(self, *args, **kwargs):
        super(MainMap, self).__init__(*args, **kwargs)


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


class DatasetInfo(dict):
    pass

#
# # FIXME: Workspace structure
# class Workspace(object):
#     def __init__(self, base_dir):
#         if not os.path.isdir(base_dir):
#             raise IOError("Workspace '%s' does not exist" % (base_dir,))
#         self.base_dir = os.path.realpath(base_dir)
#
#     def _dataset_filepath(self, item, time_step):
#         fn_pat = "HS_H08_20150714_{}_{}_FLDK_R20.merc.tif"
#         # 'time_step' is a string representing the directory name for the string to get
#         return os.path.join(self.base_dir, time_step, fn_pat.format(time_step, item))
#
#     def _get_dataset_projection_info(self, fp):
#         d = {}
#         if fp.endswith(".tif"):
#             gtiff = gdal.Open(fp)
#             ox, cw, _, oy, _, ch = gtiff.GetGeoTransform()
#             d["origin_x"] = ox
#             d["origin_y"] = oy
#             d["cell_width"] = cw
#             d["cell_height"] = ch
#             # FUTURE: Should the Workspace normalize all input data or should the Image Layer handle any projection?
#             srs = gdal.osr.SpatialReference()
#             srs.ImportFromWkt(gtiff.GetProjection())
#             d["proj"] = srs.ExportToProj4()
#         else:
#             raise ValueError("Unknown workspace format detected: %s" % (fp,))
#
#         return d
#
#     def get_dataset_info(self, item, time_step):
#         # Extra little 'fluff' so that 0 in the texture can be used for fill values
#         fill_margin = 0.005
#         dataset_info = DatasetInfo()
#         # FIXME: Make up a better name
#         dataset_info["name"] = item + "_" + time_step
#         dataset_info["filepath"] = self._dataset_filepath(item, time_step)
#
#         # Valid min and max for colormap use
#         if item in ["B01", "B02", "B03", "B04", "B05", "B06"]:
#             # Reflectance/visible data limits
#             # FIXME: Are these correct?
#             dataset_info["clim"] = (0.0 - fill_margin, 1.0)
#         else:
#             # BT data limits
#             # FIXME: Are these correct?
#             dataset_info["clim"] = (200.0 - fill_margin, 350.0)
#
#         # Full resolution shape
#         dataset_info["shape"] = self.get_dataset_data(item, time_step).shape
#
#         dataset_info.update(self._get_dataset_projection_info(dataset_info["filepath"]))
#         return dataset_info
#
#     def get_dataset_data(self, item, time_step, row_slice=None, col_slice=None):
#         fp = self._dataset_filepath(item, time_step)
#
#         if fp.endswith(".tif"):
#             gtiff = gdal.Open(fp)
#             img_data = gtiff.GetRasterBand(1).ReadAsArray()
#         else:
#             raise ValueError("Unknown workspace format detected: %s" % (fp,))
#
#         if row_slice is None or col_slice is None:
#             return img_data
#         else:
#             return img_data[row_slice, col_slice]


class Main(QtGui.QMainWindow):
    def _init_add_file_dialog(self):
        pass
        # self._b_adds_files = UserAddsFileToDoc(self, self.ui.)

    def update_progress_bar(self, status_info, *args, **kwargs):  # FIXME no workie
        active = status_info[0]
        LOG.warning('{0!r:s}'.format(status_info))
        val = active[TASK_PROGRESS]
        self.ui.progressBar.setValue(int(val*PROGRESS_BAR_MAX))
        LOG.warning('progress bar updated to {}'.format(val))


    def __init__(self, workspace_dir=None, border_shapefile=None):
        super(Main, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        # refer to objectName'd entities as self.ui.objectName

        self.queue = TaskQueue()
        self.ui.progressBar.setRange(0, PROGRESS_BAR_MAX)
        self.queue.didMakeProgress.connect(self.update_progress_bar)

        # create document
        self.document = doc = Document()
        self.workspace = Workspace(workspace_dir)

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
        # Put all the images to the -50.0 Z level
        # TODO: Make this part of whatever custom Image class we make
        self.image_list.transform *= STTransform(translate=(0, 0, -50.0))

        self.boundaries = NEShapefileLines(border_shapefile, double=True, parent=self.main_map)

        # Create Layers
        tex_tiles_per_image = 32
        # tex_tiles_per_image = 16 * 8
        for uuid, ds_info, full_data in test_layers(self.workspace, self.document):
            # add visuals to scene
            image = TiledGeolocatedImage(
                full_data,
                ds_info["origin_x"],
                ds_info["origin_y"],
                ds_info["cell_width"],
                ds_info["cell_height"],
                name=ds_info["name"],
                clim=ds_info["clim"],
                interpolation='nearest',
                method='tiled',
                double=False,
                num_tiles=tex_tiles_per_image,
                parent=self.image_list,  # FIXME move into document tilestack
            )

        # Interaction Setup
        self.setup_key_releases()

        # things to refresh the map window
        # doc.docDidChangeLayerOrder.connect(main_canvas.update)
        # doc.docDidChangeEnhancement.connect(main_canvas.update)
        # doc.docDidChangeLayer.connect(main_canvas.update)

        self.ui.mainWidgets.removeTab(0)
        self.ui.mainWidgets.removeTab(0)

        # convey action between document and layer list view
        # self.behaviorLayersList = LayerStackListViewModel(self.ui.layers, doc)

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
