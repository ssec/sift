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
from cspov.view.LayerRep import TiledImageFile

__author__ = 'rayg'
__docformat__ = 'reStructuredText'


# from PyQt4.QtGui import *
# from PyQt4.QtCore import *
from vispy import app, gloo
# import vispy
# vispy.use(app='PyQt4') #, gl='gl+')

try:
    app_object = app.use_app('pyqt4')
except Exception:
    app_object = app.use_app('pyside')
QtCore = app_object.backend_module.QtCore
QtGui = app_object.backend_module.QtGui

from cspov.view.MapWidget import CspovMainMapWidget, LOG
from cspov.view.LayerDrawingPlan import ListWidgetChangesLayerStack, LayerDrawingPlan
from cspov.model import VizDoc

# this is generated with pyuic4 pov_main.ui >pov_main_ui.py
from cspov.ui.pov_main_ui import Ui_MainWindow

# behaviors
from cspov.control.file_behaviors import UserAddsFileToDoc

import logging, unittest, argparse

LOG = logging.getLogger(__name__)



def test_merc_layers(fn):
    # FIXME: pass in the Layers object rather than building it right here (test pattern style)
    raw_layers = []  # front to back
    LOG.info('loading {}'.format(fn))
    from .view.Program import GlooRGBImageTile, GlooColormapDataTile
    cls = GlooRGBImageTile if (fn is None or fn.endswith('.jpg') or fn.endswith('.png')) else GlooColormapDataTile
    # cls = GlooColormapDataTile if (fn is not None and fn.endswith('.tif')) else GlooRGBImageTile
    layer = TiledImageFile(filename=fn, tile_class=cls)
    # layer.set_alpha(0.5)
    return [layer]


def test_layers_from_directory(layer_tiff_glob, range_txt=None):
    """
    TIFF_GLOB='/Users/keoni/Data/CSPOV/2015_07_14_195/00?0/HS*_B03_*merc.tif' VERBOSITY=3 python -m cspov
    :param model:
    :param view:
    :param layer_tiff_glob:
    :return:
    """
    from glob import glob
    from .view.Program import GlooColormapDataTile
    layers = []
    range = None
    if range_txt:
        import re
        range = tuple(map(float, re.findall(r'[\.0-9]+', range_txt)))

    for tif in glob(layer_tiff_glob):
        layer = TiledImageFile(tif, tile_class=GlooColormapDataTile, range=range)
        layers.append(layer)
    return layers


def test_layers():
    if 'TIFF_GLOB' in os.environ:
        return test_layers_from_directory(os.environ['TIFF_GLOB'], os.environ.get('RANGE',None))
    elif 'MERC' in os.environ:
        return test_merc_layers(os.environ.get('MERC', None))
    return []


class Main(QtGui.QMainWindow):

    def _init_add_file_dialog(self):
        pass
        # self._b_adds_files = UserAddsFileToDoc(self, self.ui.)

    def __init__(self):
        super(Main, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        # refer to objectName'd entities as self.ui.objectName

        self.document = doc = VizDoc()


        self.mainMap = mainMap = CspovMainMapWidget(parent=self)
        self.ui.mainWidgets.addTab(self.mainMap.native, 'Mercator')
        mainMap.drawing_plan = plan = doc.asLayerDrawingPlan

        # FIXME: do this through document operations
        for layer in test_layers():
            plan.append(layer)

        self.ui.mainWidgets.removeTab(0)
        self.ui.mainWidgets.removeTab(0)

        # convey action between layer list
        # FIXME: use the document for this, not the drawing plan
        self.behaviorLayersList = ListWidgetChangesLayerStack(self.ui.layers, plan)
        # self.ui.layers

    def updateLayerList(self):
        self.ui.layers.add


if __name__ == '__main__':
    import os
    levels = [logging.ERROR, logging.WARN, logging.INFO, logging.DEBUG]
    verbosity = int(os.environ.get('VERBOSITY', 0))
    logging.basicConfig(level=levels[min(3, verbosity)])

    app.create()
    # app = QApplication(sys.argv)
    window = Main()
    window.show()
    print("running")
    app.run()
    # sys.exit(app.exec_())
#
