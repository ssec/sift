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
from cspov.control.layer_list import ListWidgetMatchesLayerStack
from cspov.model import Document

# this is generated with pyuic4 pov_main.ui >pov_main_ui.py
from cspov.ui.pov_main_ui import Ui_MainWindow

# behaviors
from cspov.control.file_behaviors import UserAddsFileToDoc

import logging, unittest, argparse

LOG = logging.getLogger(__name__)



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




class Main(QtGui.QMainWindow):

    def _init_add_file_dialog(self):
        pass
        # self._b_adds_files = UserAddsFileToDoc(self, self.ui.)

    def __init__(self):
        super(Main, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        # refer to objectName'd entities as self.ui.objectName

        # create document
        self.document = doc = Document()

        self.mainMap = mainMap = CspovMainMapWidget(parent=self)
        self.ui.mainWidgets.addTab(self.mainMap.native, 'Mercator')

        # callable which returns an iterable of LayerReps to draw
        mainMap.drawing_plan = doc.asDrawingPlan

        doc.addShapeLayer("/Users/davidh/Downloads/ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp")
        test_layers(doc)
        mainMap.update()

        # things to refresh the map window
        doc.docDidChangeLayerOrder.connect(mainMap.update)
        doc.docDidChangeEnhancement.connect(mainMap.update)
        doc.docDidChangeLayer.connect(mainMap.update)

        self.ui.mainWidgets.removeTab(0)
        self.ui.mainWidgets.removeTab(0)

        # convey action between layer list
        # FIXME: use the document for this, not the drawing plan
        self.behaviorLayersList = ListWidgetMatchesLayerStack(self.ui.layers, doc)
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
    # bring window to front
    window.raise_()
    app.run()
    # sys.exit(app.exec_())
#
