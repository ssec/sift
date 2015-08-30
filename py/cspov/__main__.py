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


# from PyQt4.QtGui import *
# from PyQt4.QtCore import *
from vispy import app, gloo
# import vispy
# vispy.use(app='PyQt4') #, gl='gl+')

try:
    app_object = app.use_app('pyqt4')
except Exception:
    app_object = app.use_app('pyside')
QtCore = app_object.backend_module.QtCore,
QtGui = app_object.backend_module.QtGui


from cspov.view.MapWidget import CspovMainMapWidget


import logging, unittest, argparse

LOG = logging.getLogger(__name__)


# this is generated with pyuic4 pov_main.ui >pov_main_ui.py
from cspov.ui.pov_main_ui import Ui_MainWindow

class Main(QtGui.QMainWindow):
    def __init__(self):
        super(Main, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        # refer to objectName'd entities as self.ui.objectName

        self.mainMap = CspovMainMapWidget(parent=self)
        self.ui.mainWidgets.addTab(self.mainMap.native, 'Mercator')
        # self.mainMap.setVisible(True)

        # self.scenegraph = vps.SceneCanvas('vispy', app='PyQt4')
        # self.ui.mainWidgets.addTab(self.scenegraph.native, 'Tools')

        self.ui.mainWidgets.removeTab(0)
        self.ui.mainWidgets.removeTab(0)

if __name__ == '__main__':
    app.create()
    # app = QApplication(sys.argv)
    window = Main()
    window.show()
    print("running")
    app.run()
    # sys.exit(app.exec_())
#
