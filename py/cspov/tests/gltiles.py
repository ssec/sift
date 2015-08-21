#!/usr/bin/env python
# encoding: utf-8
"""
All coordinates are (y,x) pairs

We assume a mercator projection using a WGS84 geoid
Scene coordinate are meters along the 0 parallel
y range +/- 6356752.3142 * 2 * pi => 39,940,652.742
x range +/- 6378137.0 * 2 * pi => 40,075,016.6856

"eqm" = equator-meter, we're using mercator projection as a baseline with

References
https://bitbucket.org/rndblnch/opengl-programmable/raw/29d0c699c82a2ca961014e7eb5e6cd3a87fe5883/05-shader.py

Layers
    can hold more than one level of detail (LayerRepresentation), only 1-2 of which are activated at any given time
    can paint to multiple GLWidgets
    render draw-lists and textures that draw a subset of the overall layer data, with a suitable level of detail


Rendering loop
    For each layer
        call Layer's paint, giving it information about the extents and sampling so it can pick best of the ready representations
        note whether the layer returned False and is requesting a re-render when the dust has settled => add to dirty layer list

Idle loop
    For each dirty layer
        Have layer render new draw list
        If layer has problems rendering, call a purge on all layers and then re-render


"""

import sys

from PyQt4.QtGui import *
from PyQt4.QtCore import *
import vispy
vispy.use(app='PyQt4') #, gl='gl+')

import vispy.scene as vps
from cspov.view.MapWidget import CspovMainMapWidget



class CoordSystem(object):
    """
    converts (y,x) pixel coordinates to geodetic coordinates in meters or degrees (lat/lon)
    """
    UNKNOWN = 0
    DATA_PIXELS = 1     # storage pixels for a given layer
    SCREEN_PIXELS = 2   # OpenGL pixels
    LATLON_DEGREES = 4  # lat/lon


class MainWindow(QMainWindow):

    significant = pyqtSignal(str)

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        # self.windowTitleChanged.connect(self.onWindowTitleChange)
        self.setWindowTitle("gltiles")
        # layout = QStackedLayout()
        widget = QTabWidget()
        # things = [QDateEdit, QLabel, QDial, QDoubleSpinBox, QSpinBox, QProgressBar, QSlider, QRadioButton, QTimeEdit, QFontComboBox, QLineEdit]
        self.scenegraph = vps.SceneCanvas('vispy', app='PyQt4')
        widget.addTab(self.scenegraph.native, 'vispy')

        things = [CspovMainMapWidget, QTextEdit]
        for w in things: 
            if w is QLabel:
                wid = QLabel('hola')
                font = wid.font()
                font.setPointSize(32)
                wid.setFont(font)
                wid.setAlignment(Qt.AlignHCenter)
            elif w is QLineEdit:
                q = QLineEdit()
                q.setPlaceholderText("I am text hear me roar")
                q.returnPressed.connect(self.return_pressed)
                q.selectionChanged.connect(self.selection_changed)
                q.textChanged.connect(self.text_changed)
                q.textEdited.connect(self.text_edited)
                self.line = q
                wid = q
            else:
                wid = w()
            # layout.addWidget(wid)

            widget.addTab(wid, str(w.__name__))

        widget.setCurrentIndex(1)

        self.setCentralWidget(widget)
        self.significant.connect(self.on_my_signal)

    def contextMenuEvent(self, e):
        print("context menu")
        # maj,min = glGetIntegerv(GL_MAJOR_VERSION), glGetIntegerv(GL_MINOR_VERSION)
        # print("OpenGL {}.{}".format(maj,min))

        super(MainWindow, self).contextMenuEvent(e)  # can also use e.accept() or e.ignore()

    def return_pressed(self):
        print('return pressed')
        self.line.setText("BOOM")

    def selection_changed(self):
        print('selection changed')

    def text_edited(self):
        print("text edited to " + repr(self.line.text()))

    def text_changed(self):
        print("text changed to " + repr(self.line.text()))

    def on_golden_pond(self, a):
        print(a)

    def on_button_pressed(self):
        print("bort")
        self.significant.emit('-bort-')

    def on_my_signal(self, s):
        print('there has been {0:s}'.format(s))


    # def onWindowTitleChange(self, s):
    #     print(s)


if __name__=='__main__':
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    app.exec_()
