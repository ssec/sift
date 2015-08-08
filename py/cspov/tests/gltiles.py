#!/usr/bin/env python
import sys
from PyQt4.QtGui import *
from PyQt4.QtCore import *

# http://pyqt.sourceforge.net/Docs/PyQt4/modules.html
#from PyQt4.QtWidgets import *

class GLTile(object):
    """
    A lazy-loaded image which can be mapped to a texture buffer and drawn on a polygon
    Represents a single x-y coordinate range at a single level of detail
    """


class GLTileArray(object):
    """
    A single level of detail with lazy loading of tiles
    Tiles may or may not be available
    Tiles follow a predictable lookup
    """
    _drawlist = None 

    def __init__(self, index, lod):
        """
        index is a mapping which { (0,0): GLTile(), ... }
        """
        pass

    def get_drawlist(self):
        return self._drawlist

    def set_drawlist(self, number):
        self._drawlist = number

    gldrawlist = property(get_drawlist, set_drawlist)


    def tileseq_in_area(self, index_bltr): 
        """
        yield the sequence of tiles in a given rectangular tileseq_in_area
        """
        pass

    def tileseq_visible(self, data_bltr): 
        """
        given data coordinates, determine which tiles are on the canvas
        """
        pass

    def glLoadTextures(self, tile_seq):
        pass

    def glDraw(self, tile_seq, to_drawlist=None):
        """
        send drawing commands for a sequence of tiles

        """

    def glRedraw(self, data_bltr):
        """
        if bltr is inside most recently drawn bltr, redraw with draw list
        otherwise do a full draw 
        """


class LayerStack(object):
    """
    manage stack of layers including tiles, vectors, overlays etc
    """



class MainWindow(QMainWindow):

    significant = pyqtSignal(str)

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        # self.windowTitleChanged.connect(self.onWindowTitleChange)
        self.setWindowTitle("gltiles")
        # layout = QStackedLayout()
        widget = QTabWidget()
        things = [QDateEdit, QLabel, QDial, QDoubleSpinBox, QSpinBox, QProgressBar, QSlider, QRadioButton, QTimeEdit, QFontComboBox, QLineEdit]
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
            widget.addTab(wid, str(w))

        self.setCentralWidget(widget)
        self.significant.connect(self.on_my_signal)

    def contextMenuEvent(self, e):
        print("context menu")
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
