#!/usr/bin/env python
import sys
from PyQt4.QtGui import *
from PyQt4.QtCore import *

# http://pyqt.sourceforge.net/Docs/PyQt4/modules.html
#from PyQt4.QtWidgets import *


class MainWindow(QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        # self.windowTitleChanged.connect(self.onWindowTitleChange)
        self.setWindowTitle("simple0")

        label = QLabel("och!")
        label.setAlignment(Qt.AlignCenter)
        self.setCentralWidget(label)

        toolbar = QToolBar("och")
        self.addToolBar(toolbar)

        button_action = QAction("ochtuse", self)
        button_action.setStatusTip("och, just do something")
        button_action.triggered.connect(self.onMyToolBarButtonClick)
        toolbar.addAction(button_action)

        self.setStatusBar(QStatusBar(self))


    def onMyToolBarButtonClick(self, s):
        print("click", s)

    # def onWindowTitleChange(self, s):
    #     print(s)


app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec_()
