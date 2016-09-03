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
        toolbar.setIconSize(QSize(20,20))
        self.addToolBar(toolbar)

        button_action = QAction(QIcon("balance.png"), "ochtuse", self)
        button_action.setStatusTip("och, just do something")
        button_action.triggered.connect(self.onMyToolBarButtonClick)
        button_action.setCheckable(True)
        # button_action.setShortcut(QKeySequence("Ctrl+p"))
        # button_action.setShortcut(QKeySequence(Qt.CTRL + Qt.Key_P))
        button_action.setShortcut(QKeySequence.Print)
        toolbar.addAction(button_action)
        toolbar.addWidget(QLabel("OCH"))
        toolbar.addWidget(QCheckBox())

        self.setStatusBar(QStatusBar(self))


        menu = self.menuBar()

        file_menu = menu.addMenu("&File")
        file_menu.addAction(button_action)
        file_menu.addSeparator()
        file_menu.addMenu("Do not push")
#        file_menu.addAction()

    def onMyToolBarButtonClick(self, s):
        print("click", s)

    # def onWindowTitleChange(self, s):
    #     print(s)


app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec_()
