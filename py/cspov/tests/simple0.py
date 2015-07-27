#!/usr/bin/env python
import sys
from PyQt4.QtGui import *
from PyQt4.QtCore import *
#from PyQt4.QtWidgets import *


class MainWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setWindowTitle("simple0")
        label = QLabel('this is simple')
        label.setAlignment(Qt.AlignCenter)
        self.setCentralWidget(label)


app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec_()
