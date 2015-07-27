#!/usr/bin/env python
import sys
from PyQt4.QtGui import *
from PyQt4.QtCore import *

# http://pyqt.sourceforge.net/Docs/PyQt4/modules.html
#from PyQt4.QtWidgets import *


class MainWindow(QMainWindow):

    significant = pyqtSignal(str)

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        # self.windowTitleChanged.connect(self.onWindowTitleChange)
        self.setWindowTitle("simple0")
        layout = QHBoxLayout()
        for n in range(10):
            btn = QPushButton(str(n))    
            # btn.pressed.connect(lambda n=n: self.on_golden_pond(n))
            btn.pressed.connect(self.on_button_pressed)
            layout.addWidget(btn)
        widget = QWidget() 
        widget.setLayout(layout)
        self.setCentralWidget(widget)
        self.significant.connect(self.on_my_signal)

    def on_golden_pond(self, a):
        print(a)

    def on_button_pressed(self):
        print("bort")
        self.significant.emit('-bort-')

    def on_my_signal(self, s):
        print('there has been {0:s}'.format(s))


    # def onWindowTitleChange(self, s):
    #     print(s)


app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec_()
