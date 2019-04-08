#!/usr/bin/env python
import sys
import time
from random import randint, shuffle

from PyQt5.QtWidgets import (QGraphicsScene, QMainWindow, QToolBar, QAction, QLabel, QCheckBox, QStatusBar,
                             QGraphicsView, QGraphicsTextItem, QApplication)
from PyQt5.QtCore import QSize, QTimer, Qt, QTimeLine, QPointF
from PyQt5.QtGui import QKeySequence, QIcon, QFont
from PyQt5.QtOpenGL import QGLWidget, QGLFormat, QGL


# http://pyqt.sourceforge.net/Docs/PyQt4/modules.html
# from PyQt4.QtWidgets import *
# ref https://ralsina.me/stories/BBS53.html

class DemoScene(QGraphicsScene):

    def __init__(self, *args, **kwargs):
        super(DemoScene, self).__init__(*args, **kwargs)


class MainWindow(QMainWindow):
    _scene = None
    _gfx = None

    def __init__(self, scene, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        # self.windowTitleChanged.connect(self.onWindowTitleChange)
        self.setWindowTitle("timeline0")

        toolbar = QToolBar("och")
        toolbar.setIconSize(QSize(20, 20))
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

        self._scene = scene
        gfx = self._gfx = QGraphicsView(self)
        # label = QLabel("och!")
        # label.setAlignment(Qt.AlignCenter)

        # ref https://doc.qt.io/archives/qq/qq26-openglcanvas.html
        self.setCentralWidget(gfx)
        fmt = QGLFormat(QGL.SampleBuffers)
        wdgt = QGLWidget(fmt)
        assert (wdgt.isValid())
        gfx.setViewport(wdgt)
        gfx.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        gfx.setScene(scene)

        # populate fills the scene with interesting stuff.
        self.populate()

        # Make it bigger
        self.setWindowState(Qt.WindowMaximized)

        # Well... it's going to have an animation, ok?

        # So, I set a timer to 1 second
        self.animator = QTimer()

        # And when it triggers, it calls the animate method
        self.animator.timeout.connect(self.animate)

        # And I animate it once manually.
        self.animate()

    def animate(self):

        # Just a list with 60 positions
        self.animations = [None] * 60  # list(range(0, 60))

        # This is the only "hard" part
        # Given an item, and where you want it to be
        # it moves it there, smoothly, in one second.
        def animate_to(t, item, x, y, angle):
            # The QGraphicsItemAnimation class is used to
            # animate an item in specific ways
            # FIXME QGraphicsItemAnimation class is no longer supported.
            animation = QGraphicsItemAnimation()

            # You create a timeline (in this case, it is 1 second long
            timeline = QTimeLine(1000)

            # And it has 100 steps
            timeline.setFrameRange(0, 100)

            # I want that, at time t, the item be at point x,y
            animation.setPosAt(t, QPointF(x, y))

            # And it should be rotated at angle "angle"
            animation.setRotationAt(t, angle)

            # It should animate this specific item
            animation.setItem(item)

            # And the whole animation is this long, and has
            # this many steps as I set in timeline.
            animation.setTimeLine(timeline)

            # Here is the animation, use it.
            return animation

        # Ok, I confess it, this part is a mess, but... a little
        # mistery is good for you. Read this carefully, and tell
        # me if you can do it better. Or try to something nicer!

        offsets = list(range(6))
        shuffle(offsets)

        # Some items, animate with purpose
        h1, h2 = map(int, '%02d' % time.localtime().tm_hour)
        h1 += offsets[0] * 10
        h2 += offsets[1] * 10
        self.animations[h1] = animate_to(0.2, self.digits[h1], -40, 0, 0)
        self.animations[h2] = animate_to(0.2, self.digits[h2], 50, 0, 0)

        m1, m2 = map(int, '%02d' % time.localtime().tm_min)
        m1 += offsets[2] * 10
        m2 += offsets[3] * 10
        self.animations[m1] = animate_to(0.2, self.digits[m1], 230, 0, 0)
        self.animations[m2] = animate_to(0.2, self.digits[m2], 320, 0, 0)

        s1, s2 = map(int, '%02d' % time.localtime().tm_sec)
        s1 += offsets[4] * 10
        s2 += offsets[5] * 10
        self.animations[s1] = animate_to(0.2, self.digits[s1], 500, 0, 0)
        self.animations[s2] = animate_to(0.2, self.digits[s2], 590, 0, 0)

        # Other items, animate randomly
        for i in range(len(self.animations)):
            t_item = self.digits[i]
            if i in [h1, h2, m1, m2, s1, s2]:
                t_item.setOpacity(1)
                continue
            t_item.setOpacity(.3)
            self.animations[i] = animate_to(1, t_item, randint(0, 500), randint(0, 300), randint(0, 0))

        [animation.timeLine().start() for animation in self.animations]

        self.animator.start(1000)

    def onMyToolBarButtonClick(self, s):
        print("click", s)

    # def onWindowTitleChange(self, s):
    #     print(s)

    def populate(self):
        self.digits = []
        self.animations = []

        # This is just a nice font, use any font you like, or none
        font = QFont('White Rabbit')
        font.setPointSize(120)

        # Create three ":" and place them in our scene
        self.dot1 = QGraphicsTextItem(':')  # from QtGui
        self.dot1.setFont(font)
        self.dot1.setPos(140, 0)
        self._scene.addItem(self.dot1)
        self.dot2 = QGraphicsTextItem(':')
        self.dot2.setFont(font)
        self.dot2.setPos(410, 0)
        self._scene.addItem(self.dot2)

        # Create 6 sets of 0-9 digits
        for i in range(60):
            t_item = QGraphicsTextItem(str(i % 10))
            t_item.setFont(font)
            # The zvalue is what controls what appears "on top" of what.
            # Send them to "the bottom" of the scene.
            t_item.setZValue(-100)

            # Place them anywhere
            t_item.setPos(randint(0, 500), randint(150, 300))

            # Make them semi-transparent
            t_item.setOpacity(.3)

            # Put them in the scene
            self._scene.addItem(t_item)

            # Keep a reference for internal purposes
            self.digits.append(t_item)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    scene = DemoScene()
    window = MainWindow(scene)
    window.show()

    app.exec_()
