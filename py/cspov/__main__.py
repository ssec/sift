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


from vispy import app
try:
    app_object = app.use_app('pyqt4')
except Exception:
    app_object = app.use_app('pyside')
QtCore = app_object.backend_module.QtCore
QtGui = app_object.backend_module.QtGui

from cspov.control.layer_list import LayerStackListViewModel
from cspov.model import Document
from cspov.view.SceneGraphManager import SceneGraphManager
from cspov.queue import TaskQueue, test_task, TASK_PROGRESS, TASK_DOING
from cspov.workspace import Workspace
from vispy.color.colormap import Colormap

from functools import partial

# this is generated with pyuic4 pov_main.ui >pov_main_ui.py
from cspov.ui.pov_main_ui import Ui_MainWindow
from cspov.common import INFO, KIND

import os
import logging

# http://stackoverflow.com/questions/12459811/how-to-embed-matplotib-in-pyqt-for-dummies
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
import matplotlib.pyplot as plt


LOG = logging.getLogger(__name__)
PROGRESS_BAR_MAX = 1000


def test_layers_from_directory(ws, doc, layer_tiff_glob, range_txt=None):
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
        # doc.addFullGlobMercatorColormappedFloatImageLayer(tif, range=range)
        # uuid, info, overview_data = ws.import_image(tif)
        uuid, info, overview_data = doc.open_file(tif)
        LOG.info('loaded uuid {} from {}'.format(uuid, tif))
        yield uuid, info, overview_data


def test_layers(ws, doc, glob_pattern=None):
    if glob_pattern:
        return test_layers_from_directory(ws, doc, glob_pattern, os.environ.get('RANGE', None))
    LOG.warning("No image glob pattern provided")
    return []


class Main(QtGui.QMainWindow):
    _last_open_dir = None  # directory to open files in

    def make_mpl_pane(self, parent):
        """place a matplotlib figure inside a probe pane
        """
        # a figure instance to plot on
        figure = plt.figure(figsize=(3,3), dpi=72)

        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        canvas = FigureCanvas(figure, )

        # this is the Navigation widget
        # it takes the Canvas widget and a parent
        toolbar = None # NavigationToolbar(canvas, self)

        # Just some button connected to `plot` method
        # self.button = QtGui.QPushButton('Plot')
        # self.button.clicked.connect(self.plot)

        # set the layout
        layout = QtGui.QVBoxLayout()
        # layout.addWidget(toolbar)
        layout.addWidget(canvas)
        # layout.addWidget(button)
        parent.setLayout(layout)
        return figure, canvas, toolbar


    def open_files(self):
        files = QtGui.QFileDialog.getOpenFileNames(self,
                                                   "Select one or more files to open",
                                                   self._last_open_dir or os.getenv("HOME"),
                                                   'Mercator GeoTIFF (*.tiff *.tif)')
        for pathname in files:
            self.document.open_file(pathname)
            self._last_open_dir = os.path.split(pathname)[0]

    def dropEvent(self, event):
        LOG.debug('drop event on mainwindow')
        mime = event.mimeData()
        if mime.hasUrls:
            event.setDropAction(QtCore.Qt.CopyAction)
            event.accept()
            for url in mime.urls():
                path = str(url.toLocalFile())
                LOG.info('about to open {}'.format(path))
                self.document.open_file(path)
        else:
            event.ignore()

    def change_tool(self, name="pz_camera"):
        buttons = [self.ui.panZoomToolButton, self.ui.pointSelectButton, self.ui.regionSelectButton]
        names = [self.scene_manager.pz_camera.name, self.scene_manager.point_probe_camera.name, self.scene_manager.polygon_probe_camera.name]
        names = dict((name,value) for (value,name) in enumerate(names))
        dex = names[name]
        for q,b in enumerate(buttons):
            b.setDown(dex==q)
        self.scene_manager.change_camera(dex)

    def update_progress_bar(self, status_info, *args, **kwargs):
        active = status_info[0]
        LOG.debug('{0!r:s}'.format(status_info))
        val = active[TASK_PROGRESS]
        txt = active[TASK_DOING]
        self.ui.progressBar.setValue(int(val*PROGRESS_BAR_MAX))
        self.ui.progressText.setText(txt)
        #LOG.warning('progress bar updated to {}'.format(val))

    def update_frame_slider(self, frame_info):
        frame_index, frame_count, animating = frame_info[:3]
        self.ui.animationSlider.setRange(0, frame_count-1)
        self.ui.animationSlider.setValue(frame_index or 0)
        # LOG.debug('did update animation slider {} {}'.format(frame_index, frame_count))
        self.ui.animPlayPause.setDown(animating)
        self.ui.animationSlider.repaint()

    def change_layer_colormap(self, nfo):
        uuid = nfo[INFO.UUID]
        mapname = nfo[INFO.COLORMAP]
        LOG.info('changing {} to colormap {}'.format(uuid, mapname))
        self.scene_manager.set_colormap(mapname, uuid=uuid)

    def remove_layer(self, *args, **kwargs):
        uuids = self.behaviorLayersList.current_selected_uuids()
        for uuid in uuids:
            LOG.debug('removing layer {}'.format(uuid))
            self.document.remove_layer_prez(uuid)

    def animation_slider_jump_frame(self, event, *args, **kwargs):
        frame = self.ui.animationSlider.value()
        self.scene_manager.set_frame_number(frame)
        # TODO: update layer list to reflect what layers are visible/hidden?

    def _next_last_time_visibility(self, direction=0, *args, **kwargs):
        LOG.info('time incr {}'.format(direction))
        # TODO: if this frame is part of the animation sequence, update the slider as well!
        uuids = self.behaviorLayersList.current_selected_uuids()
        if not uuids:
            self.ui.cursorProbeText.setText('No layer selected?')
            pass # FIXME: notify user
        for uuid in uuids:
            new_focus = self.document.next_last_step(uuid, direction, bandwise=False)
        return new_focus

    def next_last_time(self, direction=0, *args, **kwargs):
        new_focus = self._next_last_time_visibility(direction=direction)
        self.behaviorLayersList.select([new_focus])
        return new_focus
        # self.document.animate_siblings_of_layer(new_focus)

    def next_last_band(self, direction=0, *args, **kwargs):
        LOG.info('band incr {}'.format(direction))
        uuids = self.behaviorLayersList.current_selected_uuids()
        if not uuids:
            pass # FIXME: notify user
        for uuid in uuids:
            new_focus = self.document.next_last_step(uuid, direction, bandwise=True)
        self.behaviorLayersList.select([new_focus])

    def change_animation_to_current_selection_siblings(self, *args, **kwargs):
        uuid = self._next_last_time_visibility(direction=0)
        # calculate the new animation sequence by consulting the guidebook
        uuids = self.document.animate_siblings_of_layer(uuid)
        self.ui.cursorProbeText.setText("Frame order updated")
        self.behaviorLayersList.select(uuids)
        LOG.info('using siblings of {} for animation loop'.format(uuids[0]))
        # TODO: decide if animation controls should be updated as well

    def toggle_visibility_on_selected_layers(self, *args, **kwargs):
        uuids = self.behaviorLayersList.current_selected_uuids()
        self.document.toggle_layer_visibility(uuids)

    # def accept_new_layer(self, new_order, info, overview_content):
    #     LOG.debug('accepting new layer order {0!r:s}'.format(new_order))
    #     if info[INFO.KIND] == KIND.IMAGE:
    #         LOG.info("rebuilding animation based on newly loaded image layer")
    #         self.document.animate_using_layer(info[INFO.UUID])
    #         self.animation_slider_jump_frame(None)
    #         self.behaviorLayersList.select([info[INFO.UUID]])


    def __init__(self, workspace_dir=None, glob_pattern=None, border_shapefile=None):
        super(Main, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        # refer to objectName'd entities as self.ui.objectName
        self.setAcceptDrops(True)

        self.queue = TaskQueue()
        self.ui.progressBar.setRange(0, PROGRESS_BAR_MAX)
        self.queue.didMakeProgress.connect(self.update_progress_bar)

        # create document
        self.workspace = Workspace(workspace_dir)
        self.document = doc = Document(self.workspace)
        self.scene_manager = SceneGraphManager(doc, self.workspace, self.queue,
                                               glob_pattern=glob_pattern,
                                               border_shapefile=border_shapefile,
                                               parent=self)
        self.ui.mainWidgets.addTab(self.scene_manager.main_canvas.native, 'Mercator')

        self.scene_manager.didChangeFrame.connect(self.update_frame_slider)
        self.ui.animPlayPause.clicked.connect(self.scene_manager.layer_set.toggle_animation)
        self.ui.animForward.clicked.connect(self.scene_manager.layer_set.next_frame)
        last_frame = partial(self.scene_manager.layer_set.next_frame, frame_number=-1)
        self.ui.animBack.clicked.connect(last_frame)
        self.ui.animationSlider.valueChanged.connect(self.animation_slider_jump_frame)
        # TODO: connect animation slider to frame number
        # TODO: connect step forward and step back buttons to frame number (.next_frame)
        # TODO: when animation stops, show the frames with their proper visibility again?
        # TODO: otherwise, make sure the visibility on the layer list reflects what animation is doing

        # disable close button on panes
        for pane in [self.ui.probeAPane, self.ui.probeBPane, self.ui.layersPane]:
            pane.setFeatures(QtGui.QDockWidget.DockWidgetFloatable |
                             QtGui.QDockWidget.DockWidgetMovable)

        for uuid, ds_info, full_data in test_layers(self.workspace, self.document, glob_pattern=glob_pattern):
            # this now fires off a document modification cascade resulting in a new layer going up
            pass

        # Interaction Setup
        self.setup_key_releases()
        self.scheduler = QtCore.QTimer(parent=self)
        self.scheduler.setInterval(200.0)
        self.scheduler.timeout.connect(partial(self.scene_manager.on_view_change, self.scheduler))
        def start_wrapper(timer, event):
            """Simple wrapper around a timers start method so we can accept but ignore the event provided
            """
            timer.start()
        self.scene_manager.main_canvas.transforms.changed.connect(partial(start_wrapper, self.scheduler))

        # convey action between document and layer list view
        self.behaviorLayersList = LayerStackListViewModel([self.ui.layerSet1Table, self.ui.layerSet2Table, self.ui.layerSet3Table, self.ui.layerSet4Table], doc)

        # coordinate what gets done when a layer is added by document
        # self.document.didAddLayer.connect(self.accept_new_layer)

        def update_probe_point(uuid, xy_pos):
            data_point = self.workspace.get_content_point(uuid, xy_pos)
            self.ui.cursorProbeText.setText("Point Probe: {:.03f}".format(float(data_point)))
        self.scene_manager.newProbePoint.connect(update_probe_point)

        def update_probe_polygon(uuid, points, layerlist=self.behaviorLayersList):
            selected_uuids = list(layerlist.current_selected_uuids())
            LOG.debug("selected UUID set is {0!r:s}".format(selected_uuids))
            if len(selected_uuids)==0:
                selected_uuids = [uuid]
            if (len(selected_uuids)==1):
                data_polygon = self.workspace.get_content_polygon(selected_uuids[0], points)
                self.figureA.clf()
                plt.hist(data_polygon.flatten(), bins=100)
                self.canvasA.draw()
            elif len(selected_uuids)==2:
                data1 = self.workspace.get_content_polygon(selected_uuids[0], points)
                name1 = self.workspace.get_info(selected_uuids[0])[INFO.NAME]
                data2 = self.workspace.get_content_polygon(selected_uuids[1], points)
                name2 = self.workspace.get_info(selected_uuids[1])[INFO.NAME]
                self.figureA.clf()
                plt.scatter(data1.flatten(), data2.flatten(), s=1, alpha=0.5)
                plt.xlabel(name1)
                plt.ylabel(name2)
                self.canvasA.draw()
                data_polygon = data1
            else:
                data_polygon = self.workspace.get_content_polygon(selected_uuids[0], points)
            avg = data_polygon.mean()
            self.ui.cursorProbeText.setText("Polygon Probe: {:.03f}".format(float(avg)))
            self.scene_manager.on_new_polygon(points)
        self.scene_manager.newProbePolygon.connect(update_probe_polygon)

        self.ui.mainWidgets.removeTab(0)
        self.ui.mainWidgets.removeTab(0)

        # Set up builtin colormaps
        # FIXME: Move stuff like this to document probably
        self.scene_manager.add_colormap("test", Colormap([
            (0.00, 0.00, 0.00, 1.00),
            (0.00, 0.00, 1.00, 1.00),
        ]))

        # self.queue.add('test', test_task(), 'test000')
        # self.ui.layers
        print(self.scene_manager.main_view.describe_tree(with_transform=True))
        self.document.didChangeColormap.connect(self.change_layer_colormap)

        self.ui.panZoomToolButton.clicked.connect(partial(self.change_tool, name=self.scene_manager.pz_camera.name))
        self.ui.pointSelectButton.clicked.connect(partial(self.change_tool, name=self.scene_manager.point_probe_camera.name))
        self.ui.regionSelectButton.clicked.connect(partial(self.change_tool, name=self.scene_manager.polygon_probe_camera.name))
        self.change_tool()

        self.setup_menu()
        self.setup_probe_panes()

    def setup_probe_panes(self):
        self.figureA, self.canvasA, self.toolbarA = self.make_mpl_pane(self.ui.probeAWidget)

    def toggle_animation(self, action:QtGui.QAction=None, *args):
        new_state = self.scene_manager.layer_set.toggle_animation()
        action.setChecked(new_state)

    def setup_menu(self):
        open_action = QtGui.QAction("&Open", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_files)

        exit_action = QtGui.QAction("&Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(QtGui.qApp.quit)

        menubar = self.ui.menubar
        file_menu = menubar.addMenu('&File')
        file_menu.addAction(open_action)
        file_menu.addAction(exit_action)

        next_time = QtGui.QAction("Next Time", self)
        next_time.setShortcut(QtCore.Qt.Key_Right)
        next_time.triggered.connect(partial(self.next_last_time, direction=1))

        prev_time = QtGui.QAction("Previous Time", self)
        prev_time.setShortcut(QtCore.Qt.Key_Left)
        prev_time.triggered.connect(partial(self.next_last_time, direction=-1))

        next_band = QtGui.QAction("Next Band", self)
        next_band.setShortcut(QtCore.Qt.Key_Up)
        next_band.triggered.connect(partial(self.next_last_band, direction=1))

        prev_band = QtGui.QAction("Previous Band", self)
        prev_band.setShortcut(QtCore.Qt.Key_Down)
        prev_band.triggered.connect(partial(self.next_last_band, direction=-1))

        toggle_vis = QtGui.QAction("Toggle &Visibility", self)
        toggle_vis.setShortcut('V')
        toggle_vis.triggered.connect(self.toggle_visibility_on_selected_layers)

        animate = QtGui.QAction("Animate", self)
        animate.setShortcut('A')
        animate.triggered.connect(partial(self.toggle_animation, action=animate))

        change_order = QtGui.QAction("Set Animation &Order", self)
        change_order.setShortcut('O')
        change_order.triggered.connect(self.change_animation_to_current_selection_siblings)

        view_menu = menubar.addMenu('&View')
        view_menu.addAction(animate)
        view_menu.addAction(prev_time)
        view_menu.addAction(next_time)
        view_menu.addAction(prev_band)
        view_menu.addAction(next_band)
        view_menu.addAction(change_order)
        view_menu.addAction(toggle_vis)

        remove = QtGui.QAction("Remove Layer", self)
        remove.setShortcut(QtCore.Qt.Key_Delete)
        remove.triggered.connect(self.remove_layer)

        edit_menu = menubar.addMenu('&Edit')
        edit_menu.addAction(remove)

        menubar.setEnabled(True)


    def setup_key_releases(self):
        def cb_factory(required_key, cb):
            def tmp_cb(key, cb=cb):
                if key.text == required_key:
                    return cb()
            return tmp_cb

        # self.scene_manager.main_canvas.events.key_release.connect(cb_factory("a", self.scene_manager.layer_set.toggle_animation))
        # self.scene_manager.main_canvas.events.key_release.connect(cb_factory("n", self.scene_manager.layer_set.next_frame))
        self.scene_manager.main_canvas.events.key_release.connect(cb_factory("c", self.scene_manager.next_camera))

        class ColormapSlot(object):
            def __init__(self, sgm, key='e'):
                self.index = 0
                self.key = key
                self.sgm = sgm
                self.colormaps = ["grays", "autumn", "fire", "hot", "winter", "test"]

            def __call__(self, key):
                if key.text == self.key:
                    self.sgm.set_colormap(self.colormaps[self.index])
                    self.index = (self.index + 1) % len(self.colormaps)

        self.scene_manager.main_canvas.events.key_release.connect(ColormapSlot(self.scene_manager))

    def updateLayerList(self):
        # self.ui.layers.add
        pass


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run CSPOV")
    parser.add_argument("-w", "--workspace", default='.',
                        help="Specify workspace base directory")
    parser.add_argument("--border-shapefile", default=None,
                        help="Specify alternative coastline/border shapefile")
    parser.add_argument("--glob-pattern", default=os.environ.get("TIFF_GLOB", None),
                        help="Specify glob pattern for input images")
    parser.add_argument('-v', '--verbose', dest='verbosity', action="count", default=int(os.environ.get("VERBOSITY", 2)),
                        help='each occurrence increases verbosity 1 level through ERROR-WARNING-INFO-DEBUG (default INFO)')
    args = parser.parse_args()

    levels = [logging.ERROR, logging.WARN, logging.INFO, logging.DEBUG]
    level=levels[min(3, args.verbosity)]
    logging.basicConfig(level=level)
    # logging.getLogger('vispy').setLevel(level)

    app.create()
    # app = QApplication(sys.argv)
    window = Main(
        workspace_dir=args.workspace,
        glob_pattern=args.glob_pattern,
        border_shapefile=args.border_shapefile
    )
    window.show()
    print("running")
    # bring window to front
    window.raise_()
    app.run()

if __name__ == '__main__':
    import sys
    sys.exit(main())
