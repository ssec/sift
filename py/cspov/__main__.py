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
from cspov.control.LayerManager import LayerSetsManager
from cspov.model import Document
from cspov.view.SceneGraphManager import SceneGraphManager
from cspov.view.ProbeGraphs import ProbeGraphManager
from cspov.queue import TaskQueue, test_task, TASK_PROGRESS, TASK_DOING
from cspov.workspace import Workspace
from cspov.view.Colormap import ALL_COLORMAPS

from functools import partial

# this is generated with pyuic4 pov_main.ui >pov_main_ui.py
from cspov.ui.pov_main_ui import Ui_MainWindow
from cspov.common import INFO, KIND, DEFAULT_PROJ_OBJ

import os
import logging


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

class AnimationSpeedPopupWindow(QtGui.QWidget):
    _slider = None
    _active = False

    def __init__(self, slot, *args, **kwargs):
        super(AnimationSpeedPopupWindow, self).__init__(*args, **kwargs)
        from PyQt4.QtCore import Qt
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Popup)
        self.setFocusPolicy(Qt.ClickFocus)
        self.setToolTip('Set animation speed')
        self._slider = QtGui.QSlider(parent=self)
        # n, x = self._convert(10, reverse=True), self._convert(5000, reverse=True)
        n, x = 2, 150  # frames per 10 seconds
        self._slider.setRange(n, x) #
        # self._slider.setSingleStep(1)
        # self._slider.setInvertedAppearance(True)
        self._slot = slot
        self._slider.valueChanged.connect(self._changed)
        self._layout = QtGui.QHBoxLayout()
        self._layout.addWidget(self._slider)
        self.setLayout(self._layout)

    def _convert(self, val, reverse=False):
        """
        map 1..100 nonlinearly to 10ms .. 5000ms
        :param val: ticks to convert to milliseconds
        :param reverse: when true, reverse conversion
        :return:
        """
        if reverse: # convert milliseconds to fp10s
            fp10s = 10000.0 / float(val)
            return fp10s
        else:
            ms = 10000.0 / float(val)
            return ms

    def _changed(self, value):
        if not self._active:
            return
        fps = float(value) / 10.0
        self.setToolTip('{0:.1f} fps'.format(fps))
        val = self._convert(value)
        self._slot(val)

    def show_at(self, pos, val):
        from PyQt4.QtCore import QRect, QPoint, QSize
        sz = QSize(40, 180)
        pt = QPoint(pos.x() - 20, pos.y() - 160)
        rect = QRect(pt, sz)
        self.setGeometry(rect)
        self.show()
        self._slider.setValue(int(self._convert(val, reverse=True)))
        self._active = True

    def focusOutEvent(self, *args, **kwargs):
        self.hide()
        self._active = False

class Main(QtGui.QMainWindow):
    _last_open_dir = None  # directory to open files in
    _animation_speed_popup = None  # window we'll show temporarily with animation speed popup

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
        # val = active[TASK_PROGRESS]
        txt = active[TASK_DOING]
        val = self.queue.progress_ratio()
        self.ui.progressBar.setValue(int(val*PROGRESS_BAR_MAX))
        self.ui.progressText.setText(txt)
        #LOG.warning('progress bar updated to {}'.format(val))

    def update_frame_slider(self, frame_info):
        """
        animation is in progress or completed
        update the animation slider and label to show what's going on
        :param frame_info: tuple, ultimately from scenegraphmanager.layer_set callback into sgm
        :return:
        """
        frame_index, frame_count, animating, uuid = frame_info[:4]
        self.ui.animationSlider.setRange(0, frame_count-1)
        self.ui.animationSlider.setValue(frame_index or 0)
        # LOG.debug('did update animation slider {} {}'.format(frame_index, frame_count))
        self.ui.animPlayPause.setDown(animating)
        self.ui.animationSlider.repaint()
        self.ui.animationLabel.setText(self.document.time_label_for_uuid(uuid))

    def update_frame_time_to_top_visible(self):
        # FUTURE: don't address layer set directly
        self.ui.animationLabel.setText(self.document.time_label_for_uuid(self.scene_manager.layer_set.top_layer_uuid()))

    def remove_layer(self, *args, **kwargs):
        uuids = self.behaviorLayersList.current_selected_uuids()
        if uuids:
            self.document.remove_layers_from_all_sets(uuids)

    def animation_slider_jump_frame(self, event, *args, **kwargs):
        "user has moved frame slider, update the display"
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
        new_focus = None
        for uuid in uuids:
            new_focus = self.document.next_last_step(uuid, direction, bandwise=False)
        return new_focus

    def update_slider_if_frame_is_in_animation(self, uuid):
        # FUTURE: this could be a cheaper operation but it's probably fine since it's input-driven
        cao = self.document.current_animation_order
        try:
            dex = cao.index(uuid)
        except ValueError as not_present:
            return
        frame_change_tuple = (dex, len(cao), False, uuid)
        self.update_frame_slider(frame_change_tuple)

    def next_last_time(self, direction=0, *args, **kwargs):
        self.scene_manager.layer_set.animating = False
        new_focus = self._next_last_time_visibility(direction=direction)
        self.behaviorLayersList.select([new_focus])
        # if this part of the animation cycle, update the animation slider and displayed time as well
        self.update_slider_if_frame_is_in_animation(new_focus)
        # FIXME: force animation off
        return new_focus
        # self.document.animate_siblings_of_layer(new_focus)
        self.update_frame_time_to_top_visible()

    def next_last_band(self, direction=0, *args, **kwargs):
        LOG.info('band incr {}'.format(direction))
        uuids = self.behaviorLayersList.current_selected_uuids()
        new_focus = None
        if not uuids:
            pass # FIXME: notify user
        for uuid in uuids:
            new_focus = self.document.next_last_step(uuid, direction, bandwise=True)
        if new_focus is not None:
            self.behaviorLayersList.select([new_focus])
            self.update_frame_time_to_top_visible()

    def change_animation_to_current_selection_siblings(self, *args, **kwargs):
        uuid = self._next_last_time_visibility(direction=0)
        # calculate the new animation sequence by consulting the guidebook
        uuids = self.document.animate_siblings_of_layer(uuid)
        if uuids:
            self.ui.cursorProbeText.setText("Frame order updated")
            self.behaviorLayersList.select(uuids)
        else:
            self.ui.cursorProbeText.setText("Layer with time steps needed")
        LOG.info('using siblings of {} for animation loop'.format(uuids[0]))

    def set_animation_speed(self, milliseconds):
        LOG.info('animation speed set to {}ms'.format(milliseconds))
        # FUTURE: propagate this into the document?
        self.scene_manager.layer_set.animation_speed = milliseconds

    def show_animation_speed_slider(self, pos:QtCore.QPoint, *args):
        LOG.info('menu requested for animation control')
        gpos = self.ui.animPlayPause.mapToGlobal(pos)

        if self._animation_speed_popup is None:
            self._animation_speed_popup = popup = AnimationSpeedPopupWindow(slot=self.set_animation_speed, parent=None)
        else:
            popup = self._animation_speed_popup
        if not popup.isVisible():
            popup.show_at(gpos, self.scene_manager.layer_set.animation_speed)

    def toggle_visibility_on_selected_layers(self, *args, **kwargs):
        uuids = self.behaviorLayersList.current_selected_uuids()
        self.document.toggle_layer_visibility(uuids)
        self.update_frame_time_to_top_visible()

    def toggle_animation(self, event, *args, **kwargs):
        self.scene_manager.layer_set.toggle_animation(*args, **kwargs)

    # def accept_new_layer(self, new_order, info, overview_content):
    #     LOG.debug('accepting new layer order {0!r:s}'.format(new_order))
    #     if info[INFO.KIND] == KIND.IMAGE:
    #         LOG.info("rebuilding animation based on newly loaded image layer")
    #         self.document.animate_using_layer(info[INFO.UUID])
    #         self.animation_slider_jump_frame(None)
    #         self.behaviorLayersList.select([info[INFO.UUID]])

    def __init__(self, workspace_dir=None, workspace_size=None, glob_pattern=None, border_shapefile=None, center=None):
        super(Main, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        # refer to objectName'd entities as self.ui.objectName
        self.setAcceptDrops(True)

        self.queue = TaskQueue()
        self.ui.progressBar.setRange(0, PROGRESS_BAR_MAX)
        self.queue.didMakeProgress.connect(self.update_progress_bar)

        # create document
        self.workspace = Workspace(workspace_dir, max_size_gb=workspace_size, queue=self.queue)
        self.document = doc = Document(self.workspace)
        self.scene_manager = SceneGraphManager(doc, self.workspace, self.queue,
                                               glob_pattern=glob_pattern,
                                               border_shapefile=border_shapefile,
                                               center=center,
                                               parent=self)
        self.ui.mainWidgets.addTab(self.scene_manager.main_canvas.native, 'Mercator')

        self.scene_manager.didChangeFrame.connect(self.update_frame_slider)
        self.ui.animPlayPause.clicked.connect(self.toggle_animation)
        self.ui.animPlayPause.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.ui.animPlayPause.customContextMenuRequested.connect(self.show_animation_speed_slider)
        # def next_frame(*args, **kwargs):
        #     self.scene_manager.animating = False
        #     self.scene_manager.layer_set.next_frame()
        # self.ui.animForward.clicked.connect(next_frame)
        # def prev_frame(*args, **kwargs):
        #     self.scene_manager.animating = False
        #     self.scene_manager.layer_set.next_frame(frame_number=-1)
        # self.ui.animBack.clicked.connect(prev_frame)

        # allow animation slider to set animation frame being displayed:
        self.ui.animationSlider.valueChanged.connect(self.animation_slider_jump_frame)

        # allow animation, once stopped, to propagate visibility to the document and layerlist:
        self.scene_manager.didChangeLayerVisibility.connect(self.document.animation_changed_visibility)

        # disable close button on panes
        for pane in [self.ui.areaProbePane, self.ui.layersPane]:
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
        self.layerSetsManager = LayerSetsManager(self.ui.layerSetTabs, doc)
        self.behaviorLayersList = self.layerSetsManager.getLayerStackListViewModel()

        # coordinate what gets done when a layer is added by document
        self.document.didAddLayer.connect(self.update_frame_time_to_top_visible)

        def update_probe_point(uuid, xy_pos):
            lon, lat = DEFAULT_PROJ_OBJ(xy_pos[0], xy_pos[1], inverse=True)
            lon_str = "{:.02f} {}".format(abs(lon), "W" if lon < 0 else "E")
            lat_str = "{:.02f} {}".format(abs(lat), "S" if lat < 0 else "N")
            self.ui.cursorProbeLocation.setText("Probe Location: {}, {}".format(lon_str, lat_str))

            if uuid is not None:
                data_point = self.workspace.get_content_point(uuid, xy_pos)
                data_str = "{:.03f}".format(float(data_point))
            else:
                data_str = "N/A"
            self.ui.cursorProbeText.setText("Probe Value: {} ".format(data_str))
        self.scene_manager.newProbePoint.connect(update_probe_point)

        def update_probe_polygon(uuid, points, layerlist=self.behaviorLayersList):

            selected_uuids = list(layerlist.current_selected_uuids())
            LOG.debug("selected UUID set is {0!r:s}".format(selected_uuids))

            # if the layer list doesn't have any selected UUIDs, use the one passed in
            if len(selected_uuids) <= 0:
                selected_uuids = [uuid]

            # if we have more than two uuids, just plot the very first one
            elif len(selected_uuids) > 2 :
                selected_uuids = selected_uuids[0:1]

            # now we must have 1 or 2 UUIDs in our list

            # TODO, when the plots manage their own layer selection, change this call
            # FUTURE, once the polygon is a layer, this will need to change
            # update our current plot with the new polygon
            polygon_name = self.graphManager.currentPolygonChanged (polygonPoints=points)

            # do whatever other updates the scene manager needs
            self.scene_manager.on_new_polygon(polygon_name, points)

        self.scene_manager.newProbePolygon.connect(update_probe_polygon)

        self.ui.mainWidgets.removeTab(0)
        self.ui.mainWidgets.removeTab(0)

        # self.queue.add('test', test_task(), 'test000')
        # self.ui.layers
        print(self.scene_manager.main_view.describe_tree(with_transform=True))
        self.document.didChangeColormap.connect(self.scene_manager.change_layers_colormap)
        self.document.didChangeColorLimits.connect(self.scene_manager.change_layers_color_limits)

        self.ui.panZoomToolButton.clicked.connect(partial(self.change_tool, name=self.scene_manager.pz_camera.name))
        self.ui.pointSelectButton.clicked.connect(partial(self.change_tool, name=self.scene_manager.point_probe_camera.name))
        self.ui.regionSelectButton.clicked.connect(partial(self.change_tool, name=self.scene_manager.polygon_probe_camera.name))
        self.change_tool()

        self.setup_menu()
        self.graphManager = ProbeGraphManager(self.ui.probeTabWidget, self.workspace, self.document)
        self.graphManager.didChangeTab.connect(self.scene_manager.show_only_polygons)

    def closeEvent(self, event, *args, **kwargs):
        LOG.debug('main window closing')
        self.workspace.close()

    def toggle_animation(self, action:QtGui.QAction=None, *args):
        new_state = self.scene_manager.layer_set.toggle_animation()
        self.ui.animPlayPause.setChecked(new_state)

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
        next_slot = partial(self.next_last_time, direction=1)
        next_time.triggered.connect(next_slot)
        self.ui.animForward.clicked.connect(next_slot)

        focus_current = QtGui.QAction("Focus Current Timestep", self)
        focus_current.setShortcut('.')
        focus_current.triggered.connect(partial(self.next_last_band, direction=0))

        prev_time = QtGui.QAction("Previous Time", self)
        prev_time.setShortcut(QtCore.Qt.Key_Left)
        prev_slot = partial(self.next_last_time, direction=-1)
        prev_time.triggered.connect(prev_slot)
        self.ui.animBack.clicked.connect(prev_slot)

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

        flip_colormap = QtGui.QAction("Flip Color Limits (Top Layer)", self)
        flip_colormap.setShortcut("/")
        flip_colormap.triggered.connect(lambda: self.document.flip_climits_for_layers([self.document.current_visible_layer]))

        remove = QtGui.QAction("Remove Layer", self)
        remove.setShortcut(QtCore.Qt.Key_Delete)
        remove.triggered.connect(self.remove_layer)

        cycle_borders = QtGui.QAction("Cycle &Borders", self)
        cycle_borders.setShortcut('B')
        cycle_borders.triggered.connect(self.scene_manager.cycle_borders_color)

        cycle_grid = QtGui.QAction("Cycle &Lat/Lon Grid", self)
        cycle_grid.setShortcut('L')
        cycle_grid.triggered.connect(self.scene_manager.cycle_grid_color)

        edit_menu = menubar.addMenu('&Edit')
        edit_menu.addAction(remove)

        view_menu = menubar.addMenu('&View')
        view_menu.addAction(animate)
        view_menu.addAction(prev_time)
        view_menu.addAction(focus_current)
        view_menu.addAction(next_time)
        view_menu.addAction(prev_band)
        view_menu.addAction(next_band)
        view_menu.addAction(change_order)
        view_menu.addAction(toggle_vis)
        view_menu.addAction(flip_colormap)
        view_menu.addAction(cycle_borders)
        view_menu.addAction(cycle_grid)

        menubar.setEnabled(True)

    def setup_key_releases(self):
        def cb_factory(required_key, cb):
            def tmp_cb(key, cb=cb):
                if key.text == required_key:
                    return cb()
            return tmp_cb

        self.scene_manager.main_canvas.events.key_release.connect(cb_factory("c", self.scene_manager.next_camera))

        class ColormapSlot(object):
            def __init__(self, sgm, key='e'):
                self.index = 0
                self.key = key
                self.sgm = sgm
                self.colormaps = ALL_COLORMAPS.keys()

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
    parser.add_argument("-s", "--space", default=256, type=int,
                        help="Specify max amount of data to hold in workspace in Gigabytes")
    parser.add_argument("--border-shapefile", default=None,
                        help="Specify alternative coastline/border shapefile")
    parser.add_argument("--glob-pattern", default=os.environ.get("TIFF_GLOB", None),
                        help="Specify glob pattern for input images")
    parser.add_argument("-c", "--center", nargs=2, type=float,
                        help="Specify center longitude and latitude for camera")
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
        workspace_size=args.space,
        glob_pattern=args.glob_pattern,
        border_shapefile=args.border_shapefile,
        center=args.center,
    )
    window.show()
    print("running")
    # bring window to front
    window.raise_()
    app.run()

if __name__ == '__main__':
    import sys
    sys.exit(main())
