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

from uuid import UUID
from vispy import app
app_object = app.use_app('pyqt4')
QtCore = app_object.backend_module.QtCore
QtGui = app_object.backend_module.QtGui

import sift.ui.open_cache_dialog_ui as open_cache_dialog_ui
from sift.ui import export_image_dialog_ui
from sift.control.LayerManager import LayerSetsManager
from sift.model.document import Document, DocRGBLayer, DocCompositeLayer, DocBasicLayer, DocLayer, DocLayerStack
from sift.view.SceneGraphManager import SceneGraphManager
from sift.view.ProbeGraphs import ProbeGraphManager, DEFAULT_POINT_PROBE
from sift.queue import TaskQueue, test_task, TASK_PROGRESS, TASK_DOING
from sift.workspace import Workspace
from sift.view.Colormap import ALL_COLORMAPS
from sift import __version__

from functools import partial

# this is generated with pyuic4 pov_main.ui >pov_main_ui.py
from sift.ui.pov_main_ui import Ui_MainWindow
from sift.common import INFO, KIND, DEFAULT_PROJ_OBJ, TOOL, COMPOSITE_TYPE

import os
import sys
import logging


LOG = logging.getLogger(__name__)
PROGRESS_BAR_MAX = 1000
STATUS_BAR_DURATION = 2000  # ms


def test_layers_from_directory(ws, doc, layer_tiff_glob, range_txt=None):
    """
    TIFF_GLOB='/Users/keoni/Data/CSPOV/2015_07_14_195/00?0/HS*_B03_*merc.tif' VERBOSITY=3 python -m sift
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
    return []

class OpenCacheDialog(QtGui.QWidget):
    _paths = None
    _opener = None
    _remover = None

    def __init__(self):
        super(OpenCacheDialog, self).__init__()
        self.ui = open_cache_dialog_ui.Ui_openFromCacheDialog()
        self.ui.setupUi(self)
        self.ui.cacheListWidget.setSelectionMode(QtGui.QListWidget.ExtendedSelection)
        self.ui.removeFromCacheButton.clicked.connect(self._do_remove)

    def activate(self, paths, opener, remover):
        self._opener = opener
        self._remover = remover
        self.ui.cacheListWidget.clear()
        if paths:
            pfx = _common_path_prefix(paths)
            self.ui.commonPathLabel.setText("Common prefix: " + pfx)
            lpfx = len(pfx)
            self._paths = {}
            for path in paths:
                key = path[lpfx:].strip(os.sep)
                li = QtGui.QListWidgetItem(key)
                self.ui.cacheListWidget.addItem(li)
                self._paths[key] = path
        self.show()

    def _do_remove(self, *args, **kwargs):
        items = list(self.ui.cacheListWidget.selectedItems())
        to_remove = [self._paths[x.text()] for x in items]
        for item in items:
            self.ui.cacheListWidget.removeItemWidget(item)
        self._remover(to_remove)
        self.hide()
        self._doc = self._ws = self._opener = self._paths = None

    def accept(self, *args, **kwargs):
        self.hide()
        to_open = [self._paths[x.text()] for x in self.ui.cacheListWidget.selectedItems()]
        LOG.info("opening from cache: " + repr(to_open))
        self._opener(to_open)
        self._doc = self._ws = self._opener = self._paths = None

    def reject(self, *args, **kwargs):
        self.hide()
        self._doc = self._ws = self._opener = self._paths = None


class ExportImageDialog(QtGui.QDialog):
    def __init__(self, parent):
        super(ExportImageDialog, self).__init__(parent)
        self._last_dir = os.getcwd()

        self.ui = export_image_dialog_ui.Ui_ExportImageDialog()
        self.ui.setupUi(self)

        self.ui.animationGroupBox.setDisabled(True)
        self.ui.constantDelaySpin.setDisabled(True)
        self.ui.timeLapseRadio.setChecked(True)
        self.ui.timeLapseRadio.clicked.connect(self._delay_clicked)
        self.ui.constantDelayRadio.clicked.connect(self._delay_clicked)
        self._delay_clicked()

        self.ui.frameRangeFrom.setValidator(QtGui.QIntValidator(1, 1))
        self.ui.frameRangeTo.setValidator(QtGui.QIntValidator(1, 1))
        self.ui.saveAsLineEdit.textChanged.connect(self._validate_filename)
        self.ui.saveAsButton.clicked.connect(self._show_file_dialog)
        self._validate_filename()

        self.ui.frameAllRadio.clicked.connect(self.change_frame_range)
        self.ui.frameCurrentRadio.clicked.connect(self.change_frame_range)
        self.ui.frameRangeRadio.clicked.connect(self.change_frame_range)
        self.change_frame_range()  # set default

    def set_total_frames(self, n):
        self.ui.frameRangeFrom.validator().setBottom(1)
        self.ui.frameRangeTo.validator().setBottom(2)
        self.ui.frameRangeFrom.validator().setTop(n - 1)
        self.ui.frameRangeTo.validator().setTop(n)

    def _delay_clicked(self):
        if self.ui.constantDelayRadio.isChecked():
            self.ui.constantDelaySpin.setDisabled(False)
        else:
            self.ui.constantDelaySpin.setDisabled(True)

    def _show_file_dialog(self):
        fn = QtGui.QFileDialog.getSaveFileName(self,
                                               self.tr('Screenshot Filename'),
                                               self._last_dir,
                                               self.tr('Image Files (*.png *.jpg *.gif)'))
        if fn:
            self.ui.saveAsLineEdit.setText(fn)
        # bring this dialog back in focus
        self.raise_()
        self.activateWindow()

    def _validate_filename(self):
        t = self.ui.saveAsLineEdit.text()
        bt = self.ui.buttonBox.button(QtGui.QDialogButtonBox.Save)
        if not t or os.path.splitext(t)[-1] not in ['.png', '.jpg', '.gif']:
            bt.setDisabled(True)
        else:
            self._last_dir = os.path.dirname(t)
            bt.setDisabled(False)

    def change_frame_range(self):
        if self.ui.frameRangeRadio.isChecked():
            self.ui.frameRangeFrom.setDisabled(False)
            self.ui.frameRangeTo.setDisabled(False)
        else:
            self.ui.frameRangeFrom.setDisabled(True)
            self.ui.frameRangeTo.setDisabled(True)

        if self.ui.frameCurrentRadio.isChecked():
            self.ui.animationGroupBox.setDisabled(True)
            self.ui.frameDelayGroup.setDisabled(True)
        else:
            self.ui.animationGroupBox.setDisabled(False)
            self.ui.frameDelayGroup.setDisabled(False)

    def get_frame_range(self):
        if self.ui.frameCurrentRadio.isChecked():
            frame = None
        elif self.ui.frameAllRadio.isChecked():
            frame = [None, None]
        elif self.ui.frameRangeRadio.isChecked():
            frame = [
                int(self.ui.frameRangeFrom.text()),
                int(self.ui.frameRangeTo.text())
            ]
        else:
            LOG.error("Unknown frame range selection")
            return
        return frame

    def get_info(self):
        if self.ui.timeLapseRadio.isChecked():
            delay = None
        else:
            delay = self.ui.constantDelaySpin.value()

        # loop is actually an integer of number of times to loop (0 infinite)
        info = {
            'frame_range': self.get_frame_range(),
            'include_footer': self.ui.includeFooterCheckbox.isChecked(),
            # 'transparency': self.ui.transparentCheckbox.isChecked(),
            'loop': self.ui.loopRadio.isChecked(),
            'filename': self.ui.saveAsLineEdit.text(),
            'delay': delay,
        }
        return info


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

def _recursive_split(path):
    dn,fn = os.path.split(path)
    if dn and not fn:
        yield dn
    if len(dn)>0 and dn!=path and len(fn)>0:
        for pc in _recursive_split(dn):
            yield pc
    if len(fn):
        yield fn

def _common_path_prefix_seq(paths):
    pathlists = [list(_recursive_split(path)) for path in paths]
    for component_list in zip(*pathlists):
        pc = None
        for pc1,pc2 in zip(component_list[:-1], component_list[1:]):
            if pc1!=pc2:
                return
            else:
                pc = pc1
        yield pc

def _common_path_prefix(paths):
    "find the most common directory shared by a list of paths"
    paths = list(paths)
    if len(paths)==1:
        return os.path.split(paths[0])[0]
    return os.path.join(*_common_path_prefix_seq(paths))


class Main(QtGui.QMainWindow):
    _last_open_dir = None  # directory to open files in
    _recent_files_menu = None # QMenu
    _animation_speed_popup = None  # window we'll show temporarily with animation speed popup
    _open_cache_dialog = None
    _screenshot_dialog = None

    def interactive_open_files(self, *args, files=None, **kwargs):
        self.scene_manager.layer_set.animating = False
        # http://pyqt.sourceforge.net/Docs/PyQt4/qfiledialog.html#getOpenFileNames
        files = QtGui.QFileDialog.getOpenFileNames(self,
                                                   "Select one or more files to open",
                                                   self._last_open_dir or os.getenv("HOME"),
                                                   ';;'.join(['GOES-R netCDF or Merc GTIFF (*.nc *.nc4 *.tiff *.tif)']))
        self.open_paths(files)

    def open_paths(self, paths):
        paths = list(paths)
        if not paths:
            return
        for uuid, _, _ in self.document.open_files(paths):
            pass
        self.behaviorLayersList.select([uuid])
        # set the animation based on the last added (topmost) layer
        self.document.animate_siblings_of_layer(uuid)
        # force the newest layer to be visible
        self.document.next_last_step(uuid)
        self._last_open_dir = _common_path_prefix(paths)
        self.update_recent_file_menu()

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

    def change_tool(self, checked, name=TOOL.PAN_ZOOM):
        if checked != True:
            return
        self.scene_manager.change_tool(name)

    def update_recent_file_menu(self, *args, **kwargs):
        paths = self.workspace.recently_used_cache_paths()
        paths = self.document.sort_paths(paths)
        LOG.debug('recent files: {0!r:s}'.format(paths))
        self._recent_files_menu.clear()
        for path in paths:
            def openit(*args, path=path, **kwargs):
                LOG.debug('open recent file {}'.format(path))
                self.scene_manager.layer_set.animating = False
                self.document.open_file(path)
            open_action = QtGui.QAction(os.path.split(path)[1], self)
            open_action.triggered.connect(openit)
            self._recent_files_menu.addAction(open_action)

    def update_progress_bar(self, status_info, *args, **kwargs):
        active = status_info[0] if len(status_info)>0 else None
        LOG.debug('{0!r:s}'.format(status_info))
        val = active[TASK_PROGRESS] if active else 0.0
        txt = active[TASK_DOING] if active else ''
        val = self.queue.progress_ratio(val)
        self.ui.progressBar.setValue(int(val*PROGRESS_BAR_MAX))
        self.ui.progressText.setText(txt)
        #LOG.warning('progress bar updated to {}'.format(val))

    def reset_frame_slider(self, *args, **kwargs):
        frame_count = len(self.document.current_animation_order)
        frame_index = None # self.scene_manager.layer_set._frame_number  # FIXME BAAD
        self.ui.animationSlider.setRange(0, frame_count-1)
        self.ui.animationSlider.setValue(frame_index or 0)
        self.ui.animPlayPause.setDown(False)
        self.ui.animationSlider.repaint()
        self.update_frame_time_to_top_visible()

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
        if animating:
            self.ui.animationLabel.setText(self.document.time_label_for_uuid(uuid))
        else:
            self.update_frame_time_to_top_visible()

    def update_frame_time_to_top_visible(self, *args):
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
            self.ui.statusbar.showMessage('ERROR: No layer selected', STATUS_BAR_DURATION)
        new_focus = None
        for uuid in uuids:
            new_focus = self.document.next_last_step(uuid, direction, bandwise=False)
        return new_focus

    def update_slider_if_frame_is_in_animation(self, uuid, **kwargs):
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
        return new_focus

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
            self.update_slider_if_frame_is_in_animation(new_focus)

    def change_animation_to_current_selection_siblings(self, *args, **kwargs):
        uuid = self._next_last_time_visibility(direction=0)
        if uuid is None:
            self.ui.statusbar.showMessage("ERROR: No layer selected", STATUS_BAR_DURATION)
            return
        # calculate the new animation sequence by consulting the guidebook
        uuids = self.document.animate_siblings_of_layer(uuid)
        if uuids:
            self.ui.statusbar.showMessage("INFO: Frame order updated", STATUS_BAR_DURATION)
            self.behaviorLayersList.select(uuids)
        else:
            self.ui.statusbar.showMessage("ERROR: Layer with time steps or band siblings needed", STATUS_BAR_DURATION)
        LOG.info('using siblings of {} for animation loop'.format(uuids[0] if uuids else '-unknown-'))

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

    def animation_reset_by_layer_set_switch(self, *args, **kwargs):
        self.reset_frame_slider()
        self.update_frame_time_to_top_visible()

    # def accept_new_layer(self, new_order, info, overview_content):
    #     LOG.debug('accepting new layer order {0!r:s}'.format(new_order))
    #     if info[INFO.KIND] == KIND.IMAGE:
    #         LOG.info("rebuilding animation based on newly loaded image layer")
    #         self.document.animate_using_layer(info[INFO.UUID])
    #         self.animation_slider_jump_frame(None)
    #         self.behaviorLayersList.select([info[INFO.UUID]])

    def _user_set_rgb_layer(self, uuid:UUID, rgba:str, selected:DocLayer):
        """
        handle signal from layer info panel which says that an RGB layer selected a new layer as channel
        :param layer: layer being edited
        :param rgba: char from 'rgba'
        :param selected: layer to replace, causing scene graph element to be rebuilt
        :return:
        """
        # we could just modify the layer and emit the document signal, but preference is to have document generate its own signals.
        layer = self.document[uuid]
        self.document.change_rgb_component_layer(layer, **{rgba:selected})

    def _user_set_rgb_range(self, uuid:UUID, rgba:str, lo:float, hi:float):
        layer = self.document[uuid]
        self.document.set_rgb_range(layer, rgba, lo, hi)

    def _refresh_probe_results(self, *args):
        arg1 = args[0]
        if isinstance(arg1, dict):
            # Given a dictionary of changes
            uuids = arg1.keys()
        else:
            uuids = [arg1]
        _state, _xy_pos = self.graphManager.current_point_probe_status(DEFAULT_POINT_PROBE)
        self.document.update_equalizer_values(DEFAULT_POINT_PROBE, _state, _xy_pos, uuids=uuids)

    def update_point_probe_text(self, probe_name, state=None, xy_pos=None, uuid=None, animating=None):
        if uuid is None:
            uuid = self.document.current_visible_layer_uuid
        if state is None or xy_pos is None:
            _state, _xy_pos = self.graphManager.current_point_probe_status(probe_name)
            if state is None:
                state = _state
            if xy_pos is None:
                xy_pos = _xy_pos

        if xy_pos is not None and state:
            lon, lat = xy_pos
            lon_str = "{:.02f} {}".format(abs(lon), "W" if lon < 0 else "E")
            lat_str = "{:.02f} {}".format(abs(lat), "S" if lat < 0 else "N")
            self.ui.cursorProbeLocation.setText("Probe Location: {}, {}".format(lon_str, lat_str))
        else:
            self.ui.cursorProbeLocation.setText("Probe Location: N/A")

        if animating:
            data_str = "<animating>"
        elif state and uuid is not None:
            try:
                data_point = self.workspace.get_content_point(uuid, xy_pos)
            except ValueError:
                LOG.debug("Could not get data value", exc_info=True)
                data_point = None

            if data_point is None:
                data_str = "N/A"
            else:
                format_str, unit_str, converted = self.document.convert_units(uuid, data_point)
                data_str = (format_str + "{:s}").format(float(converted), unit_str)
        else:
            data_str = "N/A"
        self.ui.cursorProbeText.setText("Probe Value: {} ".format(data_str))

    def __init__(self, workspace_dir=None, workspace_size=None, glob_pattern=None, border_shapefile=None, center=None):
        super(Main, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle(self.windowTitle().replace("|X.X.X|", __version__))
        self.tabifyDockWidget(self.ui.layersPane, self.ui.areaProbePane)
        self.tabifyDockWidget(self.ui.layerDetailsPane, self.ui.rgbConfigPane)

        # self.tabifyDockWidget(self.ui.rgbConfigPane, self.ui.layerDetailsPane)
        # Make the layer list and layer details shown
        self.ui.layersPane.raise_()
        self.ui.layerDetailsPane.raise_()
        # refer to objectName'd entities as self.ui.objectName
        self.setAcceptDrops(True)

        self.queue = TaskQueue()
        self.ui.progressBar.setRange(0, PROGRESS_BAR_MAX)
        self.queue.didMakeProgress.connect(self.update_progress_bar)

        # create document
        self.workspace = Workspace(workspace_dir, max_size_gb=workspace_size, queue=self.queue)
        self.document = doc = Document(self.workspace)
        self.scene_manager = SceneGraphManager(doc, self.workspace, self.queue,
                                               border_shapefile=border_shapefile,
                                               center=center,
                                               parent=self)
        self.ui.mainMapWidget.layout().addWidget(self.scene_manager.main_canvas.native)
        self.ui.projectionComboBox.addItems(tuple(self.document.available_projections.keys()))
        self.ui.projectionComboBox.currentIndexChanged.connect(self.document.change_projection_index)

        self.scene_manager.didChangeFrame.connect(self.update_frame_slider)
        self.ui.animPlayPause.clicked.connect(self.toggle_animation)
        self.ui.animPlayPause.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.ui.animPlayPause.customContextMenuRequested.connect(self.show_animation_speed_slider)
        def next_frame(*args, **kwargs):
            self.scene_manager.layer_set.animating = False
            self.scene_manager.layer_set.next_frame()
        self.ui.animForward.clicked.connect(next_frame)
        def prev_frame(*args, **kwargs):
            self.scene_manager.layer_set.animating = False
            self.scene_manager.layer_set.next_frame(frame_number=-1)
        self.ui.animBack.clicked.connect(prev_frame)

        # allow animation slider to set animation frame being displayed:
        self.ui.animationSlider.valueChanged.connect(self.animation_slider_jump_frame)

        # allow animation, once stopped, to propagate visibility to the document and layerlist:
        self.scene_manager.didChangeLayerVisibility.connect(self.document.animation_changed_visibility)

        # disable close button on panes
        for pane in [self.ui.areaProbePane, self.ui.layersPane, self.ui.layerDetailsPane, self.ui.rgbConfigPane]:
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
        self.layerSetsManager = LayerSetsManager(self.ui, self.ui.layerSetTabs, self.ui.layerInfoContents, self.document)
        self.behaviorLayersList = self.layerSetsManager.getLayerStackListViewModel()
        def update_probe_polygon(uuid, points, layerlist=self.behaviorLayersList):
            top_uuids = list(self.document.current_visible_layer_uuids)
            LOG.debug("top visible UUID is {0!r:s}".format(top_uuids))

            # TODO, when the plots manage their own layer selection, change this call
            # FUTURE, once the polygon is a layer, this will need to change
            # set the selection for the probe plot to the top visible layer(s)
            # new tabs should clone the information from the currently selected tab
            # the call below will check if this is a new polygon
            self.graphManager.set_default_layer_selections(*top_uuids)
            # update our current plot with the new polygon
            polygon_name = self.graphManager.currentPolygonChanged(polygonPoints=points)

            # do whatever other updates the scene manager needs
            self.scene_manager.on_new_polygon(polygon_name, points)

            if self.scene_manager._current_tool == TOOL.REGION_PROBE:
                self.ui.panZoomToolButton.click()

        self.scene_manager.newProbePolygon.connect(update_probe_polygon)

        # setup RGB configuration : FIXME clean this up into a behavior
        self.layerSetsManager.didChangeRGBLayerComponentRange.connect(self._user_set_rgb_range)
        self.layerSetsManager.didChangeRGBLayerSelection.connect(self._user_set_rgb_layer)
        self.document.didChangeComposition.connect(lambda *args: self._refresh_probe_results(*args[1:]))
        self.document.didChangeColorLimits.connect(self._refresh_probe_results)

        # self.queue.add('test', test_task(), 'test000')
        # self.ui.layers
        print(self.scene_manager.main_view.describe_tree(with_transform=True))
        self.document.didChangeColormap.connect(self.scene_manager.change_layers_colormap)
        self.document.didChangeColorLimits.connect(self.scene_manager.change_layers_color_limits)
        self.document.didSwitchLayerSet.connect(self.animation_reset_by_layer_set_switch)

        self.document.didChangeLayerVisibility.connect(self.update_frame_time_to_top_visible)
        self.document.didReorderLayers.connect(self.update_frame_time_to_top_visible)
        self.document.didRemoveLayers.connect(self.update_frame_time_to_top_visible)
        self.document.didAddBasicLayer.connect(self.update_frame_time_to_top_visible)
        self.document.didAddCompositeLayer.connect(self.update_frame_time_to_top_visible)
        self.document.didChangeProjection.connect(self.scene_manager.set_projection)

        self.ui.panZoomToolButton.toggled.connect(partial(self.change_tool, name=TOOL.PAN_ZOOM))
        self.ui.pointSelectButton.toggled.connect(partial(self.change_tool, name=TOOL.POINT_PROBE))
        self.ui.regionSelectButton.toggled.connect(partial(self.change_tool, name=TOOL.REGION_PROBE))
        self.change_tool(True)

        self.setup_menu()
        self.graphManager = ProbeGraphManager(self.ui.probeTabWidget, self.workspace, self.document, self.queue)
        self.graphManager.didChangeTab.connect(self.scene_manager.show_only_polygons)
        self.graphManager.didClonePolygon.connect(self.scene_manager.copy_polygon)
        self.graphManager.pointProbeChanged.connect(self.scene_manager.on_point_probe_set)
        self.graphManager.pointProbeChanged.connect(self.document.update_equalizer_values)
        self.graphManager.pointProbeChanged.connect(self.update_point_probe_text)
        self.graphManager.pointProbeChanged.connect(self.graphManager.update_point_probe_graph)

        self.scene_manager.newPointProbe.connect(self.graphManager.update_point_probe)
        zap = lambda *args: self.graphManager.update_point_probe(DEFAULT_POINT_PROBE)
        self.document.didAddBasicLayer.connect(zap)
        self.document.didAddCompositeLayer.connect(zap)
        # FIXME: These were added as a simple fix to update the proble value on layer changes, but this should really
        #        have its own manager-like object
        def _blackhole(*args, **kwargs):
            return self.update_point_probe_text(DEFAULT_POINT_PROBE)
        self.document.didChangeLayerVisibility.connect(_blackhole)
        self.document.didAddBasicLayer.connect(_blackhole)
        self.document.didAddCompositeLayer.connect(_blackhole)
        self.document.didRemoveLayers.connect(_blackhole)
        self.document.didReorderLayers.connect(_blackhole)
        if False:
            # XXX: Disable the below line if updating during animation is too much work
            # self.scene_manager.didChangeFrame.connect(lambda frame_info: update_probe_point(uuid=frame_info[-1]))
            pass
        else:
            # XXX: Disable the below line if updating the probe value during animation isn't a performance problem
            self.scene_manager.didChangeFrame.connect(lambda frame_info: self.ui.cursorProbeText.setText("Probe Value: <animating>"))

        # Set the projection based on the document's default
        self.document.change_projection()
        self.ui.projectionComboBox.setCurrentIndex(self.document.current_projection_index())

    def closeEvent(self, event, *args, **kwargs):
        LOG.debug('main window closing')
        self.workspace.close()

    def toggle_animation(self, action:QtGui.QAction=None, *args):
        new_state = self.scene_manager.layer_set.toggle_animation()
        self.ui.animPlayPause.setChecked(new_state)

    def _remove_paths_from_cache(self, paths):
        self.workspace.remove_paths_from_cache(paths)
        self.update_recent_file_menu()

    def open_from_cache(self, *args, **kwargs):
        if not self._open_cache_dialog:
            self._open_cache_dialog = OpenCacheDialog()
        paths = self.document.sort_paths(self.workspace.paths_in_cache)
        self._open_cache_dialog.activate(paths, self.open_paths, self._remove_paths_from_cache)

    def open_glob(self, *args, **kwargs):
        text, ok = QtGui.QInputDialog.getText(self, 'Open Glob Pattern',
            'Open files matching pattern:')
        from glob import glob
        if ok:
            paths = list(glob(text))
            self.open_paths(paths)

    def remove_region_polygon(self, action:QtGui.QAction=None, *args):
        if self.scene_manager.has_pending_polygon():
            self.scene_manager.clear_pending_polygon()
            return

        # Remove the polygon from other locations
        removed_name = self.graphManager.currentPolygonChanged(None)
        LOG.info("Clearing polygon with name '%s'", removed_name)
        self.scene_manager.remove_polygon(removed_name)

    def create_composite(self, action:QtGui.QAction=None, uuids=[], composite_type=COMPOSITE_TYPE.RGB):
        if composite_type not in [COMPOSITE_TYPE.RGB]:
            raise ValueError("Unknown or unimplemented composite type: %s" % (composite_type,))
        if len(uuids) == 0:
            # get the layers to composite from current selection
            uuids = list(self.behaviorLayersList.current_selected_uuids())
        if len(uuids)<3:  # pad with None
            uuids = uuids + ([None] * (3 - len(uuids)))
        # Don't use non-basic layers as starting points for the new composite
        uuids = [uuid if uuid is not None and self.document[uuid][INFO.KIND] == KIND.IMAGE else None for uuid in uuids]
        layer = self.document.create_rgb_composite(uuids[0], uuids[1], uuids[2])
        self.behaviorLayersList.select([layer.uuid])

    def take_screenshot(self):
        if not self._screenshot_dialog:
            self._screenshot_dialog = ExportImageDialog(self)
            self._screenshot_dialog.accepted.connect(self._save_screenshot)
        self._screenshot_dialog.set_total_frames(self.scene_manager.layer_set.max_frame + 1)
        self._screenshot_dialog.show()

    def _save_screenshot(self):
        from PIL import Image
        info = self._screenshot_dialog.get_info()
        LOG.info("Exporting image with options: {}".format(info))
        img_arrays = self.scene_manager.get_screenshot_array(info['frame_range'])
        params = {
            # 'transparency': info['transparency'],
            'duration': info['delay'],
        }

        if isinstance(img_arrays, list):
            if not len(img_arrays):
                LOG.error("Can't save zero frames returned from scene")
                return
            if info['filename'].endswith('.gif'):
                images = [Image.fromarray(x) for x in img_arrays]
                params['save_all'] = True
                if not info['loop']:
                    # rocking animation
                    # we want frames 0, 1, 2, 3, 2, 1
                    images = images + images[-2:0:-1]
                if info['delay'] is None:
                    # FIXME: Talk to the document about frame delays
                    params['delay'] = [200] * len(images)
                else:
                    params['delay'] = info['delay']
                params['loop'] = 0  # infinite number of loops
                new_img = images[0]
                params['append_images'] = images[1:]
            else:
                LOG.warning("File format does not support animation: {}".format(info['filename']))
                new_img = Image.fromarray(img_arrays[0])
        else:
            new_img = Image.fromarray(img_arrays)

        LOG.info("Saving screenshot to '{}'".format(info['filename']))
        new_img.save(info['filename'], **params)

    def setup_menu(self):
        open_action = QtGui.QAction("&Open...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.interactive_open_files)

        exit_action = QtGui.QAction("&Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(QtGui.qApp.quit)
        
        open_cache_action = QtGui.QAction("Open from Cache...", self)
        open_cache_action.setShortcut("Ctrl+A")
        open_cache_action.triggered.connect(self.open_from_cache)

        open_glob_action = QtGui.QAction("Open Filename Pattern...", self)
        open_glob_action.setShortcut("Ctrl+Shift+O")
        open_glob_action.triggered.connect(self.open_glob)

        menubar = self.ui.menubar
        file_menu = menubar.addMenu('&File')
        file_menu.addAction(open_action)
        file_menu.addAction(open_cache_action)
        file_menu.addAction(open_glob_action)
        self._recent_files_menu = file_menu.addMenu('Open Recent')

        screenshot_action = QtGui.QAction("Export Image", self)
        screenshot_action.setShortcut("Ctrl+I")
        screenshot_action.triggered.connect(self.take_screenshot)
        file_menu.addAction(screenshot_action)

        file_menu.addAction(exit_action)

        next_time = QtGui.QAction("Next Time", self)
        next_time.setShortcut(QtCore.Qt.Key_Right)
        next_slot = partial(self.next_last_time, direction=1)
        next_time.triggered.connect(next_slot)
        # self.ui.animForward.clicked.connect(next_slot)

        focus_current = QtGui.QAction("Focus Current Timestep", self)
        focus_current.setShortcut('.')
        focus_current.triggered.connect(partial(self.next_last_band, direction=0))

        prev_time = QtGui.QAction("Previous Time", self)
        prev_time.setShortcut(QtCore.Qt.Key_Left)
        prev_slot = partial(self.next_last_time, direction=-1)
        prev_time.triggered.connect(prev_slot)
        # self.ui.animBack.clicked.connect(prev_slot)

        focus_prev_band = QtGui.QAction("Next Band", self)
        focus_prev_band.setShortcut(QtCore.Qt.Key_Up)
        focus_prev_band.triggered.connect(partial(self.next_last_band, direction=-1))

        focus_next_band = QtGui.QAction("Previous Band", self)
        focus_next_band.setShortcut(QtCore.Qt.Key_Down)
        focus_next_band.triggered.connect(partial(self.next_last_band, direction=1))

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
        flip_colormap.triggered.connect(lambda: self.document.flip_climits_for_layers([self.document.current_visible_layer_uuid]))

        cycle_borders = QtGui.QAction("Cycle &Borders", self)
        cycle_borders.setShortcut('B')
        cycle_borders.triggered.connect(self.scene_manager.cycle_borders_color)

        cycle_grid = QtGui.QAction("Cycle &Lat/Lon Grid", self)
        cycle_grid.setShortcut('L')
        cycle_grid.triggered.connect(self.scene_manager.cycle_grid_color)

        remove = QtGui.QAction("Remove Layer", self)
        remove.setShortcut(QtCore.Qt.Key_Delete)
        remove.triggered.connect(self.remove_layer)

        clear = QtGui.QAction("Clear Region Selection", self)
        clear.setShortcut(QtCore.Qt.Key_Escape)
        clear.triggered.connect(self.remove_region_polygon)

        composite = QtGui.QAction("Create Composite", self)
        composite.setShortcut('C')
        composite.triggered.connect(self.create_composite)

        toggle_point = QtGui.QAction("Toggle Point Probe", self)
        toggle_point.setShortcut('X')
        toggle_point.triggered.connect(lambda: self.graphManager.toggle_point_probe(DEFAULT_POINT_PROBE))

        edit_menu = menubar.addMenu('&Edit')
        edit_menu.addAction(remove)
        edit_menu.addAction(clear)
        edit_menu.addAction(toggle_point)

        layer_menu = menubar.addMenu('&Layer')
        layer_menu.addAction(composite)

        view_menu = menubar.addMenu('&View')
        view_menu.addAction(animate)
        view_menu.addAction(prev_time)
        view_menu.addAction(focus_current)
        view_menu.addAction(next_time)
        view_menu.addAction(focus_next_band)
        view_menu.addAction(focus_prev_band)
        view_menu.addAction(change_order)
        view_menu.addAction(toggle_vis)
        view_menu.addAction(flip_colormap)
        view_menu.addAction(cycle_borders)
        view_menu.addAction(cycle_grid)

        self.update_recent_file_menu()
        menubar.setEnabled(True)

    def setup_key_releases(self):
        def cb_factory(required_key, cb):
            def tmp_cb(key, cb=cb):
                if key.text == required_key:
                    return cb()
            return tmp_cb

        self.scene_manager.main_canvas.events.key_release.connect(cb_factory("t", self.scene_manager.next_tool))

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
    if "win" in sys.platform:
        # Windows AND Mac/OSX
        default_workspace = os.path.expanduser(os.path.join("~", "Documents", "sift_workspace"))
    else:
        default_workspace = os.path.expanduser(os.path.join("~", "sift_workspace"))

    import argparse
    parser = argparse.ArgumentParser(description="Run CSPOV")
    parser.add_argument("-w", "--workspace", default=default_workspace,
                        help="Specify workspace base directory")
    parser.add_argument("-s", "--space", default=256, type=int,
                        help="Specify max amount of data to hold in workspace in Gigabytes")
    parser.add_argument("--border-shapefile", default=None,
                        help="Specify alternative coastline/border shapefile")
    parser.add_argument("--glob-pattern", default=os.environ.get("TIFF_GLOB", None),
                        help="Specify glob pattern for input images")
    parser.add_argument("-c", "--center", nargs=2, type=float,
                        help="Specify center longitude and latitude for camera")
    parser.add_argument("--desktop", type=int, default=0,
                        help="Number of monitor/display to show the main window on (0 for main, 1 for secondary, etc.)")
    parser.add_argument('-v', '--verbose', dest='verbosity', action="count", default=int(os.environ.get("VERBOSITY", 2)),
                        help='each occurrence increases verbosity 1 level through ERROR-WARNING-INFO-DEBUG (default INFO)')
    args = parser.parse_args()

    levels = [logging.ERROR, logging.WARN, logging.INFO, logging.DEBUG]
    level=levels[min(3, args.verbosity)]
    logging.basicConfig(level=level)
    # FIXME: This is needed because shapely 1.5.11 sucks
    logging.getLogger().setLevel(level)
    # logging.getLogger('vispy').setLevel(level)

    if not os.path.exists(args.workspace):
        LOG.info("Creating SIFT Workspace: %s", args.workspace)
        os.makedirs(args.workspace)
    else:
        LOG.info("Using SIFT Workspace: %s", args.workspace)

    app.create()
    # app = QApplication(sys.argv)
    window = Main(
        workspace_dir=args.workspace,
        workspace_size=args.space,
        glob_pattern=args.glob_pattern,
        border_shapefile=args.border_shapefile,
        center=args.center,
    )
    screen = QtGui.QApplication.desktop()
    screen_geometry = screen.screenGeometry(args.desktop)
    if 'darwin' not in sys.platform:
        w, h = screen_geometry.width() - 400, screen_geometry.height() - 300
        window.setGeometry(200, 150, w, h)
    else:
        size = window.size()
        w, h = size.width(), size.height()
        center = screen_geometry.center()
        screen_x, screen_y = center.x(), center.y()
        window.move(int(screen_x - w / 2.), int(screen_y - h / 2.))


    window.show()
    # bring window to front
    window.raise_()
    app.run()

if __name__ == '__main__':
    sys.exit(main())
