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

import gc
import logging
import os
import signal
import sys
import typing as typ
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from functools import partial
from glob import glob
from types import FrameType
from uuid import UUID

from PyQt5 import QtCore, QtGui, QtWidgets
from vispy import app

import uwsift.ui.open_cache_dialog_ui as open_cache_dialog_ui
from uwsift import (
    AUTO_UPDATE_MODE__ACTIVE,
    USE_INVENTORY_DB,
    USE_TILED_GEOLOCATED_IMAGES,
    __version__,
    config,
)
from uwsift.common import Info, Tool
from uwsift.control.layer_tree import LayerStackTreeViewModel
from uwsift.model.area_definitions_manager import AreaDefinitionsManager

# To have consistent logging for all modules (also for their static
# initialization) it must be set up before importing them.
from uwsift.model.composite_recipes import RecipeManager
from uwsift.model.document import Document
from uwsift.model.layer_model import LayerModel
from uwsift.model.product_dataset import ProductDataset
from uwsift.queue import TASK_DOING, TASK_PROGRESS, TaskQueue

# this is generated with pyuic4 pov_main.ui >pov_main_ui.py
from uwsift.ui.pov_main_ui import Ui_MainWindow
from uwsift.util import (
    DOCUMENT_SETTINGS_DIR,
    USER_CACHE_DIR,
    WORKSPACE_DB_DIR,
    HeapProfiler,
    check_grib_definition_dir,
    check_imageio_deps,
    get_package_data_dir,
)
from uwsift.util.logger import configure_loggers
from uwsift.view.algebraic_config import AlgebraicLayerConfigPane
from uwsift.view.colormap_editor import ColormapEditor
from uwsift.view.export_image import ExportImageHelper
from uwsift.view.layer_details import SingleLayerInfoPane
from uwsift.view.probes import DEFAULT_POINT_PROBE, ProbeGraphManager
from uwsift.view.rgb_config import RGBLayerConfigPane
from uwsift.view.scene_graph import SceneGraphManager
from uwsift.workspace import CachingWorkspace, SimpleWorkspace
from uwsift.workspace.collector import ResourceSearchPathCollector

LOG = logging.getLogger(__name__)
configure_loggers()

PROGRESS_BAR_MAX = 1000
STATUS_BAR_DURATION = 2000  # ms

WATCHDOG_DATETIME_FORMAT_DISPLAY = "%Y-%m-%d %H:%M:%S %Z"
WATCHDOG_DATETIME_FORMAT_STORE = "%Y-%m-%d %H:%M:%S %z"

UWSIFT_ANIM_INDICATOR_DISABLED = True

EXIT_FORCED_SHUTDOWN = 101
EXIT_CONFIRMED_SHUTDOWN = 102


def test_layers_from_directory(doc, layer_tiff_glob):
    return doc.open_file(glob(layer_tiff_glob))


def test_layers(doc, glob_pattern=None):
    if glob_pattern:
        return test_layers_from_directory(doc, glob_pattern)
    return []


async def do_test_cycle(txt: QtWidgets.QWidget):
    from asyncio import sleep

    n = 0
    while True:
        txt.setText(str(n))
        await sleep(1)
        n += 1


class OpenCacheDialog(QtWidgets.QDialog):
    _opener = None
    _remover = None

    def __init__(self, parent, opener, remover):
        super(OpenCacheDialog, self).__init__(parent)
        self.ui = open_cache_dialog_ui.Ui_openFromCacheDialog()
        self.ui.setupUi(self)
        self.ui.cacheListWidget.setSelectionMode(QtWidgets.QListWidget.ExtendedSelection)
        self.ui.removeFromCacheButton.clicked.connect(self._do_remove)
        self._opener = opener
        self._remover = remover

    def activate(self, uuid_to_name):
        self.ui.cacheListWidget.clear()
        # assume uuid_to_name is already an OrderedDict:
        sorted_items = uuid_to_name.items()
        for uuid, name in sorted_items:
            li = QtWidgets.QListWidgetItem(name)
            li.setData(QtCore.Qt.UserRole, uuid)
            self.ui.cacheListWidget.addItem(li)
        self.show()

    def _do_remove(self, *args, **kwargs):
        to_remove = []
        for item in self.ui.cacheListWidget.selectedItems():
            to_remove.append(item.data(QtCore.Qt.UserRole))
            self.ui.cacheListWidget.removeItemWidget(item)
        self._remover(to_remove)
        self.hide()

    def accept(self, *args, **kwargs):
        self.hide()
        to_open = [item.data(QtCore.Qt.UserRole) for item in self.ui.cacheListWidget.selectedItems()]
        LOG.info("opening from cache: " + repr(to_open))
        self._opener(to_open)

    def reject(self, *args, **kwargs):
        self.hide()


class AnimationSpeedPopupWindow(QtWidgets.QWidget):
    _slider = None
    _active = False

    def __init__(self, slot, *args, **kwargs):
        super(AnimationSpeedPopupWindow, self).__init__(*args, **kwargs)
        from PyQt5.QtCore import Qt

        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Popup)
        self.setFocusPolicy(Qt.ClickFocus)
        self.setToolTip("Set animation speed")
        self._slider = QtWidgets.QSlider(parent=self)
        n, x = 2, 150  # frames per 10 seconds
        self._slider.setRange(n, x)  #
        self._slot = slot
        self._slider.valueChanged.connect(self._changed)
        self._layout = QtWidgets.QHBoxLayout()
        self._layout.addWidget(self._slider)
        self.setLayout(self._layout)

    def _convert(self, val: int, reverse: bool = False) -> float:
        """Map 1..100 nonlinearly to 10ms .. 5000ms.

        Args:
            val: raw value to remap to milliseconds
            reverse: when True, reverse conversion milliseconds to ticks

        Returns: converted value
        """
        if reverse:  # convert milliseconds to fp10s
            fp10s = 10000.0 / float(val)
            return fp10s
        else:
            ms = 10000.0 / float(val)
            return ms

    def _changed(self, value):
        if not self._active:
            return
        fps = float(value) / 10.0
        self.setToolTip("{0:.1f} fps".format(fps))
        val = self._convert(value)
        self._slot(val)

    def show_at(self, pos, val):
        from PyQt5.QtCore import QPoint, QRect, QSize

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
    dn, fn = os.path.split(path)
    if dn and not fn:
        yield dn
    if len(dn) > 0 and dn != path and len(fn) > 0:
        for pc in _recursive_split(dn):
            yield pc
    if len(fn):
        yield fn


def _common_path_prefix_seq(paths):
    pathlists = [list(_recursive_split(path)) for path in paths]
    for component_list in zip(*pathlists):
        if all(c == component_list[0] for c in component_list[1:]):
            yield component_list[0]
        else:
            break


def _common_path_prefix(paths):
    "find the most common directory shared by a list of paths"
    paths = list(paths)
    LOG.debug("looking for common path prefix for {}".format(repr(paths)))
    if len(paths) == 1:
        return os.path.split(paths[0])[0]
    parts = list(_common_path_prefix_seq(paths))
    if parts:
        return os.path.join(*parts)
    else:
        return None


class UserControlsAnimation(QtCore.QObject):
    """Controller behavior focused around animation bar and next/last key bindings.

    Connects between document, scene graph manager, layer list view of document; uses UI elements.
    Is composed into MainWindow.
    - scrub left and right on slider
    - next and last timestep (if time animation)
    - next and last bandstep (if band animation)
    - update time display above time slider
    - update animation rate using popup widget
    """

    ui = None
    document: Document = None
    scene_manager: SceneGraphManager = None
    layer_list_model: LayerStackTreeViewModel = None
    _animation_speed_popup = None  # window we'll show temporarily with animation speed popup

    def __init__(
        self, ui, scene_manager: SceneGraphManager, document: Document, layer_list_model: LayerStackTreeViewModel
    ):
        """
        Args:
            ui: QtDesigner UI element tree for application
            scene_manager: Map display manager, needed for controller screen animation
            document: document object, needed for sibling lookups
            layer_list_model: model used to display current layer list, needed for selection
        """
        super(UserControlsAnimation, self).__init__()
        self.ui = ui
        self.scene_manager = scene_manager
        self.document = document
        self.layer_list_model = layer_list_model

        self.ui.animPlayPause.clicked.connect(self.toggle_animation)
        self.ui.animPlayPause.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.ui.animPlayPause.customContextMenuRequested.connect(self.show_animation_speed_slider)

        self.ui.animForward.clicked.connect(self.next_frame)
        self.ui.animBack.clicked.connect(self.prev_frame)

        # allow animation slider to set animation frame being displayed:
        self.ui.animationSlider.valueChanged.connect(self.animation_slider_jump_frame)

        # allow animation, once stopped, to propagate visibility to the document and layerlist:
        self.scene_manager.didChangeLayerVisibility.connect(self.document.animation_changed_visibility)

        self.document.didChangeLayerVisibility.connect(self.update_frame_time_to_top_visible)
        self.document.didReorderDatasets.connect(self.update_frame_time_to_top_visible)
        self.document.didRemoveDatasets.connect(self.update_frame_time_to_top_visible)
        self.document.didAddCompositeDataset.connect(self.update_frame_time_to_top_visible)

    def next_frame(self, *args, **kwargs):
        """Advance a frame along the animation order."""
        self.scene_manager.animation_controller.animating = False
        self.scene_manager.animation_controller.time_manager.step()

    def prev_frame(self, *args, **kwargs):
        """Retreat a frame along the animation list."""
        self.scene_manager.animation_controller.animating = False
        self.scene_manager.animation_controller.time_manager.step(backwards=True)

    def reset_frame_slider(self, *args, **kwargs):
        """Reset frame slider to show current animation state in document when aniamtion list changes."""
        frame_count = len(self.document.current_animation_order)
        frame_index = None  # self.scene_manager.layer_set._frame_number  # FIXME BAAD
        self.ui.animationSlider.setRange(0, frame_count - 1)
        self.ui.animationSlider.setValue(frame_index or 0)
        self.ui.animPlayPause.setDown(False)
        self.ui.animationSlider.repaint()
        self.update_frame_time_to_top_visible()

    def update_frame_slider(self, frame_info):
        """
        Update animation frame slider and time display to reflect current document animation state.
        Args:
            frame_info: tuple, ultimately from SceneGraphManager.layer_set callback
        """
        frame_index, frame_count, animating, uuid = frame_info[:4]
        self.ui.animationSlider.setRange(0, frame_count - 1)
        self.ui.animationSlider.setValue(frame_index or 0)
        self.ui.animPlayPause.setDown(animating)
        self.ui.animationSlider.repaint()
        if animating:
            if not UWSIFT_ANIM_INDICATOR_DISABLED:
                t_sim = self.scene_manager.animation_controller.time_manager.create_formatted_t_sim()
                self.ui.animationLabel.setText(t_sim)
        else:
            self.update_frame_time_to_top_visible()

    def update_frame_time_to_top_visible(self, *args):
        """Update frame slider's time display to show the current top layer's time."""
        self.ui.animationLabel.setText(self.document.time_label_for_uuid(self.document.current_visible_layer_uuid))

    def animation_slider_jump_frame(self, event, *args, **kwargs):
        """Update display to match frame slider change."""
        frame = self.ui.animationSlider.value()
        self.scene_manager.set_frame_number(frame)
        # TODO: update layer list to reflect what layers are visible/hidden?

    def set_animation_speed(self, milliseconds):
        """Change frame rate as measured in milliseconds."""
        LOG.info("animation speed set to {}ms".format(milliseconds))
        self.scene_manager.animation_controller.animation_speed = milliseconds

    def show_animation_speed_slider(self, pos: QtCore.QPoint, *args):
        """Show frame-rate slider as a pop-up control, at current mouse position."""
        LOG.info("menu requested for animation control")
        gpos = self.ui.animPlayPause.mapToGlobal(pos)

        if self._animation_speed_popup is None:
            self._animation_speed_popup = popup = AnimationSpeedPopupWindow(slot=self.set_animation_speed, parent=None)
        else:
            popup = self._animation_speed_popup
        if not popup.isVisible():
            popup.show_at(gpos, self.scene_manager.animation_controller.animation_speed)

    def animation_reset_by_layer_set_switch(self, *args, **kwargs):
        """Perform necessary control resets when document layer set is swapped."""
        self.reset_frame_slider()
        self.update_frame_time_to_top_visible()

    def toggle_animation(self, action: QtWidgets.QAction = None, *args):
        """Toggle animation on/off."""
        new_state = self.scene_manager.animation_controller.toggle_animation()
        self.ui.animPlayPause.setChecked(new_state)


class Main(QtWidgets.QMainWindow):
    # TODO: the following settings, is this actually the best place to store
    #  them here?
    _last_open_dir: str = None  # directory to open files in (preselection)
    _last_reader: str = None  # reader to open files with (preselection)

    _recent_files_menu: QtWidgets.QMenu = None  # QMenu
    _open_cache_dialog: QtWidgets.QDialog = None
    _screenshot_dialog: QtWidgets.QDialog = None
    _cmap_editor = None  # Gradient editor widget
    _resource_collector: ResourceSearchPathCollector = None
    _resource_collector_timer: QtCore.QTimer = None
    _last_imported_dataset_uuid: typ.Optional[UUID] = None
    _palette_text_green: QtGui.QPalette = None
    _palette_text_red: QtGui.QPalette = None
    _max_tolerable_idle_time: float = -1
    _max_tolerable_dataset_age: float = -1
    _heartbeat_file = None

    didFinishLoading = QtCore.pyqtSignal(list)

    def interactive_open_files(self, *args, files=None, **kwargs):
        self.scene_manager.animation_controller.animating = False
        # http://pyqt.sourceforge.net/Docs/PyQt4/qfiledialog.html#getOpenFileNames
        filename_filters = [
            # 'All files (*.*)',
            "All supported files (*.nc *.nc4)",
            "GOES-16 NetCDF (*.nc *.nc4)",
        ]
        filter_str = ";;".join(filename_filters)
        files = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Select one or more files to open", self._last_open_dir or os.getenv("HOME"), filter_str
        )[0]
        self.open_paths(files)

    def _bgnd_open_paths(self, paths, uuid_list, **importer_kwargs):
        """Background task runs on a secondary thread"""
        LOG.info("opening products from {} paths in background".format(len(paths)))
        for progress in self.document.import_files(paths, **importer_kwargs):
            yield progress
            uuid_list.append(progress["uuid"])
        yield {TASK_DOING: "products loaded from paths", TASK_PROGRESS: 1.0}

    def _bgnd_open_paths_finish(self, isok: bool, uuid_list: typ.List[UUID]):
        """Main thread finalization after background imports are done.

        Args:
            isok: whether _bgnd_open_paths ran without exception
            uuid_list: list of UUIDs it generated
        """
        self.didFinishLoading.emit(uuid_list)

        if not uuid_list:
            raise ValueError("no UUIDs provided by background open" " in _bgnd_open_paths_finish")
        if not isok:
            raise ValueError("background open did not succeed")

        if AUTO_UPDATE_MODE__ACTIVE:
            # Choose one of the recently loaded datasets for reporting
            # "vital signs"
            dataset = self.document[uuid_list[0]]

            self._update_dataset_timestamps(dataset)
            self._update_heartbeat_file(dataset)

    def open_paths(self, paths, **importer_kwargs):
        paths = list(paths)
        if not paths:
            return
        uli = []
        bop = partial(self._bgnd_open_paths, uuid_list=uli, **importer_kwargs)
        bopf = partial(self._bgnd_open_paths_finish, uuid_list=uli)
        self.queue.add("load_files", bop(paths), "Open {} files".format(len(paths)), and_then=bopf, interactive=False)
        # don't use <algebraic layer ...> type paths
        self._last_open_dir = _common_path_prefix([x for x in paths if x[0] != "<"]) or self._last_open_dir
        if USE_INVENTORY_DB:
            self.update_recent_file_menu()

    def activate_products_by_uuid(self, uuids):
        uuids = list(uuids)
        if not uuids:
            return
        for uuid in uuids:
            self.document.activate_product_uuid_as_new_dataset(uuid)

    def dropEvent(self, event):
        LOG.debug("drop event on mainwindow")
        mime = event.mimeData()
        if mime.hasUrls:
            event.setDropAction(QtCore.Qt.CopyAction)
            event.accept()
            paths = [str(url.toLocalFile()) for url in mime.urls()]
            self.document.import_files(paths)
        else:
            event.ignore()

    def change_tool(self, checked, name=Tool.PAN_ZOOM):
        if checked is not True:
            return
        self.scene_manager.change_tool(name)

    def update_recent_file_menu(self, *args, **kwargs):
        uuid_to_name = self.workspace.recently_used_products()
        LOG.debug("recent uuids: {}".format(repr(uuid_to_name.keys())))
        self._recent_files_menu.clear()
        for uuid, p_name in uuid_to_name.items():

            def openit(checked=False, uuid=uuid):
                LOG.debug("open recent product {}".format(uuid))
                self.scene_manager.animation_controller.animating = False
                self.activate_products_by_uuid([uuid])

            open_action = QtWidgets.QAction(p_name, self)
            open_action.triggered.connect(openit)
            self._recent_files_menu.addAction(open_action)

    def update_progress_bar(self, status_info, *args, **kwargs):
        active = status_info[0] if len(status_info) > 0 else None
        val = active[TASK_PROGRESS] if active else 0.0
        txt = active[TASK_DOING] if active else ""
        val = self.queue.progress_ratio(val)
        self.ui.progressBar.setValue(int(val * PROGRESS_BAR_MAX))
        self.ui.progressText.setText(txt)

    def toggle_visibility_on_selected_layers(self, *args, **kwargs):
        model_indexes = self.ui.treeView.selectedIndexes()
        self.layer_model.toggle_layers_visibility(model_indexes)

    def update_point_probe_text(self, probe_name, state=None, xy_pos=None, uuid=None, animating=None):
        if uuid is None:
            (
                top_probeable_layer,
                product_dataset,
            ) = self.layer_model.get_top_probeable_layer_with_active_product_dataset()
            uuid = None if product_dataset is None else product_dataset.uuid
        if state is None or xy_pos is None:
            _state, _xy_pos = self.graphManager.current_point_probe_status(probe_name)
            if state is None:
                state = _state
            if xy_pos is None:
                xy_pos = _xy_pos

        if xy_pos is not None and state:
            lon, lat = xy_pos
            lon = lon % 360 if lon > 0 else lon % -360 + 360
            lon = lon - 360 if lon > 180 else lon
            lon_str = "{:>6.02f} {}".format(abs(lon), "W" if lon < 0 else "E")
            lat_str = "{:>6.02f} {}".format(abs(lat), "S" if lat < 0 else "N")
            probe_loc = "{}, {}".format(lon_str, lat_str)
        else:
            probe_loc = "{:>6s}  , {:>6s}  ".format("N/A", "N/A")

        col, row = "N/A", "N/A"

        if animating:
            data_str = "<animating>"
        elif state and uuid is not None:
            try:
                data_point = self.workspace.get_content_point(uuid, xy_pos)
                col, row = self.workspace.position_to_grid_index(uuid, xy_pos)
            except ValueError:
                LOG.debug("Could not get data value", exc_info=True)
                data_point = None

            if data_point is None:
                data_str = "N/A"
                layer_str = "N/A"
            else:
                info = top_probeable_layer.info
                unit_info = info[Info.UNIT_CONVERSION]
                data_point = unit_info[1](data_point)
                data_str = unit_info[2](data_point, numeric=False)
                if info.get(Info.CENTRAL_WAVELENGTH):
                    wl = info[Info.CENTRAL_WAVELENGTH]
                    if wl < 4.1:
                        wl_str = "{:0.02f} µm".format(wl)
                    else:
                        wl_str = "{:0.01f} µm".format(wl)
                    layer_str = "{}, {}".format(info[Info.SHORT_NAME], wl_str)
                else:
                    layer_str = info[Info.SHORT_NAME]
        else:
            data_str = "N/A"
            layer_str = "N/A"
        self.ui.cursorProbeLayer.setText(layer_str)
        self.ui.cursorProbeText.setText("{} ({}) [{}, {}]".format(data_str, probe_loc, col, row))

    @staticmethod
    def run_gc_after_layer_deletion(new_order: tuple, removed_uuids: list, first_removed_row: int, rows_removed: int):
        """
        Trigger a full garbage collection run after the deletion of a layer from the scene graph.
        The code uses cyclic and weak references, which can only be freed by the GC.
        """
        unreachable_object_count = gc.collect()
        LOG.debug(f"GC found {unreachable_object_count} unreachable objects")

    def _update_heartbeat_file(self, dataset: ProductDataset):
        """
        Write the dataset creation time into the heartbeat file. The time of
        last dataset update can be retrieved as the file modification time.

        :param dataset: recently loaded dataset
        """
        dataset_sched_time_utc = dataset.sched_time.replace(tzinfo=timezone.utc)
        fmt_time = dataset_sched_time_utc.strftime(WATCHDOG_DATETIME_FORMAT_STORE).rstrip()

        journal_path = self._heartbeat_file + "-journal"
        with open(journal_path, "w") as file:
            file.write(f"{self.pid}\n")
            file.write(fmt_time + "\n")
            file.flush()

        os.rename(journal_path, self._heartbeat_file)

    def _update_dataset_timestamps(self, dataset: ProductDataset):
        """
        Update the timestamp displayed in timeLastDatasetCreationLineEdit and
        timeLastDatasetImportLineEdit. The import time is the current local
        time.

        If max_update_interval isn't None, then the dataset import time will be
        colored green if the time between now and last import time is smaller
        than max_update_interval. Otherwise it will be colored red.

        :param dataset: recently loaded dataset
        """

        self._last_imported_dataset_uuid = dataset.uuid

        dataset_sched_time_utc = dataset.sched_time.replace(tzinfo=timezone.utc)
        self.ui.timeLastDatasetCreationLineEdit.setText(
            dataset_sched_time_utc.strftime(WATCHDOG_DATETIME_FORMAT_DISPLAY)
        )

        self._last_imported_dataset_import_time = datetime.now(tz=timezone.utc)
        self.ui.timeLastDatasetImportLineEdit.setText(
            self._last_imported_dataset_import_time.strftime(WATCHDOG_DATETIME_FORMAT_DISPLAY)
        )

    def _clear_last_dataset_creation_time(
        self, reordered_layers: tuple, removed_uuids: typ.List[UUID], first_row_removed: int, num_rows_removed: int
    ):
        """
        Clear the LineEdit if the layer, from which the creation time was
        extracted, is deleted from the scene graph.
        """
        if self._last_imported_dataset_uuid in removed_uuids:
            self.ui.timeLastDatasetCreationLineEdit.clear()
            self._last_imported_dataset_uuid = None

    def _update_current_time(self):
        """
        Update currentTimeLineEdit with the current local time.
        """
        now_utc = datetime.now(tz=timezone.utc)
        self.ui.currentTimeLineEdit.setText(now_utc.strftime(WATCHDOG_DATETIME_FORMAT_DISPLAY))

        if not self._last_imported_dataset_uuid:
            return

        if self._max_tolerable_dataset_age > 0:
            dataset = self.document[self._last_imported_dataset_uuid]

            dataset_sched_time_utc = dataset.sched_time.replace(tzinfo=timezone.utc)
            dataset_age = now_utc - dataset_sched_time_utc
            if dataset_age.total_seconds() > self._max_tolerable_dataset_age:
                palette = self._palette_text_red
            else:
                palette = self._palette_text_green
            self.ui.timeLastDatasetCreationLineEdit.setPalette(palette)

        if self._max_tolerable_idle_time > 0:
            idle_time = now_utc - self._last_imported_dataset_import_time
            if idle_time.total_seconds() > self._max_tolerable_idle_time:
                palette = self._palette_text_red
            else:
                palette = self._palette_text_green
            self.ui.timeLastDatasetImportLineEdit.setPalette(palette)

    def _restart_handler(self, signal: int, frame: FrameType):
        if self._restart_handler_active:
            return
        self._restart_handler_active = True

        if self._restart_ask_again_interval is not None:
            if self._last_restart_request is None:
                self._last_restart_request = datetime.now()
            else:
                since_last_restart = datetime.now() - self._last_restart_request
                if since_last_restart < self._restart_ask_again_interval:
                    LOG.debug("Ignoring restart request because last restart " "request was denied recently")
                    return

        msg_box = QtWidgets.QMessageBox()
        msg_box.setIcon(QtWidgets.QMessageBox.Information)
        msg_box.setText("Do you want to perform the requested restart?")
        msg_box.setWindowTitle("Restart Request")
        msg_box.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)

        def force_restart():
            msg_box.close()
            LOG.info("forced shutdown after restart request")
            sys.exit(EXIT_FORCED_SHUTDOWN)

        if self._restart_popup_deadline is not None:
            timer = QtCore.QTimer()
            timer.setSingleShot(True)
            timer.timeout.connect(force_restart)
            timer.start(self._restart_popup_deadline * 1000)

        if msg_box.exec() == QtWidgets.QMessageBox.Yes:
            LOG.info("shutdown in order to comply with restart request")
            sys.exit(EXIT_CONFIRMED_SHUTDOWN)
        else:
            LOG.info("ignored restart request")
            self._last_restart_request = datetime.now()
            self._restart_handler_active = False

    def _init_auto_restart(self):
        restart_popup_deadline = config.get("watchdog.auto_restart_popup_deadline", 0)
        if restart_popup_deadline == 0:
            LOG.warning("deadline for the auto restart is disabled")
            self._restart_popup_deadline = None
        else:
            self._restart_popup_deadline = int(restart_popup_deadline)

        restart_ask_again_interval = config.get("watchdog.auto_restart_ask_again_interval", 0)
        if restart_ask_again_interval == 0:
            LOG.warning("User won't be asked again to restart")
            self._restart_ask_again_interval = None
        else:
            self._restart_ask_again_interval = timedelta(seconds=int(restart_ask_again_interval))

        self._restart_handler_active = False
        self._last_restart_request = None
        signal.signal(signal.SIGUSR1, self._restart_handler)

    def __init__(
        self,
        config_dir=None,
        workspace_dir=None,
        cache_size=None,
        glob_pattern=None,
        search_paths=None,
        border_shapefile=None,
        center=None,
        clear_workspace=False,
    ):
        super(Main, self).__init__()

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        if AUTO_UPDATE_MODE__ACTIVE:
            self.ui.animFrame.hide()
            self.ui.timelineFrame.hide()
            self.ui.progressBar.hide()
            self.ui.progressText.hide()

            self._init_auto_restart()
        else:
            self.ui.watchdogFrame.hide()
        # FIXME: Slider does not currently work as intended. Re-enable later
        self.ui.timelineScaleSlider.setDisabled(True)
        if UWSIFT_ANIM_INDICATOR_DISABLED:
            self.ui.animSliderFrame.hide()
        self._init_font_sizes()

        self.setWindowTitle(self.windowTitle().replace("|X.X.X|", __version__))
        self._init_arrange_panes()

        self.queue = TaskQueue()
        self.ui.progressBar.setRange(0, PROGRESS_BAR_MAX)
        self.queue.didMakeProgress.connect(self.update_progress_bar)

        # create manager and helper classes
        if USE_INVENTORY_DB:
            self.workspace = CachingWorkspace(
                workspace_dir, max_size_gb=cache_size, queue=self.queue, initial_clear=clear_workspace
            )
        else:
            self.workspace = SimpleWorkspace(workspace_dir)
        self.document = doc = Document(self.workspace, config_dir=config_dir, queue=self.queue)
        self.document.didRemoveDatasets.connect(self.run_gc_after_layer_deletion)
        self.scene_manager = SceneGraphManager(
            doc, self.workspace, self.queue, borders_shapefiles=border_shapefile, center=center, parent=self
        )
        self.export_image = ExportImageHelper(self, self.document, self.scene_manager)
        self._wizard_dialog = None

        self._init_layer_model()
        self._init_layer_panes()
        self._init_algebraic_pane()
        self._init_rgb_pane()
        self._init_recipe_manager()
        self._init_map_widget()
        self._init_qml_timeline()

        if AUTO_UPDATE_MODE__ACTIVE:
            self._init_update_times_display()

        self.animation = UserControlsAnimation(self.ui, self.scene_manager, self.document, self.layer_list_model)

        # disable close button on panes
        panes = [
            self.ui.areaProbePane,
            self.ui.rgbConfigPane,
            self.ui.algebraicConfigPane,
        ]
        for pane in panes:
            pane.setFeatures(QtWidgets.QDockWidget.DockWidgetFloatable | QtWidgets.QDockWidget.DockWidgetMovable)
        # Make the panes on the right side 375px wide
        self.resizeDocks(panes, [375] * len(panes), QtCore.Qt.Horizontal)

        test_layers(self.document, glob_pattern=glob_pattern)

        # Interaction Setup
        self._init_key_releases()

        self.scheduler = QtCore.QTimer(parent=self)
        self.scheduler.setInterval(200)
        self.scheduler.timeout.connect(partial(self.scene_manager.on_view_change, self.scheduler))

        def start_wrapper(timer, event):
            """Simple wrapper around a timers start method so we can accept but ignore the event provided"""
            timer.start()

        self.scene_manager.main_view.scene.transform.changed.connect(partial(start_wrapper, self.scheduler))

        print(self.scene_manager.main_view.describe_tree(with_transform=True))

        self._init_tool_controls()
        self._init_menu()
        self._init_point_polygon_probes()

        # Set the projection based on the document's default
        self.document.change_projection()
        self.ui.projectionComboBox.setCurrentIndex(self.document.current_projection_index())
        if USE_INVENTORY_DB:
            self._init_metadata_background_collection(search_paths)

        # FIXME: make sure sync of metadata signals sync of document potentials and track display

    def _init_metadata_background_collection(self, search_paths):
        # if search paths are provided on the command line,
        self._resource_collector = collector = ResourceSearchPathCollector(self.workspace)
        collector.paths = search_paths or []
        # periodically launch a background scan
        self._resource_collector_timer = timer = QtCore.QTimer()
        self._timer_collect_resources()
        timer.timeout.connect(self._timer_collect_resources)
        timer.start(60000)

    def _init_point_polygon_probes(self):
        self.graphManager = ProbeGraphManager(
            self.ui.probeTabWidget,
            self.ui.autoUpdateCheckbox,
            self.ui.updateButton,
            self.workspace,
            self.layer_model,
            self.queue,
        )
        self.graphManager.didChangeTab.connect(self.scene_manager.show_only_polygons)
        self.graphManager.didClonePolygon.connect(self.scene_manager.copy_polygon)
        self.graphManager.pointProbeChanged.connect(self.scene_manager.on_point_probe_set)
        self.graphManager.pointProbeChanged.connect(self.layer_model.on_point_probe_set)
        self.graphManager.pointProbeChanged.connect(self.update_point_probe_text)

        self.scene_manager.newPointProbe.connect(self.graphManager.update_point_probe)

        self.layer_model.didUpdateLayers.connect(self.graphManager.update_point_probe)
        self.layer_model.didUpdateLayers.connect(self.graphManager.handleActiveProductDatasetsChanged)
        self.layer_model.didChangeRecipeLayerNames.connect(self.graphManager.handleActiveProductDatasetsChanged)

        # Connect to an unnamed slot (lambda: ...) to strip off the argument
        # (of type dict) from the signal 'didMatchTimes'
        self.scene_manager.animation_controller.time_manager.didMatchTimes.connect(
            lambda *args: self.graphManager.update_point_probe()
        )

        def update_probe_polygon(points: list):
            probeable_layers = self.layer_model.get_probeable_layers()
            probeable_layers_uuids = [layer.uuid for layer in probeable_layers]
            LOG.debug("top visible UUID is {0!r:s}".format(probeable_layers_uuids[0]))

            # TODO, when the plots manage their own layer selection, change this call
            # FUTURE, once the polygon is a layer, this will need to change
            # set the selection for the probe plot to the top visible layer(s)
            # new tabs should clone the information from the currently selected tab
            # the call below will check if this is a new polygon
            self.graphManager.set_default_layer_selections(probeable_layers_uuids)

            # update our current plot with the new polygon
            polygon_name = self.graphManager.current_graph_set_region(polygon_points=points)

            # do whatever other updates the scene manager needs
            self.scene_manager.on_new_polygon(polygon_name, points)

            if self.scene_manager._current_tool == Tool.REGION_PROBE:
                self.ui.panZoomToolButton.click()

        self.scene_manager.newProbePolygon.connect(update_probe_polygon)

    def _init_tool_controls(self):
        self.ui.panZoomToolButton.toggled.connect(partial(self.change_tool, name=Tool.PAN_ZOOM))
        self.ui.pointSelectButton.toggled.connect(partial(self.change_tool, name=Tool.POINT_PROBE))
        self.ui.regionSelectButton.toggled.connect(partial(self.change_tool, name=Tool.REGION_PROBE))
        self.change_tool(True)

        def update_full_data_selection():
            # TODO: this slot implementation should be revised, parts of it
            #  should be merged into method provided by SceneManager, then
            #  called from here and in remove_region_polygon() and
            #  update_probe_polygon()

            # Reset graph X layer and Y layer to the two top visible layers,
            # see update_probe_polygon(), copied from there
            probeable_layers = self.layer_model.get_probeable_layers()
            probeable_layers_uuids = [layer.uuid for layer in probeable_layers]
            LOG.debug(f"Probeable layer UUIDs are {probeable_layers_uuids!r:s}")
            # TODO, when the plots manage their own layer selection, change this
            #  call (see update_probe_polygon())
            self.graphManager.set_default_layer_selections(probeable_layers_uuids)

            must_remove_polygon = self.graphManager.current_graph_has_polygon()
            current_graph_name = self.graphManager.current_graph_set_region(select_full_data=True)
            if must_remove_polygon:
                self.scene_manager.remove_polygon(current_graph_name)

            if self.scene_manager._current_tool == Tool.REGION_PROBE:
                self.ui.panZoomToolButton.click()

            if self.scene_manager.has_pending_polygon():
                self.scene_manager.clear_pending_polygon()

        menu = QtWidgets.QMenu(parent=self)
        select_full_data_action = QtWidgets.QAction("Select Full Data", parent=menu)
        select_full_data_action.triggered.connect(update_full_data_selection)
        menu.addAction(select_full_data_action)
        self.ui.regionSelectButton.setMenu(menu)

    def _init_layer_model(self):

        self.layer_model = LayerModel(self.document)

        self.document.didAddDataset.connect(self.layer_model.add_dataset)

        self.layer_model.didCreateLayer.connect(self.scene_manager.add_node_for_layer)
        self.layer_model.didAddImageDataset.connect(self.scene_manager.add_node_for_image_dataset)
        self.layer_model.didAddLinesDataset.connect(self.scene_manager.add_node_for_lines_dataset)
        self.layer_model.didAddPointsDataset.connect(self.scene_manager.add_node_for_points_dataset)

        self.layer_model.didAddSystemLayer.connect(self.scene_manager.add_node_for_system_generated_data)

        self.layer_model.didReorderLayers.connect(self.scene_manager.update_layers_z)

        self.layer_model.didChangeLayerVisible.connect(self.scene_manager.change_layer_visible)
        self.layer_model.didChangeLayerOpacity.connect(self.scene_manager.change_layer_opacity)

        self.layer_model.didChangeColormap.connect(self.scene_manager.change_dataset_nodes_colormap)
        self.layer_model.didChangeGamma.connect(self.scene_manager.change_dataset_nodes_gamma)
        self.layer_model.didChangeColorLimits.connect(self.scene_manager.change_dataset_nodes_color_limits)

        self.scene_manager.animation_controller.connect_to_model(self.layer_model)
        self.layer_model.didActivateProductDataset.connect(self.scene_manager.change_dataset_visible)
        self.layer_model.didAddCompositeDataset.connect(self.scene_manager.add_node_for_composite_dataset)
        self.layer_model.didChangeCompositeProductDataset.connect(self.scene_manager.change_node_for_composite_dataset)
        self.layer_model.didDeleteProductDataset.connect(self.scene_manager.purge_dataset)
        self.layer_model.didRequestSelectionOfLayer.connect(self.ui.treeView.setCurrentIndex)

        self.ui.treeView.setModel(self.layer_model)

        self.layer_model.init_system_layers()

    def _init_algebraic_pane(self):
        self.algebraic_config_pane = AlgebraicLayerConfigPane(
            self.ui, self.ui.algebraicScrollAreaWidget, self.layer_model
        )

        self.ui.treeView.layerSelectionChanged.connect(self.algebraic_config_pane.selection_did_change)
        self.layer_model.didAddImageLayer.connect(self.algebraic_config_pane.layer_added)
        self.algebraic_config_pane.didTriggeredUpdate.connect(self.layer_model.update_recipe_layer_timeline)

    def _init_rgb_pane(self):
        self.rgb_config_pane = RGBLayerConfigPane(self.ui, self.ui.rgbScrollAreaWidget, self.layer_model)
        self.ui.treeView.layerSelectionChanged.connect(self.rgb_config_pane.selection_did_change)
        self.layer_model.didAddImageLayer.connect(self.rgb_config_pane.layer_added)
        self.layer_model.didChangeRecipeLayerNames.connect(self.rgb_config_pane.set_combos_to_layer_names)

    def _init_recipe_manager(self):
        self.recipe_manager = RecipeManager()
        self.layer_model.didRequestCompositeRecipeCreation.connect(self.recipe_manager.create_rgb_recipe)
        self.layer_model.didRequestAlgebraicRecipeCreation.connect(self.recipe_manager.create_algebraic_recipe)
        self.recipe_manager.didCreateRGBCompositeRecipe.connect(self.layer_model.create_rgb_composite_layer)
        self.recipe_manager.didCreateAlgebraicRecipe.connect(self.layer_model.create_algebraic_composite_layer)
        self.rgb_config_pane.didChangeRGBInputLayers.connect(self.recipe_manager.update_rgb_recipe_input_layers)
        self.recipe_manager.didUpdateRGBInputLayers.connect(self.layer_model.update_recipe_layer_timeline)
        self.rgb_config_pane.didChangeRGBColorLimits.connect(self.recipe_manager.update_rgb_recipe_color_limits)
        self.recipe_manager.didUpdateRGBColorLimits.connect(self.layer_model.update_rgb_layer_color_limits)
        self.rgb_config_pane.didChangeRGBGamma.connect(self.recipe_manager.update_rgb_recipe_gammas)
        self.recipe_manager.didUpdateRGBGamma.connect(self.layer_model.update_rgb_layer_gamma)
        self.rgb_config_pane.didChangeRecipeName.connect(self.recipe_manager.update_recipe_name)
        self.algebraic_config_pane.didChangeRecipeName.connect(self.recipe_manager.update_recipe_name)
        self.recipe_manager.didUpdateRecipeName.connect(self.layer_model.update_recipe_layer_name)
        self.algebraic_config_pane.didChangeAlgebraicInputLayers.connect(
            self.recipe_manager.update_algebraic_recipe_input_layers
        )
        self.algebraic_config_pane.didChangeAlgebraicOperationKind.connect(
            self.recipe_manager.update_algebraic_recipe_operation_kind
        )
        self.algebraic_config_pane.didChangeAlgebraicOperationFormula.connect(
            self.recipe_manager.update_algebraic_recipe_operation_formula
        )

    def _init_layer_panes(self):
        # convey action between document and layer list view
        self.layer_info_pane = SingleLayerInfoPane(self.document, parent=self.ui.layerDetailsContents)
        self.layer_list_model = LayerStackTreeViewModel(
            [self.ui.layerListView], self.document, parent=self.ui.layersPaneWidget
        )
        self.layer_list_model.uuidSelectionChanged.connect(self.layer_info_pane.update_display)

    def _init_map_widget(self):
        # connect canvas and projection pieces
        self.ui.mainMapWidget.layout().addWidget(self.scene_manager.main_canvas.native)
        self.ui.projectionComboBox.addItems(tuple(AreaDefinitionsManager.available_area_def_names()))
        self.ui.projectionComboBox.currentIndexChanged.connect(self.document.change_projection_index)
        self.document.didChangeProjection.connect(self.scene_manager.set_projection)

    def _init_qml_timeline(self):
        from uwsift.control.qml_utils import QmlBackend
        from uwsift.ui import QML_PATH

        root_context = self.ui.timelineQuickWidget.engine().rootContext()

        time_manager = self.scene_manager.animation_controller.time_manager
        time_manager.qml_engine = self.ui.timelineQuickWidget.engine()
        time_manager.qml_root_object = self.ui.timelineQuickWidget.rootObject()
        time_manager.qml_backend = QmlBackend()
        time_manager.qml_backend.didJumpInTimeline.connect(self.scene_manager.animation_controller.jump)
        time_manager.qml_backend.didChangeTimebase.connect(time_manager.on_timebase_change)
        # TODO(mk): refactor all QML related objects as belonging to TimeManager's QMLBackend
        #           instance -> communication between TimeManager and QMLBackend via Signal/Slot?
        time_manager.qml_backend.qml_layer_manager = time_manager.qml_layer_manager

        root_context.setContextProperty("LayerManager", time_manager.qml_layer_manager)
        root_context.setContextProperty("timebaseModel", time_manager.qml_timestamps_model)
        root_context.setContextProperty("backend", time_manager.qml_backend)

        self.ui.timelineQuickWidget.setSource(QtCore.QUrl(str(QML_PATH / "timeline.qml")))

    # TODO(mk): replace with method to set all relevant ContextProperties?
    def _get_qml_context(self):
        engine = self.ui.timelineQuickWidget.engine()
        return engine.rootContext()

    def _init_arrange_panes(self):
        self.tabifyDockWidget(self.ui.layersPane, self.ui.areaProbePane)
        self.tabifyDockWidget(self.ui.layerDetailsPane, self.ui.rgbConfigPane)
        self.tabifyDockWidget(self.ui.layerDetailsPane, self.ui.algebraicConfigPane)
        self.layout().removeWidget(self.ui.timelinePane)
        self.ui.timelinePane.deleteLater()
        self.ui.timelinePane = None
        # Make the layer list and layer details shown
        # FIXME remove layerPane finally from the system, for now we only kind
        #  of hide it. This shall be done as part of ticket "Code cleanup (#89)"
        #  (https://gitlab.eumetsat.int/webservices/mtg-sift/-/issues/89)
        self.layout().removeWidget(self.ui.layersPane)

        # FIXME hide layerDetailsPane for now, improve and show it later again
        self.ui.layerDetailsPane.hide()

        # refer to objectName'd entities as self.ui.objectName
        self.setAcceptDrops(True)

    def _init_font_sizes(self):
        # hack some font sizes until we migrate to PyQt5 and handle it better
        # was 14 on osx
        font = QtGui.QFont("Andale Mono")
        font.setPointSizeF(14)
        self.ui.cursorProbeLayer.setFont(font)
        self.ui.cursorProbeText.setFont(font)

    def _init_update_times_display(self):
        self._palette_text_green = QtGui.QPalette()
        self._palette_text_green.setColor(QtGui.QPalette.Text, QtGui.QColor(23, 193, 23))
        self._palette_text_red = QtGui.QPalette()
        self._palette_text_red.setColor(QtGui.QPalette.Text, QtGui.QColor(220, 0, 0))

        self._max_tolerable_idle_time = config.get("watchdog.max_tolerable_idle_time", -1)
        if self._max_tolerable_idle_time <= 0:
            LOG.warning(
                "No valid configuration for"
                " 'watchdog.max_tolerable_idle_time'. Can't highlight"
                " last import time display when delayed."
            )
        else:
            LOG.info(
                f"Highlighting last import time display when delayed for"
                f" more than {self._max_tolerable_idle_time} seconds."
            )

        self._max_tolerable_dataset_age = config.get("watchdog.max_tolerable_dataset_age", -1)
        if self._max_tolerable_dataset_age <= 0:
            LOG.warning(
                "No valid configuration for"
                " 'watchdog.max_tolerable_dataset_age'. Can't highlight"
                " last data time display when delayed."
            )
        else:
            LOG.info(
                f"Highlighting last data time display when delayed for"
                f" more than {self._max_tolerable_dataset_age} seconds."
            )

        # don't clear the time of last import when the layers are removed
        self.document.didRemoveDatasets.connect(self._clear_last_dataset_creation_time)

        self.currentTimeTimer = QtCore.QTimer(parent=self)
        self.currentTimeTimer.timeout.connect(self._update_current_time)
        self.currentTimeTimer.start(250)

        heartbeat_file = config.get("watchdog.heartbeat_file", None)
        if heartbeat_file is None:
            LOG.warning("No configuration for 'watchdog.heartbeat_file'." " Can't send heartbeats to the watchdog.")
        else:
            self.pid = os.getpid()
            self._heartbeat_file = heartbeat_file.replace("$$CACHE_DIR$$", USER_CACHE_DIR)
            LOG.info(f"Communication with watchdog via heartbeat file " f" '{self._heartbeat_file}' configured.")

    def _timer_collect_resources(self):
        if self._resource_collector:
            LOG.debug("launching background resource search")
            self.queue.add(
                "resource_find",
                self._resource_collector.bgnd_look_for_new_files(),
                "look for new or modified files",
                and_then=self._finish_collecting_resources,
                interactive=False,
            )

    def _finish_collecting_resources(self, previous_stage_ok: bool = True):
        ntodo = self._resource_collector.has_pending_files
        if ntodo:
            LOG.debug("{} new resources to collect metadata from".format(ntodo))
            self.queue.add(
                "resource_collect",
                self._resource_collector.bgnd_merge_new_file_metadata_into_mdb(),
                "add metadata for newly found files",
                interactive=False,
            )
        else:
            LOG.debug("no resources to collect, skipping followup task")

    def closeEvent(self, event, *args, **kwargs):
        LOG.debug("main window closing")
        self.workspace.close()

    def open_from_cache(self, *args, **kwargs):
        def _activate_products_for_names(uuids):
            LOG.info("activating cached products with uuids: {}".format(repr(uuids)))
            self.activate_products_by_uuid(uuids)

        def _purge_content_for_names(uuids):
            LOG.info("removing cached products with uuids: {}".format(repr(uuids)))
            self.workspace.purge_content_for_product_uuids(uuids, also_products=False)
            if USE_INVENTORY_DB:
                self.update_recent_file_menu()

        if not self._open_cache_dialog:
            self._open_cache_dialog = OpenCacheDialog(self, _activate_products_for_names, _purge_content_for_names)

        uuid_to_name = self.workspace.product_names_available_in_cache
        ordered_uuids = self.document.sort_product_uuids(uuid_to_name.keys())
        ordered_uuid_to_name = OrderedDict([(u, uuid_to_name[u]) for u in ordered_uuids])
        self._open_cache_dialog.activate(ordered_uuid_to_name)

    def open_wizard(self, *args, **kwargs):
        from uwsift.view.open_file_wizard import OpenFileWizard

        wizard_dialog = OpenFileWizard(base_dir=self._last_open_dir, base_reader=self._last_reader, parent=self)
        self._wizard_dialog = wizard_dialog
        if wizard_dialog.exec_():
            LOG.info("Loading products from open wizard...")
            scenes = wizard_dialog.scenes
            reader = wizard_dialog.get_reader()

            merge_with_existing = config.get("data_reading.merge_with_existing", True)
            if USE_INVENTORY_DB and merge_with_existing:
                # TODO(AR): provide a choice in the wizard for
                #  'merge_with_existing' but only, if caching is off. The latter
                #  condition becomes obsolete, when the CachingWorkspace becomes
                #  able to merge too.
                LOG.error(
                    "Merging new data granules into existing data does not work"
                    " when the caching database is active, i.e. not both"
                    " 'storage.use_inventory_db' and"
                    " 'data_reading.merge_with_existing' can be True."
                    "  Deactivating merging, the caching database wins."
                )
                merge_with_existing = False

            if USE_TILED_GEOLOCATED_IMAGES and merge_with_existing:
                LOG.warning(
                    "Merging of new data segments into existing data"
                    " does not work well with adaptive tiled image"
                    " rendering. Consider switching it of by configuring"
                    " 'display.use_tiled_geolocated_images: False'"
                )

            importer_kwargs = {
                "reader": reader,
                "scenes": scenes,
                "dataset_ids": wizard_dialog.collect_selected_ids(),
                "resampling_info": wizard_dialog.resampling_info,
                "merge_with_existing": merge_with_existing,
            }
            self._last_reader = reader
            self._last_open_dir = wizard_dialog.get_directory()

            self.open_paths(wizard_dialog.files_to_load, **importer_kwargs)
        else:
            LOG.debug("Wizard closed, nothing to load")
        self._wizard_dialog = None

    def reload_config(self):
        config.refresh()

    def remove_region_polygon(self, action: QtWidgets.QAction = None, *args):
        if self.scene_manager._current_tool == Tool.REGION_PROBE:
            self.ui.panZoomToolButton.click()

        if self.scene_manager.has_pending_polygon():
            self.scene_manager.clear_pending_polygon()
            return

        must_remove_polygon = self.graphManager.current_graph_has_polygon()
        removed_name = self.graphManager.current_graph_set_region(None)

        if must_remove_polygon:
            # Remove the polygon from other locations
            LOG.info("Clearing polygon with name '%s'", removed_name)
            self.scene_manager.remove_polygon(removed_name)

    def _init_menu(self):
        open_action = QtWidgets.QAction("&Open...", self)
        open_action.setShortcut("Ctrl+Shift+O")
        open_action.triggered.connect(self.interactive_open_files)

        exit_action = QtWidgets.QAction("&Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(QtWidgets.QApplication.quit)

        if USE_INVENTORY_DB:
            open_cache_action = QtWidgets.QAction("Open from Cache...", self)
            open_cache_action.setShortcut("Ctrl+A")
            open_cache_action.triggered.connect(self.open_from_cache)

        open_wizard_action = QtWidgets.QAction("Open File Wizard...", self)
        open_wizard_action.setShortcuts(["Ctrl+O", "Ctrl+Alt+O"])
        open_wizard_action.triggered.connect(self.open_wizard)

        reload_config_action = QtWidgets.QAction("Reload Configuration", self)
        reload_config_action.setShortcuts(["Ctrl+K", "Ctrl+Alt+K"])
        reload_config_action.triggered.connect(self.reload_config)

        menubar = self.ui.menubar
        file_menu = menubar.addMenu("&File")
        self.addAction(open_action)  # add it to the main window, not the menu (hide it)
        if USE_INVENTORY_DB:
            file_menu.addAction(open_cache_action)
        file_menu.addAction(open_wizard_action)
        if USE_INVENTORY_DB:
            self._recent_files_menu = file_menu.addMenu("Open Recent")

        screenshot_action = QtWidgets.QAction("Export Image", self)
        screenshot_action.setShortcut("Ctrl+I")
        screenshot_action.triggered.connect(self.export_image.take_screenshot)
        file_menu.addAction(screenshot_action)

        file_menu.addSeparator()
        file_menu.addAction(reload_config_action)
        file_menu.addSeparator()

        file_menu.addAction(exit_action)

        next_time = QtWidgets.QAction("Next Time", self)
        next_time.setShortcut(QtCore.Qt.Key_Right)
        next_time.triggered.connect(self.animation.next_frame)

        prev_time = QtWidgets.QAction("Previous Time", self)
        prev_time.setShortcut(QtCore.Qt.Key_Left)
        prev_time.triggered.connect(self.animation.prev_frame)

        toggle_vis = QtWidgets.QAction("Toggle &Visibility", self)
        toggle_vis.setShortcut("V")
        toggle_vis.triggered.connect(self.toggle_visibility_on_selected_layers)

        animate = QtWidgets.QAction("Animate", self)
        animate.setShortcut("A")
        animate.triggered.connect(partial(self.animation.toggle_animation, action=animate))

        cycle_borders = QtWidgets.QAction("Cycle &Borders", self)
        cycle_borders.setShortcut("B")
        cycle_borders.triggered.connect(self.scene_manager.cycle_borders_color)

        cycle_grid = QtWidgets.QAction("Cycle &Lat/Lon Grid", self)
        cycle_grid.setShortcut("L")
        cycle_grid.triggered.connect(self.scene_manager.cycle_latlon_grid_color)

        clear = QtWidgets.QAction("Clear Region Selection", self)
        clear.setShortcut(QtCore.Qt.Key_Escape)
        clear.triggered.connect(self.remove_region_polygon)

        composite = QtWidgets.QAction("Create Composite", self)
        composite.setShortcut("C")
        composite.triggered.connect(self.layer_model.start_rgb_composite_creation)

        algebraic = QtWidgets.QAction("Create Algebraic", self)
        algebraic.triggered.connect(self.layer_model.start_algebraic_composite_creation)

        toggle_point = QtWidgets.QAction("Toggle Point Probe", self)
        toggle_point.setShortcut("X")
        toggle_point.triggered.connect(lambda: self.graphManager.toggle_point_probe(DEFAULT_POINT_PROBE))

        open_gradient = QtWidgets.QAction("Toggle Colormap Editor", self)
        open_gradient.setShortcut("Ctrl+E")
        open_gradient.triggered.connect(self.open_colormap_editor)

        edit_menu = menubar.addMenu("&Edit")
        edit_menu.addAction(clear)
        edit_menu.addAction(toggle_point)
        edit_menu.addAction(open_gradient)

        layer_menu = menubar.addMenu("&Layer")
        layer_menu.addAction(composite)
        layer_menu.addAction(algebraic)

        view_menu = menubar.addMenu("&View")
        view_menu.addAction(animate)
        view_menu.addAction(prev_time)
        view_menu.addAction(next_time)
        view_menu.addAction(toggle_vis)
        view_menu.addAction(cycle_borders)
        view_menu.addAction(cycle_grid)

        if USE_INVENTORY_DB:
            self.update_recent_file_menu()
        menubar.setEnabled(True)

    def _init_key_releases(self):
        def cb_factory(required_key, cb):
            def tmp_cb(key, cb=cb):
                if key.text == required_key:
                    return cb()

            return tmp_cb

        self.scene_manager.main_canvas.events.key_release.connect(cb_factory("t", self.scene_manager.next_tool))

    def open_colormap_editor(self):
        if self._cmap_editor is None:
            self._cmap_editor = ColormapEditor(doc=self.document)
        self._cmap_editor.show()


def set_default_geometry(window, desktop=0):
    screen = QtWidgets.QApplication.desktop()
    screen_geometry = screen.screenGeometry(desktop)
    # TODO: Remove platform specific code
    if "darwin" not in sys.platform:
        w, h = screen_geometry.width() - 400, screen_geometry.height() - 300
        window.setGeometry(200, 150, w, h)
    else:
        size = window.size()
        w, h = size.width(), size.height()
        center = screen_geometry.center()
        screen_x, screen_y = center.x(), center.y()
        window.move(int(screen_x - w / 2.0), int(screen_y - h / 2.0))


def _search_paths(arglist):
    for arg in arglist:
        for subpath in arg.split(":"):
            yield subpath


def create_app() -> (app.Application, QtWidgets.QApplication):
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    vispy_app = app.use_app("pyqt5")
    qt_app = vispy_app.create()
    if hasattr(QtWidgets.QStyleFactory, "AA_UseHighDpiPixmaps"):
        qt_app.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps)
    return vispy_app, qt_app


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run SIFT")
    parser.add_argument("-w", "--workspace-dir", default=WORKSPACE_DB_DIR, help="Specify workspace base directory")
    parser.add_argument("--cache-dir", help="(DEPRECATED: use --workspace-dir) Specify workspace directory")
    parser.add_argument("--clear-workspace", action="store_true", help="Remove workspace contents during start up")
    parser.add_argument("--config-dir", default=DOCUMENT_SETTINGS_DIR, help="Specify config directory")
    parser.add_argument(
        "-s",
        "--space",
        default=256,
        type=int,
        help="Specify max amount of data to hold in workspace cache in Gigabytes",
    )
    parser.add_argument("--border-shapefile", default=None, help="Specify alternative coastline/border shapefile")
    parser.add_argument(
        "--glob-pattern", default=os.environ.get("TIFF_GLOB", None), help="Specify glob pattern for input images"
    )
    parser.add_argument(
        "-p", "--path", dest="paths", action="append", help="directory to search for data [MULTIPLE ALLOWED]"
    )
    parser.add_argument("-c", "--center", nargs=2, type=float, help="Specify center longitude and latitude for camera")
    parser.add_argument(
        "--desktop",
        type=int,
        default=0,
        help="Number of monitor/display to show the main window on (0 for main, 1 for secondary, etc.)",
    )
    parser.add_argument(
        "--profile-heap", type=float, help="take a snapshot of the heap in the given interval (in seconds)"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbosity",
        action="count",
        default=int(os.environ.get("VERBOSITY", 2)),
        help="each occurrence increases verbosity 1 level through " "ERROR-WARNING-Info-DEBUG (default Info)",
    )
    args = parser.parse_args()

    if args.profile_heap:
        heap_profiler = HeapProfiler(args.profile_heap)
        heap_profiler.start()

    check_grib_definition_dir()
    check_imageio_deps()

    if args.cache_dir:
        LOG.warning("'--cache-dir' is deprecated, use '--workspace-dir'")
        args.workspace_dir = args.cache_dir

    LOG.info("Using configuration directory: %s", args.config_dir)
    LOG.info("Using cache directory: %s", args.cache_dir)
    vispy_app, qt_app = create_app()

    # Add our own fonts to Qt windowing system
    font_pattern = os.path.join(get_package_data_dir(), "fonts", "*")
    for fn in glob(font_pattern):
        QtGui.QFontDatabase.addApplicationFont(fn)

    data_search_paths = [] if not args.paths else list(_search_paths(args.paths))
    LOG.info("will search {} for new data periodically".format(repr(data_search_paths)))

    window = Main(
        workspace_dir=args.workspace_dir,
        config_dir=args.config_dir,
        cache_size=args.space,
        glob_pattern=args.glob_pattern,
        search_paths=data_search_paths,
        border_shapefile=args.border_shapefile,
        center=args.center,
        clear_workspace=args.clear_workspace,
    )

    set_default_geometry(window, desktop=args.desktop)
    window.show()
    # bring window to front
    window.raise_()

    if AUTO_UPDATE_MODE__ACTIVE:

        # FIXME: let the AutoUpdateManager be in control...
        from uwsift import config
        from uwsift.control.auto_update import AutoUpdateManager

        minimum_interval = config.get("auto_update.interval", None)
        if minimum_interval is None:
            raise ValueError("Auto update interval needs to be set!")
        auto_update_manager = AutoUpdateManager(window, minimum_interval)
        # connect signal to start timer anew when loading is done
        window.didFinishLoading.connect(auto_update_manager.on_loading_done)

    # run the event loop until the user closes the application
    exit_code = vispy_app.run()
    # Workaround PyCharm issue: The PyCharm dev console raises a TypeError if
    # None is passed to 'sys.exit()'. Thus replace None by 0, both represent
    # success for 'sys.exit()'.
    if exit_code is not None:
        return exit_code
    return 0


if __name__ == "__main__":
    sys.exit(main())
