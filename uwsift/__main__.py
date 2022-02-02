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

import logging
import os
import sys
from collections import OrderedDict
from functools import partial
from glob import glob
import typing as typ
from uuid import UUID

from PyQt5 import QtCore, QtGui, QtWidgets
from vispy import app

import uwsift.ui.open_cache_dialog_ui as open_cache_dialog_ui
from uwsift import __version__
from uwsift.common import Info, Tool, CompositeType
from uwsift.control.doc_ws_as_timeline_scene import SiftDocumentAsFramesInTracks
from uwsift.control.layer_tree import LayerStackTreeViewModel
from uwsift.control.rgb_behaviors import UserModifiesRGBLayers
from uwsift.model.document import Document
from uwsift.model.layer import DocRGBLayer
from uwsift.queue import TaskQueue, TASK_PROGRESS, TASK_DOING
# this is generated with pyuic4 pov_main.ui >pov_main_ui.py
from uwsift.ui.pov_main_ui import Ui_MainWindow
from uwsift.util import (WORKSPACE_DB_DIR, DOCUMENT_SETTINGS_DIR,
                         get_package_data_dir, check_grib_definition_dir, check_imageio_deps)
from uwsift.view.colormap_editor import ColormapEditor
from uwsift.view.create_algebraic import CreateAlgebraicDialog
from uwsift.view.export_image import ExportImageHelper
from uwsift.view.layer_details import SingleLayerInfoPane
from uwsift.view.probes import ProbeGraphManager, DEFAULT_POINT_PROBE
from uwsift.view.rgb_config import RGBLayerConfigPane
from uwsift.view.scene_graph import SceneGraphManager
from uwsift.workspace import Workspace
from uwsift.workspace.collector import ResourceSearchPathCollector

LOG = logging.getLogger(__name__)
PROGRESS_BAR_MAX = 1000
STATUS_BAR_DURATION = 2000  # ms


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
        # sorted_items = sorted(uuid_to_name.items(),
        #                       key=lambda x: x[1])
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
        to_open = [item.data(QtCore.Qt.UserRole)
                   for item in self.ui.cacheListWidget.selectedItems()]
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
        self.setToolTip('Set animation speed')
        self._slider = QtWidgets.QSlider(parent=self)
        # n, x = self._convert(10, reverse=True), self._convert(5000, reverse=True)
        n, x = 2, 150  # frames per 10 seconds
        self._slider.setRange(n, x)  #
        # self._slider.setSingleStep(1)
        # self._slider.setInvertedAppearance(True)
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
        self.setToolTip('{0:.1f} fps'.format(fps))
        val = self._convert(value)
        self._slot(val)

    def show_at(self, pos, val):
        from PyQt5.QtCore import QRect, QPoint, QSize
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
    LOG.debug('looking for common path prefix for {}'.format(repr(paths)))
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

    def __init__(self, ui,
                 scene_manager: SceneGraphManager,
                 document: Document,
                 layer_list_model: LayerStackTreeViewModel
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

        self.scene_manager.didChangeFrame.connect(self.update_frame_slider)
        self.ui.animPlayPause.clicked.connect(self.toggle_animation)
        self.ui.animPlayPause.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.ui.animPlayPause.customContextMenuRequested.connect(self.show_animation_speed_slider)

        self.ui.animForward.clicked.connect(self.next_frame)
        self.ui.animBack.clicked.connect(self.prev_frame)

        # allow animation slider to set animation frame being displayed:
        self.ui.animationSlider.valueChanged.connect(self.animation_slider_jump_frame)

        # allow animation, once stopped, to propagate visibility to the document and layerlist:
        self.scene_manager.didChangeLayerVisibility.connect(self.document.animation_changed_visibility)

        self.document.didSwitchLayerSet.connect(self.animation_reset_by_layer_set_switch)
        self.document.didChangeLayerVisibility.connect(self.update_frame_time_to_top_visible)
        self.document.didReorderLayers.connect(self.update_frame_time_to_top_visible)
        self.document.didRemoveLayers.connect(self.update_frame_time_to_top_visible)
        self.document.didAddBasicLayer.connect(self.update_frame_time_to_top_visible)
        self.document.didAddCompositeLayer.connect(self.update_frame_time_to_top_visible)

    def next_frame(self, *args, **kwargs):
        """Advance a frame along the animation order."""
        self.scene_manager.layer_set.animating = False
        self.scene_manager.layer_set.next_frame()

    def prev_frame(self, *args, **kwargs):
        """Retreat a frame along the animation list."""
        self.scene_manager.layer_set.animating = False
        self.scene_manager.layer_set.next_frame(frame_number=-1)

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
        # LOG.debug('did update animation slider {} {}'.format(frame_index, frame_count))
        self.ui.animPlayPause.setDown(animating)
        self.ui.animationSlider.repaint()
        if animating:
            self.ui.animationLabel.setText(self.document.time_label_for_uuid(uuid))
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

    def _next_last_time_visibility(self, direction=0, *args, **kwargs):
        LOG.info('time incr {}'.format(direction))
        # TODO: if this frame is part of the animation sequence, update the slider as well!
        uuids = self.layer_list_model.current_selected_uuids()
        if not uuids:
            self.ui.statusbar.showMessage('ERROR: No layer selected', STATUS_BAR_DURATION)
        new_focus = None
        for uuid in uuids:
            new_focus = self.document.next_last_step(uuid, direction, bandwise=False)
        return new_focus

    def update_slider_if_frame_is_in_animation(self, uuid, **kwargs):
        """Update frame slider to the specified frame UUID, but only if it's part of animation order."""
        # FUTURE: this could be a cheaper operation but it's probably fine since it's input-driven
        cao = self.document.current_animation_order
        try:
            dex = cao.index(uuid)
        except ValueError:
            return
        frame_change_tuple = (dex, len(cao), False, uuid)
        self.update_frame_slider(frame_change_tuple)

    def next_last_time(self, direction=0, *args, **kwargs):
        """Move forward (direction=+1) or backward (-1) a time step in animation order."""
        self.scene_manager.layer_set.animating = False
        new_focus = self._next_last_time_visibility(direction=direction)
        self.layer_list_model.select([new_focus])
        # if this part of the animation cycle, update the animation slider and displayed time as well
        self.update_slider_if_frame_is_in_animation(new_focus)
        return new_focus

    def next_last_band(self, direction=0, *args, **kwargs):
        """Move forward (direction=+1) or backward (-1) a band step in animation order."""
        LOG.info('band incr {}'.format(direction))
        uuids = self.layer_list_model.current_selected_uuids()
        new_focus = None
        if not uuids:
            pass  # FIXME: notify user
        for uuid in uuids:
            new_focus = self.document.next_last_step(uuid, direction, bandwise=True)
        if new_focus is not None:
            self.layer_list_model.select([new_focus])
            self.update_frame_time_to_top_visible()
            self.update_slider_if_frame_is_in_animation(new_focus)

    def set_animation_speed(self, milliseconds):
        """Change frame rate as measured in milliseconds."""
        LOG.info('animation speed set to {}ms'.format(milliseconds))
        self.scene_manager.layer_set.animation_speed = milliseconds

    def show_animation_speed_slider(self, pos: QtCore.QPoint, *args):
        """Show frame-rate slider as a pop-up control, at current mouse position."""
        LOG.info('menu requested for animation control')
        gpos = self.ui.animPlayPause.mapToGlobal(pos)

        if self._animation_speed_popup is None:
            self._animation_speed_popup = popup = AnimationSpeedPopupWindow(slot=self.set_animation_speed, parent=None)
        else:
            popup = self._animation_speed_popup
        if not popup.isVisible():
            popup.show_at(gpos, self.scene_manager.layer_set.animation_speed)

    def animation_reset_by_layer_set_switch(self, *args, **kwargs):
        """Perform necessary control resets when document layer set is swapped."""
        self.reset_frame_slider()
        self.update_frame_time_to_top_visible()

    def change_animation_to_current_selection_siblings(self, *args, **kwargs):
        """Assign new animation order based on selection."""
        uuid = self._next_last_time_visibility(direction=0)
        if uuid is None:
            self.ui.statusbar.showMessage("ERROR: No layer selected", STATUS_BAR_DURATION)
            return
        # calculate the new animation sequence by consulting the guidebook
        uuids = self.document.animate_siblings_of_layer(uuid)
        if uuids:
            self.ui.statusbar.showMessage("Info: Frame order updated", STATUS_BAR_DURATION)
            self.layer_list_model.select(uuids)
        else:
            self.ui.statusbar.showMessage("ERROR: Layer with time steps or band siblings needed", STATUS_BAR_DURATION)
        LOG.info('using siblings of {} for animation loop'.format(uuids[0] if uuids else '-unknown-'))

    def toggle_animation(self, action: QtWidgets.QAction = None, *args):
        """Toggle animation on/off."""
        new_state = self.scene_manager.layer_set.toggle_animation()
        self.ui.animPlayPause.setChecked(new_state)


class Main(QtWidgets.QMainWindow):
    _last_open_dir: str = None  # directory to open files in
    _recent_files_menu: QtWidgets.QMenu = None  # QMenu
    _open_cache_dialog: QtWidgets.QDialog = None
    _screenshot_dialog: QtWidgets.QDialog = None
    _cmap_editor = None  # Gradient editor widget
    _resource_collector: ResourceSearchPathCollector = None
    _resource_collector_timer: QtCore.QTimer = None
    _timeline_scene: SiftDocumentAsFramesInTracks = None

    def interactive_open_files(self, *args, files=None, **kwargs):
        self.scene_manager.layer_set.animating = False
        # http://pyqt.sourceforge.net/Docs/PyQt4/qfiledialog.html#getOpenFileNames
        filename_filters = [
            # 'All files (*.*)',
            'All supported files (*.nc *.nc4)',
            'GOES-16 NetCDF (*.nc *.nc4)',
        ]
        filter_str = ';;'.join(filename_filters)
        files = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Select one or more files to open", self._last_open_dir or os.getenv("HOME"), filter_str)[0]
        self.open_paths(files)

    def _bgnd_open_paths(self, paths, uuid_list, **importer_kwargs):
        """Background task runs on a secondary thread
        """
        LOG.info("opening products from {} paths in background".format(
            len(paths)))
        for progress in self.document.import_files(paths, **importer_kwargs):
            yield progress
            uuid_list.append(progress['uuid'])
        yield {TASK_DOING: 'products loaded from paths', TASK_PROGRESS: 1.0}

    def _bgnd_open_paths_finish(self, isok: bool, uuid_list: typ.List[UUID]):
        """Main thread finalization after background imports are done.

        Args:
            isok: whether _bgnd_open_paths ran without exception
            uuid_list: list of UUIDs it generated
        """
        if not uuid_list:
            raise ValueError("no UUIDs provided by background open in _bgnd_open_paths_when_done")
        if not isok:
            raise ValueError("background open did not succeed")
        uuid = uuid_list[-1]
        self.layer_list_model.select([uuid])
        # set the animation based on the last added (topmost) layer
        self.document.animate_siblings_of_layer(uuid)
        # force the newest layer to be visible
        self.document.next_last_step(uuid)

    def open_paths(self, paths, **importer_kwargs):
        paths = list(paths)
        if not paths:
            return
        uli = []
        bop = partial(self._bgnd_open_paths, uuid_list=uli, **importer_kwargs)
        bopf = partial(self._bgnd_open_paths_finish, uuid_list=uli)
        self.queue.add("load_files", bop(paths), "Open {} files".format(len(paths)), and_then=bopf, interactive=False)
        # don't use <algebraic layer ...> type paths
        self._last_open_dir = _common_path_prefix([x for x in paths if x[0] != '<']) or self._last_open_dir
        self.update_recent_file_menu()

    def activate_products_by_uuid(self, uuids):
        uuids = list(uuids)
        if not uuids:
            return
        for uuid in uuids:
            self.document.activate_product_uuid_as_new_layer(uuid)
        uuid = uuids[-1]
        self.layer_list_model.select([uuid])
        # set the animation based on the last added (topmost) layer
        self.document.animate_siblings_of_layer(uuid)
        # force the newest layer to be visible
        self.document.next_last_step(uuid)
        # don't use <algebraic layer ...> type paths

    def dropEvent(self, event):
        LOG.debug('drop event on mainwindow')
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
        LOG.debug('recent uuids: {}'.format(repr(uuid_to_name.keys())))
        self._recent_files_menu.clear()
        for uuid, p_name in uuid_to_name.items():
            def openit(checked=False, uuid=uuid):
                LOG.debug('open recent product {}'.format(uuid))
                self.scene_manager.layer_set.animating = False
                self.activate_products_by_uuid([uuid])

            open_action = QtWidgets.QAction(p_name, self)
            open_action.triggered.connect(openit)
            self._recent_files_menu.addAction(open_action)

    def update_progress_bar(self, status_info, *args, **kwargs):
        active = status_info[0] if len(status_info) > 0 else None
        # LOG.debug('{0!r:s}'.format(status_info))
        val = active[TASK_PROGRESS] if active else 0.0
        txt = active[TASK_DOING] if active else ''
        val = self.queue.progress_ratio(val)
        self.ui.progressBar.setValue(int(val * PROGRESS_BAR_MAX))
        self.ui.progressText.setText(txt)
        # LOG.warning('progress bar updated to {}'.format(val))

    def toggle_visibility_on_selected_layers(self, *args, **kwargs):
        uuids = self.layer_list_model.current_selected_uuids()
        self.document.toggle_layer_visibility(uuids)
        self.animation.update_frame_time_to_top_visible()

    def remove_layer(self, *args, **kwargs):
        uuids = self.layer_list_model.current_selected_uuids()
        rgb_uuids_handled = set()
        uuids_to_remove = set()
        # if we are deleting an RGB layer then we have to remove all of them
        uuids = list(uuids)
        for uuid in uuids:
            layer = self.document[uuid]
            if not isinstance(layer, DocRGBLayer):
                uuids_to_remove.add(uuid)
                continue
            elif uuid in rgb_uuids_handled:
                continue

            rgbs_uuids = self.document.family_uuids_for_uuid(uuid, active_only=True)
            all_rgbs_uuids = self.document.family_uuids_for_uuid(uuid)
            if all(l_uuid in uuids for l_uuid in rgbs_uuids):
                # there is only one of these RGBs so just remove it
                # or they have selected all of the layers in this family
                rgb_uuids_handled.update(all_rgbs_uuids)
                uuids_to_remove.update(all_rgbs_uuids)
                continue

            # Ask the user if this is what they want
            msg_box = QtWidgets.QMessageBox()
            msg_box.setText("Deleting RGB layer, delete all times for this RGB?")
            msg_box.setInformativeText("All related RGBs must also be deleted.")
            msg_box.setStandardButtons(msg_box.Yes | msg_box.No)
            msg_box.setDefaultButton(msg_box.No)
            response = msg_box.exec_()
            if response == msg_box.Yes:
                LOG.debug("Setting all RGB family UUIDs to be removed: %s", uuid)
                rgb_uuids_handled.update(all_rgbs_uuids)
                uuids_to_remove.update(all_rgbs_uuids)
            else:
                LOG.debug("Will not delete RGB or its family: %s", uuid)
                rgb_uuids_handled.update(all_rgbs_uuids)
                continue

        if uuids_to_remove:
            self.document.remove_layers_from_all_sets(uuids_to_remove)

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
            lon = lon % 360 if lon > 0 else lon % -360 + 360
            lon = lon - 360 if lon > 180 else lon
            lon_str = "{:>6.02f} {}".format(abs(lon), "W" if lon < 0 else "E")
            lat_str = "{:>6.02f} {}".format(abs(lat), "S" if lat < 0 else "N")
            probe_loc = "{}, {}".format(lon_str, lat_str)
        else:
            probe_loc = "{:>6s}  , {:>6s}  ".format("N/A", "N/A")

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
                layer_str = "N/A"
            else:
                info = self.document[uuid]
                unit_info = info[Info.UNIT_CONVERSION]
                data_point = unit_info[1](data_point)
                data_str = unit_info[2](data_point, numeric=False)
                if info.get(Info.CENTRAL_WAVELENGTH):
                    wl = info[Info.CENTRAL_WAVELENGTH]
                    if wl < 4.1:
                        wl_str = "{:0.02f} µm".format(wl)
                    else:
                        wl_str = "{:0.01f} µm".format(wl)
                    layer_str = "{}, {}".format(info[Info.SHORT_NAME],
                                                wl_str)
                else:
                    layer_str = info[Info.SHORT_NAME]
        else:
            data_str = "N/A"
            layer_str = "N/A"
        self.ui.cursorProbeLayer.setText(layer_str)
        self.ui.cursorProbeText.setText("{} ({})".format(data_str, probe_loc))

    def _init_timeline(self, doc: Document, ws: Workspace):
        gv = self.ui.timelineView

        # set up the widget itself
        gv.setViewportUpdateMode(QtWidgets.QGraphicsView.FullViewportUpdate)
        gv.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        gv.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        # gv.setRenderHints(QtGui.QPainter.Antialiasing)

        # connect up the scene
        doc.sync_potential_tracks_from_metadata()
        LOG.debug("Potential tracks: {}".format(repr(doc.track_order)))
        self._timeline_scene = SiftDocumentAsFramesInTracks(doc, self.workspace)
        gv.setScene(self._timeline_scene)
        QtWidgets.QApplication.instance().aboutToQuit.connect(self._timeline_scene.clear)

        self._timeline_scene.sync_items()

        def center_timeline_view_on_single_frame(frame_uuids, gv=gv, timeline_scene=self._timeline_scene):
            frame_uuids = list(frame_uuids) if not isinstance(frame_uuids, list) else frame_uuids
            # FIXME: again, we're assuming product id = frame id = layer id
            if len(frame_uuids) == 1 and gv.isVisible():
                LOG.debug('centering timeline view on single selected frame')

                timeline_scene.center_view_on_frame(gv, frame_uuids[0])

        self.layer_list_model.uuidSelectionChanged.connect(center_timeline_view_on_single_frame)

    def __init__(self, config_dir=None, workspace_dir=None, cache_size=None, glob_pattern=None, search_paths=None,
                 border_shapefile=None, center=None, clear_workspace=False):
        super(Main, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        # FIXME: Slider does not currently work as intended. Re-enable later
        self.ui.timelineScaleSlider.setDisabled(True)

        self._init_font_sizes()

        self.setWindowTitle(self.windowTitle().replace("|X.X.X|", __version__))
        self._init_arrange_panes()

        self.queue = TaskQueue()
        self.ui.progressBar.setRange(0, PROGRESS_BAR_MAX)
        self.queue.didMakeProgress.connect(self.update_progress_bar)

        # create manager and helper classes
        self.workspace = Workspace(workspace_dir, max_size_gb=cache_size, queue=self.queue,
                                   initial_clear=clear_workspace)
        self.document = doc = Document(self.workspace, config_dir=config_dir, queue=self.queue)
        self.scene_manager = SceneGraphManager(doc, self.workspace, self.queue,
                                               border_shapefile=border_shapefile,
                                               center=center,
                                               parent=self)
        self.export_image = ExportImageHelper(self, self.document, self.scene_manager)
        self._wizard_dialog = None

        self._init_layer_panes()
        self._init_rgb_pane()
        self._init_map_widget()

        self.animation = UserControlsAnimation(self.ui,
                                               self.scene_manager,
                                               self.document,
                                               self.layer_list_model
                                               )

        # disable close button on panes
        panes = [self.ui.areaProbePane, self.ui.layersPane, self.ui.layerDetailsPane, self.ui.rgbConfigPane]
        for pane in panes:
            pane.setFeatures(QtWidgets.QDockWidget.DockWidgetFloatable |
                             QtWidgets.QDockWidget.DockWidgetMovable)
        # Make the panes on the right side 375px wide
        self.resizeDocks(panes, [375] * len(panes), QtCore.Qt.Horizontal)

        test_layers(self.document, glob_pattern=glob_pattern)

        # quamash async test pattern updates a control once a second
        # loop.create_task(do_test_cycle(self.ui.cursorProbeText))

        # Interaction Setup
        self._init_key_releases()

        self.scheduler = QtCore.QTimer(parent=self)
        self.scheduler.setInterval(200)
        self.scheduler.timeout.connect(partial(self.scene_manager.on_view_change, self.scheduler))

        def start_wrapper(timer, event):
            """Simple wrapper around a timers start method so we can accept but ignore the event provided
            """
            timer.start()

        self.scene_manager.main_view.scene.transform.changed.connect(partial(start_wrapper, self.scheduler))

        print(self.scene_manager.main_view.describe_tree(with_transform=True))

        self._init_tool_controls()
        self._init_menu()
        self._init_point_polygon_probes()

        # Set the projection based on the document's default
        self.document.change_projection()
        self.ui.projectionComboBox.setCurrentIndex(self.document.current_projection_index())

        self._init_metadata_background_collection(search_paths)

        # set up timeline
        # LOG.info("potential tracks already in database: {}".format(repr(doc.potential_tracks())))
        # self._init_timeline(doc, self.workspace)

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
        self.graphManager = ProbeGraphManager(self.ui.probeTabWidget, self.workspace, self.document, self.queue)
        self.graphManager.didChangeTab.connect(self.scene_manager.show_only_polygons)
        self.graphManager.didClonePolygon.connect(self.scene_manager.copy_polygon)
        self.graphManager.pointProbeChanged.connect(self.scene_manager.on_point_probe_set)
        self.graphManager.pointProbeChanged.connect(self.document.update_equalizer_values)
        self.graphManager.pointProbeChanged.connect(self.update_point_probe_text)
        self.graphManager.pointProbeChanged.connect(self.graphManager.update_point_probe_graph)

        self.scene_manager.newPointProbe.connect(self.graphManager.update_point_probe)

        def _update_point_probe_slot(*args):
            return self.graphManager.update_point_probe(DEFAULT_POINT_PROBE)
        self.document.didAddBasicLayer.connect(_update_point_probe_slot)
        self.document.didAddCompositeLayer.connect(_update_point_probe_slot)

        # FIXME: These were added as a simple fix to update the probe value on layer changes, but this should really
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
            self.scene_manager.didChangeFrame.connect(
                lambda frame_info: self.ui.cursorProbeText.setText("Probe Value: <animating>"))

        def update_probe_polygon(uuid, points, layerlist=self.layer_list_model):
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

            if self.scene_manager._current_tool == Tool.REGION_PROBE:
                self.ui.panZoomToolButton.click()

        self.scene_manager.newProbePolygon.connect(update_probe_polygon)
        # setup RGB configuration
        self.document.didChangeComposition.connect(lambda *args: self._refresh_probe_results(*args[1:]))
        self.document.didChangeColorLimits.connect(self._refresh_probe_results)

    def _init_tool_controls(self):
        self.ui.panZoomToolButton.toggled.connect(partial(self.change_tool, name=Tool.PAN_ZOOM))
        self.ui.pointSelectButton.toggled.connect(partial(self.change_tool, name=Tool.POINT_PROBE))
        self.ui.regionSelectButton.toggled.connect(partial(self.change_tool, name=Tool.REGION_PROBE))
        self.change_tool(True)

    def _init_rgb_pane(self):
        self.rgb_config_pane = RGBLayerConfigPane(self.ui, self.ui.layersPaneWidget)
        self.user_rgb_behavior = UserModifiesRGBLayers(self.document,
                                                       self.rgb_config_pane,
                                                       self.layer_list_model,
                                                       parent=self)

    def _init_layer_panes(self):
        # convey action between document and layer list view
        self.layer_info_pane = SingleLayerInfoPane(self.document, parent=self.ui.layerDetailsContents)
        self.layer_list_model = LayerStackTreeViewModel([self.ui.layerListView], self.document,
                                                        parent=self.ui.layersPaneWidget)
        self.layer_list_model.uuidSelectionChanged.connect(self.layer_info_pane.update_display)

    def _init_map_widget(self):
        # connect canvas and projection pieces
        self.ui.mainMapWidget.layout().addWidget(self.scene_manager.main_canvas.native)
        self.ui.projectionComboBox.addItems(tuple(self.document.available_projections.keys()))
        self.ui.projectionComboBox.currentIndexChanged.connect(self.document.change_projection_index)
        self.document.didChangeProjection.connect(self.scene_manager.set_projection)

    def _init_arrange_panes(self):
        self.tabifyDockWidget(self.ui.layersPane, self.ui.areaProbePane)
        self.tabifyDockWidget(self.ui.layerDetailsPane, self.ui.rgbConfigPane)
        # self.tabifyDockWidget(self.ui.layerDetailsPane, self.ui.timelinePane)
        self.layout().removeWidget(self.ui.timelinePane)
        self.ui.timelinePane.deleteLater()
        self.ui.timelinePane = None
        # self.tabifyDockWidget(self.ui.rgbConfigPane, self.ui.layerDetailsPane)
        # Make the layer list and layer details shown
        self.ui.layersPane.raise_()
        self.ui.layerDetailsPane.raise_()
        # refer to objectName'd entities as self.ui.objectName
        self.setAcceptDrops(True)

    def _init_font_sizes(self):
        # hack some font sizes until we migrate to PyQt5 and handle it better
        # was 14 on osx
        font = QtGui.QFont('Andale Mono')
        font.setPointSizeF(14)
        self.ui.cursorProbeLayer.setFont(font)
        self.ui.cursorProbeText.setFont(font)

    def _timer_collect_resources(self):
        if self._resource_collector:
            LOG.debug("launching background resource search")
            self.queue.add('resource_find', self._resource_collector.bgnd_look_for_new_files(),
                           "look for new or modified files",
                           and_then=self._finish_collecting_resources, interactive=False)

    def _finish_collecting_resources(self, previous_stage_ok: bool = True):
        ntodo = self._resource_collector.has_pending_files
        if ntodo:
            LOG.debug("{} new resources to collect metadata from".format(ntodo))
            self.queue.add("resource_collect", self._resource_collector.bgnd_merge_new_file_metadata_into_mdb(),
                           "add metadata for newly found files", interactive=False)
        else:
            LOG.debug("no resources to collect, skipping followup task")

    def closeEvent(self, event, *args, **kwargs):
        LOG.debug('main window closing')
        self.workspace.close()

    def _remove_paths_from_cache(self, paths):
        self.workspace.remove_all_workspace_content_for_resource_paths(paths)
        self.update_recent_file_menu()

    def open_from_cache(self, *args, **kwargs):
        def _activate_products_for_names(uuids):
            LOG.info('activating cached products with uuids: {}'.format(repr(uuids)))
            self.activate_products_by_uuid(uuids)

        def _purge_content_for_names(uuids):
            LOG.info('removing cached products with uuids: {}'.format(repr(uuids)))
            self.workspace.purge_content_for_product_uuids(uuids, also_products=False)
            self.update_recent_file_menu()

        if not self._open_cache_dialog:
            self._open_cache_dialog = OpenCacheDialog(self,
                                                      _activate_products_for_names,
                                                      _purge_content_for_names)

        uuid_to_name = self.workspace.product_names_available_in_cache
        ordered_uuids = self.document.sort_product_uuids(uuid_to_name.keys())
        ordered_uuid_to_name = OrderedDict([(u, uuid_to_name[u]) for u in ordered_uuids])
        self._open_cache_dialog.activate(ordered_uuid_to_name)

    # def open_glob(self, *args, **kwargs):
    #     text, ok = QtWidgets.QInputDialog.getText(self, 'Open Glob Pattern', 'Open files matching pattern:')
    #     from glob import glob
    #     if ok:
    #         paths = list(glob(text))
    #         self.open_paths(paths)

    def open_wizard(self, *args, **kwargs):
        from uwsift.view.open_file_wizard import OpenFileWizard
        wizard_dialog = OpenFileWizard(base_dir=self._last_open_dir, parent=self)
        self._wizard_dialog = wizard_dialog
        if wizard_dialog.exec_():
            LOG.info("Loading products from open wizard...")
            scenes = wizard_dialog.scenes
            reader = wizard_dialog.previous_reader
            importer_kwargs = {
                'reader': reader,
                'scenes': scenes,
                'dataset_ids': wizard_dialog.collect_selected_ids(),
            }
            self._last_open_dir = wizard_dialog.last_open_dir
            self.open_paths(wizard_dialog.files_to_load,
                            **importer_kwargs)
        else:
            LOG.debug("Wizard closed, nothing to load")
        self._wizard_dialog = None

    def remove_region_polygon(self, action: QtWidgets.QAction = None, *args):
        if self.scene_manager.has_pending_polygon():
            self.scene_manager.clear_pending_polygon()
            return

        # Remove the polygon from other locations
        removed_name = self.graphManager.currentPolygonChanged(None)
        LOG.info("Clearing polygon with name '%s'", removed_name)
        self.scene_manager.remove_polygon(removed_name)

    def create_algebraic(self, action: QtWidgets.QAction = None, uuids=None, composite_type=CompositeType.ARITHMETIC):
        if uuids is None:
            uuids = list(self.layer_list_model.current_selected_uuids())
        dialog = CreateAlgebraicDialog(self.document, uuids, parent=self)
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _init_menu(self):
        open_action = QtWidgets.QAction("&Open...", self)
        open_action.setShortcut("Ctrl+Shift+O")
        open_action.triggered.connect(self.interactive_open_files)

        exit_action = QtWidgets.QAction("&Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(QtWidgets.QApplication.quit)

        open_cache_action = QtWidgets.QAction("Open from Cache...", self)
        open_cache_action.setShortcut("Ctrl+A")
        open_cache_action.triggered.connect(self.open_from_cache)

        # open_glob_action = QtWidgets.QAction("Open Filename Pattern...", self)
        # open_glob_action.setShortcut("Ctrl+Shift+O")
        # open_glob_action.triggered.connect(self.open_glob)

        open_wizard_action = QtWidgets.QAction("Open File Wizard...", self)
        open_wizard_action.setShortcuts(["Ctrl+O", "Ctrl+Alt+O"])
        open_wizard_action.triggered.connect(self.open_wizard)

        menubar = self.ui.menubar
        file_menu = menubar.addMenu('&File')
        self.addAction(open_action)  # add it to the main window, not the menu (hide it)
        file_menu.addAction(open_cache_action)
        # file_menu.addAction(open_glob_action)
        file_menu.addAction(open_wizard_action)
        self._recent_files_menu = file_menu.addMenu('Open Recent')

        screenshot_action = QtWidgets.QAction("Export Image", self)
        screenshot_action.setShortcut("Ctrl+I")
        screenshot_action.triggered.connect(self.export_image.take_screenshot)
        file_menu.addAction(screenshot_action)

        file_menu.addAction(exit_action)

        next_time = QtWidgets.QAction("Next Time", self)
        next_time.setShortcut(QtCore.Qt.Key_Right)
        next_slot = partial(self.animation.next_last_time, direction=1)
        next_time.triggered.connect(next_slot)
        # self.ui.animForward.clicked.connect(next_slot)

        focus_current = QtWidgets.QAction("Focus Current Timestep", self)
        focus_current.setShortcut('.')
        focus_current.triggered.connect(partial(self.animation.next_last_band, direction=0))

        prev_time = QtWidgets.QAction("Previous Time", self)
        prev_time.setShortcut(QtCore.Qt.Key_Left)
        prev_slot = partial(self.animation.next_last_time, direction=-1)
        prev_time.triggered.connect(prev_slot)
        # self.ui.animBack.clicked.connect(prev_slot)

        focus_prev_band = QtWidgets.QAction("Next Band", self)
        focus_prev_band.setShortcut(QtCore.Qt.Key_Up)
        focus_prev_band.triggered.connect(partial(self.animation.next_last_band, direction=-1))

        focus_next_band = QtWidgets.QAction("Previous Band", self)
        focus_next_band.setShortcut(QtCore.Qt.Key_Down)
        focus_next_band.triggered.connect(partial(self.animation.next_last_band, direction=1))

        toggle_vis = QtWidgets.QAction("Toggle &Visibility", self)
        toggle_vis.setShortcut('V')
        toggle_vis.triggered.connect(self.toggle_visibility_on_selected_layers)

        animate = QtWidgets.QAction("Animate", self)
        animate.setShortcut('A')
        animate.triggered.connect(partial(self.animation.toggle_animation, action=animate))

        change_order = QtWidgets.QAction("Set Animation &Order", self)
        change_order.setShortcut('O')
        change_order.triggered.connect(self.animation.change_animation_to_current_selection_siblings)

        flip_colormap = QtWidgets.QAction("Flip Color Limits (Top Layer)", self)
        flip_colormap.setShortcut("/")
        flip_colormap.triggered.connect(
            lambda: self.document.flip_climits_for_layers([self.document.current_visible_layer_uuid]))

        cycle_borders = QtWidgets.QAction("Cycle &Borders", self)
        cycle_borders.setShortcut('B')
        cycle_borders.triggered.connect(self.scene_manager.cycle_borders_color)

        cycle_grid = QtWidgets.QAction("Cycle &Lat/Lon Grid", self)
        cycle_grid.setShortcut('L')
        cycle_grid.triggered.connect(self.scene_manager.cycle_grid_color)

        remove = QtWidgets.QAction("Remove Layer", self)
        remove.setShortcut(QtCore.Qt.Key_Delete)
        remove.triggered.connect(self.remove_layer)

        clear = QtWidgets.QAction("Clear Region Selection", self)
        clear.setShortcut(QtCore.Qt.Key_Escape)
        clear.triggered.connect(self.remove_region_polygon)

        composite = QtWidgets.QAction("Create Composite", self)
        composite.setShortcut('C')
        composite.triggered.connect(self.user_rgb_behavior.create_rgb)

        algebraic = QtWidgets.QAction("Create Algebraic", self)
        algebraic.triggered.connect(self.create_algebraic)

        toggle_point = QtWidgets.QAction("Toggle Point Probe", self)
        toggle_point.setShortcut('X')
        toggle_point.triggered.connect(lambda: self.graphManager.toggle_point_probe(DEFAULT_POINT_PROBE))

        open_gradient = QtWidgets.QAction("Toggle Colormap Editor", self)
        open_gradient.setShortcut("Ctrl+E")
        open_gradient.triggered.connect(self.open_colormap_editor)

        edit_menu = menubar.addMenu('&Edit')
        edit_menu.addAction(remove)
        edit_menu.addAction(clear)
        edit_menu.addAction(toggle_point)
        edit_menu.addAction(open_gradient)

        layer_menu = menubar.addMenu('&Layer')
        layer_menu.addAction(composite)
        layer_menu.addAction(algebraic)

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

    def _init_key_releases(self):
        def cb_factory(required_key, cb):
            def tmp_cb(key, cb=cb):
                if key.text == required_key:
                    return cb()

            return tmp_cb

        self.scene_manager.main_canvas.events.key_release.connect(cb_factory("t", self.scene_manager.next_tool))

    def updateLayerList(self):
        # self.ui.layers.add
        pass

    def open_colormap_editor(self):
        if self._cmap_editor is None:
            self._cmap_editor = ColormapEditor(doc=self.document)
        self._cmap_editor.show()


def set_default_geometry(window, desktop=0):
    screen = QtWidgets.QApplication.desktop()
    screen_geometry = screen.screenGeometry(desktop)
    # TODO: Remove platform specific code
    if 'darwin' not in sys.platform:
        w, h = screen_geometry.width() - 400, screen_geometry.height() - 300
        window.setGeometry(200, 150, w, h)
    else:
        size = window.size()
        w, h = size.width(), size.height()
        center = screen_geometry.center()
        screen_x, screen_y = center.x(), center.y()
        window.move(int(screen_x - w / 2.), int(screen_y - h / 2.))


def _search_paths(arglist):
    for arg in arglist:
        for subpath in arg.split(':'):
            yield subpath


def create_app() -> (app.Application, QtWidgets.QApplication):
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    vispy_app = app.use_app('pyqt5')
    qt_app = vispy_app.create()
    if hasattr(QtWidgets.QStyleFactory, 'AA_UseHighDpiPixmaps'):
        qt_app.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps)
    return vispy_app, qt_app


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run SIFT")
    parser.add_argument("-w", "--workspace-dir", default=WORKSPACE_DB_DIR,
                        help="Specify workspace base directory")
    parser.add_argument("--cache-dir",
                        help="(DEPRECATED: use --workspace-dir) Specify workspace directory")
    parser.add_argument('--clear-workspace', action='store_true',
                        help="Remove workspace contents during start up")
    parser.add_argument("--config-dir", default=DOCUMENT_SETTINGS_DIR,
                        help="Specify config directory")
    parser.add_argument("-s", "--space", default=256, type=int,
                        help="Specify max amount of data to hold in workspace cache in Gigabytes")
    parser.add_argument("--border-shapefile", default=None,
                        help="Specify alternative coastline/border shapefile")
    parser.add_argument("--glob-pattern", default=os.environ.get("TIFF_GLOB", None),
                        help="Specify glob pattern for input images")
    parser.add_argument('-p', '--path', dest='paths', action="append",
                        help='directory to search for data [MULTIPLE ALLOWED]')
    parser.add_argument("-c", "--center", nargs=2, type=float,
                        help="Specify center longitude and latitude for camera")
    parser.add_argument("--desktop", type=int, default=0,
                        help="Number of monitor/display to show the main window on (0 for main, 1 for secondary, etc.)")
    parser.add_argument('-v', '--verbose', dest='verbosity', action="count",
                        default=int(os.environ.get("VERBOSITY", 2)),
                        help='each occurrence increases verbosity 1 level through '
                             'ERROR-WARNING-Info-DEBUG (default Info)')
    args = parser.parse_args()

    levels = [logging.ERROR, logging.WARN, logging.INFO, logging.DEBUG]
    level = levels[min(3, args.verbosity)]
    logging.basicConfig(level=level, datefmt='%H:%M:%S',
                        format='%(levelname)s %(asctime)s %(module)s:%(funcName)s:L%(lineno)d %(message)s')
    check_grib_definition_dir()
    check_imageio_deps()
    # logging.getLogger('vispy').setLevel(level)

    if args.cache_dir:
        LOG.warning("'--cache-dir' is deprecated, use '--workspace-dir'")
        args.workspace_dir = args.cache_dir

    LOG.info("Using configuration directory: %s", args.config_dir)
    LOG.info("Using cache directory: %s", args.cache_dir)
    vispy_app, qt_app = create_app()

    # Add our own fonts to Qt windowing system
    font_pattern = os.path.join(get_package_data_dir(), 'fonts', '*')
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
    vispy_app.run()


if __name__ == '__main__':
    sys.exit(main())
