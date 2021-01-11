#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of SIFT.
#
# SIFT is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# SIFT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with SIFT.  If not, see <http://www.gnu.org/licenses/>.

import os
import logging
from enum import Enum
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtCore import QPoint
from PyQt5.QtWidgets import QMenu
from collections import OrderedDict
from typing import Generator, Tuple, Union

from uwsift.satpy_compat import DataID, get_id_value
from satpy import Scene
from satpy.readers import group_files

from uwsift import config
from uwsift.ui.open_file_wizard_ui import Ui_openFileWizard
from uwsift.workspace.importer import available_satpy_readers, filter_dataset_ids

LOG = logging.getLogger(__name__)

FILE_PAGE = 0
PRODUCT_PAGE = 1


class OpenFileWizard(QtWidgets.QWizard):
    AVAILABLE_READERS = OrderedDict()
    filesChanged = QtCore.pyqtSignal()

    def __init__(self, base_dir=None, parent=None):
        super(OpenFileWizard, self).__init__(parent)
        # enable context menus
        self.setContextMenuPolicy(QtCore.Qt.ActionsContextMenu)
        # assume this doesn't change through the lifetime of the wizard
        self.config = config.get('open_file_wizard')
        self.last_open_dir = base_dir
        self._all_filenames = set()
        self._filelist_changed = False
        # tuple(filenames) -> scene object
        self.scenes = {}
        self.all_available_products = None
        self.previous_reader = None
        self.file_groups = {}
        self.unknown_files = set()
        app = QtWidgets.QApplication.instance()
        self._unknown_icon = app.style().standardIcon(QtGui.QStyle.SP_DialogCancelButton)
        self._known_icon = QtGui.QIcon()
        # self._known_icon = app.style().standardIcon(QtGui.QStyle.SP_DialogApplyButton)

        self.ui = Ui_openFileWizard()
        self.ui.setupUi(self)

        self.ui.addButton.released.connect(self.add_file)
        self.ui.removeButton.released.connect(self.remove_file)
        # Connect signals so Next buttons are determined by selections on pages
        self._connect_next_button_signals(self.ui.fileList, self.ui.fileSelectionPage)
        # self.ui.fileList.model().rowsInserted.connect(self._file_selection_or_reader_changed)
        # self.ui.fileList.model().rowsRemoved.connect(self._file_selection_or_reader_changed)
        # self.ui.fileList.itemChanged.connect(self._file_selection_or_reader_changed)
        self.filesChanged.connect(self._file_selection_or_reader_changed)
        self.ui.readerComboBox.currentIndexChanged.connect(self._file_selection_or_reader_changed)
        self.ui.statusMessage.setText("")

        # Page 2 - Product selection
        self._connect_next_button_signals(self.ui.selectIDTable, self.ui.productSelectionPage)
        self._all_selected = True
        self.ui.selectAllButton.clicked.connect(self.select_all_products_state)
        self.ui.selectIDTable.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.ui.selectIDTable.customContextMenuRequested.connect(self._product_context_menu)

    def _file_selection_or_reader_changed(self, *args, **kwargs):
        """Set status message to empty text and check if this page is complete."""
        self.ui.fileSelectionPage.sift_page_checked = True
        if self.ui.fileList.count() != 0:
            self.ui.statusMessage.setText('Checking file/reader compatibility...')
            self.ui.statusMessage.setStyleSheet('color: black')
            reader = self.ui.readerComboBox.currentData()
            groups_updated = self._group_files(reader)
            if groups_updated:
                self._mark_unknown_files()
            if not self.file_groups:
                # if none of the files were usable then the user can't click Next
                self.ui.fileSelectionPage.sift_page_checked = False
                self.ui.statusMessage.setText("ERROR: Could not load any files with specified reader")
                self.ui.statusMessage.setStyleSheet('color: red')
            else:
                self.ui.statusMessage.setText('')
        self.ui.fileSelectionPage.completeChanged.emit()

    def _mark_unknown_files(self):
        for row_idx in range(self.ui.fileList.count()):
            item = self.ui.fileList.item(row_idx)
            fn = item.text()
            if fn in self.unknown_files:
                item.setIcon(self._unknown_icon)
            else:
                item.setIcon(self._known_icon)

    def _connect_next_button_signals(self, widget, page):
        widget.model().rowsInserted.connect(page.completeChangedSlot)
        widget.model().rowsRemoved.connect(page.completeChangedSlot)
        widget.itemChanged.connect(page.completeChangedSlot)

    def _disconnect_next_button_signals(self, widget, page):
        widget.model().rowsInserted.disconnect(page.completeChangedSlot)
        widget.model().rowsRemoved.disconnect(page.completeChangedSlot)
        widget.itemChanged.disconnect(page.completeChangedSlot)

    def select_all_products_state(self, checked: bool):
        """Select all or deselect all products listed on the product table."""
        # the new state (all selected or all unselected)
        self._all_selected = not self._all_selected
        self.select_all_products(select=self._all_selected)

    def select_all_products(self, select=True, prop_key: Union[str, None] = None,
                            prop_val: Union[str, None] = None):
        """Select products based on a specific property."""
        for row_idx in range(self.ui.selectIDTable.rowCount()):
            # our check state goes on the name item (always)
            name_item = self.ui.selectIDTable.item(row_idx, 0)
            if prop_key is not None:
                item_id = name_item.data(QtCore.Qt.UserRole)
                if get_id_value(item_id, prop_key) != get_id_value(prop_val, prop_key):
                    continue
            check_state = self._get_checked(select)
            name_item.setCheckState(check_state)

    def _product_context_menu(self, position: QPoint):
        item = self.ui.selectIDTable.itemAt(position)
        col = item.column()
        id_comp = self.config['id_components'][col]
        # first column always has DataID
        id_data = self.ui.selectIDTable.item(item.row(), 0).data(QtCore.Qt.UserRole)
        menu = QMenu()
        select_action = menu.addAction("Select all by '{}'".format(id_comp))
        deselect_action = menu.addAction("Deselect all by '{}'".format(id_comp))
        action = menu.exec_(self.ui.selectIDTable.mapToGlobal(position))
        if action == select_action or action == deselect_action:
            select = action == select_action
            self.select_all_products(select=select, prop_key=id_comp, prop_val=id_data)

    def collect_selected_ids(self):
        selected_ids = []
        prime_key = self.config['id_components'][0]
        for item_idx in range(self.ui.selectIDTable.rowCount()):
            id_items = OrderedDict((key, self.ui.selectIDTable.item(item_idx, id_idx))
                                   for id_idx, key in enumerate(self.config['id_components']))
            if id_items[prime_key].checkState():
                data_id = id_items[prime_key]
                selected_ids.append(data_id.data(QtCore.Qt.UserRole))
        return selected_ids

    def initializePage(self, p_int):
        if p_int == FILE_PAGE:
            self._init_file_page()
        elif p_int == PRODUCT_PAGE:
            self._init_product_select_page()

    def validateCurrentPage(self) -> bool:
        """Check that the current page will generate the necessary data."""
        valid = super(OpenFileWizard, self).validateCurrentPage()
        if not valid:
            self.ui.statusMessage.setText('')
            return valid

        p_int = self.currentId()
        if p_int == FILE_PAGE:
            try:
                self._create_scenes()
            except (RuntimeError, ValueError):
                LOG.error("Could not load files with Satpy reader.")
                LOG.debug("Could not load files with Satpy reader.", exc_info=True)
                self.ui.statusMessage.setText("ERROR: Could not load files with specified reader")
                self.ui.statusMessage.setStyleSheet('color: red')
                return False

            if not self.all_available_products:
                LOG.error("No known products can be loaded from the available files.")
                self.ui.statusMessage.setText("ERROR: No known products can be loaded from the available files.")
                self.ui.statusMessage.setStyleSheet('color: red')
                return False
        self.ui.statusMessage.setText('')
        return True

    def _group_files(self, reader) -> bool:
        """Group provided files by time step."""
        # the files haven't changed since we were last run
        if not self._filelist_changed and self.previous_reader == reader:
            return False

        self._filelist_changed = False
        self.file_groups = {}  # reset the list, just in case
        selected_files = self._all_filenames.copy()
        self.previous_reader = reader
        file_groups = group_files(selected_files, reader=reader)
        if not file_groups:
            self.unknown_files = selected_files
            self.file_groups = {}
            return True

        scenes = {}  # recreate Scene dictionary
        file_group_map = {}
        known_files = set()
        for file_group in file_groups:
            # file_group includes what reader to use
            # NOTE: We only allow a single reader at a time
            group_id = tuple(sorted(fn for group_list in file_group.values() for fn in group_list))
            known_files.update(group_id)
            if group_id not in self.scenes:
                # never seen this exact group of files before
                scenes[group_id] = None  # filled in later
            else:
                scenes[group_id] = self.scenes[group_id]
            file_group_map[group_id] = file_group
        self.scenes = scenes
        self.file_groups = file_group_map
        self.unknown_files = selected_files - known_files
        return True

    def _create_scenes(self):
        """Create Scene objects for the selected files."""
        all_available_products = set()
        for group_id, file_group in self.file_groups.items():
            scn = self.scenes.get(group_id)
            if scn is None:
                # need to create the Scene for the first time
                # file_group includes what reader to use
                # NOTE: We only allow a single reader at a time
                self.scenes[group_id] = scn = Scene(filenames=file_group)

            all_available_products.update(scn.available_dataset_ids())

        # update the widgets
        self.all_available_products = sorted(all_available_products)

    def _init_file_page(self):
        if self.AVAILABLE_READERS:
            readers = self.AVAILABLE_READERS
        else:
            satpy_readers = config.get('data_reading.readers')
            readers = available_satpy_readers(as_dict=True)
            readers = (r for r in readers if not satpy_readers or r['name'] in satpy_readers)
            readers = sorted(readers, key=lambda x: x.get('long_name', x['name']))
            readers = OrderedDict((ri.get('long_name', ri['name']), ri['name']) for ri in readers)
            OpenFileWizard.AVAILABLE_READERS = readers

        for idx, (reader_short_name, reader_name) in enumerate(readers.items()):
            self.ui.readerComboBox.addItem(reader_short_name, reader_name)
            if self.config['default_reader'] == reader_name:
                self.ui.readerComboBox.setCurrentIndex(idx)

    def _pretty_identifiers(self, data_id: DataID) -> Generator[Tuple[str, object, str], None, None]:
        """Determine pretty version of each identifier."""
        for key in self.config['id_components']:
            value = get_id_value(data_id, key)
            if value is None:
                pretty_val = "N/A"
            elif key == 'wavelength':
                pretty_val = "{:0.02f} µm".format(value[1])
            elif key == 'level':
                pretty_val = "{:d} hPa".format(int(value))
            elif key == 'resolution':
                pretty_val = "{:d}m".format(int(value))
            elif key == 'calibration' and isinstance(value, Enum):
                # calibration is an enum in newer Satpy version
                pretty_val = value.name
                value = value.name
            else:
                pretty_val = value

            yield key, value, pretty_val

    def _init_product_select_page(self):
        # Disconnect the signals until we are done setting up the widgets
        self._disconnect_next_button_signals(self.ui.selectIDTable, self.ui.productSelectionPage)

        # name and level
        id_components = self.config['id_components']
        self.ui.selectIDTable.setColumnCount(len(id_components))
        self.ui.selectIDTable.setHorizontalHeaderLabels([x.title() for x in id_components])
        for idx, ds_id in enumerate(filter_dataset_ids(self.all_available_products)):
            col_idx = 0
            for id_key, id_val, pretty_val in self._pretty_identifiers(ds_id):
                if id_key not in id_components:
                    continue

                self.ui.selectIDTable.setRowCount(idx + 1)
                item = QtWidgets.QTableWidgetItem(pretty_val)
                if col_idx == 0:
                    item.setData(QtCore.Qt.UserRole, ds_id)
                else:
                    item.setData(QtCore.Qt.UserRole, id_val)
                item.setFlags((item.flags() ^ QtCore.Qt.ItemIsEditable) | QtCore.Qt.ItemIsUserCheckable)
                if id_key == 'name':
                    item.setCheckState(QtCore.Qt.Checked)
                self.ui.selectIDTable.setItem(idx, col_idx, item)
                col_idx += 1

        # must do this after everything is added so we don't trigger the
        # "complete" checks a ton of times
        self._connect_next_button_signals(self.ui.selectIDTable, self.ui.productSelectionPage)

    def _get_checked(self, bool_val):
        if bool_val:
            return QtCore.Qt.Checked
        else:
            return QtCore.Qt.Unchecked

    def add_file(self):
        filename_filters = ['All files (*.*)']
        filter_str = ';;'.join(filename_filters)
        files = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Select one or more files to open", self.last_open_dir or os.getenv("HOME"), filter_str)[0]
        if not files:
            return
        self.last_open_dir = os.path.dirname(files[0])
        for fn in files:
            if fn in self._all_filenames:
                continue
            item = QtWidgets.QListWidgetItem(fn, self.ui.fileList)
            # turn off checkability so we can tell the difference between this
            # and the product list when checking WizardPage.isComplete
            item.setFlags(item.flags() & ~QtCore.Qt.ItemIsUserCheckable)
            self._filelist_changed = True
            self._all_filenames.add(fn)
            self.ui.fileList.addItem(item)
        self.filesChanged.emit()

    def remove_file(self):
        # need to go backwards to index numbers don't change
        for item_idx in range(self.ui.fileList.count() - 1, -1, -1):
            item = self.ui.fileList.item(item_idx)
            if item.isSelected():
                self._filelist_changed = True
                self._all_filenames.remove(item.text())
                self.ui.fileList.takeItem(item_idx)
        self.filesChanged.emit()

    @property
    def files_to_load(self):
        """Files that should be used by the Document/Workspace."""
        return [fn for fgroup in self.file_groups.values() for fn in fgroup]
