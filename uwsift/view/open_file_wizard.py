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
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import QPoint
from PyQt5.QtWidgets import QMenu, QAction
from collections import OrderedDict
from typing import Generator, Tuple, Iterable, Union

from satpy import Scene, DatasetID
from satpy.readers import group_files

from uwsift.ui.open_file_wizard_ui import Ui_openFileWizard

FILE_PAGE = 0
PRODUCT_PAGE = 1

ID_COMPONENTS = [
    'name',
    'wavelength',
    'resolution',
    'calibration',
    'level',
]
READERS = None  # all readers
EXCLUDE_DATASETS = {'calibration': ['radiance', 'counts']}


def _pretty_identifiers(ds_id: DatasetID) -> Generator[Tuple[str, object, str], None, None]:
    """Determine pretty version of each identifier."""
    for key in ID_COMPONENTS:
        value = getattr(ds_id, key, None)
        if value is None:
            pretty_val = "N/A"
        elif key == 'wavelength':
            pretty_val = "{:0.02f} µm".format(value[1])
        elif key == 'level':
            pretty_val = "{:d} hPa".format(value)
        elif key == 'resolution':
            pretty_val = "{:d}m".format(value)
        else:
            pretty_val = value

        yield key, value, pretty_val


def _filter_identifiers(ids_to_filter: Iterable[DatasetID]) -> Generator[DatasetID, None, None]:
    """Generate only non-filtered DatasetIDs based on EXCLUDE_DATASETS global filters."""
    # skip certain DatasetIDs
    for ds_id in ids_to_filter:
        for filter_key, filtered_values in EXCLUDE_DATASETS.items():
            if getattr(ds_id, filter_key) in filtered_values:
                break
        else:
            yield ds_id


class OpenFileWizard(QtWidgets.QWizard):
    AVAILABLE_READERS = []

    def __init__(self, base_dir=None, parent=None):
        super(OpenFileWizard, self).__init__(parent)
        # enable context menus
        self.setContextMenuPolicy(QtCore.Qt.ActionsContextMenu)
        self.last_open_dir = base_dir
        self._filenames = set()
        self._selected_files = []
        # tuple(filenames) -> scene object
        self.scenes = {}
        # filename -> group tuple of filenames
        self.all_known_files = {}

        self.ui = Ui_openFileWizard()
        self.ui.setupUi(self)

        self.ui.addButton.released.connect(self.add_file)
        self.ui.removeButton.released.connect(self.remove_file)
        # Connect signals so Next buttons are determined by selections on pages
        self._connect_next_button_signals(self.ui.fileList, self.ui.fileSelectionPage)
        self.ui.fileList.model().rowsInserted.connect(lambda x: self.ui.fileSelectionPage.completeChanged.emit())
        self.ui.fileList.model().rowsRemoved.connect(lambda x: self.ui.fileSelectionPage.completeChanged.emit())
        self.ui.fileList.itemChanged.connect(lambda x: self.ui.fileSelectionPage.completeChanged.emit())

        # Page 2 - Product selection
        self._connect_next_button_signals(self.ui.selectIDTable, self.ui.productSelectionPage)
        self._all_selected = True
        self.ui.selectAllButton.clicked.connect(self.select_all_products_state)
        self.ui.selectIDTable.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.ui.selectIDTable.customContextMenuRequested.connect(self._product_context_menu)

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
        if prop_key is not None:
            prop_column = ID_COMPONENTS.index(prop_key)
        else:
            prop_column = 0

        for row_idx in range(self.ui.selectIDTable.rowCount()):
            # our check state goes on the name item (always)
            name_item = self.ui.selectIDTable.item(row_idx, 0)
            if prop_key is not None:
                prop_item = self.ui.selectIDTable.item(row_idx, prop_column)
                item_val = prop_item.data(QtCore.Qt.UserRole)
                if item_val != prop_val:
                    continue
            check_state = self._get_checked(select)
            name_item.setCheckState(check_state)

    def _product_context_menu(self, position: QPoint):
        item = self.ui.selectIDTable.itemAt(position)
        col = item.column()
        id_comp = ID_COMPONENTS[col]
        id_val = item.data(QtCore.Qt.UserRole)
        menu = QMenu()
        select_action = menu.addAction("Select all by '{}'".format(id_comp))
        deselect_action = menu.addAction("Deselect all by '{}'".format(id_comp))
        action = menu.exec_(self.ui.selectIDTable.mapToGlobal(position))
        if action == select_action or action == deselect_action:
            select = action == select_action
            self.select_all_products(select=select, prop_key=id_comp, prop_val=id_val)

    def collect_selected_ids(self):
        selected_ids = []
        for item_idx in range(self.ui.selectIDTable.rowCount()):
            id_items = OrderedDict((key, self.ui.selectIDTable.item(item_idx, id_idx))
                                   for id_idx, key in enumerate(ID_COMPONENTS))
            if id_items['name'].checkState():
                id_dict = {key: id_item.data(QtCore.Qt.UserRole)
                           for key, id_item in id_items.items() if id_item is not None}
                selected_ids.append(DatasetID(**id_dict))
        return selected_ids

    def initializePage(self, p_int):
        if p_int == FILE_PAGE:
            self._init_file_page()
        elif p_int == PRODUCT_PAGE:
            self._init_product_select_page()

    def _init_file_page(self):
        if self.AVAILABLE_READERS:
            readers = self.AVAILABLE_READERS
        else:
            from satpy import available_readers
            readers = sorted(available_readers())
            OpenFileWizard.AVAILABLE_READERS = readers

        self.ui.readerComboBox.addItems(readers)

    def _init_product_select_page(self):
        if self._selected_files == self._filenames:
            return

        # Disconnect the signals until we are done setting up the widgets
        self._disconnect_next_button_signals(self.ui.selectIDTable, self.ui.productSelectionPage)

        self._selected_files = self._filenames.copy()
        all_available_products = set()
        reader = self.ui.readerComboBox.currentText()
        for file_group in group_files(self._filenames, reader=reader):
            # file_group includes what reader to use
            # NOTE: We only allow a single reader at a time
            groups_files = tuple(sorted(fn for group_id, group_list in file_group.items() for fn in group_list))
            if groups_files not in self.scenes:
                # never seen this exact group of files before
                # let's make sure we remove any previous sub-groups
                for fn in groups_files:
                    if fn in self.all_known_files:
                        del self.scenes[self.all_known_files[fn]]
                    self.all_known_files[fn] = groups_files
                self.scenes[groups_files] = scn = Scene(filenames=file_group)
            else:
                scn = self.scenes[groups_files]

            all_available_products.update(scn.available_dataset_ids())

        # update the widgets
        all_available_products = sorted(all_available_products)
        # name and level
        self.ui.selectIDTable.setColumnCount(len(ID_COMPONENTS))
        self.ui.selectIDTable.setHorizontalHeaderLabels([x.title() for x in ID_COMPONENTS])
        for idx, ds_id in enumerate(_filter_identifiers(all_available_products)):
            col_idx = 0
            for id_key, id_val, pretty_val in _pretty_identifiers(ds_id):
                if id_key not in ID_COMPONENTS:
                    continue

                self.ui.selectIDTable.setRowCount(idx + 1)
                item = QtWidgets.QTableWidgetItem(pretty_val)
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
            if fn in self._filenames:
                continue
            item = QtWidgets.QListWidgetItem(fn, self.ui.fileList)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Checked)
            self.ui.fileList.addItem(item)
            self._filenames.add(fn)

    def remove_file(self):
        # need to go backwards to index numbers don't change
        for item_idx in range(self.ui.fileList.count() - 1, -1, -1):
            item = self.ui.fileList.item(item_idx)
            if self.ui.fileList.isItemSelected(item):
                self.ui.fileList.takeItem(item_idx)
                self._filenames.remove(item.text())
