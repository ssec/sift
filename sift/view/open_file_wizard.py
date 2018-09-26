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
from sift.ui.open_file_wizard_ui import Ui_openFileWizard
from sift.common import get_font_size
from PyQt4 import QtGui, QtCore
from satpy import Scene, DatasetID
from collections import OrderedDict


FILE_PAGE = 0
PRODUCT_PAGE = 1
SUMMARY_PAGE = 2

BY_PARAMETER_TAB = 0
BY_ID_TAB = 1


class OpenFileWizard(QtGui.QWizard):
    def __init__(self, base_dir=None, parent=None):
        super(OpenFileWizard, self).__init__(parent)
        self._last_open_dir = base_dir
        self._filenames = set()
        self._selected_files = []
        # tuple(filenames) -> scene object
        self.scenes = {}
        self.selected_ids = []

        self.ui = Ui_openFileWizard()
        self.ui.setupUi(self)

        fsize = get_font_size(8.2)
        font = QtGui.QFont('Andale Mono')
        font.setPointSizeF(fsize)
        self.ui.productSummaryText.setFont(font)

        self.ui.addButton.released.connect(self.add_file)
        self.ui.removeButton.released.connect(self.remove_file)
        # Connect signals so Next buttons are determined by selections on pages
        self._connect_next_button_signals(self.ui.fileList, self.ui.fileSelectionPage)
        self.ui.fileList.model().rowsInserted.connect(lambda x: self.ui.fileSelectionPage.completeChanged.emit())
        self.ui.fileList.model().rowsRemoved.connect(lambda x: self.ui.fileSelectionPage.completeChanged.emit())
        self.ui.fileList.itemChanged.connect(lambda x: self.ui.fileSelectionPage.completeChanged.emit())
        self.ui.selectByTabWidget.currentChanged.connect(self._product_selection_tab_change)

        # Page 2 - Product selection
        self._connect_product_select_complete()

    def _connect_next_button_signals(self, widget, page):
        widget.model().rowsInserted.connect(page.completeChangedSlot)
        widget.model().rowsRemoved.connect(page.completeChangedSlot)
        widget.itemChanged.connect(page.completeChangedSlot)

    def _disconnect_next_button_signals(self, widget, page):
        widget.model().rowsInserted.disconnect(page.completeChangedSlot)
        widget.model().rowsRemoved.disconnect(page.completeChangedSlot)
        widget.itemChanged.disconnect(page.completeChangedSlot)

    def initializePage(self, p_int):
        self.selected_ids = []
        if p_int == FILE_PAGE:
            self._init_file_page()
        elif p_int == PRODUCT_PAGE:
            self._init_product_select_page()
        elif p_int == SUMMARY_PAGE:
            self._init_summary_page()

    def _init_file_page(self):
        all_readers = bool(os.getenv("SIFT_ALLOW_ALL_READERS", ""))
        if all_readers:
            from satpy import available_readers
            readers = sorted(available_readers())
        else:
            readers = ['grib']
            self.ui.readerComboBox.setDisabled(True)

        self.ui.readerComboBox.addItems(readers)

    def _disconnect_product_select_complete(self, tabs_switched=False):
        current_idx = self.ui.selectByTabWidget.currentIndex()
        if tabs_switched:
            # we want to disconnect the previous tab
            current_idx = 0 if current_idx else 1
        if current_idx == 0:
            self._disconnect_next_button_signals(self.ui.selectByNameList, self.ui.productSelectionPage)
            self._disconnect_next_button_signals(self.ui.selectByLevelList, self.ui.productSelectionPage)
        elif current_idx == 1:
            self._disconnect_next_button_signals(self.ui.selectIDTable, self.ui.productSelectionPage)

    def _connect_product_select_complete(self):
        current_idx = self.ui.selectByTabWidget.currentIndex()
        if current_idx == 0:
            self.ui.productSelectionPage.important_children = [
                self.ui.selectByNameList, self.ui.selectByLevelList]
            self._connect_next_button_signals(self.ui.selectByNameList, self.ui.productSelectionPage)
            self._connect_next_button_signals(self.ui.selectByLevelList, self.ui.productSelectionPage)
        elif current_idx == 1:
            self.ui.productSelectionPage.important_children = [self.ui.selectIDTable]
            self._connect_next_button_signals(self.ui.selectIDTable, self.ui.productSelectionPage)

    def _init_product_select_page(self):
        if self._selected_files == self._filenames:
            return

        # Disconnect the signals until we are done setting up the widgets
        self._disconnect_product_select_complete()

        self._selected_files = self._filenames.copy()
        all_available_products = set()
        for fn in self._selected_files:
            these_files = tuple(sorted([fn]))
            # TODO: We need to be able to figure out how many paths go to each
            #       Scene (add to satpy as utility function)
            if these_files in self.scenes:
                continue
            reader = self.ui.readerComboBox.currentText()
            scn = Scene(reader=reader, filenames=these_files)
            self.scenes[these_files] = scn
            all_available_products.update(scn.available_dataset_ids())

        # update the widgets
        all_available_products = sorted(all_available_products)
        self.ui.selectIDTable.setRowCount(len(all_available_products))
        # name and level
        self.ui.selectIDTable.setColumnCount(2)
        self.ui.selectIDTable.setHorizontalHeaderLabels(['name', 'level'])
        properties = OrderedDict((
            ('name', set()),
            ('level', set()),
        ))
        for idx, ds_id in enumerate(all_available_products):
            pretty_name = ds_id.name.upper() if ds_id.level is not None else ds_id.name
            properties['name'].add((ds_id.name, pretty_name))
            pretty_level = "{:d} hPa".format(ds_id.level) if ds_id.level is not None else 'NA'
            properties['level'].add((ds_id.level, pretty_level))

            item = QtGui.QTableWidgetItem(pretty_name)
            item.setData(QtCore.Qt.UserRole, ds_id.name)
            item.setFlags((item.flags() ^ QtCore.Qt.ItemIsEditable) | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Unchecked)
            self.ui.selectIDTable.setItem(idx, 0, item)
            item = QtGui.QTableWidgetItem(pretty_level)
            item.setData(QtCore.Qt.UserRole, ds_id.level)
            item.setFlags((item.flags() ^ QtCore.Qt.ItemIsEditable) | QtCore.Qt.ItemIsUserCheckable)
            self.ui.selectIDTable.setItem(idx, 1, item)

        # Update the per-property lists
        names = sorted(properties['name'])
        for name, pretty_name in names:
            item = QtGui.QListWidgetItem(pretty_name)
            item.setData(QtCore.Qt.UserRole, name)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Unchecked)
            self.ui.selectByNameList.addItem(item)
        levels = sorted(properties['level'])
        for level, pretty_level in levels:
            item = QtGui.QListWidgetItem(pretty_level)
            item.setData(QtCore.Qt.UserRole, level)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Unchecked)
            self.ui.selectByLevelList.addItem(item)

        self._connect_product_select_complete()

    def _get_checked(self, bool_val):
        if bool_val:
            return QtCore.Qt.Checked
        else:
            return QtCore.Qt.Unchecked

    def _reinit_parameter_tab(self):
        # we were on the ID list, now one the parameter lists
        # need to update which selections should be chosen
        names = set()
        levels = set()
        for item_idx in range(self.ui.selectIDTable.rowCount()):
            name_item = self.ui.selectIDTable.item(item_idx, 0)
            level_item = self.ui.selectIDTable.item(item_idx, 1)

            if name_item.checkState():
                names.add(name_item.data(QtCore.Qt.UserRole))
                levels.add(level_item.data(QtCore.Qt.UserRole))
        # enable all the names that were checked in the ID list
        for item_idx in range(self.ui.selectByNameList.count()):
            item = self.ui.selectByNameList.item(item_idx)
            name = item.data(QtCore.Qt.UserRole)
            item.setCheckState(self._get_checked(name in names))
        # enable all the levels that were checked in the ID list
        for item_idx in range(self.ui.selectByLevelList.count()):
            item = self.ui.selectByLevelList.item(item_idx)
            level = item.data(QtCore.Qt.UserRole)
            item.setCheckState(self._get_checked(level in levels))

    def _reinit_id_tab(self):
        names = set()
        for item_idx in range(self.ui.selectByNameList.count()):
            item = self.ui.selectByNameList.item(item_idx)
            if item.checkState():
                names.add(item.data(QtCore.Qt.UserRole))

        levels = set()
        for item_idx in range(self.ui.selectByLevelList.count()):
            item = self.ui.selectByLevelList.item(item_idx)
            if item.checkState():
                levels.add(item.data(QtCore.Qt.UserRole))

        for item_idx in range(self.ui.selectIDTable.rowCount()):
            name_item = self.ui.selectIDTable.item(item_idx, 0)
            level_item = self.ui.selectIDTable.item(item_idx, 1)

            name_selected = name_item.data(QtCore.Qt.UserRole) in names
            level_selected = level_item.data(QtCore.Qt.UserRole) in levels
            name_item.setCheckState(self._get_checked(name_selected and level_selected))

    def _product_selection_tab_change(self, tab_idx, force_reinit=True):
        self._disconnect_product_select_complete(tabs_switched=True)
        if tab_idx == BY_PARAMETER_TAB:
            self._reinit_parameter_tab()
        elif tab_idx == BY_ID_TAB:
            self._reinit_id_tab()
        self._connect_product_select_complete()

    def _init_summary_page(self):
        # we are going to use the id table to get our summary
        # so make sure the values are correct
        if self.ui.selectByTabWidget.currentIndex() != BY_ID_TAB:
            self.ui.selectByTabWidget.setCurrentIndex(BY_ID_TAB)

        selected_text = []
        selected_ids = []
        id_format = "| {name:<20s} | {level:>8s} |"
        header_format = "| {name:<20s} | {level:>8s} |"
        header_line = "|-{0:-^20s}-|-{0:-^8s}-|".format('-')
        for item_idx in range(self.ui.selectIDTable.rowCount()):
            name_item = self.ui.selectIDTable.item(item_idx, 0)
            level_item = self.ui.selectIDTable.item(item_idx, 1)
            if name_item.checkState():
                name = name_item.data(QtCore.Qt.UserRole)
                level = level_item.data(QtCore.Qt.UserRole)
                selected_ids.append(DatasetID(name=name, level=level))
                selected_text.append(id_format.format(
                    name=name_item.text(),
                    level=level_item.text(),
                ))

        self.selected_ids = selected_ids

        summary_text = """Products to be loaded: {}

""".format(len(selected_ids))

        header = header_format.format(name="Name", level="Level")
        summary_text += "\n".join([header, header_line] + selected_text)
        self.ui.productSummaryText.setText(summary_text)

    def add_file(self):
        filename_filters = [
            # 'All files (*.*)',
            # 'All supported files (*.nc *.nc4 *.tiff *.tif)',
            # 'GOES-16 NetCDF (*.nc *.nc4)',
            # 'Mercator GTIFF (*.tiff *.tif)',
            'NWP GRIB2 (*.grib2 *.f???)',
        ]
        if os.getenv("SIFT_ALLOW_ALL_READERS", ""):
            filename_filters = ['All files (*.*)'] + filename_filters
        filter_str = ';;'.join(filename_filters)
        files = QtGui.QFileDialog.getOpenFileNames(self,
                                                   "Select one or more files to open",
                                                   self._last_open_dir or os.getenv("HOME"),
                                                   filter_str)
        if not files:
            return
        self._last_open_dir = os.path.dirname(files[0])
        for fn in files:
            if fn in self._filenames:
                continue
            item = QtGui.QListWidgetItem(fn, self.ui.fileList)
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
