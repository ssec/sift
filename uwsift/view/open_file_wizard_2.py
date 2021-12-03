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
import logging
import os
from collections import OrderedDict
from enum import Enum
from typing import Generator, Tuple, Union

import satpy.resample
import trollsift.parser as fnparser
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtCore import QPoint
from PyQt5.QtWidgets import QMenu
from satpy.readers import group_files

from uwsift import config
from uwsift.model.area_definitions_manager import AreaDefinitionsManager
from uwsift.satpy_compat import DataID, get_id_value
from uwsift.ui.open_file_wizard_2_ui import Ui_openFileWizard
from uwsift.util.common import create_scenes
from uwsift.workspace.importer import available_satpy_readers, filter_dataset_ids

LOG = logging.getLogger(__name__)

# Key in .yaml files where data reading information is stored for each reader
DATA_READING_CONFIG_KEY = 'data_reading'

CHECKMARK = '✔️'

FILE_PAGE = 0
PRODUCT_PAGE = 1


class Conf(Enum):
    # Just to have a well defined constant to express "skip this resampler"
    SKIP = 1


RESAMPLING_METHODS = {
    # Configure a display name for each resampling method ID as well as for
    # which geometry definition (AreaDefinition or SwathDefinition for now) it
    # works by associating an according tuple of strings where the first item is
    # the display name followed by any suitable geometry definition.
    #
    # To hide a resampling method just associate Conf.SKIP with its ID.
    #
    # This configuration is evaluated in
    # OpenFileWizard.update_resampling_method_combobox().
    'none':            ('None',
                        'AreaDefinition'),
    'kd_tree':           Conf.SKIP,  # synonym for `nearest`, don't show both
    'nearest':         ('Nearest Neighbor',
                        'AreaDefinition', 'SwathDefinition'),
    'ewa':             ('Elliptical Weighted Averaging',
                        'AreaDefinition', 'SwathDefinition'),
    'bilinear':        ('Bilinear',
                        'AreaDefinition', 'SwathDefinition'),
    'native':          ('Native',
                        'AreaDefinition'),
    'gradient_search': ('Gradient Search',
                        'AreaDefinition'),
    'bucket_avg':      ('Bucket Average',
                        'AreaDefinition', 'SwathDefinition'),
    'bucket_sum':      ('Bucket Sum',
                        'AreaDefinition'),
    'bucket_count':    ('Bucket Count',
                        'AreaDefinition'),
    'bucket_fraction': ('Bucket Fraction',
                        'AreaDefinition'),
}


class GroupingMode(Enum):
    # Keep in sync with uwsift/ui/open_file_wizard_2.ui
    # TODO initialize groupingModeComboBox programmatically
    BY_GROUP_KEYS = 0
    KEEP_SEPARATE = 1
    MERGE_ALL = 2


class OpenFileWizard(QtWidgets.QWizard):
    AVAILABLE_READERS = OrderedDict()
    inputParametersChanged = QtCore.pyqtSignal()

    def __init__(self, base_dir=None, base_reader=None, parent=None):
        super(OpenFileWizard, self).__init__(parent)

        self._initial_directory = base_dir
        self._initial_reader = base_reader

        # tuple(filenames) -> scene object
        self.scenes = {}
        self.all_available_products = None
        self.file_groups = {}
        self.unknown_files = set()

        # ------------------------------------------------------------------------------------------
        # GENERAL GUI SETTINGS
        # ------------------------------------------------------------------------------------------

        # allow maximizing the wizard and minimizing the whole application from wizard window
        self.setWindowFlags(self.windowFlags() |
                            QtCore.Qt.CustomizeWindowHint | QtCore.Qt.WindowMinimizeButtonHint |
                            QtCore.Qt.WindowMaximizeButtonHint | QtCore.Qt.WindowCloseButtonHint)
        # enable context menus
        self.setContextMenuPolicy(QtCore.Qt.ActionsContextMenu)
        # assume the config doesn't change through the lifetime of the wizard
        self.config = config.get('open_file_wizard')

        self.ui = Ui_openFileWizard()
        self.ui.setupUi(self)

        # ------------------------------------------------------------------------------------------
        # SIGNAL & SLOT CONNECTIONS
        # ------------------------------------------------------------------------------------------

        # Page 1 - File selection

        # open folder dialog on '...' button click
        self.ui.selectFolderButton.released.connect(self._open_select_folder_dialog)

        # On reader index change: update filter patterns. Also triggers
        # input_parameters_changed in the end so that file table is updated with the new pattern.
        self.ui.readerComboBox.currentIndexChanged.connect(self._update_filter_patterns)
        self.ui.readerComboBox.currentIndexChanged.connect(self._update_grouping_mode_combobox)

        # on filter pattern (displayed value, as this is editable) change: update file table
        self.ui.filterPatternComboBox.currentTextChanged.connect(self.inputParametersChanged.emit)
        # on folder change: update file table
        self.ui.folderTextBox.textChanged.connect(self.inputParametersChanged.emit)
        # on input parameter change (e.g.: filter pattern, folder): update file table
        self.inputParametersChanged.connect(self._update_file_table)
        # on change of selection: group files and check if selection is valid
        self.ui.fileTable.itemSelectionChanged.connect(
            self._synchronize_checkmarks_and_check_file_page_completeness)
        # on change of sorting: temporarily pause sorting while sorted by checked state
        self.ui.fileTable.horizontalHeader().sortIndicatorChanged.connect(
            self._file_sorting_changed)

        # Page 2 - Product selection

        self._all_selected = False
        self.ui.selectAllButton.clicked.connect(self.select_all_products_state)
        self.ui.selectIDTable.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.ui.selectIDTable.customContextMenuRequested.connect(self._product_context_menu)

        self.ui.resamplingMethodComboBox\
            .currentIndexChanged.connect(self.update_resampling_info)
        self.ui.resamplingMethodComboBox.currentIndexChanged\
            .connect(self.update_activation_of_projection_combobox)

        self.ui.radiusOfInfluenceSpinBox.valueChanged\
            .connect(self.update_resampling_info)

        self.ui.projectionComboBox.addItems(
            AreaDefinitionsManager.available_area_def_names())
        self.ui.projectionComboBox.setCurrentIndex(
            parent.document.current_projection_index())
        self.ui.projectionComboBox.currentIndexChanged.\
            connect(self.update_resampling_info)

        self._update_resampling_shape_spin_boxes()
        self.ui.projectionComboBox.currentIndexChanged\
            .connect(self._update_resampling_shape_spin_boxes)
        self.ui.resamplingShapeRowSpinBox.valueChanged.connect(self.update_resampling_info)
        self.ui.resamplingShapeColumnSpinBox.valueChanged.connect(self.update_resampling_info)

        # GUI has been initialized, make sure we have a consistent
        # resampling_info
        self.resampling_info = None
        self.update_resampling_info()

        # on cell change: check if page is complete
        self.ui.selectIDTable.cellChanged\
            .connect(self._check_product_page_completeness)

    # ==============================================================================================
    # PUBLIC GENERAL WIZARD INTERFACE
    # ==============================================================================================

    def initializePage(self, page_id: int):
        if page_id == FILE_PAGE:
            self._init_file_page()
        elif page_id == PRODUCT_PAGE:
            self._init_product_select_page()

    def validateCurrentPage(self) -> bool:
        """Check that the current page will generate the necessary data."""

        # start of general validation logic
        valid = super(OpenFileWizard, self).validateCurrentPage()
        if not valid:
            self.ui.statusMessage.setText('')
            return valid

        page_id: int = self.currentId()
        if page_id == FILE_PAGE:
            # Check for completeness when pressing NEXT, and not a lot of times when selection
            # changes. In case the check fails 'page_complete' is set to False.
            self._check_selected_files_for_compatibility_with_reader()
            if not self.ui.fileSelectionPage.page_complete:
                return False

            # try to create scenes
            try:
                self.all_available_products \
                    = create_scenes(self.scenes, self.file_groups)
            except IOError as e:
                self.ui.statusMessage.setText(f"ERROR: {e}")
                self.ui.statusMessage.setStyleSheet('color: red')
                return False
            except (RuntimeError, ValueError):
                LOG.error("Could not load files with Satpy reader.")
                LOG.debug("Could not load files with Satpy reader.", exc_info=True)
                self.ui.statusMessage.setText("ERROR: Could not load files with specified reader")
                self.ui.statusMessage.setStyleSheet('color: red')
                return False

            # ensure at least 1 product could be created
            if not self.all_available_products:
                LOG.error("No known products can be loaded from the selected files.")
                self.ui.statusMessage.setText(
                    "ERROR: No known products can be loaded from the selected files.")
                self.ui.statusMessage.setStyleSheet('color: red')
                return False

        self.ui.statusMessage.setText('')
        return True

    # ----------------------------------------------------------------------------------------------
    # PRIVATE GENERAL WIZARD INTERFACE
    # ----------------------------------------------------------------------------------------------

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

        reader_to_preselect = self._initial_reader or self.config['default_reader']
        for idx, (reader_short_name, reader_name) in enumerate(readers.items()):
            self.ui.readerComboBox.addItem(reader_short_name, reader_name)
            if reader_name == reader_to_preselect:
                self.ui.readerComboBox.setCurrentIndex(idx)

        self.ui.folderTextBox.setText(self._initial_directory)
        self._update_grouping_mode_combobox()

    def _init_product_select_page(self):
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
                item.setData(QtCore.Qt.UserRole, ds_id if col_idx == 0 else id_val)
                item.setFlags(
                    (item.flags() ^ QtCore.Qt.ItemIsEditable) | QtCore.Qt.ItemIsUserCheckable)
                if id_key == 'name':
                    item.setCheckState(_to_Qt_CheckState(self._all_selected))
                self.ui.selectIDTable.setItem(idx, col_idx, item)
                col_idx += 1

        self.update_resampling_method_combobox()

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

    # def _create_scenes(self):
    #     """Create Scene objects for the selected files."""
    #     all_available_products = set()
    #     for group_id, file_group in self.file_groups.items():
    #         scn = self.scenes.get(group_id)
    #         if scn is None:
    #             # need to create the Scene for the first time
    #             # file_group includes what reader to use
    #             # NOTE: We only allow a single reader at a time
    #             self.scenes[group_id] = scn = Scene(filenames=file_group)
    #
    #             # WORKAROUND: to decompress compressed SEVIRI HRIT files, an environment variable
    #             # needs to be set. Check if decompression might have introduced errors when using
    #             # the specific reader and loading a file with compression flag set.
    #             # NOTE: in case this workaround-check fails data cannot be loaded in SIFT although
    #             # creating the scene might have succeeded!
    #             compressed_seviri = False
    #             from satpy.readers.hrit_base import get_xritdecompress_cmd
    #             # TODO: Scene may not provide information about reader in the
    #             # future - here the "protected" variable '_readers' is used as
    #             # workaround already
    #             for r in scn._readers.values():
    #                 # only perform check when using a relevant reader, so that this is not triggered
    #                 # mistakenly when another reader uses the same meta data key for another purpose
    #                 if r.name in ['seviri_l1b_hrit']:
    #                     for fh in r.file_handlers.values():
    #                         for fh2 in fh:
    #                             if fh2.mda.get('compression_flag_for_data'):
    #                                 compressed_seviri = True
    #             if compressed_seviri:
    #                 get_xritdecompress_cmd()
    #             # END OF WORKAROUND
    #
    #         all_available_products.update(scn.available_dataset_ids())
    #
    #     # update the widgets
    #     self.all_available_products = sorted(all_available_products)

    # ==============================================================================================
    # PUBLIC CUSTOM INTERFACE
    # ==============================================================================================

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

    @property
    def files_to_load(self):
        """Return files that should be used by the Document/Workspace."""
        return [fn for fgroup in self.file_groups.values() for fn in fgroup]

    def get_reader(self) -> str:
        return self.ui.readerComboBox.currentData()

    def get_directory(self) -> str:
        return self.ui.folderTextBox.text()

    # ==============================================================================================
    # PAGE 1 RELATED FUNCTIONALITY
    # ==============================================================================================

    def _open_select_folder_dialog(self):
        """Show folder chooser and update table with files"""

        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select folder to open",
            self.ui.folderTextBox.text() or os.getenv("HOME"))
        if not folder:
            return

        if os.path.exists(folder):
            self.ui.folderTextBox.setText(folder)

    def _update_filter_patterns(self):
        """Updates available file filter patterns by reading the config. Selects first entry."""
        reader = self.ui.readerComboBox.currentData()

        if reader is None:
            return

        filter_patterns =\
            config.get(DATA_READING_CONFIG_KEY + '.' + reader + '.' + 'filter_patterns', None)
        # Example value:
        # filter_patterns = ['{rate:1s}-000-{hrit_format:_<6s}-{platform_shortname:4s}_{service:_<7s}'
        #                  '-{channel:_<6s}___-{segment:_<6s}___-{start_time:%Y%m%d%H%M}-{:1s}_']
        if filter_patterns is None:
            filter_patterns = []

        # always append wildcard to show all files, i.e. not filter at all
        filter_patterns = filter_patterns.copy()
        filter_patterns.append('')

        self.ui.filterPatternComboBox.clear()
        self.ui.filterPatternComboBox.addItems(filter_patterns)

        self.inputParametersChanged.emit()

    def _update_file_table(self):
        """
        Clears and re-populates the file table. Columns may be removed/added.
        This method considers the selected reader, filter pattern and folder.
        """

        self.ui.statusMessage.setText('')

        table = self.ui.fileTable

        # clear table: remove all rows
        table.setRowCount(0)

        # retrieve column names from pattern, prepend 2 columns for selection state and filename
        filter_pattern = self.ui.filterPatternComboBox.currentText()
        pattern_convert_dict = None
        try:
            pattern_convert_dict = fnparser.get_convert_dict(filter_pattern)
        except ValueError as e:
            LOG.error(f'Invalid filter pattern: {filter_pattern}')
            self.ui.statusMessage.setText('Invalid filter pattern')
            self.ui.statusMessage.setStyleSheet('color: red')

        column_names = list((CHECKMARK, 'Filename'))
        if pattern_convert_dict is not None:
            column_names.extend([key for key in pattern_convert_dict if len(key) > 0])

        # update columns
        table.setColumnCount(len(column_names))
        table.setHorizontalHeaderLabels([c for c in column_names])

        # find files in selected folder
        folder = self.ui.folderTextBox.text()
        if os.path.exists(folder):
            # Inserting items while sorting is active may lead to inconsistent table data.
            # It is suggested to disable sorting while inserting items when using a table widget.
            table_sorting_enabled = table.isSortingEnabled()
            table.setSortingEnabled(False)
            for file in os.listdir(folder):
                try:
                    if len(pattern_convert_dict) == 0:
                        # if pattern is empty, show all files
                        table.insertRow(table.rowCount())
                        table.setItem(table.rowCount() - 1, 1, QtWidgets.QTableWidgetItem(file))
                    elif pattern_convert_dict is None or fnparser.validate(filter_pattern, file):
                        # if pattern matches, add more columns, and show error when pattern is invalid
                        table.insertRow(table.rowCount())
                        table.setItem(table.rowCount()-1, 1, QtWidgets.QTableWidgetItem(file))
                        p = fnparser.parse(filter_pattern, file)
                        for col in range(2, len(column_names)):
                            table.setItem(table.rowCount()-1, col, QtWidgets.QTableWidgetItem(
                                str(p.get(column_names[col], ''))
                            ))
                except Exception as e:
                    # As the error thrown by trollsift's validate function in case of an
                    # unparsable pattern has no class, a general 'Exception' is caught although
                    # this is not PEP8-compliant.
                    LOG.error(f'Invalid filter pattern: {filter_pattern}')
                    self.ui.statusMessage.setText('Invalid filter pattern')
                    self.ui.statusMessage.setStyleSheet('color: red')
            table.setSortingEnabled(table_sorting_enabled)

        # Initially (and if no sorting is applied), sort by filename (column: 1)
        if table.horizontalHeader().sortIndicatorSection() >= table.columnCount():
            self.ui.fileTable.sortByColumn(1, QtCore.Qt.AscendingOrder)

        # resize columns to fit content (table's sizeAdjustPolicy is set to AdjustToContents)
        table.resizeColumnsToContents()

    def _file_sorting_changed(self, logical_index, order):
        """
        Pause sorting after sorting by checked state until choosing another column.

        When sorting by checked state is active, clicking a row will lead to reordering of the
        table as the checked state of that row changes. To prevent confusion sorting needs to be
        paused until the user clicks another column to sort by. Clicking the checked state column
        itself will still work as usual and allow to sort asc/desc.
        """

        if logical_index == 0:
            # actually sort once, then disable sorting temporarily
            self.ui.fileTable.sortByColumn(logical_index, order)
            self.ui.fileTable.setSortingEnabled(False)
        else:
            # activate sorting when choosing another column
            self.ui.fileTable.setSortingEnabled(True)

    def _synchronize_checkmarks_and_check_file_page_completeness(self, *args, **kwargs):
        """update status message, check if this page is complete."""

        self.ui.fileSelectionPage.page_complete = False

        # synchronize selection with checkmarks to enable sorting by marked files
        indices = self.ui.fileTable.selectionModel().selectedRows()
        for r in range(self.ui.fileTable.rowCount()):
            filename = self.ui.fileTable.item(r, 1).text()
            if r in [index.row() for index in indices]:
                self.ui.fileTable.setItem(r, 0, QtWidgets.QTableWidgetItem(CHECKMARK))
                self.ui.fileSelectionPage.page_complete = True
            else:
                self.ui.fileTable.setItem(r, 0, QtWidgets.QTableWidgetItem(''))

        self.ui.fileSelectionPage.completeChanged.emit()

    def _check_selected_files_for_compatibility_with_reader(self):
        """TODO: description"""
        self.ui.statusMessage.setText('Checking file/reader compatibility...')
        self.ui.statusMessage.setStyleSheet('color: black')
        reader = self.ui.readerComboBox.currentData()
        groups_updated = self._group_files(reader)
        if groups_updated:
            self._mark_unknown_files()
        if not self.file_groups:
            # if none of the files were usable then the user can't click Next
            self.ui.fileSelectionPage.page_complete = False
            LOG.error("Could not load any file with specified reader.")
            self.ui.statusMessage.setText("ERROR: Could not load any file with specified reader")
            self.ui.statusMessage.setStyleSheet('color: red')
        else:
            self.ui.statusMessage.setText('')

    def _group_files(self, reader) -> bool:
        """Group provided files by some keys, especially time step."""

        # reset state
        self.scenes = {}
        self.file_groups = {}
        self.unknown_files = set()

        # get filenames from table's 'Filename' column
        # TODO: in future, use a data model for the table and get filenames from there
        folder = self.ui.folderTextBox.text()
        selected_files = set([os.path.join(folder, self.ui.fileTable.item(r.row(), 1).text())
                              for r in self.ui.fileTable.selectionModel().selectedRows()])

        # if there's nothing to group, return
        if len(selected_files) == 0:
            return True

        # Read group_keys from SIFT reader-specific config. If not present, Satpy's config is used.
        group_keys = config.get(DATA_READING_CONFIG_KEY + '.' + reader + '.group_keys', None)

        grouping_mode = GroupingMode(
            self.ui.groupingModeComboBox.currentIndex())

        file_groups = None
        if grouping_mode == GroupingMode.BY_GROUP_KEYS:
            file_groups = group_files(selected_files, reader=reader,
                                      group_keys=group_keys)
        elif grouping_mode == GroupingMode.MERGE_ALL:
            file_groups = [{reader: list(selected_files)}]
        else:  # elif grouping_mode == GroupingModes.KEEP_SEPARATE:
            file_groups = [{reader: [file]} for file in selected_files]

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

    def _mark_unknown_files(self):
        """Mark rows unknown to the reader in red color and remove selection"""
        unknown_filenames = [os.path.basename(f) for f in self.unknown_files]
        for r in range(self.ui.fileTable.rowCount()):
            filename = self.ui.fileTable.item(r, 1).text()
            if filename in unknown_filenames:
                # change currently selected row visually (allows change of background/foreground)
                self.ui.fileTable.setCurrentCell(-1, -1)
                # remove selection from model
                self.ui.fileTable.item(r, 0).setSelected(False)
                # change visual representation of all cells in this row
                for c in range(self.ui.fileTable.columnCount()):
                    self.ui.fileTable.item(r, c).setForeground(QtGui.QColor(255, 0, 0))

    # ==============================================================================================
    # PAGE 2 RELATED FUNCTIONALITY
    # ==============================================================================================

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
            name_item.setCheckState(_to_Qt_CheckState(select))

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

    def _check_product_page_completeness(self):
        """update status message, check if this page is complete."""

        self.ui.productSelectionPage.page_complete = False

        for row_idx in range(self.ui.selectIDTable.rowCount()):
            item = self.ui.selectIDTable.item(row_idx, 0)
            # if at least 1 item is checked
            if item is not None and item.checkState():
                self.ui.productSelectionPage.page_complete = True
                break

        self.ui.productSelectionPage.completeChanged.emit()

    def update_resampling_method_combobox(self):
        reader = self.get_reader()
        geometry_definition: str = config.get(f'data_reading.{reader}'
                                              f'.geometry_definition',
                                              'AreaDefinition')

        self.ui.resamplingMethodComboBox.blockSignals(True)
        self.ui.resamplingMethodComboBox.clear()

        cb_model = self.ui.resamplingMethodComboBox.model()

        known_resampling_methods = ['none']
        known_resampling_methods.extend(satpy.resample.RESAMPLERS)
        first_enabled_item_index = -1
        for resampling_method in known_resampling_methods:
            configuration = RESAMPLING_METHODS.get(resampling_method, None)
            if configuration == Conf.SKIP:
                continue
            # If not explicitly skipped but also not "configured" in
            # RESAMPLING_METHODS add the resampler's ID, disabled (see below).
            # This makes "unknown" (newly added to Satpy) resamplers show up
            # drawing attention to ask a developer to test and enable them.
            resampling_method_name = \
                configuration[0] if configuration else resampling_method
            self.ui.resamplingMethodComboBox.addItem(
                resampling_method_name,
                userData=resampling_method)

            # Check, whether current item is approved for detected geometry
            # (area or swath). Disable if not and make sure the first enabled
            # item is preselected
            item_index = self.ui.resamplingMethodComboBox.count() - 1
            if not configuration or geometry_definition not in configuration:
                item = cb_model.item(item_index)
                item.setEnabled(False)
            elif first_enabled_item_index < 0:
                first_enabled_item_index = item_index
                self.ui.resamplingMethodComboBox\
                    .setCurrentIndex(first_enabled_item_index)

        self.update_resampling_info()
        self._set_opts_disabled(
            self.ui.resamplingMethodComboBox.currentData() == 'none')
        self.ui.resamplingMethodComboBox.blockSignals(False)

    def update_activation_of_projection_combobox(self):
        if self.ui.resamplingMethodComboBox.currentData() != 'none':
            self._set_opts_disabled(False)
        else:
            self._set_opts_disabled(True)
            self._reset_fields()

    def update_resampling_info(self):
        area_def_name = self.ui.projectionComboBox.currentText()
        area_def = AreaDefinitionsManager.area_def_by_name(area_def_name)

        resampler = self.ui.resamplingMethodComboBox.currentData()
        if not resampler or resampler.lower() == 'none':
            # gracefully interpret capitalization variants of 'None' as:
            # "do not resample"
            self.resampling_info = None
        else:
            self.resampling_info = {
                'resampler': resampler,
                'area_id': area_def.area_id,
                'projection': area_def.proj_str,
                'radius_of_influence': self.ui.radiusOfInfluenceSpinBox.value(),
                'shape': (self.ui.resamplingShapeRowSpinBox.value(),
                          self.ui.resamplingShapeColumnSpinBox.value())
            }

    def _set_opts_disabled(self, is_disabled):
        self.ui.radiusOfInfluenceSpinBox.setDisabled(is_disabled)
        # The user should not change the projection nor the resampling shape,
        # thus:
        self.ui.projectionComboBox.setDisabled(True)  # instead of 'is_disabled'
        self.ui.resamplingShapeRowSpinBox.setDisabled(True)  # instead of 'is_disabled'
        self.ui.resamplingShapeColumnSpinBox.setDisabled(True)  # instead of 'is_disabled'

    def _reset_fields(self):
        self.ui.resamplingMethodComboBox.setCurrentIndex(0)
        self.ui.radiusOfInfluenceSpinBox.setValue(5000)
        self.ui.projectionComboBox.setCurrentIndex(self.parent().document
                                                   .current_projection_index())
        self._set_opts_disabled(True)

    def _update_resampling_shape_spin_boxes(self):
        area_def_name = self.ui.projectionComboBox.currentText()
        area_def = \
            AreaDefinitionsManager.area_def_by_name(area_def_name)
        self.ui.resamplingShapeRowSpinBox.setValue(area_def.shape[0])
        self.ui.resamplingShapeColumnSpinBox.setValue(area_def.shape[1])

    def _update_grouping_mode_combobox(self):
        reader = self.get_reader()
        geometry_definition: str = config.get(f'data_reading.{reader}'
                                              f'.geometry_definition',
                                              'AreaDefinition')

        self.ui.groupingModeComboBox.blockSignals(True)

        cb_model = self.ui.groupingModeComboBox.model()

        if geometry_definition == 'SwathDefinition':
            # cb_model.item(GroupingModes.BY_GROUP_KEYS.value).setEnabled(True)
            cb_model.item(GroupingMode.KEEP_SEPARATE.value).setEnabled(True)
            cb_model.item(GroupingMode.MERGE_ALL.value).setEnabled(True)
        else:
            # cb_model.item(GroupingModes.BY_GROUP_KEYS.value).setEnabled(True)
            cb_model.item(GroupingMode.KEEP_SEPARATE.value).setEnabled(False)
            cb_model.item(GroupingMode.MERGE_ALL.value).setEnabled(False)
            self.ui.groupingModeComboBox.setCurrentIndex(
                GroupingMode.BY_GROUP_KEYS.value)

        self.ui.groupingModeComboBox.blockSignals(False)


def _to_Qt_CheckState(value: bool):
    return QtCore.Qt.Checked if value else QtCore.Qt.Unchecked