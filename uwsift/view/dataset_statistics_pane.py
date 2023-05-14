import numbers
from decimal import Decimal
from typing import Tuple

import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

from uwsift.common import Info
from uwsift.model.layer_item import LayerItem
from uwsift.ui.dataset_statistics_widget_ui import Ui_datasetStatisticsPane


class DatasetStatisticsPane(QtWidgets.QWidget):
    """The purpose of this class is to manage the statistic analysis pane.

    In the statistical analysis pane, the corresponding statistics of the first active dataset of a selected layer are
    displayed.
    """

    def __init__(self, *args, **kwargs):
        super(DatasetStatisticsPane, self).__init__(*args, **kwargs)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self._pane_ui = Ui_datasetStatisticsPane()
        self._pane_widget = QtWidgets.QWidget(self)
        self._pane_ui.setupUi(self._pane_widget)

        layout.addWidget(self._pane_widget)
        self.setLayout(layout)
        self._pane_ui.datasetNameLabel.setText("")
        self._pane_ui.decimalPlacesSpinBox.setValue(2)

        self._pane_ui.decimalPlacesSpinBox.valueChanged.connect(self._decimal_places_changed)

        self._current_selected_layer = None

        self._pane_ui.statisticsTableWidget.setSelectionMode(QtWidgets.QTableWidget.SingleSelection)
        self._pane_ui.statisticsTableWidget.resizeColumnsToContents()
        self._pane_ui.statisticsTableWidget.horizontalHeader().setStretchLastSection(True)

    # Slot functions

    def initiate_update(self):
        """Refresh the display of name and the statistics from the current active dataset or clear the pane if there is
        none.
        """
        self._clear_statistics_pane()  # TODO: start revision here if you want to keep parts of the table layout

        if not self._current_selected_layer:
            return

        first_active_dataset = self._current_selected_layer.get_first_active_product_dataset()
        if not first_active_dataset:
            return

        dataset_display_name = (
            f"{self._current_selected_layer.descriptor} {first_active_dataset.info[Info.DISPLAY_TIME]}"
        )
        self._pane_ui.datasetNameLabel.setText(dataset_display_name)

        stats = self._current_selected_layer.model._workspace.get_statistics_for_dataset_by_uuid(
            first_active_dataset.uuid
        )
        self._update_table_content(stats)

    def _decimal_places_changed(self):
        if self._current_selected_layer:
            first_active_dataset = self._current_selected_layer.get_first_active_product_dataset()
            if first_active_dataset:
                stats = self._current_selected_layer.model._workspace.get_statistics_for_dataset_by_uuid(
                    first_active_dataset.uuid
                )
                self._update_table_content(stats)

    def selection_did_change(self, layers: Tuple[LayerItem]):
        """Process if an update is needed or not. If it is then initiate the update process

        If multiple layers are selected then the statistics can not be viewed.
        """
        if layers is not None and len(layers) == 1:
            self._current_selected_layer = layers[0]
        else:
            self._current_selected_layer = None

        self.initiate_update()

    # Utility functions

    def _clear_statistics_pane(self):
        self._pane_ui.datasetNameLabel.setText("")
        self._pane_ui.statisticsTableWidget.setRowCount(0)
        self._pane_ui.statisticsTableWidget.setColumnCount(0)
        self._pane_ui.statisticsTableWidget.verticalHeader().show()
        self._pane_ui.statisticsTableWidget.horizontalHeader().show()

    @staticmethod
    def _determine_rounded_value(decimal_places: int, value: Decimal):
        if decimal_places > 1:
            decimal_places_str = f"{0:.{decimal_places - 1}f}1"
        elif decimal_places == 1:
            decimal_places_str = ".1"
        else:
            decimal_places_str = "1."
        value = value.quantize(Decimal(decimal_places_str))
        return value

    def _determine_table_content_by_stats_dict(self, stats: dict, header: list):
        keys = list(stats.keys())
        max_col_number = 0
        for key in keys:
            cur_col_number = len(stats[key])
            if cur_col_number > max_col_number:
                max_col_number = cur_col_number
        self._pane_ui.statisticsTableWidget.setRowCount(len(stats))
        self._pane_ui.statisticsTableWidget.setColumnCount(max_col_number)
        self._pane_ui.statisticsTableWidget.setVerticalHeaderLabels(list(stats.keys()))

        if header:
            self._pane_ui.statisticsTableWidget.setHorizontalHeaderLabels(header)
        else:
            self._pane_ui.statisticsTableWidget.horizontalHeader().hide()

        for row in range(len(stats)):
            curr_stat = stats[keys[row]]
            for col in range(len(curr_stat)):
                value = self._determine_value_for_table_item(curr_stat[col])
                item = self._get_tailored_item(value)
                self._pane_ui.statisticsTableWidget.setItem(row, col, item)

    def _determine_table_content_by_stats_list(self, stats: list, header: list):
        self._pane_ui.statisticsTableWidget.verticalHeader().hide()
        max_col_number = 0
        for row in stats:
            cur_col_number = len(row)
            if cur_col_number > max_col_number:
                max_col_number = cur_col_number
        self._pane_ui.statisticsTableWidget.setRowCount(len(stats))
        self._pane_ui.statisticsTableWidget.setColumnCount(max_col_number)

        if header:
            self._pane_ui.statisticsTableWidget.horizontalHeader().show()
            self._pane_ui.statisticsTableWidget.setHorizontalHeaderLabels(header)
        else:
            self._pane_ui.statisticsTableWidget.horizontalHeader().hide()

        for row in range(len(stats)):
            row_content = stats[row]
            for col in range(len(row_content)):
                col_content = row_content[col]
                value = self._determine_value_for_table_item(col_content)
                item = self._get_tailored_item(value)
                self._pane_ui.statisticsTableWidget.setItem(row, col, item)

    def _determine_value_for_table_item(self, value: float) -> str:
        decimal_places = self._pane_ui.decimalPlacesSpinBox.value()

        if isinstance(value, np.number):
            # if the given value is a numeric numpy value then it needs to be converted
            # to a standard python numeric value because the used Decimal class can't
            # convert a numpy numeric value to a Decimal class object
            value = value.item()

        value_str = str(value)
        if isinstance(value, numbers.Number):
            if isinstance(value, int):
                value_str = str(value)
            elif decimal_places > -1:
                value_str = str(self._determine_rounded_value(decimal_places, Decimal(value)))
        return value_str

    @staticmethod
    def _get_tailored_item(value: str) -> QtWidgets.QTableWidgetItem:
        item = QtWidgets.QTableWidgetItem(value)
        item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
        font = QFont("Monospace")
        font.setStyleHint(QFont.TypeWriter)
        item.setFont(font)
        item.setFlags(item.flags() & ~Qt.ItemIsEditable)
        return item

    def _update_table_content(self, data: dict):
        stats = data.get("stats")
        if not stats:
            return
        header = data.get("header", [])

        if isinstance(stats, dict):
            self._determine_table_content_by_stats_dict(stats, header)
        elif isinstance(stats, list):
            self._determine_table_content_by_stats_list(stats, header)
        self._resize_all_but_the_last_column_to_contents()

    def _resize_all_but_the_last_column_to_contents(self):
        for column in range(0, self._pane_ui.statisticsTableWidget.columnCount() - 1):
            self._pane_ui.statisticsTableWidget.resizeColumnToContents(column)
