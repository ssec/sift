from __future__ import annotations

import sys
from datetime import datetime
from typing import Callable, List, Optional

import numpy as np
from dateutil.relativedelta import relativedelta
from PyQt5.QtCore import (
    QAbstractListModel,
    QDateTime,
    QModelIndex,
    QObject,
    Qt,
    pyqtProperty,
    pyqtSignal,
    pyqtSlot,
)

from uwsift.common import DEFAULT_TIME_FORMAT

INVALID_QModelIndex = QModelIndex()


class QmlLayerManager(QObject):
    layerToDisplayChanged = pyqtSignal()
    layerModelChanged = pyqtSignal()
    convFuncModelChanged = pyqtSignal()
    dateToDisplayChanged = pyqtSignal(str)
    testChanged = pyqtSignal(int)

    def __init__(self) -> None:
        super().__init__()
        self._layer_to_display = ""
        self._date_to_display = None

        self._layer_model: None | LayerModel = None

        self._convenience_functions: dict[str, Callable] = {
            "Most Frequent": self.get_most_frequent_data_layer_index,
        }
        self.convFuncModel = LayerModel(layer_strings=list(self._convenience_functions.keys()))  # type: ignore

        # TODO(mk): make this configurable if the user wants to display dates differently?
        self._format_str = DEFAULT_TIME_FORMAT
        self._qml_layer_model: LayerModel = LayerModel(layer_strings=["No Layers loaded."])

    def get_most_frequent_data_layer_index(self) -> int:
        """
        Get index of the data layer with the least mean difference between timestamps.
        :return: -1 if no data layers exist, index of most frequent data layer otherwise.
        """
        if self._layer_model is None:
            raise RuntimeError("Layer model has not been connected to layer manager yet")
        dynamic_layers = self._layer_model.get_dynamic_layers()
        num_data_layers = len(dynamic_layers)
        if num_data_layers == 0:
            return -1
        else:
            temporal_differences = np.zeros(num_data_layers)
            for i, dynamic_layer in enumerate(dynamic_layers):
                tl = list(dynamic_layer.timeline.keys())
                t_diffs = [tl[i + 1] - tl[i] for i in range(len(tl) - 1)]
                # Calculate mean difference in timeline in seconds to support
                # comparing timelines with non-regularly occurring
                # timestamps. Timelines with zero or one entries (=> t_diffs
                # is empty) are not suitable, assign the largest possible mean
                # value for them!
                temporal_differences[i] = (
                    sys.float_info.max if not t_diffs else np.mean(list(map(lambda td: td.total_seconds(), t_diffs)))
                )
            most_frequent_data_layer_idx = np.argmin(temporal_differences)
            if not isinstance(most_frequent_data_layer_idx, np.int64):
                # If there exist multiple timelines at the same sampling rate,
                # take the first one.
                # TODO: better: take one with most datasets (= 'longest'
                #  timeline)
                most_frequent_data_layer_idx = most_frequent_data_layer_idx[0]
            return most_frequent_data_layer_idx

    def get_convenience_function(self, function_name: str) -> Optional[Callable]:
        conv_fun = self._convenience_functions.get(function_name, None)
        return conv_fun

    @pyqtProperty(QObject, notify=layerModelChanged)
    def layerModel(self):
        return self._qml_layer_model

    @layerModel.setter  # type: ignore # mypy bug #9911
    def layerModel(self, new_model):
        self._qml_layer_model = new_model
        self.layerModelChanged.emit()

    @pyqtProperty(QObject, notify=convFuncModelChanged)
    def convFuncModel(self):
        return self._conv_func_model

    @convFuncModel.setter  # type: ignore # mypy bug #9911
    def convFuncModel(self, new_model):
        self._conv_func_model = new_model
        self.convFuncModelChanged.emit()

    # TODO(mk): this only works for SEVIRI data like this, make this more general!
    @staticmethod
    def format_product_family_key(product_family_key):
        return product_family_key[0].name + "-" + product_family_key[1].name + "-" + product_family_key[2]

    # number = pyqtProperty(int, get_number, notify=numberChanged)
    # Define the getter of the 'name' property.  The C++ type of the
    # property is QString which Python will convert to and from a string.
    @pyqtProperty(str, notify=layerToDisplayChanged)
    def layerToDisplay(self):
        return self._layer_to_display

    # Define the setter of the 'name' property.
    @layerToDisplay.setter  # type: ignore # mypy bug #9911
    def layerToDisplay(self, data_layer_str):
        self._layer_to_display = data_layer_str
        self.layerToDisplayChanged.emit()

    @pyqtProperty(str, notify=dateToDisplayChanged)
    def dateToDisplay(self):
        if not self._date_to_display:
            return self._format_str.strip("%")
        else:
            return self._date_to_display

    @dateToDisplay.setter  # type: ignore # mypy bug #9911
    def dateToDisplay(self, new_date):
        self._date_to_display = new_date.strftime(self._format_str)
        self.dateToDisplayChanged.emit(self._date_to_display)


class LayerModel(QAbstractListModel):
    modelChanged = pyqtSignal()
    pushedOrPopped = pyqtSignal()

    def __init__(self, *args, layer_strings=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._layer_strings = layer_strings if layer_strings is not None else []

    @pyqtProperty("QVariantList", notify=modelChanged)
    def model(self):
        return self._layer_strings

    def push(self, data):
        data_size = len(data)
        if data_size > 1:
            start_idx = self.rowCount()
            end_idx = start_idx + data_size
            self.beginInsertRows(QModelIndex(), start_idx, end_idx)
            self._layer_strings.extend(data)
            self.endInsertRows()
        else:
            self.beginInsertRows(QModelIndex(), self.rowCount(), self.rowCount())
            self._layer_strings.extend(data)
            self.endInsertRows()
        self.pushedOrPopped.emit()

    def pop(self, row):
        self.beginRemoveColumns(QModelIndex(), row, row)
        del self._layer_strings[row]
        self.endRemoveRows()
        self.pushedOrPopped.emit()

    def clear(self):
        for row, _ in enumerate(self._layer_strings):
            self.pop(row)

    @property
    def layer_strings(self):
        return self._layer_strings

    @layer_strings.setter
    def layer_strings(self, new_layer_strings):
        self.layoutAboutToBeChanged.emit()
        self._layer_strings = new_layer_strings
        # upd. persistent indexes
        from_index_list = self.persistentIndexList()
        to_index_list = []
        for i, _ in enumerate(new_layer_strings):
            to_index_list.append(self.index(i, parent=QModelIndex()))
        self.changePersistentIndexList(from_index_list, to_index_list)
        self.layoutChanged.emit()
        # TODO(mk): rename this signal to indicate that the list of combo box options changed
        self.pushedOrPopped.emit()

    def data(self, index, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            return self._layer_strings[index.row()]

    def rowCount(self, parent: QModelIndex = INVALID_QModelIndex) -> int:  # noqa
        return len(self._layer_strings)


class TimebaseModel(QAbstractListModel):
    modelChanged = pyqtSignal()
    timebaseChanged = pyqtSignal()
    currentTimestampChanged = pyqtSignal(str)
    returnToInitialState = pyqtSignal()

    def __init__(self, *args, timestamps: None | List[QDateTime], **kwargs):
        super().__init__(*args, **kwargs)
        self._format_str = DEFAULT_TIME_FORMAT
        self.timestamps = timestamps  # sets self._timestamps

        self._current_timestamp = self._format_str.replace("%", "")

    @pyqtProperty("QVariantList", notify=modelChanged)
    def model(self):
        return self._timestamps

    def data(self, index, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            return self._timestamps[index.row()]

    def rowCount(self, parent: QModelIndex = INVALID_QModelIndex) -> int:  # noqa
        return len(self._timestamps)

    @pyqtSlot(int, result=QDateTime)
    def at(self, i):
        return self._timestamps[i]

    @pyqtProperty(str, notify=currentTimestampChanged)
    def currentTimestamp(self):
        return self._current_timestamp

    @currentTimestamp.setter  # type: ignore # mypy bug #9911
    def currentTimestamp(self, new_date):
        self._current_timestamp = new_date.strftime(self._format_str) if new_date else self._format_str.replace("%", "")
        self.currentTimestampChanged.emit(self._current_timestamp)

    @property
    def timestamps(self):
        return self._timestamps

    @timestamps.setter
    def timestamps(self, timestamps):
        self.layoutAboutToBeChanged.emit()
        self._timestamps = timestamps if timestamps else self._get_default_qdts()
        # upd. persistent indexes
        from_index_list = self.persistentIndexList()
        to_index_list = []
        for i, _ in enumerate(self._timestamps):
            to_index_list.append(self.index(i, parent=QModelIndex()))
        self.changePersistentIndexList(from_index_list, to_index_list)
        self.layoutChanged.emit()
        self.timebaseChanged.emit()

    def clear(self):
        """Set the two corresponding properties which has to do something with the timestamps to None.

        The purpose of this is to give a possibility to return to a similar state like
        when TimeBaseModel was initialized."""
        self.timestamps = None
        self.currentTimestamp = None
        self.returnToInitialState.emit()

    @staticmethod
    def _get_default_qdts(steps=5):
        """Generate default QDateTime steps.

        If no data is loaded, there is no way to specify a time interval for
        the data. Create some dummy time steps as a substitute.

        These QDateTime steps are not meant to be drawn with time markers for
        data. They only assist drawing the QML timeline.

        :param steps: Number of QDateTime objects to be generated
        :return: List of generated QDateTime objects

        """
        now_dt = datetime.utcnow()
        now_dt = datetime(now_dt.year, now_dt.month, now_dt.day, now_dt.hour)
        return list(map(lambda dt: QDateTime(dt), [now_dt + relativedelta(hours=i) for i in range(steps)]))


class QmlBackend(QObject):
    doRefreshTimeline = pyqtSignal()
    doLoadTimeline = pyqtSignal()
    doClearTimeline = pyqtSignal()
    doNotifyTimelineIndexChanged = pyqtSignal(int)
    didJumpInTimeline = pyqtSignal(int)
    # The timebase of the animation/display might be a driving layer or a timeline generated
    # according to some policy. didChangeTimebase is emitted when i.e. the driving layer is supposed
    # to change.
    didChangeTimebase = pyqtSignal(int)

    def __init__(self) -> None:
        super().__init__()
        self.qml_layer_manager: QmlLayerManager | None = None

    def clear_timeline(self):
        self.doClearTimeline.emit()

    def load_timeline(self):
        self.doLoadTimeline.emit()

    def refresh_timeline(self):
        self.doRefreshTimeline.emit()

    @pyqtSlot(int)
    def clickTimelineAtIndex(self, idx):
        """
        Slot that gets triggered if the user clicks the timeline at a position. The index
        corresponding to that position is passed to this slot which then forwards it via
        emittance of the didJumpInTimeline signal.
        :param idx: Timebase index of the clicked location on the UI timeline
        """
        self.didJumpInTimeline.emit(idx)

    @pyqtSlot(int)
    def clickComboBoxAtIndex(self, idx):
        """
        Slot triggered by the user clicking an entry in the timebase selection Combo Box. A timebase
        change is then triggered by emitting the didChangeTimebase signal with the corresponding
        index.
        :param idx: Timebase index of the clicked combo box item
        """
        self.didChangeTimebase.emit(idx)

    @pyqtSlot(str)
    def clickConvFuncMenu(self, conv_func_name):
        """
        Slot triggered by the user clicking an entry in the convenience function popup menu.
        A convenience function is then selected by its index in the popup menu and executed. The
        resulting data_layer_index is then emitted via the didChangeTimebase signal.
        :param conv_func_name: Name of the clicked convenience function as a string
        """
        if self.qml_layer_manager is not None:
            data_layer_index = self.qml_layer_manager.get_convenience_function(conv_func_name)()
            if data_layer_index >= 0:
                self.didChangeTimebase.emit(data_layer_index)
