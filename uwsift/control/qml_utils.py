from typing import List

from uwsift.common import DEFAULT_TIME_FORMAT

from PyQt5.QtCore import QObject, pyqtSignal, pyqtProperty, pyqtSlot, QAbstractListModel
from PyQt5.QtCore import Qt, QModelIndex, QDateTime


class QmlLayerManager(QObject):

    layerToDisplayChanged = pyqtSignal()
    layerModelChanged = pyqtSignal()
    convFuncModelChanged = pyqtSignal()
    dateToDisplayChanged = pyqtSignal(str)
    testChanged = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self._layer_to_display = ""
        self._date_to_display = None

        self._convenience_functions = {}
        self._conv_func_model: LayerModel = LayerModel(layer_strings=[])
        self.convenience_functions = {}

        # TODO(mk): make this configurable if the user wants to display dates differently?
        self._format_str = DEFAULT_TIME_FORMAT
        self._layer_model: LayerModel = LayerModel(layer_strings=["No Layers loaded."])

        # self._conv_func_model: LayerModel = LayerModel(
        #     layer_strings=list(self._convenience_functions.keys()))

    @property
    def convenience_functions(self):
        return self._convenience_functions

    @convenience_functions.setter
    def convenience_functions(self, conv_funcs):
        self._convenience_functions = conv_funcs
        self.convFuncModel = LayerModel(layer_strings=list(conv_funcs.keys()))

    @pyqtProperty(QObject, notify=layerModelChanged)
    def layerModel(self):
        return self._layer_model

    @layerModel.setter
    def layerModel(self, new_model):
        self._layer_model = new_model
        self.layerModelChanged.emit()

    @pyqtProperty(QObject, notify=convFuncModelChanged)
    def convFuncModel(self):
        return self._conv_func_model

    @convFuncModel.setter
    def convFuncModel(self, new_model):
        self._conv_func_model = new_model
        self.convFuncModelChanged.emit()

    # TODO(mk): deprecated methods, remove?
    def change_layer_model(self, idx, elem):
        insert_idx = idx + self.num_convenience_functions
        if insert_idx < len(self._layer_model._layer_strings):
            self._layer_model._layer_strings[insert_idx] = elem
        else:
            self._layer_model._layer_strings.append(elem)
        for el in self._layer_model._layer_strings:
            print(f"{el}")
        self.layerModelChanged.emit()

    def clear_layer_model(self):
        topIdx = len(self._layer_model._layer_strings) - 1
        while topIdx > self.num_convenience_functions:
            del self._layer_model._layer_strings[topIdx]
            topIdx -= 1
        self.layerModelChanged.emit()



    # TODO(mk): this only works for SEVIRI data like this, make this more general!
    @staticmethod
    def format_product_family_key(product_family_key):
        return product_family_key[0].name + "-" + product_family_key[1].name + "-" + \
               product_family_key[2]

    # number = pyqtProperty(int, get_number, notify=numberChanged)
    # Define the getter of the 'name' property.  The C++ type of the
    # property is QString which Python will convert to and from a string.
    @pyqtProperty(str, notify=layerToDisplayChanged)
    def layerToDisplay(self):
        return self._layer_to_display

    # Define the setter of the 'name' property.
    @layerToDisplay.setter
    def layerToDisplay(self, data_layer_str):
        self._layer_to_display = data_layer_str
        self.layerToDisplayChanged.emit()

    @pyqtProperty(str, notify=dateToDisplayChanged)
    def dateToDisplay(self):
        if not self._date_to_display:
            return self._format_str.strip("%")
        else:
            return self._date_to_display

    @dateToDisplay.setter
    def dateToDisplay(self, new_date):
        self._date_to_display = new_date.strftime(self._format_str)
        self.dateToDisplayChanged.emit(self._date_to_display)


class LayerModel(QAbstractListModel):
    modelChanged = pyqtSignal()
    pushedOrPopped = pyqtSignal()

    def __init__(self, *args, layer_strings=[], **kwargs):
        super().__init__(*args, **kwargs)
        self._layer_strings = layer_strings

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

    def rowCount(self, parent=QModelIndex()):
        return len(self._layer_strings)


class TimebaseModel(QAbstractListModel):
    modelChanged = pyqtSignal()
    timebaseChanged = pyqtSignal()
    currentTimestampChanged = pyqtSignal(str)

    def __init__(self, *args, timestamps: List[QDateTime] = None, **kwargs):
        super().__init__(*args, **kwargs)
        if timestamps is None:
            timestamps = []
        self._timestamps = None
        self.timestamps = timestamps
        self._current_timestamp = None
        self._format_str = DEFAULT_TIME_FORMAT

    @pyqtProperty("QVariantList", notify=modelChanged)
    def model(self):
        return self._timestamps

    def data(self, index, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            return self._timestamps[index.row()]

    def rowCount(self, parent=QModelIndex()):
        return len(self._timestamps)

    @pyqtSlot(int, result=QDateTime)
    def at(self, i):
        return self._timestamps[i]

    @pyqtProperty(str, notify=currentTimestampChanged)
    def currentTimestamp(self):
        if not self._current_timestamp:
            return self._format_str.strip("%")
        else:
            return self._current_timestamp

    @currentTimestamp.setter
    def currentTimestamp(self, new_date):
        self._current_timestamp = new_date.strftime(self._format_str)
        self.currentTimestampChanged.emit(self._current_timestamp)

    @property
    def timestamps(self):
        return self._timestamps

    @timestamps.setter
    def timestamps(self, new_timetamps):
        self.layoutAboutToBeChanged.emit()
        self._timestamps = new_timetamps
        # upd. persistent indexes
        from_index_list = self.persistentIndexList()
        to_index_list = []
        for i, _ in enumerate(new_timetamps):
            to_index_list.append(self.index(i, parent=QModelIndex()))
        self.changePersistentIndexList(from_index_list, to_index_list)
        self.layoutChanged.emit()
        self.timebaseChanged.emit()


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

    def __init__(self):
        super().__init__()
        self.qml_layer_manager: QmlLayerManager = None

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
        A convenience function is then selected by it's index in the popup menu and executed. The
        resulting data_layer_index is then emitted via the didChangeTimebase signal.
        :param conv_func_name: Name of the clicked convenience function as a string
        """
        if self.qml_layer_manager:
            data_layer_index = self.qml_layer_manager.convenience_functions[conv_func_name]()
            if data_layer_index >= 0:
                self.didChangeTimebase.emit(data_layer_index)
