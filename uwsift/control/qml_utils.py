from PyQt5.QtCore import QObject, pyqtSignal, pyqtProperty, pyqtSlot, QAbstractListModel
from PyQt5.QtCore import QStringListModel, Qt, QModelIndex, QVariant


class QmlLayerManager(QObject):
    def __init__(self):
        super().__init__()
        self._layer_to_display = ""
        self._date_to_display = None
        self._test = 0
        # TODO(mk): make this configurable if the user wants to display dates differently?
        self._format_str = "%Y-%m-%d %H:%M%Z"
        self.testModel = QStringListModel()

    layerToDisplayChanged = pyqtSignal()
    dateToDisplayChanged = pyqtSignal(str)
    testChanged = pyqtSignal(int)

    # number = pyqtProperty(int, get_number, notify=numberChanged)
    # Define the getter of the 'name' property.  The C++ type of the
    # property is QString which Python will convert to and from a string.
    @pyqtProperty(str, notify=layerToDisplayChanged)
    def layerToDisplay(self):
        return self._layer_to_display

    # Define the setter of the 'name' property.
    @layerToDisplay.setter
    def layerToDisplay(self, driving_layer_pfkey):
        layer_name = driving_layer_pfkey[0].name + "-" + driving_layer_pfkey[1].name + "-" +\
                     driving_layer_pfkey[2].split(":")[0]
        self._layer_to_display = layer_name
        self.layerToDisplayChanged.emit()

    @pyqtProperty(str, notify=dateToDisplayChanged)
    def dateToDisplay(self):
        if not self._date_to_display:
            return self._format_str.strip("%")
        else:
            return self._date_to_display
            #date_str = self._date_to_display.strftime(self._format_str)
            #return date_str

    @dateToDisplay.setter
    def dateToDisplay(self, new_date):
        self._date_to_display = new_date.strftime(self._format_str)
        self.dateToDisplayChanged.emit(self._date_to_display)

    @pyqtProperty(int, notify=testChanged)
    def test(self):
        return self._test

    @test.setter
    def test(self, val):
        self._test = val
        self.testChanged.emit(self._test)


class QmlTimelineManager(QObject):
    def __init__(self):
        super().__init__()
        self.timeStampQStringModel = QStringListModel()

    def update(self, timeline, format_str="%d %h %Y %H:%M"):
        timeline_string_iterable = map(lambda dt: dt.strftime(format_str), timeline)
        self.timeStampQStringModel.setStringList(timeline_string_iterable)


# class NumberGenerator(QObject):
#
#     def __init__(self):
#         super().__init__()
#         self.__number = 42
#         self.__max_number = 99
#
#     maxNumberChanged = pyqtSignal()
#     numberChanged = pyqtSignal(int)
#
#     #number = pyqtProperty(int, get_number, notify=numberChanged)
#     # Define the getter of the 'name' property.  The C++ type of the
#     # property is QString which Python will convert to and from a string.
#     @pyqtProperty(int, notify=numberChanged)
#     def number(self):
#         return self.__number
#
#     # Define the setter of the 'name' property.
#     @number.setter
#     def number(self, val):
#         if self.__number != val:
#             self.__number = val
#             self.numberChanged.emit(self.__number)
#
#     @pyqtProperty(int, notify=maxNumberChanged)
#     def maxNumber(self):
#         return self.__max_number
#
#     @maxNumber.setter
#     def maxNumber(self, val):
#         if val < 0:
#             val = 0
#
#         if self.__max_number != val:
#             self.__max_number = val
#             self.maxNumberChanged.emit()
#
#         if self.__number > self.__max_number:
#             self.number = self.__max_number
#
#     @pyqtSlot()
#     def updateNumber(self):
#         new_num = randint(0, self.__max_number)
#         self.number = new_num
#
#     @pyqtSlot(int)
#     def updateMaxNumber(self, val):
#         self.maxNumber = val

class MyModel(QObject):
    modelChanged = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__()
        self.propertyList = [1, 2, 3]

    # def model(self):
    #     return self.propertyList
    #model = QtCore.Property("QVariantList", fget=model, notify=modelChanged)

    @pyqtProperty(int, notify=modelChanged)
    def model(self):
        return self.propertyList


class MyTestModel2(QAbstractListModel):
    modelChanged = pyqtSignal()

    def __init__(self, *args, vals=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.vals = vals or []

    # @property
    # def vals(self):
    #     return self.vals
    # @vals.setter
    # def vals(self, new_vals):
    #     self.vals = new_vals
    #     onDataChanged.emit()

    @pyqtProperty("QVariantList", notify=modelChanged)
    def model(self):
        return self.vals

    def data(self, index, role):
        if role == Qt.DisplayRole:
            #status, text
            text = self.vals[index.row()]
            return text

    def rowCount(self, index):
        return len(self.vals)


class QmlBackend(QObject):

    # doAddPin = pyqtSignal()
    doRefreshTimeline = pyqtSignal()

    def __init__(self):
        super().__init__()

    def refresh_timeline(self):
        self.doRefreshTimeline.emit()


        # class Backend: public
        #
        # QObject
        # {
        #     signals:
        #         void doAddPin(float latitude, float longitude, QString name, QString address);
        #
        # ........
        #     void
        # callAddPinInQML()
        # {
        #     emit
        # doAddPin(12.34, 45.67, "hello", "world");
        # }
        # }