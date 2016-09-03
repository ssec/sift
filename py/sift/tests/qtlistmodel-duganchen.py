#!/usr/bin/env python

from sip import setapi

setapi("QDate", 2)
setapi("QDateTime", 2)
setapi("QTextStream", 2)
setapi("QTime", 2)
setapi("QVariant", 2)
setapi("QString", 2)
setapi("QUrl", 2)


from PyQt4.QtCore import QAbstractListModel, QMimeData, QModelIndex, Qt
from PyQt4.QtGui import QApplication, QListView, QMainWindow
import sys


def main():
    app = QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())


class MainWindow(QMainWindow):

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        view = QListView()
        view.setDragEnabled(True)
        view.setAcceptDrops(True)
        view.setDropIndicatorShown(True)
        view.setSelectionMode(view.ExtendedSelection)
        self.model = ListModel(view)
        view.setModel(self.model)
        self.setCentralWidget(view)


class ListModel(QAbstractListModel):

    Mimetype = 'application/vnd.row.list'

    def __init__(self, parent=None):
        super(ListModel, self).__init__(parent)
        self.__data = ['line 1', 'line 2', 'line 3', 'line 4']

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None

        if index.row() > len(self.__data):
            return None

        if role == Qt.DisplayRole or role == Qt.EditRole:
            return self.__data[index.row()]

        return None

    def dropMimeData(self, data, action, row, column, parent):
        if action == Qt.IgnoreAction:
            return True
        if not data.hasFormat(self.Mimetype):
            return False
        if column > 0:
            return False

        strings = str(data.data(self.Mimetype)).split('\n')
        self.insertRows(row, len(strings))
        for i, text in enumerate(strings):
            self.setData(self.index(row + i, 0), text)

        return True

    def flags(self, index):
        flags = super(ListModel, self).flags(index)

        if index.isValid():
            flags |= Qt.ItemIsEditable
            flags |= Qt.ItemIsDragEnabled
        else:
            flags = Qt.ItemIsDropEnabled

        return flags

    def insertRows(self, row, count, parent=QModelIndex()):

        self.beginInsertRows(QModelIndex(), row, row + count - 1)
        self.__data[row:row] = [''] * count
        self.endInsertRows()
        return True

    def mimeData(self, indexes):
        sortedIndexes = sorted([index for index in indexes
            if index.isValid()], key=lambda index: index.row())
        encodedData = '\n'.join(self.data(index, Qt.DisplayRole)
                for index in sortedIndexes)
        mimeData = QMimeData()
        mimeData.setData(self.Mimetype, encodedData)
        return mimeData

    def mimeTypes(self):
        return [self.Mimetype]

    def removeRows(self, row, count, parent=QModelIndex()):
        self.beginRemoveRows(QModelIndex(), row, row + count - 1)
        del self.__data[row:row + count]
        self.endRemoveRows()
        return True

    def rowCount(self, parent=QModelIndex()):
        return len(self.__data)

    def setData(self, index, value, role=Qt.EditRole):
        if not index.isValid() or role != Qt.EditRole:
            return False

        self.__data[index.row()] = value
        self.dataChanged.emit(index, index)
        return True

    def supportedDropActions(self):
        return Qt.MoveAction


if __name__ == '__main__':
    main()