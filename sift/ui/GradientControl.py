import pyqtgraph as pg
from PyQt4 import QtGui, QtCore
import json
import math
import logging

LOG = logging.getLogger(__name__)


class GradientControl(QtGui.QDialog):
    def __init__(self, doc, parent=None, **kwargs):
        super(GradientControl, self).__init__(parent)

        l = QtGui.QGridLayout()
        l.setSpacing(0)
        self.setLayout(l)
        self.doc = doc
        self.gData = {}
        self.autoImportData = {}
        self.ALL_COLORMAPS = self.doc.colormaps
        self.USER_MAPS = self.doc.usermaps

        # Setup Color Bar & clear its data
        self.ColorBar = pg.GradientWidget(orientation='bottom')
        tickList = self.ColorBar.listTicks()
        for tick in tickList:
            self.ColorBar.removeTick(tick[0])
        self.ColorBar.setEnabled(False)

        self.CloneButton = QtGui.QPushButton("Clone Gradient")
        self.CloneButton.clicked.connect(self.cloneGradient)
        self.CloneButton.setEnabled(False)

        # Create Import button
        self.ImportButton = QtGui.QPushButton("Import Gradient")
        self.ImportButton.clicked.connect(self.importButtonClick)

        # Create Gradient List and Related Functions
        self.List = QtGui.QListWidget()
        self.List.setSelectionMode(QtGui.QAbstractItemView.ExtendedSelection)
        self.List.itemSelectionChanged.connect(self.updateColorBar)

        # Create SQRT Button and Related Functions
        self.sqrt = QtGui.QCheckBox("SQRT")
        self.sqrt.stateChanged.connect(self.sqrtAction)
        self.sqrt.setEnabled(False)

        # Create Close button
        self.CloseButton = QtGui.QPushButton("Close")
        self.CloseButton.clicked.connect(self.closeButtonClick)

        # Create Delete Button and Related Functions
        self.DeleteButton = QtGui.QPushButton("Delete Gradient")
        self.DeleteButton.clicked.connect(self.deleteButtonClick)
        self.DeleteButton.setEnabled(False)

        # Create Export Button and Related Functions
        self.ExportButton = QtGui.QPushButton("Export Gradient")
        self.ExportButton.clicked.connect(self.exportButtonClick)
        self.ExportButton.setEnabled(False)

        # Create Save button
        self.SaveButton = QtGui.QPushButton("Save Gradient")
        self.SaveButton.clicked.connect(self.saveButtonClick)
        self.SaveButton.setEnabled(False)

        # Add widgets to their respective spots in the UI grid
        l.addWidget(self.ImportButton, 0, 0)
        l.addWidget(self.SaveButton, 0, 2)
        l.addWidget(self.sqrt, 1, 2)
        l.addWidget(self.ColorBar, 4, 1)
        l.addWidget(self.CloneButton, 1, 0)
        l.addWidget(self.List, 1, 1, 3, 1)
        l.addWidget(self.CloseButton, 6, 2)
        l.addWidget(self.ExportButton, 2, 2)
        l.addWidget(self.DeleteButton, 2, 0)

        # Import custom colormaps
        for cmap in self.USER_MAPS:
            if self.USER_MAPS[cmap].colors and (hasattr(self.USER_MAPS[cmap], "_controls")):
                self.importGradients(cmap, self.USER_MAPS[cmap].colors.hex, self.USER_MAPS[cmap]._controls, True)

        # Import internal colormaps
        for cmap in self.ALL_COLORMAPS:
            if cmap in self.USER_MAPS:
                continue
            if self.ALL_COLORMAPS[cmap].colors and (hasattr(self.ALL_COLORMAPS[cmap], "_controls")):
                self.importGradients(cmap, self.ALL_COLORMAPS[cmap].colors.hex, self.ALL_COLORMAPS[cmap]._controls,
                                     False)

    # Save Custom Gradient
    def saveButtonClick(self):
        name = self.List.item(self.List.currentRow()).text()
        self.gData[name] = self.ColorBar.saveState()
        self.saveNewMap(self.ColorBar.saveState(), name)

    # Clone existing gradient
    def cloneGradient(self):
        text, ok = QtGui.QInputDialog.getText(self, 'Clone Gradient', 'Enter gradient name:')

        if ok:
            SaveName = str(text)
            if SaveName in self.gData:
                overwrite_msg = "There is already a save with this name. Would you like to Overwrite?"
                reply = QtGui.QMessageBox.question(self, 'Message',
                                                   overwrite_msg, QtGui.QMessageBox.Yes, QtGui.QMessageBox.No)

                if reply == QtGui.QMessageBox.Yes:
                    if SaveName in self.autoImportData:
                        QtGui.QMessageBox.information(self, "Error",
                                                      "You cannot save a gradient with the same name as one of the internal gradients.")
                        reply.close()
                        return

                    self.gData[SaveName] = self.ColorBar.saveState()
            else:

                if SaveName in self.autoImportData:
                    QtGui.QMessageBox.information(self, "Error",
                                                  "You cannot save a gradient with the same name as one of the internal gradients.")
                    return

                self.gData[SaveName] = self.ColorBar.saveState()
            self.updateListWidget(SaveName)
            self.saveNewMap(self.ColorBar.saveState(), SaveName)

    # Determine if an internal gradient is selected, returns boolean
    def toRemoveDelete(self):
        toReturn = False

        ListCount = self.List.count()

        index = 0
        while index < ListCount:
            if (self.List.item(index).isSelected()):
                if self.List.item(index).text() in self.autoImportData:
                    toReturn = True
            index = index + 1

        return toReturn

    # Call document function with new gradient
    def saveNewMap(self, UpdatedMap, name):
        self.doc.updateGCColorMap(UpdatedMap, name)

    # Import gtadient
    def importButtonClick(self):
        fname = QtGui.QFileDialog.getOpenFileName(None, 'Get File', 'Export.txt')
        try:
            ifile = open(fname, "r")
            toImport = json.loads(ifile.read())

            for impItem in toImport:
                if impItem in self.autoImportData:
                    QtGui.QMessageBox.information(self, "Error",
                                                  "You cannot save a gradient with the same name as one of the internal gradients.")
                    return

            for impItem in toImport:
                self.saveNewMap(toImport[impItem], impItem)

            self.gData.update(toImport)

            self.updateListWidget(impItem)
        except IOError:
            LOG.error("Error opening file or reading!")

    # Import a gradient into either the internal or custom gradient lists
    def importGradients(self, name, hex, floats, editable):
        try:
            data = {}

            for i in range(len(hex)):
                data[floats.astype(float)[i]] = hex[i]

            newWidget = pg.GradientWidget()
            newWidget.hide()

            tickList = newWidget.listTicks()
            for tick in tickList:
                newWidget.removeTick(tick[0])

            for key in data:
                newWidget.addTick(key, QtGui.QColor(data[key]), movable=editable)

            if editable:
                self.gData[name] = newWidget.saveState()
            else:
                self.autoImportData[name] = newWidget.saveState()

            self.updateListWidget()

        except AssertionError as e:
            LOG.error(e)

    # Close the widget
    def closeButtonClick(self):
        self.close()

    # Update list widget with new gradient list
    def updateListWidget(self, toShow=None):
        self.List.clear()

        totalCount = 0
        corVal = 0
        for key2 in self.autoImportData:
            self.List.addItem(key2)
            totalCount = totalCount + 1
            if toShow is not None and key2 == toShow:
                corVal = totalCount

        self.List.addItem("----------------------------- Below Are Custom ColorMaps -----------------------------")
        barrier_item = self.List.item(totalCount)
        barrier_item.setFlags(QtCore.Qt.NoItemFlags)
        totalCount = totalCount + 1

        for key in self.gData:
            self.List.addItem(key)
            totalCount = totalCount + 1
            if toShow is not None and key == toShow:
                corVal = totalCount

        if toShow is not None:
            self.List.setCurrentRow(corVal, QtGui.QItemSelectionModel.Select)

    # Update the colorbar with the newly selected gradient
    def updateColorBar(self):
        self.sqrt.setCheckState(False)

        if self.List.item(self.List.currentRow()).text() in self.gData:
            NewBar = self.gData[self.List.item(self.List.currentRow()).text()]
            self.ColorBar.restoreState(NewBar)

        if self.List.item(self.List.currentRow()).text() in self.autoImportData:
            NewBar = self.autoImportData[self.List.item(self.List.currentRow()).text()]
            self.ColorBar.restoreState(NewBar)

        SelectedThings = self.List.selectedItems()

        # Bunch of functions determing which buttons to enable / disable

        showDel = True
        for thing in SelectedThings:
            if thing.text() in self.autoImportData:
                showDel = False

        if len(SelectedThings) > 1:
            self.SaveButton.setEnabled(False)
            self.sqrt.setEnabled(False)
            self.CloneButton.setEnabled(False)

            tickList = self.ColorBar.listTicks()
            for tick in tickList:
                self.ColorBar.removeTick(tick[0])
            self.ColorBar.setEnabled(False)
        elif len(SelectedThings) == 1 and showDel:
            self.ColorBar.setEnabled(True)
            self.DeleteButton.setEnabled(True)
            self.sqrt.setEnabled(True)
            self.CloneButton.setEnabled(True)
            self.SaveButton.setEnabled(True)
            self.ExportButton.setEnabled(True)
        elif len(SelectedThings) == 1 and not showDel:
            self.CloneButton.setEnabled(True)
            self.ExportButton.setEnabled(True)
            self.DeleteButton.setEnabled(False)
            self.sqrt.setEnabled(False)
            self.SaveButton.setEnabled(False)

        if not showDel:
            self.DeleteButton.setEnabled(False)

    # If square root button is checked/unchecked, modify the ticks as such
    def sqrtAction(self):
        if self.sqrt.isChecked():
            tickList = self.ColorBar.listTicks()
            for tick in tickList:
                self.ColorBar.setTickValue(tick[0], math.sqrt(self.ColorBar.tickValue(tick[0])))
        else:
            tickList = self.ColorBar.listTicks()
            for tick in tickList:
                self.ColorBar.setTickValue(tick[0], self.ColorBar.tickValue(tick[0]) * self.ColorBar.tickValue(tick[0]))

    # Delete gradient(s)
    def deleteButtonClick(self):
        block = self.toRemoveDelete()
        if block is True:
            # This shouldn't happen
            QtGui.QMessageBox.information(self, "Error: Can not delete internal gradients.")
            return

        selectedGradients = self.List.selectedItems()

        toPrint = ",".join([x.text() for x in selectedGradients])

        delete_msg = "Please confirm you want to delete the Gradient(s): " + toPrint
        reply = QtGui.QMessageBox.question(self, 'Message',
                                           delete_msg, QtGui.QMessageBox.Yes, QtGui.QMessageBox.No)
        if reply == QtGui.QMessageBox.Yes:
            for index in selectedGradients:
                del self.gData[index.text()]
                self.doc.removeGCColorMap(index.text())
            self.updateListWidget()

    # Export gradient(s)
    def exportButtonClick(self):
        selectedGradients = self.List.selectedItems()
        fname = QtGui.QFileDialog.getSaveFileName(None, 'Save As', 'Export.json')
        toExport = set()
        for index in selectedGradients:
            toExport.add(index.text())
        done = {}

        for k in self.gData:
            if k in toExport:
                done[k] = self.gData[k]

        for k in self.autoImportData:
            if k in toExport:
                done[k] = self.autoImportData[k]
        try:
            file = open(fname, 'w')
            file.write(str(done))
        except IOException:
            LOG.error("Error opening or writing!")


def main():
    app = QtGui.QApplication([])
    w = GradientControl()
    w.show()
    app.exec_()
    return 0


if __name__ == '__main__':
    sys.exit(main())
