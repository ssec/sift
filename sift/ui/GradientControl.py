import pyqtgraph as pg
from PyQt4 import QtGui, QtCore
import json
import os
import math
import numpy
# from ipywidgets import *
# from tkinter import *
import ast
from colormap import rgb2hex
from vispy.color.colormap import Colormap, BaseColormap, _mix_simple, _colormaps



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
        l.addWidget(self.List, 1, 1,3,1)
        l.addWidget(self.CloseButton, 6, 2)
        l.addWidget(self.ExportButton, 2, 2)
        l.addWidget(self.DeleteButton, 2, 0)



        # Import custom colormaps
        for map in self.USER_MAPS:
            if self.USER_MAPS[map].colors and (hasattr(self.USER_MAPS[map], "_controls")):
                self.importGradients(map, self.USER_MAPS[map].colors.hex, self.USER_MAPS[map]._controls, True)


        # Import internal colormaps
        for map in self.ALL_COLORMAPS:
            if self.ALL_COLORMAPS[map].colors and (hasattr(self.ALL_COLORMAPS[map], "_controls")):
                if map not in self.USER_MAPS:
                    self.importGradients(map, self.ALL_COLORMAPS[map].colors.hex, self.ALL_COLORMAPS[map]._controls, False)

    # Save Custom Gradient
    def saveButtonClick(self):
        name = self.List.item(self.List.currentRow()).text()
        self.gData[name] = self.ColorBar.saveState()
        self.saveNewMap(self.ColorBar.saveState(), name)


    # Clone existing gradient, make UI box
    def cloneGradient(self):
        self.p = QtGui.QWidget()
        self.p.setWindowTitle('Save Gradient As:')
        self.p.textbox = QtGui.QLineEdit(self.p)
        self.p.textbox.move(20, 20)
        self.p.textbox.resize(280, 40)
        self.p.resize(320, 150)
        button = QtGui.QPushButton('Save', self.p)
        button.move(20, 80)
        button.clicked.connect(self.cloneGradient2)
        self.p.setWindowModality(QtCore.Qt.WindowModal)
        self.p.show()

    # Clone existing gradient, clone the gradient
    def cloneGradient2(self):
        SaveName = self.p.textbox.text()
        if SaveName in self.gData.keys():
            overwrite_msg = "There is already a save with this name. Would you like to Overwrite?"
            reply = QtGui.QMessageBox.question(self, 'Message',
                                               overwrite_msg, QtGui.QMessageBox.Yes, QtGui.QMessageBox.No)

            if reply == QtGui.QMessageBox.Yes:
                if SaveName in self.autoImportData.keys():
                    QtGui.QMessageBox.information(self, "Error",
                                                  "You cannot save a gradient with the same name as one of the internal gradients.")
                    self.p.close()
                    reply.done(1)
                    return

                self.gData[SaveName] = self.ColorBar.saveState()
                self.p.close()
        else:

            if SaveName in self.autoImportData.keys():
                QtGui.QMessageBox.information(self, "Error",
                                              "You cannot save a gradient with the same name as one of the internal gradients.")
                self.p.close()
                return

            self.gData[SaveName] = self.ColorBar.saveState()
        self.updateListWidget(SaveName)
        self.saveNewMap(self.ColorBar.saveState(), SaveName)
        self.p.close()


    # Return list of selected gradients
    def getSelected(self):
        toReturn = []

        ListCount = self.List.count()

        index = 0
        while index < ListCount:
            if (self.List.item(index).isSelected()):
                toReturn.append(self.List.item(index))
            index = index + 1

        return toReturn

    # Determine if an internal gradient is selected, returns boolean
    def toRemoveDelete(self):
        toReturn = False

        ListCount = self.List.count()

        index = 0
        while index < ListCount:
            if (self.List.item(index).isSelected()):
                if self.List.item(index).text() in self.autoImportData.keys():
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
            file = open(fname, "r")
            toImport = ast.literal_eval(file.read())

            for impItem in toImport.keys():
                if impItem in self.autoImportData.keys():
                    QtGui.QMessageBox.information(self, "Error",
                                                  "You cannot save a gradient with the same name as one of the internal gradients.")
                    return

            for impItem in toImport.keys():
                self.saveNewMap(toImport[impItem], impItem)

            self.gData.update(toImport)

            self.updateListWidget(impItem)
        except:
            print("Error opening file or reading!")

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

        except Exception as e:
            print(e)

    # Close the widget
    def closeButtonClick(self):
        self.done(0)

    # Update list widget with new gradient list
    def updateListWidget(self, toShow = None):
        self.List.clear()

        totalCount = 0
        corVal = 0
        for key2 in self.autoImportData.keys():
            self.List.addItem(key2)
            totalCount = totalCount + 1
            if toShow is not None and key2 == toShow:
                corVal = totalCount

        self.List.addItem("----------------------------- Below Are Custom ColorMaps -----------------------------")
        totalCount = totalCount + 1

        for key in self.gData.keys():
            self.List.addItem(key)
            totalCount = totalCount + 1
            if toShow is not None and key == toShow:
                corVal = totalCount

        if toShow is not None:
            self.List.setCurrentRow(corVal, QtGui.QItemSelectionModel.Select)


    # Update the colorbar with the newly selected gradient
    def updateColorBar(self):
        self.sqrt.setCheckState(0)

        if self.List.item(self.List.currentRow()).text() in self.gData.keys():
            NewBar = self.gData[self.List.item(self.List.currentRow()).text()]
            self.ColorBar.restoreState(NewBar)

        if self.List.item(self.List.currentRow()).text() in self.autoImportData.keys():
            NewBar = self.autoImportData[self.List.item(self.List.currentRow()).text()]
            self.ColorBar.restoreState(NewBar)

        SelectedThings = self.getSelected()

        # Bunch of functions determing which buttons to enable / disable

        if len(SelectedThings) > 1:
            self.SaveButton.setEnabled(False)
            self.sqrt.setEnabled(False)
            self.CloneButton.setEnabled(False)

            tickList = self.ColorBar.listTicks()
            for tick in tickList:
                self.ColorBar.removeTick(tick[0])
            self.ColorBar.setEnabled(False)
        elif len(SelectedThings) == 1:
            self.SaveButton.setEnabled(True)
            self.sqrt.setEnabled(True)
            self.CloneButton.setEnabled(True)
            self.ColorBar.setEnabled(True)

        showDel = True
        for thing in SelectedThings:
            print(thing.text())
            if thing.text() in self.autoImportData.keys():
                showDel = False

        # Disable buttons if internal gradient is selected
        if showDel is False:
            self.DeleteButton.setEnabled(False)
            self.sqrt.setEnabled(False)
            self.SaveButton.setEnabled(False)

        # Disable buttons if divider is selected
        if len(SelectedThings) > 0 and SelectedThings[0].text() == "----------------------------- Below Are Custom ColorMaps -----------------------------":
            self.DeleteButton.setEnabled(False)
            self.sqrt.setEnabled(False)
            self.CloneButton.setEnabled(False)
            self.SaveButton.setEnabled(False)
            self.ExportButton.setEnabled(False)

            # Clear colorbar
            tickList = self.ColorBar.listTicks()
            for tick in tickList:
                self.ColorBar.removeTick(tick[0])
            self.ColorBar.setEnabled(False)

        elif len(SelectedThings) == 1 and showDel is True:
            self.ColorBar.setEnabled(True)
            self.DeleteButton.setEnabled(True)
            self.sqrt.setEnabled(True)
            self.CloneButton.setEnabled(True)
            self.SaveButton.setEnabled(True)
            self.ExportButton.setEnabled(True)

        elif len(SelectedThings) == 1 and showDel is False:
            self.CloneButton.setEnabled(True)
            self.ExportButton.setEnabled(True)

    # If square root button is checked/unchecked, modify the ticks as such
    def sqrtAction(self):
        if self.sqrt.isChecked() == True:
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
            QtGui.QMessageBox.information(self, "Please Unselect Gradient(s)", "You have a \"native\" gradient selected. Please "
                                                                               "unselect it before deleting.")
            return

        selectedGradients = self.getSelected()
        toPrint = ""
        for index in selectedGradients:
            toPrint = toPrint + index.text() + ", "

        toPrint = toPrint[:-2]

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
        selectedGradients = self.getSelected()
        fname = QtGui.QFileDialog.getSaveFileName(None, 'Save As', 'Export.txt')
        toExport = set()
        for index in selectedGradients:
            toExport.add(index.text())
        done = {}

        for k in self.gData.keys():
            if k in toExport:
                done[k] = self.gData[k]

        for k in self.autoImportData.keys():
            if k in toExport:
                done[k] = self.autoImportData[k]
        try:
            file = open(fname, 'w')
            file.write(str(done))
        except Exception as e:
            print("Error opening or writing!")
            print(e)


def main():
    app = QtGui.QApplication([])
    w = GradientControl()
    w.show()
    app.exec_()
    return 0


if __name__ == '__main__':
    sys.exit(main())
