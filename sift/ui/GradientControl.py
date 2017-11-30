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

import sift
from sift.view.Colormap import ALL_COLORMAPS
import sift.view.colormap_dialogs as colord


class GradientControl(QtGui.QDialog):
    def __init__(self, doc, parent=None, **kwargs):
        super(GradientControl, self).__init__(parent)

        l = QtGui.QGridLayout()
        l.setSpacing(0)
        self.setLayout(l)
        self.doc = doc
        self.gData = {}
        self.autoImportData = {}
        self.INTERNAL_ALL_COLORMAPS = ALL_COLORMAPS


        try:
            f = open('data.json', 'x+')
            print("No file found, continuing")
        except FileExistsError:
            if os.stat("data.json").st_size != 0:
                try:
                    self.gData = json.loads(open('data.json').read())
                    print("Loaded data from data.json")
                except:
                    print("Error reading json file")
            else:
                print("Empty json file found.")

        self.ColorBar = pg.GradientWidget(orientation='bottom')
        self.ColorBar.hide()

        self.SaveButton = QtGui.QPushButton("Save Gradient")
        self.SaveButton.clicked.connect(self.on_save_click)

        self.ImportButton = QtGui.QPushButton("Import Gradient")
        self.ImportButton.clicked.connect(self.importButtonClick)

        # Create Gradient List and Related Functions
        self.List = QtGui.QListWidget()
        self.List.setSelectionMode(QtGui.QAbstractItemView.ExtendedSelection)
        self.List.itemSelectionChanged.connect(self.updateColorBar)

        # Create SQRT Button and Related Functions
        self.sqrt = QtGui.QCheckBox("SQRT")
        self.sqrt.stateChanged.connect(self.sqrtAction)

        self.CloseButton = QtGui.QPushButton("Close")
        self.CloseButton.clicked.connect(self.closeButtonClick)

        # Create Delete Button and Related Functions
        self.DeleteButton = QtGui.QPushButton("Delete Gradient")
        self.DeleteButton.clicked.connect(self.deleteButtonClick)
        self.DeleteButton.hide()

        # Create Export Button and Related Functions
        self.ExportButton = QtGui.QPushButton("Export Gradient")
        self.ExportButton.clicked.connect(self.exportButtonClick)
        self.ExportButton.hide()

        # Create Update Map Button & Related Functions
        self.UpdateMapButton = QtGui.QPushButton("Update SIFT Map")
        self.UpdateMapButton.clicked.connect(self.updateButtonClick)

        self.updateListWidget()


        l.addWidget(self.ImportButton, 0, 0)
        l.addWidget(self.sqrt, 1, 2)
        l.addWidget(self.ColorBar, 2, 1)
        l.addWidget(self.SaveButton, 1, 0)
        l.addWidget(self.List, 1, 1)
        l.addWidget(self.CloseButton, 6, 2)
        l.addWidget(self.ExportButton, 2, 2)
        l.addWidget(self.DeleteButton, 3, 0)
        l.addWidget(self.UpdateMapButton, 2, 0)

        for map in ALL_COLORMAPS:
            if ALL_COLORMAPS[map].colors and (hasattr(ALL_COLORMAPS[map], "_controls")):
                self.importGradients(map, ALL_COLORMAPS[map].colors.hex, ALL_COLORMAPS[map]._controls)

                print("Imported Gradient!")
            else:
                print("Skipping: " + map)
                print("Type: " + str(type(ALL_COLORMAPS[map])))


    def updateButtonClick(self):
        self.autoImportData[self.List.item(self.List.currentRow()).text()] = self.ColorBar.saveState()

        for item in self.autoImportData.keys():
            print("Doing " + item)
            pointList = self.autoImportData[item]["ticks"]
            floats = []
            hex = []
            for point in pointList:
                floats.append(point[0])
                rgb = point[1]
                hexCode = rgb2hex(rgb[0], rgb[1], rgb[2])
                hex.append(hexCode)

            for i in range(len(floats)):
                for k in range(len(floats) - 1, i, -1):
                    if (floats[k] < floats[k - 1]):
                        self.bubbleSortSwap(floats, k, k - 1)
                        self.bubbleSortSwap(hex, k, k - 1)

            print(ALL_COLORMAPS[item].colors.hex)
            print(numpy.array(hex))
            print("\n")
            print(ALL_COLORMAPS[item]._controls)
            print(floats)
            print("\n")
            try:
                toAdd = Colormap(colors=hex, controls=floats)
                ALL_COLORMAPS[item] = toAdd
            except:
                print("Error creating or setting colormap")
            #try:
            #    toAdd = Colormap(colors=hex, controls=floats)
            #    ALL_COLORMAPS[item] = toAdd
            #except Exception as e:
            #    print("Error updating gradient!")
            #    print(e)

        self.doc.update_colormaps(ALL_COLORMAPS)

        print(self.List.item(self.List.currentRow()).text())
        self.doc.change_colormap_for_layers(self.List.item(self.List.currentRow()).text())

    def bubbleSortSwap(self, A, x, y):
        tmp = A[x]
        A[x] = A[y]
        A[y] = tmp

    def on_save_click(self):
        self.p = QtGui.QWidget()
        self.p.setWindowTitle('Save Gradient As:')
        self.p.textbox = QtGui.QLineEdit(self.p)
        self.p.textbox.move(20, 20)
        self.p.textbox.resize(280, 40)
        self.p.resize(320, 150)
        button = QtGui.QPushButton('Save', self.p)
        button.move(20, 80)
        button.clicked.connect(self.on_save_click_2)
        self.p.setWindowModality(QtCore.Qt.WindowModal)
        self.p.show()

    def getSelected(self):
        toReturn = []

        ListCount = self.List.count()

        index = 0
        while index < ListCount:
            if (self.List.item(index).isSelected()):
                toReturn.append(self.List.item(index))
            index = index + 1

        return toReturn


    def toRemoveDelete(self):
        toReturn = False

        ListCount = self.List.count()

        index = 0
        #self.List.item(index).text() in self.autoImportData.keys()
        while index < ListCount:
            if (self.List.item(index).isSelected()):
                if self.List.item(index).text() in self.autoImportData.keys():
                    toReturn = True
            index = index + 1

        return toReturn

    def on_save_click_2(self):
        SaveName = self.p.textbox.text()
        if SaveName in self.gData.keys():
            overwrite_msg = "There is already a save with this name. Would you like to Overwrite?"
            reply = QtGui.QMessageBox.question(self, 'Message',
                                               overwrite_msg, QtGui.QMessageBox.Yes, QtGui.QMessageBox.No)

            if reply == QtGui.QMessageBox.Yes:
                self.gData[SaveName] = self.ColorBar.saveState()
                self.saveData()
                self.p.close()
        else:
            self.gData[SaveName] = self.ColorBar.saveState()
            self.saveData()
            self.p.close()
        self.updateListWidget()

    def saveData(self):
        with open('data.json', 'w') as fp:
            json.dump(self.gData, fp)

    def importButtonClick(self):

        reply2 = QtGui.QMessageBox()
        reply2.setWindowTitle("File or Arrays?")
        reply2.setText("Would you like to import from a file or from point + color arrays?")
        reply2.setStandardButtons(QtGui.QMessageBox.Yes | QtGui.QMessageBox.No | QtGui.QMessageBox.Cancel)
        buttonY = reply2.button(QtGui.QMessageBox.Yes)
        buttonY.setText('File')
        buttonN = reply2.button(QtGui.QMessageBox.No)
        buttonN.setText('Arrays')

        reply2.exec_()

        if reply2.clickedButton() == buttonY:
            print("Starting File")
            fname = QtGui.QFileDialog.getOpenFileName(None, 'Get File', 'Export.txt')
            try:
                file = open(fname, "r")
                toImport = ast.literal_eval(file.read())

                self.gData.update(toImport)

                self.saveData()

                self.updateListWidget()
            except:
                print("Error opening file or reading!")
            print("Done")
        elif reply2.clickedButton() == buttonN:
            print("Going")
            pointArray = QtGui.QInputDialog.getText(self, self.tr("Input Points"),
                                                    self.tr("Point Array:"), QtGui.QLineEdit.Normal)

            if pointArray[0]:
                colorArray = QtGui.QInputDialog.getText(self, self.tr("Input Hex Colors"),
                                                        self.tr("Color Array:"), QtGui.QLineEdit.Normal)

                if colorArray[0]:
                    try:
                        points_values = pointArray[0]

                        points_values = points_values.replace('(', '')
                        points_values = points_values.replace(')', '')
                        points_values = points_values.replace('[', '')
                        points_values = points_values.replace(']', '')
                        points_values = points_values.replace('{', '')
                        points_values = points_values.replace('}', '')
                        points_values = points_values.split(',')

                        points_values = [float(i) for i in points_values]

                        colors_values = colorArray[0]

                        colors_values = colors_values.replace('(', '')
                        colors_values = colors_values.replace(')', '')
                        colors_values = colors_values.replace('[', '')
                        colors_values = colors_values.replace(']', '')
                        colors_values = colors_values.replace('{', '')
                        colors_values = colors_values.replace('}', '')
                        colors_values = colors_values.replace('\'', '')
                        colors_values = colors_values.replace(' ', '')
                        colors_values = colors_values.split(',')

                        data = {}

                        for i in range(len(points_values)):
                            data[points_values[i]] = colors_values[i]

                        newName = QtGui.QInputDialog.getText(self, self.tr("Input ColorBar Name"),
                                                             self.tr("Name:"), QtGui.QLineEdit.Normal)

                        if newName[0]:
                            newWidget = pg.GradientWidget()
                            newWidget.hide()
                            for key in data:
                                newWidget.addTick(key, QtGui.QColor(data[key]), True)
                            self.gData[newName[0]] = newWidget.saveState()
                            self.saveData()
                            self.updateListWidget()


                    except:
                        QtGui.QMessageBox.information(self, "Error", "Error loading the arrays!")

    def importGradients(self, name, hex, floats):
        try:
            data = {}

            for i in range(len(hex)):
                data[floats.astype(float)[i]] = hex[i]

            newWidget = pg.GradientWidget()
            newWidget.hide()
            for key in data:
                newWidget.addTick(key, QtGui.QColor(data[key]), True)
            #self.gData[name] = newWidget.saveState()
            self.autoImportData[name] = newWidget.saveState()
            #self.saveData()
            self.updateListWidget()

        except Exception as e:
            print(e)
            #QtGui.QMessageBox.information(self, "Error", "Error loading the arrays!")

    def closeButtonClick(self):
        self.done(0)

    def updateListWidget(self):
        self.List.clear()
        self.ExportButton.hide()
        self.DeleteButton.hide()
        for key in self.gData.keys():
            self.List.addItem(key)

        for key2 in self.autoImportData.keys():
            self.List.addItem(key2)

    def updateColorBar(self):
        self.ExportButton.show()
        self.ColorBar.show()
        self.DeleteButton.show()
        self.sqrt.setCheckState(0)

        if self.List.item(self.List.currentRow()).text() in self.gData.keys():
            NewBar = self.gData[self.List.item(self.List.currentRow()).text()]
            self.ColorBar.restoreState(NewBar)


        if self.List.item(self.List.currentRow()).text() in self.autoImportData.keys():
            NewBar = self.autoImportData[self.List.item(self.List.currentRow()).text()]
            self.ColorBar.restoreState(NewBar)

        SelectedThings = self.getSelected()

        if len(SelectedThings) > 1:
            self.ColorBar.hide()
            self.SaveButton.hide()
            self.sqrt.hide()
        else:
            self.ColorBar.show()
            self.SaveButton.show()
            self.sqrt.show()

    def sqrtAction(self):
        if self.sqrt.isChecked() == True:
            tickList = self.ColorBar.listTicks()
            for tick in tickList:
                self.ColorBar.setTickValue(tick[0], math.sqrt(self.ColorBar.tickValue(tick[0])))
        else:
            tickList = self.ColorBar.listTicks()
            for tick in tickList:
                self.ColorBar.setTickValue(tick[0], self.ColorBar.tickValue(tick[0]) * self.ColorBar.tickValue(tick[0]))

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
            self.updateListWidget()
            self.saveData()

    def exportButtonClick(self):
        selectedGradients = self.getSelected()
        fname, _ = QtGui.QFileDialog.getSaveFileName(None, 'Save As', 'Export.txt')
        toExport = set()
        for index in selectedGradients:
            toExport.add(index.text())
        done = {k: self.gData[k] for k in self.gData.keys() & toExport}
        try:
            file = open(fname, 'w')
            file.write(str(done))
        except:
            print("Error opening or writing!")


def main():
    app = QtGui.QApplication([])
    w = GradientControl()
    w.show()
    app.exec_()
    return 0


if __name__ == '__main__':
    sys.exit(main())
