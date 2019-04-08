import json
import logging
import math
import os
from PyQt5 import QtCore, QtGui, QtWidgets

import pyqtgraph as pg

LOG = logging.getLogger(__name__)


class ColormapEditor(QtWidgets.QDialog):
    def __init__(self, doc, parent=None, **kwargs):
        super(ColormapEditor, self).__init__(parent)

        layout = QtWidgets.QGridLayout()
        layout.setSpacing(0)
        self.setLayout(layout)
        self.doc = doc
        self.user_colormap_states = {}
        self.builtin_colormap_states = {}

        # Setup Color Bar & clear its data
        self.ColorBar = pg.GradientWidget(orientation='bottom')
        tickList = self.ColorBar.listTicks()
        for tick in tickList:
            self.ColorBar.removeTick(tick[0])
        self.ColorBar.setEnabled(False)

        self.CloneButton = QtWidgets.QPushButton("Clone Colormap")
        self.CloneButton.clicked.connect(self.clone_colormap)
        self.CloneButton.setEnabled(False)

        # Create Import button
        self.ImportButton = QtWidgets.QPushButton("Import Colormap")
        self.ImportButton.clicked.connect(self.importButtonClick)

        # Create Colormap List and Related Functions
        self.cmap_list = QtWidgets.QListWidget()
        self.cmap_list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.cmap_list.itemSelectionChanged.connect(self.update_color_bar)

        # Create SQRT Button and Related Functions
        self.sqrt = QtWidgets.QCheckBox("SQRT")
        self.sqrt.clicked.connect(self.sqrt_action)
        self.sqrt.setEnabled(False)

        # Create Close button
        self.CloseButton = QtWidgets.QPushButton("Close")
        self.CloseButton.clicked.connect(self.close)

        # Create Delete Button and Related Functions
        self.DeleteButton = QtWidgets.QPushButton("Delete Colormap")
        self.DeleteButton.clicked.connect(self.handle_delete_click)
        self.DeleteButton.setEnabled(False)

        # Create Export Button and Related Functions
        self.ExportButton = QtWidgets.QPushButton("Export Colormap")
        self.ExportButton.clicked.connect(self.exportButtonClick)
        self.ExportButton.setEnabled(False)

        # Create Save button
        self.SaveButton = QtWidgets.QPushButton("Save Colormap")
        self.SaveButton.clicked.connect(self.save_button_click)
        self.SaveButton.setEnabled(False)

        # Add widgets to their respective spots in the UI grid
        layout.addWidget(self.ImportButton, 0, 0)
        layout.addWidget(self.SaveButton, 0, 2)
        layout.addWidget(self.sqrt, 1, 2)
        layout.addWidget(self.ColorBar, 4, 1)
        layout.addWidget(self.CloneButton, 1, 0)
        layout.addWidget(self.cmap_list, 1, 1, 3, 1)
        layout.addWidget(self.CloseButton, 6, 2)
        layout.addWidget(self.ExportButton, 2, 2)
        layout.addWidget(self.DeleteButton, 2, 0)

        # Import custom colormaps
        cmap_manager = self.doc.colormaps
        for cmap in cmap_manager.iter_colormaps():
            editable = cmap_manager.is_writeable_colormap(cmap)
            cmap_obj = cmap_manager[cmap]
            if cmap_obj.colors and hasattr(cmap_obj, "_controls"):
                is_sqrt = getattr(cmap_obj, 'sqrt', False)
                self.import_colormaps(cmap, cmap_obj._controls, cmap_obj.colors._rgba, sqrt=is_sqrt, editable=editable)

    def save_button_click(self):
        # save custom colormap
        name = self.cmap_list.item(self.cmap_list.currentRow()).text()
        self.user_colormap_states[name] = self.ColorBar.saveState()
        self.user_colormap_states[name]["sqrt"] = self.sqrt.isChecked()
        self.save_new_map(self.user_colormap_states[name], name)

    def clone_colormap(self):
        # Clone existing colormap
        text, ok = QtWidgets.QInputDialog.getText(self, 'Clone Colormap', 'Enter colormap name:')
        protected_names = ['mode', 'ticks', 'step']

        if ok:
            save_name = str(text)
            if save_name in self.user_colormap_states or save_name in protected_names:
                overwrite_msg = "There is already a save with this name. Would you like to Overwrite?"
                reply = QtWidgets.QMessageBox.question(
                    self, 'Message', overwrite_msg, QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)

                if reply == QtWidgets.QMessageBox.Yes:
                    if save_name in self.builtin_colormap_states or save_name in protected_names:
                        QtWidgets.QMessageBox.information(
                            self, "Error", "You cannot save a colormap with "
                                           "the same name as one of the "
                                           "internal colormaps or one of the "
                                           "protected names ('mode', "
                                           "'ticks', 'step').")
                        reply.close()
                        return

                    self.user_colormap_states[save_name] = self.ColorBar.saveState()
            else:
                if save_name in self.builtin_colormap_states:
                    QtWidgets.QMessageBox.information(
                        self, "Error",
                        "You cannot save a colormap with the same name as one of the internal colormaps.")
                    return

                self.user_colormap_states[save_name] = self.ColorBar.saveState()
            self.updateListWidget(save_name)
            self.save_new_map(self.user_colormap_states[save_name], save_name)

    def toRemoveDelete(self):
        # Determine if an internal colormap is selected, returns boolean
        toReturn = False

        ListCount = self.cmap_list.count()

        index = 0
        while index < ListCount:
            if self.cmap_list.item(index).isSelected():
                if self.cmap_list.item(index).text() in self.builtin_colormap_states:
                    toReturn = True
            index = index + 1

        return toReturn

    def save_new_map(self, new_cmap, name):
        # Call document function with new colormap
        self.doc.update_user_colormap(new_cmap, name)

    def import_colormaps(self, name, controls, colors, sqrt=False, editable=False):
        # Import a colormap into either the internal or custom colormap lists
        try:
            # FIXME: GradientWidget can accept 'allowAdd' flag for whether or
            #        not a widget is editable
            newWidget = pg.GradientWidget()
            newWidget.hide()

            for tick in newWidget.listTicks():
                newWidget.removeTick(tick[0])

            if not isinstance(controls, (tuple, list)):
                # convert numpy arrays to a list so we can JSON serialize
                # numpy data types
                controls = controls.tolist()
            for control, color in zip(controls, colors):
                newWidget.addTick(control, QtGui.QColor(*(color * 255.)), movable=editable)

            if editable:
                self.user_colormap_states[name] = newWidget.saveState()
                self.user_colormap_states[name]["sqrt"] = sqrt
            else:
                self.builtin_colormap_states[name] = newWidget.saveState()
            self.updateListWidget()
        except AssertionError as e:
            LOG.error(e)

    def updateListWidget(self, to_show=None):
        # Update list widget with new colormap list
        self.cmap_list.clear()

        total_count = 0
        corVal = 0
        for key in self.user_colormap_states:
            self.cmap_list.addItem(key)
            total_count += 1
            if to_show is not None and key == to_show:
                corVal = total_count

        self.cmap_list.addItem("----------------------------- "
                               "Below Are Builtin ColorMaps"
                               " -----------------------------")
        barrier_item = self.cmap_list.item(total_count)
        barrier_item.setFlags(QtCore.Qt.NoItemFlags)
        total_count += 1

        for key2 in self.builtin_colormap_states:
            self.cmap_list.addItem(key2)
            total_count += 1
            if to_show is not None and key2 == to_show:
                corVal = total_count

        if to_show is not None:
            self.cmap_list.setCurrentRow(corVal, QtCore.QItemSelectionModel.Select)

    def update_color_bar(self):
        # Update the colorbar with the newly selected colormap
        # FIXME: isn't this redundant?
        self.sqrt.setCheckState(False)

        cmap_name = self.cmap_list.item(self.cmap_list.currentRow()).text()
        if cmap_name in self.user_colormap_states:
            NewBar = self.user_colormap_states[self.cmap_list.item(self.cmap_list.currentRow()).text()]
            self.ColorBar.restoreState(NewBar)
            if "sqrt" in NewBar:
                self.sqrt.setChecked(NewBar["sqrt"])

        if cmap_name in self.builtin_colormap_states:
            NewBar = self.builtin_colormap_states[self.cmap_list.item(self.cmap_list.currentRow()).text()]
            self.ColorBar.restoreState(NewBar)

        # Bunch of functions determining which buttons to enable / disable
        showDel = True
        SelectedThings = self.cmap_list.selectedItems()
        for thing in SelectedThings:
            if thing.text() in self.builtin_colormap_states:
                showDel = False

        if len(SelectedThings) > 1:
            self.SaveButton.setEnabled(False)
            self.sqrt.setEnabled(False)
            self.CloneButton.setEnabled(False)

            tickList = self.ColorBar.listTicks()
            for tick in tickList:
                self.ColorBar.removeTick(tick[0])
            self.ColorBar.setEnabled(False)
        elif len(SelectedThings) == 1:
            self.ColorBar.setEnabled(showDel)
            self.sqrt.setEnabled(showDel)
            self.CloneButton.setEnabled(True)
            self.SaveButton.setEnabled(showDel)
            self.ExportButton.setEnabled(True)

        self.DeleteButton.setEnabled(showDel)

    def sqrt_action(self):
        # If square root button is checked/unchecked, modify the ticks as such
        if self.sqrt.isChecked():
            tickList = self.ColorBar.listTicks()
            for tick in tickList:
                self.ColorBar.setTickValue(tick[0], math.sqrt(self.ColorBar.tickValue(tick[0])))
        else:
            tickList = self.ColorBar.listTicks()
            for tick in tickList:
                self.ColorBar.setTickValue(tick[0], self.ColorBar.tickValue(tick[0]) * self.ColorBar.tickValue(tick[0]))

    def handle_delete_click(self):
        # Delete colormap(s)
        block = self.toRemoveDelete()
        if block is True:
            # This shouldn't happen
            QtWidgets.QMessageBox.information(self, "Error: Can not delete internal colormaps.")
            return

        selected_colormaps = self.cmap_list.selectedItems()
        to_print = ",".join([x.text() for x in selected_colormaps])

        delete_msg = "Please confirm you want to delete the colormap(s): " + to_print
        reply = QtWidgets.QMessageBox.question(
            self, 'Message', delete_msg, QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
        if reply == QtWidgets.QMessageBox.Yes:
            for index in selected_colormaps:
                del self.user_colormap_states[index.text()]
                self.doc.remove_user_colormap(index.text())
            self.updateListWidget()

    def importButtonClick(self):
        # Import colormap
        fname = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Get Colormap File', os.path.expanduser('~'), "Colormaps (*.json)")[0]
        self._import_single_file(fname)

    def _import_single_file(self, filename):
        try:
            cmap_content = json.loads(open(filename, 'r').read())
            # FUTURE: Handle all types of colormaps, make sure they are copied to the settings directory
            if isinstance(cmap_content, dict) and 'ticks' in cmap_content:
                # single colormap file
                cmap_name = os.path.splitext(os.path.basename(filename))[0]
                cmap_content = {cmap_name: cmap_content}
            elif isinstance(cmap_content, list) and isinstance(cmap_content[0], dict):
                # list of individual colormap objects (not currently used
                cmap_content = {cmap['name']: cmap for cmap in cmap_content}
            elif not isinstance(cmap_content, dict):
                raise ValueError("Unknown colormap file format: {}".format(filename))

            for cmap_name in cmap_content:
                if cmap_name in self.builtin_colormap_states:
                    QtGui.QMessageBox.information(
                        self, "Error", "You cannot import a colormap with "
                                       "the same name as one of the internal "
                                       "colormaps: {}".format(cmap_name))
                    return

            for cmap_name, cmap_info in cmap_content.items():
                if cmap_name in self.user_colormap_states:
                    LOG.info("Overwriting colormap '{}'".format(cmap_name))
                else:
                    LOG.info("Importing new colormap '{}'".format(cmap_name))
                self.save_new_map(cmap_info, cmap_name)
            self.user_colormap_states.update(cmap_content)
            self.updateListWidget(cmap_name)
        except IOError:
            LOG.error("Error importing colormap from file "
                      "{}".format(filename), exc_info=True)

    def exportButtonClick(self):
        # Export colormap(s)
        selected_colormaps = self.cmap_list.selectedItems()
        fname = QtWidgets.QFileDialog.getSaveFileName(None, 'Save As', 'Export.json')[0]
        toExport = set()
        for index in selected_colormaps:
            toExport.add(index.text())
        done = {}

        for k in self.user_colormap_states:
            if k in toExport:
                done[k] = self.user_colormap_states[k]

        for k in self.builtin_colormap_states:
            if k in toExport:
                done[k] = self.builtin_colormap_states[k]
        try:
            file = open(fname, 'w')
            file.write(json.dumps(done, indent=2, sort_keys=True))
            file.close()
        except IOError:
            LOG.error("Error exporting colormaps: {}".format(fname),
                      exc_info=True)


def main():
    app = QtWidgets.QApplication([])
    w = ColormapEditor()
    w.show()
    app.exec_()
    return 0


if __name__ == '__main__':
    import sys

    sys.exit(main())
