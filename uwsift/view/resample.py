from PyQt5 import QtWidgets, QtGui

from uwsift.ui.resample_dialog_ui import Ui_ResampleDialog


class ResampleDialog(QtWidgets.QDialog):

    def __init__(self, parent=None):
        super(ResampleDialog, self).__init__(parent)

        self.available_resampling_methods = ['None', 'Nearest Neighbor']
        self.available_projections = parent.parent().document.available_projections

        self.ui = Ui_ResampleDialog()
        self.ui.setupUi(self)

        self.ui.projectionComboBox.addItems(self.available_projections)
        self.ui.projectionComboBox.activated[str].connect(self.set_projection)
        self.ui.resamplingMethodComboBox.addItems(self.available_resampling_methods)
        self.ui.resamplingMethodComboBox.activated[str].connect(self.set_resampler)

        self._set_opts_disabled(True)

        validator1 = QtGui.QDoubleValidator()
        validator2 = QtGui.QDoubleValidator()
        validator1.setNotation(QtGui.QDoubleValidator.StandardNotation)
        validator2.setNotation(QtGui.QDoubleValidator.StandardNotation)
        validator1.setBottom(0.0)
        self.ui.resXLineEdit.setValidator(validator1)
        self.ui.resYLineEdit.setValidator(validator1)
        self.ui.resXLineEdit.textChanged.connect(self.update_res)
        self.ui.resYLineEdit.textChanged.connect(self.update_res)
        self.ui.shapeXLineEdit.setValidator(validator1)
        self.ui.shapeYLineEdit.setValidator(validator1)
        self.ui.shapeXLineEdit.textChanged.connect(self.update_shape)
        self.ui.shapeYLineEdit.textChanged.connect(self.update_shape)

        self.ui.areaExLowerLeftXLineEdit.setValidator(validator2)
        self.ui.areaExLowerLeftYLineEdit.setValidator(validator2)
        self.ui.areaExUpperRightXLineEdit.setValidator(validator2)
        self.ui.areaExUpperRightYLineEdit.setValidator(validator2)
        self.ui.areaExLowerLeftXLineEdit.textChanged.connect(self._check_fields)
        self.ui.areaExLowerLeftYLineEdit.textChanged.connect(self._check_fields)
        self.ui.areaExUpperRightXLineEdit.textChanged.connect(self._check_fields)
        self.ui.areaExUpperRightYLineEdit.textChanged.connect(self._check_fields)

        self.ui.buttonBox.button(QtWidgets.QDialogButtonBox.Ok).clicked.connect(self.update_info)

        self.resampler = None
        self.projection = None
        self.shape = None
        self.resolution = None
        self.area_extent = None

    def _set_opts_disabled(self, is_disabled):
        self.ui.projectionGroupBox.setDisabled(is_disabled)
        self.ui.resGroupBox.setDisabled(is_disabled)
        self.ui.shapeGroupBox.setDisabled(is_disabled)
        self.ui.areaExGroupBox.setDisabled(is_disabled)

    def _reset_fields(self):
        self.ui.resamplingMethodComboBox.setCurrentIndex(0)
        self.ui.projectionComboBox.setCurrentIndex(0)
        self.ui.resXLineEdit.clear()
        self.ui.resYLineEdit.clear()
        self.ui.shapeXLineEdit.clear()
        self.ui.shapeYLineEdit.clear()
        self.ui.areaExLowerLeftXLineEdit.clear()
        self.ui.areaExLowerLeftYLineEdit.clear()
        self.ui.areaExUpperRightXLineEdit.clear()
        self.ui.areaExUpperRightYLineEdit.clear()
        self._set_opts_disabled(True)
        self.resampler = None

    def _is_empty(self, field):
        if field == 'shape':
            if self.ui.shapeXLineEdit.text() == "" and self.ui.shapeYLineEdit.text() == "":
                return True
        elif field == 'res':
            if self.ui.resXLineEdit.text() == "" and self.ui.resYLineEdit.text() == "":
                return True
        elif field == 'area_extent':
            if self.ui.areaExLowerLeftXLineEdit.text() == "" and \
               self.ui.areaExLowerLeftYLineEdit.text() == "" and \
               self.ui.areaExUpperRightXLineEdit.text() == "" and \
               self.ui.areaExUpperRightYLineEdit.text() == "":
                return True
        return False

    def _is_filled(self):
        return (self.ui.areaExUpperRightYLineEdit.text() != "" and
                self.ui.areaExUpperRightXLineEdit.text() != "" and
                self.ui.areaExLowerLeftYLineEdit.text() != "" and
                self.ui.areaExLowerLeftXLineEdit.text() != "")

    def _is_exclusive(self):
        return (self.ui.shapeXLineEdit.text() != "" and
                self.ui.shapeYLineEdit.text() != "" and
                self.ui.resXLineEdit.text() == "" and
                self.ui.resYLineEdit.text() == "") or \
               (self.ui.shapeYLineEdit.text() == "" and
                self.ui.shapeXLineEdit.text() == "" and
                self.ui.resYLineEdit.text() != "" and
                self.ui.resXLineEdit.text() != "")

    def _check_fields(self):
        if not self._is_filled() and not self._is_empty('area_extent'):
            self.ui.buttonBox.button(QtWidgets.QDialogButtonBox.Ok).setDisabled(True)
            return

        if not self._is_exclusive() and not (self._is_empty('shape') and self._is_empty('res')):
            self.ui.buttonBox.button(QtWidgets.QDialogButtonBox.Ok).setDisabled(True)
            return

        self.ui.buttonBox.button(QtWidgets.QDialogButtonBox.Ok).setDisabled(False)

    def update_res(self):
        if self._is_empty('res'):
            self.ui.shapeGroupBox.setDisabled(False)
        else:
            self.ui.shapeGroupBox.setDisabled(True)
        self._check_fields()

    def update_shape(self):
        if self._is_empty('shape'):
            self.ui.resGroupBox.setDisabled(False)
        else:
            self.ui.resGroupBox.setDisabled(True)
        self._check_fields()

    def set_resampler(self, text):
        resample_opts = {'None': None, 'Nearest Neighbor': 'nearest'}
        if self.ui.resamplingMethodComboBox.currentIndex() != 0:
            self._set_opts_disabled(False)
        else:
            self._set_opts_disabled(True)
            self._reset_fields()
        return resample_opts[text]

    def set_projection(self, text):
        self.projection = (text, self.available_projections[text])

    def update_info(self):
        self.resampler = self.set_resampler(self.ui.resamplingMethodComboBox.currentText())
        self.projection = (self.ui.projectionComboBox.currentText(),
                           self.available_projections[self.ui.projectionComboBox.currentText()])
        self.shape = None if self._is_empty('shape') \
            else (float(self.ui.shapeXLineEdit.text()), float(self.ui.shapeYLineEdit.text()))

        self.resolution = None if self._is_empty('res') \
            else (float(self.ui.resXLineEdit.text()), float(self.ui.shapeYLineEdit.text()))
        self.area_extent = None if self._is_empty('area_extent') \
            else (float(self.ui.areaExLowerLeftXLineEdit.text()), float(self.ui.areaExLowerLeftYLineEdit.text()),
                  float(self.ui.areaExUpperRightXLineEdit.text()), float(self.ui.areaExUpperRightYLineEdit.text()))

        self.parent().resampling_info = {
            'resampler': self.resampler,
            'projection': self.projection,
            'shape': self.shape,
            'resolution': self.resolution,
            'area extent': self.area_extent,
        }
