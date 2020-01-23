from PyQt5 import QtWidgets, QtGui

from uwsift.ui.resample_dialog_ui import Ui_ResampleDialog


class ResampleDialog(QtWidgets.QDialog):

    def __init__(self, parent=None):
        super(ResampleDialog, self).__init__(parent)

        self.available_resampling_methods = ['None', 'Nearest Neighbor', 'Bilinear']
        self.available_projections = parent.parent().document.available_projections

        self.ui = Ui_ResampleDialog()
        self.ui.setupUi(self)

        self.ui.projectionComboBox.addItems(self.available_projections)
        self.ui.projectionComboBox.activated[str].connect(self.set_projection)
        self.ui.projectionComboBox.setCurrentIndex(parent.parent().document.current_projection_index())

        self.ui.resamplingMethodComboBox.addItems(self.available_resampling_methods)
        self.ui.resamplingMethodComboBox.activated[str].connect(self.set_resampler)

        self._set_opts_disabled(True)

        validator = QtGui.QDoubleValidator()
        validator.setNotation(QtGui.QDoubleValidator.StandardNotation)
        validator.setBottom(0.0)
        self.ui.resXLineEdit.setValidator(validator)
        self.ui.resYLineEdit.setValidator(validator)
        self.ui.resXLineEdit.textChanged.connect(self.update_info)
        self.ui.resYLineEdit.textChanged.connect(self.update_info)

        self.ui.buttonBox.button(QtWidgets.QDialogButtonBox.Ok).clicked.connect(self.update_info)

        self.resampler = None
        self.projection = None
        self.resolution = None

    def _set_opts_disabled(self, is_disabled):
        self.ui.projectionGroupBox.setDisabled(is_disabled)
        self.ui.resGroupBox.setDisabled(is_disabled)

    def _reset_fields(self):
        self.ui.resamplingMethodComboBox.setCurrentIndex(0)
        self.ui.projectionComboBox.setCurrentIndex(self.parent().parent().document.current_projection_index())
        self.ui.resXLineEdit.clear()
        self.ui.resYLineEdit.clear()
        self._set_opts_disabled(True)
        self.resampler = None

    def set_resampler(self, text):
        resample_opts = {'None': None, 'Nearest Neighbor': 'nearest', 'Bilinear': 'bilinear'}
        if self.ui.resamplingMethodComboBox.currentIndex() != 0:
            self._set_opts_disabled(False)
        else:
            self._set_opts_disabled(True)
            self._reset_fields()
        return resample_opts[text]

    def set_projection(self, text):
        self.projection = (text, self.available_projections[text])

    def update_info(self):
        if (self.ui.resXLineEdit.text() == "") != (self.ui.resYLineEdit.text() == ""):
            self.ui.buttonBox.button(QtWidgets.QDialogButtonBox.Ok).setDisabled(True)
            return
        else:
            self.ui.buttonBox.button(QtWidgets.QDialogButtonBox.Ok).setDisabled(False)

        self.resampler = self.set_resampler(self.ui.resamplingMethodComboBox.currentText())
        self.projection = (self.ui.projectionComboBox.currentText(),
                           self.available_projections[self.ui.projectionComboBox.currentText()])
        self.resolution = None if self.ui.resXLineEdit.text() == "" and self.ui.resYLineEdit.text() == "" \
            else (float(self.ui.resXLineEdit.text()), float(self.ui.resYLineEdit.text()))

        self.parent().resampling_info = {
            'resampler': self.resampler,
            'projection': self.projection,
            'resolution': self.resolution,
        }
