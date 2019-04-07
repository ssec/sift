# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'change_colormap_dialog.ui'
#
# Created by: PyQt4 UI code generator 4.11.4
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

try:
    _encoding = QtWidgets.QApplication.UnicodeUTF8


    def _translate(context, text, disambig):
        return QtCore.QCoreApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtCore.QCoreApplication.translate(context, text, disambig)


class Ui_changeColormapDialog(object):
    def setupUi(self, changeColormapDialog):
        changeColormapDialog.setObjectName("changeColormapDialog")
        changeColormapDialog.resize(351, 199)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(changeColormapDialog.sizePolicy().hasHeightForWidth())
        changeColormapDialog.setSizePolicy(sizePolicy)
        self.buttons = QtWidgets.QDialogButtonBox(changeColormapDialog)
        self.buttons.setGeometry(QtCore.QRect(20, 160, 321, 32))
        self.buttons.setOrientation(QtCore.Qt.Horizontal)
        self.buttons.setStandardButtons(QtWidgets.QDialogButtonBox.Close | QtWidgets.QDialogButtonBox.Reset)
        self.buttons.setCenterButtons(False)
        self.buttons.setObjectName("buttons")
        self.cmap_combobox = QtWidgets.QComboBox(changeColormapDialog)
        self.cmap_combobox.setGeometry(QtCore.QRect(10, 10, 331, 26))
        self.cmap_combobox.setObjectName("cmap_combobox")
        self.vmin_edit = QtWidgets.QLineEdit(changeColormapDialog)
        self.vmin_edit.setGeometry(QtCore.QRect(260, 50, 71, 21))
        self.vmin_edit.setObjectName("vmin_edit")
        self.vmin_slider = QtWidgets.QSlider(changeColormapDialog)
        self.vmin_slider.setGeometry(QtCore.QRect(10, 50, 241, 21))
        self.vmin_slider.setOrientation(QtCore.Qt.Horizontal)
        self.vmin_slider.setObjectName("vmin_slider")
        self.vmax_slider = QtWidgets.QSlider(changeColormapDialog)
        self.vmax_slider.setGeometry(QtCore.QRect(10, 90, 241, 21))
        self.vmax_slider.setOrientation(QtCore.Qt.Horizontal)
        self.vmax_slider.setObjectName("vmax_slider")
        self.vmax_edit = QtWidgets.QLineEdit(changeColormapDialog)
        self.vmax_edit.setGeometry(QtCore.QRect(260, 90, 71, 21))
        self.vmax_edit.setObjectName("vmax_edit")
        self.gammaSpinBox = QNoScrollDoubleSpinBox(changeColormapDialog)
        self.gammaSpinBox.setEnabled(True)
        self.gammaSpinBox.setGeometry(QtCore.QRect(260, 120, 71, 24))
        self.gammaSpinBox.setDecimals(1)
        self.gammaSpinBox.setMaximum(5.0)
        self.gammaSpinBox.setSingleStep(0.1)
        self.gammaSpinBox.setProperty("value", 1.0)
        self.gammaSpinBox.setObjectName("gammaSpinBox")
        self.gammaLabel = QtWidgets.QLabel(changeColormapDialog)
        self.gammaLabel.setGeometry(QtCore.QRect(200, 120, 56, 24))
        self.gammaLabel.setAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.gammaLabel.setObjectName("gammaLabel")

        self.retranslateUi(changeColormapDialog)
        self.buttons.accepted.connect(changeColormapDialog.accept)
        self.buttons.rejected.connect(changeColormapDialog.reject)
        QtCore.QMetaObject.connectSlotsByName(changeColormapDialog)

    def retranslateUi(self, changeColormapDialog):
        changeColormapDialog.setWindowTitle(_translate("changeColormapDialog", "Change Colormap", None))
        self.vmin_edit.setToolTip(_translate("changeColormapDialog", "minimum color limit", None))
        self.vmin_slider.setToolTip(_translate("changeColormapDialog", "minimum color limit", None))
        self.vmax_slider.setToolTip(_translate("changeColormapDialog", "maximum color limit", None))
        self.vmax_edit.setToolTip(_translate("changeColormapDialog", "maximum color limit", None))
        self.gammaLabel.setText(_translate("changeColormapDialog", "Gamma: ", None))


from uwsift.ui.custom_widgets import QNoScrollDoubleSpinBox
