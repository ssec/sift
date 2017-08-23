# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'change_colormap_dialog.ui'
#
# Created by: PyQt4 UI code generator 4.11.4
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_changeColormapDialog(object):
    def setupUi(self, changeColormapDialog):
        changeColormapDialog.setObjectName(_fromUtf8("changeColormapDialog"))
        changeColormapDialog.resize(351, 172)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(changeColormapDialog.sizePolicy().hasHeightForWidth())
        changeColormapDialog.setSizePolicy(sizePolicy)
        self.buttons = QtGui.QDialogButtonBox(changeColormapDialog)
        self.buttons.setGeometry(QtCore.QRect(30, 130, 311, 32))
        self.buttons.setOrientation(QtCore.Qt.Horizontal)
        self.buttons.setStandardButtons(QtGui.QDialogButtonBox.Close|QtGui.QDialogButtonBox.Reset)
        self.buttons.setCenterButtons(False)
        self.buttons.setObjectName(_fromUtf8("buttons"))
        self.cmap_combobox = QtGui.QComboBox(changeColormapDialog)
        self.cmap_combobox.setGeometry(QtCore.QRect(10, 10, 331, 26))
        self.cmap_combobox.setObjectName(_fromUtf8("cmap_combobox"))
        self.vmin_edit = QtGui.QLineEdit(changeColormapDialog)
        self.vmin_edit.setGeometry(QtCore.QRect(260, 50, 71, 21))
        self.vmin_edit.setObjectName(_fromUtf8("vmin_edit"))
        self.vmin_slider = QtGui.QSlider(changeColormapDialog)
        self.vmin_slider.setGeometry(QtCore.QRect(10, 50, 241, 22))
        self.vmin_slider.setOrientation(QtCore.Qt.Horizontal)
        self.vmin_slider.setObjectName(_fromUtf8("vmin_slider"))
        self.vmax_slider = QtGui.QSlider(changeColormapDialog)
        self.vmax_slider.setGeometry(QtCore.QRect(10, 90, 241, 22))
        self.vmax_slider.setOrientation(QtCore.Qt.Horizontal)
        self.vmax_slider.setObjectName(_fromUtf8("vmax_slider"))
        self.vmax_edit = QtGui.QLineEdit(changeColormapDialog)
        self.vmax_edit.setGeometry(QtCore.QRect(260, 90, 71, 21))
        self.vmax_edit.setObjectName(_fromUtf8("vmax_edit"))

        self.retranslateUi(changeColormapDialog)
        QtCore.QObject.connect(self.buttons, QtCore.SIGNAL(_fromUtf8("accepted()")), changeColormapDialog.accept)
        QtCore.QObject.connect(self.buttons, QtCore.SIGNAL(_fromUtf8("rejected()")), changeColormapDialog.reject)
        QtCore.QMetaObject.connectSlotsByName(changeColormapDialog)

    def retranslateUi(self, changeColormapDialog):
        changeColormapDialog.setWindowTitle(_translate("changeColormapDialog", "Change Colormap", None))
        self.vmin_edit.setToolTip(_translate("changeColormapDialog", "minimum color limit", None))
        self.vmin_slider.setToolTip(_translate("changeColormapDialog", "minimum color limit", None))
        self.vmax_slider.setToolTip(_translate("changeColormapDialog", "maximum color limit", None))
        self.vmax_edit.setToolTip(_translate("changeColormapDialog", "maximum color limit", None))

