# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'resample_dialog.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtWidgets


class Ui_ResampleDialog(object):
    def setupUi(self, ResampleDialog):
        ResampleDialog.setObjectName("ResampleDialog")
        ResampleDialog.resize(310, 282)
        self.buttonBox = QtWidgets.QDialogButtonBox(ResampleDialog)
        self.buttonBox.setGeometry(QtCore.QRect(130, 240, 171, 32))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel | QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.resGroupBox = QtWidgets.QGroupBox(ResampleDialog)
        self.resGroupBox.setGeometry(QtCore.QRect(20, 150, 131, 81))
        self.resGroupBox.setObjectName("resGroupBox")
        self.resXLineEdit = QtWidgets.QLineEdit(self.resGroupBox)
        self.resXLineEdit.setGeometry(QtCore.QRect(10, 30, 113, 21))
        self.resXLineEdit.setText("")
        self.resXLineEdit.setObjectName("resXLineEdit")
        self.resYLineEdit = QtWidgets.QLineEdit(self.resGroupBox)
        self.resYLineEdit.setGeometry(QtCore.QRect(10, 50, 113, 21))
        self.resYLineEdit.setObjectName("resYLineEdit")
        self.resamplingMethodGroupBox = QtWidgets.QGroupBox(ResampleDialog)
        self.resamplingMethodGroupBox.setGeometry(QtCore.QRect(20, 10, 271, 71))
        self.resamplingMethodGroupBox.setObjectName("resamplingMethodGroupBox")
        self.resamplingMethodComboBox = QtWidgets.QComboBox(self.resamplingMethodGroupBox)
        self.resamplingMethodComboBox.setGeometry(QtCore.QRect(10, 30, 251, 26))
        self.resamplingMethodComboBox.setObjectName("resamplingMethodComboBox")
        self.projectionGroupBox = QtWidgets.QGroupBox(ResampleDialog)
        self.projectionGroupBox.setGeometry(QtCore.QRect(20, 80, 271, 71))
        self.projectionGroupBox.setObjectName("projectionGroupBox")
        self.projectionComboBox = QtWidgets.QComboBox(self.projectionGroupBox)
        self.projectionComboBox.setGeometry(QtCore.QRect(10, 30, 251, 26))
        self.projectionComboBox.setObjectName("projectionComboBox")

        self.retranslateUi(ResampleDialog)
        self.buttonBox.accepted.connect(ResampleDialog.accept)
        self.buttonBox.rejected.connect(ResampleDialog.reject)
        QtCore.QMetaObject.connectSlotsByName(ResampleDialog)

    def retranslateUi(self, ResampleDialog):
        _translate = QtCore.QCoreApplication.translate
        ResampleDialog.setWindowTitle(_translate("ResampleDialog", "Resample"))
        self.resGroupBox.setTitle(_translate("ResampleDialog", "Resolution"))
        self.resXLineEdit.setPlaceholderText(_translate("ResampleDialog", "pixel size in x dir"))
        self.resYLineEdit.setPlaceholderText(_translate("ResampleDialog", "pixel size in y dir"))
        self.resamplingMethodGroupBox.setTitle(_translate("ResampleDialog", "Resampling Method"))
        self.projectionGroupBox.setTitle(_translate("ResampleDialog", "Projection"))
