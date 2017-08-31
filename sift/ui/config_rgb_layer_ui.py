# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'config_rgb_layer.ui'
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

class Ui_config_rgb_layer(object):
    def setupUi(self, config_rgb_layer):
        config_rgb_layer.setObjectName(_fromUtf8("config_rgb_layer"))
        config_rgb_layer.resize(421, 447)
        self.verticalLayout = QtGui.QVBoxLayout(config_rgb_layer)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.groupBox_3 = QtGui.QGroupBox(config_rgb_layer)
        self.groupBox_3.setObjectName(_fromUtf8("groupBox_3"))
        self.comboRed = QtGui.QComboBox(self.groupBox_3)
        self.comboRed.setGeometry(QtCore.QRect(10, 20, 380, 26))
        self.comboRed.setObjectName(_fromUtf8("comboRed"))
        self.slideMinRed = QtGui.QSlider(self.groupBox_3)
        self.slideMinRed.setGeometry(QtCore.QRect(10, 50, 380, 22))
        self.slideMinRed.setMaximum(32767)
        self.slideMinRed.setOrientation(QtCore.Qt.Horizontal)
        self.slideMinRed.setObjectName(_fromUtf8("slideMinRed"))
        self.slideMaxRed = QtGui.QSlider(self.groupBox_3)
        self.slideMaxRed.setGeometry(QtCore.QRect(10, 80, 380, 22))
        self.slideMaxRed.setMaximum(32767)
        self.slideMaxRed.setOrientation(QtCore.Qt.Horizontal)
        self.slideMaxRed.setObjectName(_fromUtf8("slideMaxRed"))
        self.redGammaSpinBox = QtGui.QDoubleSpinBox(self.groupBox_3)
        self.redGammaSpinBox.setGeometry(QtCore.QRect(70, 110, 62, 24))
        self.redGammaSpinBox.setObjectName(_fromUtf8("redGammaSpinBox"))
        self.redGammaLabel = QtGui.QLabel(self.groupBox_3)
        self.redGammaLabel.setGeometry(QtCore.QRect(10, 110, 56, 24))
        self.redGammaLabel.setObjectName(_fromUtf8("redGammaLabel"))
        self.comboRed.raise_()
        self.slideMinRed.raise_()
        self.slideMaxRed.raise_()
        self.redGammaSpinBox.raise_()
        self.redGammaLabel.raise_()
        self.verticalLayout.addWidget(self.groupBox_3)
        self.groupBox_2 = QtGui.QGroupBox(config_rgb_layer)
        self.groupBox_2.setObjectName(_fromUtf8("groupBox_2"))
        self.comboGreen = QtGui.QComboBox(self.groupBox_2)
        self.comboGreen.setGeometry(QtCore.QRect(10, 20, 380, 26))
        self.comboGreen.setObjectName(_fromUtf8("comboGreen"))
        self.slideMinGreen = QtGui.QSlider(self.groupBox_2)
        self.slideMinGreen.setGeometry(QtCore.QRect(10, 50, 380, 22))
        self.slideMinGreen.setMaximum(32767)
        self.slideMinGreen.setOrientation(QtCore.Qt.Horizontal)
        self.slideMinGreen.setObjectName(_fromUtf8("slideMinGreen"))
        self.slideMaxGreen = QtGui.QSlider(self.groupBox_2)
        self.slideMaxGreen.setGeometry(QtCore.QRect(10, 80, 380, 22))
        self.slideMaxGreen.setMaximum(32767)
        self.slideMaxGreen.setOrientation(QtCore.Qt.Horizontal)
        self.slideMaxGreen.setObjectName(_fromUtf8("slideMaxGreen"))
        self.greenGammaLabel = QtGui.QLabel(self.groupBox_2)
        self.greenGammaLabel.setGeometry(QtCore.QRect(10, 110, 56, 24))
        self.greenGammaLabel.setObjectName(_fromUtf8("greenGammaLabel"))
        self.greenGammaSpinBox = QtGui.QDoubleSpinBox(self.groupBox_2)
        self.greenGammaSpinBox.setGeometry(QtCore.QRect(70, 110, 62, 24))
        self.greenGammaSpinBox.setObjectName(_fromUtf8("greenGammaSpinBox"))
        self.comboGreen.raise_()
        self.slideMinGreen.raise_()
        self.slideMaxGreen.raise_()
        self.groupBox_3.raise_()
        self.groupBox_3.raise_()
        self.greenGammaLabel.raise_()
        self.greenGammaSpinBox.raise_()
        self.verticalLayout.addWidget(self.groupBox_2)
        self.groupBox = QtGui.QGroupBox(config_rgb_layer)
        self.groupBox.setObjectName(_fromUtf8("groupBox"))
        self.comboBlue = QtGui.QComboBox(self.groupBox)
        self.comboBlue.setGeometry(QtCore.QRect(10, 20, 380, 26))
        self.comboBlue.setObjectName(_fromUtf8("comboBlue"))
        self.slideMaxBlue = QtGui.QSlider(self.groupBox)
        self.slideMaxBlue.setGeometry(QtCore.QRect(10, 80, 380, 22))
        self.slideMaxBlue.setMaximum(32767)
        self.slideMaxBlue.setOrientation(QtCore.Qt.Horizontal)
        self.slideMaxBlue.setObjectName(_fromUtf8("slideMaxBlue"))
        self.slideMinBlue = QtGui.QSlider(self.groupBox)
        self.slideMinBlue.setGeometry(QtCore.QRect(10, 50, 380, 22))
        self.slideMinBlue.setMaximum(32767)
        self.slideMinBlue.setOrientation(QtCore.Qt.Horizontal)
        self.slideMinBlue.setObjectName(_fromUtf8("slideMinBlue"))
        self.blueGammaLabel = QtGui.QLabel(self.groupBox)
        self.blueGammaLabel.setGeometry(QtCore.QRect(10, 110, 56, 24))
        self.blueGammaLabel.setObjectName(_fromUtf8("blueGammaLabel"))
        self.blueGammaSpinBox = QtGui.QDoubleSpinBox(self.groupBox)
        self.blueGammaSpinBox.setGeometry(QtCore.QRect(70, 110, 62, 24))
        self.blueGammaSpinBox.setObjectName(_fromUtf8("blueGammaSpinBox"))
        self.comboBlue.raise_()
        self.slideMaxBlue.raise_()
        self.slideMinBlue.raise_()
        self.groupBox_3.raise_()
        self.blueGammaLabel.raise_()
        self.blueGammaSpinBox.raise_()
        self.verticalLayout.addWidget(self.groupBox)

        self.retranslateUi(config_rgb_layer)
        QtCore.QMetaObject.connectSlotsByName(config_rgb_layer)

    def retranslateUi(self, config_rgb_layer):
        config_rgb_layer.setWindowTitle(_translate("config_rgb_layer", "Form", None))
        self.groupBox_3.setTitle(_translate("config_rgb_layer", "Red", None))
        self.comboRed.setToolTip(_translate("config_rgb_layer", "Red", None))
        self.comboRed.setStatusTip(_translate("config_rgb_layer", "Red channel", None))
        self.redGammaLabel.setText(_translate("config_rgb_layer", "Gamma: ", None))
        self.groupBox_2.setTitle(_translate("config_rgb_layer", "Green", None))
        self.comboGreen.setToolTip(_translate("config_rgb_layer", "Green", None))
        self.comboGreen.setStatusTip(_translate("config_rgb_layer", "Green channel", None))
        self.greenGammaLabel.setText(_translate("config_rgb_layer", "Gamma: ", None))
        self.groupBox.setTitle(_translate("config_rgb_layer", "Blue", None))
        self.comboBlue.setToolTip(_translate("config_rgb_layer", "Blue", None))
        self.comboBlue.setStatusTip(_translate("config_rgb_layer", "Blue channel", None))
        self.blueGammaLabel.setText(_translate("config_rgb_layer", "Gamma: ", None))

