# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'config_rgb_layer.ui'
#
# Created: Thu Jun 16 11:22:43 2016
#      by: PyQt4 UI code generator 4.10.4
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
        config_rgb_layer.resize(403, 352)
        self.verticalLayoutWidget = QtGui.QWidget(config_rgb_layer)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(9, 9, 381, 311))
        self.verticalLayoutWidget.setObjectName(_fromUtf8("verticalLayoutWidget"))
        self.verticalLayout = QtGui.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setMargin(0)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.label = QtGui.QLabel(self.verticalLayoutWidget)
        self.label.setObjectName(_fromUtf8("label"))
        self.verticalLayout.addWidget(self.label)
        self.comboRed = QtGui.QComboBox(self.verticalLayoutWidget)
        self.comboRed.setObjectName(_fromUtf8("comboRed"))
        self.verticalLayout.addWidget(self.comboRed)
        self.slideMinRed = QtGui.QSlider(self.verticalLayoutWidget)
        self.slideMinRed.setMaximum(32767)
        self.slideMinRed.setOrientation(QtCore.Qt.Horizontal)
        self.slideMinRed.setObjectName(_fromUtf8("slideMinRed"))
        self.verticalLayout.addWidget(self.slideMinRed)
        self.slideMaxRed = QtGui.QSlider(self.verticalLayoutWidget)
        self.slideMaxRed.setMaximum(32767)
        self.slideMaxRed.setOrientation(QtCore.Qt.Horizontal)
        self.slideMaxRed.setObjectName(_fromUtf8("slideMaxRed"))
        self.verticalLayout.addWidget(self.slideMaxRed)
        self.comboGreen = QtGui.QComboBox(self.verticalLayoutWidget)
        self.comboGreen.setObjectName(_fromUtf8("comboGreen"))
        self.verticalLayout.addWidget(self.comboGreen)
        self.slideMinGreen = QtGui.QSlider(self.verticalLayoutWidget)
        self.slideMinGreen.setMaximum(32767)
        self.slideMinGreen.setOrientation(QtCore.Qt.Horizontal)
        self.slideMinGreen.setObjectName(_fromUtf8("slideMinGreen"))
        self.verticalLayout.addWidget(self.slideMinGreen)
        self.slideMaxGreen = QtGui.QSlider(self.verticalLayoutWidget)
        self.slideMaxGreen.setMaximum(32767)
        self.slideMaxGreen.setOrientation(QtCore.Qt.Horizontal)
        self.slideMaxGreen.setObjectName(_fromUtf8("slideMaxGreen"))
        self.verticalLayout.addWidget(self.slideMaxGreen)
        self.comboBlue = QtGui.QComboBox(self.verticalLayoutWidget)
        self.comboBlue.setObjectName(_fromUtf8("comboBlue"))
        self.verticalLayout.addWidget(self.comboBlue)
        self.slideMinBlue = QtGui.QSlider(self.verticalLayoutWidget)
        self.slideMinBlue.setMaximum(32767)
        self.slideMinBlue.setOrientation(QtCore.Qt.Horizontal)
        self.slideMinBlue.setObjectName(_fromUtf8("slideMinBlue"))
        self.verticalLayout.addWidget(self.slideMinBlue)
        self.slideMaxBlue = QtGui.QSlider(self.verticalLayoutWidget)
        self.slideMaxBlue.setMaximum(32767)
        self.slideMaxBlue.setOrientation(QtCore.Qt.Horizontal)
        self.slideMaxBlue.setObjectName(_fromUtf8("slideMaxBlue"))
        self.verticalLayout.addWidget(self.slideMaxBlue)

        self.retranslateUi(config_rgb_layer)
        QtCore.QMetaObject.connectSlotsByName(config_rgb_layer)

    def retranslateUi(self, config_rgb_layer):
        config_rgb_layer.setWindowTitle(_translate("config_rgb_layer", "Form", None))
        self.label.setText(_translate("config_rgb_layer", "RGB Layer Channels and Ranges", None))
        self.comboRed.setToolTip(_translate("config_rgb_layer", "Red", None))
        self.comboRed.setStatusTip(_translate("config_rgb_layer", "Red channel", None))
        self.comboGreen.setToolTip(_translate("config_rgb_layer", "Green", None))
        self.comboGreen.setStatusTip(_translate("config_rgb_layer", "Green channel", None))
        self.comboBlue.setToolTip(_translate("config_rgb_layer", "Blue", None))
        self.comboBlue.setStatusTip(_translate("config_rgb_layer", "Blue channel", None))

