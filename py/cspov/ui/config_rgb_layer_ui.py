# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'config_rgb_layer.ui'
#
# Created: Thu Jun 16 11:03:16 2016
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
        config_rgb_layer.resize(400, 300)
        self.verticalLayoutWidget = QtGui.QWidget(config_rgb_layer)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(9, 9, 381, 311))
        self.verticalLayoutWidget.setObjectName(_fromUtf8("verticalLayoutWidget"))
        self.verticalLayout = QtGui.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setMargin(0)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.label = QtGui.QLabel(self.verticalLayoutWidget)
        self.label.setObjectName(_fromUtf8("label"))
        self.verticalLayout.addWidget(self.label)
        self.comboBox_2 = QtGui.QComboBox(self.verticalLayoutWidget)
        self.comboBox_2.setObjectName(_fromUtf8("comboBox_2"))
        self.verticalLayout.addWidget(self.comboBox_2)
        self.horizontalSlider = QtGui.QSlider(self.verticalLayoutWidget)
        self.horizontalSlider.setMaximum(32767)
        self.horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider.setObjectName(_fromUtf8("horizontalSlider"))
        self.verticalLayout.addWidget(self.horizontalSlider)
        self.horizontalSlider_4 = QtGui.QSlider(self.verticalLayoutWidget)
        self.horizontalSlider_4.setMaximum(32767)
        self.horizontalSlider_4.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_4.setObjectName(_fromUtf8("horizontalSlider_4"))
        self.verticalLayout.addWidget(self.horizontalSlider_4)
        self.comboBox = QtGui.QComboBox(self.verticalLayoutWidget)
        self.comboBox.setObjectName(_fromUtf8("comboBox"))
        self.verticalLayout.addWidget(self.comboBox)
        self.horizontalSlider_2 = QtGui.QSlider(self.verticalLayoutWidget)
        self.horizontalSlider_2.setMaximum(32767)
        self.horizontalSlider_2.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_2.setObjectName(_fromUtf8("horizontalSlider_2"))
        self.verticalLayout.addWidget(self.horizontalSlider_2)
        self.horizontalSlider_5 = QtGui.QSlider(self.verticalLayoutWidget)
        self.horizontalSlider_5.setMaximum(32767)
        self.horizontalSlider_5.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_5.setObjectName(_fromUtf8("horizontalSlider_5"))
        self.verticalLayout.addWidget(self.horizontalSlider_5)
        self.comboBox_3 = QtGui.QComboBox(self.verticalLayoutWidget)
        self.comboBox_3.setObjectName(_fromUtf8("comboBox_3"))
        self.verticalLayout.addWidget(self.comboBox_3)
        self.horizontalSlider_3 = QtGui.QSlider(self.verticalLayoutWidget)
        self.horizontalSlider_3.setMaximum(32767)
        self.horizontalSlider_3.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_3.setObjectName(_fromUtf8("horizontalSlider_3"))
        self.verticalLayout.addWidget(self.horizontalSlider_3)
        self.horizontalSlider_6 = QtGui.QSlider(self.verticalLayoutWidget)
        self.horizontalSlider_6.setMaximum(32767)
        self.horizontalSlider_6.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_6.setObjectName(_fromUtf8("horizontalSlider_6"))
        self.verticalLayout.addWidget(self.horizontalSlider_6)

        self.retranslateUi(config_rgb_layer)
        QtCore.QMetaObject.connectSlotsByName(config_rgb_layer)

    def retranslateUi(self, config_rgb_layer):
        config_rgb_layer.setWindowTitle(_translate("config_rgb_layer", "Form", None))
        self.label.setText(_translate("config_rgb_layer", "RGB Layer Channels and Ranges", None))
        self.comboBox_2.setToolTip(_translate("config_rgb_layer", "Red", None))
        self.comboBox_2.setStatusTip(_translate("config_rgb_layer", "Red channel", None))
        self.comboBox.setToolTip(_translate("config_rgb_layer", "Green", None))
        self.comboBox.setStatusTip(_translate("config_rgb_layer", "Green channel", None))
        self.comboBox_3.setToolTip(_translate("config_rgb_layer", "Blue", None))
        self.comboBox_3.setStatusTip(_translate("config_rgb_layer", "Blue channel", None))

