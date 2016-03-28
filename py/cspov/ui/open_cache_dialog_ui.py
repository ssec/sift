# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'open_cache_dialog.ui'
#
# Created: Mon Mar 28 13:36:49 2016
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

class Ui_openFromCacheDialog(object):
    def setupUi(self, openFromCacheDialog):
        openFromCacheDialog.setObjectName(_fromUtf8("openFromCacheDialog"))
        openFromCacheDialog.resize(593, 427)
        self.verticalLayout = QtGui.QVBoxLayout(openFromCacheDialog)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.label = QtGui.QLabel(openFromCacheDialog)
        self.label.setObjectName(_fromUtf8("label"))
        self.horizontalLayout.addWidget(self.label)
        spacerItem = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.removeFromCacheButton = QtGui.QPushButton(openFromCacheDialog)
        self.removeFromCacheButton.setObjectName(_fromUtf8("removeFromCacheButton"))
        self.horizontalLayout.addWidget(self.removeFromCacheButton)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.commonPathLabel = QtGui.QLabel(openFromCacheDialog)
        self.commonPathLabel.setObjectName(_fromUtf8("commonPathLabel"))
        self.verticalLayout.addWidget(self.commonPathLabel)
        self.cacheListWidget = QtGui.QListWidget(openFromCacheDialog)
        self.cacheListWidget.setMinimumSize(QtCore.QSize(400, 300))
        self.cacheListWidget.setObjectName(_fromUtf8("cacheListWidget"))
        self.verticalLayout.addWidget(self.cacheListWidget)
        self.buttonBox = QtGui.QDialogButtonBox(openFromCacheDialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtGui.QDialogButtonBox.Cancel|QtGui.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName(_fromUtf8("buttonBox"))
        self.verticalLayout.addWidget(self.buttonBox)

        self.retranslateUi(openFromCacheDialog)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL(_fromUtf8("accepted()")), openFromCacheDialog.accept)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL(_fromUtf8("rejected()")), openFromCacheDialog.reject)
        QtCore.QMetaObject.connectSlotsByName(openFromCacheDialog)

    def retranslateUi(self, openFromCacheDialog):
        openFromCacheDialog.setWindowTitle(_translate("openFromCacheDialog", "Open Cached Layers", None))
        self.label.setText(_translate("openFromCacheDialog", "Pre-processed layers stored in cache will load quickly.", None))
        self.removeFromCacheButton.setText(_translate("openFromCacheDialog", "Remove Selected Cached Layers", None))
        self.commonPathLabel.setText(_translate("openFromCacheDialog", "/", None))

