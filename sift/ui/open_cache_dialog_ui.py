# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'open_cache_dialog.ui'
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


class Ui_openFromCacheDialog(object):
    def setupUi(self, openFromCacheDialog):
        openFromCacheDialog.setObjectName("openFromCacheDialog")
        openFromCacheDialog.resize(593, 427)
        self.verticalLayout = QtWidgets.QVBoxLayout(openFromCacheDialog)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(openFromCacheDialog)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.removeFromCacheButton = QtWidgets.QPushButton(openFromCacheDialog)
        self.removeFromCacheButton.setObjectName("removeFromCacheButton")
        self.horizontalLayout.addWidget(self.removeFromCacheButton)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.cacheListWidget = QtWidgets.QListWidget(openFromCacheDialog)
        self.cacheListWidget.setMinimumSize(QtCore.QSize(400, 300))
        self.cacheListWidget.setObjectName("cacheListWidget")
        self.verticalLayout.addWidget(self.cacheListWidget)
        self.buttonBox = QtWidgets.QDialogButtonBox(openFromCacheDialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel | QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.verticalLayout.addWidget(self.buttonBox)

        self.retranslateUi(openFromCacheDialog)
        self.buttonBox.accepted.connect(openFromCacheDialog.accept)
        self.buttonBox.rejected.connect(openFromCacheDialog.reject)
        QtCore.QMetaObject.connectSlotsByName(openFromCacheDialog)

    def retranslateUi(self, openFromCacheDialog):
        openFromCacheDialog.setWindowTitle(_translate("openFromCacheDialog", "Open Cached Layers", None))
        self.label.setText(
            _translate("openFromCacheDialog", "Pre-processed layers stored in cache will load quickly.", None))
        self.removeFromCacheButton.setText(_translate("openFromCacheDialog", "Remove Selected Cached Layers", None))
