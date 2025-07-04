# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'export_image_dialog.ui'
#
# Created by: PyQt5 UI code generator 5.15.11
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_ExportImageDialog(object):
    def setupUi(self, ExportImageDialog):
        ExportImageDialog.setObjectName("ExportImageDialog")
        ExportImageDialog.resize(272, 580)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(ExportImageDialog.sizePolicy().hasHeightForWidth())
        ExportImageDialog.setSizePolicy(sizePolicy)
        ExportImageDialog.setStyleSheet("")
        self.buttonBox = QtWidgets.QDialogButtonBox(ExportImageDialog)
        self.buttonBox.setEnabled(True)
        self.buttonBox.setGeometry(QtCore.QRect(110, 540, 161, 32))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel | QtWidgets.QDialogButtonBox.Save)
        self.buttonBox.setCenterButtons(False)
        self.buttonBox.setObjectName("buttonBox")
        self.frameRangeGroupBox = QtWidgets.QGroupBox(ExportImageDialog)
        self.frameRangeGroupBox.setGeometry(QtCore.QRect(10, 30, 251, 111))
        self.frameRangeGroupBox.setAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self.frameRangeGroupBox.setFlat(False)
        self.frameRangeGroupBox.setCheckable(False)
        self.frameRangeGroupBox.setObjectName("frameRangeGroupBox")
        self.frameCurrentRadio = QtWidgets.QRadioButton(self.frameRangeGroupBox)
        self.frameCurrentRadio.setGeometry(QtCore.QRect(10, 20, 121, 21))
        self.frameCurrentRadio.setChecked(True)
        self.frameCurrentRadio.setObjectName("frameCurrentRadio")
        self.frameAllRadio = QtWidgets.QRadioButton(self.frameRangeGroupBox)
        self.frameAllRadio.setGeometry(QtCore.QRect(10, 50, 100, 21))
        self.frameAllRadio.setObjectName("frameAllRadio")
        self.frameRangeRadio = QtWidgets.QRadioButton(self.frameRangeGroupBox)
        self.frameRangeRadio.setGeometry(QtCore.QRect(10, 80, 71, 21))
        self.frameRangeRadio.setObjectName("frameRangeRadio")
        self.label = QtWidgets.QLabel(self.frameRangeGroupBox)
        self.label.setGeometry(QtCore.QRect(80, 80, 41, 21))
        self.label.setAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.label.setObjectName("label")
        self.frameRangeFrom = QtWidgets.QLineEdit(self.frameRangeGroupBox)
        self.frameRangeFrom.setGeometry(QtCore.QRect(120, 80, 31, 21))
        self.frameRangeFrom.setText("")
        self.frameRangeFrom.setReadOnly(False)
        self.frameRangeFrom.setPlaceholderText("")
        self.frameRangeFrom.setProperty("clearButtonEnabled", False)
        self.frameRangeFrom.setObjectName("frameRangeFrom")
        self.label_2 = QtWidgets.QLabel(self.frameRangeGroupBox)
        self.label_2.setGeometry(QtCore.QRect(160, 80, 21, 21))
        self.label_2.setObjectName("label_2")
        self.frameRangeTo = QtWidgets.QLineEdit(self.frameRangeGroupBox)
        self.frameRangeTo.setGeometry(QtCore.QRect(180, 80, 31, 21))
        self.frameRangeTo.setObjectName("frameRangeTo")
        self.saveAsLineEdit = QtWidgets.QLineEdit(ExportImageDialog)
        self.saveAsLineEdit.setGeometry(QtCore.QRect(10, 10, 221, 21))
        self.saveAsLineEdit.setText("")
        self.saveAsLineEdit.setObjectName("saveAsLineEdit")
        self.saveAsButton = QtWidgets.QPushButton(ExportImageDialog)
        self.saveAsButton.setGeometry(QtCore.QRect(230, 10, 31, 21))
        self.saveAsButton.setObjectName("saveAsButton")
        self.animationGroupBox = QtWidgets.QGroupBox(ExportImageDialog)
        self.animationGroupBox.setGeometry(QtCore.QRect(10, 140, 141, 61))
        self.animationGroupBox.setObjectName("animationGroupBox")
        self.loopRadio = QtWidgets.QRadioButton(self.animationGroupBox)
        self.loopRadio.setGeometry(QtCore.QRect(10, 30, 61, 21))
        self.loopRadio.setChecked(True)
        self.loopRadio.setObjectName("loopRadio")
        self.rockRadio = QtWidgets.QRadioButton(self.animationGroupBox)
        self.rockRadio.setGeometry(QtCore.QRect(80, 30, 61, 18))
        self.rockRadio.setObjectName("rockRadio")
        self.frameDelayGroup = QtWidgets.QGroupBox(ExportImageDialog)
        self.frameDelayGroup.setGeometry(QtCore.QRect(10, 210, 251, 131))
        self.frameDelayGroup.setObjectName("frameDelayGroup")
        self.timeLapseRadio = QtWidgets.QRadioButton(self.frameDelayGroup)
        self.timeLapseRadio.setGeometry(QtCore.QRect(10, 30, 97, 21))
        self.timeLapseRadio.setChecked(True)
        self.timeLapseRadio.setObjectName("timeLapseRadio")
        self.constantDelayRadio = QtWidgets.QRadioButton(self.frameDelayGroup)
        self.constantDelayRadio.setGeometry(QtCore.QRect(10, 60, 81, 21))
        self.constantDelayRadio.setObjectName("constantDelayRadio")
        self.constantDelaySpin = QtWidgets.QSpinBox(self.frameDelayGroup)
        self.constantDelaySpin.setGeometry(QtCore.QRect(90, 60, 81, 21))
        self.constantDelaySpin.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.constantDelaySpin.setFrame(True)
        self.constantDelaySpin.setButtonSymbols(QtWidgets.QAbstractSpinBox.UpDownArrows)
        self.constantDelaySpin.setAccelerated(True)
        self.constantDelaySpin.setKeyboardTracking(True)
        self.constantDelaySpin.setMinimum(25)
        self.constantDelaySpin.setMaximum(5000)
        self.constantDelaySpin.setObjectName("constantDelaySpin")
        self.fpsDelayRadio = QtWidgets.QRadioButton(self.frameDelayGroup)
        self.fpsDelayRadio.setGeometry(QtCore.QRect(10, 90, 81, 21))
        self.fpsDelayRadio.setObjectName("fpsDelayRadio")
        self.fpsDelaySpin = QtWidgets.QSpinBox(self.frameDelayGroup)
        self.fpsDelaySpin.setGeometry(QtCore.QRect(90, 90, 81, 21))
        self.fpsDelaySpin.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.fpsDelaySpin.setFrame(True)
        self.fpsDelaySpin.setButtonSymbols(QtWidgets.QAbstractSpinBox.UpDownArrows)
        self.fpsDelaySpin.setAccelerated(True)
        self.fpsDelaySpin.setKeyboardTracking(True)
        self.fpsDelaySpin.setSuffix("")
        self.fpsDelaySpin.setMinimum(1)
        self.fpsDelaySpin.setMaximum(60)
        self.fpsDelaySpin.setProperty("value", 1)
        self.fpsDelaySpin.setObjectName("fpsDelaySpin")
        self.frameDelayGroup_2 = QtWidgets.QGroupBox(ExportImageDialog)
        self.frameDelayGroup_2.setGeometry(QtCore.QRect(10, 340, 251, 61))
        self.frameDelayGroup_2.setObjectName("frameDelayGroup_2")
        self.colorbarNoneRadio = QtWidgets.QRadioButton(self.frameDelayGroup_2)
        self.colorbarNoneRadio.setGeometry(QtCore.QRect(10, 30, 61, 21))
        self.colorbarNoneRadio.setChecked(True)
        self.colorbarNoneRadio.setObjectName("colorbarNoneRadio")
        self.colorbarHorizontalRadio = QtWidgets.QRadioButton(self.frameDelayGroup_2)
        self.colorbarHorizontalRadio.setGeometry(QtCore.QRect(80, 30, 91, 21))
        self.colorbarHorizontalRadio.setObjectName("colorbarHorizontalRadio")
        self.colorbarVerticalRadio = QtWidgets.QRadioButton(self.frameDelayGroup_2)
        self.colorbarVerticalRadio.setGeometry(QtCore.QRect(170, 30, 81, 21))
        self.colorbarVerticalRadio.setObjectName("colorbarVerticalRadio")
        self.outputResolutionGroup = QtWidgets.QGroupBox(ExportImageDialog)
        self.outputResolutionGroup.setGeometry(QtCore.QRect(10, 410, 251, 91))
        self.outputResolutionGroup.setObjectName("outputResolutionGroup")
        self.resolutionDefaultRadio = QtWidgets.QRadioButton(self.outputResolutionGroup)
        self.resolutionDefaultRadio.setGeometry(QtCore.QRect(10, 30, 71, 21))
        self.resolutionDefaultRadio.setChecked(True)
        self.resolutionDefaultRadio.setObjectName("resolutionDefaultRadio")
        self.resolutionOptimalRadio = QtWidgets.QRadioButton(self.outputResolutionGroup)
        self.resolutionOptimalRadio.setGeometry(QtCore.QRect(90, 30, 71, 21))
        self.resolutionOptimalRadio.setObjectName("resolutionOptimalRadio")
        self.resolutionCustomRadio = QtWidgets.QRadioButton(self.outputResolutionGroup)
        self.resolutionCustomRadio.setGeometry(QtCore.QRect(170, 30, 71, 21))
        self.resolutionCustomRadio.setObjectName("resolutionCustomRadio")
        self.customSizeLabel = QtWidgets.QLabel(self.outputResolutionGroup)
        self.customSizeLabel.setGeometry(QtCore.QRect(10, 60, 81, 21))
        self.customSizeLabel.setObjectName("customSizeLabel")
        self.customWidthEdit = QtWidgets.QLineEdit(self.outputResolutionGroup)
        self.customWidthEdit.setGeometry(QtCore.QRect(90, 60, 61, 21))
        self.customWidthEdit.setObjectName("customWidthEdit")
        self.customSizeSeparatorLabel = QtWidgets.QLabel(self.outputResolutionGroup)
        self.customSizeSeparatorLabel.setGeometry(QtCore.QRect(155, 60, 16, 21))
        self.customSizeSeparatorLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.customSizeSeparatorLabel.setObjectName("customSizeSeparatorLabel")
        self.customHeightEdit = QtWidgets.QLineEdit(self.outputResolutionGroup)
        self.customHeightEdit.setGeometry(QtCore.QRect(170, 60, 61, 21))
        self.customHeightEdit.setObjectName("customHeightEdit")
        self.layoutWidget = QtWidgets.QWidget(ExportImageDialog)
        self.layoutWidget.setGeometry(QtCore.QRect(10, 500, 255, 33))
        self.layoutWidget.setObjectName("layoutWidget")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.layoutWidget)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.includeFooterCheckbox = QtWidgets.QCheckBox(self.layoutWidget)
        self.includeFooterCheckbox.setChecked(True)
        self.includeFooterCheckbox.setObjectName("includeFooterCheckbox")
        self.horizontalLayout_2.addWidget(self.includeFooterCheckbox)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem1)
        self.label_3 = QtWidgets.QLabel(self.layoutWidget)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout.addWidget(self.label_3)
        self.footerFontSizeSpinBox = QtWidgets.QSpinBox(self.layoutWidget)
        self.footerFontSizeSpinBox.setMinimum(8)
        self.footerFontSizeSpinBox.setMaximum(72)
        self.footerFontSizeSpinBox.setProperty("value", 11)
        self.footerFontSizeSpinBox.setObjectName("footerFontSizeSpinBox")
        self.horizontalLayout.addWidget(self.footerFontSizeSpinBox)
        self.horizontalLayout_2.addLayout(self.horizontalLayout)

        self.retranslateUi(ExportImageDialog)
        self.buttonBox.accepted.connect(ExportImageDialog.accept)  # type: ignore
        self.buttonBox.rejected.connect(ExportImageDialog.reject)  # type: ignore
        QtCore.QMetaObject.connectSlotsByName(ExportImageDialog)

    def retranslateUi(self, ExportImageDialog):
        _translate = QtCore.QCoreApplication.translate
        ExportImageDialog.setWindowTitle(_translate("ExportImageDialog", "Export Image"))
        self.frameRangeGroupBox.setTitle(_translate("ExportImageDialog", "Frame Range"))
        self.frameCurrentRadio.setText(_translate("ExportImageDialog", "Current"))
        self.frameAllRadio.setText(_translate("ExportImageDialog", "All"))
        self.frameRangeRadio.setText(_translate("ExportImageDialog", "Frames"))
        self.label.setText(_translate("ExportImageDialog", "from:"))
        self.label_2.setText(_translate("ExportImageDialog", "to:"))
        self.saveAsButton.setText(_translate("ExportImageDialog", "..."))
        self.animationGroupBox.setTitle(_translate("ExportImageDialog", "Animation Type"))
        self.loopRadio.setText(_translate("ExportImageDialog", "Loop"))
        self.rockRadio.setText(_translate("ExportImageDialog", "Rock"))
        self.frameDelayGroup.setTitle(_translate("ExportImageDialog", "Frame Delay"))
        self.timeLapseRadio.setToolTip(_translate("ExportImageDialog", "delay based on dataset observation time"))
        self.timeLapseRadio.setText(_translate("ExportImageDialog", "Time Lapse"))
        self.constantDelayRadio.setText(_translate("ExportImageDialog", "Constant:"))
        self.constantDelaySpin.setSuffix(_translate("ExportImageDialog", "ms"))
        self.fpsDelayRadio.setText(_translate("ExportImageDialog", "FPS:"))
        self.frameDelayGroup_2.setTitle(_translate("ExportImageDialog", "Export Colorbar"))
        self.colorbarNoneRadio.setText(_translate("ExportImageDialog", "None"))
        self.colorbarHorizontalRadio.setText(_translate("ExportImageDialog", "Horizontal"))
        self.colorbarVerticalRadio.setText(_translate("ExportImageDialog", "Vertical"))
        self.outputResolutionGroup.setTitle(_translate("ExportImageDialog", "Output resolution"))
        self.resolutionDefaultRadio.setText(_translate("ExportImageDialog", "Default"))
        self.resolutionOptimalRadio.setText(_translate("ExportImageDialog", "Optimal"))
        self.resolutionCustomRadio.setText(_translate("ExportImageDialog", "Custom"))
        self.customSizeLabel.setText(_translate("ExportImageDialog", "Image size:"))
        self.customSizeSeparatorLabel.setText(_translate("ExportImageDialog", "x"))
        self.includeFooterCheckbox.setText(_translate("ExportImageDialog", "Include Footer"))
        self.label_3.setText(_translate("ExportImageDialog", "Font Size:"))
        self.footerFontSizeSpinBox.setSuffix(_translate("ExportImageDialog", "px"))
