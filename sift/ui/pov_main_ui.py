# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'pov_main.ui'
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

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.resize(1397, 800)
        MainWindow.setAcceptDrops(True)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.verticalLayout_2 = QtGui.QVBoxLayout(self.centralwidget)
        self.verticalLayout_2.setObjectName(_fromUtf8("verticalLayout_2"))
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        self.panZoomToolButton = QtGui.QToolButton(self.centralwidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.panZoomToolButton.sizePolicy().hasHeightForWidth())
        self.panZoomToolButton.setSizePolicy(sizePolicy)
        self.panZoomToolButton.setMinimumSize(QtCore.QSize(0, 0))
        self.panZoomToolButton.setCheckable(True)
        self.panZoomToolButton.setChecked(True)
        self.panZoomToolButton.setAutoExclusive(False)
        self.panZoomToolButton.setAutoRaise(False)
        self.panZoomToolButton.setObjectName(_fromUtf8("panZoomToolButton"))
        self.buttonGroup = QtGui.QButtonGroup(MainWindow)
        self.buttonGroup.setObjectName(_fromUtf8("buttonGroup"))
        self.buttonGroup.addButton(self.panZoomToolButton)
        self.horizontalLayout_2.addWidget(self.panZoomToolButton)
        self.pointSelectButton = QtGui.QToolButton(self.centralwidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pointSelectButton.sizePolicy().hasHeightForWidth())
        self.pointSelectButton.setSizePolicy(sizePolicy)
        self.pointSelectButton.setMinimumSize(QtCore.QSize(0, 0))
        self.pointSelectButton.setCheckable(True)
        self.pointSelectButton.setObjectName(_fromUtf8("pointSelectButton"))
        self.buttonGroup.addButton(self.pointSelectButton)
        self.horizontalLayout_2.addWidget(self.pointSelectButton)
        self.regionSelectButton = QtGui.QToolButton(self.centralwidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.regionSelectButton.sizePolicy().hasHeightForWidth())
        self.regionSelectButton.setSizePolicy(sizePolicy)
        self.regionSelectButton.setCheckable(True)
        self.regionSelectButton.setObjectName(_fromUtf8("regionSelectButton"))
        self.buttonGroup.addButton(self.regionSelectButton)
        self.horizontalLayout_2.addWidget(self.regionSelectButton)
        self.projectionLabel = QtGui.QLabel(self.centralwidget)
        self.projectionLabel.setEnabled(True)
        self.projectionLabel.setObjectName(_fromUtf8("projectionLabel"))
        self.horizontalLayout_2.addWidget(self.projectionLabel)
        self.projectionComboBox = QNoScrollComboBox(self.centralwidget)
        self.projectionComboBox.setEnabled(True)
        self.projectionComboBox.setObjectName(_fromUtf8("projectionComboBox"))
        self.horizontalLayout_2.addWidget(self.projectionComboBox)
        self.line = QtGui.QFrame(self.centralwidget)
        self.line.setFrameShape(QtGui.QFrame.VLine)
        self.line.setFrameShadow(QtGui.QFrame.Sunken)
        self.line.setObjectName(_fromUtf8("line"))
        self.horizontalLayout_2.addWidget(self.line)
        self.cursorProbeText = QtGui.QLabel(self.centralwidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.cursorProbeText.sizePolicy().hasHeightForWidth())
        self.cursorProbeText.setSizePolicy(sizePolicy)
        self.cursorProbeText.setMinimumSize(QtCore.QSize(250, 0))
        self.cursorProbeText.setMaximumSize(QtCore.QSize(256, 16777215))
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(17, 71, 5))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(17, 71, 5))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(17, 71, 5))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(17, 71, 5))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(69, 69, 69))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(69, 69, 69))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Text, brush)
        self.cursorProbeText.setPalette(palette)
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Andale Mono"))
        font.setPointSize(14)
        self.cursorProbeText.setFont(font)
        self.cursorProbeText.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.cursorProbeText.setObjectName(_fromUtf8("cursorProbeText"))
        self.horizontalLayout_2.addWidget(self.cursorProbeText)
        self.cursorProbeLayer = QtGui.QLabel(self.centralwidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.cursorProbeLayer.sizePolicy().hasHeightForWidth())
        self.cursorProbeLayer.setSizePolicy(sizePolicy)
        self.cursorProbeLayer.setMinimumSize(QtCore.QSize(200, 0))
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Andale Mono"))
        font.setPointSize(14)
        self.cursorProbeLayer.setFont(font)
        self.cursorProbeLayer.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.cursorProbeLayer.setObjectName(_fromUtf8("cursorProbeLayer"))
        self.horizontalLayout_2.addWidget(self.cursorProbeLayer)
        self.verticalLayout_2.addLayout(self.horizontalLayout_2)
        self.mainMapWidget = QtGui.QWidget(self.centralwidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.mainMapWidget.sizePolicy().hasHeightForWidth())
        self.mainMapWidget.setSizePolicy(sizePolicy)
        self.mainMapWidget.setObjectName(_fromUtf8("mainMapWidget"))
        self.verticalLayout_5 = QtGui.QVBoxLayout(self.mainMapWidget)
        self.verticalLayout_5.setObjectName(_fromUtf8("verticalLayout_5"))
        self.verticalLayout_3 = QtGui.QVBoxLayout()
        self.verticalLayout_3.setObjectName(_fromUtf8("verticalLayout_3"))
        self.verticalLayout_5.addLayout(self.verticalLayout_3)
        self.verticalLayout_2.addWidget(self.mainMapWidget)
        self.line_2 = QtGui.QFrame(self.centralwidget)
        self.line_2.setFrameShape(QtGui.QFrame.VLine)
        self.line_2.setFrameShadow(QtGui.QFrame.Sunken)
        self.line_2.setObjectName(_fromUtf8("line_2"))
        self.verticalLayout_2.addWidget(self.line_2)
        self.groupBox = QtGui.QGroupBox(self.centralwidget)
        self.groupBox.setTitle(_fromUtf8(""))
        self.groupBox.setObjectName(_fromUtf8("groupBox"))
        self.horizontalLayout = QtGui.QHBoxLayout(self.groupBox)
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.verticalLayout_10 = QtGui.QVBoxLayout()
        self.verticalLayout_10.setObjectName(_fromUtf8("verticalLayout_10"))
        self.progressText = QtGui.QLabel(self.groupBox)
        self.progressText.setMinimumSize(QtCore.QSize(240, 0))
        self.progressText.setMaximumSize(QtCore.QSize(256, 16777215))
        self.progressText.setAlignment(QtCore.Qt.AlignCenter)
        self.progressText.setObjectName(_fromUtf8("progressText"))
        self.verticalLayout_10.addWidget(self.progressText)
        self.progressBar = QtGui.QProgressBar(self.groupBox)
        self.progressBar.setMaximumSize(QtCore.QSize(256, 16777215))
        self.progressBar.setProperty("value", 24)
        self.progressBar.setObjectName(_fromUtf8("progressBar"))
        self.verticalLayout_10.addWidget(self.progressBar)
        self.horizontalLayout.addLayout(self.verticalLayout_10)
        self.verticalLayout_11 = QtGui.QVBoxLayout()
        self.verticalLayout_11.setObjectName(_fromUtf8("verticalLayout_11"))
        self.animationLabel = QtGui.QLabel(self.groupBox)
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Andale Mono"))
        font.setPointSize(14)
        self.animationLabel.setFont(font)
        self.animationLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.animationLabel.setObjectName(_fromUtf8("animationLabel"))
        self.verticalLayout_11.addWidget(self.animationLabel)
        self.animationSlider = QtGui.QSlider(self.groupBox)
        self.animationSlider.setOrientation(QtCore.Qt.Horizontal)
        self.animationSlider.setObjectName(_fromUtf8("animationSlider"))
        self.verticalLayout_11.addWidget(self.animationSlider)
        self.horizontalLayout.addLayout(self.verticalLayout_11)
        self.animBack = QtGui.QToolButton(self.groupBox)
        self.animBack.setToolTip(_fromUtf8(""))
        self.animBack.setObjectName(_fromUtf8("animBack"))
        self.horizontalLayout.addWidget(self.animBack)
        self.animPlayPause = QtGui.QToolButton(self.groupBox)
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Andale Mono"))
        font.setPointSize(18)
        font.setItalic(False)
        self.animPlayPause.setFont(font)
        self.animPlayPause.setObjectName(_fromUtf8("animPlayPause"))
        self.horizontalLayout.addWidget(self.animPlayPause)
        self.animForward = QtGui.QToolButton(self.groupBox)
        self.animForward.setObjectName(_fromUtf8("animForward"))
        self.horizontalLayout.addWidget(self.animForward)
        self.verticalLayout_2.addWidget(self.groupBox)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1397, 22))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        MainWindow.setStatusBar(self.statusbar)
        self.areaProbePane = QtGui.QDockWidget(MainWindow)
        self.areaProbePane.setFeatures(QtGui.QDockWidget.DockWidgetFloatable|QtGui.QDockWidget.DockWidgetMovable)
        self.areaProbePane.setObjectName(_fromUtf8("areaProbePane"))
        self.probeWidget = QtGui.QWidget()
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.probeWidget.sizePolicy().hasHeightForWidth())
        self.probeWidget.setSizePolicy(sizePolicy)
        self.probeWidget.setObjectName(_fromUtf8("probeWidget"))
        self.verticalLayout_12 = QtGui.QVBoxLayout(self.probeWidget)
        self.verticalLayout_12.setObjectName(_fromUtf8("verticalLayout_12"))
        self.probeTabWidget = QtGui.QTabWidget(self.probeWidget)
        self.probeTabWidget.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.probeTabWidget.setObjectName(_fromUtf8("probeTabWidget"))
        self.tab_plus = QtGui.QWidget()
        self.tab_plus.setObjectName(_fromUtf8("tab_plus"))
        self.probeTabWidget.addTab(self.tab_plus, _fromUtf8(""))
        self.verticalLayout_12.addWidget(self.probeTabWidget)
        self.areaProbePane.setWidget(self.probeWidget)
        MainWindow.addDockWidget(QtCore.Qt.DockWidgetArea(2), self.areaProbePane)
        self.layersPane = QtGui.QDockWidget(MainWindow)
        self.layersPane.setFeatures(QtGui.QDockWidget.DockWidgetFloatable|QtGui.QDockWidget.DockWidgetMovable)
        self.layersPane.setObjectName(_fromUtf8("layersPane"))
        self.dockWidgetContents_5 = QtGui.QWidget()
        self.dockWidgetContents_5.setObjectName(_fromUtf8("dockWidgetContents_5"))
        self.verticalLayout = QtGui.QVBoxLayout(self.dockWidgetContents_5)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.layerSetTabs = QtGui.QTabWidget(self.dockWidgetContents_5)
        self.layerSetTabs.setMinimumSize(QtCore.QSize(160, 0))
        self.layerSetTabs.setObjectName(_fromUtf8("layerSetTabs"))
        self.tab_plusL = QtGui.QWidget()
        self.tab_plusL.setObjectName(_fromUtf8("tab_plusL"))
        self.layerSetTabs.addTab(self.tab_plusL, _fromUtf8(""))
        self.verticalLayout.addWidget(self.layerSetTabs)
        self.layersPane.setWidget(self.dockWidgetContents_5)
        MainWindow.addDockWidget(QtCore.Qt.DockWidgetArea(2), self.layersPane)
        self.layerDetailsPane = QtGui.QDockWidget(MainWindow)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.layerDetailsPane.sizePolicy().hasHeightForWidth())
        self.layerDetailsPane.setSizePolicy(sizePolicy)
        self.layerDetailsPane.setFeatures(QtGui.QDockWidget.DockWidgetFloatable|QtGui.QDockWidget.DockWidgetMovable)
        self.layerDetailsPane.setObjectName(_fromUtf8("layerDetailsPane"))
        self.layerInfoContents = QtGui.QWidget()
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.layerInfoContents.sizePolicy().hasHeightForWidth())
        self.layerInfoContents.setSizePolicy(sizePolicy)
        self.layerInfoContents.setObjectName(_fromUtf8("layerInfoContents"))
        self.gridLayout = QtGui.QGridLayout(self.layerInfoContents)
        self.gridLayout.setMargin(0)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.layerDetailsScrollArea = QtGui.QScrollArea(self.layerInfoContents)
        self.layerDetailsScrollArea.setWidgetResizable(True)
        self.layerDetailsScrollArea.setObjectName(_fromUtf8("layerDetailsScrollArea"))
        self.layerDetailsContents = QtGui.QWidget()
        self.layerDetailsContents.setGeometry(QtCore.QRect(0, 0, 195, 76))
        self.layerDetailsContents.setObjectName(_fromUtf8("layerDetailsContents"))
        self.layerDetailsScrollArea.setWidget(self.layerDetailsContents)
        self.gridLayout.addWidget(self.layerDetailsScrollArea, 0, 0, 1, 1)
        self.layerDetailsPane.setWidget(self.layerInfoContents)
        MainWindow.addDockWidget(QtCore.Qt.DockWidgetArea(2), self.layerDetailsPane)
        self.rgbConfigPane = QtGui.QDockWidget(MainWindow)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.rgbConfigPane.sizePolicy().hasHeightForWidth())
        self.rgbConfigPane.setSizePolicy(sizePolicy)
        self.rgbConfigPane.setMaximumSize(QtCore.QSize(800, 800))
        self.rgbConfigPane.setBaseSize(QtCore.QSize(128, 128))
        self.rgbConfigPane.setFeatures(QtGui.QDockWidget.DockWidgetFloatable|QtGui.QDockWidget.DockWidgetMovable)
        self.rgbConfigPane.setObjectName(_fromUtf8("rgbConfigPane"))
        self.dockWidgetContents = QtGui.QWidget()
        self.dockWidgetContents.setObjectName(_fromUtf8("dockWidgetContents"))
        self.verticalLayout_4 = QtGui.QVBoxLayout(self.dockWidgetContents)
        self.verticalLayout_4.setMargin(0)
        self.verticalLayout_4.setObjectName(_fromUtf8("verticalLayout_4"))
        self.scrollArea = QtGui.QScrollArea(self.dockWidgetContents)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName(_fromUtf8("scrollArea"))
        self.scrollAreaWidgetContents_8 = QtGui.QWidget()
        self.scrollAreaWidgetContents_8.setGeometry(QtCore.QRect(0, 0, 225, 524))
        self.scrollAreaWidgetContents_8.setObjectName(_fromUtf8("scrollAreaWidgetContents_8"))
        self.verticalLayout_33 = QtGui.QVBoxLayout(self.scrollAreaWidgetContents_8)
        self.verticalLayout_33.setObjectName(_fromUtf8("verticalLayout_33"))
        self.redGroupBox = QtGui.QGroupBox(self.scrollAreaWidgetContents_8)
        self.redGroupBox.setObjectName(_fromUtf8("redGroupBox"))
        self.verticalLayout_6 = QtGui.QVBoxLayout(self.redGroupBox)
        self.verticalLayout_6.setObjectName(_fromUtf8("verticalLayout_6"))
        self.comboRed = QNoScrollComboBox(self.redGroupBox)
        self.comboRed.setObjectName(_fromUtf8("comboRed"))
        self.verticalLayout_6.addWidget(self.comboRed)
        self.horizontalLayout_3 = QtGui.QHBoxLayout()
        self.horizontalLayout_3.setObjectName(_fromUtf8("horizontalLayout_3"))
        self.slideMinRed = QNoScrollSlider(self.redGroupBox)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.slideMinRed.sizePolicy().hasHeightForWidth())
        self.slideMinRed.setSizePolicy(sizePolicy)
        self.slideMinRed.setMaximum(32767)
        self.slideMinRed.setOrientation(QtCore.Qt.Horizontal)
        self.slideMinRed.setObjectName(_fromUtf8("slideMinRed"))
        self.horizontalLayout_3.addWidget(self.slideMinRed)
        self.editMinRed = QtGui.QLineEdit(self.redGroupBox)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.editMinRed.sizePolicy().hasHeightForWidth())
        self.editMinRed.setSizePolicy(sizePolicy)
        self.editMinRed.setMinimumSize(QtCore.QSize(80, 0))
        self.editMinRed.setObjectName(_fromUtf8("editMinRed"))
        self.horizontalLayout_3.addWidget(self.editMinRed)
        self.verticalLayout_6.addLayout(self.horizontalLayout_3)
        self.horizontalLayout_4 = QtGui.QHBoxLayout()
        self.horizontalLayout_4.setObjectName(_fromUtf8("horizontalLayout_4"))
        self.slideMaxRed = QNoScrollSlider(self.redGroupBox)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.slideMaxRed.sizePolicy().hasHeightForWidth())
        self.slideMaxRed.setSizePolicy(sizePolicy)
        self.slideMaxRed.setMaximum(32767)
        self.slideMaxRed.setOrientation(QtCore.Qt.Horizontal)
        self.slideMaxRed.setObjectName(_fromUtf8("slideMaxRed"))
        self.horizontalLayout_4.addWidget(self.slideMaxRed)
        self.editMaxRed = QtGui.QLineEdit(self.redGroupBox)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.editMaxRed.sizePolicy().hasHeightForWidth())
        self.editMaxRed.setSizePolicy(sizePolicy)
        self.editMaxRed.setMinimumSize(QtCore.QSize(80, 0))
        self.editMaxRed.setObjectName(_fromUtf8("editMaxRed"))
        self.horizontalLayout_4.addWidget(self.editMaxRed)
        self.verticalLayout_6.addLayout(self.horizontalLayout_4)
        self.horizontalLayout_5 = QtGui.QHBoxLayout()
        self.horizontalLayout_5.setObjectName(_fromUtf8("horizontalLayout_5"))
        self.redGammaLabel = QtGui.QLabel(self.redGroupBox)
        self.redGammaLabel.setObjectName(_fromUtf8("redGammaLabel"))
        self.horizontalLayout_5.addWidget(self.redGammaLabel)
        self.redGammaSpinBox = QNoScrollDoubleSpinBox(self.redGroupBox)
        self.redGammaSpinBox.setDecimals(1)
        self.redGammaSpinBox.setMaximum(5.0)
        self.redGammaSpinBox.setSingleStep(0.1)
        self.redGammaSpinBox.setProperty("value", 1.0)
        self.redGammaSpinBox.setObjectName(_fromUtf8("redGammaSpinBox"))
        self.horizontalLayout_5.addWidget(self.redGammaSpinBox)
        self.verticalLayout_6.addLayout(self.horizontalLayout_5)
        self.verticalLayout_33.addWidget(self.redGroupBox)
        self.greenGroupBox = QtGui.QGroupBox(self.scrollAreaWidgetContents_8)
        self.greenGroupBox.setObjectName(_fromUtf8("greenGroupBox"))
        self.verticalLayout_7 = QtGui.QVBoxLayout(self.greenGroupBox)
        self.verticalLayout_7.setObjectName(_fromUtf8("verticalLayout_7"))
        self.comboGreen = QNoScrollComboBox(self.greenGroupBox)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.comboGreen.sizePolicy().hasHeightForWidth())
        self.comboGreen.setSizePolicy(sizePolicy)
        self.comboGreen.setObjectName(_fromUtf8("comboGreen"))
        self.verticalLayout_7.addWidget(self.comboGreen)
        self.horizontalLayout_43 = QtGui.QHBoxLayout()
        self.horizontalLayout_43.setObjectName(_fromUtf8("horizontalLayout_43"))
        self.slideMinGreen = QNoScrollSlider(self.greenGroupBox)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.slideMinGreen.sizePolicy().hasHeightForWidth())
        self.slideMinGreen.setSizePolicy(sizePolicy)
        self.slideMinGreen.setMaximum(32767)
        self.slideMinGreen.setOrientation(QtCore.Qt.Horizontal)
        self.slideMinGreen.setObjectName(_fromUtf8("slideMinGreen"))
        self.horizontalLayout_43.addWidget(self.slideMinGreen)
        self.editMinGreen = QtGui.QLineEdit(self.greenGroupBox)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.editMinGreen.sizePolicy().hasHeightForWidth())
        self.editMinGreen.setSizePolicy(sizePolicy)
        self.editMinGreen.setMinimumSize(QtCore.QSize(80, 0))
        self.editMinGreen.setObjectName(_fromUtf8("editMinGreen"))
        self.horizontalLayout_43.addWidget(self.editMinGreen)
        self.verticalLayout_7.addLayout(self.horizontalLayout_43)
        self.horizontalLayout_44 = QtGui.QHBoxLayout()
        self.horizontalLayout_44.setObjectName(_fromUtf8("horizontalLayout_44"))
        self.slideMaxGreen = QNoScrollSlider(self.greenGroupBox)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.slideMaxGreen.sizePolicy().hasHeightForWidth())
        self.slideMaxGreen.setSizePolicy(sizePolicy)
        self.slideMaxGreen.setMaximum(32767)
        self.slideMaxGreen.setOrientation(QtCore.Qt.Horizontal)
        self.slideMaxGreen.setObjectName(_fromUtf8("slideMaxGreen"))
        self.horizontalLayout_44.addWidget(self.slideMaxGreen)
        self.editMaxGreen = QtGui.QLineEdit(self.greenGroupBox)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.editMaxGreen.sizePolicy().hasHeightForWidth())
        self.editMaxGreen.setSizePolicy(sizePolicy)
        self.editMaxGreen.setMinimumSize(QtCore.QSize(80, 0))
        self.editMaxGreen.setObjectName(_fromUtf8("editMaxGreen"))
        self.horizontalLayout_44.addWidget(self.editMaxGreen)
        self.verticalLayout_7.addLayout(self.horizontalLayout_44)
        self.horizontalLayout_6 = QtGui.QHBoxLayout()
        self.horizontalLayout_6.setObjectName(_fromUtf8("horizontalLayout_6"))
        self.greenGammaLabel = QtGui.QLabel(self.greenGroupBox)
        self.greenGammaLabel.setObjectName(_fromUtf8("greenGammaLabel"))
        self.horizontalLayout_6.addWidget(self.greenGammaLabel)
        self.greenGammaSpinBox = QNoScrollDoubleSpinBox(self.greenGroupBox)
        self.greenGammaSpinBox.setDecimals(1)
        self.greenGammaSpinBox.setMaximum(5.0)
        self.greenGammaSpinBox.setSingleStep(0.1)
        self.greenGammaSpinBox.setProperty("value", 1.0)
        self.greenGammaSpinBox.setObjectName(_fromUtf8("greenGammaSpinBox"))
        self.horizontalLayout_6.addWidget(self.greenGammaSpinBox)
        self.verticalLayout_7.addLayout(self.horizontalLayout_6)
        self.verticalLayout_33.addWidget(self.greenGroupBox)
        self.blueGroupBox = QtGui.QGroupBox(self.scrollAreaWidgetContents_8)
        self.blueGroupBox.setObjectName(_fromUtf8("blueGroupBox"))
        self.verticalLayout_8 = QtGui.QVBoxLayout(self.blueGroupBox)
        self.verticalLayout_8.setObjectName(_fromUtf8("verticalLayout_8"))
        self.comboBlue = QNoScrollComboBox(self.blueGroupBox)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.comboBlue.sizePolicy().hasHeightForWidth())
        self.comboBlue.setSizePolicy(sizePolicy)
        self.comboBlue.setObjectName(_fromUtf8("comboBlue"))
        self.verticalLayout_8.addWidget(self.comboBlue)
        self.horizontalLayout_41 = QtGui.QHBoxLayout()
        self.horizontalLayout_41.setObjectName(_fromUtf8("horizontalLayout_41"))
        self.slideMinBlue = QNoScrollSlider(self.blueGroupBox)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.slideMinBlue.sizePolicy().hasHeightForWidth())
        self.slideMinBlue.setSizePolicy(sizePolicy)
        self.slideMinBlue.setMaximum(32767)
        self.slideMinBlue.setOrientation(QtCore.Qt.Horizontal)
        self.slideMinBlue.setObjectName(_fromUtf8("slideMinBlue"))
        self.horizontalLayout_41.addWidget(self.slideMinBlue)
        self.editMinBlue = QtGui.QLineEdit(self.blueGroupBox)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.editMinBlue.sizePolicy().hasHeightForWidth())
        self.editMinBlue.setSizePolicy(sizePolicy)
        self.editMinBlue.setMinimumSize(QtCore.QSize(80, 0))
        self.editMinBlue.setObjectName(_fromUtf8("editMinBlue"))
        self.horizontalLayout_41.addWidget(self.editMinBlue)
        self.verticalLayout_8.addLayout(self.horizontalLayout_41)
        self.horizontalLayout_42 = QtGui.QHBoxLayout()
        self.horizontalLayout_42.setObjectName(_fromUtf8("horizontalLayout_42"))
        self.slideMaxBlue = QNoScrollSlider(self.blueGroupBox)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.slideMaxBlue.sizePolicy().hasHeightForWidth())
        self.slideMaxBlue.setSizePolicy(sizePolicy)
        self.slideMaxBlue.setMaximum(32767)
        self.slideMaxBlue.setOrientation(QtCore.Qt.Horizontal)
        self.slideMaxBlue.setObjectName(_fromUtf8("slideMaxBlue"))
        self.horizontalLayout_42.addWidget(self.slideMaxBlue)
        self.editMaxBlue = QtGui.QLineEdit(self.blueGroupBox)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.editMaxBlue.sizePolicy().hasHeightForWidth())
        self.editMaxBlue.setSizePolicy(sizePolicy)
        self.editMaxBlue.setMinimumSize(QtCore.QSize(80, 0))
        self.editMaxBlue.setObjectName(_fromUtf8("editMaxBlue"))
        self.horizontalLayout_42.addWidget(self.editMaxBlue)
        self.verticalLayout_8.addLayout(self.horizontalLayout_42)
        self.horizontalLayout_7 = QtGui.QHBoxLayout()
        self.horizontalLayout_7.setObjectName(_fromUtf8("horizontalLayout_7"))
        self.blueGammaLabel = QtGui.QLabel(self.blueGroupBox)
        self.blueGammaLabel.setObjectName(_fromUtf8("blueGammaLabel"))
        self.horizontalLayout_7.addWidget(self.blueGammaLabel)
        self.blueGammaSpinBox = QNoScrollDoubleSpinBox(self.blueGroupBox)
        self.blueGammaSpinBox.setDecimals(1)
        self.blueGammaSpinBox.setMaximum(5.0)
        self.blueGammaSpinBox.setSingleStep(0.1)
        self.blueGammaSpinBox.setProperty("value", 1.0)
        self.blueGammaSpinBox.setObjectName(_fromUtf8("blueGammaSpinBox"))
        self.horizontalLayout_7.addWidget(self.blueGammaSpinBox)
        self.verticalLayout_8.addLayout(self.horizontalLayout_7)
        self.verticalLayout_33.addWidget(self.blueGroupBox)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents_8)
        self.verticalLayout_4.addWidget(self.scrollArea)
        self.rgbConfigPane.setWidget(self.dockWidgetContents)
        MainWindow.addDockWidget(QtCore.Qt.DockWidgetArea(2), self.rgbConfigPane)

        self.retranslateUi(MainWindow)
        self.probeTabWidget.setCurrentIndex(0)
        self.layerSetTabs.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "SIFT Beta |X.X.X|", None))
        self.panZoomToolButton.setToolTip(_translate("MainWindow", "Pan/Zoom Tool", None))
        self.panZoomToolButton.setStatusTip(_translate("MainWindow", "Left mouse click and drag pans the scene", None))
        self.panZoomToolButton.setText(_translate("MainWindow", "Pan/Zoom", None))
        self.pointSelectButton.setToolTip(_translate("MainWindow", "Point Probe Tool", None))
        self.pointSelectButton.setStatusTip(_translate("MainWindow", "Left mouse click probes a single point in the scene", None))
        self.pointSelectButton.setText(_translate("MainWindow", "Point", None))
        self.regionSelectButton.setToolTip(_translate("MainWindow", "Region Probe Tool", None))
        self.regionSelectButton.setStatusTip(_translate("MainWindow", "Left mouse click creates a polygon vertex (click initial vertex to complete)", None))
        self.regionSelectButton.setText(_translate("MainWindow", "Region", None))
        self.projectionLabel.setText(_translate("MainWindow", "Projection:", None))
        self.cursorProbeText.setToolTip(_translate("MainWindow", "Value under cursor", None))
        self.cursorProbeText.setStatusTip(_translate("MainWindow", "Top layer data value under point probe", None))
        self.cursorProbeText.setText(_translate("MainWindow", "N/A", None))
        self.cursorProbeLayer.setText(_translate("MainWindow", "N/A", None))
        self.progressText.setText(_translate("MainWindow", "idle", None))
        self.progressBar.setToolTip(_translate("MainWindow", "Activity Progress", None))
        self.animationLabel.setText(_translate("MainWindow", "HH:MM", None))
        self.animBack.setStatusTip(_translate("MainWindow", "Step backward", None))
        self.animBack.setText(_translate("MainWindow", "|◀", None))
        self.animPlayPause.setStatusTip(_translate("MainWindow", "Start or stop animation", None))
        self.animPlayPause.setText(_translate("MainWindow", "▶", None))
        self.animForward.setStatusTip(_translate("MainWindow", "Step forward", None))
        self.animForward.setText(_translate("MainWindow", "▶|", None))
        self.areaProbePane.setWindowTitle(_translate("MainWindow", "Area Probe Graphs", None))
        self.probeTabWidget.setTabText(self.probeTabWidget.indexOf(self.tab_plus), _translate("MainWindow", "+", None))
        self.layersPane.setWindowTitle(_translate("MainWindow", "Layers", None))
        self.layerSetTabs.setToolTip(_translate("MainWindow", "Alternate layer sets", None))
        self.layerSetTabs.setStatusTip(_translate("MainWindow", "Select alternate layer set", None))
        self.layerSetTabs.setTabText(self.layerSetTabs.indexOf(self.tab_plusL), _translate("MainWindow", "+", None))
        self.layerDetailsPane.setStatusTip(_translate("MainWindow", "Selected layer information", None))
        self.layerDetailsPane.setWindowTitle(_translate("MainWindow", "Layer Details", None))
        self.rgbConfigPane.setWindowTitle(_translate("MainWindow", "RGB Bounds", None))
        self.redGroupBox.setTitle(_translate("MainWindow", "Red", None))
        self.comboRed.setToolTip(_translate("MainWindow", "Red", None))
        self.comboRed.setStatusTip(_translate("MainWindow", "Red channel", None))
        self.slideMinRed.setToolTip(_translate("MainWindow", "value at 0% red", None))
        self.slideMaxRed.setToolTip(_translate("MainWindow", "value at 100% red", None))
        self.redGammaLabel.setText(_translate("MainWindow", "Gamma: ", None))
        self.greenGroupBox.setTitle(_translate("MainWindow", "Green", None))
        self.comboGreen.setToolTip(_translate("MainWindow", "Green", None))
        self.comboGreen.setStatusTip(_translate("MainWindow", "Red channel", None))
        self.slideMinGreen.setToolTip(_translate("MainWindow", "value at 0% green", None))
        self.slideMaxGreen.setToolTip(_translate("MainWindow", "value at 100% green", None))
        self.greenGammaLabel.setText(_translate("MainWindow", "Gamma: ", None))
        self.blueGroupBox.setTitle(_translate("MainWindow", "Blue", None))
        self.comboBlue.setToolTip(_translate("MainWindow", "Red", None))
        self.comboBlue.setStatusTip(_translate("MainWindow", "Red channel", None))
        self.slideMinBlue.setToolTip(_translate("MainWindow", "value at 0% blue", None))
        self.slideMaxBlue.setToolTip(_translate("MainWindow", "value at 100% blue", None))
        self.blueGammaLabel.setText(_translate("MainWindow", "Gamma: ", None))

from sift.ui.custom_widgets import QNoScrollComboBox, QNoScrollDoubleSpinBox, QNoScrollSlider
