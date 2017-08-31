# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'create_algebraic_dialog.ui'
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

class Ui_create_algebraic_dialog(object):
    def setupUi(self, create_algebraic_dialog):
        create_algebraic_dialog.setObjectName(_fromUtf8("create_algebraic_dialog"))
        create_algebraic_dialog.resize(369, 306)
        self.buttons = QtGui.QDialogButtonBox(create_algebraic_dialog)
        self.buttons.setGeometry(QtCore.QRect(10, 270, 351, 32))
        self.buttons.setOrientation(QtCore.Qt.Horizontal)
        self.buttons.setStandardButtons(QtGui.QDialogButtonBox.Cancel|QtGui.QDialogButtonBox.Ok)
        self.buttons.setObjectName(_fromUtf8("buttons"))
        self.operations_text = QtGui.QPlainTextEdit(create_algebraic_dialog)
        self.operations_text.setGeometry(QtCore.QRect(10, 180, 351, 78))
        self.operations_text.setLineWidth(2)
        self.operations_text.setMidLineWidth(1)
        self.operations_text.setObjectName(_fromUtf8("operations_text"))
        self.formLayoutWidget = QtGui.QWidget(create_algebraic_dialog)
        self.formLayoutWidget.setGeometry(QtCore.QRect(10, 10, 351, 143))
        self.formLayoutWidget.setObjectName(_fromUtf8("formLayoutWidget"))
        self.formLayout = QtGui.QFormLayout(self.formLayoutWidget)
        self.formLayout.setFieldGrowthPolicy(QtGui.QFormLayout.AllNonFixedFieldsGrow)
        self.formLayout.setLabelAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.formLayout.setFormAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.formLayout.setObjectName(_fromUtf8("formLayout"))
        self.name_label = QtGui.QLabel(self.formLayoutWidget)
        self.name_label.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.name_label.setObjectName(_fromUtf8("name_label"))
        self.formLayout.setWidget(0, QtGui.QFormLayout.LabelRole, self.name_label)
        self.layer_name_edit = QtGui.QLineEdit(self.formLayoutWidget)
        self.layer_name_edit.setMaxLength(35)
        self.layer_name_edit.setObjectName(_fromUtf8("layer_name_edit"))
        self.formLayout.setWidget(0, QtGui.QFormLayout.FieldRole, self.layer_name_edit)
        self.operation_label = QtGui.QLabel(self.formLayoutWidget)
        self.operation_label.setObjectName(_fromUtf8("operation_label"))
        self.formLayout.setWidget(1, QtGui.QFormLayout.LabelRole, self.operation_label)
        self.operation_combo = QtGui.QComboBox(self.formLayoutWidget)
        self.operation_combo.setFrame(True)
        self.operation_combo.setObjectName(_fromUtf8("operation_combo"))
        self.formLayout.setWidget(1, QtGui.QFormLayout.FieldRole, self.operation_combo)
        self.x_label = QtGui.QLabel(self.formLayoutWidget)
        self.x_label.setObjectName(_fromUtf8("x_label"))
        self.formLayout.setWidget(2, QtGui.QFormLayout.LabelRole, self.x_label)
        self.x_combo = QtGui.QComboBox(self.formLayoutWidget)
        self.x_combo.setObjectName(_fromUtf8("x_combo"))
        self.formLayout.setWidget(2, QtGui.QFormLayout.FieldRole, self.x_combo)
        self.y_label = QtGui.QLabel(self.formLayoutWidget)
        self.y_label.setObjectName(_fromUtf8("y_label"))
        self.formLayout.setWidget(3, QtGui.QFormLayout.LabelRole, self.y_label)
        self.y_combo = QtGui.QComboBox(self.formLayoutWidget)
        self.y_combo.setObjectName(_fromUtf8("y_combo"))
        self.formLayout.setWidget(3, QtGui.QFormLayout.FieldRole, self.y_combo)
        self.z_label = QtGui.QLabel(self.formLayoutWidget)
        self.z_label.setObjectName(_fromUtf8("z_label"))
        self.formLayout.setWidget(4, QtGui.QFormLayout.LabelRole, self.z_label)
        self.z_combo = QtGui.QComboBox(self.formLayoutWidget)
        self.z_combo.setObjectName(_fromUtf8("z_combo"))
        self.formLayout.setWidget(4, QtGui.QFormLayout.FieldRole, self.z_combo)
        self.operations_label = QtGui.QLabel(create_algebraic_dialog)
        self.operations_label.setGeometry(QtCore.QRect(10, 160, 71, 20))
        font = QtGui.QFont()
        font.setBold(False)
        font.setUnderline(True)
        font.setWeight(50)
        self.operations_label.setFont(font)
        self.operations_label.setObjectName(_fromUtf8("operations_label"))

        self.retranslateUi(create_algebraic_dialog)
        QtCore.QObject.connect(self.buttons, QtCore.SIGNAL(_fromUtf8("accepted()")), create_algebraic_dialog.accept)
        QtCore.QObject.connect(self.buttons, QtCore.SIGNAL(_fromUtf8("rejected()")), create_algebraic_dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(create_algebraic_dialog)

    def retranslateUi(self, create_algebraic_dialog):
        create_algebraic_dialog.setWindowTitle(_translate("create_algebraic_dialog", "Create Algebraic Layer", None))
        self.operations_text.setPlainText(_translate("create_algebraic_dialog", "result = x - y", None))
        self.name_label.setText(_translate("create_algebraic_dialog", "Name:", None))
        self.operation_label.setText(_translate("create_algebraic_dialog", "Operation:", None))
        self.x_label.setText(_translate("create_algebraic_dialog", "x:", None))
        self.y_label.setText(_translate("create_algebraic_dialog", "y:", None))
        self.z_label.setText(_translate("create_algebraic_dialog", "z:", None))
        self.operations_label.setText(_translate("create_algebraic_dialog", "Operations", None))

