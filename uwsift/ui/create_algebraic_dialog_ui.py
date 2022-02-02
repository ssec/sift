# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'create_algebraic_dialog.ui'
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


class Ui_create_algebraic_dialog(object):
    def setupUi(self, create_algebraic_dialog):
        create_algebraic_dialog.setObjectName("create_algebraic_dialog")
        create_algebraic_dialog.resize(369, 306)
        self.buttons = QtWidgets.QDialogButtonBox(create_algebraic_dialog)
        self.buttons.setGeometry(QtCore.QRect(10, 270, 351, 32))
        self.buttons.setOrientation(QtCore.Qt.Horizontal)
        self.buttons.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel | QtWidgets.QDialogButtonBox.Ok)
        self.buttons.setObjectName("buttons")
        self.operations_text = QtWidgets.QPlainTextEdit(create_algebraic_dialog)
        self.operations_text.setGeometry(QtCore.QRect(10, 180, 351, 78))
        self.operations_text.setLineWidth(2)
        self.operations_text.setMidLineWidth(1)
        self.operations_text.setObjectName("operations_text")
        self.formLayoutWidget = QtWidgets.QWidget(create_algebraic_dialog)
        self.formLayoutWidget.setGeometry(QtCore.QRect(10, 10, 351, 143))
        self.formLayoutWidget.setObjectName("formLayoutWidget")
        self.formLayout = QtWidgets.QFormLayout(self.formLayoutWidget)
        self.formLayout.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        self.formLayout.setLabelAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.formLayout.setFormAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self.formLayout.setObjectName("formLayout")
        self.name_label = QtWidgets.QLabel(self.formLayoutWidget)
        self.name_label.setAlignment(int(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter))
        self.name_label.setObjectName("name_label")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.name_label)
        self.layer_name_edit = QtWidgets.QLineEdit(self.formLayoutWidget)
        self.layer_name_edit.setMaxLength(35)
        self.layer_name_edit.setObjectName("layer_name_edit")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.layer_name_edit)
        self.operation_label = QtWidgets.QLabel(self.formLayoutWidget)
        self.operation_label.setObjectName("operation_label")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.operation_label)
        self.operation_combo = QtWidgets.QComboBox(self.formLayoutWidget)
        self.operation_combo.setFrame(True)
        self.operation_combo.setObjectName("operation_combo")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.operation_combo)
        self.x_label = QtWidgets.QLabel(self.formLayoutWidget)
        self.x_label.setObjectName("x_label")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.x_label)
        self.x_combo = QtWidgets.QComboBox(self.formLayoutWidget)
        self.x_combo.setObjectName("x_combo")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.x_combo)
        self.y_label = QtWidgets.QLabel(self.formLayoutWidget)
        self.y_label.setObjectName("y_label")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.y_label)
        self.y_combo = QtWidgets.QComboBox(self.formLayoutWidget)
        self.y_combo.setObjectName("y_combo")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.y_combo)
        self.z_label = QtWidgets.QLabel(self.formLayoutWidget)
        self.z_label.setObjectName("z_label")
        self.formLayout.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.z_label)
        self.z_combo = QtWidgets.QComboBox(self.formLayoutWidget)
        self.z_combo.setObjectName("z_combo")
        self.formLayout.setWidget(4, QtWidgets.QFormLayout.FieldRole, self.z_combo)
        self.operations_label = QtWidgets.QLabel(create_algebraic_dialog)
        self.operations_label.setGeometry(QtCore.QRect(10, 160, 71, 20))
        font = QtGui.QFont()
        font.setBold(False)
        font.setUnderline(True)
        font.setWeight(50)
        self.operations_label.setFont(font)
        self.operations_label.setObjectName("operations_label")

        self.retranslateUi(create_algebraic_dialog)
        self.buttons.accepted.connect(create_algebraic_dialog.accept)
        self.buttons.rejected.connect(create_algebraic_dialog.reject)
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
