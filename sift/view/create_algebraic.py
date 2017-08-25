import ast
from PyQt4 import QtGui, QtCore
from sift.common import KIND, INFO
from sift.ui.create_algebraic_dialog_ui import Ui_create_algebraic_dialog


DIFF_OP_NAME = 'Difference'
NDI_OP_NAME = 'Normalized Difference Index'
CUSTOM_OP_NAME = "Custom..."
PRESET_OPERATIONS = {
    DIFF_OP_NAME: ('result = x - y', 2),
    NDI_OP_NAME: ('result = (x - y) / (x + y)', 2),
}


class CreateAlgebraicDialog(QtGui.QDialog):
    def __init__(self, doc, selected_uuids=None, parent=None):
        super(QtGui.QDialog, self).__init__(parent)
        self.ui = Ui_create_algebraic_dialog()
        self.ui.setupUi(self)
        # NOTE: For now disable the operations area
        self.ui.operations_text.setDisabled(True)
        self._name_validator = QtGui.QRegExpValidator()
        # allow names that don't start with numbers but accept all other "word" characters
        self._name_validator.setRegExp(QtCore.QRegExp(r'^\w[-\+\*/\w\s]+'))
        self.ui.layer_name_edit.setValidator(self._name_validator)
        self.doc = doc

        # setup UI elements
        available_info = {}
        available_short_names = []
        selected_short_names = []
        # use DATASET_NAME as unique group identifier
        for idx, prez, layer in self.doc.current_layers_where(kinds=(KIND.IMAGE, KIND.COMPOSITE)):
            # use the UUID as a representative when talking to the document
            available_info.setdefault(layer[INFO.SHORT_NAME],
                                      layer[INFO.UUID])
            available_short_names.append(layer[INFO.SHORT_NAME])
            if layer[INFO.UUID] in selected_uuids:
                selected_short_names.append(layer[INFO.SHORT_NAME])

        self.ui.layer_name_edit.textEdited.connect(self._validate)
        self.accepted.connect(self._create_algebraic)
        for idx, combo in enumerate(self.layer_combos):
            if len(selected_short_names) > idx:
                a_select = selected_short_names[idx]
            elif len(available_info) > idx:
                a_select = available_short_names[idx]
            else:
                a_select = None
            self._init_combo(combo, available_info, a_select)
            combo.currentIndexChanged.connect(self._validate)

        for op_name in PRESET_OPERATIONS.keys():
            self.ui.operation_combo.addItem(op_name)
        self.ui.operation_combo.addItem(CUSTOM_OP_NAME)
        self.ui.operation_combo.setCurrentIndex(0)
        self.ui.operation_combo.currentIndexChanged.connect(self._set_operation)
        self._custom_formula = PRESET_OPERATIONS[DIFF_OP_NAME][0]

        self._set_operation()
        self._validate()

    @property
    def layer_combos(self):
        return (
            self.ui.x_combo,
            self.ui.y_combo,
            self.ui.z_combo,
        )

    @property
    def layer_combos_names(self):
        return zip(
            ('x', 'y', 'z'),
            self.layer_combos,
        )

    def _set_operation(self, *args, **kwargs):
        was_custom = self.ui.operations_text.isEnabled()
        if was_custom:
            self._custom_formula = self.ui.operations_text.toPlainText()

        operation = self.ui.operation_combo.currentText()
        op_formula, num_layers = PRESET_OPERATIONS.get(operation, (None, 3))
        if op_formula is None:
            op_formula = self._custom_formula
            self.ui.operations_text.setDisabled(False)
        else:
            self.ui.operations_text.setDisabled(True)
        self.ui.operations_text.setPlainText(op_formula)

        # enable only the layer combos we need
        for idx, combo in enumerate(self.layer_combos):
            combo.setDisabled(idx >= num_layers)

    def _validate(self, *args, **kwargs):
        valid_name = bool(self.ui.layer_name_edit.text())
        ok_button = self.ui.buttons.button(QtGui.QDialogButtonBox.Ok)
        if valid_name:
            ok_button.setDisabled(False)
        else:
            ok_button.setDisabled(True)

    def _init_combo(self, c, available_info, select_uuid):
        for idx, (short_name, uuid) in enumerate(available_info.items()):
            c.addItem(short_name, uuid)
            if select_uuid and short_name == select_uuid:
                c.setCurrentIndex(idx)

    def _create_algebraic(self):
        new_name = self.ui.layer_name_edit.text()
        namespace = {}
        for name, combo in self.layer_combos_names:
            namespace[name] = combo.itemData(combo.currentIndex())
        operations = self.ui.operations_text.toPlainText()
        info = {
            INFO.SHORT_NAME: new_name,
        }

        self.doc.create_algebraic_composite(operations=operations, namespace=namespace, info=info)

    def done(self, r):
        if r == QtGui.QDialog.Accepted:
            operation = self.ui.operation_combo.currentText()

            status_title = ''
            status_text = ''
            if operation in [DIFF_OP_NAME, NDI_OP_NAME]:
                namespace = {}
                for name, combo in self.layer_combos_names:
                    namespace[name] = combo.itemData(combo.currentIndex())
                x = self.doc[namespace['x']]
                y = self.doc[namespace['y']]
                if namespace['x'] == namespace['y']:
                    status_title = 'Error: Bad layer selection'
                    status_text = 'Error: Subtracting a layer from itself'
                elif x[INFO.STANDARD_NAME] != y[INFO.STANDARD_NAME]:
                    status_title = 'Warning: Bad layer selection'
                    status_text = 'Warning: Subtracting layers with different units'

            try:
                operations = self.ui.operations_text.toPlainText()
                ops_ast = ast.parse(operations, mode='exec')
                ops = compile(ops_ast, '<string>', 'exec')
                result_name = ops_ast.body[-1].targets[0].id
            except (AttributeError, KeyError, IndexError, SyntaxError):
                ret_button = QtGui.QMessageBox.critical(
                    self,
                    'Error: Syntax Error',
                    'Error: Syntax Error in Operations code. Must be valid Python 3 syntax.',
                    QtGui.QMessageBox.Ok,
                )
                return

            if status_title.startswith("Warning"):
                msg_box = QtGui.QMessageBox(
                    QtGui.QMessageBox.Warning,
                    status_title,
                    status_text,
                    QtGui.QMessageBox.No | QtGui.QMessageBox.Yes,
                    self,
                )
                msg_box.setInformativeText("Do you want to create the layer anyway?")
                msg_box.setDefaultButton(QtGui.QMessageBox.No)
                ret = msg_box.exec_()
                if ret == QtGui.QMessageBox.No:
                    return
            elif status_title.startswith('Error'):
                ret_button = QtGui.QMessageBox.critical(
                    self,
                    status_title,
                    status_text,
                    QtGui.QMessageBox.Ok,
                    )
                return

        return super(CreateAlgebraicDialog, self).done(r)
