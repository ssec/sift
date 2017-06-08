from PyQt4 import QtGui, QtCore
from sift.common import KIND, INFO
from sift.ui.create_algebraic_dialog_ui import Ui_create_algebraic_dialog


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
        # TODO: Turn off OK button until everything is satisfied
        available_info = {}
        selected_short_names = []
        # use DATASET_NAME as unique group identifier
        for idx, prez, layer in self.doc.current_layers_where(kinds=(KIND.IMAGE, KIND.COMPOSITE)):
            # use the UUID as a representative when talking to the document
            available_info.setdefault(layer[INFO.SHORT_NAME],
                                      layer[INFO.UUID])
            if layer[INFO.UUID] in selected_uuids:
                selected_short_names.append(layer[INFO.SHORT_NAME])
        a_select = selected_short_names[0] if len(selected_short_names) > 0 else None
        b_select = selected_short_names[1] if len(selected_short_names) > 1 else None
        self._init_combo(self.ui.a_combo, available_info, a_select)
        self._init_combo(self.ui.b_combo, available_info, b_select)

        self.ui.a_combo.currentIndexChanged.connect(self._validate)
        self.ui.b_combo.currentIndexChanged.connect(self._validate)
        self.ui.layer_name_edit.textEdited.connect(self._validate)
        self.accepted.connect(self._create_algebraic)

        self._validate()

    def _validate(self, *args, **kwargs):
        # FIXME: Once subtraction isn't the only thing, this won't work
        same_layers = self.ui.a_combo.currentIndex() == self.ui.b_combo.currentIndex()
        valid_name = bool(self.ui.layer_name_edit.text())
        ok_button = self.ui.buttons.button(QtGui.QDialogButtonBox.Ok)
        if not same_layers and valid_name:
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
        a_uuid = self.ui.a_combo.itemData(self.ui.a_combo.currentIndex())
        b_uuid = self.ui.b_combo.itemData(self.ui.b_combo.currentIndex())
        operations = self.ui.operations_text.toPlainText()
        namespace = {
            'A': a_uuid,
            'B': b_uuid,
        }
        info = {
            INFO.SHORT_NAME: new_name,
        }

        self.doc.create_algebraic_composite(operations=operations, namespace=namespace, info=info)
