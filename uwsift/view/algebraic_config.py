import logging
from functools import partial
from typing import Optional, Tuple

from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtWidgets import QComboBox

from uwsift.common import Kind
from uwsift.model.composite_recipes import (
    CUSTOM_OP_NAME,
    DIFF_OP_NAME,
    PRESET_OPERATIONS,
    AlgebraicRecipe,
)
from uwsift.model.layer_item import LayerItem
from uwsift.model.layer_model import LayerModel

LOG = logging.getLogger(__name__)


class AlgebraicLayerConfigPane(QObject):
    # Recipe, Channel, layer uuid
    didChangeAlgebraicInputLayers = pyqtSignal(AlgebraicRecipe, str, object)

    # Recipe, operation kind
    didChangeAlgebraicOperationKind = pyqtSignal(AlgebraicRecipe, str)
    # Recipe, operation kind
    didChangeAlgebraicOperationFormula = pyqtSignal(AlgebraicRecipe, str)

    didChangeRecipeName = pyqtSignal(AlgebraicRecipe, str)

    didTriggeredUpdate = pyqtSignal(AlgebraicRecipe)

    _operation_combo = None
    _operations_text = None
    _algebraic_name_edit = None
    _algebraic_channels = None
    _custom_formula = None

    def __init__(self, ui, parent, model: LayerModel):
        super(AlgebraicLayerConfigPane, self).__init__(parent)
        self.ui = ui
        self._layer_uuids: list = []
        self.recipe: Optional[AlgebraicRecipe] = None
        self.model = model

        [
            x.currentIndexChanged.connect(partial(self._combo_changed, combo=x, channel=xyz))
            for xyz, x in zip(("z", "y", "x"), (self.ui.z_combo, self.ui.y_combo, self.ui.x_combo))
        ]
        self.ui.layer_name_edit.textEdited.connect(self._algebraic_name_edit_changed)
        self.ui.operations_text.textChanged.connect(self._operation_formular_changed)
        self.ui.algebraicUpdateButton.clicked.connect(self._on_algebraic_update_button_clicked)

        self._algebraic_name_edit = self.ui.layer_name_edit
        self._operation_combo = self.ui.operation_combo
        self._operations_text = self.ui.operations_text

        self._init_operation_selection()
        self._set_combos_to_layer_names()
        self._show_settings_for_layer()

    @property
    def algebraic_channels(self):
        if self._algebraic_channels is None:
            self._algebraic_channels = [self.ui.x_combo, self.ui.y_combo, self.ui.z_combo]
        return self._algebraic_channels

    @property
    def algebraic_name_edit(self):
        return self._algebraic_name_edit

    @property
    def operation_combo(self):
        return self._operation_combo

    @property
    def operations_text(self):
        return self._operations_text

    def layer_added(self, layer: LayerItem):
        if layer.kind in [Kind.IMAGE] and layer.recipe is None:
            self._layer_uuids.append(layer.uuid)
            self._set_combos_to_layer_names()

    def layer_removed(self, layer_uuid):
        if layer_uuid in self._layer_uuids:
            idx = self._layer_uuids.index(layer_uuid)
            del self._layer_uuids[idx]
            self._set_combos_to_layer_names()
            self._show_settings_for_layer(self.recipe)

    def _combo_changed(self, index, combo: QComboBox = None, channel=None):
        layer_uuid = combo.itemData(index)
        if not layer_uuid:
            layer_uuid = None

        LOG.debug(f"Algebraic: user selected {repr(layer_uuid)} for {channel}")

        self.didChangeAlgebraicInputLayers.emit(self.recipe, channel, layer_uuid)

        self._show_settings_for_layer(self.recipe)

    def _algebraic_name_edit_changed(self, text):
        self.didChangeRecipeName.emit(self.recipe, text)

    def selection_did_change(self, layers: Tuple[LayerItem]):
        """Change UI elements to reflect the provided recipe"""
        if layers is not None and len(layers) == 1:
            layer = layers[0]
            self.recipe = layer.recipe if isinstance(layer.recipe, AlgebraicRecipe) else None
            self._show_settings_for_layer(self.recipe)

    def _show_settings_for_layer(self, recipe=None):
        if not isinstance(recipe, AlgebraicRecipe):
            self.ui.algebraicScrollAreaWidgetContents.setDisabled(True)
            return
        else:
            self.ui.algebraicScrollAreaWidgetContents.setDisabled(False)

        for widget in self.algebraic_channels:
            widget.blockSignals(True)

        # Update widgets
        self._select_components_for_recipe(recipe)
        self._set_algebraic_name_edit(recipe)
        self._select_operation_of_recipe(recipe)

        for widget in self.algebraic_channels:
            widget.blockSignals(False)

    def _select_components_for_recipe(self, recipe=None):
        if recipe is not None:
            for layer_uuid, widget in zip(recipe.input_layer_ids, self.algebraic_channels):
                if not layer_uuid:
                    widget.setCurrentIndex(0)
                else:
                    dex = widget.findData(layer_uuid)
                    if dex <= 0:
                        widget.setCurrentIndex(0)
                        LOG.error(f"Layer with  uuid '{layer_uuid}' not" f" available to be selected")
                    else:
                        widget.setCurrentIndex(dex)
        else:
            for widget in self.algebraic_channels:
                widget.setCurrentIndex(0)

    def _set_algebraic_name_edit(self, recipe):
        if recipe is not None:
            self.algebraic_name_edit.setText(recipe.name)
        else:
            self.algebraic_name_edit.setText("")

    def _set_combos_to_layer_names(self):
        """
        update combo boxes with the list of layer names and then select
         the right x, y, z layers if they're not None
        """
        current_layers = [x.itemData(x.currentIndex()) for x in self.algebraic_channels]

        for widget in self.algebraic_channels:
            widget.blockSignals(True)

        for widget in self.algebraic_channels:
            widget.clear()
            widget.addItem("None", None)

        for widget, selected_layer_uuid in zip(self.algebraic_channels, current_layers):
            if not selected_layer_uuid or selected_layer_uuid not in self._layer_uuids:
                widget.setCurrentIndex(0)

            for idx, layer_uuid in enumerate(self._layer_uuids):
                layer: LayerItem = self.model.get_layer_by_uuid(layer_uuid)
                display_name = layer.descriptor

                widget.addItem(display_name, layer_uuid)
                widget.findData(layer_uuid)

                if layer_uuid == selected_layer_uuid:
                    widget.setCurrentIndex(idx + 1)

        for widget in self.algebraic_channels:
            widget.blockSignals(False)

    def _init_operation_selection(self):
        for op_name in PRESET_OPERATIONS.keys():
            self.operation_combo.addItem(op_name)
        self.operation_combo.addItem(CUSTOM_OP_NAME)
        self.operation_combo.setCurrentIndex(0)
        self.operation_combo.currentIndexChanged.connect(self._set_operation)
        self.ui.operation_combo.currentIndexChanged.connect(self._operation_kind_changed)
        self._custom_formula = PRESET_OPERATIONS[DIFF_OP_NAME][0]

    def _set_operation(self, *args, **kwargs):
        was_custom = self.operations_text.isEnabled()
        if was_custom:
            self._custom_formula = self.operations_text.toPlainText()

        operation = self.operation_combo.currentText()
        op_formular, num_layers = PRESET_OPERATIONS.get(operation, (None, 3))
        if op_formular is None:
            op_formular = self._custom_formula
            self.operations_text.setDisabled(False)
        else:
            self.operations_text.setDisabled(True)
        self.operations_text.setPlainText(op_formular)

        for idx, combo in enumerate(self.algebraic_channels):
            combo.setDisabled(idx >= num_layers)

    def _select_operation_of_recipe(self, recipe):
        if recipe and isinstance(recipe, AlgebraicRecipe):
            dex = self.operation_combo.findText(recipe.operation_kind)
            self.operation_combo.setCurrentIndex(dex)
        self._set_operation()

    def _operation_kind_changed(self, index: QComboBox):
        operation_kind = self.operation_combo.itemText(index)
        self.didChangeAlgebraicOperationKind.emit(self.recipe, operation_kind)

    def _operation_formular_changed(self):
        self.didChangeAlgebraicOperationFormula.emit(self.recipe, self.operations_text.toPlainText())

    def _on_algebraic_update_button_clicked(self):
        self.didTriggeredUpdate.emit(self.recipe)
