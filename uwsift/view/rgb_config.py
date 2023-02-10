#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""UI objects for configuring RGB layers."""
import logging
import uuid
from functools import partial
from typing import Optional, Tuple

from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import QComboBox, QLineEdit

from uwsift.common import DEFAULT_GAMMA_VALUE, Info, Kind
from uwsift.model.composite_recipes import RGBA2IDX, CompositeRecipe
from uwsift.model.layer_item import LayerItem
from uwsift.model.layer_model import LayerModel

LOG = logging.getLogger(__name__)


class RGBLayerConfigPane(QObject):
    """Configures RGB channel selection and ranges on behalf of document.
    Document in turn generates update signals which cause the SceneGraph to refresh.
    """

    # Recipe, Channel (RGB), layer uuid, color limits, gamma
    didChangeRGBInputLayers = pyqtSignal(CompositeRecipe, str, object, tuple, float)
    # Recipe, Channel (RGB), color limits
    didChangeRGBColorLimits = pyqtSignal(CompositeRecipe, str, tuple)
    # Recipe, Channel (RGB), gamma
    didChangeRGBGamma = pyqtSignal(CompositeRecipe, str, float)

    didChangeRecipeName = pyqtSignal(CompositeRecipe, str)

    _rgb = None  # combo boxes in r,g,b order; cache
    _sliders = None  # sliders in r,g,b order; cache
    _edits = None
    _gamma_boxes = None  # tuple of each component's gamma spin boxes

    def __init__(self, ui, parent, model: LayerModel):
        super(RGBLayerConfigPane, self).__init__(parent)
        self.ui = ui
        self._valid_ranges = [(None, None), (None, None), (None, None)]  # tuples of each component's c-limits
        self._layer_uuids: list = []
        self.recipe: Optional[CompositeRecipe] = None
        self.model = model

        self._slider_steps = 100
        self.ui.slideMinRed.setRange(0, self._slider_steps)
        self.ui.slideMaxRed.setRange(0, self._slider_steps)
        self.ui.slideMinGreen.setRange(0, self._slider_steps)
        self.ui.slideMaxGreen.setRange(0, self._slider_steps)
        self.ui.slideMinBlue.setRange(0, self._slider_steps)
        self.ui.slideMaxBlue.setRange(0, self._slider_steps)

        self._double_validator = qdoba = QDoubleValidator()
        self.ui.editMinRed.setValidator(qdoba)
        self.ui.editMinRed.setText("0.0")
        self.ui.editMaxRed.setValidator(qdoba)
        self.ui.editMaxRed.setText("0.0")
        self.ui.editMinGreen.setValidator(qdoba)
        self.ui.editMinGreen.setText("0.0")
        self.ui.editMaxGreen.setValidator(qdoba)
        self.ui.editMaxGreen.setText("0.0")
        self.ui.editMinBlue.setValidator(qdoba)
        self.ui.editMinBlue.setText("0.0")
        self.ui.editMaxBlue.setValidator(qdoba)
        self.ui.editMaxBlue.setText("0.0")

        [
            x.currentIndexChanged.connect(partial(self._combo_changed, combo=x, color=rgb))
            for rgb, x in zip(("b", "g", "r"), (self.ui.comboBlue, self.ui.comboGreen, self.ui.comboRed))
        ]
        [
            x.valueChanged.connect(partial(self._slider_changed, slider=x, color=rgb, is_max=False))
            for rgb, x in zip(("b", "g", "r"), (self.ui.slideMinBlue, self.ui.slideMinGreen, self.ui.slideMinRed))
        ]
        [
            x.valueChanged.connect(partial(self._slider_changed, slider=x, color=rgb, is_max=True))
            for rgb, x in zip(("b", "g", "r"), (self.ui.slideMaxBlue, self.ui.slideMaxGreen, self.ui.slideMaxRed))
        ]
        [
            x.editingFinished.connect(partial(self._edit_changed, line_edit=x, color=rgb, is_max=False))
            for rgb, x in zip(("b", "g", "r"), (self.ui.editMinBlue, self.ui.editMinGreen, self.ui.editMinRed))
        ]
        [
            x.editingFinished.connect(partial(self._edit_changed, line_edit=x, color=rgb, is_max=True))
            for rgb, x in zip(("b", "g", "r"), (self.ui.editMaxBlue, self.ui.editMaxGreen, self.ui.editMaxRed))
        ]
        [
            x.valueChanged.connect(self._gamma_changed)
            for rgb, x in zip(
                ("b", "g", "r"), (self.ui.blueGammaSpinBox, self.ui.greenGammaSpinBox, self.ui.redGammaSpinBox)
            )
        ]

        self.ui.nameEdit.textEdited.connect(self._rgb_name_edit_changed)

        self._rgb_name_edit = self.ui.nameEdit

        # initialize the combo boxes
        self.set_combos_to_layer_names()
        # disable all UI elements to start
        self._show_settings_for_layer()

    @property
    def rgb(self):
        if self._rgb is None:
            self._rgb = [self.ui.comboRed, self.ui.comboGreen, self.ui.comboBlue]
            return self._rgb
        else:
            return self._rgb

    @property
    def sliders(self):
        if self._sliders is None:
            self._sliders = [
                (self.ui.slideMinRed, self.ui.slideMaxRed),
                (self.ui.slideMinGreen, self.ui.slideMaxGreen),
                (self.ui.slideMinBlue, self.ui.slideMaxBlue),
            ]
        return self._sliders

    @property
    def line_edits(self):
        if self._edits is None:
            self._edits = [
                (self.ui.editMinRed, self.ui.editMaxRed),
                (self.ui.editMinGreen, self.ui.editMaxGreen),
                (self.ui.editMinBlue, self.ui.editMaxBlue),
            ]
        return self._edits

    @property
    def rgb_name_edit(self):
        return self._rgb_name_edit

    @property
    def gamma_boxes(self):
        if self._gamma_boxes is None:
            self._gamma_boxes = (
                self.ui.redGammaSpinBox,
                self.ui.greenGammaSpinBox,
                self.ui.blueGammaSpinBox,
            )
        return self._gamma_boxes

    def layer_added(self, layer: LayerItem):
        if layer.kind in [Kind.IMAGE, Kind.COMPOSITE]:
            self._layer_uuids.append(layer.uuid)
            self.set_combos_to_layer_names()

    def layer_removed(self, layer_uuid):
        if layer_uuid in self._layer_uuids:
            idx = self._layer_uuids.index(layer_uuid)
            del self._layer_uuids[idx]
            self.set_combos_to_layer_names()
            self._show_settings_for_layer(self.recipe)

    def _gamma_changed(self, value):
        gamma = tuple(x.value() for x in self.gamma_boxes)
        recipe_gamma = self.recipe.gammas
        channels = ["r", "g", "b"]

        changed_channel = ""

        for idx in range(len(channels)):
            if gamma[idx] != recipe_gamma[idx]:
                changed_channel = channels[idx]

        self.didChangeRGBGamma.emit(self.recipe, changed_channel, value)

    def _combo_changed(self, index, combo: QComboBox, color):
        layer_uuid = combo.itemData(index)
        if not layer_uuid:
            # we use None as no-selection value, not empty string
            layer_uuid = None

        LOG.debug("RGB: user selected %s for %s" % (repr(layer_uuid), color))
        # reset slider position to min and max for layer
        self._set_minmax_slider(color, layer_uuid)

        layer = self.model.get_layer_by_uuid(layer_uuid)

        if not layer:
            clim = (None, None)
        else:
            valid_range = layer.valid_range
            # TODO: is the actual range really used correctly here?
            actual_range = layer.get_actual_range_from_layer()
            assert actual_range != (None, None)  # nosec B101
            clim = valid_range if valid_range else actual_range

        self.didChangeRGBInputLayers.emit(self.recipe, color, layer_uuid, clim, DEFAULT_GAMMA_VALUE)

        self._show_settings_for_layer(self.recipe)

    def _rgb_name_edit_changed(self, text):
        self.didChangeRecipeName.emit(self.recipe, text)

    def _display_to_data(self, color: str, values):
        """Convert display value to data value."""
        if self.recipe is None:
            # No recipe has been set yet
            return values
        layer_uuid = self.recipe.input_layer_ids[RGBA2IDX[color]]
        if layer_uuid is None:
            return values
        layer = self.model.get_layer_by_uuid(layer_uuid)
        assert layer is not None  # nosec B101 # suppress mypy [union-attr]
        layer_info = layer.info

        return (
            layer_info[Info.UNIT_CONVERSION][1](values, inverse=True)
            if layer_info.get(Info.UNIT_CONVERSION)
            else values
        )

    def _data_to_display(self, color: str, values):
        "convert data value to display value"
        if self.recipe is None:
            # No recipe has been set yet
            return values
        layer_uuid = self.recipe.input_layer_ids[RGBA2IDX[color]]
        if layer_uuid is None:
            return values
        layer = self.model.get_layer_by_uuid(layer_uuid)
        assert layer is not None  # nosec B101 # suppress mypy [union-attr]
        layer_info = layer.info

        return layer_info[Info.UNIT_CONVERSION][1](values) if layer_info.get(Info.UNIT_CONVERSION) else values

    def _get_slider_value(self, valid_min, valid_max, slider_val):
        return (slider_val / self._slider_steps) * (valid_max - valid_min) + valid_min

    def _create_slider_value(self, valid_min, valid_max, channel_val):
        return int((channel_val - valid_min) / (valid_max - valid_min)) * self._slider_steps

    def _update_line_edits(self, color: str, n: Optional[float] = None, x: Optional[float] = None):
        """
        update edit controls to match non-None values provided
        if called with just color, returns current min and max
        implicitly convert data values to and from display values
        :param color: in 'rgba'
        :param n: minimum data value or None
        :param x: max data value or None
        :return: new min, new max
        """
        idx = RGBA2IDX[color]
        edn, edx = self.line_edits[idx]
        edn.blockSignals(True)
        edx.blockSignals(True)
        if n is not None:
            ndis = self._data_to_display(color, n)
            edn.setText("%f" % ndis)
        else:
            ndis = float(edn.text())
            n = self._display_to_data(color, ndis)
        if x is not None:
            xdis = self._data_to_display(color, x)
            edx.setText("%f" % xdis)
        else:
            xdis = float(edx.text())
            x = self._display_to_data(color, xdis)
        edn.blockSignals(False)
        edx.blockSignals(False)
        return n, x

    def _signal_color_changing_range(self, color: str, n: float, x: float):
        new_limits = (n, x)
        self.didChangeRGBColorLimits.emit(self.recipe, color, new_limits)

    def _slider_changed(self, value, slider, color: str, is_max: bool):
        """
        handle slider update event from user
        :param slider: control
        :param color: char in 'rgba'
        :param is_max: whether slider's value represents the max or the min
        :return:
        """
        idx = RGBA2IDX[color]
        valid_min, valid_max = self._valid_ranges[idx]
        if value is None:
            value = slider.value()
        value = self._get_slider_value(valid_min, valid_max, value)
        LOG.debug("slider %s %s => %f" % (color, "max" if is_max else "min", value))
        n, x = self._update_line_edits(color, value if not is_max else None, value if is_max else None)
        self._signal_color_changing_range(color, n, x)

    def _edit_changed(self, line_edit: QLineEdit, color: str, is_max: bool):
        """
        update relevant slider value, propagate to the document
        :param line_edit: field that got a new value
        :param color: in 'rgba'
        :param is_max: whether the min or max edit field was changed
        :return:
        """
        idx = RGBA2IDX[color]
        vn, vx = self._valid_ranges[idx]
        vdis = float(line_edit.text())
        val = self._display_to_data(color, vdis)
        LOG.debug("line edit %s %s => %f => %f" % (color, "max" if is_max else "min", vdis, val))
        sv = self._create_slider_value(vn, vx, val)
        slider = self.sliders[idx][1 if is_max else 0]
        slider.blockSignals(True)
        slider.setValue(sv)
        slider.blockSignals(False)
        self._signal_color_changing_range(color, *self._update_line_edits(color))

    def selection_did_change(self, layers: Tuple[LayerItem]):
        """Change UI elements to reflect the provided recipe."""
        if layers is not None and len(layers) == 1:
            layer = layers[0]
            self.recipe = layer.recipe if isinstance(layer.recipe, CompositeRecipe) else None
            self._show_settings_for_layer(self.recipe)

    def _show_settings_for_layer(self, recipe=None):
        if not isinstance(recipe, CompositeRecipe):
            self.ui.rgbScrollAreaWidgetContents.setDisabled(True)
            return
        else:
            self.ui.rgbScrollAreaWidgetContents.setDisabled(False)

        for widget in self.rgb:
            # block signals so an existing RGB layer doesn't get overwritten with new layer selections
            widget.blockSignals(True)

        # update the combo boxes
        self._select_components_for_recipe(recipe)
        self._set_minmax_sliders(recipe)
        self._set_gamma_boxes(recipe)
        self._set_rgb_name_edit(recipe)

        for widget in self.rgb:
            # block signals so an existing RGB layer doesn't get overwritten with new layer selections
            widget.blockSignals(False)

    def _set_rgb_name_edit(self, recipe):
        if recipe is not None:
            self.rgb_name_edit.setText(recipe.name)
        else:
            self.rgb_name_edit.setText("")

    def _set_minmax_slider(self, color: str, layer_uuid: uuid.UUID, clims: Optional[Tuple[float, float]] = None):
        idx = RGBA2IDX[color]
        slider = self.sliders[idx]
        editn, editx = self.line_edits[idx]
        if layer_uuid not in self._layer_uuids:
            LOG.debug("Could not find {} in layer_uuids {}".format(repr(layer_uuid), self._layer_uuids))
        # block signals so the changed sliders don't trigger updates
        slider[0].blockSignals(True)
        slider[1].blockSignals(True)
        if clims is None or clims == (None, None) or layer_uuid not in self._layer_uuids:
            self._valid_ranges[idx] = (None, None)
            slider[0].setSliderPosition(0)
            slider[1].setSliderPosition(0)
            editn.blockSignals(True)
            editx.blockSignals(True)
            editn.setText("0.0")
            editx.setText("0.0")
            editn.blockSignals(False)
            editx.blockSignals(False)
            slider[0].setDisabled(True)
            slider[1].setDisabled(True)
            editn.setDisabled(True)
            editx.setDisabled(True)
        else:
            slider[0].setDisabled(False)
            slider[1].setDisabled(False)
            editn.setDisabled(False)
            editx.setDisabled(False)

            layer = self.model.get_layer_by_uuid(layer_uuid)
            assert layer is not None  # nosec B101 # suppress mypy [union-attr]
            valid_range = layer.valid_range
            # TODO: is the actual range really used correctly here?
            actual_range = layer.get_actual_range_from_layer()
            assert actual_range != (None, None)  # nosec B101
            layer_range = valid_range if valid_range else actual_range
            self._valid_ranges[idx] = layer_range

            slider_val = self._create_slider_value(layer_range[0], layer_range[1], clims[0])
            slider[0].setSliderPosition(max(slider_val, 0))
            slider_val = self._create_slider_value(layer_range[0], layer_range[1], clims[1])
            slider[1].setSliderPosition(min(slider_val, self._slider_steps))

            self._update_line_edits(color, *clims)
        slider[0].blockSignals(False)
        slider[1].blockSignals(False)

    def _set_minmax_sliders(self, recipe):
        if recipe:
            for idx, (color, clim) in enumerate(zip("rgb", self.recipe.color_limits)):
                layer_uuid = self.recipe.input_layer_ids[idx]
                self._set_minmax_slider(color, layer_uuid, clim)
        else:
            self._set_minmax_slider("r", None)
            self._set_minmax_slider("g", None)
            self._set_minmax_slider("b", None)

    def _select_components_for_recipe(self, recipe=None):
        if recipe is not None:
            for layer_uuid, widget in zip(recipe.input_layer_ids, self.rgb):
                if not layer_uuid:
                    widget.setCurrentIndex(0)
                else:
                    # Qt can't handle item data being tuples
                    dex = widget.findData(layer_uuid)
                    if dex <= 0:
                        widget.setCurrentIndex(0)
                        LOG.error("Layer with uuid '%s' not available to" " be selected" % (layer_uuid,))
                    else:
                        widget.setCurrentIndex(dex)
        else:
            for widget in self.rgb:
                widget.setCurrentIndex(0)

    def set_combos_to_layer_names(self):
        """
        update combo boxes with the list of layer names and then select the right r,g,b,a layers if they're not None
        :return:
        """
        # Get the current selected families so we can reselect them when we
        # rebuild the lists.
        current_layers = [x.itemData(x.currentIndex()) for x in self.rgb]

        for widget in self.rgb:
            # block signals so an existing RGB layer doesn't get overwritten with new layer selections
            widget.blockSignals(True)

        # clear out the current lists
        for widget in self.rgb:
            widget.clear()
            widget.addItem("None", None)

        # fill up our lists of layers
        for widget, selected_layer_uuid in zip(self.rgb, current_layers):
            if not selected_layer_uuid or selected_layer_uuid not in self._layer_uuids:
                # if the selection is None or the current layer was removed
                # if the current layer was removed by the document then the
                # document should have updated the recipe
                widget.setCurrentIndex(0)

            for idx, layer_uuid in enumerate(self._layer_uuids):
                layer: LayerItem = self.model.get_layer_by_uuid(layer_uuid)
                display_name = layer.descriptor

                widget.addItem(display_name, layer_uuid)
                widget.findData(uuid)

                if layer_uuid == selected_layer_uuid:
                    widget.setCurrentIndex(idx + 1)

        for widget in self.rgb:
            # block signals so an existing RGB layer doesn't get overwritten with new layer selections
            widget.blockSignals(False)

    def _set_gamma_boxes(self, recipe=None):
        if recipe is not None:
            for idx, sbox in enumerate(self.gamma_boxes):
                sbox.setDisabled(recipe.input_layer_ids[idx] is None)
                sbox.setValue(recipe.gammas[idx])
        else:
            for sbox in self.gamma_boxes:
                sbox.setDisabled(True)
                sbox.setValue(1.0)
