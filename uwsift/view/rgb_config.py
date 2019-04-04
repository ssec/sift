#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""UI objects for configuring RGB layers."""

import logging
from functools import partial
from typing import List, Tuple, Optional, Mapping

from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import QComboBox, QLineEdit

from uwsift.common import Info, Kind

LOG = logging.getLogger(__name__)
RGBA2IDX: Mapping[str, int] = dict(r=0, g=1, b=2, a=3)


class RGBLayerConfigPane(QObject):
    """Configures RGB channel selection and ranges on behalf of document.
    Document in turn generates update signals which cause the SceneGraph to refresh.
    """
    # recipe being changed, character from 'rgba', layer being assigned
    didChangeRGBComponentSelection = pyqtSignal(tuple, str, object)
    # recipe being changed, ((min, max), (min, max), (min, max))
    didChangeRGBComponentLimits = pyqtSignal(tuple, tuple)
    # recipe being changed, (new-gamma, new-gamma, new-gamma)
    didChangeRGBComponentGamma = pyqtSignal(tuple, tuple)

    _rgb = None  # combo boxes in r,g,b order; cache
    _sliders = None  # sliders in r,g,b order; cache
    _edits = None
    _valid_ranges: List[Tuple[float, float]] = None  # tuples of each component's c-limits
    _gamma_boxes = None  # tuple of each component's gamma spin boxes

    def __init__(self, ui, parent):
        super(RGBLayerConfigPane, self).__init__(parent)
        self.ui = ui
        self._valid_ranges = [None, None, None]
        self._selected_family = [None, None, None]
        self._families = {}
        self.recipe = None

        self._slider_steps = 100
        self.ui.slideMinRed.setRange(0, self._slider_steps)
        self.ui.slideMaxRed.setRange(0, self._slider_steps)
        self.ui.slideMinGreen.setRange(0, self._slider_steps)
        self.ui.slideMaxGreen.setRange(0, self._slider_steps)
        self.ui.slideMinBlue.setRange(0, self._slider_steps)
        self.ui.slideMaxBlue.setRange(0, self._slider_steps)

        self._double_validator = qdoba = QDoubleValidator()
        self.ui.editMinRed.setValidator(qdoba)
        self.ui.editMaxRed.setValidator(qdoba)
        self.ui.editMinGreen.setValidator(qdoba)
        self.ui.editMaxGreen.setValidator(qdoba)
        self.ui.editMinBlue.setValidator(qdoba)
        self.ui.editMaxBlue.setValidator(qdoba)

        [x.currentIndexChanged.connect(partial(self._combo_changed, combo=x, color=rgb))
         for rgb, x in zip(('b', 'g', 'r'), (self.ui.comboBlue, self.ui.comboGreen, self.ui.comboRed))]
        [x.sliderReleased.connect(partial(self._slider_changed, slider=x, color=rgb, is_max=False))
         for rgb, x in zip(('b', 'g', 'r'), (self.ui.slideMinBlue, self.ui.slideMinGreen, self.ui.slideMinRed))]
        [x.sliderReleased.connect(partial(self._slider_changed, slider=x, color=rgb, is_max=True))
         for rgb, x in zip(('b', 'g', 'r'), (self.ui.slideMaxBlue, self.ui.slideMaxGreen, self.ui.slideMaxRed))]
        [x.editingFinished.connect(partial(self._edit_changed, line_edit=x, color=rgb, is_max=False))
         for rgb, x in zip(('b', 'g', 'r'), (self.ui.editMinBlue, self.ui.editMinGreen, self.ui.editMinRed))]
        [x.editingFinished.connect(partial(self._edit_changed, line_edit=x, color=rgb, is_max=True))
         for rgb, x in zip(('b', 'g', 'r'), (self.ui.editMaxBlue, self.ui.editMaxGreen, self.ui.editMaxRed))]
        [x.valueChanged.connect(self._gamma_changed)
         for rgb, x in
         zip(('b', 'g', 'r'), (self.ui.redGammaSpinBox, self.ui.greenGammaSpinBox, self.ui.blueGammaSpinBox))]

        # initialize the combo boxes
        self._set_combos_to_family_names()
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
    def gamma_boxes(self):
        if self._gamma_boxes is None:
            self._gamma_boxes = (
                self.ui.redGammaSpinBox,
                self.ui.greenGammaSpinBox,
                self.ui.blueGammaSpinBox,
            )
        return self._gamma_boxes

    def family_added(self, family, family_info):
        if family_info[Info.KIND] in [Kind.RGB, Kind.CONTOUR]:
            # can't choose RGBs as components of RGBs
            return

        self._families[family] = family_info
        self._set_combos_to_family_names()

    def family_removed(self, family):
        if family in self._families:
            del self._families[family]
            self._set_combos_to_family_names()
            self._show_settings_for_layer(self.recipe)

    def _gamma_changed(self, value):
        gamma = tuple(x.value() for x in self.gamma_boxes)
        self.didChangeRGBComponentGamma.emit(self.recipe, gamma)

    def _combo_changed(self, index, combo: QComboBox = None, color=None):
        family = combo.itemData(index)
        if not family:
            # we use None as no-selection value, not empty string
            family = None

        LOG.debug("RGB: user selected %s for %s" % (repr(family), color))
        # reset slider position to min and max for layer
        self._set_minmax_slider(color, family)
        self.didChangeRGBComponentSelection.emit(self.recipe, color, family)

    def _display_to_data(self, color: str, values):
        """Convert display value to data value."""
        if self.recipe is None:
            # No recipe has been set yet
            return values
        family = self.recipe.input_ids[RGBA2IDX[color]]
        if family is None:
            return values
        family_info = self._families[family]
        return family_info[Info.UNIT_CONVERSION][1](values, inverse=True)

    def _data_to_display(self, color: str, values):
        "convert data value to display value"
        family = self.recipe.input_ids[RGBA2IDX[color]]
        if family is None:
            return values
        family_info = self._families[family]
        return family_info[Info.UNIT_CONVERSION][1](values)

    def _get_slider_value(self, valid_min, valid_max, slider_val):
        return (slider_val / self._slider_steps) * (valid_max - valid_min) + valid_min

    def _create_slider_value(self, valid_min, valid_max, channel_val):
        return ((channel_val - valid_min) / (valid_max - valid_min)) * self._slider_steps

    def _min_max_for_color(self, rgba: str):
        """
        return min value, max value as represented in sliders
        :param rgba: char in 'rgba'
        :return: (min-value, max-value) where min can be > max
        """
        idx = RGBA2IDX[rgba]
        slider = self.sliders[idx]
        valid_min, valid_max = self._valid_ranges[idx]
        n = self._get_slider_value(valid_min, valid_max, slider[0].value())
        x = self._get_slider_value(valid_min, valid_max, slider[1].value())
        return n, x

    def _update_line_edits(self, color: str, n: float = None, x: float = None):
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
        if n is not None:
            ndis = self._data_to_display(color, n)
            edn.setText('%f' % ndis)
        else:
            ndis = float(edn.text())
            n = self._display_to_data(color, ndis)
        if x is not None:
            xdis = self._data_to_display(color, x)
            edx.setText('%f' % xdis)
        else:
            xdis = float(edx.text())
            x = self._display_to_data(color, xdis)
        return n, x

    def _signal_color_changing_range(self, color: str, n: float, x: float):
        idx = RGBA2IDX[color]
        new_limits = list(self.recipe.color_limits)
        new_limits[idx] = (n, x)
        self.didChangeRGBComponentLimits.emit(self.recipe, tuple(new_limits))

    def _slider_changed(self, slider=None, color: str = None, is_max: bool = False):
        """
        handle slider update event from user
        :param slider: control
        :param color: char in 'rgba'
        :param is_max: whether slider's value represents the max or the min
        :return:
        """
        idx = RGBA2IDX[color]
        valid_min, valid_max = self._valid_ranges[idx]
        val = self._get_slider_value(valid_min, valid_max, slider.value())
        LOG.debug('slider %s %s => %f' % (color, 'max' if is_max else 'min', val))
        n, x = self._update_line_edits(color, val if not is_max else None, val if is_max else None)
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
        LOG.debug('line edit %s %s => %f => %f' % (color, 'max' if is_max else 'min', vdis, val))
        sv = self._create_slider_value(vn, vx, val)
        slider = self.sliders[idx][1 if is_max else 0]
        slider.setValue(sv)
        self._signal_color_changing_range(color, *self._update_line_edits(color))

    def selection_did_change(self, recipe):
        """Change UI elements to reflect the provided recipe."""
        self.recipe = recipe
        self._show_settings_for_layer(recipe)

    def _show_settings_for_layer(self, recipe=None):
        if recipe is None:
            for slider in self.sliders:
                slider[0].setDisabled(True)
                slider[1].setDisabled(True)
            for combo in self.rgb:
                combo.setDisabled(True)
            for edit in self.line_edits:
                edit[0].setDisabled(True)
                edit[1].setDisabled(True)
            for sbox in self.gamma_boxes:
                sbox.setDisabled(True)
            return
        else:
            # re-enable all the widgets
            for slider in self.sliders:
                slider[0].setDisabled(False)
                slider[1].setDisabled(False)
            for combo in self.rgb:
                combo.setDisabled(False)
            for edit in self.line_edits:
                edit[0].setDisabled(False)
                edit[1].setDisabled(False)
            for sbox in self.gamma_boxes:
                sbox.setDisabled(False)

        for widget in self.rgb:
            # block signals so an existing RGB layer doesn't get overwritten with new layer selections
            widget.blockSignals(True)

        # update the combo boxes
        self._select_components_for_recipe(recipe)
        self._set_minmax_sliders(recipe)
        self._set_gamma_boxes(recipe)

        for widget in self.rgb:
            # block signals so an existing RGB layer doesn't get overwritten with new layer selections
            widget.blockSignals(False)

    def _set_minmax_slider(self, color: str, family: str, clims: Optional[Tuple[float, float]] = None):
        idx = RGBA2IDX[color]
        slider = self.sliders[idx]
        editn, editx = self.line_edits[idx]
        if family not in self._families:
            LOG.debug(
                "Could not find {} in families {}".format(repr(family), repr(list(sorted(self._families.keys())))))
        if clims is None or clims == (None, None) or \
                family not in self._families:
            self._valid_ranges[idx] = None
            slider[0].setSliderPosition(0)
            slider[1].setSliderPosition(0)
            editn.setText('0.0')
            editx.setText('0.0')
            slider[0].setDisabled(True)
            slider[1].setDisabled(True)
            editn.setDisabled(True)
            editx.setDisabled(True)
        else:
            slider[0].setDisabled(False)
            slider[1].setDisabled(False)
            editn.setDisabled(False)
            editx.setDisabled(False)

            valid_range = self._families[family][Info.VALID_RANGE]
            self._valid_ranges[idx] = valid_range

            slider_val = self._create_slider_value(valid_range[0], valid_range[1], clims[0])
            slider[0].setSliderPosition(max(slider_val, 0))
            slider_val = self._create_slider_value(valid_range[0], valid_range[1], clims[1])
            slider[1].setSliderPosition(min(slider_val, self._slider_steps))
            self._update_line_edits(color, *clims)

    def _set_minmax_sliders(self, recipe):
        if recipe:
            for idx, (color, clim) in enumerate(zip("rgb", recipe.color_limits)):
                family = recipe.input_ids[idx]
                self._set_minmax_slider(color, family, clim)
        else:
            self._set_minmax_slider("r", None)
            self._set_minmax_slider("g", None)
            self._set_minmax_slider("b", None)

    def _select_components_for_recipe(self, recipe=None):
        if recipe is not None:
            for family_name, widget in zip(recipe.input_ids, self.rgb):
                if family_name is None:
                    widget.setCurrentIndex(0)
                else:
                    # Qt can't handle item data being tuples
                    dex = widget.findData(family_name)
                    if dex <= 0:
                        widget.setCurrentIndex(0)
                        LOG.error("Layer family '%s' not available to be selected" % (family_name,))
                    else:
                        widget.setCurrentIndex(dex)
        else:
            for widget in self.rgb:
                widget.setCurrentIndex(0)

    def _set_combos_to_family_names(self):
        """
        update combo boxes with the list of layer names and then select the right r,g,b,a layers if they're not None
        :return:
        """
        # Get the current selected families so we can reselect them when we
        # rebuild the lists.
        current_families = [x.itemData(x.currentIndex()) for x in self.rgb]

        for widget in self.rgb:
            # block signals so an existing RGB layer doesn't get overwritten with new layer selections
            widget.blockSignals(True)

        # clear out the current lists
        for widget in self.rgb:
            widget.clear()
            widget.addItem('None', '')

        # fill up our lists of layers
        for widget, selected_family in zip(self.rgb, current_families):
            if not selected_family or selected_family not in self._families:
                # if the selection is None or the current family was removed
                # if the current family was removed by the document then the
                # document should have updated the recipe
                widget.setCurrentIndex(0)
            for idx, (family_name, family_info) in enumerate(
                    sorted(self._families.items(), key=lambda x: x[1][Info.DISPLAY_FAMILY])):
                # Qt can't handle tuples as
                display_name = family_info[Info.DISPLAY_FAMILY]
                LOG.debug('adding to widget family {} as "{}"'.format(family_name, display_name))
                widget.addItem(display_name, family_name)
                widget.findData(family_name)  # sanity check
                if family_name == selected_family:
                    # None is 0 so add 1 to index
                    widget.setCurrentIndex(idx + 1)

        for widget in self.rgb:
            # block signals so an existing RGB layer doesn't get overwritten with new layer selections
            widget.blockSignals(False)

    def _set_gamma_boxes(self, recipe=None):
        if recipe is not None:
            for idx, sbox in enumerate(self.gamma_boxes):
                sbox.setDisabled(recipe.input_ids[idx] is None)
                sbox.setValue(recipe.gammas[idx])
        else:
            for idx, sbox in enumerate(self.gamma_boxes):
                sbox.setDisabled(True)
                sbox.setValue(1.)
