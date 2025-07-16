#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

PURPOSE
Manage the layer sets.

REFERENCES

REQUIRES

:author: Eva Schiffer <evas@ssec.wisc.edu>
:copyright: 2015 by University of Wisconsin Regents, see AUTHORS for more details
:license: GPLv3, see LICENSE for more details
"""
__author__ = "evas"
__docformat__ = "reStructuredText"

import logging
import math
from functools import partial
from typing import Optional, Tuple

from PyQt5 import QtWidgets

from uwsift.common import FALLBACK_RANGE, INVALID_COLOR_LIMITS, Info, Kind
from uwsift.model.area_definitions_manager import AreaDefinitionsManager
from uwsift.model.layer_item import LayerItem
from uwsift.model.layer_model import LayerModel
from uwsift.model.product_dataset import ProductDataset
from uwsift.ui.layer_details_widget_ui import Ui_LayerDetailsPane
from uwsift.util.common import (
    format_clims,
    format_resolution,
    get_initial_gamma,
    range_hull,
    range_hull_no_fail,
)
from uwsift.view.colormap import COLORMAP_MANAGER

LOG = logging.getLogger(__name__)


SLIDER_MIN_VAL = -2147483648
SLIDER_MAX_VAL = 2147483647


class SingleLayerInfoPane(QtWidgets.QWidget):
    """Shows details about one layer that is currently selected."""

    _slider_steps = 1000000
    _resampling_info = None

    # Flags that indicate whether to perform invert current time fit data or invert all times fit data
    _invert_current_time = False
    _invert_all_times = False

    def __init__(self, *args, **kwargs):
        """Initialise subwidgets and layout.

        Hide the subwidgets at the beginning because no layer is selected."""
        super(SingleLayerInfoPane, self).__init__(*args, **kwargs)

        self._valid_min, self._valid_max = FALLBACK_RANGE

        layout = QtWidgets.QVBoxLayout(self)
        self.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self._details_pane_ui = Ui_LayerDetailsPane()
        self._details_pane_widget = QtWidgets.QWidget(self)
        self._details_pane_ui.setupUi(self._details_pane_widget)

        layout.addWidget(self._details_pane_widget)
        self.setLayout(layout)

        self._clear_details_pane()

        self._current_selected_layer: Optional[LayerItem] = None

        self._init_cmap_combo()

        self._details_pane_ui.cmap_combobox.currentIndexChanged.connect(self._cmap_changed)
        self._details_pane_ui.vmin_slider.valueChanged.connect(partial(self._slider_changed, is_max=False))
        self._details_pane_ui.vmax_slider.valueChanged.connect(partial(self._slider_changed, is_max=True))

        # Detect if one of the buttons of the vmin/vmax spinbox has been clicked
        def on_spinbox_arrow_clicked():
            self._spinbox_button_was_clicked = True

        # If the text of one of the vmin/vmax spinbox has been changed this resets the
        # "spin button was clicked"-state.
        def on_spinbox_text_changed(_):
            self._spinbox_button_was_clicked = False

        self._details_pane_ui.vmin_spinbox.valueChanged.connect(partial(self._spin_box_changed, is_max=False))
        self._details_pane_ui.vmin_spinbox.upArrowClicked.connect(on_spinbox_arrow_clicked)
        self._details_pane_ui.vmin_spinbox.downArrowClicked.connect(on_spinbox_arrow_clicked)
        self._details_pane_ui.vmin_spinbox.lineEdit().textEdited.connect(on_spinbox_text_changed)
        self._details_pane_ui.vmax_spinbox.valueChanged.connect(partial(self._spin_box_changed, is_max=True))
        self._details_pane_ui.vmax_spinbox.upArrowClicked.connect(on_spinbox_arrow_clicked)
        self._details_pane_ui.vmax_spinbox.downArrowClicked.connect(on_spinbox_arrow_clicked)
        self._details_pane_ui.vmax_spinbox.lineEdit().textEdited.connect(on_spinbox_text_changed)
        self._spinbox_button_was_clicked = False

        self._details_pane_ui.gammaSpinBox.valueChanged.connect(self._gamma_changed)

        self._details_pane_ui.buttonGroup.buttonClicked.connect(self._on_button_clicked)

        self._details_pane_ui.climitsCurrentTime.clicked.connect(self._fit_clims_to_current_time)
        self._details_pane_ui.climitsAllTimes.clicked.connect(self._fit_clims_to_all_times)

        self._details_pane_ui.climitsAllTimesInvert.clicked.connect(self._invert_fit_clims_to_all_times)
        self._details_pane_ui.climitsCurrentTimeInvert.clicked.connect(self._invert_fit_clims_to_current_times)

        self._details_pane_ui.colormap_reset_button.clicked.connect(self._reset_to_initial_state)
        self._details_pane_ui.colormap_reassign_button.clicked.connect(self._reassign_color_limits)

    # Slot functions

    def initiate_update(self):
        """Start the update process if a layer is currently selected."""
        self._clear_details_pane()
        if not self._current_selected_layer:
            return

        clims = self._current_selected_layer.presentation.climits
        valid_range = self._current_selected_layer.valid_range
        if valid_range:
            self._valid_min, self._valid_max = range_hull(clims, valid_range)
        elif (
            clims and clims != INVALID_COLOR_LIMITS and isinstance(clims, tuple)
        ):  # Internal RGB Composites have a list of color limit (clims) tuples
            self._valid_min, self._valid_max = clims
        else:
            self._valid_min, self._valid_max = FALLBACK_RANGE

        self._update_displayed_info()

    def selection_did_change(self, layers: Tuple[LayerItem]):
        """Update the displayed values only when one layer is selected.

        Also reset the display to its initial state at the beginning.

        :param layers: Layers which are currently selected
        """
        if layers is not None and len(layers) == 1:
            self._current_selected_layer = layers[0]
            self.initiate_update()
        else:
            self._clear_details_pane()

    def update_displayed_clims(self):
        """Update the corresponding viewed values for the color limits of the layer

        Exclude the listed layer types for displaying color limit values even if they have some."""
        if self._current_selected_layer:
            if self._current_selected_layer.presentation.climits and self._current_selected_layer.info.get(
                Info.KIND
            ) not in [Kind.MC_IMAGE]:
                self._determine_display_value_for_clims()

    def update_displayed_colormap(self):
        """Update the currently viewed colormap values of the layer"""
        if self._current_selected_layer:
            if self._current_selected_layer.presentation.colormap:
                self._details_pane_ui.layerColormapValue.setText(self._current_selected_layer.presentation.colormap)

                idx = self._details_pane_ui.cmap_combobox.findData(self._current_selected_layer.presentation.colormap)
                self._details_pane_ui.cmap_combobox.setCurrentIndex(idx)

                cmap = COLORMAP_MANAGER.get(self._current_selected_layer.presentation.colormap)
                if cmap:
                    cmap_html = cmap._repr_html_()
                    cmap_html = cmap_html.replace("height", "border-collapse: collapse;\nheight")
                    self._details_pane_ui.layerColormapVisual.setHtml(
                        f"""<html><head></head><body style="margin: 0px"><div>{cmap_html}</div></body></html>"""
                    )

    # Utility functions

    def _clear_details_pane(self):
        self._details_pane_ui.layerNameValue.setText("<no single layer selected>")
        self._details_pane_ui.layerVisibleSchedTimeValue.setText("N/A")
        self._details_pane_ui.layerInstrumentValue.setText("N/A")
        self._details_pane_ui.layerWavelengthValue.setText("N/A")
        self._details_pane_ui.layerResolutionValue.setText("N/A")
        self._details_pane_ui.layerAreaResolutionValue.setText("N/A")
        self._details_pane_ui.layerColormapValue.setText("N/A")
        self._details_pane_ui.layerColorLimitsValue.setText("N/A")
        self._details_pane_ui.layerColormapVisual.setHtml("")
        self._details_pane_ui.kindDetailsStackedWidget.setCurrentWidget(self._details_pane_ui.page_others)

    def _cmap_changed(self, index):
        model = self._current_selected_layer.model
        cmap_str = self._details_pane_ui.cmap_combobox.itemData(index)
        self._current_cmap = str(cmap_str)
        model.change_colormap_for_layer(self._current_selected_layer.uuid, self._current_cmap)

    def _set_valid_min_max(self, channel_val):
        if self._valid_max >= self._valid_min:
            if channel_val < self._valid_min:
                self._valid_min = channel_val
            if channel_val > self._valid_max:
                self._valid_max = channel_val
        else:
            if channel_val < self._valid_max:
                self._valid_max = channel_val
            if channel_val > self._valid_min:
                self._valid_min = channel_val

    def _create_slider_value(self, channel_val):
        if self._valid_max == self._valid_min:
            return 0
        mini = min(self._valid_min, self._valid_max)
        maxi = max(self._valid_min, self._valid_max)
        return int((channel_val - mini) / (maxi - mini) * self._slider_steps)

    def _determine_display_value_for_clims(self):
        try:
            clims = self._current_selected_layer.presentation.climits
            if self._current_selected_layer.kind == Kind.RGB:
                model: LayerModel = self._current_selected_layer.model
                input_layers_info = model.get_input_layers_info(self._current_selected_layer)
                clims_str = self._get_multichannel_clims_str(clims, input_layers_info)
            else:
                unit_conv = self._current_selected_layer.info[Info.UNIT_CONVERSION]
                clims_str = format_clims(clims, unit_conv)

            self._details_pane_ui.layerColorLimitsValue.setText(clims_str)
        except TypeError:
            LOG.warning(
                f"Unable to set the value for color limits."
                f" Instead for {self._current_selected_layer.uuid} will the value 'N/A' be shown."
            )
        except KeyError:
            LOG.warning(
                f"Unable to convert color limits of layer {self._current_selected_layer}."
                f" Because there is no unit conversion in layer info: '{self._current_selected_layer.info}'."
                f" Instead for {self._current_selected_layer.uuid} will the value 'N/A' be shown."
            )

    def _enable_all_buttons(self):
        for btn in self._details_pane_ui.buttonGroup.buttons():
            btn.setEnabled(True)

    def _on_button_clicked(self, button):
        self._enable_all_buttons()
        # Disable the clicked button and mark it
        button.setEnabled(False)

    def _invert_fit_clims_to_all_times(self):
        self._invert_all_times = True
        self._fit_clims_to_all_times()
        self._invert_all_times = False

    def _invert_fit_clims_to_current_times(self):
        self._invert_current_time = True
        self._fit_clims_to_current_time()
        self._invert_current_time = False

    def _fit_clims_to_all_times(self):
        model = self._current_selected_layer.model
        actual_range = self._current_selected_layer.get_actual_range_from_layer()

        if self._invert_all_times:
            actual_range = (actual_range[1], actual_range[0])  # Swap min and max

        model.change_color_limits_for_layer(self._current_selected_layer.uuid, actual_range)
        valid_range = self._current_selected_layer.valid_range

        assert actual_range != (None, None)  # nosec B101
        self._valid_min, self._valid_max = range_hull_no_fail(actual_range, valid_range, actual_range)

        self._update_vmin(True)
        self._update_vmax(True)

    def _fit_clims_to_current_time(self):
        first_active_dataset = self._current_selected_layer.get_first_active_product_dataset()
        if first_active_dataset:
            model = self._current_selected_layer.model
            actual_range = self._current_selected_layer.get_actual_range_from_first_active_dataset()

            if self._invert_current_time:
                actual_range = (actual_range[1], actual_range[0])  # Swap min and max

            model.change_color_limits_for_layer(self._current_selected_layer.uuid, actual_range)

            valid_range = self._current_selected_layer.valid_range

            assert actual_range != (None, None)  # nosec B101
            self._valid_min, self._valid_max = range_hull_no_fail(actual_range, valid_range, actual_range)

            self._update_vmin(True)
            self._update_vmax(True)

    def _gamma_changed(self, val):
        model = self._current_selected_layer.model
        model.change_gamma_for_layer(self._current_selected_layer.uuid, val)

    def _get_multichannel_clims_str(self, clims, input_layers_info):
        assert len(input_layers_info) == len(clims)  # nosec B101
        multichannel_clims_strs = []
        for idx in range(len(clims)):
            curr_clims = clims[idx]
            curr_layer_info = input_layers_info[idx]
            if curr_layer_info:
                curr_unit_conv = curr_layer_info.get(Info.UNIT_CONVERSION)
                multichannel_clims_strs.append(format_clims(curr_clims, curr_unit_conv))
            else:
                multichannel_clims_strs.append("N/A")

        return ", ".join(multichannel_clims_strs)

    def _get_multichannel_instrument_str(self):
        used_instruments = set()
        instruments = []
        for instrument in self._current_selected_layer.info.get(Info.INSTRUMENT):
            if not instrument:
                instruments.append("N/A")
                continue

            instruments.append(instrument.value)
            used_instruments.add(instrument.value)
        if len(used_instruments) == 1:
            instrument_str = list(used_instruments)[0]
        else:
            instrument_str = ", ".join(instruments)
        return instrument_str

    @staticmethod
    def _get_multichannel_wavelength_str(wavelength):
        wavelength_tmp = []
        for wv in wavelength:
            if not wv:
                wavelength_tmp.append("N/A")
                continue

            wavelength_tmp.append(f"{wv.central} {wv.unit}")
        wavelength_str = ", ".join(wavelength_tmp)
        return wavelength_str

    def _get_slider_value(self, slider_val):
        if self._valid_max == self._valid_min:
            return self._valid_min
        mini = min(self._valid_min, self._valid_max)
        maxi = max(self._valid_min, self._valid_max)
        return (slider_val / self._slider_steps) * (maxi - mini) + mini

    def _init_cmap_combo(self):
        # FIXME: We should do this by colormap category
        for colormap in COLORMAP_MANAGER.keys():
            self._details_pane_ui.cmap_combobox.addItem(colormap, colormap)

    def _reset_to_initial_state(self):
        self._enable_all_buttons()

        model = self._current_selected_layer.model
        # rejecting (Cancel button) means reset previous settings
        model.change_colormap_for_layer(
            self._current_selected_layer.uuid, self._current_selected_layer.info[Info.COLORMAP]
        )

        clims = self._current_selected_layer.determine_initial_clims()
        model.change_color_limits_for_layer(self._current_selected_layer.uuid, clims)
        model.change_gamma_for_layer(
            self._current_selected_layer.uuid, get_initial_gamma(self._current_selected_layer.info)
        )
        actual_range = self._current_selected_layer.get_actual_range_from_first_active_dataset()
        valid_range = self._current_selected_layer.valid_range

        self._valid_min, self._valid_max = valid_range if valid_range else actual_range
        self._update_vmin()
        self._update_vmax()
        self._update_gamma()

    def _set_reassign_button_state(self, val1, val2):
        if math.isclose(val1, val2, rel_tol=1e-6):
            self._details_pane_ui.colormap_reassign_button.setEnabled(False)
        else:
            self._details_pane_ui.colormap_reassign_button.setEnabled(True)

    def _reassign_color_limits(self):
        # Get spin box values
        spin_box_max = self._details_pane_ui.vmax_spinbox
        spin_box_min = self._details_pane_ui.vmin_spinbox
        val_max = spin_box_max.value()
        val_min = spin_box_min.value()

        # Convert spin box values using the inverse unit conversion
        unit_converter = self._current_selected_layer.info[Info.UNIT_CONVERSION][1]
        val_max = self._create_slider_value(unit_converter(val_max, inverse=True))
        val_min = self._create_slider_value(unit_converter(val_min, inverse=True))

        # Determine min and max values for sliders
        real_max = max(val_min, val_max)
        real_min = min(val_min, val_max)

        # Access sliders
        slider_max = self._details_pane_ui.vmax_slider
        slider_min = self._details_pane_ui.vmin_slider

        # Block signals to prevent unwanted updates during setup
        for slider in (slider_max, slider_min):
            slider.blockSignals(True)
            slider.setMaximum(real_max)
            slider.setMinimum(real_min)

        # Set current slider values
        slider_max.setValue(val_max)
        slider_min.setValue(val_min)

        # Re-enable signals after setup
        for slider in (slider_max, slider_min):
            slider.blockSignals(False)

    def _set_new_clims(self, val, is_max):
        if is_max:
            new_clims = (self._current_selected_layer.presentation.climits[0], val)
        else:
            new_clims = (val, self._current_selected_layer.presentation.climits[1])
        model = self._current_selected_layer.model
        model.change_color_limits_for_layer(self._current_selected_layer.uuid, new_clims)

    def set_resampling_info(self, resampling_info):
        self._resampling_info = resampling_info

    def _slider_changed(self, value=None, is_max=True):
        self._enable_all_buttons()

        spin_box = self._details_pane_ui.vmax_spinbox if is_max else self._details_pane_ui.vmin_spinbox
        spin_box2 = self._details_pane_ui.vmin_spinbox if is_max else self._details_pane_ui.vmax_spinbox

        if value is None:
            slider = self._details_pane_ui.vmax_slider if is_max else self._details_pane_ui.vmin_slider
            value = slider.value()
        value = self._get_slider_value(value)
        LOG.debug("slider %s %s => %f" % (self._current_selected_layer.uuid, "max" if is_max else "min", value))
        display_val = self._current_selected_layer.info[Info.UNIT_CONVERSION][1](value)
        spin_box.blockSignals(True)
        spin_box.setValue(display_val, True)
        spin_box.blockSignals(False)

        self._set_reassign_button_state(spin_box.value(), spin_box2.value())
        return self._set_new_clims(value, is_max)

    def _spin_box_changed(self, is_max=True):
        self._enable_all_buttons()

        slider = self._details_pane_ui.vmax_slider if is_max else self._details_pane_ui.vmin_slider
        spin_box = self._details_pane_ui.vmax_spinbox if is_max else self._details_pane_ui.vmin_spinbox
        dis_val = spin_box.value()
        val = self._current_selected_layer.info[Info.UNIT_CONVERSION][1](dis_val, inverse=True)
        LOG.debug(
            "spin box %s %s => %f => %f" % (self._current_selected_layer.uuid, "max" if is_max else "min", dis_val, val)
        )
        # The val value probably affects _valid_min and _valid_max
        self._set_valid_min_max(val)
        sv = self._create_slider_value(val)

        # take the data for the second slider and spinbox
        slider2 = self._details_pane_ui.vmin_slider if is_max else self._details_pane_ui.vmax_slider
        spin_box2 = self._details_pane_ui.vmin_spinbox if is_max else self._details_pane_ui.vmax_spinbox
        dis_val2 = spin_box2.value()
        val2 = self._current_selected_layer.info[Info.UNIT_CONVERSION][1](dis_val2, inverse=True)
        sv2 = self._create_slider_value(val2)

        # block the slider signals
        slider.blockSignals(True)
        slider2.blockSignals(True)

        sv = max(SLIDER_MIN_VAL, min(SLIDER_MAX_VAL, sv))  # avoid OverflowError in slider.setvalue(...)
        sv2 = max(SLIDER_MIN_VAL, min(SLIDER_MAX_VAL, sv2))

        # set value to the slider
        slider.setValue(sv)
        # set the value of the second slider, as the mapping of the slider values may have changed
        slider2.setValue(sv2)

        # unblock the slider signals
        slider.blockSignals(False)
        slider2.blockSignals(False)

        # Workaround for the strange behavior in the spinbox, where the value
        # increases or decreases twice with a single arrow click when the maximum
        # or minimum of the range is changed
        if self._spinbox_button_was_clicked:
            spin_box.setEnabled(False)
            spin_box.setEnabled(True)
            self._details_pane_ui.gammaSpinBox.clearFocus()
            self._spinbox_button_was_clicked = False

        self._set_reassign_button_state(dis_val, spin_box2.value())
        return self._set_new_clims(val, is_max)

    def _update_displayed_info(self):
        self._update_displayed_layer_name()
        self._update_displayed_time()
        self._update_displayed_instrument()
        self._update_displayed_wavelength()
        self._update_displayed_resolution()
        self._update_displayed_area_resolution()
        self.update_displayed_clims()

        self._update_displayed_kind_details()

    def _update_displayed_instrument(self):
        if self._current_selected_layer.info.get(Info.INSTRUMENT):
            if self._current_selected_layer.info.get(Info.KIND) == Kind.RGB:
                instrument_str = self._get_multichannel_instrument_str()
            else:
                instrument_str = self._current_selected_layer.info[Info.INSTRUMENT].value

            self._details_pane_ui.layerInstrumentValue.setText(instrument_str)

    def _update_displayed_kind_details(self):
        if (
            self._current_selected_layer.kind == Kind.IMAGE
            and self._current_selected_layer.get_first_active_product_dataset()
        ):
            self._details_pane_ui.kindDetailsStackedWidget.setCurrentWidget(self._details_pane_ui.page_IMAGE)

            self.update_displayed_colormap()

            self._update_gamma()
            self._update_vmin()
            self._update_vmax()

    def _update_displayed_layer_name(self):
        layer_descriptor = self._current_selected_layer.descriptor
        if layer_descriptor:
            self._details_pane_ui.layerNameValue.setText(self._current_selected_layer.descriptor)

    def _update_displayed_resolution(self):
        resolution_str = "N/A"
        if self._current_selected_layer.info.get("resolution"):
            resolution_str = format_resolution(self._current_selected_layer.info["resolution"])
        elif self._current_selected_layer.info.get("resolution-x") or self._current_selected_layer.info.get(
            "resolution-y"
        ):
            resolution_x = self._current_selected_layer.info.get("resolution-x")
            resolution_y = self._current_selected_layer.info.get("resolution-y")

            resolution_str = format_resolution(resolution_x) if resolution_x else "N/A"
            resolution_str += " / "
            resolution_str += format_resolution(resolution_y) if resolution_y else "N/A"

        self._details_pane_ui.layerResolutionValue.setText(resolution_str)

    def _update_displayed_area_resolution(self):
        if self._resampling_info is None:
            self._details_pane_ui.layerAreaResolutionValue.setText("N/A")
        else:
            area_def = AreaDefinitionsManager.area_def_by_id(self._resampling_info["area_id"])
            if self._resampling_info["custom"]:
                resolution_values = AreaDefinitionsManager.area_def_custom_resolution_values(
                    area_def, self._resampling_info["shape"][0], self._resampling_info["shape"][1]
                )
            else:
                resolution_values = (area_def.pixel_size_x, area_def.pixel_size_y)

            resolution_x = resolution_values[0]
            resolution_y = resolution_values[1]

            self._details_pane_ui.layerAreaResolutionValue.setText(
                f"{format_resolution(resolution_x)} / {format_resolution(resolution_y)} "
            )

    def _update_displayed_time(self):
        active_product_dataset: Optional[ProductDataset] = (
            self._current_selected_layer.get_first_active_product_dataset()
        )
        if active_product_dataset:
            self._details_pane_ui.layerVisibleSchedTimeValue.setText(
                self._current_selected_layer.get_first_active_product_dataset().info.get(Info.DISPLAY_TIME)
            )

    def _update_displayed_wavelength(self):
        if self._current_selected_layer.info.get("wavelength"):
            wavelength = self._current_selected_layer.info.get("wavelength")
            if self._current_selected_layer.kind == Kind.RGB:
                wavelength_str = self._get_multichannel_wavelength_str(wavelength)
            else:
                wavelength_str = f"{wavelength.central} {wavelength.unit}"
            self._details_pane_ui.layerWavelengthValue.setText(wavelength_str)
        else:
            wavelength_str = "N/A" if self._current_selected_layer.kind != Kind.RGB else "N/A, N/A, N/A"
            self._details_pane_ui.layerWavelengthValue.setText(wavelength_str)

    def _update_gamma(self):
        self._details_pane_ui.gammaSpinBox.setValue(self._current_selected_layer.presentation.gamma)

    def _update_vmin(self, keep_displ_decs=False):
        current_vmin = self._current_selected_layer.presentation.climits[0]
        self._details_pane_ui.vmin_slider.setRange(0, self._slider_steps)
        slider_val = self._create_slider_value(current_vmin)
        self._details_pane_ui.vmin_slider.setSliderPosition(max(slider_val, 0))

        conv = self._current_selected_layer.info[Info.UNIT_CONVERSION]
        self._details_pane_ui.vmin_spinbox.setValue(
            conv[1](self._valid_min) if conv[1](self._valid_min) > conv[1](current_vmin) else conv[1](current_vmin),
            keep_displ_decs,
        )

    def _update_vmax(self, keep_displ_decs=False):
        current_vmax = self._current_selected_layer.presentation.climits[1]
        self._details_pane_ui.vmax_slider.setMaximum(32767)
        self._details_pane_ui.vmax_slider.setRange(0, self._slider_steps)
        slider_val = self._create_slider_value(current_vmax)
        self._details_pane_ui.vmax_slider.setSliderPosition(min(slider_val, 32767))

        conv = self._current_selected_layer.info[Info.UNIT_CONVERSION]
        self._details_pane_ui.vmax_spinbox.setValue(
            conv[1](self._valid_max) if conv[1](self._valid_max) < conv[1](current_vmax) else conv[1](current_vmax),
            keep_displ_decs,
        )
