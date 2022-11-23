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
from typing import Optional, Tuple

import numpy as np
from PyQt5 import QtWidgets

from uwsift.common import INVALID_COLOR_LIMITS, Info, Kind
from uwsift.model.layer_item import LayerItem
from uwsift.model.layer_model import LayerModel
from uwsift.model.product_dataset import ProductDataset
from uwsift.ui.layer_details_widget_ui import Ui_LayerDetailsPane
from uwsift.view.colormap import COLORMAP_MANAGER

LOG = logging.getLogger(__name__)


class SingleLayerInfoPane(QtWidgets.QWidget):
    """Shows details about one layer that is currently selected."""

    def __init__(self, *args, **kwargs):
        """Initialise subwidgets and layout.

        Hide the subwidgets at the beginning because no layer is selected."""
        super(SingleLayerInfoPane, self).__init__(*args, **kwargs)

        layout = QtWidgets.QVBoxLayout(self)
        self.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self._details_pane_ui = Ui_LayerDetailsPane()
        self._details_pane_widget = QtWidgets.QWidget(self)
        self._details_pane_ui.setupUi(self._details_pane_widget)

        layout.addWidget(self._details_pane_widget)
        self.setLayout(layout)

        self._clear_details_pane()

        self._current_selected_layer: Optional[LayerItem] = None

    # Slot functions

    def initiate_update(self):
        """Start the update process if a layer is currently selected."""
        if self._current_selected_layer:
            self._update_displayed_info()

    def selection_did_change(self, layers: Tuple[LayerItem]):
        """Update the displayed values only when one layer is selected.

        If no or multiple layer are selected then reset the display to its initial state.

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
            else:
                self._details_pane_ui.layerColorLimitsValue.setText("N/A")

    def update_displayed_colormap(self):
        """Update the currently viewed colormap values of the layer"""
        if self._current_selected_layer:
            if self._current_selected_layer.presentation.colormap:
                self._details_pane_ui.layerColormapValue.setText(self._current_selected_layer.presentation.colormap)
                cmap = COLORMAP_MANAGER.get(self._current_selected_layer.presentation.colormap)
                if cmap:
                    cmap_html = cmap._repr_html_()
                    cmap_html = cmap_html.replace("height", "border-collapse: collapse;\nheight")
                    self._details_pane_ui.layerColormapVisual.setHtml(
                        f"""<html><head></head><body style="margin: 0px"><div>{cmap_html}</div></body></html>"""
                    )
                else:
                    self._details_pane_ui.layerColormapValue.setText("N/A")
                    self._details_pane_ui.layerColormapVisual.page().runJavaScript(
                        'document.body.style.overflow = "hidden";'
                    )

            else:
                self._details_pane_ui.layerColormapValue.setText("N/A")
                self._details_pane_ui.layerColormapVisual.setHtml("")

    # Utility functions

    def _clear_details_pane(self):
        self._details_pane_ui.layerNameValue.setText("<no single layer selected>")
        self._details_pane_ui.layerVisibleSchedTimeValue.setText("N/A")
        self._details_pane_ui.layerInstrumentValue.setText("N/A")
        self._details_pane_ui.layerWavelengthValue.setText("N/A")
        self._details_pane_ui.layerColormapValue.setText("N/A")
        self._details_pane_ui.layerColorLimitsValue.setText("N/A")
        self._details_pane_ui.layerColormapVisual.setHtml("")

    def _determine_clim_str_for_channel(self, curr_clim, curr_layer_info, multichannel_clim_strs):
        curr_unit_conv = curr_layer_info.get(Info.UNIT_CONVERSION) if curr_layer_info else None

        multichannel_clim_strs.append(self._get_single_clim_str(curr_clim, curr_unit_conv))

    def _determine_display_value_for_clims(self):
        try:
            clims = self._current_selected_layer.presentation.climits
            if self._current_selected_layer.kind == Kind.RGB:
                clim_str = self._get_multichannel_clim_str(clims)
            else:
                unit_conv = self._current_selected_layer.info[Info.UNIT_CONVERSION]
                clim_str = self._get_single_clim_str(clims, unit_conv)

            self._details_pane_ui.layerColorLimitsValue.setText(clim_str)
        except TypeError:
            LOG.warning(
                f"Unable to set the value for clims."
                f" Instead for {self._current_selected_layer.uuid} will the value 'N/A' be shown."
            )
            self._details_pane_ui.layerColorLimitsValue.setText("N/A")
        except KeyError:
            LOG.warning(
                f"Unable to convert clims of layer {self._current_selected_layer}."
                f" Because there is no unit conversion in layer info: '{self._current_selected_layer.info}'."
                f" Instead for {self._current_selected_layer.uuid} will the value 'N/A' be shown."
            )
            self._details_pane_ui.layerColorLimitsValue.setText("N/A")

    def _get_multichannel_clim_str(self, clims):

        model: LayerModel = self._current_selected_layer.model
        input_layers_info = model.get_input_layers_info(self._current_selected_layer)

        if len(input_layers_info) == len(clims):
            multichannel_clim_strs = []
            for idx in range(len(clims)):
                curr_clim = clims[idx]
                curr_layer_info = input_layers_info[idx]
                self._determine_clim_str_for_channel(curr_clim, curr_layer_info, multichannel_clim_strs)

            if len(multichannel_clim_strs) != len(input_layers_info) and len(multichannel_clim_strs) != len(clims):
                return None

            return ", ".join(multichannel_clim_strs)

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

    @staticmethod
    def _get_single_clim_str(clim, unit_conv):
        if clim == INVALID_COLOR_LIMITS:
            return "N/A"

        display_clim = unit_conv[1](np.array(clim))
        min_clim = unit_conv[2](display_clim[0], include_units=False)
        max_clim = unit_conv[2](display_clim[1])
        clim_str = f"{min_clim} ~ {max_clim}"
        return clim_str

    def _update_displayed_info(self):
        self._update_displayed_layer_name()

        self._update_displayed_time()

        self._update_displayed_instrument()

        self._update_displayed_wavelength()

        self.update_displayed_colormap()

        self.update_displayed_clims()

    def _update_displayed_instrument(self):
        if self._current_selected_layer.info.get(Info.INSTRUMENT):

            if self._current_selected_layer.info.get(Info.KIND) == Kind.RGB:
                instrument_str = self._get_multichannel_instrument_str()
            else:
                instrument_str = self._current_selected_layer.info[Info.INSTRUMENT].value

            self._details_pane_ui.layerInstrumentValue.setText(instrument_str)
        else:
            self._details_pane_ui.layerInstrumentValue.setText("N/A")

    def _update_displayed_layer_name(self):
        layer_descriptor = self._current_selected_layer.descriptor
        if layer_descriptor:
            self._details_pane_ui.layerNameValue.setText(self._current_selected_layer.descriptor)
        else:
            self._details_pane_ui.layerNameValue.setText("<no layer selected>")

    def _update_displayed_time(self):
        active_product_dataset: Optional[
            ProductDataset
        ] = self._current_selected_layer.get_first_active_product_dataset()
        if active_product_dataset:
            self._details_pane_ui.layerVisibleSchedTimeValue.setText(
                self._current_selected_layer.get_first_active_product_dataset().info.get(Info.DISPLAY_TIME)
            )
        else:
            self._details_pane_ui.layerVisibleSchedTimeValue.setText("N/A")

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
