#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test changes through the colormap dialog."""
from unittest import mock

from uwsift.common import Info, Kind, Presentation
from uwsift.view.colormap_dialogs import ChangeColormapDialog


def test_slider_change(qtbot):
    layer_model = mock.MagicMock()
    layer = mock.MagicMock()
    presentation = Presentation(
        uuid="some_uuid",
        kind=Kind.IMAGE,
        visible=True,
        a_order=0,
        colormap="viridis",
        climits=(0.0, 50.0),
        gamma=1.0,
        opacity=1.0,
    )
    layer_info = {
        Info.VALID_RANGE: (0.0, 150.0),
        Info.KIND: Kind.IMAGE,
        Info.DISPLAY_FAMILY: "family1",
        Info.SHORT_NAME: "some_name",
        Info.UNIT_CONVERSION: (lambda x, inverse=False: x, lambda x, inverse=False: x),
    }
    layer.info = layer_info
    layer.presentation = presentation
    layer.valid_range = layer_info[Info.VALID_RANGE]
    layer_model.get_layer_by_uuid.return_value = layer

    widget = ChangeColormapDialog(layer_model, "some_uuid")

    # Set the red min value manually
    widget.ui.vmin_slider.setValue(8)
    layer_model.change_color_limits_for_layer.assert_called_once()
    layer_model.change_color_limits_for_layer.assert_called_with("some_uuid", (12.0, 50.0))
