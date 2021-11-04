#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test changes through the colormap dialog."""
from uwsift.common import Info, Kind, Presentation
from uwsift.model.layer import DocBasicLayer
from uwsift.view.colormap_dialogs import ChangeColormapDialog
from unittest import mock


def test_slider_change(qtbot):
    doc = mock.MagicMock()
    prez = Presentation(
        uuid="some_uuid",
        kind=Kind.IMAGE,
        visible=True,
        a_order=0,
        colormap="viridis",
        climits=(0.0, 50.0),
        gamma=1.0,
        mixing=None,
    )
    layer_info = {
        Info.VALID_RANGE: (0.0, 150.0),
        Info.KIND: Kind.IMAGE,
        Info.DISPLAY_FAMILY: "family1",
        Info.SHORT_NAME: "some_name",
        Info.UNIT_CONVERSION: (lambda x, inverse=False: x, lambda x, inverse=False: x),

    }
    layer = DocBasicLayer(doc, layer_info)
    doc.prez_for_uuid.return_value = prez
    doc.valid_range_for_uuid.return_value = layer_info[Info.VALID_RANGE]
    doc.__getitem__.return_value = layer
    widget = ChangeColormapDialog(doc, "some_uuid")

    # Set the red min value manually
    widget.ui.vmin_slider.setValue(8)
    doc.change_clims_for_siblings.assert_called_once()
    doc.change_clims_for_siblings.assert_called_with("some_uuid", (12.0, 50.0))
