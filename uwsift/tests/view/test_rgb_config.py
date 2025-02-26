#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test RGB configuration changes through the UI."""
from unittest import mock

from PyQt5.QtWidgets import QMainWindow

from uwsift.common import Info, Kind
from uwsift.model.composite_recipes import CompositeRecipe
from uwsift.model.layer_model import LayerModel
from uwsift.ui.pov_main_ui import Ui_MainWindow
from uwsift.view.rgb_config import RGBLayerConfigPane


class _PaneWrapper(QMainWindow):
    def __init__(self, model):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.pane = RGBLayerConfigPane(self.ui, None, model)


def test_slider_change(qtbot):
    doc = mock.MagicMock()
    workspace = mock.MagicMock()
    doc._workspace = workspace

    layer_model = LayerModel(doc)
    widget = _PaneWrapper(layer_model)
    pane = widget.pane

    layer_one = mock.MagicMock()
    layer_two = mock.MagicMock()

    layer_one.uuid = "some_uuid"
    layer_one.kind = Kind.IMAGE
    layer_one.descriptor = "layer_one"
    layer_one.info = {
        Info.VALID_RANGE: (0.0, 150.0),
        Info.KIND: Kind.IMAGE,
        Info.DISPLAY_FAMILY: "family1",
        Info.UNIT_CONVERSION: (lambda x, inverse=False: x, lambda x, inverse=False: x),
    }
    layer_one.valid_range = (0.0, 150.0)

    layer_two.uuid = "some_other_uuid"
    layer_two.kind = Kind.IMAGE
    layer_two.descriptor = "layer_two"
    layer_two.info = {
        Info.VALID_RANGE: (0.0, 150.0),
        Info.KIND: Kind.IMAGE,
        Info.DISPLAY_FAMILY: "family2",
        Info.UNIT_CONVERSION: (lambda x, inverse=False: x, lambda x, inverse=False: x),
    }
    layer_two.valid_range = (0.0, 150.0)

    layer_model.layers.append(layer_one)
    layer_model.layers.append(layer_two)

    pane.layer_added(layer_one)
    pane.layer_added(layer_two)

    rgb_recipe = CompositeRecipe.from_rgb(
        name="my_rgb", r="some_uuid", g="some_other_uuid", b=None, color_limits=((0.0, 90.0), (0.0, 90.0), (None, None))
    )
    rgb_layer = mock.MagicMock()
    rgb_layer.recipe = rgb_recipe

    pane.selection_did_change([rgb_layer])

    # Set the red min value manually
    with qtbot.waitSignal(pane.didChangeRGBColorLimits) as blocker:
        pane.sliders[0][0].setValue(8)
    recipe = blocker.args[0]
    color = blocker.args[1]
    clim = blocker.args[2]
    assert recipe is rgb_recipe
    assert color == "r"
    assert clim == ((8.0 / 100.0) * 150, 90.0)
