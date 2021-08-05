#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test RGB configuration changes through the UI."""
from uwsift.common import Info, Kind
from uwsift.model.composite_recipes import CompositeRecipe
from uwsift.view.rgb_config import RGBLayerConfigPane
from uwsift.ui.pov_main_ui import Ui_MainWindow

from PyQt5.QtWidgets import QMainWindow


class _PaneWrapper(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.pane = RGBLayerConfigPane(self.ui, self.ui.layersPaneWidget)


def test_slider_change(qtbot):
    widget = _PaneWrapper()
    pane = widget.pane

    pane.family_added("family1", {
        Info.VALID_RANGE: (0.0, 150.0),
        Info.KIND: Kind.IMAGE,
        Info.DISPLAY_FAMILY: "family1",
        Info.UNIT_CONVERSION: (lambda x, inverse=False: x, lambda x, inverse=False: x),
    })
    pane.family_added("family2", {
        Info.VALID_RANGE: (0.0, 150.0),
        Info.KIND: Kind.IMAGE,
        Info.DISPLAY_FAMILY: "family2",
        Info.UNIT_CONVERSION: (lambda x, inverse=False: x, lambda x, inverse=False: x),
    })
    rgb_recipe = CompositeRecipe("my_rgb",
                                 input_ids=["family1", "family2", None],
                                 color_limits=((0.0, 90.0), (0.0, 90.0), (None, None)))
    pane.selection_did_change(rgb_recipe)

    # Set the red min value manually
    with qtbot.waitSignal(pane.didChangeRGBComponentLimits) as blocker:
        pane.sliders[0][0].setValue(8)
    recipe = blocker.args[0]
    clims_r = blocker.args[1][0]
    clims_g = blocker.args[1][1]
    clims_b = blocker.args[1][2]
    assert recipe is rgb_recipe
    assert clims_r == ((8.0 / 100.0) * 150, 90.0)
    assert clims_g == (0.0, 90.0)
    assert clims_b == (None, None)
