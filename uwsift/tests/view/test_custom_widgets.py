#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test custom widgets."""
import pytest
from PyQt5.QtCore import Qt
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QMainWindow

from uwsift.ui.custom_widgets import QAdaptiveDoubleSpinBox
from uwsift.ui.pov_main_ui import Ui_MainWindow


@pytest.mark.parametrize(
    "spinbox_obj",
    [
        "vmin_spinbox",
        "vmax_spinbox",
    ],
)
def test_adaptive_spin_box(qtbot, spinbox_obj):

    main_window = QMainWindow()
    ui_main_window = Ui_MainWindow()
    ui_main_window.setupUi(main_window)

    sd_spinbox: QAdaptiveDoubleSpinBox = getattr(ui_main_window.layerDetailsPane._details_pane_ui, spinbox_obj)
    sd_spinbox.blockSignals(True)

    line_edit = sd_spinbox.lineEdit()

    assert sd_spinbox._decimal_places == 2  # should be 2 by default
    sd_spinbox.setValue(1.019999)  # test for correct rounding
    assert sd_spinbox.value() == 1.02  # the value itself
    assert line_edit.text() == "1.02"  # as well as the textual representation

    # init the obj for subsequent tests
    sd_spinbox._decimal_places = 0
    sd_spinbox.setValue(1)
    line_edit.setFocus()

    sd_spinbox.stepBy(1)
    assert line_edit.text() == "2"  # one digit w/o decimals, so just an integral increment

    # add a digit
    line_edit.setCursorPosition(len(line_edit.text()))
    qtbot.keyClicks(line_edit, ".0")
    qtbot.keyClick(line_edit, Qt.Key_Enter)
    sd_spinbox.stepBy(1)
    assert line_edit.text() == "2.1"  # should increment only on the digit added

    sd_spinbox.stepBy(10)
    assert line_edit.text() == "3.1"  # do higher increments work?

    # add further digits
    line_edit.setCursorPosition(len(line_edit.text()))
    qtbot.keyClicks(line_edit, "00")
    qtbot.keyClick(line_edit, Qt.Key_Enter)
    sd_spinbox.stepBy(1)
    assert line_edit.text() == "3.101"  # works correct on last digit?
    QTest.keyPress(sd_spinbox, Qt.Key_Shift)  # use shift modifier
    sd_spinbox.stepBy(1)
    assert line_edit.text() == "3.201"  # increment 100 times higher?
    sd_spinbox.stepBy(10)
    assert line_edit.text() == "4.201"  # correct increment in combination with modifier and higher increment step?
    QTest.keyRelease(sd_spinbox, Qt.Key_Shift)
    sd_spinbox.blockSignals(False)
