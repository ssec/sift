#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of SIFT.
#
# SIFT is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# SIFT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with SIFT.  If not, see <http://www.gnu.org/licenses/>.
"""Test various parts of the Open File Wizard dialog."""

from PyQt5.QtCore import Qt, QTimer
from PyQt5 import QtWidgets
from uwsift.view.open_file_wizard import OpenFileWizard
from functools import partial


def get_wizard_object(window):
    """Wait until the OpenFileWizard dialog is available and return a reference."""
    wiz = window.findChildren(OpenFileWizard)
    while len(wiz) < 1:
        # force Qt event loop to process events until our dialog is shown
        QtWidgets.QApplication.processEvents()
    assert len(wiz) == 1
    return wiz[0]


def need_wizard_test(func, *args):
    """Decorator used to simplify tests between the main window and wizard."""
    def new_func(qtbot, window):
        """Wrap decorated function to simplify getting the wizard object."""
        def check_dialog():
            """Keep checking that the wizard is closed until it is closed."""
            wiz = window.findChildren(OpenFileWizard)
            # make sure it is closed and deleted
            assert len(wiz) == 0

        qtbot.addWidget(window)
        QTimer.singleShot(0, partial(func, qtbot, window))
        qtbot.keyClick(window, Qt.Key_O, Qt.ControlModifier)
        qtbot.waitUntil(check_dialog)
    return new_func


def create_scene(dataset_ids):
    class _SceneMock(object):
        """Fake Satpy Scene object."""

        def __init__(self, *args, **kwargs):
            """Allow any arguments or kwargs."""
            pass

        def available_dataset_ids(self):
            return dataset_ids

    return _SceneMock


def _get_group_files_mock(groups):
    """Create a fake group_files function for testing."""
    def _group_files_mock(*args, **kwargs):
        """Fake Satpy group_files function."""
        return groups
    return _group_files_mock


@need_wizard_test
def test_cmd_open_export_image_dialog(qtbot, window):
    """Run our actual in-dialog tests."""
    wiz = get_wizard_object(window)
    qtbot.addWidget(wiz)
    wiz.close()


def _create_get_open_file_names_mock(returned_files):
    def _get_open_file_names(self_, msg, start_dir, filter_str):
        return [returned_files]
    return _get_open_file_names


def test_wizard_abi_l1b(qtbot, monkeypatch):
    from satpy import DatasetID
    files = ['OR_ABI-L1b-RadM1-M3C01_G16_s20182541300210_e20182541300267_c20182541300308.nc']
    dataset_ids = [
        # test that floating point resolutions don't crash
        DatasetID(name='C01', resolution=1000.5, calibration='reflectance'),
        # radiance calibrations should be ignored by default
        DatasetID(name='C01', resolution=1000.5, calibration='radiance'),
    ]
    # Don't actually talk to Satpy
    monkeypatch.setattr('uwsift.view.open_file_wizard.Scene', create_scene(dataset_ids))
    monkeypatch.setattr('uwsift.view.open_file_wizard.group_files',
                        _get_group_files_mock([{'abi_l1b': files}]))
    # Add some ABI L1b files to the file list when add button is clicked
    monkeypatch.setattr('uwsift.view.open_file_wizard.QtWidgets.QFileDialog.getOpenFileNames',
                        _create_get_open_file_names_mock(
                            ['OR_ABI-L1b-RadM1-M3C01_G16_s20182541300210_e20182541300267_c20182541300308.nc']))

    # open the dialog and do some things
    wiz = OpenFileWizard()
    qtbot.addWidget(wiz)
    wiz.show()

    ## Page 1
    # Set reader to ABI L1b
    wiz.ui.readerComboBox.setCurrentIndex(wiz.ui.readerComboBox.findData('abi_l1b'))
    # Add a single file (see mock above)
    qtbot.mouseClick(wiz.ui.addButton, Qt.LeftButton)
    # Adding this file for the 'abi_l1b' should make this page complete
    assert wiz.button(wiz.WizardButton.NextButton).isEnabled()
    # Go to the next page
    wiz.next()

    ## Page 2
    # One product should be listed from our mocking above
    assert wiz.ui.selectIDTable.rowCount() == 1
    for row_idx in range(wiz.ui.selectIDTable.rowCount()):
        item = wiz.ui.selectIDTable.item(row_idx, 0)
        assert item.checkState() == Qt.Checked
    # A product is selected, that should be to good for the next page
    assert wiz.button(wiz.WizardButton.FinishButton).isEnabled()
    # Go to the next page
    wiz.next()

    # Verify the wizard is left in a usable state for the MainWindow
    assert len(wiz.scenes) == 1
    sel_ids = wiz.collect_selected_ids()
    assert len(sel_ids) == 1
