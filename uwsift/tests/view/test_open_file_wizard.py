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

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWizard
from uwsift.view.open_file_wizard import OpenFileWizard


def create_scene(dataset_ids):
    """Create a fake Scene object with specified datasets available."""
    class _SceneMock(object):
        """Fake Satpy Scene object."""

        def __init__(self, *args, **kwargs):
            """Allow any arguments or kwargs."""
            pass

        def available_dataset_ids(self):
            """Fake available dataset IDs."""
            return dataset_ids

    return _SceneMock


def _get_group_files_mock(groups):
    """Create a fake group_files function for testing."""
    def _group_files_mock(*args, **kwargs):
        """Fake Satpy group_files function."""
        return groups
    return _group_files_mock


def _create_get_open_file_names_mock(returned_files):
    """Create a mocked open file dialog box."""
    def _get_open_file_names(self_, msg, start_dir, filter_str):
        """Return filenames like a open file dialog."""
        return [returned_files]
    return _get_open_file_names


def test_wizard_abi_l1b(qtbot, monkeypatch):
    """Test that the open file wizard works all the way through."""
    from satpy.tests.utils import make_dataid
    files = ['OR_ABI-L1b-RadM1-M3C01_G16_s20182541300210_e20182541300267_c20182541300308.nc']
    dataset_ids = [
        # test that floating point resolutions don't crash
        make_dataid(name='C01', resolution=1000.5, calibration='reflectance'),
        # radiance calibrations should be ignored by default
        make_dataid(name='C01', resolution=1000.5, calibration='radiance'),
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
    # HACK: Bug in Windows, no enum's by name
    assert wiz.button(getattr(QWizard.WizardButton, 'NextButton', 1)).isEnabled()
    # Go to the next page
    wiz.next()  # noqa: B305

    ## Page 2
    # One product should be listed from our mocking above
    assert wiz.ui.selectIDTable.rowCount() == 1
    for row_idx in range(wiz.ui.selectIDTable.rowCount()):
        item = wiz.ui.selectIDTable.item(row_idx, 0)
        assert item.checkState() == Qt.Checked
    # A product is selected, that should be to good for the next page
    # HACK: Bug in Windows, no enum's by name
    assert wiz.button(getattr(QWizard.WizardButton, 'FinishButton', 3)).isEnabled()
    # Go to the next page
    wiz.next()  # noqa: B305

    # Verify the wizard is left in a usable state for the MainWindow
    assert len(wiz.scenes) == 1
    assert wiz.previous_reader == 'abi_l1b'
    sel_ids = wiz.collect_selected_ids()
    assert len(sel_ids) == 1
