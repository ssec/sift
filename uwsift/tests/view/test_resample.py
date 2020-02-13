from PyQt5 import QtWidgets
from uwsift.view.open_file_wizard import OpenFileWizard


def _create_mock_doc(a=None):
    """Create a mocked Document object."""

    class _DocMock:
        def __init__(self):
            self.available_projections = {'Mercator': 'proj_str1', 'LCC Conus': 'proj_str2'}
            self.current_projection = 'Mercator'

        def current_projection_index(self):
            return 0

    class _ParentMock():
        def __init__(self):
            self.document = _DocMock()

        def parent(self):
            return self

    return _ParentMock()


def test_resample_set_projection(monkeypatch):
    """Test that selecting a projection should store information of that projection."""
    monkeypatch.setattr('uwsift.view.open_file_wizard.OpenFileWizard.parent', _create_mock_doc)
    res = OpenFileWizard().resample_dialog
    res.set_projection('Mercator')
    assert res.projection == ('Mercator', 'proj_str1')


def test_resample_set_resampler(monkeypatch):
    """Test that setting the resampler to certain options should disable/enable other options."""
    monkeypatch.setattr('uwsift.view.open_file_wizard.OpenFileWizard.parent', _create_mock_doc)
    res = OpenFileWizard().resample_dialog
    resampler = res.set_resampler('None')
    assert resampler is None
    assert not res.ui.resGroupBox.isEnabled()

    res.ui.resamplingMethodComboBox.setCurrentIndex(1)
    resampler = res.set_resampler('Nearest Neighbor')
    assert resampler == 'nearest'
    assert res.ui.resGroupBox.isEnabled()


def test_resample_update_info(monkeypatch):
    """Test changing the values in the resample dialog disables boxes when incomplete and
    updates resampling information."""
    monkeypatch.setattr('uwsift.view.open_file_wizard.OpenFileWizard.parent', _create_mock_doc)
    res = OpenFileWizard().resample_dialog

    assert res.resolution is None

    res.ui.resamplingMethodComboBox.setCurrentIndex(1)

    res.ui.resXLineEdit.setText('123')
    assert not res.ui.buttonBox.button(QtWidgets.QDialogButtonBox.Ok).isEnabled()

    res.ui.resYLineEdit.setText('123')
    assert res.ui.buttonBox.button(QtWidgets.QDialogButtonBox.Ok).isEnabled()

    assert res.resampler == 'nearest'
    assert res.resolution == (123, 123)
    assert res.projection == ('Mercator', 'proj_str1')

    assert res.parent().resampling_info == {'resampler': res.resampler, 'projection': res.projection,
                                            'resolution': res.resolution}
