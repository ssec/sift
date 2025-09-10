from unittest.mock import MagicMock, patch

import pytest
from PyQt5 import QtGui

from uwsift.__main__ import Main
from uwsift.util.default_paths import USER_CONFIG_DIR, WORKSPACE_TEMP_DIR


@pytest.fixture
def patched_window(qtbot):
    global mocked_system_font
    mocked_system_font = QtGui.QFont()
    mocked_system_font.setPointSize(23)
    menubar_mock = MagicMock()
    menubar_mock.font.return_value = mocked_system_font

    with patch("uwsift.__main__.Main.menuBar", return_value=menubar_mock):
        main_window = Main(USER_CONFIG_DIR, WORKSPACE_TEMP_DIR)
        qtbot.addWidget(main_window)
        yield main_window


def test_apply_system_font(patched_window):
    """Assert applying the system font works as intended."""
    global mocked_system_font

    assert patched_window.ui.cursorProbeText.font() == mocked_system_font
    assert patched_window.ui.projectionLabel.font() == mocked_system_font
    assert patched_window.ui.timeLastDatasetCreation.font() == mocked_system_font
    assert patched_window.ui.timeLastDatasetCreationLineEdit.font() != mocked_system_font  # font has been customised
    assert patched_window.ui.timeLastDatasetImportLineEdit.font() != mocked_system_font  # font has been customised
    assert patched_window.ui.currentTime.font() == mocked_system_font
    assert patched_window.ui.currentTimeLineEdit.font() != mocked_system_font  # font has been customised
    assert patched_window.ui.progressText.font() == mocked_system_font
    assert patched_window.ui.redGammaLabel.font() == mocked_system_font
    assert patched_window.ui.greenGammaLabel.font() == mocked_system_font
    assert patched_window.ui.blueGammaLabel.font() == mocked_system_font
    assert patched_window.ui.name_label.font() == mocked_system_font
    assert patched_window.ui.operation_label.font() == mocked_system_font
    assert patched_window.ui.x_label.font() == mocked_system_font
    assert patched_window.ui.y_label.font() == mocked_system_font
    assert patched_window.ui.z_label.font() == mocked_system_font
    assert patched_window.ui.operations_label.font() != mocked_system_font  # font has been customised
