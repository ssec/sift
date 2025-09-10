from unittest.mock import MagicMock, patch

import pytest
from PyQt5 import QtGui

from uwsift import queue
from uwsift.__main__ import Main
from uwsift.util.default_paths import USER_CONFIG_DIR, WORKSPACE_TEMP_DIR


@pytest.fixture
def patched_main_window(qtbot):
    mocked_system_font = QtGui.QFont()
    mocked_system_font.setPointSize(23)
    menubar_mock = MagicMock()
    menubar_mock.font.return_value = mocked_system_font

    with patch("uwsift.__main__.Main.menuBar", return_value=menubar_mock):
        queue.TheQueue = None  # Need to reset queue.TheQueue
        main_window = Main(USER_CONFIG_DIR, WORKSPACE_TEMP_DIR)
        qtbot.addWidget(main_window)
        yield mocked_system_font, main_window


def test_apply_system_font(patched_main_window):
    """Assert applying the system font works as intended."""
    mocked_system_font, main_window = patched_main_window

    assert main_window.ui.cursorProbeText.font() == mocked_system_font
    assert main_window.ui.projectionLabel.font() == mocked_system_font
    assert main_window.ui.timeLastDatasetCreation.font() == mocked_system_font
    assert main_window.ui.timeLastDatasetCreationLineEdit.font() != mocked_system_font  # font has been customised
    assert main_window.ui.timeLastDatasetImportLineEdit.font() != mocked_system_font  # font has been customised
    assert main_window.ui.currentTime.font() == mocked_system_font
    assert main_window.ui.currentTimeLineEdit.font() != mocked_system_font  # font has been customised
    assert main_window.ui.progressText.font() == mocked_system_font
    assert main_window.ui.redGammaLabel.font() == mocked_system_font
    assert main_window.ui.greenGammaLabel.font() == mocked_system_font
    assert main_window.ui.blueGammaLabel.font() == mocked_system_font
    assert main_window.ui.name_label.font() == mocked_system_font
    assert main_window.ui.operation_label.font() == mocked_system_font
    assert main_window.ui.x_label.font() == mocked_system_font
    assert main_window.ui.y_label.font() == mocked_system_font
    assert main_window.ui.z_label.font() == mocked_system_font
    assert main_window.ui.operations_label.font() != mocked_system_font  # font has been customised
