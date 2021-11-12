import pytest
from uwsift.__main__ import Main, create_app
from uwsift.util.default_paths import USER_CONFIG_DIR
from PyQt5.QtTest import QTest


@pytest.fixture(scope="session")
def window(tmp_path_factory):
    """Provides the SIFT GUI to tests."""
    vispy_app, qt_app = create_app()  # noqa
    d = tmp_path_factory.mktemp("tmp")
    window = Main(config_dir=USER_CONFIG_DIR, workspace_dir=str(d))
    window.show()
    QTest.qWaitForWindowExposed(window)
    QTest.qWaitForWindowActive(window)
    window.raise_()
    window.activateWindow()
    return window
