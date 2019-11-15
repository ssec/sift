import pytest
from uwsift.__main__ import Main
from uwsift.util.default_paths import USER_CONFIG_DIR


@pytest.fixture(scope="session")
def window(tmp_path_factory):
    d = tmp_path_factory.mktemp("tmp")
    window = Main(config_dir=USER_CONFIG_DIR, workspace_dir=str(d))
    return window
