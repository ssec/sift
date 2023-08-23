import importlib
import importlib.util
import os
import subprocess
import sys
from ast import literal_eval
from typing import Union

import appdirs
import pytest


def get_python_path() -> str:
    process = subprocess.run([sys.executable, "--version"], stdout=subprocess.PIPE, check=True)
    if process.stdout.decode("utf-8").startswith("Python "):
        return sys.executable
    else:
        raise RuntimeError("Python has been embedded into another application")


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="can only set the config directory on Linux")
def test_xdg_config_home(tmp_path):
    # get the folder structure of the config directory
    uwsift_module = importlib.import_module("uwsift")
    relative_config_dir = os.path.relpath(uwsift_module.USER_CONFIG_PATHS[0], appdirs.user_config_dir())

    # Don't import the `uwsift` module using the current Python Interpreter, because the code
    # in the __init__.py is just executed once. Use a subprocess, such that the code can be
    # executed multiple times with different config files and corrupted Satpy modules.
    def check_uwsift_paths(xdg_config_home: Union[str, None], base_config_dir: str) -> None:
        if xdg_config_home is None:
            modified_env = {}
        else:
            modified_env = {"XDG_CONFIG_HOME": xdg_config_home}

        # Use stderr, because other print statements will confuse the `literal_eval`.
        command = [
            get_python_path(),
            "-c",
            "import uwsift; import sys, satpy, satpy.readers\n"
            "print({'base_config_dir': uwsift.USER_CONFIG_PATHS[0]})",
        ]
        working_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
        process = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=modified_env, cwd=working_dir, check=True
        )
        results = literal_eval(process.stdout.decode("utf-8"))

        # Can we configure the config file path with the environment variable `XDG_CONFIG_HOME`?
        assert results["base_config_dir"] == base_config_dir

    check_uwsift_paths(None, uwsift_module.USER_CONFIG_PATHS[0])

    # test that XDG_CONFIG_HOME changes which configuration directory is used
    # recreate that folder structure inside the temp directory `config`
    tmp_config_dir = os.path.join(tmp_path, "config")
    sift_tmp_config_dir = os.path.join(tmp_config_dir, relative_config_dir)
    os.makedirs(sift_tmp_config_dir)
    check_uwsift_paths(tmp_config_dir, sift_tmp_config_dir)
