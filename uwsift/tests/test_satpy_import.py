import importlib
import importlib.util
import os
import re
import shutil
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
def test_config_satpy_import_path(tmp_path):
    satpy_module = importlib.import_module("satpy")

    # get the folder structure of the config directory
    uwsift_module = importlib.import_module("uwsift")
    relative_config_dir = os.path.relpath(uwsift_module.BASE_CONFIG_DIR, appdirs.user_config_dir())

    # Don't import the `uwsift` module using the current Python Interpreter, because the code
    # in the __init__.py is just executed once. Use a subprocess, such that the code can be
    # executed multiple times with different config files and corrupted Satpy modules.
    def check_uwsift_paths(
        xdg_config_home: Union[str, None],
        base_config_dir: str,
        overwritten_satpy_import_path: Union[str, None],
        satpy_init_path: str,
    ):
        if xdg_config_home is None:
            modified_env = {}
        else:
            modified_env = {"XDG_CONFIG_HOME": xdg_config_home}

        # Use stderr, because other print statements will confuse the `literal_eval`.
        # If an import is overwritten, then `overwrite_import` will output to stdout.
        command = [
            get_python_path(),
            "-c",
            "import sys, uwsift, satpy, satpy.readers\n"
            "print({'base_config_dir': uwsift.BASE_CONFIG_DIR, "
            "'overwritten_satpy_import_path': uwsift.config.get('satpy_import_path', None), "
            "'satpy_init_path': satpy.__file__}, file=sys.stderr)",
        ]
        working_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
        process = subprocess.run(command, stderr=subprocess.PIPE, env=modified_env, cwd=working_dir, check=True)
        results = literal_eval(process.stderr.decode("utf-8"))

        # Can we configure the config file path with the environment variable `XDG_CONFIG_HOME`?
        assert results["base_config_dir"] == base_config_dir
        # Is the Satpy module path correctly read from the config file `general_settings.yml`?
        assert results["overwritten_satpy_import_path"] == overwritten_satpy_import_path
        # Where is the `__init__.py` from Satpy located?
        assert results["satpy_init_path"] == satpy_init_path

    # If the environment variable `XDG_CONFIG_HOME` isn't set, then we search for the config in the user
    # specific config directory. If the file exists, we search for the `satpy_import_path`. Otherwise the
    # `satpy_import_path` must be None, in which case we don't overwrite the default Satpy import path.
    user_general_settings_path = os.path.join(uwsift_module.BASE_CONFIG_DIR, "general_settings.yml")
    if os.path.exists(user_general_settings_path):
        if os.path.isfile(user_general_settings_path):
            with open(user_general_settings_path) as file:
                import_path_regex = re.compile('^satpy_import_path: "(.*)"$')
                for line in file.readlines():
                    match = import_path_regex.match(line)
                    if match:
                        user_satpy_import_path = match.group(1)
                        break
                else:
                    # the file is empty or `satpy_import_path` is commented out
                    user_satpy_import_path = None
        else:
            raise FileNotFoundError(f"general_settings.yml isn't a file: {user_general_settings_path}")
    else:
        user_satpy_import_path = None
    check_uwsift_paths(None, uwsift_module.BASE_CONFIG_DIR, user_satpy_import_path, satpy_module.__file__)

    # recreate that folder structure inside the temp directory `config`
    tmp_config_dir = os.path.join(tmp_path, "config")
    sift_tmp_config_dir = os.path.join(tmp_config_dir, relative_config_dir)
    os.makedirs(sift_tmp_config_dir)

    # `satpy_import_path` is None, because the general_settings.yml doesn't exist yet.
    # Therefore the Satpy import path won't be overwritten and the default path is used.
    check_uwsift_paths(tmp_config_dir, sift_tmp_config_dir, None, satpy_module.__file__)

    def set_satpy_import_path(import_path: str):
        """create a config file, which points to the Satpy module in the temp `satpy` directory"""
        general_settings_path = os.path.join(sift_tmp_config_dir, "general_settings.yml")
        with open(general_settings_path, "w") as file:
            file.write(f'satpy_import_path: "{tmp_satpy_dir}"\n')

    tmp_satpy_dir = os.path.join(tmp_path, "satpy")
    set_satpy_import_path(tmp_satpy_dir)
    with pytest.raises((FileNotFoundError, subprocess.CalledProcessError)):
        # The temp `Satpy` directory is still empty, thus a FileNotFoundError will be raised.
        # Note: Only the first parameter is used, because the exception will be raised before the asserts.
        check_uwsift_paths(tmp_config_dir, sift_tmp_config_dir, None, tmp_satpy_dir)

    # copy the Satpy module to the temp directory `satpy`
    shutil.copytree(os.path.dirname(satpy_module.__file__), tmp_satpy_dir)
    tmp_satpy_init = os.path.join(tmp_satpy_dir, "__init__.py")
    check_uwsift_paths(tmp_config_dir, sift_tmp_config_dir, tmp_satpy_dir, tmp_satpy_init)

    # delete the `readers` subpackage from the copied Satpy installation
    readers_submodule = os.path.join(tmp_satpy_dir, "readers")
    shutil.rmtree(readers_submodule)
    with pytest.raises((ImportError, subprocess.CalledProcessError)):
        check_uwsift_paths(tmp_config_dir, sift_tmp_config_dir, tmp_satpy_dir, tmp_satpy_init)
