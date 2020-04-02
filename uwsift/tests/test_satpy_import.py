import os
import sys
import importlib
import importlib.util
import appdirs
import subprocess
import shutil
import pytest

from ast import literal_eval


def remove_cached_module(module_name: str):
    filtered_modules = [module for module in sys.modules.keys() if module.startswith(module_name)]
    for module in filtered_modules:
        del sys.modules[module]

    cached_paths = [file_path for file_path in sys.path_importer_cache.keys() if module_name in file_path]
    for cached_file_path in cached_paths:
        del sys.path_importer_cache[cached_file_path]

    importlib.invalidate_caches()


def get_python_path() -> str:
    process = subprocess.run([sys.executable, "--version"], stdout=subprocess.PIPE, check=True)
    if process.stdout.decode("utf-8").startswith("Python "):
        return sys.executable
    else:
        raise RuntimeError("Python has been embedded into another application")


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="can only set the config directory on Linux")
def test_config_satpy_import_path(tmp_path):
    # get the folder structure of the config directory
    uwsift_module = importlib.import_module("uwsift")
    relative_config_dir = os.path.relpath(uwsift_module.BASE_CONFIG_DIR, appdirs.user_config_dir())

    # recreate that folder structure inside the temp directory `config`
    tmp_config_dir = os.path.join(tmp_path, "config")
    sift_tmp_config_dir = os.path.join(tmp_config_dir, relative_config_dir)
    os.makedirs(sift_tmp_config_dir)

    # copy the Satpy module to the temp directory `satpy`
    tmp_satpy_dir = os.path.join(tmp_path, "satpy")
    satpy_module = importlib.import_module("satpy")
    shutil.copytree(os.path.dirname(satpy_module.__file__), tmp_satpy_dir)

    # create a config file, which points to the Satpy module in the temp `satpy` directory
    general_settings_path = os.path.join(sift_tmp_config_dir, "general_settings.yml")
    with open(general_settings_path, "w") as file:
        file.write(f'satpy_import_path: "{tmp_satpy_dir}"\n')

    # Don't import the `uwsift` using the current Python Interpreter, because the code in the
    # __init__.py is just executed once. Use a subprocess, such that the code can be be exected
    # multiple times with different config files and corrupted Satpy modules.
    modified_env = {"PATH": os.getenv("PATH"), "XDG_CONFIG_HOME": tmp_config_dir}
    command = [get_python_path(), "-c", "import uwsift\nprint({'base_config_dir': uwsift.BASE_CONFIG_DIR, "
                                     "'satpy_import_path': uwsift.config['satpy_import_path']})"]
    working_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
    process = subprocess.run(command, stdout=subprocess.PIPE, env=modified_env, cwd=working_dir, check=True)
    results = literal_eval(process.stdout.decode("utf-8"))

    # Can we configure the config file path with the environment variable `XDG_CONFIG_HOME`?
    assert results["base_config_dir"] == sift_tmp_config_dir

    # Is the Satpy module path correctly read from the config file `general_settings.yml`?
    assert results["satpy_import_path"] == tmp_satpy_dir


@pytest.mark.skip
def test_overwrite_satpy_import(tmp_path):
    import uwsift

    satpy_module = importlib.import_module("satpy")
    satpy_default_path = os.path.dirname(satpy_module.__file__)

    # set the import path to a directory, which doesn't exist
    with pytest.raises(FileNotFoundError):
        uwsift.overwrite_import("satpy", os.path.join(tmp_path, "does_not_exist"), verbose=False)

    # set the import path to an empty directory
    remove_cached_module("satpy")
    uwsift.overwrite_import("satpy", tmp_path, verbose=False)
    satpy_module = importlib.import_module("satpy")
    assert satpy_module.__file__ == os.path.join(tmp_path, "__init__.py")

    # copy the default Satpy package to the temp directory
    tmp_satpy_dir = os.path.join(tmp_path, "satpy")
    shutil.copytree(satpy_default_path, tmp_satpy_dir)
    remove_cached_module("satpy")
    uwsift.overwrite_import("satpy", tmp_satpy_dir, verbose=False)
    satpy_module = importlib.import_module("satpy")
    assert satpy_module.__file__ == os.path.join(tmp_satpy_dir, "__init__.py")

    # delete the `readers` subpackage from the copied Satpy installation
    shutil.rmtree(os.path.join(tmp_satpy_dir, "readers"))
    remove_cached_module("satpy")
    uwsift.overwrite_import("satpy", tmp_satpy_dir, verbose=False)
    with pytest.raises(ImportError):
        importlib.import_module("satpy.readers")
