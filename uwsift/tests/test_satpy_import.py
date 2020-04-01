import os
import sys
import importlib
import importlib.util
import appdirs
import subprocess
import runpy
import shutil
import pytest


def remove_cached_module(module_name: str):
    filtered_modules = [module for module in sys.modules.keys() if module.startswith(module_name)]
    for module in filtered_modules:
        del sys.modules[module]

    cached_paths = [file_path for file_path in sys.path_importer_cache.keys() if module_name in file_path]
    for cached_file_path in cached_paths:
        del sys.path_importer_cache[cached_file_path]

    importlib.invalidate_caches()


@pytest.mark.skip
@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="can only set the config directory on Linux")
def test_config_satpy_import_path(tmp_path):
    uwsift_module = importlib.import_module("uwsift")
    relative_config_dir = os.path.relpath(uwsift_module.BASE_CONFIG_DIR, appdirs.user_config_dir())

    tmp_config_dir = os.path.join(tmp_path, "config")
    sift_tmp_config_dir = os.path.join(tmp_config_dir, relative_config_dir)
    os.makedirs(sift_tmp_config_dir)

    tmp_satpy_dir = os.path.join(tmp_path, "satpy")
    satpy_module = importlib.import_module("satpy")
    shutil.copytree(os.path.dirname(satpy_module.__file__), tmp_satpy_dir)

    general_settings = os.path.join(sift_tmp_config_dir, "general_settings.yml")
    with open(general_settings, "w") as file:
        file.write(f'satpy_import_path: "{tmp_satpy_dir}"\n')

    os.environ["XDG_CONFIG_HOME"] = tmp_config_dir
    assert appdirs.user_config_dir() == tmp_config_dir

    # remove all `uwsift` modules from the cache in order to execute the __init__.py again
    remove_cached_module("uwsift")

    # 1: try to import the uwsift package (we must reload the module in order to execute __init__.py again)
    uwsift_module = importlib.import_module("uwsift")
    uwsift_util_module = importlib.import_module("uwsift.util")
    importlib.reload(uwsift_module)
    importlib.reload(uwsift_util_module)
    assert uwsift_module.config["satpy_import_path"] == tmp_satpy_dir

    # 2: try to execute the uwsift module with runpy
    uwsift_gloval_vars = runpy.run_module("uwsift")
    assert uwsift_gloval_vars["DOCUMENT_SETTINGS_DIR"] == sift_tmp_config_dir

    # 3: try to test the config with a subprocess
    modified_env = {"PATH": os.getenv("PATH"), "XDG_CONFIG_HOME": tmp_satpy_dir}
    command = ["python", "-c", "import uwsift\nprint(uwsift.config['satpy_import_path'])"]
    cwd = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
    process = subprocess.run(command, stdout=subprocess.PIPE, env=modified_env, cwd=cwd)
    assert process.stdout.decode("utf-8").rstrip("\n") == tmp_satpy_dir


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
