#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a conda package and all-in-one installer for the current platform.

.. note::

    A current anaconda environment with all dependencies must be loaded
    before the below commands will complete. Usually `source activate sift`
    or `activate sift` on Windows.

Run the following commands to build an anaconda package and a all-in-one
installer application for the platform this script is executed on.
These commands assume the current machine has direct access to the SSEC
or other servers needed to complete the package uploads.

.. warning::

    This script does not do the final upload to the FTP server.

Mac OSX
-------

    python build_release.py

Windows 64-bit
--------------

    python build_release.py --no-conda-upload --no-installer-upload

The above assumes that an SCP and SSH client are not available on the Windows
build machine otherwise the uploads can be attempted. If they are not
available then a shared folder should be used to transport the conda package
and installer ".exe" to the appropriate servers.

Linux 64-bit
------------

    python build_release.py

Environment Variables
---------------------

SIFT_CHANNEL_HOST: (default 'larch') Server where the conda packages are
    served via HTTP.

SIFT_CHANNEL_PATH: (default '/var/apache/larch/htdocs/channels/sift')
    Path on SIFT_CHANNEL_HOST where conda packages are served. There should
    be `win-64`, `linux-64`, `osx-64`, and `noarch` directories in this
    directory.

SIFT_FTP_HOST: (default 'bumi')
    Server that has permission to upload to the SSEC FTP server.

SIFT_FTP_HOST_PATH: (default '~/repos/git/sift/dist') Where on the FTP host
    should all-in-one installers be placed before being uploaded to the FTP
    server.

SIFT_FTP_PATH: (default 'pub/sift/dist') Where on the SSEC FTP server should
    packages be uploaded.

"""

import logging
import os
import shutil
import subprocess  # nosec: B404
import sys
from glob import glob

from uwsift import version

if sys.version_info < (3, 5):
    run = subprocess.check_call
else:
    run = subprocess.run


log = logging.getLogger(__name__)

SIFT_CHANNEL = "http://larch.ssec.wisc.edu/channels/sift"
CONDA_RECIPE = os.path.join("conda-recipe", "uwsift")
CHANNEL_HOST = os.environ.get("SIFT_CHANNEL_HOST", "larch")
CHANNEL_PATH = os.environ.get("SIFT_CHANNEL_PATH", "/var/apache/larch/htdocs/channels/sift")
# server that is allowed to add to FTP site
FTP_HOST = os.environ.get("SIFT_FTP_HOST", "bumi")
FTP_HOST_PATH = os.environ.get("SIFT_FTP_HOST_PATH", "repos/git/uwsift/dist")
FTP_PATH = os.environ.get("SIFT_FTP_PATH", "pub/sift/dist")
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
DIST_DIR = os.path.join(SCRIPT_DIR, "dist")
ISCC_PATH = os.path.join("C:/", "Program Files (x86)", "Inno Setup 5", "ISCC.exe")


PLATFORMS = ["darwin", "linux", "win"]
CONDA_PLAT = {
    "darwin": "osx-64",
    "linux": "linux-64",
    "win": "win-64",
    # 'darwin': 'noarch',
    # 'linux': 'noarch',
    # 'win': 'noarch',
}


def get_platform():
    for k in PLATFORMS:
        if sys.platform.startswith(k):
            return k
    return "linux"


platform = get_platform()


def _build_conda(python_version, output_dir=DIST_DIR):
    try:
        os.makedirs(output_dir)
    except FileExistsError:
        pass

    log.info("Building conda package...")
    CONDA_BUILD_CMD = "conda build -c {} --python {} --output-folder {} {}".format(
        SIFT_CHANNEL, python_version, DIST_DIR, CONDA_RECIPE
    )
    run(CONDA_BUILD_CMD.split(" "))
    # check for build revisision
    for i in range(4, -1, -1):
        f = os.path.join(DIST_DIR, CONDA_PLAT[platform], "uwsift-{}-*{}.tar.bz2".format(version.__version__, i))
        glob_results = glob(f)
        if len(glob_results) == 1:
            log.info("Conda package name is: %s", glob_results[0])
            return glob_results[0]
    raise FileNotFoundError("Conda package was not built")


def _scp(src, dst):
    cmd = "pscp" if platform == "win" else "scp"
    log.info("SCPing {} to {}".format(src, dst))
    run("{} {} {}".format(cmd, src, dst).split(" "))


def _ssh(host, command):
    log.info("SSHing {} to run command '{}'".format(host, command))
    run("ssh {} {}".format(host, command).split(" "))


def _run_pyinstaller():
    log.info("Building installer...")
    shutil.rmtree(os.path.join("dist", "SIFT"), ignore_errors=True)
    run("pyinstaller --clean -y sift-pyinstaller-package.spec".split(" "))


def package_installer_osx():
    os.chdir("dist")
    vol_name = "SIFT_{}".format(version.__version__)
    dmg_name = vol_name + ".dmg"
    run(f"hdiutil create -volname {vol_name} -fs HFS+ -srcfolder SIFT.app -ov -format UDZO {dmg_name}".split(" "))
    return dmg_name


def package_installer_linux():
    os.chdir("dist")
    vol_name = "SIFT_{}.tar.gz".format(version.__version__)
    run("tar -czf {} SIFT".format(vol_name).split(" "))
    return vol_name


def package_installer_win():
    from uwsift.util.default_paths import DOCUMENT_SETTINGS_DIR, WORKSPACE_DB_DIR

    new_env = os.environ.copy()
    new_env["WORKSPACE_DB_DIR"] = WORKSPACE_DB_DIR
    new_env["DOCUMENT_SETTINGS_DIR"] = DOCUMENT_SETTINGS_DIR
    run([ISCC_PATH, "uwsift.iss"], env=new_env)
    vol_name = "SIFT_{}.exe".format(version.__version__)
    vol_name = os.path.join("sift_inno_setup_output", vol_name)
    old_name = os.path.join("sift_inno_setup_output", "setup.exe")
    shutil.move(old_name, vol_name)
    return vol_name


INSTALLER_PACKAGER = {
    "darwin": package_installer_osx,
    "linux": package_installer_linux,
    "win": package_installer_win,
}


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Build conda and all-in-one installers for SIFT")
    parser.add_argument("--no-conda", dest="build_conda", action="store_false", help="Don't build a conda package")
    parser.add_argument(
        "--no-conda-upload",
        dest="upload_conda",
        action="store_false",
        help="Don't upload conda package to local channel server",
    )
    parser.add_argument(
        "--no-conda-index", dest="index_conda", action="store_false", help="Don't update remote conda index"
    )
    parser.add_argument(
        "--python",
        default="3.7",
        help="Specify what version of python to build the conda package for (see conda-build " "documentation.)",
    )
    parser.add_argument(
        "--no-installer", dest="build_installer", action="store_false", help="Don't build an installer with pyinstaller"
    )
    parser.add_argument(
        "--no-installer-upload",
        dest="upload_installer",
        action="store_false",
        help="Don't upload installer to server permitted to upload to FTP",
    )
    parser.add_argument("--conda-host-user", default=os.getlogin(), help="Username on conda channel server")
    parser.add_argument("--ftp-host-user", default=os.getlogin(), help="Username on server permitted to upload to FTP")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG)

    os.chdir(SCRIPT_DIR)
    if args.build_conda:
        conda_pkg = _build_conda(python_version=args.python)
        if args.upload_conda:
            ch_path = os.path.join(CHANNEL_PATH, CONDA_PLAT[platform])
            _scp(conda_pkg, "{}@{}:{}".format(args.ftp_host_user, CHANNEL_HOST, ch_path))
            if args.index_conda:
                _ssh(CHANNEL_HOST, "/home/davidh/miniconda3/bin/conda index {}".format(ch_path))

    if args.build_installer:
        _run_pyinstaller()
        pkg_name = INSTALLER_PACKAGER[platform]()
        if args.upload_installer:
            _scp(pkg_name, "{}@{}:{}".format(args.ftp_host_user, FTP_HOST, FTP_HOST_PATH))


if __name__ == "__main__":
    sys.exit(main())
