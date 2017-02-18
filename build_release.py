#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import shutil
import logging
import subprocess
from sift import version

if sys.version_info < (3, 5):
    run = subprocess.check_call
else:
    run = subprocess.run


log = logging.getLogger(__name__)

SIFT_CHANNEL = "http://larch.ssec.wisc.edu/channels/sift"
CONDA_RECIPE = os.path.join('conda-recipe', 'sift')
CHANNEL_HOST = os.environ.get("SIFT_CHANNEL_HOST", "larch")
CHANNEL_PATH = os.environ.get("SIFT_CHANNEL_PATH", "/var/apache/larch/htdocs/channels/sift")
# server that is allowed to add to FTP site
FTP_HOST = os.environ.get("SIFT_FTP_HOST", "meelo")
FTP_HOST_PATH = os.environ.get("SIFT_FTP_HOST_PATH", "repos/git/CSPOV/dist")
FTP_PATH = os.environ.get("SIFT_FTP_PATH", "pub/sift/dist")
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
DIST_DIR = os.path.join(SCRIPT_DIR, 'dist')


PLATFORMS = ['darwin', 'linux', 'win']
CONDA_PLAT = {
    'darwin': 'osx-64',
    'linux': 'linux-64',
    'win': 'win-64',
}


def get_platform():
    for k in PLATFORMS:
        if sys.platform.startswith(k):
            return k
    return 'linux'
platform = get_platform()


def _build_conda(output_dir=DIST_DIR):
    try:
        os.makedirs(output_dir)
    except FileExistsError:
        pass

    log.info("Building conda package...")
    CONDA_BUILD_CMD = "conda build -c {} --output-folder {} {}".format(
        SIFT_CHANNEL, DIST_DIR, CONDA_RECIPE)
    run(CONDA_BUILD_CMD.split(' '))
    # check for build revisision
    for i in range(4, -1, -1):
        f = os.path.join(DIST_DIR, 'sift-{}-{}.tar.bz2'.format(version.__version__, i))
        if os.path.isfile(f):
            return f
    raise FileNotFoundError("Conda package was not built")


def _scp(src, dst):
    cmd = 'pscp' if platform == 'win' else 'scp'
    log.info("SCPing {} to {}".format(src, dst))
    run("{} {} {}".format(cmd, src, dst).split(' '))


def _ssh(host, command):
    log.info("SSHing {} to run command '{}'".format(host, command))
    run("ssh {} {}".format(host, command).split(' '))


def _run_pyinstaller():
    log.info("Building installer...")
    run("pyinstaller -y sift.spec".split(' '))


def package_installer_osx():
    os.chdir('dist')
    vol_name = "SIFT_{}".format(version.__version__)
    dmg_name = vol_name + ".dmg"
    run("hdiutil create -volname {} -srcfolder SIFT.app -ov -format UDZO {}".format(vol_name, dmg_name).split(' '))
    return dmg_name


def package_installer_linux():
    os.chdir('dist')
    vol_name = "SIFT_{}.tar.gz".format(version.__version__)
    run("tar -czf {} SIFT".format(vol_name).split(' '))
    return vol_name


def package_installer_win():
    run("iscc \"sift.iss\"")
    vol_name = "SIFT_{}.exe".format(version.__version__)
    vol_name = os.path.join('sift_inno_setup_output', vol_name)
    old_name = os.path.join('sift_inno_setup_output', 'setup.exe')
    shutil.move(old_name, vol_name)
    return vol_name


INSTALLER_PACKAGER = {
    'darwin': package_installer_osx,
    'linux': package_installer_linux,
    'win': package_installer_win,
}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Build conda and all-in-one installers for SIFT")
    parser.add_argument('--no-conda', dest='build_conda', action='store_false',
                        help="Don't build a conda package")
    parser.add_argument('--no-conda-upload', dest='upload_conda', action='store_false',
                        help="Don't upload conda package to local channel server")
    parser.add_argument('--no-conda-index', dest='index_conda', action='store_false',
                        help="Don't update remote conda index")
    parser.add_argument('--no-installer', dest='build_installer', action='store_false',
                        help="Don't build an installer with pyinstaller")
    parser.add_argument('--no-installer-upload', dest='upload_installer', action='store_false',
                        help="Don't upload installer to server permitted to upload to FTP")
    parser.add_argument('--conda-host-user', default=os.getlogin(),
                        help="Username on conda channel server")
    parser.add_argument('--ftp-host-user', default=os.getlogin(),
                        help="Username on server permitted to upload to FTP")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG)

    os.chdir(SCRIPT_DIR)
    if args.build_conda:
        conda_pkg = _build_conda()
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
