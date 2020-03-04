#!/usr/bin/env python3
"""Create a conda-pack'd SIFT installation tarball.

Note: This script will place extra files in the currently activated python
environment in order to include these files in the produced tarball.

SIFT must be installed in the current environment with::

    pip install --no-deps .

Instead of installing it in development mode (`-e`).

Example::

    python build_conda_pack.py -c
"""

import os
import sys
import shutil
import subprocess


def get_version():
    try:
        from uwsift import __version__
        return __version__
    except ImportError:
        raise RuntimeError("Could not determine SIFT version. Is SIFT installed?")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Build SIFT installation tarball (remaining arguments "
                    "are passed to conda-pack)")
    parser.add_argument('--arcroot',
                        help="Directory name inside the tarball (default: SIFT_X.Y.Z)")
    parser.add_argument('-o', '--output',
                        help="Pathname for bundled file. Default is "
                             "'SIFT_X.Y.Z_<platform>.<ext>' where platform is "
                             "'linux', 'darwin', or 'win32' and ext is "
                             "'.tar.gz' for linux and OSX, '.zip' for Windows.")
    args, unknown_args = parser.parse_known_args()

    version = get_version()
    if args.arcroot is None:
        args.arcroot = f"SIFT_{version}"
    if args.output is None:
        ext = '.zip' if 'win' in sys.platform else '.tar.gz'
        args.output = f"SIFT_{version}_{sys.platform}.{ext}"

    # Copy appropriate wrapper scripts
    dst = sys.prefix
    script_dir = os.path.realpath(os.path.dirname(__file__))
    if 'nux' in sys.platform:
        script = os.path.join(script_dir, 'bundle_scripts', 'SIFT.sh')
        shutil.copyfile(script, os.path.join(dst, 'SIFT.sh'))
    elif 'darwin' in sys.platform:
        script = os.path.join(script_dir, 'bundle_scripts', 'SIFT.sh')
        shutil.copyfile(script, os.path.join(dst, 'SIFT.command'))
    elif 'win' in sys.platform:
        script = os.path.join(script_dir, 'bundle_scripts', 'SIFT.bat')
        shutil.copyfile(script, os.path.join(dst, 'SIFT.bat'))
    else:
        raise RuntimeError(f"Unknown platform: {sys.platform}")

    subprocess.check_call(['conda-pack', '--arcroot', args.arcroot,
                          '--output', args.output] + unknown_args)

    # TODO: Do additional risky cleanup to reduce output file size


if __name__ == "__main__":
    sys.exit(main())