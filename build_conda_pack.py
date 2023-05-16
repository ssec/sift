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
import re
import shutil
import subprocess  # nosec: B404
import sys


def get_version():
    try:
        from uwsift import __version__

        return __version__
    except ImportError:
        raise RuntimeError("Could not determine SIFT version. Is SIFT installed?")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Build SIFT installation tarball (remaining arguments " "are passed to conda-pack)"
    )
    parser.add_argument("--arcroot", help="Directory name inside the tarball (default: SIFT_X.Y.Z)")
    parser.add_argument(
        "-o",
        "--output",
        help="Pathname for bundled file. Default is "
        "'SIFT_X.Y.Z_<platform>.<ext>' where platform is "
        "'linux', 'darwin', or 'win32' and ext is "
        "'.tar.gz' for linux and OSX, '.zip' for Windows.",
    )
    args, unknown_args = parser.parse_known_args()

    version = get_version()
    if args.arcroot is None:
        args.arcroot = f"SIFT_{version}"
    if args.output is None:
        ext = ".zip" if sys.platform.startswith("win") else ".tar.gz"
        args.output = f"SIFT_{version}_{sys.platform}{ext}"

    # Copy appropriate wrapper scripts
    dst = sys.prefix
    script_dir = os.path.realpath(os.path.dirname(__file__))
    if "nux" in sys.platform:
        script = os.path.join(script_dir, "bundle_scripts", "SIFT.sh")
        shutil.copy(script, os.path.join(dst, "SIFT.sh"))
    elif "darwin" in sys.platform:
        script = os.path.join(script_dir, "bundle_scripts", "SIFT.sh")
        shutil.copy(script, os.path.join(dst, "SIFT.command"))
    elif "win" in sys.platform:
        script = os.path.join(script_dir, "bundle_scripts", "SIFT.bat")
        shutil.copy(script, os.path.join(dst, "SIFT.bat"))
    else:
        raise RuntimeError(f"Unknown platform: {sys.platform}")

    # HACK: https://github.com/conda/conda-pack/issues/141
    if sys.platform.startswith("win"):
        _hack_conda_packed_qtconf_on_windows()

    subprocess.check_call(  # nosec: B603
        ["conda-pack", "--arcroot", args.arcroot, "--output", args.output] + unknown_args
    )
    os.chmod(args.output, 0o755)  # nosec: B103

    # TODO: Do additional risky cleanup to reduce output file size


def _hack_conda_packed_qtconf_on_windows():
    qt_conf_path = os.path.join(sys.prefix, "Library", "bin", "qt.conf")
    if not os.path.exists(qt_conf_path):
        return
    with open(qt_conf_path, "rt") as qtconf:
        old_text = qtconf.read()
    (old_prefix,) = tuple(re.findall(r"^Prefix\s*=\s*(.*?).Library\s*$", old_text, re.MULTILINE))
    new_prefix = old_prefix.replace("/", "\\")
    new_text = old_text.replace(old_prefix, new_prefix)
    with open(os.path.join(sys.prefix, "qt.conf"), "wt") as qtconf:
        qtconf.write(new_text)
    with open(os.path.join(sys.prefix, "Library", "bin", "qt.conf"), "wt") as qtconf:
        qtconf.write(new_text)


if __name__ == "__main__":
    sys.exit(main())
