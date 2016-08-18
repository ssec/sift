#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Setuptools installation script for the CSPOV python package.

To install from source run the following command::

    python setup.py install

To install for development replace 'install' with 'develop' in the above
command.

.. note::

    PyQt4 is required for GUI operations, but must be install manually
    since it is not 'pip' installable.

"""

import os
import re
from setuptools import setup, find_packages, Command

script_dir = os.path.dirname(os.path.realpath(__file__))
version_pathname = os.path.join(script_dir, "cspov", "version.py")
version_str = open(version_pathname).readlines()[-1].split()[-1].strip("\"\'")
version_regex = re.compile('^(?P<major>\d+)\.(?P<minor>\d+)\.(?P<micro>\d+)(?P<dev_level>[ab]?)(?P<dev_version>\d+)?')
version_info = version_regex.match(version_str).groupdict()
assert version_info is not None, "Invalid version in version.py: {}".format(version_str)
version_info["major"] = int(version_info["major"])
version_info["minor"] = int(version_info["minor"])
version_info["micro"] = int(version_info["micro"])
version_info["dev_version"] = int(version_info["dev_version"])

extras_require = {
    "docs": ['blockdiag', 'sphinx', 'sphinx_rtd_theme',
             'sphinxcontrib-seqdiag', 'sphinxcontrib-blockdiag'],
}


class BumpCommand(Command):
    description = "bump package version by one micro, minor, or major version number (major.minor.micro[a/b])"
    user_options = [
        ("bump-level=", 'b', "major, minor, micro (default: None)"),
        ("dev-level=", 'd', "alpha or beta (default: None"),
        ("commit", "c", "Run the git commit command but do not push (default: False)"),
    ]
    boolean_options = ["commit"]

    def initialize_options(self):
        self.bump_level = None
        self.dev_level = None
        self.commit = False

    def finalize_options(self):
        if self.bump_level not in ["major", "minor", "micro", None]:
            raise ValueError("Bump level must be one of ['major', 'minor', 'micro', <unspecified>]")
        if self.dev_level not in ["alpha", "beta", None]:
            raise ValueError("Dev level must be one of ['alpha', 'beta', <unspecified>]")

    def run(self):
        current_version = version_info
        new_version = current_version.copy()
        if self.bump_level == "micro":
            new_version["micro"] += 1
        elif self.bump_level == "minor":
            new_version["minor"] += 1
            new_version["micro"] = 0
        elif self.bump_level == "major":
            new_version["major"] += 1
            new_version["minor"] = 0
            new_version["micro"] = 0
        new_version_str = "{major:d}.{minor:d}.{micro:d}".format(**new_version)

        if self.dev_level == "alpha":
            if current_version["dev_level"] == "a" and self.bump_level is None:
                new_dev_version = current_version["dev_version"] + 1
            else:
                new_dev_version = 0
            new_version_str += "a{:d}".format(new_dev_version)
        elif self.dev_level == "beta":
            if current_version["dev_level"] == "b" and self.bump_level is None:
                new_dev_version = current_version["dev_version"] + 1
            else:
                new_dev_version = 0
            new_version_str += "b{:d}".format(new_dev_version)

        # Update the version test in the version.py file
        print("Old Version: {}".format(version_str))
        print("New Version: {}".format(new_version_str))

        # Update the version.py
        print("Updating version.py...")
        version_data = open(version_pathname, "r").read()
        version_data = version_data.replace("__version__ = \"{}\"".format(version_str), "__version__ = \"{}\"".format(new_version_str))
        open(version_pathname, "w").write(version_data)

        # Updating Windows Inno Setup file
        # XXX: Once PyInstaller executable is properly encoded with version this may be removed after proper fixes
        print("Updating Inno Setup Version number...")
        iss_pathname = os.path.join(script_dir, "sift.iss")
        file_data = open(iss_pathname, "rb").read()
        _old = "AppVersion={}".format(version_str).encode()
        _new = "AppVersion={}".format(new_version_str).encode()
        file_data = file_data.replace(_old, _new)
        open(iss_pathname, "wb").write(file_data)

        # Tag git repository commit
        add_args = ["git", "add", version_pathname, iss_pathname]
        commit_args = ["git", "commit", "-m", "Bump version from {} to {}".format(version_str, new_version_str)]
        tag_args = ["git", "tag", "-a", new_version_str, "-m", "Version {}".format(new_version_str)]
        if self.commit:
            import subprocess
            print("Adding files to git staging area...")
            subprocess.check_call(add_args)
            print("Committing changes...")
            subprocess.check_call(commit_args)
            print("Tagging commit...")
            subprocess.check_call(tag_args)
        else:
            commit_args[-1] = "\"" + commit_args[-1] + "\""
            tag_args[-1] = "\"" + tag_args[-1] + "\""
            print("To appropriate files:")
            print("    ", " ".join(add_args))
            print("To commit after run:")
            print("    ", " ".join(commit_args))
            print("Followed by:")
            print("    ", " ".join(tag_args))
        print("Run:\n    git push --follow-tags")

setup(
    name='cspov',
    version=version_str,
    description="Satellite Information Familiarization Tool for mercator geotiff files",
    author='R.K.Garcia, University of Wisconsin - Madison Space Science & Engineering Center',
    author_email='rkgarcia@wisc.edu',
    url='https://www.ssec.wisc.edu/',
    zip_safe=False,
    include_package_data=True,
    install_requires=['numpy', 'pillow', 'scipy', 'numba', 'vispy>0.4.0',
                      'PyOpenGL', 'netCDF4', 'h5py', 'pyproj', 'gdal',
                      'pyshp', 'shapely', 'rasterio',
                      ],
    extras_require=extras_require,
    packages=find_packages(),
    entry_points={},
    cmdclass={
        'bump': BumpCommand,
    }
)
