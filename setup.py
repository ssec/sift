#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of SIFT.
#
# SIFT is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# SIFT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with SIFT.  If not, see <http://www.gnu.org/licenses/>.
"""Setuptools installation script for the SIFT python package.

To install from source run the following command::

    python setup.py install

To install for development replace 'install' with 'develop' in the above
command.

.. note::

    PyQt4 is required for GUI operations, but must be install manually
    since it is not 'pip' installable.

For Developers
--------------

To bump the version run:

    python setup.py bump -b minor -d alpha

Or to tag and commit the new version:

    python setup.py bump -b major -t -c

See the `-h` options for more info.

"""

import os
import re
from setuptools import setup, find_packages, Command

script_dir = os.path.dirname(os.path.realpath(__file__))
version_pathname = os.path.join(script_dir, "uwsift", "version.py")
version_str = open(version_pathname).readlines()[-1].split()[-1].strip("\"\'")
version_regex = re.compile('^(?P<major>\d+)\.(?P<minor>\d+)\.(?P<micro>\d+)(?:(?P<dev_level>(a|b|rc))(?P<dev_version>\d))?$')
version_info = version_regex.match(version_str).groupdict()
assert version_info is not None, "Invalid version in version.py: {}".format(version_str)
version_info["major"] = int(version_info["major"])
version_info["minor"] = int(version_info["minor"])
version_info["micro"] = int(version_info["micro"])
version_info["dev_version"] = int(version_info["dev_version"] or 0)

extras_require = {
    "docs": ['blockdiag', 'sphinx', 'sphinx_rtd_theme',
             'sphinxcontrib-seqdiag', 'sphinxcontrib-blockdiag'],
}


class BumpCommand(Command):
    description = "bump package version by one micro, minor, or major version number (major.minor.micro[a/b])"
    user_options = [
        ("bump-level=", 'b', "major, minor, micro (default: None)"),
        ("dev-level=", 'd', "alpha, beta, rc (default: None)"),
        ("new-version=", 'v', "specify exact new version number (default: None"),
        ("dry-run", 'n', "dry run, don't change anything"),
        ("tag", "t", "add a git tag for this version (default: False)"),
        ("commit", "c", "Run the git commit command but do not push (default: False)"),
    ]
    boolean_options = ["dry_run", "tag", "commit"]

    def initialize_options(self):
        self.bump_level = None
        self.dev_level = None
        self.new_version = None
        self.dry_run = False
        self.tag = False
        self.commit = False

    def finalize_options(self):
        if self.bump_level not in ["major", "minor", "micro", None]:
            raise ValueError("Bump level must be one of ['major', 'minor', 'micro', <unspecified>]")
        if self.dev_level not in ["alpha", "beta", "rc", None]:
            raise ValueError("Dev level must be one of ['alpha', 'beta', <unspecified>]")

    def run(self):
        current_version = version_info
        new_version = current_version.copy()
        if self.new_version is not None:
            new_version_str = self.new_version
            assert version_regex.match(new_version_str) is not None
        else:
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

            if self.dev_level:
                short_level = {'alpha': 'a', 'beta': 'b', 'rc': 'rc'}[self.dev_level]
                if current_version["dev_level"] == short_level and self.bump_level is None:
                    new_dev_version = current_version["dev_version"] + 1
                else:
                    new_dev_version = 0
                new_version_str += "{:s}{:d}".format(short_level, new_dev_version)

        # Update the version test in the version.py file
        print("Old Version: {}".format(version_str))
        print("New Version: {}".format(new_version_str))

        if self.dry_run:
            print("### Dry Run: No modifications ###")
            return

        # Update the version.py
        print("Updating version.py...")
        version_data = open(version_pathname, "r").read()
        version_data = version_data.replace("__version__ = \"{}\"".format(version_str),
                                            "__version__ = \"{}\"".format(new_version_str))
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
        if self.commit:
            import subprocess
            print("Adding files to git staging area...")
            subprocess.check_call(add_args)
            print("Committing changes...")
            subprocess.check_call(commit_args)
        else:
            commit_args[-1] = "\"" + commit_args[-1] + "\""
            print("To appropriate files:")
            print("    ", " ".join(add_args))
            print("To commit after run:")
            print("    ", " ".join(commit_args))

        tag_args = ["git", "tag", "-a", new_version_str, "-m", "Version {}".format(new_version_str)]
        if self.tag:
            print("Tagging commit...")
            subprocess.check_call(tag_args)
        else:
            tag_args[-1] = "\"" + tag_args[-1] + "\""
            print("To tag:")
            print("    ", " ".join(tag_args))

        print("To push git changes to remote, run:\n    git push --follow-tags")


readme = open(os.path.join(script_dir, 'README.md')).read()

setup(
    name='uwsift',
    version=version_str,
    description="Satellite Information Familiarization Tool",
    long_description=readme,
    long_description_content_type='text/markdown',
    author='R.K.Garcia, University of Wisconsin - Madison Space Science & Engineering Center',
    author_email='rkgarcia@wisc.edu',
    url='https://github.com/ssec/sift',
    classifiers=["Development Status :: 5 - Production/Stable",
                 "Intended Audience :: Science/Research",
                 "License :: OSI Approved :: GNU General Public License v3 " +
                 "or later (GPLv3+)",
                 "Operating System :: OS Independent",
                 "Programming Language :: Python",
                 "Programming Language :: Python :: 3",
                 "Topic :: Scientific/Engineering"],
    zip_safe=False,
    include_package_data=True,
    install_requires=['numpy', 'pillow', 'numba', 'vispy>=0.7.1',
                      'netCDF4', 'h5py', 'pyproj',
                      'pyshp', 'shapely', 'rasterio', 'sqlalchemy',
                      'appdirs', 'pyyaml', 'pyqtgraph', 'satpy', 'matplotlib',
                      'scikit-image', 'donfig',
                      'pygrib;sys_platform=="linux" or sys_platform=="darwin"', 'imageio', 'pyqt5>=5.9'
                      ],
    tests_requires=['pytest', 'pytest-qt', 'pytest-mock'],
    python_requires='>=3.7',
    extras_require=extras_require,
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "SIFT = uwsift.__main__:main",
        ],
    },
    cmdclass={
        'bump': BumpCommand,
    }
)
