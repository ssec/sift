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
from setuptools import setup, find_packages

script_dir = os.path.dirname(os.path.realpath(__file__))
version_pathname = os.path.join(script_dir, "cspov", "version.py")
version = open(version_pathname).readlines()[-1].split()[-1].strip("\"\'")

extras_require = {
    "docs": ['blockdiag', 'sphinx', 'sphinx_rtd_theme',
             'sphinxcontrib-seqdiag', 'sphinxcontrib-blockdiag'],
}

setup(
    name='cspov',
    version=version,
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
    entry_points={}
)
