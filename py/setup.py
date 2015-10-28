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

from setuptools import setup, find_packages

setup(
    name='cspov',
    version='0.4.1',
    description="Fluid high resolution satellite and meteorological imagery viewer",
    author='Ray Garcia, SSEC',
    author_email='ray.garcia@ssec.wisc.edu',
    url='https://www.ssec.wisc.edu/',
    zip_safe=False,
    include_package_data=True,
    install_requires=['numpy', 'pillow', 'scipy', 'numba', 'vispy>0.4.0', 'numpy', 'PyOpenGL', 'netCDF4', 'h5py', 'pyproj', 'gdal', 'pyshp', 'shapely', 'rasterio'],
    packages=find_packages(),
    entry_points={'console_scripts': ['cspov = cspov.__main__:main']}
)
