#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
goesr_pug.py
============

PURPOSE
Toolbox for extracting imagery from GOES-16 PUG L1B NetCDF4 files

REFERENCES
GOES-R Product User's Guide Rev E

REQUIRES
numpy
netCDF4


:author: R.K.Garcia <rayg@ssec.wisc.edu>
:copyright: 2016 by University of Wisconsin Regents, see AUTHORS for more details
:license: GPLv3, see LICENSE for more details
"""
import os, sys
import logging, unittest, argparse
import numpy as np
import netCDF4 as nc4

__author__ = 'rayg'
__docformat__ = 'reStructuredText'

LOG = logging.getLogger(__name__)

# default variable names for PUG L1b files
DEFAULT_RADIANCE_VAR_NAME = 'Rad'
DEFAULT_Y_CENTER_VAR_NAME = 'y_image'
DEFAULT_X_CENTER_VAR_NAME = 'x_image'


# ref https://gitlab.ssec.wisc.edu/scottm/QL_package/blob/master/cspp_ql/geo_proj.py
def proj4_params(longitude_of_projection_origin=0.0, perspective_point_height=0.0,
                 semi_major_axis=0.0, semi_minor_axis=0.0,
                 sweep_angle_axis='x', latitude_of_projection_origin=0.0, y_0=0, x_0=0, **etc):
    """
    Generate PROJ.4 parameters for Fixed Grid projection
    :param longitude_of_projection_origin: longitude at center of projection, related to sub-satellite point
    :param perspective_point_height: effective projection height in m
    :param semi_major_axis: ellipsoid semi-major axis in m
    :param semi_minor_axis: ellipsoid semi-minor axis in m
    :param sweep_angle_axis: 'x' for GOES-R, 'y' for original CGMS
    :param y_0:
    :param x_0:
    :return: tuple of (key,value) pairs
    """
    assert(latitude_of_projection_origin==0.0)  # assert your assumptions: need to revisit this routine if this is not the case
    return (('proj', 'geos'),
            ('lon_0', longitude_of_projection_origin),
            ('h', perspective_point_height),
            ('a', semi_major_axis),
            ('b', semi_minor_axis),
            ('x_0', x_0),
            ('y_0', y_0),
            ('sweep', sweep_angle_axis),
            ('ellps', 'GRS80'),  # FIXME: probably redundant given +a and +b, PUG states GRS80 and not WGS84
            ('units', 'm'),
            )


def proj4_string(**kwargs):
    """
    see proj4_params()
    """
    p4p = proj4_params(**kwargs)
    return ' '.join( ('+%s=%s' % q) for q in p4p )
    # return '+proj=geos +sweep=x +lon_0=%f +h=%f +x_0=%d +y_0=%d +a=%f +b=%f +units=m +ellps=GRS80' % (
    #     lon_center_of_projection,
    #     perspective_point_height,
    #     x_plus,
    #     y_plus,
    #     semi_major_axis,
    #     semi_minor_axis
    # )


def calc_bt(L: np.ndarray, fk1: float, fk2: float, bc1: float, bc2: float, **etc):
    """
    convert brightness temperature bands from radiance
    ref PUG Vol4 7.1.3.1 Radiances Product : Description
    Note: marginally negative radiances can occur and will result in NaN values!
    :param L: raw radiance image data as masked array
    :param fk1: calibration constant
    :param fk2: calibration constant
    :param bc1: calibration constant
    :param bc2: calibration constant
    :return: BT converted radiance data, K
    """
    T = (fk2 / (np.log(fk1 / L) + 1.0) - bc1) / bc2
    # Tmin = -bc1 / bc2   # when L<=0
    T[L <= 0.0] = 0.0  # Tmin, but for now truncate to absolute 0
    return T


def calc_refl(L: np.ndarray, kappa0: float, **etc):
    """
    convert reflectance bands from radiance
    ref PUG Vol4 7.1.3.1 Radiances Product : Description
    :param L: raw radiance image data as masked array
    :param kappa0: conversion factor radiance to reflectance
    :return:
    """
    return L * kappa0


def is_band_refl_or_bt(band: int):
    """
    :param band: band number, 1..16
    :return: "refl", "bt" or None
    """
    return 'refl' if (1 <= band <= 6) else 'bt' if (7 <= band <= 16) else None


def nc_cal_values(nc: nc4.Dataset):
    """
    extract a dictionary of calibration parameters to use
    :param nc: PUG netCDF4 instance
    :return: dict
    """
    return {
        'fk1': float(nc['planck_fk1'][:]),
        'fk2': float(nc['planck_fk2'][:]),
        'bc1': float(nc['planck_bc1'][:]),
        'bc2': float(nc['planck_bc2'][:]),
        'kappa0': float(nc['kappa0'][:])
    }


def nc_nav_values(nc, radiance_var_name: str=None,
                  proj_var_name: str=None,
                  y_image_var_name: str=DEFAULT_Y_CENTER_VAR_NAME,
                  x_image_var_name: str=DEFAULT_X_CENTER_VAR_NAME):
    """
    extract a dictionary of navigation parameters to use
    :param radiance_var_name: radiance variable name to follow .grid_mapping of, optional with default
    :param proj_var_name: alternately, projection variable to fetch parameters from
    :param y_image_var_name: variable holding center of image in FGF y/x radiance, if available
    :param x_image_var_name: variable holding center of image in FGF y/x radiance, if available
    :param nc: PUG netCDF4 instance
    :return: dict
    """
    if proj_var_name is None:
        v = nc.variables.get(radiance_var_name or DEFAULT_RADIANCE_VAR_NAME, None)
        if v is not None:
            a = getattr(v, 'grid_mapping', None)
            if a is not None:
                proj_var_name = a
    if proj_var_name is None:
        raise ValueError('unknown projection variable to read')
    proj = nc[proj_var_name]
    nav = dict((name, getattr(proj, name, None)) for name in (
        'grid_mapping_name', 'perspective_point_height', 'semi_major_axis', 'semi_minor_axis',
        'inverse_flattening', 'latitude_of_projection_origin', 'longitude_of_projection_origin', 'sweep_angle_axis'))

    x_0, y_0 = 0, 0
    if y_image_var_name in nc.variables.keys():
        y_0 = float(nc[y_image_var_name][:])
    if x_image_var_name in nc.variables.keys():
        x_0 = float(nc[x_image_var_name][:])
    nav['y_0'], nav['x_0'] = y_0, x_0

    return nav


class PugL1bTools(object):
    """
    PUG helper routines for a single band, given a single-band NetCDF4 file
    or a multi-band NetCDF4 file and an offset in the band dimension
    """
    band = None  # band number
    cal = None  # dictionary of cal parameters
    nav = None  # dictionary of nav parameters
    shape = None  # (lines, elements)
    rad_var_name = None  # radiance variable to focus on, default is DEFAULT_RADIANCE_VAR_NAME

    @property
    def bt_or_refl(self):
        return is_band_refl_or_bt(self.band)

    def convert(self, rad):
        """
        calculate BT or Refl based on band
        :param rad: radiance array from file
        :return: ('refl' or 'bt', measurement array, 'K' or '')
        """
        kind = self.bt_or_refl
        if kind == 'refl':
            return kind, calc_refl(rad, **self.cal), ''
        elif kind == 'bt':
            return kind, calc_bt(rad, **self.cal), 'K'

    def __init__(self, nc: nc4.Dataset, radiance_var: str=DEFAULT_RADIANCE_VAR_NAME, band_offset: int=0):
        """
        Initialize cal and nav helper object from a PUG NetCDF4 L1B file
        :param nc:
        :param radiance_var: radiance variable to convert to BT/Refl
        :param band_offset: typically 0, eventually nonzero for multi-band files
        """
        super(PugL1bTools, self).__init__()
        self.rad_var_name = radiance_var
        self.band = int(nc['band_id'][band_offset])  # FUTURE: this has a band dimension
        self.cal = nc_cal_values(nc)            # but for some reason these do not?
        self.nav = nc_nav_values(nc, self.rad_var_name)
        self.shape = nc[radiance_var].shape

    @property
    def proj4(self):
        return proj4_string(**self.cal)

    @property
    def proj4_params(self):
        return proj4_params(**self.nav)

    @property
    def proj4_string(self):
        return proj4_string(**self.nav)

    def convert_from_nc(self, nc: nc4.Dataset):
        rad = nc[self.rad_var_name][:]
        return self.convert(rad)


class tests(unittest.TestCase):
    data_file = os.environ.get('TEST_DATA', os.path.expanduser("~/mnt/changeo/home/rayg/Data/test_data/goesr/awgonly/2017/020/OR_ABI-L1b-RadF-M3C12_G16_s20170200635589_e20170200646362_c20170200646413.nc"))

    def setUp(self):
        pass

    def test_proj4_string(self):
        nc = nc4.Dataset(self.data_file)
        t = PugL1bTools(nc)
        print(t.proj4_string)
        nc.close()

    @staticmethod
    def hello():
        nc = nc4.Dataset(tests.data_file)
        t = PugL1bTools(nc)
        t.nc = nc
        return t



def _debug(type, value, tb):
    "enable with sys.excepthook = debug"
    if not sys.stdin.isatty():
        sys.__excepthook__(type, value, tb)
    else:
        import traceback, pdb
        traceback.print_exception(type, value, tb)
        # …then start the debugger in post-mortem mode.
        pdb.post_mortem(tb)  # more “modern”


def main():
    parser = argparse.ArgumentParser(
        description="PURPOSE",
        epilog="",
        fromfile_prefix_chars='@')
    parser.add_argument('-v', '--verbose', dest='verbosity', action="count", default=0,
                        help='each occurrence increases verbosity 1 level through ERROR-WARNING-INFO-DEBUG')
    parser.add_argument('-d', '--debug', dest='debug', action='store_true',
                        help="enable interactive PDB debugger on exception")
    # http://docs.python.org/2.7/library/argparse.html#nargs
    # parser.add_argument('--stuff', nargs='5', dest='my_stuff',
    #                    help="one or more random things")
    parser.add_argument('inputs', nargs='*',
                        help="input files to process")
    args = parser.parse_args()

    levels = [logging.ERROR, logging.WARN, logging.INFO, logging.DEBUG]
    logging.basicConfig(level=levels[min(3, args.verbosity)])

    if args.debug:
        sys.excepthook = _debug

    if not args.inputs:
        unittest.main()
        return 0

    for pn in args.inputs:
        pass

    return 0


if __name__ == '__main__':
    sys.exit(main())
