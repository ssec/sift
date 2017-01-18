#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
.py
~~~

PURPOSE


REFERENCES


REQUIRES


:author: R.K.Garcia <rayg@ssec.wisc.edu>
:copyright: 2016 by University of Wisconsin Regents, see AUTHORS for more details
:license: GPLv3, see LICENSE for more details
"""
__author__ = 'rayg'
__docformat__ = 'reStructuredText'

import os, sys
import logging, unittest, argparse
import numpy as np
import netCDF4 as nc4

LOG = logging.getLogger(__name__)

DEFAULT_RADIANCE_VAR_NAME = 'Rad'

# ref https://gitlab.ssec.wisc.edu/scottm/QL_package/blob/master/cspp_ql/geo_proj.py
def proj4_params(lon_center_of_projection, perspective_point_height, semi_major_axis, semi_minor_axis,
                sweep_angle_axis='x', x_plus=0, y_plus=0, **etc):
    return (('proj', 'geos'),
            ('lon_0', lon_center_of_projection),
            ('h', perspective_point_height),
            ('x_0', x_plus),
            ('y_0', y_plus),
            ('a', semi_major_axis),
            ('b', semi_minor_axis),
            ('sweep', sweep_angle_axis),
            ('ellps', 'GRS80'),  # FIXME: redundant given +a and +b?
            ('units', 'm'),
            )


def proj4_string(lon_center_of_projection, perspective_point_height, x_plus, y_plus, semi_major_axis, semi_minor_axis,
                 **etc):
    p4p = proj4_params(lon_center_of_projection, perspective_point_height, x_plus, y_plus, semi_major_axis, semi_minor_axis, **etc)
    return ' '.join( ('+%s=%s' % q) for q in p4p )
    # return '+proj=geos +sweep=x +lon_0=%f +h=%f +x_0=%d +y_0=%d +a=%f +b=%f +units=m +ellps=GRS80' % (
    #     lon_center_of_projection,
    #     perspective_point_height,
    #     x_plus,
    #     y_plus,
    #     semi_major_axis,
    #     semi_minor_axis
    # )


def calc_bt(L: np.ndarray, fk1, fk2, bc1, bc2, **etc):
    """
    convert brightness temperature bands from radiance
    ref PUG Vol4 7.1.3.1 Radiances Product : Description
    Note: marginally negative radiances can occur and will result in NaN values!
    :param: bt: output array, preallocated to match size of input
    :param L: raw image data as masked array (dtype=(np.float32, np.float64))
    :param fk1: calibration constant
    :param fk2: calibration constant
    :param bc1: calibration constant
    :param bc2: calibration constant
    :return: BT converted radiance data
    """
    T = (fk2 / (np.log(fk1 / L) + 1.0) - bc1) / bc2
    # Tmin = -bc1 / bc2
    T[L <= 0.0] = 0.0  # Tmin
    return T


def calc_refl(L: np.ndarray, kappa0, **etc):
    """
    convert reflectance bands from radiance
    ref PUG Vol4 7.1.3.1 Radiances Product : Description
    :param L: radiance array
    :param kappa0: conversion factor radiance to reflectance
    :return:
    """
    return L * kappa0


def is_band_refl_or_bt(band):
    """
    :param band: band number, 1..16
    :return: "refl", "bt" or None
    """
    return 'refl' if (1 <= band <= 6) else 'bt' if (7 <= band <= 16) else None


def nc_cal_values(nc):
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


def nc_nav_values(nc, radiance_var_name=None, proj_var_name=None):
    """
    extract a dictionary of navigation parameters to use
    :param nc: PUG netCDF4 instance
    :return: dict
    """
    if proj_var_name is None:
        v = nc.get(radiance_var_name or DEFAULT_RADIANCE_VAR_NAME, None)
        if v is not None:
            a = getattr(v, 'grid_mapping', None)
            if a is not None:
                proj_var_name = a
    if proj_var_name is None:
        raise ValueError('unknown projection variable to read')
    proj = nc[proj_var_name]
    return dict((name, getattr(proj, name, None)) for name in (
        'grid_mapping_name', 'perspective_point_height', 'semi_major_axis', 'semi_minor_axis',
        'inverse_flattening', 'latitude_of_projection_origin', 'longitude_of_projection_origin', 'sweep_angle_axis'))


class PugBandTools(object):
    """
    PUG helper routines for a single band, given a single-band NetCDF4 file
    or a multi-band NetCDF4 file and an offset in the band dimension
    """
    band = None
    cal = None
    nav = None
    rad_var_name = None

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
        if kind=='refl':
            return (kind, calc_refl(rad, **self.cal), '')
        elif kind=='bt':
            return (kind, calc_bt(rad, **self.cal), 'K')

    def __init__(self, nc:nc4.Dataset, radiance_var=DEFAULT_RADIANCE_VAR_NAME, band_offset=0):
        super(PugBandTools, self).__init__()
        self.rad_var_name = radiance_var
        self.band = nc['band_id'][band_offset]  # FUTURE: this has a band dimension
        self.cal = nc_cal_values(nc)            # but for some reason these do not?
        self.nav = nc_nav_values(nc, self.rad_var_name)

    @property
    def proj4(self):
        return proj4_string(**self.cal)

    @property
    def proj4_params(self):
        return proj4_params(**self.cal)

    def convert_from_nc(self, nc):
        rad = nc[self.rad_var_name][:]
        return self.convert(rad)


class tests(unittest.TestCase):
    data_file = os.environ.get('TEST_DATA', os.path.expanduser("~/Data/test_files/thing.dat"))

    def setUp(self):
        pass

    def test_something(self):
        pass


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
