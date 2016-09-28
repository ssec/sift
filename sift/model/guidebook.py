#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
guidebook.py
~~~~~~~~~~~~

PURPOSE
This module is the "scientific expert knowledge" that is consulted.



:author: R.K.Garcia <rayg@ssec.wisc.edu>
:copyright: 2014 by University of Wisconsin Regents, see AUTHORS for more details
:license: GPLv3, see LICENSE for more details
"""
__author__ = 'rayg'
__docformat__ = 'reStructuredText'

import logging
import os
import re
from datetime import datetime

from sift.common import INFO, KIND, INSTRUMENT, PLATFORM
from sift.view.Colormap import DEFAULT_IR, DEFAULT_VIS, DEFAULT_UNKNOWN

LOG = logging.getLogger(__name__)

GUIDEBOOKS = {}


class Guidebook(object):
    """
    guidebook which knows about AHI, ABI, AMI bands, timing, file naming conventions
    """
    @staticmethod
    def is_relevant(pathname):
        return False

    @staticmethod
    def for_info(info=None, path=None):
        """
        given an info dictionary, figure out which
        :param info:
        :return:
        """
        if info and not path:
            path = info.get(INFO.PATHNAME, None)

    def channel_siblings(self, uuid, infos):
        """
        determine the channel siblings of a given dataset
        :param uuid: uuid of the dataset we're interested in
        :param infos: datasetinfo_dict sequence, available datasets
        :return: (list,offset:int): list of [uuid,uuid,uuid] for siblings in order; offset of where the input is found in list
        """
        return None, None

    def time_siblings(self, uuid, infos):
        """
        determine the time siblings of a given dataset
        :param uuid: uuid of the dataset we're interested in
        :param infos: datasetinfo_dict sequence, available datasets
        :return: (list,offset:int): list of [uuid,uuid,uuid] for siblings in order; offset of where the input is found in list
        """
        return None, None


# Instrument -> Band Number -> Nominal Wavelength
NOMINAL_WAVELENGTHS = {
    PLATFORM.HIMAWARI_8: {
        INSTRUMENT.AHI: {
            1: 0.47,
            2: 0.51,
            3: 0.64,
            4: 0.86,
            5: 1.6,
            6: 2.3,
            7: 3.9,
            8: 6.2,
            9: 6.9,
            10: 7.3,
            11: 8.6,
            12: 9.6,
            13: 10.4,
            14: 11.2,
            15: 12.4,
            16: 13.3,
        },
    },
    # PLATFORM.HIMAWARI_9: {
    #     INSTRUMENT.AHI: {
    #     },
    # },
    # PLATFORM.GOES_16: {
    #     INSTRUMENT.ABI: {
    #     },
    # },
}



class AHI_HSF_Guidebook(Guidebook):
    "e.g. HS_H08_20150714_0030_B10_FLDK_R20.merc.tif"
    REFL_BANDS = [1, 2, 3, 4, 5, 6]
    BT_BANDS = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

    def collect_info(self, info):
        """Collect information that may not come from the dataset.

        This method should only be called once to "fill in" metadata
        that isn't originally known about an opened file.
        """
        z = {}
        # TODO: Used to be 'info *not* in (KIND.IMAGE, KIND.COMPOSITE)', what's right?
        if info[INFO.KIND] in (KIND.IMAGE, KIND.COMPOSITE):
            if z.get(INFO.CENTRAL_WAVELENGTH) is None:
                try:
                    wl = NOMINAL_WAVELENGTHS[info[INFO.PLATFORM]][info[INFO.INSTRUMENT]][info[INFO.BAND]]
                except KeyError:
                    wl = None
                z[INFO.CENTRAL_WAVELENGTH] = wl
        return z

    def climits(self, dsi):
        # Valid min and max for colormap use for data values in file (unconverted)
        band = dsi.get(INFO.BAND)
        if band in self.REFL_BANDS:
            # Reflectance/visible data limits
            return -0.012, 1.192
        elif band in self.BT_BANDS:
            # BT data limits
            return -109.0 + 273.15, 55 + 273.15
        else:
            return None, None

    def units_conversion(self, dsi):
        "return UTF8 unit string, lambda v,inverse=False: convert-raw-data-to-unis"
        if dsi[INFO.BAND] in self.REFL_BANDS:
            # Reflectance/visible data limits
            return ('{:.03f}', '', lambda x, inverse=False: x)
        elif dsi[INFO.BAND] in self.BT_BANDS:
            # BT data limits, Kelvin to degC
            return ("{:.02f}", "°C", lambda x, inverse=False: x - 273.15 if not inverse else x + 273.15)
        else:
            return ("", "", lambda x, inverse=False: 0.0)

    def default_colormap(self, dsi):
        band = dsi.get(INFO.BAND)
        if band in self.REFL_BANDS:
            return DEFAULT_VIS
        elif band in self.BT_BANDS:
            return DEFAULT_IR
        else:
            return DEFAULT_UNKNOWN

    def display_time(self, dsi):
        return dsi.get(INFO.DISPLAY_TIME, '--:--')

    def display_name(self, dsi):
        return dsi.get(INFO.DISPLAY_NAME, '-unknown-')



# if __name__ == '__main__':
#     sys.exit(main())
