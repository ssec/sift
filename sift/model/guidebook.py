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

# CF compliant Standard Names (should be provided by input files or the workspace)
# Instrument -> Band Number -> Standard Name
STANDARD_NAMES = {
    PLATFORM.HIMAWARI_8: {
        INSTRUMENT.AHI: {
            1: "toa_bidirectional_reflectance",
            2: "toa_bidirectional_reflectance",
            3: "toa_bidirectional_reflectance",
            4: "toa_bidirectional_reflectance",
            5: "toa_bidirectional_reflectance",
            6: "toa_bidirectional_reflectance",
            7: "toa_brightness_temperature",
            8: "toa_brightness_temperature",
            9: "toa_brightness_temperature",
            10: "toa_brightness_temperature",
            11: "toa_brightness_temperature",
            12: "toa_brightness_temperature",
            13: "toa_brightness_temperature",
            14: "toa_brightness_temperature",
            15: "toa_brightness_temperature",
            16: "toa_brightness_temperature",
        }
    }
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

        if info[INFO.KIND] in (KIND.IMAGE, KIND.COMPOSITE):
            if z.get(INFO.CENTRAL_WAVELENGTH) is None:
                try:
                    wl = NOMINAL_WAVELENGTHS[info[INFO.PLATFORM]][info[INFO.INSTRUMENT]][info[INFO.BAND]]
                except KeyError:
                    wl = None
                z[INFO.CENTRAL_WAVELENGTH] = wl

        if INFO.BAND in info:
            band_short_name = "B{:02d}".format(info[INFO.BAND])
        else:
            band_short_name = info.get(INFO.NAME, '???')
        z[INFO.SHORT_NAME] = info.get(INFO.SHORT_NAME, band_short_name)
        z[INFO.LONG_NAME] = info.get(INFO.LONG_NAME, z[INFO.SHORT_NAME])

        try:
            z[INFO.STANDARD_NAME] = info.get(INFO.STANDARD_NAME, STANDARD_NAMES[info[INFO.PLATFORM]][info[INFO.INSTRUMENT]][info[INFO.BAND]])
        except KeyError:
            z[INFO.STANDARD_NAME] = info.get(INFO.STANDARD_NAME, z.get(INFO.SHORT_NAME))

        return z

    def _is_refl(self, dsi):
        # work around for old `if band in BAND_TYPE`
        return dsi.get(INFO.BAND) in self.REFL_BANDS or \
               dsi.get(INFO.STANDARD_NAME) == "toa_bidirectional_reflectance"

    def _is_bt(self, dsi):
        return dsi.get(INFO.BAND) in self.BT_BANDS or \
               dsi.get(INFO.STANDARD_NAME) == "toa_brightness_temperature"

    def climits(self, dsi):
        # Valid min and max for colormap use for data values in file (unconverted)
        if self._is_refl(dsi):
            # Reflectance/visible data limits
            return -0.012, 1.192
        elif self._is_bt(dsi):
            # BT data limits
            return -109.0 + 273.15, 55 + 273.15
        else:
            return None, None

    def units_conversion(self, dsi):
        "return UTF8 unit string, lambda v,inverse=False: convert-raw-data-to-unis"
        if self._is_refl(dsi):
            # Reflectance/visible data limits
            return ('{:.03f}', '', lambda x, inverse=False: x)
        elif self._is_bt(dsi):
            # BT data limits, Kelvin to degC
            return ("{:.02f}", "°C", lambda x, inverse=False: x - 273.15 if not inverse else x + 273.15)
        else:
            return ("", "", lambda x, inverse=False: 0.0)

    def default_colormap(self, dsi):
        if self._is_refl(dsi):
            return DEFAULT_VIS
        elif self._is_bt(dsi):
            return DEFAULT_IR
        else:
            return DEFAULT_UNKNOWN

    # def display_time(self, dsi):
    #     return dsi.get(INFO.DISPLAY_TIME, '--:--')
    #
    # def display_name(self, dsi):
    #     return dsi.get(INFO.DISPLAY_NAME, '-unknown-')

    def _default_display_time(self, ds_info):
        # FUTURE: This can be customized by the user
        when = ds_info.get(INFO.SCHED_TIME, ds_info.get(INFO.OBS_TIME))
        if when is None:
            dtime = '--:--'
        else:
            dtime = when.strftime('%Y-%m-%d %H:%M')
        return dtime

    def _default_display_name(self, ds_info):
        # FUTURE: This can be customized by the user
        sat = ds_info[INFO.PLATFORM]
        inst = ds_info[INFO.INSTRUMENT]
        name = ds_info.get(INFO.SHORT_NAME, '-unknown-')

        label = ds_info.get(INFO.STANDARD_NAME, '')
        if label == 'toa_bidirectional_reflectance':
            label = 'Refl'
        elif label == 'toa_brightness_temperature':
            label = 'BT'

        dtime = ds_info.get(INFO.DISPLAY_TIME, self._default_display_time(ds_info))
        name = "{inst} {name} {standard_name} {dtime}".format(inst=inst.value, name=name, standard_name=label, dtime=dtime)
        return name



# if __name__ == '__main__':
#     sys.exit(main())
