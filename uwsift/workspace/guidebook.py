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
__author__ = "rayg"
__docformat__ = "reStructuredText"

import logging

from uwsift.common import INVALID_COLOR_LIMITS, Info, Instrument, Platform
from uwsift.view.colormap import DEFAULT_IR, DEFAULT_UNKNOWN, DEFAULT_VIS

LOG = logging.getLogger(__name__)
GUIDEBOOKS: dict = {}


class Guidebook(object):
    """
    guidebook which knows about AHI, ABI, AMI bands, timing, file naming conventions
    """

    def channel_siblings(self, uuid, infos):
        """Determine the channel siblings of a given dataset.

        :param uuid: uuid of the dataset we're interested in
        :param infos: datasetinfo_dict sequence, available datasets
        :return: (list,offset:int): list of [uuid,uuid,uuid] for siblings in order;
                 offset of where the input is found in list
        """
        return None, None

    def time_siblings(self, uuid, infos):
        """Determine the time siblings of a given dataset.

        :param uuid: uuid of the dataset we're interested in
        :param infos: datasetinfo_dict sequence, available datasets
        :return: (list,offset:int): list of [uuid,uuid,uuid] for siblings in order;
            offset of where the input is found in list
        """
        return None, None


DEFAULT_COLORMAPS = {
    "toa_bidirectional_reflectance": DEFAULT_VIS,
    "toa_brightness_temperature": DEFAULT_IR,
    "brightness_temperature": DEFAULT_IR,
    "height_at_cloud_top": "Cloud Top Height",
    "air_temperature": DEFAULT_IR,
    "relative_humidity": DEFAULT_IR,
    # 'thermodynamic_phase_of_cloud_water_particles_at_cloud_top': 'Cloud Phase',
}

_NW_GOESR_ABI = {
    Instrument.ABI: {  # http://www.goes-r.gov/education/ABI-bands-quick-info.html
        1: 0.47,
        2: 0.64,
        3: 0.86,
        4: 1.37,
        5: 1.6,
        6: 2.2,
        7: 3.9,
        8: 6.2,
        9: 6.9,
        10: 7.3,
        11: 8.4,
        12: 9.6,
        13: 10.3,
        14: 11.2,
        15: 12.3,
        16: 13.3,
    },
}

_NW_HIMAWARI_AHI = {
    Instrument.AHI: {
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
}

# Instrument -> Band Number -> Nominal Wavelength
NOMINAL_WAVELENGTHS = {
    Platform.HIMAWARI_8: _NW_HIMAWARI_AHI,
    Platform.HIMAWARI_9: _NW_HIMAWARI_AHI,
    Platform.GOES_16: _NW_GOESR_ABI,
    Platform.GOES_17: _NW_GOESR_ABI,
    Platform.GOES_18: _NW_GOESR_ABI,
    Platform.GOES_19: _NW_GOESR_ABI,
}

# CF compliant Standard Names (should be provided by input files or the workspace)
# Instrument -> Band Number -> Standard Name

_SN_GOESR_ABI = {
    Instrument.ABI: {
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

_SN_HIMAWARI_AHI = {
    Instrument.AHI: {
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
    },
}

STANDARD_NAMES = {
    Platform.HIMAWARI_8: _SN_HIMAWARI_AHI,
    Platform.HIMAWARI_9: _SN_HIMAWARI_AHI,
    Platform.GOES_16: _SN_GOESR_ABI,
    Platform.GOES_17: _SN_GOESR_ABI,
}

BT_STANDARD_NAMES = ["toa_brightness_temperature", "brightness_temperature", "air_temperature"]


class ABI_AHI_Guidebook(Guidebook):
    "e.g. HS_H08_20150714_0030_B10_FLDK_R20.merc.tif"
    _cache = None  # {uuid:metadata-dictionary, ...}

    def __init__(self):
        self._cache = {}

    def collect_info(self, info):
        """Collect information that may not come from the dataset.

        This method should only be called once to "fill in" metadata
        that isn't originally known about an opened file. The provided `info`
        is used as a starting point, but is not modified by this method.

        """
        z = {}

        band_short_name = info.get(Info.DATASET_NAME, "???")
        # FIXME: Don't use pure DATASET_NAME since resolution should not be part of the SHORT_NAME
        #        And/or don't use SHORT_NAME for grouping
        if Info.SHORT_NAME not in info:
            z[Info.SHORT_NAME] = band_short_name
        else:
            z[Info.SHORT_NAME] = info[Info.SHORT_NAME]
        if Info.LONG_NAME not in info:
            z[Info.LONG_NAME] = info.get(Info.SHORT_NAME, z[Info.SHORT_NAME])

        z.setdefault(Info.STANDARD_NAME, info.get(Info.STANDARD_NAME, "unknown"))
        if info.get(Info.UNITS, z.get(Info.UNITS)) in ["K", "Kelvin"]:
            z[Info.UNITS] = "kelvin"

        return z

    def _is_refl(self, info):
        return info.get(Info.STANDARD_NAME) == "toa_bidirectional_reflectance"

    def _is_bt(self, info):
        return info.get(Info.STANDARD_NAME) in BT_STANDARD_NAMES

    def valid_range(self, info):
        from uwsift.workspace.utils.metadata_utils import get_default_climits

        configured_climits = get_default_climits(info)
        if configured_climits != INVALID_COLOR_LIMITS:
            valid_range = configured_climits
        elif self._is_refl(info):
            valid_range = (-0.012, 1.192)
            if info[Info.UNITS] == "%":
                # Reflectance/visible data limits
                valid_range = (valid_range[0] * 100.0, valid_range[1] * 100.0)
        elif self._is_bt(info):
            # BT data limits
            valid_range = (-109.0 + 273.15, 55 + 273.15)
        elif "valid_min" in info:
            valid_range = (info["valid_min"], info["valid_max"])
        elif "valid_range" in info:
            valid_range = tuple(info["valid_range"])
        elif "flag_values" in info:
            valid_range = (min(info["flag_values"]), max(info["flag_values"]))
        else:
            valid_range = None
        return valid_range

    def default_colormap(self, info):
        return DEFAULT_COLORMAPS.get(info.get(Info.STANDARD_NAME), DEFAULT_UNKNOWN)

    def _default_display_time(self, info):
        # FUTURE: This can be customized by the user
        when = info.get(Info.SCHED_TIME, info.get(Info.OBS_TIME))
        if when is None:
            dtime = "--:--:--"
        elif "model_time" in info:
            dtime = "{}Z +{}h".format(info["model_time"].strftime("%Y-%m-%d %H:%M"), when.strftime("%H"))
        else:
            dtime = when.strftime("%Y-%m-%d %H:%M:%S")
        return dtime

    def _default_display_name(self, info, display_time=None):
        # FUTURE: This can be customized by the user
        platform = info.get(Info.PLATFORM, "-unknown-")
        instrument = info.get(Info.INSTRUMENT, "-unknown-")
        short_name = info.get(Info.SHORT_NAME, "-unknown-")

        standard_name = info.get(Info.STANDARD_NAME, "")
        if standard_name == "toa_bidirectional_reflectance":
            label = " Refl"  # Don't remove the leading space
        elif standard_name == "toa_brightness_temperature":
            label = " BT"  # Don't remove the leading space
        else:
            label = ""

        if display_time is None:
            display_time = info.get(Info.DISPLAY_TIME, self._default_display_time(info))
        display_name = f"{platform.value} {instrument.value} {short_name}{label} {display_time}"
        return display_name
