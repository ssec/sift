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

import os, sys, re
from datetime import datetime
import logging, unittest, argparse
from enum import Enum
from cspov.common import INFO, KIND
from cspov.view.Colormap import DEFAULT_IR, DEFAULT_VIS

LOG = logging.getLogger(__name__)

class INSTRUMENT(Enum):
    UNKNOWN = '???'
    AHI = 'AHI'
    ABI = 'ABI'
    AMI = 'AMI'


class PLATFORM(Enum):
    HIMAWARI_8 = 'Himawari-8'
    HIMAWARI_9 = 'Himawari-9'
    GOES_16 = 'GOES-16'

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


class GUIDE(Enum):
    """
    standard dictionary keys for guidebook metadata
    """
    UUID = 'uuid'  # dataset UUID, if available
    PLATFORM = 'spacecraft' # full standard name of spacecraft
    SCHED_TIME = 'timeline'  # scheduled time for observation
    OBS_TIME = 'obstime'  # actual time for observation
    BAND = 'band'  # band number (multispectral instruments)
    SCENE = 'scene'  # standard scene identifier string for instrument, e.g. FLDK
    INSTRUMENT = 'instrument'  # INSTRUMENT enumeration, or string with full standard name
    DISPLAY_TIME = 'display_time' # time to show on animation control
    DISPLAY_NAME = 'display_name' # preferred name in the layer list
    UNIT_CONVERSION = 'unit_conversion'  # (unit string, lambda x, inverse=False: convert-to-units)
    CENTRAL_WAVELENGTH = 'nominal_wavelength'

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
    _cache = None  # {uuid:metadata-dictionary, ...}

    REFL_BANDS = [1, 2, 3, 4, 5, 6]
    BT_BANDS = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

    def __init__(self):
        self._cache = {}

    @staticmethod
    def is_relevant(pathname):
        if not pathname:
            return False
        return True if re.match(r'HS_H\d\d_\d{8}_\d{4}_B\d\d.*', os.path.split(pathname)[1]) else False

    @staticmethod
    def _metadata_for_path(pathname):
        if not pathname:
            return {}
        m = re.match(r'HS_H(\d\d)_(\d{8})_(\d{4})_B(\d\d)_([A-Za-z0-9]+).*', os.path.split(pathname)[1])
        if not m:
            return {}
        plat, yyyymmdd, hhmm, bb, scene = m.groups()
        when = datetime.strptime(yyyymmdd + hhmm, '%Y%m%d%H%M')
        plat = PLATFORM('Himawari-{}'.format(int(plat)))
        band = int(bb)
        dtime = when.strftime('%Y-%m-%d %H:%M')
        label = 'Refl' if band in [1, 2, 3, 4, 5, 6] else 'BT'
        name = "AHI B{0:02d} {1:s} {2:s}".format(band, label, dtime)
        return {
            GUIDE.PLATFORM: plat,
            GUIDE.BAND: band,
            GUIDE.SCHED_TIME: when,
            GUIDE.DISPLAY_TIME: dtime,
            GUIDE.SCENE: scene,
            GUIDE.DISPLAY_NAME: name
        }

    def _relevant_info(self, seq):
        "filter datasetinfo dictionaries in sequence, if they're not relevant to us (i.e. not AHI)"
        for dsi in seq:
            if self.is_relevant(dsi.get(INFO.PATHNAME, None)):
                yield dsi

    def sort_pathnames_into_load_order(self, paths):
        """
        given a list of paths, sort them into order, assuming layers are added at top of layer list
        first: unknown paths
        outer order: descending band number
        inner order: descending time step
        :param paths: list of path strings
        :return: ordered list of path strings
        """
        nfo = dict((path, self._metadata_for_path(path)) for path in paths)
        riffraff = [path for path in paths if not nfo[path]]
        ahi = [nfo[path] for path in paths if nfo[path]]
        names = [path for path in paths if nfo[path]]
        bands = [x.get(GUIDE.BAND, None) for x in ahi]
        times = [x.get(GUIDE.SCHED_TIME, None) for x in ahi]
        order = [(band, time, path) for band,time,path in zip(bands,times,names)]
        order.sort(reverse=True)
        LOG.debug(order)
        return riffraff + [path for band,time,path in order]

    def collect_info(self, info):
        md = self._cache.get(info[INFO.UUID], None)
        if md is None and info[INFO.KIND] not in (KIND.IMAGE, KIND.COMPOSITE):
            md = self._metadata_for_path(info.get(INFO.PATHNAME, None))
            md[GUIDE.UUID] = info[INFO.UUID]
            md[GUIDE.INSTRUMENT] = INSTRUMENT.AHI
            md[GUIDE.CENTRAL_WAVELENGTH] = NOMINAL_WAVELENGTHS[md[GUIDE.PLATFORM]][md[GUIDE.INSTRUMENT]][md[GUIDE.BAND]]
            # md[GUIDE.UNIT_CONVERSION] = self.units_conversion(info)  # FUTURE: decide whether this should be done for most queries
            self._cache[info[INFO.UUID]] = md
        if md is None:
            return dict(info)
        z = md.copy()
        z.update(info)
        return z

    def collect_info_from_seq(self, seq):
        "collect AHI metadata about a sequence of datasetinfo dictionaries"
        # FUTURE: cache uuid:metadata info in the guidebook instance for quick lookup
        for each in self._relevant_info(seq):
            md = self.collect_info(each)
            yield each[INFO.UUID], md

    def climits(self, dsi):
        # Valid min and max for colormap use for data values in file (unconverted)
        md = self.collect_info(dsi)
        if md[GUIDE.BAND] in self.REFL_BANDS:
            # Reflectance/visible data limits
            return -0.012, 1.192
        elif md[GUIDE.BAND] in self.BT_BANDS:
            # BT data limits
            return -109.0 + 273.15, 55 + 273.15
        else:
            return None

    def units_conversion(self, dsi):
        "return UTF8 unit string, lambda v,inverse=False: convert-raw-data-to-unis"
        md = self.collect_info(dsi)
        if md[GUIDE.BAND] in self.REFL_BANDS:
            # Reflectance/visible data limits
            return ('{:.03f}', '', lambda x, inverse=False: x)
        elif md[GUIDE.BAND] in self.BT_BANDS:
            # BT data limits, Kelvin to degC
            return ("{:.02f}", "°C", lambda x, inverse=False: x - 273.15 if not inverse else x + 273.15)
        else:
            return ("", "", lambda x, inverse=False: 0.0)

    def default_colormap(self, dsi):
        md = self.collect_info(dsi)
        if md[GUIDE.BAND] in self.REFL_BANDS:
            return DEFAULT_VIS
        elif md[GUIDE.BAND] in self.BT_BANDS:
            return DEFAULT_IR
        else:
            return None

    def display_time(self, dsi):
        md = self.collect_info(dsi)
        return md.get(GUIDE.DISPLAY_TIME, '--:--')

    def display_name(self, dsi):
        md = self.collect_info(dsi)
        return md.get(GUIDE.DISPLAY_NAME, '-unknown-')

    def flush(self):
        self._cache = {}

    def _filter(self, seq, reference, keys):
        "filter a sequence of metadata dictionaries to matching keys with reference"
        for md in seq:
            fail = False
            for key in keys:
                val = reference.get(key, None)
                v = md.get(key, None)
                if val != v:
                    fail=True
            if not fail:
                yield md

    def channel_siblings(self, uuid, infos):
        """
        filter document info to just dataset of the same channels
        :param uuid:
        :param infos:
        :return: sorted list of sibling uuids in channel order
        """
        meta = dict(self.collect_info_from_seq(infos))
        it = meta.get(uuid, None)
        if it is None:
            return None
        sibs = [(x[GUIDE.BAND], x[GUIDE.UUID]) for x in
                self._filter(meta.values(), it, {GUIDE.SCENE, GUIDE.SCHED_TIME, GUIDE.INSTRUMENT, GUIDE.PLATFORM})]
        # then sort it by bands
        sibs.sort()
        offset = [i for i,x in enumerate(sibs) if x[1]==uuid]
        return [x[1] for x in sibs], offset[0]

    def time_siblings(self, uuid, infos):
        """
        return time-ordered list of datasets which have the same band, in time order
        :param uuid: focus UUID we're trying to build around
        :param infos: list of dataset infos available, some of which may not be relevant
        :return: sorted list of sibling uuids in time order, index of where uuid is in the list
        """
        meta = dict(self.collect_info_from_seq(infos))
        it = meta.get(uuid, None)
        if it is None:
            return None
        sibs = [(x[GUIDE.SCHED_TIME], x[GUIDE.UUID]) for x in
                self._filter(meta.values(), it, {GUIDE.SCENE, GUIDE.BAND, GUIDE.INSTRUMENT, GUIDE.PLATFORM})]
        # then sort it into time order
        sibs.sort()
        offset = [i for i,x in enumerate(sibs) if x[1]==uuid]
        return [x[1] for x in sibs], offset[0]

    def time_siblings_uuids(self, uuids, infos):
        """
        return generator uuids for datasets which have the same band as the uuids provided
        :param uuids: iterable of uuids
        :param infos: list of dataset infos available, some of which may not be relevant
        :return: generate sorted list of sibling uuids in time order and in provided uuid order
        """
        for requested_uuid in uuids:
            for sibling_uuid in self.time_siblings(requested_uuid, infos)[0]:
                yield sibling_uuid



# if __name__ == '__main__':
#     sys.exit(main())
