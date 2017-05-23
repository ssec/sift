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

import os
import re
import logging
from datetime import datetime
from sift.common import INFO, KIND, PLATFORM, INSTRUMENT
from sift.view.Colormap import DEFAULT_IR, DEFAULT_VIS
import sift.workspace.goesr_pug as pug

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
    PLATFORM.GOES_16: {
        INSTRUMENT.ABI: {  # http://www.goes-r.gov/education/ABI-bands-quick-info.html
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
    },
}

# map .platform_id in PUG format files to SIFT platform enum
PLATFORM_ID_TO_PLATFORM = {
    'G16': PLATFORM.GOES_16,
    'G17': PLATFORM.GOES_17,
    # hsd2nc export of AHI data as PUG format
    'Himawari-8': PLATFORM.HIMAWARI_8,
    'Himawari-9': PLATFORM.HIMAWARI_9
}


class ABI_AHI_Guidebook(Guidebook):
    "e.g. HS_H08_20150714_0030_B10_FLDK_R20.merc.tif"
    _cache = None  # {uuid:metadata-dictionary, ...}

    REFL_BANDS = [1, 2, 3, 4, 5, 6]
    BT_BANDS = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

    def __init__(self):
        self._cache = {}

    @staticmethod
    def identify_instrument_for_path(pathname):
        if not pathname:
            return None
        if pathname.endswith('.nc') or pathname.endswith('.nc4'):  # FUTURE: tighten this constraint to not assume ABI
            return INSTRUMENT.ABI
        elif re.match(r'HS_H\d\d_\d{8}_\d{4}_B\d\d.*', os.path.split(pathname)[1]):
            return INSTRUMENT.AHI
        return None

    @staticmethod
    def is_relevant(pathname):
        return ABI_AHI_Guidebook.identify_instrument_for_path(pathname) in {INSTRUMENT.ABI, INSTRUMENT.AHI}

    @staticmethod
    def _metadata_for_ahi_path(pathname):
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
            INFO.PLATFORM: plat,
            INFO.BAND: band,
            INFO.INSTRUMENT: INSTRUMENT.AHI,
            INFO.SCHED_TIME: when,
            INFO.DISPLAY_TIME: dtime,
            INFO.SCENE: scene,
            INFO.DISPLAY_NAME: name
        }

    @staticmethod
    def _metadata_for_abi_path(pathname):  # FUTURE: since for ABI this is really coming from the file, decide whether the guidebook should be doing it
        abi = pug.PugL1bTools(pathname)
        return {
            INFO.PLATFORM: PLATFORM_ID_TO_PLATFORM[abi.platform],  # e.g. G16
            INFO.BAND: abi.band,
            INFO.INSTRUMENT: INSTRUMENT.ABI,
            INFO.SCHED_TIME: abi.sched_time,
            INFO.DISPLAY_TIME: abi.display_time,
            INFO.SCENE: abi.scene_id,
            INFO.DISPLAY_NAME: abi.display_name
        }

    @staticmethod
    def _metadata_for_path(pathname):
        which = ABI_AHI_Guidebook.identify_instrument_for_path(pathname)
        try:
            return {INSTRUMENT.ABI: ABI_AHI_Guidebook._metadata_for_abi_path,
                    INSTRUMENT.AHI: ABI_AHI_Guidebook._metadata_for_ahi_path}[which](pathname)
        except (OSError, ValueError, KeyError):
            LOG.error("Unable to get metadata for path '{}'".format(pathname))
            return None

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
        bands = [x.get(INFO.BAND, None) for x in ahi]
        times = [x.get(INFO.SCHED_TIME, None) for x in ahi]
        order = [(band, time, path) for band,time,path in zip(bands,times,names)]
        order.sort(reverse=True)
        LOG.debug(order)
        return riffraff + [path for band,time,path in order]

    def collect_info(self, info):
        md = self._cache.get(info[INFO.UUID], None)
        # TODO: Used to be 'info *not* in (KIND.IMAGE, KIND.COMPOSITE)', what's right?
        if md is None and info[INFO.KIND] in (KIND.IMAGE, KIND.COMPOSITE):
            md = self._metadata_for_path(info.get(INFO.PATHNAME, None))
            md[INFO.UUID] = info[INFO.UUID]
            md[INFO.INSTRUMENT] = info.get(INFO.INSTRUMENT, self.identify_instrument_for_path(info[INFO.PATHNAME]))
            md[INFO.CENTRAL_WAVELENGTH] = NOMINAL_WAVELENGTHS[md[INFO.PLATFORM]][md[INFO.INSTRUMENT]][md[INFO.BAND]]
            # md[INFO.UNIT_CONVERSION] = self.units_conversion(info)  # FUTURE: decide whether this should be done for most queries
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
        if md[INFO.BAND] in self.REFL_BANDS:
            # Reflectance/visible data limits
            return -0.012, 1.192
        elif md[INFO.BAND] in self.BT_BANDS:
            # BT data limits
            return -109.0 + 273.15, 55 + 273.15
        else:
            return None

    def units_conversion(self, dsi):
        "return UTF8 unit string, lambda v,inverse=False: convert-raw-data-to-unis"
        md = self.collect_info(dsi)
        if md[INFO.BAND] in self.REFL_BANDS:
            # Reflectance/visible data limits
            return ('{:.03f}', '', lambda x, inverse=False: x)
        elif md[INFO.BAND] in self.BT_BANDS:
            # BT data limits, Kelvin to degC
            return ("{:.02f}", "°C", lambda x, inverse=False: x - 273.15 if not inverse else x + 273.15)
        else:
            return ("", "", lambda x, inverse=False: 0.0)

    def default_colormap(self, dsi):
        md = self.collect_info(dsi)
        if md[INFO.BAND] in self.REFL_BANDS:
            return DEFAULT_VIS
        elif md[INFO.BAND] in self.BT_BANDS:
            return DEFAULT_IR
        else:
            return None

    def display_time(self, dsi):
        md = self.collect_info(dsi)
        return md.get(INFO.DISPLAY_TIME, '--:--')

    def display_name(self, dsi):
        md = self.collect_info(dsi)
        return md.get(INFO.DISPLAY_NAME, '-unknown-')

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
        sibs = [(x[INFO.BAND], x[INFO.UUID]) for x in
                self._filter(meta.values(), it, {INFO.SCENE, INFO.SCHED_TIME, INFO.INSTRUMENT, INFO.PLATFORM})]
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
            return [], 0
        sibs = [(x[INFO.SCHED_TIME], x[INFO.UUID]) for x in
                self._filter(meta.values(), it, {INFO.SCENE, INFO.BAND, INFO.INSTRUMENT, INFO.PLATFORM})]
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
