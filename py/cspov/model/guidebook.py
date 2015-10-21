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

LOG = logging.getLogger(__name__)

class INSTRUMENT(Enum):
    UNKNOWN = '???'
    AHI = 'AHI'
    ABI = 'ABI'
    AMI = 'AMI'

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
            path = info[INFO.PATHNAME]

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
    SPACECRAFT = 'spacecraft' # full standard name of spacecraft
    SCHED_TIME = 'timeline'  # scheduled time for observation
    OBS_TIME = 'obstime'  # actual time for observation
    BAND = 'band'  # band number (multispectral instruments)
    SCENE = 'scene'  # standard scene identifier string for instrument, e.g. FLDK
    INSTRUMENT = 'instrument'  # INSTRUMENT enumeration, or string with full standard name
    DISPLAY_TIME = 'display_time' # time to show on animation control


class AHI_HSF_Guidebook(Guidebook):
    "e.g. HS_H08_20150714_0030_B10_FLDK_R20.merc.tif"
    _cache = None  # {uuid:metadata-dictionary, ...}

    def __init__(self):
        self._cache = {}

    @staticmethod
    def is_relevant(pathname):
        return True if re.match(r'HS_H\d\d_\d{8}_\d{4}_B\d\d.*', os.path.split(pathname)[1]) else False

    @staticmethod
    def metadata_for_path(pathname):
        m = re.match(r'HS_H(\d\d)_(\d{8})_(\d{4})_B(\d\d)_([A-Za-z0-9]+).*', os.path.split(pathname)[1])
        if not m:
            return {}
        sat, yyyymmdd, hhmm, bb, scene = m.groups()
        when = datetime.strptime(yyyymmdd + hhmm, '%Y%m%d%H%M')
        sat = 'Himawari-{}'.format(int(sat))
        band = int(bb)
        dtime = when.strftime('%Y-%m-%d %H:%M')
        return {
            GUIDE.SPACECRAFT: sat,
            GUIDE.BAND: band,
            GUIDE.SCHED_TIME: when,
            GUIDE.DISPLAY_TIME: dtime,
            GUIDE.SCENE: scene
        }

    def _relevant_info(self, seq):
        "filter datasetinfo dictionaries in sequence, if they're not relevant to us (i.e. not AHI)"
        for dsi in seq:
            if self.is_relevant(dsi[INFO.PATHNAME]):
                yield dsi

    def _collect_info(self, seq):
        "collect AHI metadata about a sequence of datasetinfo dictionaries"
        # FUTURE: cache uuid:metadata info in the guidebook instance for quick lookup
        for each in self._relevant_info(seq):
            md = self._cache.get(each[INFO.UUID], None)
            if md is not None:
                yield each[INFO.UUID], md
            else:
                md = self.metadata_for_path(each[INFO.PATHNAME])
                md[GUIDE.UUID] = each[INFO.UUID]
                md[GUIDE.INSTRUMENT] = INSTRUMENT.AHI
                self._cache[each[INFO.UUID]] = md
                yield each[INFO.UUID], md

    def display_time(self, dsi):
        nfo, = self._collect_info([dsi])
        uuid, md = nfo
        return nfo[GUIDE.DISPLAY_TIME]

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
        meta = dict(self._collect_info(infos))
        it = meta.get(uuid, None)
        if it is None:
            return None
        sibs = [(x[GUIDE.BAND], x[GUIDE.UUID]) for x in
                self._filter(meta.values(), it, {GUIDE.SCENE, GUIDE.SCHED_TIME, GUIDE.INSTRUMENT, GUIDE.SPACECRAFT})]
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
        meta = dict(self._collect_info(infos))
        it = meta.get(uuid, None)
        if it is None:
            return None
        sibs = [(x[GUIDE.SCHED_TIME], x[GUIDE.UUID]) for x in
                self._filter(meta.values(), it, {GUIDE.SCENE, GUIDE.BAND, GUIDE.INSTRUMENT, GUIDE.SPACECRAFT})]
        # then sort it into time order
        sibs.sort()
        offset = [i for i,x in enumerate(sibs) if x[1]==uuid]
        return [x[1] for x in sibs], offset[0]



# if __name__ == '__main__':
#     sys.exit(main())
