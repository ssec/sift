#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
uwsift.model.area_definitions_manager
----------------------------

Manage area definitions from Satpy/Pyresample.

:author: A.Rettig <alexander.rettig@askvisual.de>
"""

__docformat__ = 'reStructuredText'

import logging
import typing as typ

from pyresample.area_config import AreaNotFound
from pyresample.geometry import AreaDefinition
from satpy.resample import get_area_def

from uwsift import config

LOG = logging.getLogger(__name__)


class AreaDefinitionsManager:
    """
    Manage Pyresample AreaDefinitions, i.e., check their availability in
    accessible configuration and provide them based on their *display name*. The
    *display name* is the one to be used in the GUI, it should be less abstract
    than the *area_id* and shorter than the *description* provided in the
    AreaDefinition object.
    """
    _available_area_defs = None

    @classmethod
    def init_available_area_defs(cls):

        cls._available_area_defs: typ.Dict[str, AreaDefinition] = {}
        desired_area_defs = config.get("area_definitions", {})

        for area_def_name, area_id in desired_area_defs.items():
            try:
                area_def = get_area_def(area_id)
            except AreaNotFound as e:
                LOG.warning(f"Area definition configured for display name"
                            f" '{area_def_name}' unknown: {e}. Skipping...")
                continue

            LOG.info(f"Adding area definition: {area_def_name} -> {area_id}")
            cls._available_area_defs[area_def_name] = area_def

    @classmethod
    def available_area_def_names(cls):
        return cls._available_area_defs.keys()

    @classmethod
    def area_def_by_name(cls, name):
        return cls._available_area_defs.get(name)

    @classmethod
    def default_area_def_name(cls):  # TODO: take from configuration, make robust
        # Since nothing has been configured, take the first key (i.e. the
        # display_name) of the known area definitions
        return next(iter(cls._available_area_defs))


AreaDefinitionsManager.init_available_area_defs()
