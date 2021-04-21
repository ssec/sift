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

from pyresample.area_config import AreaNotFound, load_area_from_string
from pyresample.geometry import AreaDefinition
from satpy.resample import get_area_def

from uwsift import config

LOG = logging.getLogger(__name__)

DEFAULT_AREAS = {
    # A plate carree Projection (aka geographic projection, lat/lon projection,
    # plane chart) should be available always. It will be only added though,
    # if there is no other area definition found in configuration which uses the
    # 'latlong' projection (or its PROJ.4 aliases)
    'Plate Carree': 'plate_carree'
}

DEFAULT_AREA_DEFINITIONS_YAML = """
plate_carree:
  description:
    Plate Carree with 0.1 degree resolution
  projection:
    proj: longlat
    ellps: WGS84
  shape:
    height: 1800
    width: 3600
  area_extent:
    lower_left_xy: [-180.0, -90.0]
    upper_right_xy: [180.0, 90.0]
"""


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

        # Check for existence of at least one 'latlong' projection
        # (https://proj.org/operations/conversions/latlon.html)
        for area_def in cls._available_area_defs.values():
            proj: str = area_def.proj_dict["proj"]
            if proj in ("latlon", "latlong", "lonlat", "longlat"):
                print(f"FOUND {proj}")
                return

        # Add default area definition(s)?
        for area_def_name, area_id in DEFAULT_AREAS.items():
            area_def = \
                load_area_from_string(DEFAULT_AREA_DEFINITIONS_YAML, area_id)

            LOG.info(f"Adding default area definition:"
                     f" {area_def_name} -> {area_id}")
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
