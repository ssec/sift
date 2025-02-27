#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
uwsift.model.area_definitions_manager
-------------------------------------

Manage area definitions from Satpy/Pyresample.

:author: A.Rettig <alexander.rettig@askvisual.de>
"""
from __future__ import annotations

import logging

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
    "Plate Carree": {"plate_carree": "plate_carree"}
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

    _available_area_defs_by_id: None | dict[str, AreaDefinition] = None
    _available_area_defs_group_by_group_name: None | dict[str, dict[str, str]] = (
        None  # changed | group_name : { name : id name : id}
    )

    @classmethod
    def init_available_area_defs(cls) -> None:
        cls._available_area_defs_by_id = {}
        cls._available_area_defs_group_by_group_name = {}

        desired_area_defs = config.get("area_definitions", {})

        for area_def_group_name, area_def_group in desired_area_defs.items():

            for area_resolution, area_id in area_def_group.items():
                try:
                    area_def = get_area_def(area_id)
                except AreaNotFound as e:
                    LOG.warning(
                        f"Area definition configured for display name" f" '{area_id}' unknown: {e}. Skipping..."
                    )
                    continue
                # adding ar
                LOG.info(f"Adding area definition: {area_def_group_name} , {area_resolution} -> {area_id}")
                cls._available_area_defs_by_id[area_id] = area_def

            cls._available_area_defs_group_by_group_name[area_def_group_name] = area_def_group

        # cls._available_area_defs_group_by_group_name and cls._available_area_defs_by_id are initialised

        # Check for existence of at least one 'latlong' projection
        # (https://proj.org/operations/conversions/latlon.html)
        for area_def in cls._available_area_defs_by_id.values():
            if area_def.crs.is_geographic:
                return

        # Add default area definition(s)?
        for area_def_group_name, area_def_group in DEFAULT_AREAS.items():
            for area_resolution, area_id in area_def_group.items():
                area_def = load_area_from_string(DEFAULT_AREA_DEFINITIONS_YAML, area_id)

                LOG.info(f"Adding area definition: {area_def_group_name} , {area_resolution} -> {area_id}")
                cls._available_area_defs_by_id[area_id] = area_def

            cls._available_area_defs_group_by_group_name[area_def_group_name] = area_def_group
        #

    # returns all area projections
    @classmethod
    def available_area_def_names(cls):
        return cls._available_area_defs_group_by_group_name.keys()

    @classmethod
    def area_def_by_id(cls, id):
        return cls._available_area_defs_by_id.get(id)

    # returns the first area definition from the group
    @classmethod
    def area_def_by_name(cls, name):
        area_group = cls._available_area_defs_group_by_group_name.get(name)
        first_key = next(iter(area_group))
        area_id = area_group[first_key]
        return cls.area_def_by_id(area_id)

    @classmethod
    def default_area_def_name(cls):  # TODO: take from configuration, make robust
        # Since nothing has been configured, take the first key (i.e. the
        # display_name) of the known area definitions
        return next(iter(cls._available_area_defs_group_by_group_name))

    # returns all possible resolutions for one area projection
    @classmethod
    def available_area_def_group_resolutions(cls, group_name):
        return cls._available_area_defs_group_by_group_name.get(group_name).keys()

    # returns area group by its name --- for example if a group_name = MSG SEVIRI FES, a return value will be
    # {3 km: msg_seviri_fes_3km, 1 km: msg_seviri_fes_1km}
    @classmethod
    def area_group_by_group_name(cls, group_name):
        return cls._available_area_defs_group_by_group_name.get(group_name)

    # returns area definition by its name and resolution --- for example if a group_name = MSG SEVIRI FES and
    # resolution = 3km , a return value will be area definition for id = msg_seviri_fes_3km
    @classmethod
    def area_def_by_group_name_and_resolution(cls, group_name, resolution):
        return cls.area_def_by_id(cls._available_area_defs_group_by_group_name.get(group_name).get(resolution))

    # prepares area def for resampling
    @classmethod
    def prepare_area_def_for_resampling(cls, area_def, width, height):
        # when the shape is changed, it also affects other parameters in the area definition
        area_def.width = width
        area_def.height = height
        area_def.pixel_size_x = (area_def.area_extent[2] - area_def.area_extent[0]) / float(area_def.width)
        area_def.pixel_size_y = (area_def.area_extent[3] - area_def.area_extent[1]) / float(area_def.height)
        area_def.pixel_upper_left = (
            float(area_def.area_extent[0]) + float(area_def.pixel_size_x) / 2,
            float(area_def.area_extent[3]) - float(area_def.pixel_size_y) / 2,
        )
        area_def.pixel_offset_x = -area_def.area_extent[0] / area_def.pixel_size_x
        area_def.pixel_offset_y = area_def.area_extent[3] / area_def.pixel_size_y

    # calculates pixel_size_x and pixel_size_y for an area definition with custom resolution values
    @classmethod
    def area_def_custom_resolution_values(cls, area_def, width, height):
        if width and height:
            pixel_size_x = (area_def.area_extent[2] - area_def.area_extent[0]) / float(width)
            pixel_size_y = (area_def.area_extent[3] - area_def.area_extent[1]) / float(height)
            return pixel_size_x, pixel_size_y


# TODO: Why does this need to be a class being updated instead of an instance of a class?
AreaDefinitionsManager.init_available_area_defs()
