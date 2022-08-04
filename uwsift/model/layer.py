#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
layer.py
~~~~~~~~~

PURPOSE
Document Layer & LayerDoc

REFERENCES


REQUIRES


:author: R.K.Garcia <rayg@ssec.wisc.edu>
:copyright: 2014 by University of Wisconsin Regents, see AUTHORS for more details
:license: GPLv3, see LICENSE for more details
"""
from collections import ChainMap
from enum import Enum

from _weakref import ref

from uwsift.common import Info, Kind

__author__ = "rayg"
__docformat__ = "reStructuredText"

import logging

import numpy as np

LOG = logging.getLogger(__name__)


class Mixing(Enum):
    UNKNOWN = 0
    NORMAL = 1
    ADD = 2
    SUBTRACT = 3


class DocDataset(ChainMap):
    """Container for layer metadata

    Use dictionary-like access for metadata information.
    """

    _doc = None  # weakref to document that owns us

    def __init__(self, doc, info, *args, **kwargs):
        self._doc = ref(doc)
        # store of metadata that comes from the input content and does not change
        # FIXME: review whether we can get by without copying this (is it harmless to do so?)
        self._definitive = info  # definitive information provided by workspace, read-only from our perspective
        self._additional = dict(*args, **kwargs)  # FUTURE: can we deprecate this?
        self._user_modified = {}
        super(DocDataset, self).__init__(self._user_modified, self._additional, self._definitive)

    @property
    def product_family_key(self):
        """Unique key for this layer and its group of siblings"""
        return self.platform, self.instrument, self.dataset_name

    @property
    def parent(self):
        """
        parent layer, if any
        :return:
        """
        return None

    @property
    def children(self):
        """
        return dictionary of weakrefs to layers we require in order to function
        :return:
        """
        return {}

    @property
    def uuid(self):
        """
        UUID of the layer, which for basic layers is likely to be the UUID of the dataset in the workspace.
        :return:
        """
        return self[Info.UUID]

    @property
    def kind(self):
        """
        which kind of layer it is - RGB, Algebraic, etc. This can also be tested by the class of the layer typically.
        We may deprecate this eventually?
        :return:
        """
        return self[Info.KIND]

    @property
    def instrument(self):
        return self.get(Info.INSTRUMENT)

    @property
    def platform(self):
        return self.get(Info.PLATFORM)

    @property
    def sched_time(self):
        return self.get(Info.SCHED_TIME)

    @property
    def dataset_name(self):
        return self[Info.DATASET_NAME]

    @property
    def display_name(self):
        return self[Info.DISPLAY_NAME]

    @property
    def default_display_name(self):
        for d in reversed(self.maps):
            if Info.DISPLAY_NAME in d:
                return d[Info.DISPLAY_NAME]
        return None


class DocBasicDataset(DocDataset):
    """
    A layer consistent of a simple scalar floating point value field, which can have color maps applied
    """

    pass


class DocCompositeDataset(DocDataset):
    """
    A layer which combines other layers, be they basic or composite themselves
    """


def _concurring(*q, remove_none=False):
    """Check that all provided inputs are the same"""
    if remove_none:
        q = [x for x in q if x is not None]
    if q:
        return q[0] if all(x == q[0] for x in q[1:]) else False
    else:
        return False


class DocRGBDataset(DocCompositeDataset):
    def __init__(self, doc, recipe, info, *args, **kwargs):
        self.layers = [None, None, None, None]  # RGBA upstream layers
        self.mins = [None, None, None, None]  # RGBA minimum value from upstream layers
        self.maxs = [None, None, None, None]  # RGBA maximum value from upstream layers
        self.recipe = recipe
        info.setdefault(Info.KIND, Kind.RGB)
        super().__init__(doc, info, *args, **kwargs)

    @property
    def r(self):
        return self.layers[0]

    @property
    def g(self):
        return self.layers[1]

    @property
    def b(self):
        return self.layers[2]

    @property
    def a(self):
        return self.layers[3]

    @r.setter
    def r(self, x):
        self.layers[0] = x

    @g.setter
    def g(self, x):
        self.layers[1] = x

    @b.setter
    def b(self, x):
        self.layers[2] = x

    @a.setter
    def a(self, x):
        self.layers[3] = x

    def _get_if_not_none_func(self, attr=None, item=None):
        def _get_not_none_item(layer):
            return None if layer is None else layer.get(item)

        def _get_not_none_attr(layer):
            return None if layer is None else getattr(layer, attr)

        if attr is not None:
            return _get_not_none_attr
        return _get_not_none_item

    def dep_info(self, key, include_alpha=False):
        max_idx = 4 if include_alpha else 3
        return [x.get(key) for x in self.layers[:max_idx]]

    def product_family_keys(self, include_alpha=False):
        max_idx = 4 if include_alpha else 3
        gb = self._get_if_not_none_func(attr="product_family_key")
        return [gb(x) for x in self.layers[:max_idx]]

    @property
    def has_deps(self):
        return self.r is not None or self.g is not None or self.b is not None

    @property
    def shared_projections(self):
        return all(x[Info.PROJ] == self[Info.PROJ] for x in self.layers[:3] if x is not None)

    @property
    def shared_origin(self):
        if all(x is None for x in self.layers[:3]):
            return False

        atol = max(abs(x[Info.CELL_WIDTH]) for x in self.layers[:3] if x is not None)
        shared_x = all(
            np.isclose(x[Info.ORIGIN_X], self[Info.ORIGIN_X], atol=atol) for x in self.layers[:3] if x is not None
        )

        atol = max(abs(x[Info.CELL_HEIGHT]) for x in self.layers[:3] if x is not None)
        shared_y = all(
            np.isclose(x[Info.ORIGIN_Y], self[Info.ORIGIN_Y], atol=atol) for x in self.layers[:3] if x is not None
        )
        return shared_x and shared_y

    @property
    def recipe_layers_match(self):
        def _get_family(layer):
            return layer[Info.FAMILY] if layer else None

        return all([_get_family(x) == self.recipe.input_layer_ids[idx] for idx, x in enumerate(self.layers[:3])])

    @property
    def is_valid(self):
        return self.has_deps and self.shared_projections and self.shared_origin and self.recipe_layers_match

    @property
    def sched_time(self):
        gst = self._get_if_not_none_func(attr="sched_time")
        return _concurring(gst(self.r), gst(self.g), gst(self.b), remove_none=True)

    @property
    def instrument(self):
        gst = self._get_if_not_none_func(attr="instrument")
        return _concurring(gst(self.r), gst(self.g), gst(self.b), remove_none=True)

    @property
    def platform(self):
        gst = self._get_if_not_none_func(attr="platform")
        return _concurring(gst(self.r), gst(self.g), gst(self.b), remove_none=True)

    @property
    def scene(self):
        gst = self._get_if_not_none_func(item=Info.SCENE)
        return _concurring(gst(self.r), gst(self.g), gst(self.b), remove_none=True)

    def _default_display_time(self):
        dep_info = [self.r, self.g, self.b]
        valid_times = [nfo.get(Info.SCHED_TIME, None) for nfo in dep_info if nfo is not None]
        valid_times = [x.strftime("%Y-%m-%d %H:%M:%S") if x is not None else "<unknown time>" for x in valid_times]
        if len(valid_times) == 0:
            display_time = "<unknown time>"
        else:
            display_time = (
                valid_times[0]
                if len(valid_times) and all(t == valid_times[0] for t in valid_times[1:])
                else "<multiple times>"
            )

        return display_time

    def _default_short_name(self):
        dep_info = [self.r, self.g, self.b]
        try:
            names = []
            for color, dep_layer in zip("RGB", dep_info):
                if dep_layer is None:
                    name = "{}:---".format(color)
                else:
                    name = "{}:{}".format(color, dep_layer[Info.SHORT_NAME])
                names.append(name)
            name = " ".join(names)
        except KeyError:
            LOG.error("unable to create new name from {0!r:s}".format(dep_info))
            name = "-RGB-"

        return name

    def _default_display_name(self, short_name=None, display_time=None):
        if display_time is None:
            display_time = self._default_display_time()
        if short_name is None:
            short_name = self._default_short_name()

        return short_name + " " + display_time
