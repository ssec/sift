#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
sift.model.document
--------------------

Core (low-level) document model for SIFT.
The core is sometimes accessed via Facets, which are like database views for a specific group of use cases

The document model is a metadata representation which permits the workspace to be constructed and managed.

Document is primarily a composition of layers.
Layers come in several flavors:

 - Image : a float32 field shown as tiles containing strides and/or alternate LODs, having a colormap
 - Outline : typically a geographic political map
 - Shape : a highlighted region selected by the user: point, line (great circle), polygon
 - Combination : calculated from two or more image layers, e.g. RGBA combination of images
                 combinations may be limited to areas specified by region layers.

Future Work:

 - Volume : a 3D dataset, potentially sparse
 - DenseVolume
 - SparseVolume : (x,y,z) point cloud

Layers are represented in 1 or more LayerSets, which are alternate configurations of the display.
Users may wish to display the same data in several different ways for illustration purposes.
Only one LayerSet is used on a given Map window at a time.

Layers have presentation settings that vary with LayerSet:

 - z_order: bottom to top in the map display
 - visible: whether or not it's being drawn on the map
 - a_order: animation order, when the animation button is hit
 - colormap: how the data is converted to pixels
 - mixing: mixing mode when drawing (normal, additive)

Document has zero or more Probes. Layers also come in multiple
flavors that may be be attached to plugins or helper applications.

 - Scatter: (layerA, layerB, region) -> xy plot
 - Slice: (volume, line) -> curtain plot
 - Profile: (volume, point) -> profile plot

Document has zero or more Colormaps, determining how they're presented

The document does not own data (content). It only owns metadata (info).
At most, document holds coarse overview data content for preview purposes.

All entities in the Document have a UUID that is their identity throughout their lifecycle, and is often used as shorthand
between subsystems. Document rarely deals directly with content.

:author: R.K.Garcia <rayg@ssec.wisc.edu>
:copyright: 2015 by University of Wisconsin Regents, see AUTHORS for more details
:license: GPLv3, see LICENSE for more details
"""
from sift.model.layer import Mixing, DocLayer, DocBasicLayer, DocRGBLayer,DocCompositeLayer

__author__ = 'rayg'
__docformat__ = 'reStructuredText'

import sys
import logging
import unittest
import argparse
from collections import MutableSequence, OrderedDict, defaultdict
from itertools import groupby, chain
from uuid import UUID, uuid1 as uuidgen
from datetime import datetime, timedelta
import typing as T
import enum
import numpy as np
from weakref import ref
import os
import json

from sift.workspace.metadatabase import Product
from sift.common import KIND, INFO, prez, span, FCS_SEP, ZList, flags
from sift.util.default_paths import DOCUMENT_SETTINGS_DIR
from sift.model.composite_recipes import RecipeManager, CompositeRecipe
from sift.view.Colormap import ALL_COLORMAPS, USER_COLORMAPS
from PyQt4.QtCore import QObject, pyqtSignal

from colormap import rgb2hex
from vispy.color.colormap import Colormap


LOG = logging.getLogger(__name__)

DEFAULT_LAYER_SET_COUNT = 1  # this should match the ui configuration!


def unit_symbol(unit):
    # FUTURE: Use cfunits or cf_units package
    # cf_units gives the wrong symbol for celsius
    if unit == '1':
        return ''
    elif unit == 'degrees_Celsius' or unit == 'C':
        return '°C'
    elif unit == 'kelvin' or unit == 'K':
        return 'K'
    else:
        return unit or ""


def _unit_format_func(layer, units):
    units = unit_symbol(units)

    # default formatting string
    def _format_unit(val, numeric=True, include_units=True):
        return '{:.03f}{units:s}'.format(val, units=units if include_units else "")

    if layer[INFO.STANDARD_NAME] in ('toa_brightness_temperature', 'brightness_temperature'):
        # BT data limits, Kelvin to degC
        def _format_unit(val, numeric=True, include_units=True):
            return '{:.02f}{units}'.format(val, units=units if include_units else "")
    elif "flag_values" in layer:
        # flag values don't have units
        if "flag_meanings" in layer:
            flag_masks = layer["flag_masks"] if "flag_masks" in layer else [-1] * len(layer["flag_values"])
            flag_info = tuple(zip(layer["flag_meanings"], layer["flag_values"], flag_masks))
            def _format_unit(val, numeric=True, include_units=True, flag_info=flag_info):
                val = int(val)
                if numeric:
                    return '{:d}'.format(val)

                meanings = []
                for fmean, fval, fmask in flag_info:
                    if (val & fmask) == fval:
                        meanings.append(fmean)
                return "{:d} ({:s})".format(val, ", ".join(meanings))
        else:
            def _format_unit(val, numeric=True, include_units=True):
                return '{:d}'.format(int(val))

    return _format_unit


def preferred_units(dsi):
    # FUTURE: Use cfunits or cf_units package
    if dsi[INFO.STANDARD_NAME] == 'toa_bidirectional_reflectance':
        return '1'
    elif dsi[INFO.STANDARD_NAME] in ('toa_brightness_temperature', 'brightness_temperature'):
        return 'degrees_Celsius'
    else:
        return dsi.get(INFO.UNITS, None)


def units_conversion(dsi):
    "return UTF8 unit string, lambda v,inverse=False: convert-raw-data-to-unis"
    # the dataset might be in one unit, but the user may want something else
    # FUTURE: Use cfunits or cf_units package
    punits = preferred_units(dsi)

    # Conversion functions
    # FUTURE: Use cfunits or cf_units package
    if dsi.get(INFO.UNITS) in ('kelvin', 'K') and punits in ('degrees_Celsius', 'C'):
        conv_func = lambda x, inverse=False: x - 273.15 if not inverse else x + 273.15
    else:
        conv_func = lambda x, inverse=False: x

    # Format strings
    format_func = _unit_format_func(dsi, punits)
    return punits, conv_func, format_func


class DocLayerStack(MutableSequence):
    """
    list-like layer set which will slowly eat functionality from Document as warranted, and provide cleaner interfacing to GUI elements
    """
    _doc = None  # weakref to document we belong to
    _store = None
    _u2r = None  # uuid-to-row correspondence cache

    def __init__(self, doc, *args, **kwargs):
        if isinstance(doc, DocLayerStack):
            self._doc = ref(doc._doc())
            self._store = list(doc._store)
        elif isinstance(doc, Document):
            self._doc = ref(doc)
            self._store = list(*args)
        else:
            raise ValueError('cannot initialize DocLayerStack using %s' % type(doc))

    def __setitem__(self, index:int, value: prez):
        if index>=0 and index<len(self._store):
            self._store[index] = value
        elif index == len(self._store):
            self._store.append(value)
        else:
            raise IndexError('%d not a valid index' % index)
        self._u2r = None

    @property
    def uuid2row(self):
        if self._u2r is None:
            self._u2r = dict((p.uuid,i) for (i,p) in enumerate(self._store))
        return self._u2r

    def __getitem__(self, index:int):  # then return layer object
        if isinstance(index, int):
            return self._store[index]
        elif isinstance(index, UUID):  # then return 0..n-1 index in stack
            return self.uuid2row.get(index, None)
        elif isinstance(index, DocLayer):
            return self.uuid2row.get(index.uuid, None)
        elif isinstance(index, prez):
            return self.uuid2row.get(index.uuid, None)
        else:
            raise ValueError('unable to index LayerStack using %s' % repr(index))

    def __iter__(self):
        for each in self._store:
            yield each

    def __len__(self):
        return len(self._store)

    def __delitem__(self, index:int):
        del self._store[index]
        self._u2r = None

    def insert(self, index:int, value: prez):
        self._store.insert(index, value)
        self._u2r = None

    def clear_animation_order(self):
        for i,q in enumerate(self._store):
            self._store[i] = q._replace(a_order=None)

    def index(self, uuid):
        assert(isinstance(uuid, UUID))
        u2r = self.uuid2row
        return u2r.get(uuid, None)

    def change_order_by_indices(self, new_order):
        self._u2r = None
        revised = [self._store[n] for n in new_order]
        self._store = revised

    @property
    def animation_order(self):
        aouu = [(x.a_order, x.uuid) for x in self._store if (x.a_order is not None)]
        aouu.sort()
        ao = tuple(u for a,u in aouu)
        LOG.debug('animation order is {0!r:s}'.format(ao))
        return ao

    @animation_order.setter
    def animation_order(self, layer_or_uuid_seq):
        self.clear_animation_order()
        for nth,lu in enumerate(layer_or_uuid_seq):
            try:
                idx = self[lu]
            except ValueError:
                LOG.warning('unable to find layer in LayerStack')
                raise
            self._store[idx] = self._store[idx]._replace(a_order=nth)


# FUTURE: move these into separate modules

class DocumentAsContextBase(object):
    """Base class for high-level document context objects
    When used as a context object, defers modifications and signals until exit!
    """
    doc = None
    mdb = None
    ws = None


    def __init__(self, doc, mdb, ws):
        self.doc = doc
        self.mdb = mdb
        self.ws = ws


    def __enter__(self):
        pass


    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val is not None:
            # abort code
            raise exc_val
        else:
            # commit code
            pass


# class DocumentAsContext(DocumentAsContextBase):
#     def __enter__(self):
#         raise NotImplementedError()
#
#     def __exit__(self, exc_type, exc_val, exc_tb):
#         if exc_val is not None:
#             # abort code
#             pass
#         else:
#             # commit code
#             pass
#         raise NotImplementedError()


class DocumentAsLayerStack(DocumentAsContextBase):
    """ Represent the document as a list of current layers
    As we transition to timeline model, this stops representing products and starts being a track stack
    """

    def __enter__(self):
        raise NotImplementedError()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val is not None:
            # abort code
            pass
        else:
            # commit code
            pass
        raise NotImplementedError()


class FrameInfo(T.NamedTuple):
    """Represent a data Product as information to display as a frame on timeline
    """
    uuid: UUID
    ident: str  # family::category::serial
    when: span  # time and duration of this frame
    state: flags  # logical state for timeline to display with color and glyphs
    primary: str  # primary description for timeline, e.g. "G16 ABI B06"
    secondary: str  # secondary description, typically time information
    # thumb: QImage  # thumbnail image to embed in timeline item


class TrackInfo(T.NamedTuple):
    track: str  # family::category
    presentation: prez  # colorbar, ranges, gammas, etc
    when: span  # available time-span of the data
    state: flags  # any status or special flags set on the track, according to document / workspace
    primary: str  # primary label for UI
    secondary: str  # secondary label
    frames: T.List[FrameInfo]  # list of frames within specified time span


class DocumentAsTrackStack(DocumentAsContextBase):
    """Work with document as tracks, named by family::category, e.g. IMAGE:geo:toa_reflectance:0.47µm::GOES-16:ABI:CONUS
    This is primarily used by timeline QGraphicsScene bridge, which displays metadatabase + document + workspace content
    zorder >=0 implies active track, i.e. one user has selected as participating in this document
    zorder <0 implies available but inactive track, i.e. metadata / resource / content are unrealized currently
    """

    _actions: T.List[T.Callable] = None  # only available when used as a context manager
    animating: bool = False

    @property
    def playhead_time(self) -> T.Optional[datetime]:
        """current document playhead time, or None if animating
        """
        return self.doc.playhead_time if not self.animating else None

    @playhead_time.setter
    def playhead_time(self, t: datetime):
        """Update the document's playhead time and trigger any necessary signals
        """
        self.doc.playhead_time = t

    @property
    def playback_span(self):
        return self.doc.playback_span

    @playback_span.setter
    def playback_span(self, when: span):
        self.doc.playback_span = when
        # FIXME: signal

    # def track_order_at_time(self, when:datetime = None,
    #                         only_active=False,
    #                         include_products=True
    #                         ) -> T.Iterable[T.Tuple[int, str, T.Optional[Product]]]:
    #     """List of tracks from highest Z order to lowest, at a given time;
    #     (zorder, track) pairs are returned, with zorder>=0 being "part of the document" and <0 being "available but nothing activated"
    #     include_products implies (zorder, track, product-or-None) tuples be yielded; otherwise tracks without products are not yielded
    #     Inactive tracks can be filtered by stopping iteration when zorder<0
    #     tracks are returned as track-name strings
    #     Product instances are readonly metadatabase entries
    #     defaults to current document time
    #     """
    #     if when is None:
    #         when = self.doc.playhead_time
    #     if when is None:
    #         raise RuntimeError("unknown time for iterating track order, animation likely in progress")
    #     with self.mdb as s:
    #         que = s.query(Product).filter((Product.obs_time < when) and ((Product.obs_time + Product.obs_duration) <= when))
    #         if only_active:
    #             famtab = dict((track, z) for (z, track) in self.doc.track_order.enumerate() if z>=0)
    #             active_families = set(famtab.keys())
    #             que = s.filter((Product.family + FCS_SEP + Product.category) in active_families)
    #         else:
    #             famtab = dict((track, z) for (z, track) in self.doc.track_order.enumerate())
    #         prods = list(que.all())
    #         # sort into z order according to document
    #         if include_products:
    #             zult = [(famtab[p.track], p.track, p) for p in prods]
    #         else:
    #             zult = [(famtab[p.track], p.track) for p in prods]
    #         zult.sort(reverse=True)
    #         return zult

    def enumerate_track_names(self, only_active=False) -> T.Iterable[T.Tuple[int, str]]:
        """All the names of the tracks, from highest zorder to lowest
        z>=0 implies an active track in the document, <0 implies potentials that have products either cached or potential
        """
        for z, track in self.doc.track_order.enumerate():
            if only_active and z < 0:
                break
            yield z, track

    def enumerate_tracks_frames(self, only_active: bool = False, when: span = None) -> T.Iterable[TrackInfo]:
        """enumerate tracks as TrackInfo and FrameInfo structures for timeline use, in top-Z to bottom-Z order
        """
        if when is None:  # default to the document's span
            when = self.doc.timeline_span
        when_e = when.e
        with self.mdb as s:
            for z, track in self.doc.track_order.enumerate():  # enumerates from high Z to low Z
                if only_active and (z < 0):
                    break
                fam, ctg = track.split(FCS_SEP)
                LOG.debug("yielding TrackInfo and FrameInfos for {}".format(track))
                frames = []
                # fam_nfo = self.doc.family_info(fam)
                que = s.query(Product).filter((Product.family == fam) and (Product.category == ctg))
                for prod in que.all():
                    prod_e = prod.obs_time + prod.obs_duration
                    if (prod_e <= when.s) or (prod.obs_time >= when_e):
                        # does not intersect our desired span, skip it
                        continue
                    nfo = prod.info
                    fin = FrameInfo(
                        uuid=prod.uuid,
                        ident=prod.ident,
                        when=span(prod.obs_time, prod.obs_duration),
                        state=self.doc.product_state.get(prod.uuid) or flags(),
                        primary=nfo[INFO.DISPLAY_NAME],
                        secondary=nfo[INFO.DISPLAY_TIME],  # prod.obs_time.strftime("%Y-%m-%d %H:%M:%S")
                        # thumb=
                    )
                    frames.append(fin)
                frames.sort(key=lambda x: x.when.s)
                track_span = span.from_s_e(frames[0].when.s, frames[-1].when.e) if frames else None
                trk = TrackInfo(
                    track=track,
                    presentation=self.doc.family_presentation.get(track),
                    when=track_span,
                    frames=frames,
                    state=flags(),  # FIXME
                    primary=' '.join(reversed(fam.split(FCS_SEP))),  #fam_nfo[INFO.DISPLAY_FAMILY],
                    secondary=' '.join(reversed(ctg.split(FCS_SEP)))
                )
                yield z, trk

    def lock_track_to_frame(self, track: str, frame: UUID = None):
        """ User
        """
        if frame is None:
            if frame in self.doc.track_frame_locks:
                del self.doc.track_frame_locks[frame]
        else:
            self.doc.track_frame_locks[track] = frame

        # FIXME: signal, since this will cause effects on animation and potentially static display order
        # this needs to invalidate the current display and any animation
        self.doc.didChangeLayerVisibility.emit({frame: True})

    # def move_track(self, track: str, atop_z: int):
    #     self.doc.track_order.move(atop_z, track)

    # def reorder_tracks(self, new_order: T.Iterable[T.Tuple[int, str]]):

    def tracks_in_family(self, family: str, only_active: bool = True) -> T.Sequence[str]:
        """ yield track names in document that share a common family
        """
        for z, track in self.doc.track_order.enumerate():
            if only_active and z < 0:
                break
            tfam = track.split(FCS_SEP)[0]
            if tfam == family:
                yield track

    # @property
    # def _deferring(self):
    #     "Am I a deferred-action context or an immediate context helper"
    #     return self._actions is not None

    def __init__(self, *args, as_readwrite_context=False, **kwargs):
        super(DocumentAsTrackStack, self).__init__(*args, **kwargs)
        if as_readwrite_context:
            self._actions = []

    def __enter__(self):
        # set up a copy with deferred action until we're done
        return DocumentAsTrackStack(doc=self.doc, mdb=self.mdb, ws=self.ws, as_readwrite_context=True)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val is not None:
            # abort code
            LOG.error("write to DocumentAsTrackStack via read-write context caused an exception")
            raise exc_val
        else:
            # commit code
            while self._actions:
                action = self._actions.pop(0)
                action()


class DocumentAsRegionProbes(DocumentAsContextBase):
    """Document is probed over a variety of regions
    """
    def __enter__(self):
        raise NotImplementedError()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val is not None:
            # abort code
            pass
        else:
            # commit code
            pass
        raise NotImplementedError()


class DocumentAsStyledFamilies(DocumentAsContextBase):
    """Document is composed of products falling into families, each represented onscreen with a style

    """
    def __enter__(self):
        raise NotImplementedError()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val is not None:
            # abort code
            pass
        else:
            # commit code
            pass
        raise NotImplementedError()


class DocumentAsResourcePools(DocumentAsContextBase):
    """Document allows user to specify files and directory systems to search for Resources, Products, Content
    """
    def __enter__(self):
        raise NotImplementedError()

    @property
    def search_paths(self):
        raise NotImplementedError()

    @search_paths.setter
    def search_paths(self, list_of_paths):
        raise NotImplementedError()

    def add_search_path(self, dirname):
        raise NotImplementedError()

    def interactive_open(self, *file_paths):
        raise NotImplementedError()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val is not None:
            # abort code
            pass
        else:
            # commit code
            pass
        raise NotImplementedError()


class DocumentAsRecipeCollection(DocumentAsContextBase):
    def __enter__(self):
        raise NotImplementedError()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val is not None:
            # abort code
            pass
        else:
            # commit code
            pass
        raise NotImplementedError()


class DocumentAsAnimationSequence(DocumentAsContextBase):
    """Document as sequence of product frames that appear and disappear when animated at a multiple of real-time
    """
    def __enter__(self):
        raise NotImplementedError()

    def iterate(self, multiple_of_realtime:float = None, start:datetime=None, stop:datetime=None) -> T.Sequence[T.Tuple[float, T.List[UUID]]]:
        """Yield series of (wall-seconds-offset, [back-to-front-list-of-product-uuids])
        """
        return

    @property
    def playhead_time(self) -> datetime:
        raise NotImplementedError()

    @playhead_time.setter
    def playhead_time(self, when: datetime):
        raise NotImplementedError()

    @property
    def playback_time_range(self) -> T.Tuple[datetime, datetime]:
        raise NotImplementedError()

    @playback_time_range.setter
    def playback_time_range(self, start_end: T.Tuple[datetime, datetime]):
        # set document playback time range
        # if we're in a with-clause, defer signals until outermost exit
        # if we're not in a with-clause, raise ContextNeededForEditing exception
        raise NotImplementedError()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val is not None:
            # abort code
            pass
        else:
            # commit code
            pass
        raise NotImplementedError()


class ProductDataArrayProxy(object):
    """
    As-pythonic-as-possible dataset proxy for SIFT content.
    Hides document, metadatabase, and workspace accesses into something less scattered.
    Intended to evolve toward being usable within a user-accessible friendly namespace.
    Also intended to provide a stable API behind which we can add incremental compatibility with dask/xarray.

    Metadata keys are accessible by .info dictionary-style, or by query
    Data are accessible by .data, or numpy bracket semantics
    """

    def __init__(self, context, uuid: UUID):
        self._ctx = context
        self._uuid = uuid

    def data(self, field=None):
        """Return an xarray-like entity
        """

    @property
    def info(self):
        """Return dictionary-like metadata dictionary
        """


class DocumentAsProductArrayCollection(DocumentAsContextBase):
    """Document as MeasurementDatasetProxy object collection
    merges WS, MDB, and Doc
    """
    _proxies = None

    def __init__(self, *args, **kwargs):
        super(DocumentAsProductArrayCollection, self).__init__(*args, **kwargs)
        self._proxies = {}

    def __getitem__(self, product_uuid: UUID) -> ProductDataArrayProxy:

        it = self._proxies.get(product_uuid, None)
        if it is not None:
            return it

        not_in_doc = False
        not_in_ws = False
        try:
            self.doc[product_uuid]
        except KeyError:
            not_in_doc = True
        nfo = self.ws.get_info(product_uuid)
        if nfo is None:
            not_in_ws = True

        if not_in_doc and not_in_ws:
            raise KeyError("UUID {} is unknown".format(product_uuid))

        self._proxies[product_uuid] = it = ProductDataArrayProxy(self, product_uuid)
        return it

    def __enter__(self):
        raise NotImplementedError()


    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val is not None:
            # abort code
            pass
        else:
            # commit code
            pass
        raise NotImplementedError()


# TimelineDataArrayProxy  # multiple products as a timeline, i.e. a time dimension is added representing all the products on the same timeline
# FUTURE: class DatasetAsTimelineArrayCollection(DocumentAsContextBase):



class Document(QObject):  # base class is rightmost, mixins left of that
    """Document stores user intent
    Document is a set of tracks in a Z order, with Z>=0 for "active" tracks the user is working with
    Tracks with Z-order <0 are inactive, but may be displayed in the timeline as potentials for the user to drag to active
    Document has a playhead, a playback time range, an active timeline display range
    Tracks and frames (aka Products) can have state information set

    This is the low-level "internal" interface that acts as a signaling hub.
    Direct access to the document is being deprecated.
    Most direct access patterns should be migrated to using a contextual view of the document,
    in order to reduce abstraction leakage and permit the document storage to evolve.
    """
    _workspace = None

    # timeline the user has specified:
    track_order: ZList = None  # (zorder, family-name) with higher z above lower z; z<0 should not occur

    # overall visible range of the active data
    timeline_span: span = None

    # playback information
    playhead_time: datetime = None  # document stored playhead time
    playback_per_sec: float = 60.0  # data time increment per wall-second

    # playback time range, if not None is a subset of overall timeline
    playback_span: span = None

    # user-directed overrides on tracks and frames (products)
    track_state: T.Mapping[str, flags] = None
    product_state: T.Mapping[UUID, flags] = None

    # user can lock tracks to a single frame throughout
    track_frame_locks: T.Mapping[str, UUID] = None

    # Maps of family names to their document recipes
    family_presentation: T.Mapping[str, prez] = None
    family_composition: T.Mapping[str, CompositeRecipe] = None  # using multiple products to present RGBA
    family_calculation: T.Mapping[str, object] = None  # algebraic combinations of multiple products

    # DEPRECATION in progress: layer sets
    """
    Document has one or more LayerSets choosable by the user (one at a time) as currentLayerSet
    LayerSets configure animation order, visibility, enhancements and linear combinations
    LayerSets can be cloned from the prior active LayerSet when unconfigured
    Document has Probes, which operate on the currentLayerSet
    Probes have spatial areas (point probes, shaped areas)
    Probe areas are translated into localized data masks against the workspace raw data content
    """
    current_set_index = 0
    _layer_sets = None  # list(DocLayerSet(prez, ...) or None)
    _layer_with_uuid = None  # dict(uuid:Doc____Layer)

    # signals
    # Clarification: Layer interfaces migrate to layer meaning "current active products under the playhead"
    didAddBasicLayer = pyqtSignal(tuple, UUID, prez)  # new order list with None for new layer; info-dictionary, overview-content-ndarray
    didAddCompositeLayer = pyqtSignal(tuple, UUID, prez)  # comp layer is derived from multiple basic layers and has its own UUID
    didRemoveLayers = pyqtSignal(tuple, list, int, int)  # new order, UUIDs that were removed from current layer set, first row removed, num rows removed
    willPurgeLayer = pyqtSignal(UUID)  # UUID of the layer being removed
    didReorderLayers = pyqtSignal(tuple)  # list of original indices in their new order, None for new layers
    didChangeLayerVisibility = pyqtSignal(dict)  # {UUID: new-visibility, ...} for changed layers
    didReorderAnimation = pyqtSignal(tuple)  # list of UUIDs representing new animation order
    didChangeLayerName = pyqtSignal(UUID, str)  # layer uuid, new name
    didSwitchLayerSet = pyqtSignal(int, DocLayerStack, tuple)  # new layerset number typically 0..3, list of prez tuples representing new display order, new animation order
    didChangeColormap = pyqtSignal(dict)  # dict of {uuid: colormap-name-or-UUID, ...} for all changed layers
    didChangeColorLimits = pyqtSignal(dict)  # dict of {uuid: (vmin, vmax), ...} for all changed layers
    didChangeGamma = pyqtSignal(dict)  # dict of {uuid: gamma float, ...} for all changed layers
    didChangeComposition = pyqtSignal(tuple, UUID, prez)  # new-layer-order, changed-layer, new-prez
    didChangeCompositions = pyqtSignal(tuple, list, list)  # new-layer-order, changed-layers, new-prezs
    didCalculateLayerEqualizerValues = pyqtSignal(dict)  # dict of {uuid: (value, normalized_value_within_clim)} for equalizer display
    didChangeProjection = pyqtSignal(str, dict)  # name of projection, dict of projection information
    # didChangeShapeLayer = pyqtSignal(dict)
    didAddFamily = pyqtSignal(str, dict)  # name of the newly added family and dict of family info
    didReorderTracks = pyqtSignal(set, set)  # added track names, removed track names

    # high-level contexts providing purposed access to low-level document and its storage, as well as MDB and WS
    # layer display shows active products under the playhead
    as_layer_stack: DocumentAsLayerStack = None
    # track display shows all available products according to metadatabase; some tracks are active, i.e. they have are allowed to present as part of document
    as_track_stack: DocumentAsTrackStack = None
    # presentation migrates from being layer-owned to being family-owned
    as_styled_families: DocumentAsStyledFamilies = None
    # metadata is collected from resource pools, which can be local directories or remote URIs
    as_resource_pools: DocumentAsResourcePools = None
    # presentation, calculation, and composition recipes are part of the document
    as_recipe_collection: DocumentAsRecipeCollection = None
    # SceneGraphManager needs a plan on how to present and animate product content
    as_animation_sequence: DocumentAsAnimationSequence = None
    # content gets probes applied across points and regions
    as_region_probes: DocumentAsRegionProbes = None

    def __init__(self, workspace, config_dir=DOCUMENT_SETTINGS_DIR, layer_set_count=DEFAULT_LAYER_SET_COUNT, **kwargs):
        super(Document, self).__init__(**kwargs)
        self.config_dir = config_dir
        if not os.path.isdir(self.config_dir):
            LOG.info("Creating settings directory {}".format(self.config_dir))
            os.makedirs(self.config_dir)

        # high level context managers provide access patterns for use-case based behaviors
        self.as_layer_stack = DocumentAsLayerStack(self, workspace.metadatabase, workspace)
        self.as_track_stack = DocumentAsTrackStack(self, workspace.metadatabase, workspace)
        self.as_styled_families= DocumentAsStyledFamilies(self, workspace.metadatabase, workspace)
        self.as_resource_pools = DocumentAsResourcePools(self, workspace.metadatabase, workspace)
        self.as_recipe_collection = DocumentAsRecipeCollection(self, workspace.metadatabase, workspace)
        self.as_animation_sequence = DocumentAsAnimationSequence(self, workspace.metadatabase, workspace)
        self.as_region_probes = DocumentAsRegionProbes(self, workspace.metadatabase, workspace)

        self._workspace = workspace
        self._layer_sets = [DocLayerStack(self)] + [None] * (layer_set_count - 1)
        self._layer_with_uuid = {}
        # FIXME: Copy?
        self.colormaps = ALL_COLORMAPS
        self.usermaps = USER_COLORMAPS
        self.available_projections = OrderedDict((
            ('Mercator', {
                'proj4_str': '+proj=merc +datum=WGS84 +ellps=WGS84 +over',
                'default_center': (144.8, 13.5),  # lon, lat center point (Guam)
                'default_width': 20.,  # degrees from left edge to right edge
                'default_height': 20.,  # degrees from bottom edge to top edge
            }),
            ('LCC (CONUS)', {
                'proj4_str': '+proj=lcc +a=6371200 +b=6371200 +lat_0=25 +lat_1=25 +lon_0=-95 +units=m +no_defs +over',
                'default_center': (-95, 35.),
                'default_width': 25.,
                'default_height': 25.,
            }),
            # ('Platte Carre', {}),
            ('Himawari Geos', {
                'proj4_str': '+proj=geos +a=6378137 +b=6356752.299581327 +lon_0=140.7 +h=35785863 +over',
                'default_center': (144.8, 13.5),  # lon, lat center point (Guam)
                'default_width': 20.,  # degrees from left edge to right edge
                'default_height': 20.,  # degrees from bottom edge to top edge
            }),
            ('GOES East', {
                'proj4_str': '+proj=geos +lon_0=-75 +h=35786023.0 +a=6378137.0 +b=6356752.31414 +sweep=x +units=m',
                'default_center': (-75, 13.5),  # lon, lat center point (Guam)
                'default_width': 20.,  # degrees from left edge to right edge
                'default_height': 20.,  # degrees from bottom edge to top edge
            }),
            ('GOES Test', {
                'proj4_str': '+proj=geos +lon_0=-89.5 +h=35786023.0 +a=6378137.0 +b=6356752.31414 +sweep=x +units=m',
                'default_center': (-89.5, 13.5),  # lon, lat center point (Guam)
                'default_width': 20.,  # degrees from left edge to right edge
                'default_height': 20.,  # degrees from bottom edge to top edge
            }),
            ('GOES Central', {
                'proj4_str': '+proj=geos +lon_0=-105 +h=35786023.0 +a=6378137.0 +b=6356752.31414 +sweep=x +units=m',
                'default_center': (-105, 13.5),  # lon, lat center point (Guam)
                'default_width': 20.,  # degrees from left edge to right edge
                'default_height': 20.,  # degrees from bottom edge to top edge
            }),
            ('GOES West', {
                'proj4_str': '+proj=geos +lon_0=-137 +h=35786023.0 +a=6378137.0 +b=6356752.31414 +sweep=x +units=m',
                'default_center': (-137, 13.5),  # lon, lat center point (Guam)
                'default_width': 20.,  # degrees from left edge to right edge
                'default_height': 20.,  # degrees from bottom edge to top edge
            }),
        ))
        self.default_projection = 'LCC (CONUS)'
        self.current_projection = self.default_projection
        self.recipe_manager = RecipeManager(self.config_dir)
        self._recipe_layers = {}
        # HACK: This should probably be part of the metadata database in the future
        self._families = defaultdict(list)

        # Create directory if it does not exist
        filepath = os.path.join(self.config_dir, 'colormaps')
        if not os.path.exists(filepath):
            os.makedirs(filepath)

        # Import data
        qtData = {}
        for subdir, dirs, files in os.walk(filepath):
            for ToImportFile in files:
                nfp = os.path.join(subdir, ToImportFile)
                try:
                    if os.path.splitext(ToImportFile)[1] != ".json":
                        continue
                    ifile = open(nfp, "r")
                    toImport = json.loads(ifile.read())
                    qtData[os.path.splitext(ToImportFile)[0]] = toImport
                except IOError:
                    LOG.error("Error importing gradient")
                    raise
                except ValueError:
                    raise

        for item in qtData:
            self.add_to_maps(qtData[item], item)

        # timeline document storage setup with initial track order and time range
        self.product_state = defaultdict(flags)
        self.track_state = defaultdict(flags)
        self.track_order = ZList()
        self.track_frame_locks = {}
        self.family_calculation = {}
        self.family_composition = {}
        self.family_presentation = {}
        # scan available metadata for initial state
        # FIXME: refresh this once background scan finishes
        self.timeline_span = self.playback_span = self.potential_product_span()
        self.sync_potential_tracks_from_metadata()

    def potential_product_span(self) -> span:
        with self._workspace.metadatabase as S:
            all_times = list(S.query(Product.obs_time, Product.obs_duration).distinct())
        if not all_times:
            LOG.warning("no data available, using default time span")
            nau = datetime.utcnow()
            sixh = timedelta(hours=6)
            return span(nau - sixh, sixh * 2)
        starts = [s for (s, d) in all_times]
        ends = [(s + d) for (s, d) in all_times]
        s = min(starts)
        e = max(ends)
        return span(s, e - s)

    def potential_tracks(self) -> T.Iterable[str]:
        """List the names of available tracks (both active and potential) according to the metadatabase
        """
        with self._workspace.metadatabase as S:
            return ((f + FCS_SEP + c) for (f, c) in S.query(Product.family, Product.category).distinct())

    def sync_potential_tracks_from_metadata(self):
        """update track_order to include any newly available tracks
        """
        all_tracks = list(self.potential_tracks())
        all_tracks.sort()
        old_tracks = set(name for z, name in self.track_order.enumerate())
        for track in all_tracks:
            self.track_order.append(track, start_negative=True, not_if_present=True)
        for dismissed in old_tracks - set(all_tracks):
            LOG.debug("removing track {} from track_order".format(dismissed))
            self.track_order.remove(dismissed)
        new_tracks = set(name for z, name in self.track_order.enumerate())
        if old_tracks != new_tracks:
            LOG.info("went from {} available tracks to {}".format(len(old_tracks), len(new_tracks)))
            self.didReorderTracks.emit(new_tracks - old_tracks, old_tracks - new_tracks)

    def find_colormap(self, colormap):
        if isinstance(colormap, str) and colormap in self.colormaps:
            colormap = self.colormaps[colormap]
        return colormap

    def projection_info(self, projection_name=None):
        return self.available_projections[projection_name or self.current_projection]

    def change_projection(self, projection_name=None):
        if projection_name is None:
            projection_name = self.default_projection
        assert projection_name in self.available_projections
        if projection_name != self.current_projection:
            LOG.info("Changing projection from '{}' to '{}'".format(self.current_projection, projection_name))
            self.current_projection = projection_name
            self.didChangeProjection.emit(
                self.current_projection,
                self.projection_info(self.current_projection)
            )

    def add_to_maps(self, colorItem, name):
        pointList = colorItem["ticks"]
        floats = []
        hexes = []
        for point in pointList:
            floats.append(point[0])
            rgb = point[1]
            hexCode = rgb2hex(rgb[0], rgb[1], rgb[2])
            hexes.append(hexCode)

        floats, hexes = zip(*sorted(zip(floats, hexes)))

        floats = list(floats)
        hexes = list(hexes)

        if floats[0] != 0:
            floats = [0] + floats
            hexes = [hexes[0]] + hexes
        if floats[-1] != 1:
            floats.append(1)
            hexes.append(hexes[-1])

        try:
            toAdd = Colormap(colors=hexes, controls=floats)
            self.colormaps[name] = toAdd
            self.usermaps[name] = toAdd
        except AssertionError:
            LOG.error("Error creating or setting colormap")
            raise

    # Update new gradient into save location
    def updateGCColorMap(self, colorMap, name):
        filepath = os.path.join(self.config_dir, 'colormaps')

        try:
            iFile = open(os.path.join(filepath, name + '.json'), 'w')
            iFile.write(json.dumps(colorMap, indent=2, sort_keys=True))
            iFile.close()
        except IOError:
            LOG.error("Error saving gradient")

        self.add_to_maps(colorMap, name)

        # Update live map
        self.change_colormap_for_layers(name)


    # Remove gradient from save location (on delete)
    def removeGCColorMap(self, name):
        try:
            os.remove(os.path.join(self.config_dir, 'colormaps', name + '.json'))
        except OSError:
            pass

        del self.colormaps[name]
        del self.usermaps[name]



    def current_projection_index(self):
        return list(self.available_projections.keys()).index(self.current_projection)

    def change_projection_index(self, idx):
        return self.change_projection(tuple(self.available_projections.keys())[idx])

    @property
    def current_layer_set(self):
        cls = self._layer_sets[self.current_set_index]
        assert(isinstance(cls, DocLayerStack))
        return cls

    def _insert_layer_with_info(self, info: DocLayer, cmap=None, insert_before=0):
        """
        insert a layer into the presentations but do not signal
        :return: new prez tuple, new reordered indices tuple
        """
        if cmap is None:
            cmap = info.get(INFO.COLORMAP)
        gamma = 1.
        if isinstance(info, DocRGBLayer):
            gamma = (1.,) * 3
        elif hasattr(info, 'l'):
            gamma = (1.,) * len(info.l)

        p = prez(uuid=info[INFO.UUID],
                 kind=info[INFO.KIND],
                 visible=True,
                 a_order=None,
                 colormap=cmap,
                 climits=info[INFO.CLIM],
                 gamma=gamma,
                 mixing=Mixing.NORMAL)

        q = p._replace(visible=False)  # make it available but not visible in other layer sets
        old_layer_count = len(self._layer_sets[self.current_set_index])
        for dex, lset in enumerate(self._layer_sets):
            if lset is not None:  # uninitialized layer sets will be None
                lset.insert(insert_before, p if dex == self.current_set_index else q)

        reordered_indices = tuple([None] + list(range(old_layer_count)))  # FIXME: this should obey insert_before, currently assumes always insert at top
        return p, reordered_indices

    def open_file(self, path, insert_before=0):
        """
        open an arbitrary file and make it the new top layer.
        emits docDidChangeLayer followed by docDidChangeLayerOrder
        :param path: file to open and add
        :return: overview (uuid:UUID, datasetinfo:dict, overviewdata:numpy.ndarray)
        """
        # collect product and resource information but don't yet import content
        products = list(self._workspace.collect_product_metadata_for_paths([path]))
        if not products:
            raise ValueError('no products available in {}'.format(path))
        if len(products) > 1:
            LOG.warning('more than one product available at this path - FIXME')
        info = products[0]
        assert(info is not None)
        uuid = info[INFO.UUID]

        if uuid in self._layer_with_uuid:
            LOG.warning("layer with UUID {} already in document?".format(uuid))
            active_content_data = self._workspace.get_content(uuid)
            return uuid, info, active_content_data

        return self.activate_product_uuid_as_new_layer(uuid, insert_before=insert_before)

    def activate_product_uuid_as_new_layer(self, uuid: UUID, insert_before=0):
        if uuid in self._layer_with_uuid:
            LOG.debug("Layer already loaded: {}".format(uuid))
            active_content_data = self._workspace.import_product_content(uuid)
            return uuid, self[uuid], active_content_data

        # FUTURE: Load this async, the slots for the below signal need to be OK with that
        active_content_data = self._workspace.import_product_content(uuid)
        # updated metadata with content information (most importantly nav information)
        info = self._workspace.get_info(uuid)
        assert(info is not None)
        LOG.info('cell_width: {}'.format(repr(info[INFO.CELL_WIDTH])))

        LOG.info('new layer info: {}'.format(repr(info)))
        self._layer_with_uuid[uuid] = dataset = DocBasicLayer(self, info)
        if INFO.UNIT_CONVERSION not in dataset:
            dataset[INFO.UNIT_CONVERSION] = units_conversion(dataset)
        if INFO.FAMILY not in dataset:
            dataset[INFO.FAMILY] = self._family_for_layer(dataset)
        presentation, reordered_indices = self._insert_layer_with_info(dataset, insert_before=insert_before)

        # signal updates from the document
        self.didAddBasicLayer.emit(reordered_indices, dataset.uuid, presentation)
        self._add_layer_family(dataset)
        # update any RGBs that could use this to make an RGB
        self.sync_composite_layer_prereqs([dataset[INFO.SCHED_TIME]])

        return uuid, dataset, active_content_data

    def _family_for_layer(self, uuid_or_layer):
        if isinstance(uuid_or_layer, UUID):
            uuid_or_layer = self[uuid_or_layer]
        if INFO.FAMILY in uuid_or_layer:
            LOG.debug('using pre-existing family {}'.format(uuid_or_layer[INFO.FAMILY]))
            return uuid_or_layer[INFO.FAMILY]
        # kind:pointofreference:measurement:wavelength
        kind = uuid_or_layer[INFO.KIND]
        refpoint = 'unknown'  # FUTURE: geo/leo
        measurement = uuid_or_layer.get(INFO.STANDARD_NAME)
        if uuid_or_layer.get('recipe'):
            # RGB
            subcat = uuid_or_layer['recipe'].name
        elif uuid_or_layer.get(INFO.CENTRAL_WAVELENGTH):
            # basic band
            subcat = uuid_or_layer[INFO.CENTRAL_WAVELENGTH]
        else:
            # higher level product or algebraic layer
            subcat = uuid_or_layer[INFO.DATASET_NAME]
        return "{}:{}:{}:{}".format(kind.name, refpoint, measurement, subcat)

    def _add_layer_family(self, layer):
        family = layer[INFO.FAMILY]
        is_new = family not in self._families
        self._families[family].append(layer[INFO.UUID])
        if is_new:
            self.didAddFamily.emit(family, self.family_info(family))
        return family

    def _remove_layer_from_family(self, uuid):
        family = self[uuid][INFO.FAMILY]
        self._families[family].remove(uuid)

    def family_info(self, family_or_layer_or_uuid):
        family = layer = family_or_layer_or_uuid
        if isinstance(family_or_layer_or_uuid, UUID):
            layer = self[family_or_layer_or_uuid]
        if isinstance(layer, DocBasicLayer):
            family = layer[INFO.FAMILY]

        # one layer that represents all the layers in this family
        family_rep = self[self._families[family][0]]

        # convert family subcategory to displayable name
        # if isinstance(family[2], UUID):
        #     # RGB Recipes, this needs more thinking
        #     display_family = family_rep[INFO.SHORT_NAME]
        # elif not isinstance(family[2], str):
        #     display_family = "{:.02f} µm".format(family[2])
        # else:
        #     display_family = family[2]
        family_name_components = family.split(':')
        display_family = family_rep[INFO.SHORT_NAME] + ' ' + ' '.join(reversed(family_name_components))
        # display_family = str(family)

        # NOTE: For RGBs the SHORT_NAME will update as the RGB changes
        return {
            INFO.VALID_RANGE: family_rep[INFO.VALID_RANGE],
            INFO.UNIT_CONVERSION: family_rep[INFO.UNIT_CONVERSION],
            INFO.SHORT_NAME: family_rep[INFO.SHORT_NAME],
            INFO.UNITS: family_rep[INFO.UNITS],
            INFO.KIND: family_rep[INFO.KIND],
            INFO.DISPLAY_FAMILY: display_family,
        }

    def open_files(self, paths, insert_before=0):
        """
        sort paths into preferred load order
        open files in order, yielding uuid, info, overview_content
        :param paths: paths to open
        :param insert_before: where to insert them in layer list
        :return:
        """
        # Load all the metadata so we can sort the files
        infos = list(self._workspace.collect_product_metadata_for_paths(paths))

        # Use the metadata to sort the paths
        paths = list(self.sort_datasets_into_load_order(infos))
        for path in paths:
            yield self.open_file(path, insert_before)

    def sort_paths(self, paths):
        """
        :param paths: list of paths
        :return: list of paths
        """
        LOG.info("DEPRECATED: sort products, not files, since files may have multiple products")
        return list(sorted(paths))
        # products = list(self._workspace.collect_product_metadata_for_paths(paths))
        # LOG.debug('sorting products {} for paths {}'.format(repr(products), repr(paths)))
        # infos = [x.info for x in products]
        # LOG.info('path info for sorting: {}'.format(repr(infos)))
        # paths = list(reversed(self.sort_datasets_into_load_order(infos)))  # go from load order to display order by reversing
        # return paths

    def sort_datasets_into_load_order(self, infos):
        """
        given a list of paths, sort them into order, assuming layers are added at top of layer list
        first: unknown paths
        outer order: descending band number
        inner order: descending time step
        :param infos: iterable of info dictionaries
        :return: ordered list of path strings
        """

        # FIXME: It is not possible for a pathname to be considered "irrelevant"
        # riffraff = [path for path in paths if not nfo[path]]
        riffraff = []
        # ahi = [nfo[path] for path in paths if nfo[path]]
        # names = [path for path in paths if nfo[path]]
        # bands = [x.get(INFO.BAND, None) for x in ahi]
        # times = [x.get(INFO.SCHED_TIME, None) for x in ahi]
        # order = [(band, time, path) for band,time,path in zip(bands,times,names)]

        def _sort_key(info):
            return (info.get(INFO.DATASET_NAME),
                    info.get(INFO.OBS_TIME),
                    info.get(INFO.PATHNAME))
        order = sorted(infos, key=_sort_key, reverse=True)
        paths = riffraff + [info.get(INFO.PATHNAME) for info in order]
        LOG.debug(paths)
        return paths

    def time_label_for_uuid(self, uuid):
        """used to update animation display when a new frame is shown
        """
        if not uuid:
            return "YYYY-MM-DD HH:MM"
        info = self._layer_with_uuid[uuid]
        return info.get(INFO.DISPLAY_TIME, '--:--')

    def prez_for_uuids(self, uuids, lset=None):
        if lset is None:
            lset = self.current_layer_set
        for p in lset:
            if p.uuid in uuids:
                yield p

    def prez_for_uuid(self, uuid, lset=None):
        for p in self.prez_for_uuids((uuid,), lset=lset):
            return p

    def colormap_for_uuids(self, uuids, lset=None):
        for p in self.prez_for_uuids(uuids, lset=lset):
            yield p.colormap

    def colormap_for_uuid(self, uuid, lset=None):
        for p in self.colormap_for_uuids((uuid,), lset=lset):
            return p

    def valid_range_for_uuid(self, uuid):
        # Limit ourselves to what information
        # in the future valid range may be different than the default CLIMs
        return self[uuid][INFO.CLIM]

    def convert_value(self, uuid, x, inverse=False):
        return self[uuid][INFO.UNIT_CONVERSION][1](x, inverse=inverse)

    def format_value(self, uuid, x, numeric=True, units=True):
        return self[uuid][INFO.UNIT_CONVERSION][2](x, numeric=numeric, units=units)

    def flipped_for_uuids(self, uuids, lset=None):
        for p in self.prez_for_uuids(uuids, lset=lset):
            default_clim = self._layer_with_uuid[p.uuid][INFO.CLIM]
            yield ((p.climits[1] - p.climits[0]) > 0) != ((default_clim[1] - default_clim[0]) > 0)

    def _get_equalizer_values_image(self, lyr, pinf, xy_pos):
        value = self._workspace.get_content_point(pinf.uuid, xy_pos)
        unit_info = lyr[INFO.UNIT_CONVERSION]
        nc, xc = unit_info[1](np.array(pinf.climits))
        # calculate normalized bar width relative to its current clim
        new_value = unit_info[1](value)
        if nc > xc:  # sometimes clim is swapped to reverse color scale
            nc, xc = xc, nc
        if np.isnan(new_value):
            return None
        else:
            if xc == nc:
                bar_width = 0
            else:
                bar_width = (np.clip(new_value, nc, xc) - nc) / (xc - nc)
            return new_value, bar_width, unit_info[2](new_value, numeric=False)

    def _get_equalizer_values_rgb(self, lyr, pinf, xy_pos):
        # We can show a valid RGB
        # Get 3 values for each channel
        # XXX: Better place for this?
        def _sci_to_rgb(v, cmin, cmax):
            if np.isnan(v):
                return None

            if cmin == cmax:
                return 0
            elif cmin > cmax:
                if v > cmin:
                    v = cmin
                elif v < cmax:
                    v = cmax
            else:
                if v < cmin:
                    v = cmin
                elif v > cmax:
                    v = cmax

            return int(round(abs(v - cmin) / abs(cmax - cmin) * 255.))
        values = []
        for dep_lyr, clims in zip(lyr.l[:3], pinf.climits):
            if dep_lyr is None:
                values.append(None)
            elif clims is None or clims[0] is None:
                values.append(None)
            else:
                value = self._workspace.get_content_point(dep_lyr[INFO.UUID], xy_pos)
                values.append(_sci_to_rgb(value, clims[0], clims[1]))

        nc = 0
        xc = 255
        bar_widths = [(np.clip(value, nc, xc) - nc) / (xc - nc) for value in values if value is not None]
        bar_width = np.mean(bar_widths) if len(bar_widths) > 0 else 0
        values = ",".join(["{:3d}".format(v if v is not None else 0) for v in values])
        return values, bar_width, values

    def update_equalizer_values(self, probe_name, state, xy_pos, uuids=None):
        """user has clicked on a point probe; determine relative and absolute values for all document image layers
        """
        # if the point probe was turned off then we don't want to have the equalizer
        if not state:
            self.didCalculateLayerEqualizerValues.emit({})
            return

        if uuids is None:
            uuids = [(pinf.uuid, pinf) for pinf in self.current_layer_set]
        else:
            uuids = [(pinf.uuid, pinf) for pinf in self.prez_for_uuids(uuids)]
        zult = {}
        for uuid, pinf in uuids:
            try:
                lyr = self._layer_with_uuid[pinf.uuid]
                if lyr[INFO.KIND] in [KIND.IMAGE, KIND.COMPOSITE]:
                    zult[pinf.uuid] = self._get_equalizer_values_image(lyr, pinf, xy_pos)
                elif lyr[INFO.KIND] == KIND.RGB:
                    zult[pinf.uuid] = self._get_equalizer_values_rgb(lyr, pinf, xy_pos)
            except ValueError:
                LOG.warning("Could not get equalizer values for {}".format(uuid))
                zult[pinf.uuid] = (0, 0, 0)

        self.didCalculateLayerEqualizerValues.emit(zult)  # is picked up by layer list model to update display

    def _clone_layer_set(self, existing_layer_set):
        return DocLayerStack(existing_layer_set)

    @property
    def current_animation_order(self):
        """
        return tuple of UUIDs representing the animation order in the currently selected layer set
        :return: tuple of UUIDs
        """
        return self.current_layer_set.animation_order

    @property
    def current_layer_uuid_order(self):
        """
        list of UUIDs (top to bottom) currently being displayed, independent of visibility/validity
        :return:
        """
        return tuple(x.uuid for x in self.current_layer_set)

    @property
    def current_visible_layer_uuid(self):
        """
        :return: the topmost visible layer's UUID
        """
        for x in self.current_layer_set:
            layer = self._layer_with_uuid[x.uuid]
            if x.visible and layer.is_valid:
                return x.uuid
        return None

    @property
    def current_visible_layer_uuids(self):
        for x in self.current_layer_set:
            layer = self._layer_with_uuid[x.uuid]
            if x.visible and layer.is_valid:
                yield x.uuid

    # TODO: add a document style guide which says how different bands from different instruments are displayed

    @property
    def active_layer_order(self):
        """
        return list of valid (can-be-displayed) layers which are either visible or in the animation order
        typically this is used by the scenegraphmanager to synchronize the scenegraph elements
        :return: sequence of (layer_prez, layer) pairs, with order=0 for non-animating layers
        """
        for layer_prez in self.current_layer_set:
            if layer_prez.visible or layer_prez.a_order is not None:
                layer = self._layer_with_uuid[layer_prez.uuid]
                if not layer.is_valid:
                    # we don't have enough information to display this layer yet, it's still loading or being configured
                    continue
                yield layer_prez, layer

    def layers_where(self, is_valid=None, is_active=None, in_type_set=None,
                     have_proj=None):
        """
        query current layer set for layers matching criteria
        :param is_valid: None, or True/False whether layer is valid (could be displayed)
        :param is_active: None, or True/False whether layer is active (valid & (visible | animatable))
        :param in_type_set: None, or set of Python types that the layer falls into
        :return: sequence of layers in no particular order
        """
        for layer_prez in self.current_layer_set:
            layer = self._layer_with_uuid[layer_prez.uuid]
            valid = layer.is_valid
            if is_valid is not None:
                if valid != is_valid:
                    continue
            if is_active is not None:
                active = valid and (layer_prez.visible or layer_prez.a_order is not None)
                if active != is_active:
                    continue
            if in_type_set is not None:
                if type(layer) not in in_type_set:
                    continue
            if have_proj is not None:
                if layer[INFO.PROJ] != have_proj:
                    continue
            yield layer

    def select_layer_set(self, layer_set_index:int):
        """Change the selected layer set, 0..N (typically 0..3), cloning the old set if needed
        emits docDidChangeLayerOrder with an empty list implying complete reassessment,
        if cloning of layer set didn't occur

        :param layer_set_index: which layer set to switch to

        """

        # the number of layer sets is no longer fixed, but you can't select more than 1 beyond the end of the list!
        assert(layer_set_index <= len(self._layer_sets) and layer_set_index >= 0)

        # if we are adding a layer set, do that now
        if layer_set_index == len(self._layer_sets) :
            self._layer_sets.append(None)

        # if the selected layer set doesn't exist yet, clone another set to make it
        did_clone = False
        if self._layer_sets[layer_set_index] is None:
            self._layer_sets[layer_set_index] = self._clone_layer_set(self._layer_sets[self.current_set_index])
            did_clone = True

        # switch to the new layer set and set off events to let others know about the change
        self.current_set_index = layer_set_index
        self.didSwitchLayerSet.emit(layer_set_index, self.current_layer_set, self.current_animation_order)

    def row_for_uuid(self, *uuids):
        d = dict((q.uuid,i) for i,q in enumerate(self.current_layer_set))
        if len(uuids)==1:
            return d[uuids[0]]
        else:
            return [d[x] for x in uuids]

    def toggle_layer_visibility(self, rows_or_uuids, visible=None):
        """
        change the visibility of a layer or layers
        :param rows_or_uuids: layer index or index list, 0..n-1, alternately UUIDs of layers
        :param visible: True, False, or None (toggle)
        """
        L = self.current_layer_set
        zult = {}
        if isinstance(rows_or_uuids, int) or isinstance(rows_or_uuids, UUID):
            rows_or_uuids = [rows_or_uuids]
        for dex in rows_or_uuids:
            if isinstance(dex, UUID):
                dex = L.index(dex)  # returns row index
            old = L[dex]
            vis = (not old.visible) if visible is None else visible
            nu = old._replace(visible=vis)
            L[dex] = nu
            zult[nu.uuid] = nu.visible
        self.didChangeLayerVisibility.emit(zult)

    def animation_changed_visibility(self, changes):
        """
        this is triggered by animation being stopped,
        via signal scenegraphmanager.didChangeLayerVisibility
        in turn we generate our own didChangeLayerVisibility to ensure document views are up to date
        :param changes: dictionary of {uuid:bool} with new visibility state
        :return:
        """
        u2r = dict((q.uuid,i) for i,q in enumerate(self.current_layer_set))
        L = self.current_layer_set
        for uuid,visible in changes.items():
            dex = L[uuid]
            old = L[dex]
            L[dex] = old._replace(visible=visible)
        self.didChangeLayerVisibility.emit(changes)

    def next_last_step(self, uuid, delta=0, bandwise=False):
        """
        given a selected layer uuid,
        find the next or last time/bandstep (default: the layer itself) in the document
        make all layers in the sibling group invisible save that timestep
        :param uuid: layer we're focusing on as reference
        :param delta: -1 => last step, 0 for focus step, +1 for next step
        :param bandwise: True if we want to change by band instead of time
        :return: UUID of new focus layer
        """
        if bandwise:  # next or last band
            consult_guide = self.channel_siblings
        else:
            consult_guide = self.time_siblings
        sibs, dex = consult_guide(uuid)
        # LOG.debug('layer {0} family is +{1} of {2!r:s}'.format(uuid, dex, sibs))
        if not sibs:
            LOG.info('nothing to do in next_last_timestep')
            self.toggle_layer_visibility(uuid, True)
            return uuid
        dex += delta + len(sibs)
        dex %= len(sibs)
        new_focus = sibs[dex]
        del sibs[dex]
        if sibs:
            self.toggle_layer_visibility(sibs, False)
        self.toggle_layer_visibility(new_focus, True) # FUTURE: do these two commands in one step
        return new_focus

    def is_layer_visible(self, row):
        return self.current_layer_set[row].visible

    def layer_animation_order(self, layer_number):
        return self.current_layer_set[layer_number].a_order

    def change_layer_name(self, row, new_name):
        uuid = self.current_layer_set[row].uuid if not isinstance(row, UUID) else row
        info = self._layer_with_uuid[uuid]
        assert(uuid==info[INFO.UUID])
        if not new_name:
            # empty string, reset to default DISPLAY_NAME
            new_name = info.default_display_name
        info[INFO.DISPLAY_NAME] = new_name
        self.didChangeLayerName.emit(uuid, new_name)

    def change_colormap_for_layers(self, name, uuids=None):
        L = self.current_layer_set
        if uuids is not None:
            uuids = self.time_siblings_uuids(uuids)
        else:  # all data layers
            uuids = [pinfo.uuid for pinfo in L]

        nfo = {}
        for uuid in uuids:
            for dex,pinfo in enumerate(L):
                if pinfo.uuid==uuid:
                    L[dex] = pinfo._replace(colormap=name)
                    nfo[uuid] = name
        self.didChangeColormap.emit(nfo)

    def current_layers_where(self, kinds=None, bands=None, uuids=None,
                             dataset_names=None, wavelengths=None):
        """check current layer list for criteria and yield"""
        L = self.current_layer_set
        for idx,p in enumerate(L):
            if (uuids is not None) and (p.uuid not in uuids):
                continue
            layer = self._layer_with_uuid[p.uuid]
            if (kinds is not None) and (layer.kind not in kinds):
                continue
            if (bands is not None) and (layer[INFO.BAND] not in bands):
                continue
            if (dataset_names is not None) and (layer[INFO.DATASET_NAME] not in dataset_names):
                continue
            if (wavelengths is not None) and (layer.get(INFO.CENTRAL_WAVELENGTH) not in wavelengths):
                continue
            yield (idx, p, layer)

    def change_clims_for_layers_where(self, clims, **query):
        """
        query using .current_layers_where() and change clims en masse
        :param clims: new color limits consistent with layer's presentation
        :param query: see current_layers_where()
        :return:
        """
        nfo = {}
        L = self.current_layer_set
        for idx, pz, layer in self.current_layers_where(**query):
            new_pz = pz._replace(climits=clims)
            nfo[layer.uuid] = new_pz.climits
            L[idx] = new_pz
        self.didChangeColorLimits.emit(nfo)

    def change_clims_for_siblings(self, uuid, clims):
        uuids = self.time_siblings(uuid)[0]
        return self.change_clims_for_layers_where(clims, uuids=uuids)

    def flip_climits_for_layers(self, uuids=None):
        L = self.current_layer_set
        if uuids is not None:
            uuids = self.time_siblings_uuids(uuids)
        else:  # all data layers
            uuids = [pinfo.uuid for pinfo in L]

        nfo = {}
        for uuid in uuids:
            for dex,pinfo in enumerate(L):
                if pinfo.uuid == uuid:
                    nfo[uuid] = pinfo.climits[::-1]
                    L[dex] = pinfo._replace(climits=nfo[uuid])
        self.didChangeColorLimits.emit(nfo)

    def change_gamma_for_layers_where(self, gamma, **query):
        nfo = {}
        L = self.current_layer_set
        for idx, pz, layer in self.current_layers_where(**query):
            new_pz = pz._replace(gamma=gamma)
            nfo[layer.uuid] = new_pz.gamma
            L[idx] = new_pz
        self.didChangeGamma.emit(nfo)

    def change_gamma_for_siblings(self, uuid, gamma):
        uuids = self.time_siblings(uuid)[0]
        return self.change_gamma_for_layers_where(gamma, uuids=uuids)

    def create_algebraic_composite(self, operations, namespace, info=None, insert_before=0):
        if info is None:
            info = {}

        # Map a UUID's short name to the variable name in the namespace
        # Keep track of multiple ns variables being the same UUID
        short_name_to_ns_name = {}
        for k, u in namespace.items():
            sname = self[u][INFO.SHORT_NAME]
            short_name_to_ns_name.setdefault(sname, []).append(k)

        namespace_siblings = {k: self.time_siblings(u)[0] for k, u in namespace.items()}
        # go out of our way to make sure we make as many sibling layers as possible
        # even if one or more time steps are missing
        # NOTE: This does not handle if one product has a missing step and
        # another has a different missing time step
        time_master = max(namespace_siblings.values(), key=lambda v: len(v))
        for idx in range(len(time_master)):
            t = self[time_master[idx]][INFO.SCHED_TIME]
            channel_siblings = [(self[u][INFO.SHORT_NAME], u) for u in self.channel_siblings(time_master[idx])[0]]
            temp_namespace = {}
            for sn, u in channel_siblings:
                if sn not in short_name_to_ns_name:
                    continue
                # set each ns variable to this UUID
                for ns_name in short_name_to_ns_name[sn]:
                    temp_namespace[ns_name] = u
            if len(temp_namespace) != len(namespace):
                LOG.info("Missing some layers to create algebraic layer at {:%Y-%m-%d %H:%M:%S}".format(t))
                continue
            LOG.info("Creating algebraic layer '{}' for time {:%Y-%m-%d %H:%M:%S}".format(info.get(INFO.SHORT_NAME), self[time_master[idx]].get(INFO.SCHED_TIME)))

            uuid, layer_info, data = self._workspace.create_algebraic_composite(operations, temp_namespace, info.copy())
            self._layer_with_uuid[uuid] = dataset = DocBasicLayer(self, layer_info)
            presentation, reordered_indices = self._insert_layer_with_info(dataset, insert_before=insert_before)
            if INFO.UNIT_CONVERSION not in dataset:
                dataset[INFO.UNIT_CONVERSION] = units_conversion(dataset)
            if INFO.FAMILY not in dataset:
                dataset[INFO.FAMILY] = self._family_for_layer(dataset)
            self._add_layer_family(dataset)
            self.didAddCompositeLayer.emit(reordered_indices, dataset.uuid, presentation)

    def available_rgb_components(self):
        non_rgb_classes = [DocBasicLayer, DocCompositeLayer]
        valid_ranges = {}
        for layer in self.layers_where(is_valid=True, in_type_set=non_rgb_classes):
            sname = layer.get(INFO.CENTRAL_WAVELENGTH, layer[INFO.DATASET_NAME])
            valid_ranges.setdefault(sname, layer[INFO.VALID_RANGE])
        return valid_ranges

    def create_rgb_composite(self, r=None, g=None, b=None, clim=None, gamma=None):
        """Create an RGB recipe and the layers that can be created from it.

        Args:
            r (str): Family name to use for the Red component of the RGB
            g (str): Family name to use for the Green component of the RGB
            b (str): Family name to use for the Blue component of the RGB
            clim (tuple): 3-element tuple of min and max floats for each
                          component.
                          Example: ``((0.0, 1.2), (None, None), (273.15, 310.0))``
            gamma (tuple): 3-element tuple of Gamma value for each component

        Returns:
            Iterator of RGB layers created

        """
        # create a new recipe object
        recipe = CompositeRecipe.from_rgb(uuidgen(), r=r, g=g, b=b, color_limits=clim, gammas=gamma)
        self.recipe_manager.add_recipe(recipe)
        # recipe.name -> (sat, inst) -> {t: layer}
        self._recipe_layers[recipe.name] = {}
        self.update_rgb_composite_layers(recipe)
        return chain(*(x.values() for x in self._recipe_layers[recipe.name].values()))

    def change_rgb_recipe_components(self, recipe, **rgba):
        if recipe.read_only:
            raise ValueError("Recipe is read only, can't modify")
        for idx, channel in enumerate('rgba'):
            if channel not in rgba:
                continue
            new_comp_family = rgba[channel]
            recipe.input_ids[idx] = new_comp_family
        self.update_rgb_composite_layers(recipe, rgba=set(rgba.keys()))

    def _uuids_for_recipe(self, recipe, valid_only=True):
        prez_uuids = self.current_layer_uuid_order
        for inst_key, time_layers in self._recipe_layers[recipe.name].items():
            for t, rgb_layer in time_layers.items():
                u = rgb_layer[INFO.UUID]
                if not valid_only:
                    yield u
                elif u in prez_uuids:
                    # only provide UUIDs if the layer is valid and presentable
                    yield u

    def change_rgb_recipe_prez(self, recipe, climits=None, gamma=None, uuids=None):
        if uuids is None:
            uuids = list(self._uuids_for_recipe(recipe))
        if climits:
            # modify each element separately
            for i in range(3):
                recipe.color_limits[i] = climits[i]
            self.change_clims_for_layers_where(recipe.color_limits, uuids=uuids)
        if gamma:
            for i in range(3):
                recipe.gammas[i] = gamma[i]
            self.change_gamma_for_layers_where(recipe.gammas, uuids=uuids)

    def update_rgb_composite_layers(self, recipe, times=None, rgba='rgb'):
        """Analyze each RGB layer for `recipe` and update it is needed.

        Args:
            recipe (CompositeRecipe): Recipe whose layers will be updated
            times (iterable of datetimes): Limit analyzed layers by these
                                           times.
            rgba (iterable of strings): Limit updated RGB components to these
                                        components ('r', 'g', 'b', 'a').

        """
        # find all the layer combinations
        changed_uuids = []
        prez_uuids = self.current_layer_uuid_order
        for t, sat, inst, r, g, b in self._composite_layers(recipe, times=times, rgba=rgba):
            inst_key = (sat, inst)
            # (sat, inst) -> {time -> layer}
            layers = self._recipe_layers[recipe.name].setdefault(inst_key, {})
            # NOTE: combinations may be returned that don't match the recipe
            if t not in layers:
                # create a new blank RGB
                uuid = uuidgen()
                ds_info = {
                    INFO.UUID: uuid,
                    INFO.KIND: KIND.RGB,
                    "recipe": recipe,
                }
                # better place for this?
                ds_info[INFO.FAMILY] = self._family_for_layer(ds_info)
                LOG.debug("Creating new RGB layer for recipe '{}'".format(recipe.name))
                rgb_layer = layers[t] = DocRGBLayer(self, recipe, ds_info)
                self._layer_with_uuid[uuid] = rgb_layer
                layers[t].update_metadata_from_dependencies()
                # maybe we shouldn't add the family until the layers are set
                self._add_layer_family(rgb_layer)
                LOG.info('generating incomplete (invalid) composite for user to configure')
                # maybe we shouldn't send this out for invalid layers
            else:
                rgb_layer = layers[t]

            # only try to change the layers that were specified as being changed
            if rgba:
                changed_components = {comp: comp_layer for comp, comp_layer in zip('rgb', [r, g, b]) if comp in rgba}
            else:
                changed_components = {'r': r, 'g': g, 'b': b}

            if not changed_components:
                continue

            # update the component layers and tell which ones changed
            changed = self._change_rgb_component_layer(rgb_layer,
                                                       **changed_components)

            # check recipes color limits and update them
            # but only if this RGB layer matches the layers the recipe has
            if not recipe.read_only and changed and rgb_layer.recipe_layers_match:
                def_limits = {comp: rgb_layer[INFO.CLIM][idx] for idx, comp in enumerate('rgb') if comp in changed}
                recipe.set_default_color_limits(**def_limits)

            # only tell other components about this layer if it is valid
            should_show = rgb_layer.is_valid or rgb_layer.recipe_layers_match
            if rgb_layer[INFO.UUID] not in prez_uuids:
                if should_show:
                    presentation, reordered_indices = self._insert_layer_with_info(rgb_layer)
                    self.didAddCompositeLayer.emit(reordered_indices, rgb_layer[INFO.UUID], presentation)
                else:
                    continue
            elif not should_show:
                # is being shown, but shouldn't be
                self.remove_layer_prez(rgb_layer[INFO.UUID], purge=False)
                continue

            if rgb_layer is not None:
                changed_uuids.append(rgb_layer[INFO.UUID])

        self.change_rgb_recipe_prez(recipe, climits=recipe.color_limits,
                                    gamma=recipe.gammas, uuids=changed_uuids)
        if changed_uuids:
            # self.didChangeComposition.emit((), layer.uuid, prez, rgba)
            self.didChangeCompositions.emit((), changed_uuids,
                                            list(self.prez_for_uuids(changed_uuids)))

    def _composite_layers(self, recipe, times=None, rgba=None):
        if times:
            times = set(times)
        if rgba is None:
            rgba = []

        def _key_func(x):
            return x[INFO.PLATFORM], x[INFO.INSTRUMENT]

        def _component_generator(family, this_rgba):
            # limit what layers are returned
            if this_rgba not in rgba:
                return {}

            # use wavelength if it exists, use short name otherwise
            if family is None:
                layers = self.current_layers_where(kinds=[KIND.IMAGE, KIND.COMPOSITE])
                layers = (x[-1] for x in layers)
                # return empty `None` layers since we don't know what is wanted right now
                # we look at all possible times
                inst_layers = {k: {l[INFO.SCHED_TIME]: None for l in g} for k, g in groupby(sorted(layers, key=_key_func), _key_func)}
                return inst_layers
            else:
                family_uuids = self._families[family]
                family_layers = [self[u] for u in family_uuids]
            # (sat, inst) -> {time -> layer}
            inst_layers = {k: {l[INFO.SCHED_TIME]: l for l in g} for k, g in groupby(sorted(family_layers, key=_key_func), _key_func)}
            return inst_layers

        r_layers = _component_generator(recipe.input_ids[0], 'r')
        g_layers = _component_generator(recipe.input_ids[1], 'g')
        b_layers = _component_generator(recipe.input_ids[2], 'b')
        instruments = r_layers.keys() | g_layers.keys() | b_layers.keys()
        for inst_key in instruments:
            # any new times plus existing times if RGBs already exist
            rgb_times = r_layers.setdefault(inst_key, {}).keys() | \
                        g_layers.setdefault(inst_key, {}).keys() | \
                        b_layers.setdefault(inst_key, {}).keys() | \
                        self._recipe_layers[recipe.name].setdefault(inst_key, {}).keys()
            if times:
                rgb_times &= times
            # time order doesn't really matter
            for t in rgb_times:
                yield t, inst_key[0], inst_key[1],\
                      r_layers[inst_key].get(t), g_layers[inst_key].get(t),\
                      b_layers[inst_key].get(t)

    def sync_composite_layer_prereqs(self, new_times):
        """Check if we can make more RGBs based on newly added times"""
        # add a blank layer object if we've never seen this time before
        # update the layer object with newly available layers if possible
        # add the layer object to the document if it should be included with
        # the rest of them
        for recipe_name, inst_rgbs in self._recipe_layers.items():
            recipe = self.recipe_manager[recipe_name]
            self.update_rgb_composite_layers(recipe, times=new_times)

    def _change_rgb_component_layer(self, layer:DocRGBLayer, **rgba):
        """Update RGB Layer with specified components

        If the layer is not valid

        change the layer composition for an RGB layer, and signal
        by default, propagate the changes to sibling layers matching this layer's configuration
        """
        LOG.debug('revising RGB layer config for %s: %s' % (layer.uuid, repr(list(rgba.keys()))))
        if layer is None or not rgba:
            return
        # identify siblings before we make any changes!
        changed = []
        clims = list(layer[INFO.CLIM])
        for k, v in rgba.items():
            # assert(k in 'rgba')
            idx = 'rgba'.index(k)
            if getattr(layer, k, None) is v:
                continue
            changed.append(k)
            setattr(layer, k, v)
            clims[idx] = None  # causes update_metadata to pull in upstream clim values
        if not changed:
            return changed
        # force an update of clims for components that changed
        # These clims are the current state of the default clims for each sub-layer
        layer[INFO.CLIM] = tuple(clims)
        updated = layer.update_metadata_from_dependencies()
        LOG.info('updated metadata for layer %s: %s' % (layer.uuid, repr(list(updated.keys()))))
        return changed

    def set_rgb_range(self, recipe:CompositeRecipe, rgba:str, min:float, max:float):
        new_clims = tuple(x if c != rgba else (min, max) for c, x in zip("rgba", recipe.color_limits))
        # update the ranges on this layer and all it's siblings
        self.change_rgb_recipe_prez(recipe, climits=new_clims)

    def _directory_of_layers(self, kind=KIND.IMAGE):
        if not isinstance(kind, (list, tuple)):
            kind = [kind]
        for x in [q for q in self._layer_with_uuid.values() if q.kind in kind]:
            yield x.uuid, x.sched_time, x.product_family_key

    def __len__(self):
        # FIXME: this should be consistent with __getitem__, not self.current_layer_set
        return len(self.current_layer_set)

    def uuid_for_current_layer(self, row):
        uuid = self.current_layer_set[row].uuid
        return uuid

    def remove_layers_from_all_sets(self, uuids):
        for uuid in list(uuids):
            # FUTURE: make this removal of presentation tuples from inactive layer sets less sucky
            LOG.debug('removing {}'.format(uuid))
            for dex,layer_set in enumerate(self._layer_sets):
                if dex==self.current_set_index or layer_set is None:
                    continue
                for pdex, presentation in enumerate(layer_set):
                    if presentation.uuid==uuid:
                        del layer_set[pdex]
                        break
            # remove from available family layers
            self._remove_layer_from_family(uuid)
            # now remove from the active layer set
            self.remove_layer_prez(uuid)  # this will send signal and start purge

    def animate_siblings_of_layer(self, row_or_uuid):
        uuid = self.current_layer_set[row_or_uuid].uuid if not isinstance(row_or_uuid, UUID) else row_or_uuid
        layer = self._layer_with_uuid[uuid]
        if isinstance(layer, DocRGBLayer):
            return self.loop_rgb_layers_following(layer.uuid)
        new_anim_uuids, _ = self.time_siblings(uuid)
        if new_anim_uuids is None or len(new_anim_uuids)<2:
            LOG.info('no time siblings to chosen band, will try channel siblings to chosen time')
            new_anim_uuids, _ = self.channel_siblings(uuid)
        if new_anim_uuids is None or len(new_anim_uuids)<2:
            LOG.warning('No animation found')
            return []
        LOG.debug('new animation order will be {0!r:s}'.format(new_anim_uuids))
        L = self.current_layer_set
        L.animation_order = new_anim_uuids
        # L.clear_animation_order()
        # for dex,u in enumerate(new_anim_uuids):
        #     LOG.debug(u)
        #     row = L.uuid2row.get(u, None)
        #     if row is None:
        #         LOG.error('unable to find row for uuid {} in current layer set'.format(u))
        #         continue
        #     old = L[row]
        #     new = old._replace(a_order=dex)
        #     L[row] = new
        self.didReorderAnimation.emit(tuple(new_anim_uuids))
        return new_anim_uuids

    def get_info(self, row=None, uuid=None):
        if row is not None:
            uuid_temp = self.current_layer_set[row].uuid
            nfo = self._layer_with_uuid[uuid_temp]
            return nfo
        elif uuid is not None:
            nfo = self._layer_with_uuid[uuid]
            return nfo
        return None

    def get_algebraic_namespace(self, uuid):
        return self._workspace.get_algebraic_namespace(uuid)

    def __getitem__(self, layer_uuid):
        """
        return layer with the given UUID
        """
        if layer_uuid is None:
            raise KeyError("Key 'None' does not exist in document or workspace")
        elif not isinstance(layer_uuid, UUID):
            raise ValueError('document[UUID] required, %r was used' % type(layer_uuid))

        if layer_uuid in self._layer_with_uuid:
            return self._layer_with_uuid[layer_uuid]

        # check the workspace for information
        try:
            LOG.debug("Checking workspace for information on inactive product")
            info = self._workspace.get_info(layer_uuid)
        except KeyError:
            info = None

        if info is None:
            raise KeyError("Key '{}' does not exist in document or workspace".format(layer_uuid))
        return info

    def reorder_by_indices(self, new_order, uuids=None, layer_set_index=None):
        """given a new layer order, replace the current layer set
        emits signal to other subsystems
        """
        if layer_set_index is None:
            layer_set_index = self.current_set_index
        assert(len(new_order)==len(self._layer_sets[layer_set_index]))
        new_layer_set = DocLayerStack(self, [self._layer_sets[layer_set_index][n] for n in new_order])
        self._layer_sets[layer_set_index] = new_layer_set
        self.didReorderLayers.emit(tuple(new_order))

    def insert_layer_prez(self, row:int, layer_prez_seq):
        cls = self.current_layer_set
        clo = list(range(len(cls)))
        lps = list(layer_prez_seq)
        lps.reverse()
        if not lps:
            LOG.warning('attempt to drop empty content')
            return
        for p in lps:
            if not isinstance(p, prez):
                LOG.error('attempt to drop a new layer with the wrong type: {0!r:s}'.format(p))
                continue
            cls.insert(row, p)
            clo.insert(row, None)

    def is_using(self, uuid:UUID, layer_set:int=None):
        "return true if this dataset is still in use in one of the layer sets"
        layer = self._layer_with_uuid[uuid]
        if layer_set is not None:
            lss = [self._layer_sets[layer_set]]
        else:
            lss = [q for q in self._layer_sets if q is not None]
        for ls in lss:
            for p in ls:
                if p.uuid == uuid:
                    return True
                parent_layer = self._layer_with_uuid[p.uuid]
                if parent_layer.kind == KIND.RGB and layer in parent_layer.l:
                    return True
        return False

    def remove_layer_prez(self, row_or_uuid, count:int=1, purge=True):
        """
        remove the presentation of a given layer/s in the current set
        :param row: which current layer set row to remove
        :param count: how many rows to remove
        :return:
        """
        if isinstance(row_or_uuid, UUID) and count==1:
            row = self.row_for_uuid(row_or_uuid)
            uuids = [row_or_uuid]
        else:
            row = row_or_uuid
            uuids = [x.uuid for x in self.current_layer_set[row:row+count]]
        self.toggle_layer_visibility(list(range(row, row+count)), False)
        clo = list(range(len(self.current_layer_set)))
        del clo[row:row+count]
        del self.current_layer_set[row:row+count]
        self.didRemoveLayers.emit(tuple(clo), uuids, row, count)
        # Remove this layer from any RGBs it is a part of
        for uuid in uuids:
            layer = self._layer_with_uuid[uuid]
            self.sync_composite_layer_prereqs([layer[INFO.SCHED_TIME]])

        # Purge this layer if we can
        if purge:
            for uuid in uuids:
                if not self.is_using(uuid):
                    LOG.info('purging layer {}, no longer in use'.format(uuid))
                    self.willPurgeLayer.emit(uuid)
                    # remove from our bookkeeping
                    del self._layer_with_uuid[uuid]
                    # remove from workspace
                    self._workspace.remove(uuid)

    def channel_siblings(self, uuid, sibling_infos=None):
        """
        filter document info to just dataset of the same channels
        meaning all channels at a specific time, in alphabetical name order
        :param uuid: focus UUID we're trying to build around
        :param sibling_infos: dictionary of UUID -> Dataset Info to sort through
        :return: sorted list of sibling uuids in channel order
        """
        if sibling_infos is None:
            sibling_infos = self._layer_with_uuid
        it = sibling_infos.get(uuid, None)
        if it is None:
            return None
        sibs = [(x[INFO.SHORT_NAME], x[INFO.UUID]) for x in
                self._filter(sibling_infos.values(), it, {INFO.SCENE, INFO.SCHED_TIME, INFO.INSTRUMENT, INFO.PLATFORM})]
        # then sort it by bands
        sibs.sort()
        offset = [i for i, x in enumerate(sibs) if x[1] == uuid]
        return [x[1] for x in sibs], offset[0]

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

    def time_siblings(self, uuid, sibling_infos=None):
        """
        return time-ordered list of datasets which have the same band, in time order
        :param uuid: focus UUID we're trying to build around
        :param sibling_infos: dictionary of UUID -> Dataset Info to sort through
        :return: sorted list of sibling uuids in time order, index of where uuid is in the list
        """
        if sibling_infos is None:
            sibling_infos = self._layer_with_uuid
        it = sibling_infos.get(uuid, None)
        if it is None:
            return [], 0
        sibs = [(x[INFO.SCHED_TIME], x[INFO.UUID]) for x in
                self._filter(sibling_infos.values(), it, {INFO.SHORT_NAME, INFO.STANDARD_NAME, INFO.SCENE, INFO.INSTRUMENT, INFO.PLATFORM, INFO.KIND})]
        # then sort it into time order
        sibs.sort()
        offset = [i for i,x in enumerate(sibs) if x[1]==uuid]
        return [x[1] for x in sibs], offset[0]

    def time_siblings_uuids(self, uuids, sibling_infos=None):
        """
        return generator uuids for datasets which have the same band as the uuids provided
        :param uuids: iterable of uuids
        :param infos: list of dataset infos available, some of which may not be relevant
        :return: generate sorted list of sibling uuids in time order and in provided uuid order
        """
        for requested_uuid in uuids:
            for sibling_uuid in self.time_siblings(requested_uuid, sibling_infos=sibling_infos)[0]:
                yield sibling_uuid



#
# class DocumentTreeBranch(QObject):
#     pass
#
# class DocumentTreeLeaf(QObject):
#     pass
#
#
# class DocumentAsLayerTree(QObject):
#     """
#      DocumentAsLayerTree is a facet or wrapper (if it were a database, it would be a view; but view is already taken)
#      It allows the layer controls - specifically a LayerStackTreeViewModel - to easily access and modify
#      the document on behalf of the user.
#      It includes both queries for display and changes which then turn into document updates
#      The base model is just a list of basic layers.
#      Composite and Algebraic layers, however, are more like folders.
#      Other additional layer types may also have different responses to being dragged or having items dropped on them
#     """
#
#     def __init__(self, doc, *args, **kwargs):
#         self._doc = doc
#         super(DocumentAsLayerTree, self).__init__()
#
#     def
#


def main():
    parser = argparse.ArgumentParser(
        description="PURPOSE",
        epilog="",
        fromfile_prefix_chars='@')
    parser.add_argument('-v', '--verbose', dest='verbosity', action="count", default=0,
                        help='each occurrence increases verbosity 1 level through ERROR-WARNING-INFO-DEBUG')
    # http://docs.python.org/2.7/library/argparse.html#nargs
    # parser.add_argument('--stuff', nargs='5', dest='my_stuff',
    #                    help="one or more random things")
    parser.add_argument('pos_args', nargs='*',
                        help="positional arguments don't have the '-' prefix")
    args = parser.parse_args()

    levels = [logging.ERROR, logging.WARN, logging.INFO, logging.DEBUG]
    logging.basicConfig(level=levels[min(3, args.verbosity)])

    if not args.pos_args:
        unittest.main()
        return 0

    for pn in args.pos_args:
        pass

    return 0


if __name__ == '__main__':
    sys.exit(main())


