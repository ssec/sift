#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
uwsift.model.document
---------------------

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

All entities in the Document have a UUID that is their identity throughout their lifecycle,
and is often used as shorthand between subsystems. Document rarely deals directly with content.

:author: R.K.Garcia <rayg@ssec.wisc.edu>
:copyright: 2015 by University of Wisconsin Regents, see AUTHORS for more details
:license: GPLv3, see LICENSE for more details
"""
from __future__ import annotations

__author__ = "rayg"
__docformat__ = "reStructuredText"

import dataclasses
import json
import logging
import os
import typing as typ
import warnings
from collections import defaultdict
from collections.abc import MutableSequence
from datetime import datetime, timedelta
from uuid import UUID
from weakref import ref

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal

from uwsift.common import FCS_SEP, Flags, Info, Kind, Presentation, Span, ZList
from uwsift.model.area_definitions_manager import AreaDefinitionsManager
from uwsift.model.layer import (
    DocBasicDataset,
    DocDataset,
    DocRGBDataset,
    Mixing,
)
from uwsift.queue import TASK_DOING, TASK_PROGRESS, TaskQueue
from uwsift.util.common import units_conversion
from uwsift.util.default_paths import DOCUMENT_SETTINGS_DIR
from uwsift.view.colormap import (
    COLORMAP_MANAGER,
    SITE_CATEGORY,
    USER_CATEGORY,
    PyQtGraphColormap,
)
from uwsift.workspace import BaseWorkspace, CachingWorkspace, SimpleWorkspace
from uwsift.workspace.metadatabase import Product
from uwsift.workspace.workspace import frozendict

LOG = logging.getLogger(__name__)

DEFAULT_LAYER_SET_COUNT = 1  # this should match the ui configuration!


class DocLayerStack(MutableSequence):
    """list-like layer set which will slowly eat functionality from Document as warranted

    Provide cleaner interfacing to GUI elements.

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
            raise ValueError("cannot initialize DocLayerStack using %s" % type(doc))

    def __setitem__(self, index: int, value: Presentation):
        if index >= 0 and index < len(self._store):
            self._store[index] = value
        elif index == len(self._store):
            self._store.append(value)
        else:
            raise IndexError("%d not a valid index" % index)
        self._u2r = None

    @property
    def uuid2row(self):
        if self._u2r is None:
            self._u2r = dict((p.uuid, i) for (i, p) in enumerate(self._store))
        return self._u2r

    def __getitem__(self, index: int):  # then return layer object
        if isinstance(index, int):
            return self._store[index]
        elif isinstance(index, UUID):  # then return 0..n-1 index in stack
            return self.uuid2row.get(index, None)
        elif isinstance(index, DocDataset):
            return self.uuid2row.get(index.uuid, None)
        elif isinstance(index, Presentation):
            return self.uuid2row.get(index.uuid, None)
        else:
            raise ValueError("unable to index LayerStack using %s" % repr(index))

    def __iter__(self):
        for each in self._store:
            yield each

    def __len__(self):
        return len(self._store)

    def __delitem__(self, index: int):
        del self._store[index]
        self._u2r = None

    def insert(self, index: int, value: Presentation):
        self._store.insert(index, value)
        self._u2r = None

    def clear_animation_order(self):
        for i, q in enumerate(self._store):
            self._store[i] = dataclasses.replace(q, a_order=None)

    def index(self, uuid):
        assert isinstance(uuid, UUID)
        u2r = self.uuid2row
        return u2r.get(uuid, None)

    @property
    def animation_order(self):
        aouu = [(x.a_order, x.uuid) for x in self._store if (x.a_order is not None)]
        aouu.sort()
        ao = tuple(u for a, u in aouu)
        LOG.debug("animation order is {0!r:s}".format(ao))
        return ao

    @animation_order.setter
    def animation_order(self, layer_or_uuid_seq):
        self.clear_animation_order()
        for nth, lu in enumerate(layer_or_uuid_seq):
            try:
                idx = self[lu]
            except ValueError:
                LOG.warning("unable to find layer in LayerStack")
                raise
            self._store[idx] = dataclasses.replace(self._store[idx], a_order=nth)

###################################################################################################################


class DataLayer:
    """
    A DataLayer is unique to a satellite, channel and calibration
    over a certain period of time.

    Represents images over time belonging to a certain (satellite, channel, calibration)
                            for a given span of time
    """

    def __init__(self, uuids_timestamps: typ.List, product_family_key: typ.Tuple):
        # metadata redundant w.r.t. satellite, channel, calibration, so any metadata object
        #                           from the list of tuples will do
        self._t_matched: typ.Optional[datetime] = None
        self.product_family_key = product_family_key

        # sort timestamps while keeping them synchronous with images
        uuids_tstamps_asc = sorted(uuids_timestamps, key=lambda tup: tup[1])
        # Create timeline according to temporal resolution

        self.timeline = {uuid_dt_tup[1]: uuid_dt_tup[0] for uuid_dt_tup in uuids_tstamps_asc}

    @property
    def t_matched(self):
        return self._t_matched

    @t_matched.setter
    def t_matched(self, t_matched):
        if not isinstance(t_matched, datetime):
            raise ValueError("Matched time needs to be datetime object.")
        self._t_matched = t_matched

    @property
    def satellite(self):
        return self.product_family_key[0].value

    @satellite.setter
    def satellite(self, *args, **kwargs):
        raise ValueError("Changing type of a DataLayer forbidden.")

    @property
    def product_str(self):
        return self.product_family_key[2]

    @product_str.setter
    def product_str(self, *args, **kwargs):
        raise ValueError("Changing type of a DataLayer forbidden.")


class DataLayerCollection(QObject):
    """
    Collection of data layers (representation of time series of images) with one data layer
    as driving layer, designated by its ProductFamilyKey.
    """

    didUpdateCollection = pyqtSignal()

    def __init__(self, data_layers: typ.List[DataLayer]):
        super().__init__()
        # dict mapping product family keys to data layers
        self.data_layers = {}
        # Convenience functions must return a data layer. This is used by caller code to i.e. set a
        # new driving layer either according to some policy or an index
        self.convenience_functions: typ.Dict[str, typ.Callable] = {
            "Most Frequent": self.get_most_frequent_data_layer_index
        }
        self.product_family_keys = set()
        if data_layers:
            self.notify_update_collection(data_layers)

    def get_most_frequent_data_layer_index(self) -> int:
        """
        Get index of the data layer with the least mean difference between timestamps.
        :return: -1 if no data layers exist, index of most frequent data layer otherwise.
        """
        num_data_layers = len(self.data_layers.values())
        if num_data_layers == 0:
            return -1
        else:
            temporal_differences = np.zeros(num_data_layers)
            for i, data_layer in enumerate(self.data_layers.values()):
                tl = list(data_layer.timeline.keys())
                t_diffs = [tl[i + 1] - tl[i] for i in range(len(tl) - 1)]
                # Calculate mean difference in timeline in seconds to support
                # comparing timelines with non-regularly occurring timestamps
                temporal_differences[i] = np.mean(list(map(lambda td: td.total_seconds(), t_diffs)))
            most_frequent_data_layer_idx = np.argmin(temporal_differences)
            if not isinstance(most_frequent_data_layer_idx, np.int64):
                # If there exist multiple timelines at the same sampling rate, take the first one.
                most_frequent_data_layer_idx = most_frequent_data_layer_idx[0]
            return most_frequent_data_layer_idx

    @property
    def num_data_layers(self):
        return len(list(self.data_layers.values()))

    def notify_update_collection(self, new_data_layers: typ.List[DataLayer]):
        """Update state of collection when new data is being loaded or old data is discarded.

        Notifies dependents via Qt signal.

        Two cases need to be taken into account:
            1) entirely new data layer as shown by previously unknown product family key
            2) data layer with same pfkey already exists
               -> add new timesteps without duplication

        """
        for data_layer in new_data_layers:

            if data_layer.product_family_key in self.product_family_keys:
                old_data_layer = self.data_layers[data_layer.product_family_key]
                # Timeline maps: timestamp -> uuid
                for tstamp, uuid in data_layer.timeline.items():
                    if uuid not in old_data_layer.timeline.values():
                        old_data_layer.timeline[tstamp] = uuid
                self.data_layers[data_layer.product_family_key].timeline = self._sort_timeline(old_data_layer.timeline)
            else:
                self.product_family_keys.add(data_layer.product_family_key)
                self.data_layers[data_layer.product_family_key] = data_layer
        self.didUpdateCollection.emit()

    @staticmethod
    def _sort_timeline(timeline):
        """
        Convenience method to sort timelines of data layers by ordered
        ascending by their timestamps.
        :param timeline: Dict mapping datetime objects to uuids
        :return: Sorted timeline dict
        """
        sorted_tl = sorted(timeline.items(), key=lambda kv: kv[0])
        return {k: v for k, v in sorted_tl}

    def remove_by_uuid(self, uuid_to_remove: UUID):
        for _, data_layer in self.data_layers.items():
            new_timeline = data_layer.timeline.copy()
            for dt, uuid in data_layer.timeline.items():
                if uuid_to_remove == uuid:
                    del new_timeline[dt]
                    # NOTE(mk): Able to exit prematurely here if guaranteed that
                    # no uuid present in multiple data layers
            data_layer.timeline = self._sort_timeline(new_timeline)
        self.didUpdateCollection.emit()


class Document(QObject):  # base class is rightmost, mixins left of that
    """Storage for layer and user information.

    Document is a set of tracks in a Z order, with Z>=0 for "active" tracks the user is working with
    Tracks with Z-order <0 are inactive, but may be displayed in the timeline as potentials for the
    user to drag to active
    Document has a playhead, a playback time range, an active timeline display range
    Tracks and frames (aka Products) can have state information set

    This is the low-level "internal" interface that acts as a signaling hub.
    Direct access to the document is being deprecated.
    Most direct access patterns should be migrated to using a contextual view of the document,
    in order to reduce abstraction leakage and permit the document storage to evolve.
    """

    config_dir: str = None
    queue: TaskQueue = None
    _workspace: BaseWorkspace = None

    # timeline the user has specified:
    track_order: ZList = None  # (zorder, family-name) with higher z above lower z; z<0 should not occur

    # overall visible range of the active data if specified by user,
    # else None means assume use the product timespan from metadatabase
    timeline_span: Span = None

    # playback information
    playhead_time: datetime = None  # document stored playhead time
    playback_per_sec: timedelta = timedelta(seconds=60.0)  # data time increment per wall-second

    # playback time range, if not None is a subset of overall timeline
    playback_span: Span = None

    # user-directed overrides on tracks and frames (products)
    track_state: typ.Mapping[str, Flags] = None
    product_state: typ.Mapping[UUID, Flags] = None

    # user can lock tracks to a single frame throughout
    track_frame_locks: typ.Mapping[str, UUID] = None

    # Maps of family names to their document recipes
    family_presentation: typ.Mapping[str, Presentation] = None
    family_calculation: typ.Mapping[str, object] = None  # algebraic combinations of multiple products

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
    _layer_sets = None  # list(DocLayerSet(Presentation, ...) or None)
    _layer_with_uuid = None  # dict(uuid:Doc____Layer)

    # signals
    # Clarification: Layer interfaces migrate to layer meaning "current active products under the playhead"
    # new order list with None for new layer; info-dictionary, overview-content-ndarray
    didAddDataset = pyqtSignal(frozendict, Presentation)
    didUpdateBasicDataset = pyqtSignal(UUID, Kind)
    # comp layer is derived from multiple basic layers and has its own UUID
    didAddCompositeDataset = pyqtSignal(tuple, UUID, Presentation)
    # new order, UUIDs that were removed from current layer set, first row removed, num rows removed
    didRemoveDatasets = pyqtSignal(tuple, list, int, int)
    willPurgeDataset = pyqtSignal(UUID)  # UUID of the layer being removed
    didReorderDatasets = pyqtSignal(tuple)  # list of original indices in their new order, None for new layers
    didChangeLayerVisibility = pyqtSignal(dict)  # {UUID: new-visibility, ...} for changed layers
    didReorderAnimation = pyqtSignal(tuple)  # list of UUIDs representing new animation order
    didChangeLayerName = pyqtSignal(UUID, str)  # layer uuid, new name
    didChangeColormap = pyqtSignal(dict)  # dict of {uuid: colormap-name-or-UUID, ...} for all changed layers
    didChangeColorLimits = pyqtSignal(dict)  # dict of {uuid: (vmin, vmax), ...} for all changed layers
    didChangeProjection = pyqtSignal(str)  # name of projection (area definition)
    didAddFamily = pyqtSignal(str, dict)  # name of the newly added family and dict of family info
    didRemoveFamily = pyqtSignal(str)  # name of the newly added family and dict of family info
    didReorderTracks = pyqtSignal(set, set)  # added track names, removed track names
    didChangeImageKind = pyqtSignal(dict)

    def __init__(
        self,
        workspace: BaseWorkspace,
        queue,
        config_dir=DOCUMENT_SETTINGS_DIR,
        layer_set_count=DEFAULT_LAYER_SET_COUNT,
        **kwargs,
    ):
        super(Document, self).__init__(**kwargs)
        self.config_dir = config_dir
        self.queue = queue
        if not os.path.isdir(self.config_dir):
            LOG.debug("Creating settings directory {}".format(self.config_dir))
            os.makedirs(self.config_dir)

        self._workspace = workspace
        self._layer_sets = [DocLayerStack(self)] + [None] * (layer_set_count - 1)
        self._layer_with_uuid = {}

        # data_layers represents time series of images ordered by product family
        self.data_layers: typ.List[DataLayer] = []
        self.data_layer_collection: DataLayerCollection = DataLayerCollection([])

        self.colormaps = COLORMAP_MANAGER
        self.default_area_def_name = AreaDefinitionsManager.default_area_def_name()
        self.current_area_def_name = self.default_area_def_name
        # HACK: This should probably be part of the metadata database in the future
        self._families = defaultdict(list)

        # Create directory if it does not exist
        cmap_base_dir = os.path.join(self.config_dir, "colormaps")
        read_cmap_dir = os.path.join(cmap_base_dir, "site")  # read-only
        write_cmap_dir = os.path.join(cmap_base_dir, "user")  # writeable
        self.read_cmap_dir = read_cmap_dir
        self.write_cmap_dir = write_cmap_dir
        importable_cmap_cats = [(True, SITE_CATEGORY, read_cmap_dir), (False, USER_CATEGORY, write_cmap_dir)]
        for read_only, cmap_cat, cmap_dir in importable_cmap_cats:
            if not os.path.exists(cmap_dir):
                os.makedirs(cmap_dir)
            else:
                self.colormaps.import_colormaps(cmap_dir, read_only=read_only, category=cmap_cat)

        # timeline document storage setup with initial track order and time range
        self.product_state = defaultdict(Flags)
        self.track_state = defaultdict(Flags)
        self.track_order = ZList()
        self.track_frame_locks = {}
        self.family_calculation = {}
        self.family_composition = {}
        self.family_presentation = {}

        # scan available metadata for initial state
        # FIXME: refresh this once background scan finishes and new products are found
        # self.timeline_span = self.playback_span = self.potential_product_span()
        if isinstance(self._workspace, CachingWorkspace):
            self.sync_potential_tracks_from_metadata()

    def potential_tracks(self) -> typ.List[str]:
        """List the names of available tracks (both active and potential) according to the metadatabase"""
        with self._workspace.metadatabase as S:
            return list((f + FCS_SEP + c) for (f, c) in S.query(Product.family, Product.category).distinct())

    def sync_potential_tracks_from_metadata(self):
        """update track_order to include any newly available tracks"""
        all_tracks = list(self.potential_tracks())
        all_tracks.sort()
        old_tracks = set(name for z, name in self.track_order.items())
        for track in all_tracks:
            self.track_order.append(track, start_negative=True, not_if_present=True)
        for dismissed in old_tracks - set(all_tracks):
            LOG.debug("removing track {} from track_order".format(dismissed))
            self.track_order.remove(dismissed)
        new_tracks = set(name for z, name in self.track_order.items())
        if old_tracks != new_tracks:
            LOG.info("went from {} available tracks to {}".format(len(old_tracks), len(new_tracks)))
            self.didReorderTracks.emit(new_tracks - old_tracks, old_tracks - new_tracks)

    def find_colormap(self, colormap):
        if isinstance(colormap, str) and colormap in self.colormaps:
            colormap = self.colormaps[colormap]
        return colormap

    def area_definition(self, area_definition_name=None):
        return AreaDefinitionsManager.area_def_by_name(area_definition_name or self.current_area_def_name)

    def change_projection(self, area_def_name=None):
        if area_def_name is None:
            area_def_name = self.default_area_def_name
        assert area_def_name in AreaDefinitionsManager.available_area_def_names()
        if area_def_name != self.current_area_def_name:
            LOG.debug(
                f"Changing projection (area definition) from" f" '{self.current_area_def_name}' to '{area_def_name}'"
            )
            self.current_area_def_name = area_def_name
            self.didChangeProjection.emit(self.current_area_def_name)

    def update_user_colormap(self, colormap, name):
        # Update new gradient into save location
        try:
            filepath = self.write_cmap_dir
            cmap_file = open(os.path.join(filepath, name + ".json"), "w")
            cmap_file.write(json.dumps(colormap, indent=2, sort_keys=True))
            cmap_file.close()
        except IOError:
            LOG.error("Error saving gradient: {}".format(name), exc_info=True)

        cmap = PyQtGraphColormap(colormap)
        self.colormaps[name] = cmap

        # Update live map
        uuids = [p.uuid for _, p, _ in self.current_layers_where(colormaps=[name])]
        self.change_colormap_for_layers(name, uuids)

    def remove_user_colormap(self, name):
        try:
            os.remove(os.path.join(self.config_dir, "colormaps", "user", name + ".json"))
        except OSError:
            pass

        del self.colormaps[name]

    def current_projection_index(self):
        return list(AreaDefinitionsManager.available_area_def_names()).index(self.current_area_def_name)

    def change_projection_index(self, idx):
        return self.change_projection(tuple(AreaDefinitionsManager.available_area_def_names())[idx])

    @property
    def current_layer_set(self):
        cls = self._layer_sets[self.current_set_index]
        assert isinstance(cls, DocLayerStack)
        return cls

    def _insert_layer_with_info(self, info: DocDataset, cmap=None, style=None, insert_before=0):
        """
        insert a layer into the presentations but do not signal
        :return: new Presentation tuple, new reordered indices tuple
        """
        if cmap is None:
            cmap = info.get(Info.COLORMAP)
        if style is None:
            style = info.get(Info.STYLE)
        gamma = 1.0
        if isinstance(info, DocRGBDataset):
            gamma = (1.0,) * 3
        elif hasattr(info, "layers"):
            gamma = (1.0,) * len(info.layers)

        # get the presentation for another layer in our family
        family_uuids = self.family_uuids(info[Info.FAMILY])
        family_prez = self.prez_for_uuid(family_uuids[0]) if family_uuids else None
        p = Presentation(
            uuid=info[Info.UUID],
            kind=info[Info.KIND],
            visible=True,
            a_order=None,
            colormap=cmap if family_prez is None else family_prez.colormap,
            style=style if family_prez is None else family_prez.style,
            climits=info[Info.CLIM] if family_prez is None else family_prez.climits,
            gamma=gamma if family_prez is None else family_prez.gamma,
            mixing=Mixing.NORMAL,
            opacity=1.0,
        )

        q = dataclasses.replace(p, visible=False)  # make it available but not visible in other layer sets
        old_layer_count = len(self._layer_sets[self.current_set_index])
        for dex, lset in enumerate(self._layer_sets):
            if lset is not None:  # uninitialized layer sets will be None
                lset.insert(insert_before, p if dex == self.current_set_index else q)

        reordered_indices = tuple(
            [None] + list(range(old_layer_count))
        )  # FIXME: this should obey insert_before, currently assumes always insert at top
        return p, reordered_indices

    def activate_product_uuid_as_new_dataset(self, uuid: UUID, insert_before=0, **importer_kwargs):
        if uuid in self._layer_with_uuid:
            LOG.debug("Layer already loaded: {}".format(uuid))
            active_content_data = self._workspace.import_product_content(uuid, **importer_kwargs)
            return uuid, self[uuid], active_content_data

        # FUTURE: Load this async, the slots for the below signal need to be OK
        # with that
        active_content_data = self._workspace.import_product_content(uuid, **importer_kwargs)
        # updated metadata with content information (most importantly navigation
        # information)
        info = self._workspace.get_info(uuid)
        assert info is not None
        LOG.debug("cell_width: {}".format(repr(info[Info.CELL_WIDTH])))

        LOG.debug("new layer info: {}".format(repr(info)))
        self._layer_with_uuid[uuid] = dataset = DocBasicDataset(self, info)
        if Info.UNIT_CONVERSION not in info:
            dataset[Info.UNIT_CONVERSION] = units_conversion(dataset)
        if Info.FAMILY not in dataset:
            dataset[Info.FAMILY] = self.family_for_product_or_layer(dataset)
        presentation, reordered_indices = self._insert_layer_with_info(dataset, insert_before=insert_before)

        # signal updates from the document
        self.didAddDataset.emit(info, presentation)

        return uuid, dataset, active_content_data

    def family_for_product_or_layer(self, uuid_or_layer):
        if isinstance(uuid_or_layer, UUID):
            if isinstance(self._workspace, CachingWorkspace):
                with self._workspace.metadatabase as s:
                    fam = s.query(Product.family).filter_by(uuid_str=str(uuid_or_layer)).first()
            if isinstance(self._workspace, SimpleWorkspace):
                fam = self._workspace.get_info(uuid_or_layer)[Info.FAMILY]
            if fam:
                return fam[0]
            uuid_or_layer = self[uuid_or_layer]
        if Info.FAMILY in uuid_or_layer:
            LOG.debug("using pre-existing family {}".format(uuid_or_layer[Info.FAMILY]))
            return uuid_or_layer[Info.FAMILY]
        # kind:pointofreference:measurement:wavelength
        kind = uuid_or_layer[Info.KIND]
        refpoint = "unknown"  # FUTURE: geo/leo
        measurement = uuid_or_layer.get(Info.STANDARD_NAME)
        if uuid_or_layer.get("recipe"):
            # RGB
            subcat = uuid_or_layer["recipe"].name
        elif uuid_or_layer.get(Info.CENTRAL_WAVELENGTH):
            # basic band
            subcat = uuid_or_layer[Info.CENTRAL_WAVELENGTH]
        else:
            # higher level product or algebraic layer
            subcat = uuid_or_layer[Info.DATASET_NAME]
        return "{}:{}:{}:{}".format(kind.name, refpoint, measurement, subcat)

    def _add_layer_family(self, layer):
        family = layer[Info.FAMILY]
        is_new = family not in self._families or not len(self._families[family])
        self._families[family].append(layer[Info.UUID])
        if is_new:
            self.didAddFamily.emit(family, self.family_info(family))
        return family

    def _remove_layer_from_family(self, uuid):
        family = self[uuid][Info.FAMILY]
        self._families[family].remove(uuid)

        if not self._families[family]:
            # remove the family entirely if it is empty
            LOG.debug("Removing empty family: {}".format(family))
            del self._families[family]
            self.didRemoveFamily.emit(family)

    def family_info(self, family_or_layer_or_uuid):
        family = layer = family_or_layer_or_uuid
        if isinstance(family_or_layer_or_uuid, UUID):
            layer = self[family_or_layer_or_uuid]
        if isinstance(layer, DocBasicDataset):
            family = layer[Info.FAMILY]

        # one layer that represents all the layers in this family
        family_rep = self[self._families[family][0]]

        family_name_components = family.split(":")
        display_family = family_rep[Info.SHORT_NAME] + " " + " ".join(reversed(family_name_components))

        # NOTE: For RGBs the SHORT_NAME will update as the RGB changes
        return {
            Info.VALID_RANGE: family_rep[Info.VALID_RANGE],
            Info.UNIT_CONVERSION: family_rep[Info.UNIT_CONVERSION],
            Info.SHORT_NAME: family_rep[Info.SHORT_NAME],
            Info.UNITS: family_rep[Info.UNITS],
            Info.KIND: family_rep[Info.KIND],
            Info.DISPLAY_FAMILY: display_family,
        }

    def import_files(self, paths, insert_before=0, **importer_kwargs) -> dict:
        """Load product metadata and content from provided file paths.

        :param paths: paths to open
        :param insert_before: where to insert them in layer list
        :return:

        """

        # NOTE: if the importer argument 'merge_with_existing' is not set it
        # defaults to True here.
        # TODO(AR) make 'merge_with_existing' an explicit argument to this
        #  method.
        do_merge_with_existing = importer_kwargs.get("merge_with_existing", True) and not importer_kwargs.get(
            "resampling_info"
        )
        # Load all the metadata so we can sort the files
        # assume metadata collection is in the most user-friendly order
        infos = self._workspace.collect_product_metadata_for_paths(paths, **importer_kwargs)
        uuids = []
        merge_target_uuids = {}  # map new files uuids to merge target uuids
        total_products = 0
        for dex, (num_prods, info) in enumerate(infos):
            assert info is not None

            uuid = info[Info.UUID]
            merge_target_uuid = uuid
            if do_merge_with_existing:
                # real_paths because for satpy imports the methods paths parameter actually
                # contains the reader names
                real_paths = info["paths"]
                merge_target = self._workspace.find_merge_target(uuid, real_paths, info)
                if merge_target:
                    merge_target_uuid = merge_target.uuid

            yield {
                TASK_DOING: "Collecting metadata {}/{}".format(dex + 1, num_prods),
                TASK_PROGRESS: float(dex + 1) / float(num_prods),
                "uuid": merge_target_uuid,
                "num_products": num_prods,
            }
            # redundant but also more explicit than depending on num_prods
            total_products = num_prods
            uuids.append(uuid)
            merge_target_uuids[uuid] = merge_target_uuid

        if not total_products:
            raise ValueError("no products available in {}".format(paths))

        if isinstance(self._workspace, CachingWorkspace):
            # reverse list since we always insert a top layer
            uuids = list(reversed(self.sort_product_uuids(uuids)))

        # collect product and resource information but don't yet import content
        for dex, uuid in enumerate(uuids):
            merge_target_uuid = merge_target_uuids[uuid]
            if do_merge_with_existing and uuid != merge_target_uuid:  # merge products
                active_content_data = self._workspace.import_product_content(
                    uuid, merge_target_uuid=merge_target_uuid, **importer_kwargs
                )
                # active_content_data is none if all segments are already loaded
                # and there is nothing new to import
                if active_content_data:
                    dataset = self[merge_target_uuid]
                    self.didUpdateBasicDataset.emit(merge_target_uuid, dataset[Info.KIND])
            elif uuid in self._layer_with_uuid:
                LOG.warning("layer with UUID {} already in document?".format(uuid))
                self._workspace.get_content(uuid)
            else:
                self.activate_product_uuid_as_new_dataset(uuid, insert_before=insert_before, **importer_kwargs)

            yield {
                TASK_DOING: "Loading content {}/{}".format(dex + 1, total_products),
                TASK_PROGRESS: float(dex + 1) / float(total_products),
                "uuid": merge_target_uuid,
                "num_products": total_products,
            }

    def sort_product_uuids(self, uuids: typ.Iterable[UUID]) -> typ.List[UUID]:
        uuidset = set(str(x) for x in uuids)
        if not uuidset:
            return []
        with self._workspace.metadatabase as S:
            zult = [
                (x.uuid, x.ident)
                for x in S.query(Product)
                .filter(Product.uuid_str.in_(uuidset))
                .order_by(Product.family, Product.category, Product.serial)
                .all()
            ]
        LOG.debug("sorted products: {}".format(repr(zult)))
        return [u for u, _ in zult]

    def time_label_for_uuid(self, uuid):
        """used to update animation display when a new frame is shown"""
        if not uuid:
            return "YYYY-MM-DD HH:MM"
        info = self._layer_with_uuid[uuid]
        return info.get(Info.DISPLAY_TIME, "--:--")

    def prez_for_uuids(self, uuids: typ.List[UUID], lset: list = None) -> typ.Iterable[typ.Tuple[UUID, Presentation]]:
        if lset is None:
            lset = self.current_layer_set
        for p in lset:
            if p.uuid in uuids:
                yield p.uuid, p

    def prez_for_uuid(self, uuid: UUID, lset: list = None) -> Presentation:
        for _, p in self.prez_for_uuids((uuid,), lset=lset):
            return p

    def colormap_for_uuids(self, uuids: typ.List[UUID], lset: list = None) -> typ.Iterable[typ.Tuple[UUID, str]]:
        for u, p in self.prez_for_uuids(uuids, lset=lset):
            yield u, p.colormap

    def colormap_for_uuid(self, uuid: UUID, lset: list = None) -> str:
        for _, p in self.colormap_for_uuids((uuid,), lset=lset):
            return p

    def valid_range_for_uuid(self, uuid):
        # Limit ourselves to what information
        # in the future valid range may be different than the default CLIMs
        return self[uuid][Info.CLIM]

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

    def row_for_uuid(self, *uuids):
        d = dict((q.uuid, i) for i, q in enumerate(self.current_layer_set))
        if len(uuids) == 1:
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
            nu = dataclasses.replace(old, visible=vis)
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
        curr_set = self.current_layer_set
        for uuid, visible in changes.items():
            dex = curr_set[uuid]
            old = curr_set[dex]
            curr_set[dex] = dataclasses.replace(old, visible=visible)
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
        if not sibs:
            LOG.info("nothing to do in next_last_timestep")
            self.toggle_layer_visibility(uuid, True)
            return uuid
        dex += delta + len(sibs)
        dex %= len(sibs)
        new_focus = sibs[dex]
        del sibs[dex]
        if sibs:
            LOG.debug("hiding {} siblings".format(len(sibs)))
            self.toggle_layer_visibility(sibs, False)
        LOG.debug("showing new preferred {}step {}".format("time" if not bandwise else "band", new_focus))
        self.toggle_layer_visibility(new_focus, True)  # FUTURE: do these two commands in one step
        return new_focus

    def is_layer_visible(self, row):
        return self.current_layer_set[row].visible

    def layer_animation_order(self, layer_number):
        return self.current_layer_set[layer_number].a_order

    def change_layer_name(self, row, new_name):
        uuid = self.current_layer_set[row].uuid if not isinstance(row, UUID) else row
        info = self._layer_with_uuid[uuid]
        assert uuid == info[Info.UUID]
        if not new_name:
            # empty string, reset to default DISPLAY_NAME
            new_name = info.default_display_name
        info[Info.DISPLAY_NAME] = new_name
        self.didChangeLayerName.emit(uuid, new_name)

    def change_colormap_for_layers(self, name, uuids=None):
        L = self.current_layer_set
        if uuids is not None:
            uuids = self.time_siblings_uuids(uuids)
        else:  # all data layers
            uuids = [pinfo.uuid for pinfo in L]

        nfo = {}
        for uuid in uuids:
            for dex, pinfo in enumerate(L):
                if pinfo.uuid == uuid:
                    L[dex] = dataclasses.replace(pinfo, colormap=name)
                    nfo[uuid] = name
        self.didChangeColormap.emit(nfo)

    def current_layers_where(self, kinds=None, uuids=None, dataset_names=None, wavelengths=None, colormaps=None):
        """check current layer list for criteria and yield"""
        L = self.current_layer_set
        for idx, p in enumerate(L):
            if (uuids is not None) and (p.uuid not in uuids):
                continue
            layer = self._layer_with_uuid[p.uuid]
            if (kinds is not None) and (layer.kind not in kinds):
                continue
            if (dataset_names is not None) and (layer[Info.DATASET_NAME] not in dataset_names):
                continue
            if (wavelengths is not None) and (layer.get(Info.CENTRAL_WAVELENGTH) not in wavelengths):
                continue
            if (colormaps is not None) and (p.colormap not in colormaps):
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
            new_pz = dataclasses.replace(pz, climits=clims)
            nfo[layer.uuid] = new_pz.climits
            L[idx] = new_pz
        self.didChangeColorLimits.emit(nfo)

    def change_clims_for_siblings(self, uuid, clims):
        uuids = self.time_siblings(uuid)[0]
        return self.change_clims_for_layers_where(clims, uuids=uuids)

    def change_layers_image_kind(self, uuids, new_kind):
        """Change an image or contour layer to present as a different kind."""
        nfo = {}
        layer_set = self.current_layer_set
        assert new_kind in Kind
        all_uuids = set()
        for u in uuids:
            fam = self.family_for_product_or_layer(u)
            all_uuids.update(self._families[fam])
        for idx, pz, layer in self.current_layers_where(uuids=all_uuids):
            if pz.kind not in [Kind.IMAGE, Kind.CONTOUR]:
                LOG.warning("Can't change image kind for Kind: %s", pz.kind.name)
                continue
            new_pz = dataclasses.replace(pz, kind=new_kind)
            nfo[layer.uuid] = new_pz
            layer_set[idx] = new_pz
        self.didChangeImageKind.emit(nfo)

    def __len__(self):
        # FIXME: this should be consistent with __getitem__, not self.current_layer_set
        return len(self.current_layer_set)

    def uuid_for_current_layer(self, row):
        uuid = self.current_layer_set[row].uuid
        return uuid

    def family_uuids(self, family, active_only=False):
        uuids = self._families[family]
        if not active_only:
            return uuids
        current_visible = list(self.current_visible_layer_uuids)
        return [u for u in uuids if u in current_visible]

    def family_uuids_for_uuid(self, uuid, active_only=False):
        return self.family_uuids(self[uuid][Info.FAMILY], active_only=active_only)

    def remove_layers_from_all_sets(self, uuids):
        # find RGB layers family
        all_uuids = set()
        for uuid in list(uuids):
            all_uuids.add(uuid)
            if isinstance(self[uuid], DocRGBDataset):
                all_uuids.update(self.family_uuids_for_uuid(uuid))

        # collect all times for these layers to update RGBs later
        times = [self[u][Info.SCHED_TIME] for u in all_uuids]

        # delete all these layers from the layer list/presentation
        for uuid in all_uuids:
            LOG.debug("removing {} from family and Presentation lists".format(uuid))
            # remove from available family layers
            self._remove_layer_from_family(uuid)
            # remove from the layer set
            self.remove_layer_prez(uuid)  # this will send signal and start purge
            # remove from data layer collection
            if self.data_layer_collection is not None:
                self.data_layer_collection.remove_by_uuid(uuid)

        # Remove this layer from any RGBs it is a part of
        self.sync_composite_layer_prereqs(times)

        # Remove data from the workspace
        self.purge_layer_prez(uuids)

    def get_uuids(self):
        return list(self._layer_with_uuid.keys())

    def clear(self):
        """
        Convenience function to clear document
        """
        all_uuids = self.get_uuids()
        self.remove_layers_from_all_sets(all_uuids)

    def animate_siblings_of_layer(self, row_or_uuid):
        uuid = self.current_layer_set[row_or_uuid].uuid if not isinstance(row_or_uuid, UUID) else row_or_uuid
        layer = self._layer_with_uuid[uuid]
        L = self.current_layer_set

        if isinstance(layer, DocRGBDataset):
            pass
        else:
            new_anim_uuids, _ = self.time_siblings(uuid)
            if new_anim_uuids is None or len(new_anim_uuids) < 2:
                LOG.info("no time siblings to chosen band, will try channel siblings to chosen time")
                new_anim_uuids, _ = self.channel_siblings(uuid)
            if new_anim_uuids is None or len(new_anim_uuids) < 2:
                LOG.info("No animation found")
                return []

        LOG.debug("new animation order will be {0!r:s}".format(new_anim_uuids))
        L.animation_order = new_anim_uuids
        self.didReorderAnimation.emit(tuple(new_anim_uuids))
        return new_anim_uuids

    def get_info(self, row: int = None, uuid: UUID = None) -> typ.Optional[DocBasicDataset]:
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
            raise ValueError("document[UUID] required, %r was used" % type(layer_uuid))

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
        assert len(new_order) == len(self._layer_sets[layer_set_index])
        new_layer_set = DocLayerStack(self, [self._layer_sets[layer_set_index][n] for n in new_order])
        self._layer_sets[layer_set_index] = new_layer_set
        self.didReorderDatasets.emit(tuple(new_order))

    def is_using(self, uuid: UUID, layer_set: int = None):
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
                if parent_layer.kind == Kind.RGB and layer in parent_layer.layers:
                    return True
        return False

    def remove_layer_prez(self, row_or_uuid, count: int = 1):
        """
        remove the presentation of a given layer/s in the current set
        :param row: which current layer set row to remove
        :param count: how many rows to remove
        :return:
        """
        if isinstance(row_or_uuid, UUID) and count == 1:
            try:
                row = self.row_for_uuid(row_or_uuid)
            except KeyError:
                LOG.debug("Can't remove in-active layer: {}".format(row_or_uuid))
                return
            uuids = [row_or_uuid]
        else:
            row = row_or_uuid
            uuids = [x.uuid for x in self.current_layer_set[row : row + count]]
        self.toggle_layer_visibility(list(range(row, row + count)), False)
        clo = list(range(len(self.current_layer_set)))
        del clo[row : row + count]
        del self.current_layer_set[row : row + count]
        self.didRemoveDatasets.emit(tuple(clo), uuids, row, count)

    def purge_layer_prez(self, uuids):
        """Purge layers from the workspace"""
        for uuid in uuids:
            if not self.is_using(uuid):
                LOG.debug("purging layer {}, no longer in use".format(uuid))
                self.willPurgeDataset.emit(uuid)
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
        sibs = [
            (x[Info.SHORT_NAME], x[Info.UUID])
            for x in self._filter(
                sibling_infos.values(), it, {Info.SCENE, Info.SCHED_TIME, Info.INSTRUMENT, Info.PLATFORM}
            )
        ]
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
                    fail = True
            if not fail:
                yield md

    def time_siblings(self, uuid, sibling_infos=None):
        """
        return time-ordered list of datasets which have the same band, in time order
        :param uuid: focus UUID we're trying to build around
        :param sibling_infos: dictionary of UUID -> Dataset Info to sort through
        :return: sorted list of sibling uuids in time order, index of where uuid is in the list
        """
        # NOTE(mk): create_data_layers is the bridge between document and collection
        # as it uses state from document to create a list of data_layers that can then
        # be passed to methods of data_layer_collection
        # self.create_data_layers()
        # TODO(mk) pull create_data_layers into DataLayerCollection
        if sibling_infos is None:
            sibling_infos = self._layer_with_uuid
        it = sibling_infos.get(uuid, None)
        if it is None:
            return [], 0
        sibs = [
            (x[Info.SCHED_TIME], x[Info.UUID])
            for x in self._filter(
                sibling_infos.values(),
                it,
                {Info.SHORT_NAME, Info.STANDARD_NAME, Info.SCENE, Info.INSTRUMENT, Info.PLATFORM, Info.KIND},
            )
        ]
        # then sort it into time order
        sibs.sort()
        offset = [i for i, x in enumerate(sibs) if x[1] == uuid]
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
