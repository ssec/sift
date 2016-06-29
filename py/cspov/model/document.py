#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cspov.model.document
--------------------

PURPOSE
Core (low-level) document model for CSPOV.
The core is sometimes accessed via Facets, which are like database views for a specific group of use cases

The document model is a metadata representation which permits the workspace to be constructed and managed.

Document is primarily a composition of layers.
Layers come in several flavors:
- Image : a float32 field shown as tiles containing strides and/or alternate LODs, having a colormap
- Outline : typically a geographic political map
- Shape : a highlighted region selected by the user: point, line (great circle), polygon
- Combination : calculated from two or more image layers, e.g. RGBA combination of images
                combinations may be limited to areas specified by region layers.
=== advanced stuff for later ===
- Volume : a 3D dataset, potentially sparse
-- DenseVolume
-- SparseVolume : (x,y,z) point cloud

Layers are represented in 1 or more LayerSets, which are alternate configurations of the display.
Users may wish to display the same data in several different ways for illustration purposes.
Only one LayerSet is used on a given Map window at a time.

Layers have presentation settings that vary with LayerSet:
- z_order: bottom to top in the map display
- visible: whether or not it's being drawn on the map
- a_order: animation order, when the animation button is hit
- colormap: how the data is converted to pixels
- mixing: mixing mode when drawing (normal, additive)

Document has zero or more Probes.
Layers also come in multiple flavors that may be be attached to plugins or helper applications.
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
from cspov.model.layer import mixing, DocLayer, DocBasicLayer, DocCompositeLayer, DocRGBLayer

__author__ = 'rayg'
__docformat__ = 'reStructuredText'

import sys
import logging
import unittest
import argparse
from collections import namedtuple, MutableSequence
from uuid import UUID
import numpy as np
from abc import ABCMeta, abstractmethod, abstractproperty
from weakref import ref

from cspov.common import KIND, INFO, COMPOSITE_TYPE
from cspov.model.guidebook import AHI_HSF_Guidebook, GUIDE

from PyQt4.QtCore import QObject, pyqtSignal


LOG = logging.getLogger(__name__)

DEFAULT_LAYER_SET_COUNT = 1  # this should match the ui configuration!

# presentation information for a layer; z_order comes from the layerset
prez = namedtuple('prez', [
    'uuid',     # UUID: dataset in the document/workspace
    'kind',     # what kind of layer it is
    'visible',  # bool: whether it's visible or not
    'a_order',  # int: None for non-animating, 0..n-1 what order to draw in during animation
    'colormap', # name or uuid: color map to use; name for default, uuid for user-specified
    'climits',     # tuple: valid min and valid max used for color mapping normalization
    'mixing'    # mixing mode constant
])


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

    def __setitem__(self, index:int, value:prez):
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

    def __len__(self):
        return len(self._store)

    def __delitem__(self, index:int):
        del self._store[index]
        self._u2r = None

    def insert(self, index:int, value:prez):
        self._store.insert(index, value)
        self._u2r = None





#
#
# class DocAsLayerStack(metaclass=ABCMeta):
#     """
#     interface used by SceneGraphManager
#     """
#     @abstractmethod
#     def layers_in_z_order(self):
#         """
#         return current enabled list of (active layer, animation order)
#         animation order of 0 implies not part of current animation
#         :return:
#         """
#         pass
#
#
# class DocAsDataSpace(metaclass=ABCMeta):
#     """
#     interface used to coordinate left-right-up-down keys.
#     application behavior managing these keypresses uses this interface
#     Typically left-right is time, up-down is channel.
#     Consultation with the guidebook may be needed.
#     'l', 'r', 'u', 'd', 'i', 'o': left, right, up, down, in, out
#     """
#     @abstractmethod
#     def neighboring_layer(self, direction:str, amount:int=1):
#         """
#         return neighboring layer
#         :param direction: l, r, u, d string indicating direction
#         :param amount: number of steps to displace, typically +1
#         :return:
#         """
#         pass
#
#
# class DocAsLayerTree(metaclass=ABCMeta):
#     """
#     interface (facet) used to coordinate drag-and-drop layer tree
#     """
#     pass
#





class Document(QObject):  # base class is rightmost, mixins left of that
    """
    Document has one or more LayerSets choosable by the user (one at a time) as currentLayerSet
    LayerSets configure animation order, visibility, enhancements and linear combinations
    LayerSets can be cloned from the prior active LayerSet when unconfigured
    Document has Probes, which operate on the currentLayerSet
    Probes have spatial areas (point probes, shaped areas)
    Probe areas are translated into localized data masks against the workspace raw data content

    """
    current_set_index = 0
    _workspace = None
    _layer_sets = None  # list(DocLayerSet(prez, ...) or None)
    _layer_with_uuid = None  # dict(uuid:Doc____Layer)
    _guidebook = None  # FUTURE: this is currently an AHI_HSF_Guidebook, make it a general guidebook

    # signals
    didAddBasicLayer = pyqtSignal(list, DocBasicLayer, prez, np.ndarray)  # new order list with None for new layer; info-dictionary, overview-content-ndarray
    didAddCompositeLayer = pyqtSignal(list, DocCompositeLayer, list, list, list, object)  # comp layer is derived from multiple basic layers and has its own UUID
    didRemoveLayers = pyqtSignal(list, list, int, int)  # new order, UUIDs that were removed from current layer set, first row removed, num rows removed
    willPurgeLayer = pyqtSignal(UUID)  # UUID of the layer being removed
    didReorderLayers = pyqtSignal(list)  # list of original indices in their new order, None for new layers
    didChangeLayerVisibility = pyqtSignal(dict)  # {UUID: new-visibility, ...} for changed layers
    didReorderAnimation = pyqtSignal(list)  # list of UUIDs representing new animation order
    didChangeLayerName = pyqtSignal(UUID, str)  # layer uuid, new name
    didSwitchLayerSet = pyqtSignal(int, DocLayerStack, list)  # new layerset number typically 0..3, list of prez tuples representing new display order, new animation order
    didChangeColormap = pyqtSignal(dict)  # dict of {uuid: colormap-name-or-UUID, ...} for all changed layers
    didChangeColorLimits = pyqtSignal(dict)  # dict of {uuid: colormap-name-or-UUID, ...} for all changed layers
    didChangeComposition = pyqtSignal(list, DocCompositeLayer, prez, dict)  # new-layer-order, changed-layer, change-info: composite channels were reassigned or polynomial altered
    didCalculateLayerEqualizerValues = pyqtSignal(dict)  # dict of {uuid: (value, normalized_value_within_clim)} for equalizer display
    # didChangeShapeLayer = pyqtSignal(dict)

    def __init__(self, workspace, layer_set_count=DEFAULT_LAYER_SET_COUNT, **kwargs):
        super(Document, self).__init__(**kwargs)
        self._guidebook = AHI_HSF_Guidebook()
        self._workspace = workspace
        self._layer_sets = [DocLayerStack(self)] + [None] * (layer_set_count - 1)
        self._layer_with_uuid = {}
        # TODO: connect signals from workspace to slots including update_dataset_info

    def _default_colormap(self, datasetinfo):
        """
        consult guidebook and user preferences for which enhancement should be used for a given datasetinfo
        :param datasetinfo: dictionary of metadata about dataset
        :return: enhancement info and siblings participating in the enhancement
        """
        return self._guidebook.default_colormap(datasetinfo)

    @property
    def current_layer_set(self):
        return self._layer_sets[self.current_set_index]

    # def _additional_guidebook_information(self, info):
    #     """
    #     when adding a file, return any additional information we want from the guidebook
    #     :param info: existing datasetinfo
    #     :return: dictionary of information not immediately available from the file itself
    #     """
    #     md =
    #     return {
    #         INFO.DISPLAY_TIME: self._guidebook.display_time(info)
    #     }

    def _insert_layer_with_info(self, info: DocLayer, cmap=None, insert_before=0):
        """
        insert a layer into the presentations but do not signal
        """
        p = prez(uuid=info[INFO.UUID],
                 kind=info[INFO.KIND],
                 visible=True,
                 a_order=None,
                 colormap=cmap,
                 climits=info[INFO.CLIM],
                 mixing=mixing.NORMAL)

        q = p._replace(visible=False)  # make it available but not visible in other layer sets
        old_layer_count = len(self._layer_sets[self.current_set_index])
        for dex, lset in enumerate(self._layer_sets):
            if lset is not None:  # uninitialized layer sets will be None
                lset.insert(insert_before, p if dex == self.current_set_index else q)

        reordered_indices = [None] + list(range(old_layer_count))  # FIXME: this should obey insert_before, currently assumes always insert at top
        return p, reordered_indices

    def open_file(self, path, insert_before=0):
        """
        open an arbitrary file and make it the new top layer.
        emits docDidChangeLayer followed by docDidChangeLayerOrder
        :param path: file to open and add
        :return: overview (uuid:UUID, datasetinfo:dict, overviewdata:numpy.ndarray)
        """
        uuid, info, content = self._workspace.import_image(source_path=path)
        if uuid in self._layer_with_uuid:
            LOG.warning("layer with UUID {0:s} already in document?".format(uuid))
            return uuid, info, content
        # info.update(self._additional_guidebook_information(info))
        self._layer_with_uuid[uuid] = dataset = DocBasicLayer(self, info)
        # also get info for this layer from the guidebook
        gbinfo = self._guidebook.collect_info(dataset)
        dataset.update(gbinfo)  # FUTURE: should guidebook be integrated into DocBasicLayer?

        # add as visible to the front of the current set, and invisible to the rest of the available sets
        cmap = self._default_colormap(dataset)
        dataset[INFO.CLIM] = self._guidebook.climits(dataset)
        dataset[INFO.NAME] = self._guidebook.display_name(dataset) or dataset[INFO.NAME]
        presentation, reordered_indices = self._insert_layer_with_info(dataset, cmap=cmap, insert_before=insert_before)
        # signal updates from the document
        self.didAddBasicLayer.emit(reordered_indices, dataset, presentation, content)
        return uuid, dataset, content

    def open_files(self, paths, insert_before=0):
        """
        sort paths into preferred load order (see guidebook.py)
        open files in order, yielding uuid, info, overview_content
        :param paths: paths to open
        :param insert_before: where to insert them in layer list
        :return:
        """
        paths = list(self._guidebook.sort_pathnames_into_load_order(paths))
        for path in paths:
            yield self.open_file(path, insert_before)

    def sort_paths(self, paths):
        """
        :param paths: list of paths
        :return: list of paths
        """
        paths = list(reversed(self._guidebook.sort_pathnames_into_load_order(paths)))  # go from load order to display order by reversing
        return paths

    def time_label_for_uuid(self, uuid):
        """used to update animation display when a new frame is shown
        """
        if not uuid:
            return "YYYY-MM-DD HH:MM"
        info = self._layer_with_uuid[uuid]
        return self._guidebook.display_time(info)

    def prez_for_uuids(self, uuids, lset=None):
        if lset is None:
            lset = self.current_layer_set
        for p in lset:
            if p.uuid in uuids:
                yield p

    def colormap_for_uuids(self, uuids, lset=None):
        for p in self.prez_for_uuids(uuids, lset=lset):
            yield p.colormap

    def convert_units(self, uuid, data, inverse=False):
        formatstr, unitstr, lam = self._guidebook.units_conversion(self._layer_with_uuid[uuid])
        return formatstr, unitstr, lam(data, inverse)

    def flipped_for_uuids(self, uuids, lset=None):
        for p in self.prez_for_uuids(uuids, lset=lset):
            default_clim = self._layer_with_uuid[p.uuid][INFO.CLIM]
            yield ((p.climits[1] - p.climits[0]) > 0) != ((default_clim[1] - default_clim[0]) > 0)

    def update_equalizer_values(self, uuid, state, xy_pos):
        """user has clicked on a point probe; determine relative and absolute values for all document image layers
        """
        # if the point probe was turned off then we don't want to have the equalizer
        if not state:
            self.didCalculateLayerEqualizerValues.emit({})
            return

        zult = {}
        for pinf in self.current_layer_set:
            if pinf.uuid in zult:
                continue
            lyr = self._layer_with_uuid[pinf.uuid]
            if lyr[INFO.KIND] != KIND.IMAGE:
                continue
            value = self._workspace.get_content_point(pinf.uuid, xy_pos)
            fmt, unit_str, unit_conv = self._guidebook.units_conversion(self._layer_with_uuid[pinf.uuid])
            # calculate normalized bar width relative to its current clim
            nc, xc = unit_conv(np.array(pinf.climits))
            if nc > xc:  # sometimes clim is swapped to reverse color scale
                nc, xc = xc, nc
            value = unit_conv(value)
            if np.isnan(value):
                zult[pinf.uuid] = None
            else:
                bar_width = (np.clip(value, nc, xc) - nc) / (xc - nc)
                zult[pinf.uuid] = (value, bar_width, fmt, unit_str)
        self.didCalculateLayerEqualizerValues.emit(zult)  # is picked up by layer list model to update display

    # TODO, find out if this is needed/used and whether or not it's correct
    def update_dataset_info(self, new_info):
        """
        slot which updates document on new information workspace has provided us about a dataset
        typically signaled by importer operating in the workspace
        :param new_info: information dictionary including projection, levels of detail, etc
        :return: None
        """
        uuid = new_info[INFO.UUID]
        if uuid not in self._layer_with_uuid:
            LOG.warning('new information on uuid {0!r:s} is not for a known dataset'.format(new_info))
        self._layer_with_uuid[new_info[INFO.UUID]].update(new_info)
        # TODO, also get information about this layer from the guidebook?

        # TODO: see if this affects any presentation information; view will handle redrawing on its own

    def _clone_layer_set(self, existing_layer_set):
        return DocLayerStack(existing_layer_set)

    @property
    def current_animation_order(self):
        """
        return list of UUIDs representing the animation order in the currently selected layer set
        :return: list of UUIDs
        """
        cls = self.current_layer_set
        aouu = [(x.a_order, x.uuid) for x in cls if (x.a_order is not None)]
        aouu.sort()
        ao = [u for a,u in aouu]
        LOG.debug('animation order is {0!r:s}'.format(ao))
        return ao
        # return list(reversed(self.current_layer_order))
        # FIXME DEBUG - use this code once we have animation order setting commands
        # q = [(x.a_order, x.uuid) for x in self.current_layer_set if x.a_order is not None]
        # q.sort()
        # return [u for _,u in q]

    @property
    def current_layer_uuid_order(self):
        """
        list of UUIDs (top to bottom) currently being displayed, independent of visibility/validity
        :return:
        """
        return [x.uuid for x in self.current_layer_set]

    @property
    def current_visible_layer_uuid(self):
        """
        :return: the topmost visible layer's UUID
        """
        for x in self.current_layer_set:
            if x.visible:
                return x.uuid
        return None

    # def current_visible_layer_uuids(self, max_layers=None):
    #     """
    #     :param max_layers:
    #     :yield: the visible layers in the current layer set
    #     """
    #     count = 0
    #     for x in self.current_layer_set:
    #         if x.visible:
    #             count += 1
    #             yield x.uuid
    #         if max_layers is not None and count >= max_layers:
    #             break

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

    def select_layer_set(self, layer_set_index:int):
        """
        change the selected layer set, 0..N (typically 0..3), cloning the old set if needed
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

    # def change_layer_order(self, old_index, new_index):
    #     L = self.current_layer_set
    #     order = list(range(len(L)))
    #     p = L[old_index]
    #     d = order[old_index]
    #     del L[old_index]
    #     del order[old_index]
    #     L.insert(new_index, p)
    #     L.insert(new_index, d)
    #     self.didReorderLayers.emit(order)

    # def swap_layer_order(self, row1, row2):
    #     L = self.current_layer_set
    #     order = list(range(len(L)))
    #     L[row1], L[row2] = L[row2], L[row1]
    #     order[row1], order[row2] = order[row2], order[row1]
    #     self.didReorderLayers.emit(order)

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
                dex = L[dex]  # returns row index
            old = L[dex]
            vis = (not old.visible) if visible is None else visible
            # print(vis)
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
        use the data guidebook to
        find the next or last time/bandstep (default: the layer itself) in the document
        make all layers in the sibling group invisible save that timestep
        :param uuid: layer we're focusing on as reference
        :param delta: -1 => last step, 0 for focus step, +1 for next step
        :param bandwise: True if we want to change by band instead of time
        :return: UUID of new focus layer
        """
        # get list of UUIDs in time order, plus index where the focus uuid is
        if bandwise:  # next or last band
            consult_guide = self._guidebook.channel_siblings
        else:
            consult_guide = self._guidebook.time_siblings
        sibs, dex = consult_guide(uuid, self._layer_with_uuid.values())
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
        info[INFO.NAME] = new_name
        self.didChangeLayerName.emit(uuid, new_name)

    def change_colormap_for_layers(self, name, uuids=None):
        L = self.current_layer_set
        if uuids is not None:
            uuids = self._guidebook.time_siblings_uuids(uuids, self._layer_with_uuid.values())
        else:  # all data layers
            uuids = [pinfo.uuid for pinfo in L]

        nfo = {}
        for uuid in uuids:
            for dex,pinfo in enumerate(L):
                if pinfo.uuid==uuid:
                    L[dex] = pinfo._replace(colormap=name)
                    nfo[uuid] = name
        self.didChangeColormap.emit(nfo)

    def flip_climits_for_layers(self, uuids=None):
        L = self.current_layer_set
        if uuids is not None:
            uuids = self._guidebook.time_siblings_uuids(uuids, self._layer_with_uuid.values())
        else:  # all data layers
            uuids = [pinfo.uuid for pinfo in L]

        nfo = {}
        for uuid in uuids:
            for dex,pinfo in enumerate(L):
                if pinfo.uuid==uuid:
                    nfo[uuid] = pinfo.climits[::-1]
                    L[dex] = pinfo._replace(climits=nfo[uuid])
        self.didChangeColorLimits.emit(nfo)

    def create_rgb_composite(self, r=None, g=None, b=None, all_timesteps=True):
        """
        user has specified that a band trio should be shown as RGB
        disable display of the three layers
        add a composite layer at the z level of the topmost of the three
        do likewise for other timesteps with the same bands
        """
        from uuid import uuid1 as uuidgen
        from functools import reduce
        if r is None or g is None or b is None:
            # we have an invalid composite which needs user initialization
            LOG.info('generating invalid composite for user to configure')
            uuid = uuidgen()
            name = '-RGB-'
            ds_info = {
                INFO.UUID: uuid,
                INFO.NAME: name,
                INFO.KIND: KIND.RGB,
                GUIDE.BAND: [],
                GUIDE.DISPLAY_TIME: None,
                INFO.ORIGIN_X: None,
                INFO.ORIGIN_Y: None,
                INFO.CELL_WIDTH: None,
                INFO.CELL_HEIGHT: None,
                INFO.CLIM: None,
            }
            self._layer_with_uuid[uuid] = ds_info = DocRGBLayer(self, ds_info)
            presentation, reordered_indices = self._insert_layer_with_info(ds_info)
            return

        # disable visibility of the existing layers FUTURE: remove them entirely? probably not
        self.toggle_layer_visibility([x for x in [r,g,b] if x], False)
        # add notation to document on RGB affinity
        # register with workspace so that it can persist info to disk if needed
        # insert new RGB layer into layer list and scenegraph

        uuids = [r,g,b]
        LOG.debug("New Composite UUIDs: %r" % uuids)
        # FIXME: register this with workspace!

        dep_info = [(None if uuid is None else self.get_info(uuid=uuid)) for uuid in uuids]
        highest_res_dep = min(dep_info, key=lambda x: x[INFO.CELL_WIDTH])
        _dt = lambda nfo: nfo.get(GUIDE.DISPLAY_TIME, '<unknown time>')
        display_time = reduce(lambda dta,b: dta if dta==_dt(b) else '<multiple times>', dep_info, _dt(dep_info[0]))
        uuid = uuidgen()  # FUTURE: workspace should be providing this?
        try:
            bands = (dep_info[0][GUIDE.BAND],
                     dep_info[1][GUIDE.BAND],
                     dep_info[2][GUIDE.BAND])
            name = u"R:B{0:02d} G:B{1:02d} B:B{2:02d}".format(*bands)
        except KeyError:
            LOG.error('unable to create new name from {0!r:s}'.format(dep_info))
            name = "RGB"
            bands = []
        ds_info = {
            INFO.UUID: uuid,
            INFO.NAME: name,
            INFO.KIND: KIND.RGB,
            GUIDE.BAND: bands,
            GUIDE.DISPLAY_TIME: display_time,
            INFO.ORIGIN_X: highest_res_dep[INFO.ORIGIN_X],
            INFO.ORIGIN_Y: highest_res_dep[INFO.ORIGIN_Y],
            INFO.CELL_WIDTH: highest_res_dep[INFO.CELL_WIDTH],
            INFO.CELL_HEIGHT: highest_res_dep[INFO.CELL_HEIGHT],
            INFO.CLIM: tuple(d[INFO.CLIM] for d in dep_info),
        }
        self._layer_with_uuid[uuid] = ds_info = DocRGBLayer(self, ds_info)
        presentation, reordered_indices = self._insert_layer_with_info(ds_info)

        prezs = None  # not used right now FIXME
        overview_content = list(self._workspace.get_content(d[INFO.UUID]) for d in dep_info)
        self.didAddCompositeLayer.emit([None], ds_info, [None], overview_content, uuids, COMPOSITE_TYPE.RGB)  # FIXME first parm (new_order) is wrong
        return ds_info

    def create_empty_rgb_composite(self):
        """
        an empty RGB composite is added to the top of the current layer list.
        since it needs R, G, and B in order to be valid, we do not yet announce it to the scenegraph
        :return:
        """
        self.create_rgb_composite(None, None, None)

    def __len__(self):
        return len(self.current_layer_set)

    def uuid_for_layer(self, row):
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
            # now remove from the active layer set
            self.remove_layer_prez(uuid)  # this will send signal and start purge

    def clear_animation_order(self):
        cls = self.current_layer_set
        for i,q in enumerate(cls):
            cls[i] = q._replace(a_order=None)

    def animate_siblings_of_layer(self, row_or_uuid):
        uuid = self.current_layer_set[row_or_uuid].uuid if not isinstance(row_or_uuid, UUID) else row_or_uuid
        new_anim_uuids, _ = self._guidebook.time_siblings(uuid, self._layer_with_uuid.values())
        if new_anim_uuids is None or len(new_anim_uuids)<2:
            LOG.info('no time siblings to chosen band, will try channel siblings to chosen time')
            new_anim_uuids, _ = self._guidebook.channel_siblings(uuid, self._layer_with_uuid.values())
        if new_anim_uuids is None or len(new_anim_uuids)<2:
            LOG.warning('No animation found')
            return []
        LOG.debug('new animation order will be {0!r:s}'.format(new_anim_uuids))
        L = self.current_layer_set
        self.clear_animation_order()
        for dex,u in enumerate(new_anim_uuids):
            LOG.debug(u)
            row = L.uuid2row.get(u, None)
            if row is None:
                LOG.error('unable to find row for uuid {} in current layer set'.format(u))
                continue
            old = L[row]
            new = old._replace(a_order=dex)
            L[row] = new
        self.didReorderAnimation.emit(new_anim_uuids)
        return new_anim_uuids

    def get_info(self, row=None, uuid=None):
        if row is not None :
            uuid_temp = self.current_layer_set[row].uuid
            nfo = self._layer_with_uuid[uuid_temp]
            return nfo
        elif uuid is not None :
            nfo = self._layer_with_uuid[uuid]
            return nfo
        return None

    def __getitem__(self, row_or_uuid):
        """
        return presentation tuple for a given layer index from the current active layer set
        """
        if isinstance(row_or_uuid, UUID):
            return self._layer_with_uuid[row_or_uuid]
        else:
            return self.current_layer_set[row_or_uuid]

    def reorder_by_indices(self, new_order, uuids=None, layer_set_index=None):
        """given a new layer order, replace the current layer set
        emits signal to other subsystems
        """
        if layer_set_index is None:
            layer_set_index = self.current_set_index
        assert(len(new_order)==len(self._layer_sets[layer_set_index]))
        new_layer_set = [self._layer_sets[layer_set_index][n] for n in new_order]
        self._layer_sets[layer_set_index] = new_layer_set
        self.didReorderLayers.emit(new_order)

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
        # FIXME: this needs to check not just which layers are being displayed, but which layers which may be in use but as part of a composite instead of a direct scenegraph entry
        LOG.error('composite layers currently not checked for dependencies')
        if layer_set is not None:
            lss = [self._layer_sets[layer_set]]
        else:
            lss = [q for q in self._layer_sets if q is not None]
        for ls in lss:
            for p in ls:
                if p.uuid==uuid:
                    return True
        return False

    def remove_layer_prez(self, row_or_uuid, count:int=1):
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
        self.didRemoveLayers.emit(clo, uuids, row, count)
        for uuid in uuids:
            if not self.is_using(uuid):
                LOG.info('purging layer {}, no longer in use'.format(uuid))
                self.willPurgeLayer.emit(uuid)
                # remove from our bookkeeping
                del self._layer_with_uuid[uuid]
                # remove from workspace
                self._workspace.remove(uuid)


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


