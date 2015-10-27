#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cspov.model.core
~~~~~~~~~~~~~~~~

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

__author__ = 'rayg'
__docformat__ = 'reStructuredText'

import sys
import logging
import unittest
import argparse
from collections import namedtuple
from enum import Enum
from uuid import UUID
import numpy as np
from copy import deepcopy

from cspov.common import KIND, INFO
from cspov.model.guidebook import AHI_HSF_Guidebook

from PyQt4.QtCore import QObject, pyqtSignal

# FIXME: move these out of the document and into a factory
from ..view.LayerRep import TiledGeolocatedImage, NEShapefileLines
from ..view.Program import GlooColormapDataTile, GlooRGBImageTile

from .probes import Probe, Shape


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

class mixing(Enum):
    UNKNOWN = 0
    NORMAL = 1
    ADD = 2
    SUBTRACT = 3


# class LayerSet(object):
#     """
#     LayerSet is a visual configuration of a stack of layers
#     """
#     def __init__(self, prior=None):
#         """
#         initialize, optionally by copying a prior LayerSet
#         """
#
#     def __getitem__(self, uuid):
#         """ retrieve layer information by uuid
#         """
#         # FIXME
#
#     @property
#     def animation_order


class Document(QObject):
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
    _layer_sets = None  # list(list(prez) or None)
    _layer_with_uuid = None  # dict(uuid:datasetinfo)
    _guidebook = None  # FUTURE: this is currently an AHI_HSF_Guidebook, make it a general guidebook

    # signals
    didAddLayer = pyqtSignal(list, dict, prez, np.ndarray)  # new order list with None for new layer; info-dictionary, overview-content-ndarray
    didRemoveLayers = pyqtSignal(list, list, int, int)  # new order, UUIDs that were removed from current layer set, first row removed, num rows removed
    willPurgeLayer = pyqtSignal(UUID)  # UUID of the layer being removed
    didReorderLayers = pyqtSignal(list)  # list of original indices in their new order, None for new layers
    didChangeLayerVisibility = pyqtSignal(dict)  # {UUID: new-visibility, ...} for changed layers
    didReorderAnimation = pyqtSignal(list)  # list of UUIDs representing new animation order
    didChangeLayerName = pyqtSignal(UUID, str)  # layer uuid, new name
    didSwitchLayerSet = pyqtSignal(int, list, list)  # new layerset number typically 0..3, list of prez tuples representing new display order, new animation order
    didChangeColormap = pyqtSignal(dict)  # dict of {uuid: colormap-name-or-UUID, ...} for all changed layers
    didChangeColorLimits = pyqtSignal(dict)  # dict of {uuid: colormap-name-or-UUID, ...} for all changed layers
    # didChangeShapeLayer = pyqtSignal(dict)

    def __init__(self, workspace, layer_set_count=DEFAULT_LAYER_SET_COUNT, **kwargs):
        super(Document, self).__init__(**kwargs)
        self._guidebook = AHI_HSF_Guidebook()
        self._workspace = workspace
        self._layer_sets = [list()] + [None] * (layer_set_count-1)
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

    def open_file(self, path, insert_before=0):
        """
        open an arbitrary file and make it the new top layer.
        emits docDidChangeLayer followed by docDidChangeLayerOrder
        :param path: file to open and add
        :return: overview (uuid:UUID, datasetinfo:dict, overviewdata:numpy.ndarray)
        """
        uuid, info, content = self._workspace.import_image(source_path=path)
        if uuid in self._layer_with_uuid:
            return uuid, info, content
        # info.update(self._additional_guidebook_information(info))
        self._layer_with_uuid[uuid] = info

        # add as visible to the front of the current set, and invisible to the rest of the available sets
        cmap = self._default_colormap(info)
        info[INFO.CLIM] = self._guidebook.climits(info)
        info[INFO.NAME] = self._guidebook.display_name(info) or info[INFO.NAME]
        p = prez(uuid=uuid,
                 kind=info[INFO.KIND],
                 visible=True,
                 a_order=None,
                 colormap=cmap,
                 climits=info[INFO.CLIM],
                 mixing=mixing.NORMAL)
        q = p._replace(visible=False)  # make it available but not visible in other layer sets
        old_layer_count = len(self._layer_sets[self.current_set_index])
        for dex,lset in enumerate(self._layer_sets):
            if lset is not None:  # uninitialized layer sets will be None
                lset.insert(insert_before, p if dex==self.current_set_index else q)

        reordered_indices = [None] + list(range(old_layer_count))
        # signal updates from the document
        self.didAddLayer.emit(reordered_indices, info, p, content)
        if info[INFO.KIND]==KIND.IMAGE:  # TODO: decide if this is correct and useful behavior
            self.animate_siblings_of_layer(uuid)

        return uuid, info, content

    def open_files(self, paths, insert_before=0):
        paths = list(self._guidebook.sort_paths(paths))
        for path in reversed(paths):
            self.open_file(path, insert_before)

    def time_label_for_uuid(self, uuid):
        """used to update animation display when a new frame is shown
        """
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

    def flipped_for_uuids(self, uuids, lset=None):
        for p in self.prez_for_uuids(uuids, lset=lset):
            default_clim = self._layer_with_uuid[p.uuid][INFO.CLIM]
            yield ((p.climits[1] - p.climits[0]) > 0) != ((default_clim[1] - default_clim[0]) > 0)

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
        self._layer_with_uuid[new_info[INFO.UUID]] = new_info
        # TODO: see if this affects any presentation information; view will handle redrawing on its own

    def _clone_layer_set(self, existing_layer_set):
        return deepcopy(existing_layer_set)

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
    def current_layer_order(self):
        """
        list of UUIDs (top to bottom) currently being displayed, independent of visibility
        :return:
        """
        return [x.uuid for x in self.current_layer_set]

    @property
    def current_visible_layer(self):
        for x in self.current_layer_set:
            if x.visible:
                return x.uuid
        return None

    def select_layer_set(self, layer_set_index):
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

    # TODO, not being used?
    def change_layer_order(self, old_index, new_index):
        L = self.current_layer_set
        order = list(range(len(L)))
        p = L[old_index]
        d = order[old_index]
        del L[old_index]
        del order[old_index]
        L.insert(new_index, p)
        L.insert(new_index, d)
        self.didReorderLayers.emit(order)

    # TODO, not being used?
    def swap_layer_order(self, row1, row2):
        L = self.current_layer_set
        order = list(range(len(L)))
        L[row1], L[row2] = L[row2], L[row1]
        order[row1], order[row2] = order[row2], order[row1]
        self.didReorderLayers.emit(order)

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
        r2u = dict((q.uuid,i) for i,q in enumerate(self.current_layer_set))
        if isinstance(rows_or_uuids, int) or isinstance(rows_or_uuids, UUID):
            rows_or_uuids = [rows_or_uuids]
        for dex in rows_or_uuids:
            if isinstance(dex, UUID):
                dex = r2u[dex]
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
        r2u = dict((q.uuid,i) for i,q in enumerate(self.current_layer_set))
        L = self.current_layer_set
        for uuid,visible in changes.items():
            dex = r2u[uuid]
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
        LOG.debug('new animation order will be {0!r:s}'.format(new_anim_uuids))
        cls = self.current_layer_set
        u2r = dict((x.uuid, i) for i,x in enumerate(cls))
        if not new_anim_uuids or len(new_anim_uuids)==1:
            return []
        self.clear_animation_order()
        for dex,u in enumerate(new_anim_uuids):
            LOG.debug(u)
            row = u2r.get(u, None)
            if row is None:
                LOG.error('unable to find row for uuid {} in current layer set'.format(u))
                continue
            old = cls[row]
            new = old._replace(a_order=dex)
            cls[row] = new
        self.didReorderAnimation.emit(new_anim_uuids)
        return new_anim_uuids

    def get_info(self, row=None):
        if row is not None:
            uuid = self.current_layer_set[row].uuid
            nfo = self._layer_with_uuid[uuid]
            return nfo
        return None

    def __getitem__(self, row:int):
        """
        return info for a given layer index
        """
        return self.current_layer_set[row]
        # uuid = self.current_layer_set[dex].uuid
        # nfo = self._layer_with_uuid[uuid]
        # return nfo

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


