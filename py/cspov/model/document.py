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
from cspov.common import kind

from PyQt4.QtCore import QObject, pyqtSignal

# FIXME: move these out of the document and into a factory
from ..view.LayerRep import TiledGeolocatedImage, NEShapefileLines
from ..view.Program import GlooColormapDataTile, GlooRGBImageTile

from .probes import Probe, Shape


LOG = logging.getLogger(__name__)

DEFAULT_LAYER_SET_COUNT = 4  # this should match the ui configuration!

# presentation information for a layer; z_order comes from the layerset
prez = namedtuple('prez', [
    'uuid',     # UUID: dataset in the document/workspace
    'kind',     # what kind of layer it is
    'visible',  # bool: whether it's visible or not
    'a_order',  # int: None for non-animating, 0..n-1 what order to draw in during animation
    'colormap', # name or uuid: color map to use; name for default, uuid for user-specified
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

    # signals
    didChangeLayer = pyqtSignal(dict)  # 'change' key is add/remove/visible
    didChangeLayerOrder = pyqtSignal(list)  # list of original indices in their new order, None for new layers
    didChangeLayerVisibility = pyqtSignal(list)  # list of (uuid,visible) tuples in layer order

    didSwitchLayerSet = pyqtSignal(int)  # new layerset number, typically 0..3

    didChangeColormap = pyqtSignal(dict)  # includes colormaps
    # didChangeShapeLayer = pyqtSignal(dict)

    def __init__(self, workspace, layer_set_count=DEFAULT_LAYER_SET_COUNT, **kwargs):
        super(Document, self).__init__(**kwargs)
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
        return None

    @property
    def current_layer_set(self):
        return self._layer_sets[self.current_set_index]

    def open_file(self, path, insert_before=0):
        """
        open an arbitrary file and make it the new top layer.
        emits docDidChangeLayer followed by docDidChangeLayerOrder
        :param path: file to open and add
        :return: overview (uuid:UUID, datasetinfo:dict, overviewdata:numpy.ndarray)
        """
        uuid, info, content = self._workspace.import_image(source_path=path)

        self._layer_with_uuid[uuid] = info

        # add as visible to the front of the current set, and invisible to the rest of the available sets
        colormap = self._default_colormap(info)
        p = prez(uuid=uuid,
                 kind=info["kind"],
                 visible=True,
                 a_order=None,
                 colormap=colormap,
                 mixing=mixing.NORMAL)
        q = p._replace(visible=False)
        old_layer_count = len(self._layer_sets[self.current_set_index])
        for dex,lset in enumerate(self._layer_sets):
            if lset is not None:  # uninitialized layer sets will be None
                lset.insert(insert_before, p if dex==self.current_set_index else q)

        # signal updates from the document
        self.didChangeLayer.emit({
            'change': 'add',
            'uuid': uuid,
            'info': info,
            'content': content,
            'order': 0
        })
        # express new layer order using old layer order indices
        reordered_indices = [None] + list(range(old_layer_count))
        self.didChangeLayerOrder.emit(reordered_indices)
        return uuid, info, content

    def update_dataset_info(self, new_info):
        """
        slot which updates document on new information workspace has provided us about a dataset
        typically signaled by importer operating in the workspace
        :param new_info: information dictionary including projection, levels of detail, etc
        :return: None
        """
        uuid = new_info['uuid']
        if uuid not in self._layer_with_uuid:
            LOG.warning('new information on uuid {0!r:s} is not for a known dataset'.format(new_info))
        self._layer_with_uuid[new_info['uuid']] = new_info
        # TODO: see if this affects any presentation information; view will handle redrawing on its own

    def _clone_layer_set(self, existing_layer_set):
        return existing_layer_set.deepcopy()

    def select_layer_set(self, layer_set_index):
        """
        change the selected layer set, 0..N (typically 0..3), cloning the old set if needed
        emits docDidChangeLayerOrder with an empty list implying complete reassessment,
          if cloning of layer set didn't occur
        :param layer_set_index: which layer set to switch to
        """
        assert(layer_set_index<len(self._layer_sets) and layer_set_index>=0)
        did_clone = False
        if self._layer_sets[layer_set_index] is None:
            self._layer_sets[layer_set_index] = self._clone_layer_set(self._layer_sets[self.current_set_index])
            did_clone = True
        self.current_set_index = layer_set_index
        self.didSwitchLayerSet.emit(layer_set_index)
        if not did_clone:
            self.didChangeLayerOrder.emit([])  # indicate that pretty much everything has changed

    def change_layer_order(self, old_index, new_index):
        L = self.current_layer_set
        order = list(range(len(L)))
        p = L[old_index]
        d = order[old_index]
        del L[old_index]
        del order[old_index]
        L.insert(new_index, p)
        L.insert(new_index, d)
        self.didChangeLayerOrder.emit(order)

    def swap_layer_order(self, first_index, second_index):
        L = self.current_layer_set
        order = list(range(len(L)))
        L[first_index], L[second_index] = L[second_index], L[first_index]
        order[first_index], order[second_index] = order[second_index], order[first_index]
        self.didChangeLayerOrder.emit(order)

    def toggle_layer_visibility(self, dex, visible=None):
        """
        change the visibility of a layer
        :param dex: layer index
        :param visible: True, False, or None (toggle)
        """
        L = self.current_layer_set
        old = L[dex]
        visible = ~old.visible if visible is None else visible
        nu = old._replace(visible=visible)
        L[dex] = nu
        self.didChangeLayer.emit({
            'change': 'visible',
            'visible': visible,
            'uuid': nu.uuid,
            'order': dex
        })
        lvl = [(x.uuid, x.visibility) for x in self.current_layer_set]
        self.didChangeLayerVisibility.emit(lvl)

    def is_layer_visible(self, dex):
        return self.current_layer_set[dex].visible

    def layer_animation_order(self, dex):
        return self.current_layer_set[dex].a_order

    def change_colormap_for_layers(self, name, uuids=None):
        L = self.current_layer_set
        if uuids is not None:
            # LOG.error('layer selection not implemented in change_colormap_for_layers')
            for uuid in uuids:
                nfo = {'uuid': uuid, 'colormap': name, 'change': 'colormap'}
                self.didChangeColormap.emit(nfo)
        else:  # all data layers
            uuids = []
            for dex in range(len(self.current_layer_set)):
                uuid = self.current_layer_set[dex].uuid
                nfo = {'uuid': uuid, 'colormap': name, 'change': 'colormap'}
                uuids.append(uuid)
                self.didChangeColormap.emit(nfo)
        for uuid in uuids:
            for dex,pinfo in enumerate(L):
                if pinfo.uuid==uuid:
                    L[dex] = pinfo._replace(colormap=name)

    def __len__(self):
        return len(self.current_layer_set)

    def uuid_for_layer(self, dex):
        uuid = self.current_layer_set[dex].uuid
        return uuid

    def get_info(self, dex=None):
        if dex is not None:
            uuid = self.current_layer_set[dex].uuid
            nfo = self._layer_with_uuid[uuid]
            return nfo
        return None

    def __getitem__(self, dex):
        """
        return info for a given layer index
        """
        return self.current_layer_set[dex]
        # uuid = self.current_layer_set[dex].uuid
        # nfo = self._layer_with_uuid[uuid]
        # return nfo

    def insert_layer_prez(self, row, layer_prez_seq):
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
        self.didChangeLayerOrder.emit(clo)

    def remove_layer_prez(self, row, count=1):
        clo = list(range(len(self.current_layer_set)))
        del clo[row:row+count]
        del self.current_layer_set[row:row+count]
        self.didChangeLayerOrder.emit(clo)



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


