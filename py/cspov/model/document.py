#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cspov.model.core
~~~~~~~~~~~~~~~~

PURPOSE
Core (low-level) document model for CSPOV.
The core is sometimes accessed via Facets, which are like database views for a specific group of use cases

The document model is a metadata representation which permits the workspace to be constructed and managed.


REFERENCES


REQUIRES
sqlite
sqlalchemy


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

from PyQt4.QtCore import QObject, pyqtSignal

# FIXME: move these out of the document and into a factory
from ..view.LayerRep import TiledGeolocatedImage, NEShapefileLines
from ..view.Program import GlooColormapDataTile, GlooRGBImageTile

from .probes import Probe, Shape


LOG = logging.getLogger(__name__)

DEFAULT_LAYER_SET_COUNT = 4  # this should match the ui configuration!

# presentation information for a layer
prez = namedtuple('prez', [
    'uuid',     # UUID: dataset in the document/workspace
    'visible',  # bool: whether it's visible or not
    'order',    # int: animation order, 1..N or None/False/0 if non-animating
    'enhancement',  # weakref: to enhancement it's observed through
    'siblings'  # set(UUID): of other datasets contributing to the enhancement (e.g. RGB/RGBA enhancements)
])


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
    _available = None  # dict(uuid:datasetinfo)

    # signals
    docDidChangeLayer = pyqtSignal(dict)  # add/remove/alter
    docDidChangeLayerOrder = pyqtSignal(list)  # list of original indices in their new order, None for new layers
    docDidChangeEnhancement = pyqtSignal(dict)  # includes colormaps
    docDidChangeShape = pyqtSignal(dict)

    def __init__(self, workspace, layer_set_count=DEFAULT_LAYER_SET_COUNT, **kwargs):
        super(Document, self).__init__(**kwargs)
        self._workspace = workspace
        self._layer_sets = [list()] + [None] * (layer_set_count-1)
        self._available = {}
        # TODO: connect signals from workspace to slots including update_dataset_info

    def _default_enhancement(self, datasetinfo):
        """
        consult guidebook and user preferences for which enhancement should be used for a given datasetinfo
        :param datasetinfo: dictionary of metadata about dataset
        :return: enhancement info and siblings participating in the enhancement
        """
        return None, None

    @property
    def current_layer_set(self):
        return self._layer_sets[self.current_set_index]

    def open_file(self, path):
        """
        open an arbitrary file and make it the new top layer.
        emits docDidChangeLayer followed by docDidChangeLayerOrder
        :param path: file to open and add
        :return: overview (uuid:UUID, datasetinfo:dict, overviewdata:numpy.ndarray)
        """
        uuid, info, content = self._workspace.import_image(source_path=path)

        self._available[uuid] = info

        # add as visible to the front of the current set, and invisible to the rest of the available sets
        enhancement, siblings = self._default_enhancement(info)
        p = prez(uuid=uuid,
                 visible=True,
                 order=None,
                 enhancement=enhancement,
                 siblings=siblings)
        q = prez(uuid=uuid,
                 visible=False,
                 order=None,
                 enhancement=enhancement,
                 siblings=siblings)
        old_layer_count = len(self._layer_sets[self.current_set_index])
        for dex,lset in enumerate(self._layer_sets):
            if lset is not None:  # uninitialized layer sets will be None
                lset.insert(0, p if dex==self.current_set_index else q)

        # signal updates from the document
        self.docDidChangeLayer.emit({
            'change': 'add',
            'uuid': uuid,
            'info': info,
            'content': content,
            'order': 0
        })
        # express new layer order using old layer order indices
        reordered_indices = [None] + list(range(old_layer_count))
        self.docDidChangeLayerOrder.emit(reordered_indices)
        return uuid, info, content

    def update_dataset_info(self, new_info):
        """
        slot which updates document on new information workspace has provided us about a dataset
        :param new_info: information dictionary including projection, levels of detail, etc
        :return: None
        """
        uuid = new_info['uuid']
        if uuid not in self._available:
            LOG.warning('new information on uuid {0!r:s} is not for a known dataset'.format(new_info))
        self._available[new_info['uuid']] = new_info
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
        if not did_clone:
            self.docDidChangeLayerOrder.emit([])  # indicate that pretty much everything has changed

    def change_layer_order(self, old_index, new_index):
        L = self.current_layer_set
        order = list(range(len(L)))
        p = L[old_index]
        d = order[old_index]
        del L[old_index]
        del order[old_index]
        L.insert(new_index, p)
        L.insert(new_index, d)
        self.docDidChangeLayerOrder.emit(order)

    def swap_layer_order(self, first_index, second_index):
        L = self.current_layer_set
        order = list(range(len(L)))
        L[first_index], L[second_index] = L[second_index], L[first_index]
        order[first_index], order[second_index] = order[second_index], order[first_index]
        self.docDidChangeLayerOrder.emit(order)

    def toggle_layer_visibility(self, dex, visible=None):
        """
        change the visibility of a layer
        :param dex: layer index
        :param visible: True, False, or None (toggle)
        """
        L = self.current_layer_set
        old = L[dex]
        visible = ~old.visible if visible is None else visible
        nu = prez(
            uuid=old.uuid,
            visible=visible,
            order=old.order,
            enhancement=old.enhancement,
            siblings=old.siblings
        )
        L[dex] = nu
        self.docDidChangeLayer.emit({
            'change': 'visible',
            'uuid': nu.uuid,
            'order': dex
        })

    def __len__(self):
        return len(self.current_layer_set)

    def __getitem__(self, dex):
        """
        return prez for a given layer index
        """
        uuid = self.current_layer_set[dex].uuid
        nfo = self._available[uuid]
        return nfo

    # def asDrawingPlan(self, frame=None):
    #     """
    #     delegate callable yielding a sequence of LayerReps to draw
    #     """
    #     if frame is not None:
    #         yield self._layer_reps[frame % len(self._layer_reps)]
    #         return
    #     for layer_rep in self._layer_reps:
    #         yield layer_rep


    # def asListing(self):
    #     return [{'name': q.name} for q in self._layer_reps]

    #
    # def addRGBImageLayer(self, filename, range=None):
    #     rep = TiledGeolocatedImage(filename, tile_class=GlooRGBImageTile)
    #     self._layer_reps.append(rep)
    #     self.docDidChangeLayer.emit({'filename': filename})



    # def addFullGlobMercatorColormappedFloatImageLayer(self, filename, range=None):
    #     rep = TiledGeolocatedImage(filename, tile_class=GlooColormapDataTile, range=range)
    #     self._layer_reps.append(rep)
    #     self.docDidChangeLayer.emit({'filename': filename})

    # def addShapeLayer(self, filename, **kwargs):
    #     # FIXME: Figure out what the required arguments and stuff are
    #     rep = NEShapefileLines(filename)
    #     self._layer_reps.append(rep)
    #     self.docDidChangeLayer.emit({'filename': filename})

#
    def asProbeGuidance(self, **kwargs):
        """
        Retrieve delegate to be used by Probe objects to access and update the data selection (lasso et cetera)
        """
        return None

#     def swap(self, adex, bdex):
#         order = list(range(len(self)))
#         order[bdex], order[adex] = adex, bdex
#         new_list = [self._layerlist[dex] for dex in order]
#         self._layerlist = new_list
#         # self.layerStackDidChangeOrder.emit(tuple(order))
#


class Preferences(QObject):
    """
    Preferences doc. Holds many of the same resources, but not a layer stack.
    """
    pass


class DocElement(QObject):
    pass


class Source(DocElement):
    """
    is effectively a URI containing data we wish to convert to a Resource and visualize
    a helper/plugin is often used to render the source into the workspace
    """

class Dataset(DocElement):
    """
    is a Source rendered in such a way that the display engine can realize it rapidly.
    """

class Layer(DocElement):
    pass


class Shape(DocElement):
    pass


class Tool(DocElement):
    pass


class Transform(DocElement):
    """
    Metadata describing a transform of multiple Resources to a virtual Resource
    """

class ColorMap(Transform):
    pass




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


