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
from _weakref import ref
from collections import MutableMapping
from enum import Enum

from cspov.common import INFO, KIND

__author__ = 'rayg'
__docformat__ = 'reStructuredText'

import os, sys
import logging, unittest, argparse

LOG = logging.getLogger(__name__)

# class LayerStack(list):
#     """
#     A LayerStack includes
#     - stacking of layers
#     - reordering of layers
#     - time-ordered grouping of similar layers
#     - time sequence animation
#     Layer types includes
#     - background images
#     - RGB imagery tiles from GeoTIFF or other
#     - Imagery field tiles with color maps
#     - markers and selections
#     - geopolitical vector boundaries
#
#     Everything in LayerDoc should be serializable in order to load an identical configuration later.
#     LayerDoc interacts with view classes to manage drawing plan and layer representations.
#     Scripts and UI Controls can modify the LayerDoc, which propagates update signals to its dependencies
#     Facets - LayerDocAsTimeSeries for instance, will be introduced as needed to manage complexity
#     Multiple LayerDocs may be present, especially if we're copying content like color maps over
#     """
#
#     _layerlist = None
#
#
#     def __init__(self, document, init_layers=None):
#         super(LayerStack, self).__init__()
#         self._layerlist = list(init_layers or [])
#
#
#     def __iter__(self):
#         # FIXME: this is an animation test pattern
#         if self.animating:
#             todraw = self._layerlist[self.frame_number % len(self._layerlist)]
#             todraw.z = 0
#             yield todraw
#             return
#         for level,layer in enumerate(reversed(self._layerlist)):
#             layer.z = float(level)
#             yield layer
#
#
#     def __getitem__(self, item):
#         return self._layerlist[item]
#
#
#     def __len__(self):
#         return len(self._layerlist)
#
#
#     def append(self, layer):
#         self._layerlist.append(layer)
#         # self.layerStackDidChangeOrder.emit(tuple(range(len(self))))
#
#
#     def __delitem__(self, dex):
#         order = list(range(len(self)))
#         del self._layerlist[dex]
#         del order[dex]
#         # self.layerStackDidChangeOrder(tuple(order))
#
#
#     def swap(self, adex, bdex):
#         order = list(range(len(self)))
#         order[bdex], order[adex] = adex, bdex
#         new_list = [self._layerlist[dex] for dex in order]
#         self._layerlist = new_list
#         # self.layerStackDidChangeOrder.emit(tuple(order))
#
#
#
#     # @property
#     # def top(self):
#     #     return self._layerlist[0] if self._layerlist else None
#     #
#
#     @property
#     def listing(self):
#         """
#         return representation summary for layer list - name, icon, source, etc
#         """
#         for layer in self._layerlist:
#             yield {'name': str(layer.name)}
#



class mixing(Enum):
    UNKNOWN = 0
    NORMAL = 1
    ADD = 2
    SUBTRACT = 3


class DocLayer(MutableMapping):
    """
    Layer as represented within the document
    Essentially: A helper representation of a layer and part of the public interface to the document.
    Substitutes in for document layer information dictionaries initially,
      but eventually dictionary interfaces will be limited to non-standard annotation keys.
    Incrementally migrate functionality from Document into these classes as appropriate, to keep Document from getting too complicated.
    Make sure that DocLayer classes remain serializable and long-term connected only to the document, not to UI elements or workspace.
    """
    _doc = None  # weakref to document that owns us

    def __init__(self, doc, *args, **kwargs):
        # assert (isinstance(doc, Document))
        self._doc = ref(doc)
        self._store = dict()
        self.update(dict(*args, **kwargs))  # use the free update to set keys

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
        return self._store[INFO.UUID]

    @property
    def kind(self):
        """
        which kind of layer it is - RGB, Algebraic, etc. This can also be tested by the class of the layer typically.
         We may deprecate this eventually?
        :return:
        """
        return self._store[INFO.KIND]

    @property
    def name(self):
        return self._store[INFO.NAME]

    @property
    def is_valid(self):
        """
        invalid layers cannot be displayed (are never visible)
        valid layers may or may not be visible
        visibility is managed by the scenegraph
        validity is managed by the document
        example of an invalid layer: an RGB or algebraic layer that's insufficiently specified to actually display,
        be it through lack of data or lack of projection information
        however, an invalid layer may still be configurable in order to allow it to become valid and then visible
        :return: bool
        """
        return True

    @property
    def is_flat_field(self):
        """
        return whether the layer can be represented as a flat numerical field or not (RGB layers cannot)
        :return: bool
        """
        return True

    @property
    def range(self):
        """
        :return: (min, max) numerical range of the data or (None, None)
        """
        return (None, None)

    def __getitem__(self, key):
        return self._store[self.__keytransform__(key)]

    def __setitem__(self, key, value):
        self._store[self.__keytransform__(key)] = value

    def __delitem__(self, key):
        del self._store[self.__keytransform__(key)]

    def __iter__(self):
        return iter(self._store)

    def __len__(self):
        return len(self._store)

    def __keytransform__(self, key):
        return key


class DocBasicLayer(DocLayer):
    """
    A layer consistent of a simple scalar floating point value field, which can have color maps applied
    """
    pass


class DocCompositeLayer(DocLayer):
    """
    A layer which combines other layers, be they basic or composite themselves
    """
    def __getitem__(self, key):
        # FIXME debug
        if key==INFO.KIND:
            assert(self._store[INFO.KIND]==KIND.RGB)
        return self._store[self.__keytransform__(key)]


class DocRGBLayer(DocCompositeLayer):
    l = [None, None, None, None]  # RGBA upstream layers
    n = [None, None, None, None]  # RGBA minimum value from upstream layers
    x = [None, None, None, None]  # RGBA maximum value from upstream layers

    @property
    def r(self):
        return self.l[0]
    @property
    def g(self):
        return self.l[1]
    @property
    def b(self):
        return self.l[2]
    @property
    def a(self):
        return self.l[3]

    @r.setter
    def r(self, x):
        self.l[0] = x
    @g.setter
    def g(self, x):
        self.l[1] = x
    @b.setter
    def b(self, x):
        self.l[2] = x
    @a.setter
    def a(self, x):
        self.l[3] = x

    @property
    def is_valid(self):
        return (self.r is not None and
                self.g is not None and
                self.b is not None)

    @property
    def is_flat_field(self):
        return False



class DocAlgebraicLayer(DocCompositeLayer):
    """
    A value field derived from other value fields algebraically
    """
    pass



# class DocMapLayer(DocLayer):
#     """
#     FUTURE: A layer containing a background map as vector
#     """
#     pass
#
# class DocShapeLayer(DocLayer):
#     """
#     FUTURE: A layer represented in the scene graph as an editable shape
#     """
#     pass
#
# class DocProbeLayer(DocShapeLayer):
#     """
#     FUTURE: A shape layer which feeds probe values to another UI element or helper.
#     """



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

