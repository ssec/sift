#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
layers.py
~~~~~~~~~

PURPOSE
Document Layer & LayerDoc

REFERENCES


REQUIRES


:author: R.K.Garcia <rayg@ssec.wisc.edu>
:copyright: 2014 by University of Wisconsin Regents, see AUTHORS for more details
:license: GPLv3, see LICENSE for more details
"""
__author__ = 'rayg'
__docformat__ = 'reStructuredText'

import os, sys
import logging, unittest, argparse
from QtCore import QObject

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
