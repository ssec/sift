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

class LayerDoc(QObject):
    """
    A LayerDoc includes
    - stacking of layers
    - reordering of layers
    - time-ordered grouping of similar layers
    - time sequence animation
    Layer types includes
    - background images
    - RGB imagery tiles from GeoTIFF or other
    - Imagery field tiles with color maps
    - markers and selections
    - geopolitical vector boundaries

    Everything in LayerDoc should be serializable in order to load an identical configuration later.
    LayerDoc interacts with view classes to manage drawing plan and layer representations.
    Scripts and UI Controls can modify the LayerDoc, which propagates update signals to its dependencies
    Facets - LayerDocAsTimeSeries for instance, will be introduced as needed to manage complexity
    Multiple LayerDocs may be present, especially if we're copying content like color maps over
    """

    # FIXME migrate function from .view.LayerDrawingPlan and make LDP closer to a dumb playlist
    # FIXME have main program hook LayerDoc to LayerDrawingPlan


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
