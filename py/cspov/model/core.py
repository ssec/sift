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

from PyQt4.QtCore import QObject, pyqtSignal
from ..view.LayerDrawingPlan import LayerDrawingPlan

LOG = logging.getLogger(__name__)

class VizDoc(QObject):
    """
    low level queries
    event handling
    cross-process concurrency
    lazy updates
    """
    docDidChangeLayer = pyqtSignal(str)  # add/remove
    docDidChangeLayerOrder = pyqtSignal(str)
    docDidChangeEnhancement = pyqtSignal(str)  # includes colormaps
    docDidChangeShape = pyqtSignal(str)

    _drawing_plan = None

    @property
    def asLayerDrawingPlan(self):
        return self._drawing_plan


    def __init__(self, **kwargs):
        super(VizDoc, self).__init__(**kwargs)
        self._drawing_plan = LayerDrawingPlan()




class PrefsDoc(QObject):
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


class LayerStack(DocElement):
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


