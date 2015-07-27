#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cspov.model.core
~~~~~~~~~~~~~~~~

PURPOSE
Core (low-level) document model for CSPOV.
The core is typically accessed via Facets, which are like database views for a specific group of use cases

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

import os, sys
import logging, unittest, argparse

LOG = logging.getLogger(__name__)

class CoreDocument(object):
    """
    low level queries
    event handling
    cross-process concurrency
    lazy updates
    """
    pass


class Source(object):
    """
    is effectively a URI containing data we wish to convert to a Resource and visualize
    a helper/plugin is often used to render the source into the workspace
    """

class Resource(object):
    """
    is a Source rendered in such a way that the display engine can realize it rapidly.
    """

class Layer(object):
    pass


class LayerStack(object):
    pass


class Shape(object):
    pass


class Tool(object):
    pass


class Transform(object):
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
