#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
.py
~~~

PURPOSE
Document elements supporting Probes, including Shapes and Masks.


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
from PyQt4.QtCore import QObject, pyqtSignal

LOG = logging.getLogger(__name__)


class Shape(QObject):
    pass



class Probe(QObject):
    """
    A Probe transforms a Shape into a visualization widget aka ProbeView.
    Typically is attached to one or more layers - this is so that we can update when animation is taking place.
    It offers a delegate to the view that it's driving.
    """
    kind = None  # preferred output type

    didChangeAreaSelection = pyqtSignal()  # probably WKB or WKT representation of new shape? or shapely object?
    didChangeLayerSelection = pyqtSignal()
    didUpdateCalculatedContent = pyqtSignal()  #

    # kinds of delegates that will be supported
    HTML = 'html'
    MATPLOTLIB = 'matplotlib'
    VISPY = 'vispy'
    QT = 'qt'

    def recalculate(self, *args, **kwargs):
        """
        Slot used to recalculate content that delegates provide to the view
        :return:
        """
        return None

    def asViewDelegate(self, kind, **kwargs):
        """
        Delegate is
        :param kind:
        :param kwargs:
        :return: delegate object which a matched view will use to update its contents
        """
        assert(kind==self.kind)
        return None


class HtmlProbeViewDelegate(object):

    def set_widget(self, html_widget):
        raise NotImplementedError()

    def get_widget(self):
        return None

    widget = property(get_widget, set_widget)  # WebKit widget we'll be setting up and updating

    def update(self):
        """
        interact with WebKit widget to update or initialize content
        :return:
        """





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
