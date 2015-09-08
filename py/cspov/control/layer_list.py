#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

PURPOSE
Behaviors involving layer list controls

REFERENCES


REQUIRES


:author: R.K.Garcia <rayg@ssec.wisc.edu>
:copyright: 2014 by University of Wisconsin Regents, see AUTHORS for more details
:license: GPLv3, see LICENSE for more details
"""
from PyQt4.QtCore import QObject

__author__ = 'rayg'
__docformat__ = 'reStructuredText'

import os, sys
import logging, unittest, argparse

LOG = logging.getLogger(__name__)


class ListWidgetMatchesLayerStack(QObject):
    """ behavior connecting list widget to layer stack (both ways)
    """
    widget = None
    doc = None

    def __init__(self, widget, doc):
        super(ListWidgetMatchesLayerStack, self).__init__()
        self.widget = widget
        self.doc = doc
        self.updateList()
        doc.docDidChangeLayerOrder.connect(self.updateList)
        doc.docDidChangeLayer.connect(self.updateList)
        # FIXME: connect and configure list widget signals

    def updateList(self):
        self.widget.clear()
        for x in self.doc.asListing():
            self.widget.addItem(x['name'])

