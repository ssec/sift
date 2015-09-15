#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

PURPOSE
Behaviors involving layer list controls
Layer list shows:
- visibility status
- animation order
- active color bar
- active enhancements or formulas, if any
- indicator colors

Layer stack actions:
- rearrange layers
- de/select one or more layers
- rename layer
- bring up context menu for layer

REFERENCES
http://pyqt.sourceforge.net/Docs/PyQt4/qabstractlistmodel.html
https://github.com/Werkov/PyQt4/blob/8cc9d541119226703668d34e06b430370fbfb482/examples/itemviews/simpledommodel.py

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
import weakref
from PyQt4.QtCore import QAbstractListModel
from PyQt4.QtGui import QAbstractItemDelegate

LOG = logging.getLogger(__name__)


class LayerWidgetDelegate(QAbstractItemDelegate):
    """
    set for a specific column, controls the rendering and editing of items in that column or row of a list or table
    see QAbstractItemView.setItemDelegateForRow/Column
    """
    pass


class LayerStackListViewModel(QAbstractListModel):
    """ behavior connecting list widget to layer stack (both ways)
    """
    widget = None
    doc = None

    def __init__(self, widget, doc):
        super(LayerStackListViewModel, self).__init__()
        self.widget = weakref.ref(widget)
        self.doc = weakref.ref(doc)
        widget.clicked.connect(self.layer_clicked)
        widget.indexesMoved.connect(self.layers_moved)
        widget.customContextMenuRequested.connect(self.context_menu)
        widget.entered.connect(self.layer_entered)
        widget.pressed.connect(self.layer_pressed)
        doc.docDidChangeLayerOrder.connect(self.updateList)
        doc.docDidChangeLayer.connect(self.updateList)
        widget.setModel(self)
        self.updateList()

    def columnCount(self, *args, **kwargs):
        return 1

    def rowCount(self, QModelIndex_parent=None, *args, **kwargs):
        el = self.doc().asListing()
        LOG.info('{} layers'.format(len(el)))
        return len(el)

    def data(self, index, int_role=None):
        el = self.doc().asListing()
        row = index.row()
        LOG.info('row {} is {}'.format(row, el[row]))
        return el[row]['name']

    # def flags(self, QModelIndex):
    #     return None
    #
    # def headerData(self, p_int, Qt_Orientation, int_role=None):
    #     return None
    #
    # def index(self, p_int, int_column=0, QModelIndex_parent=None, *args, **kwargs):
    #     return None
    #
    # def parent(self, child, *args, **kwargs):
    #     return None
    #
    def layer_clicked(self, qindex):
        pass

    def layers_moved(self, qindices):
        pass

    def layer_entered(self, qindex):
        pass

    def layer_pressed(self, qindex):
        pass

    def context_menu(self, qpoint):
        pass

    def updateList(self):
        pass
        # self.widget().clear()
        # for x in self.doc.asListing():
        #     self.widget().addItem(x['name'])

