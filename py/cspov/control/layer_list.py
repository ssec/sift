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
from PyQt4.QtCore import QAbstractListModel, QVariant, Qt, QSize
from PyQt4.QtGui import QAbstractItemDelegate, QListView, QStyledItemDelegate

LOG = logging.getLogger(__name__)

COLUMNS=('Visibility', 'Name', 'Animation')

class LayerWidgetDelegate(QStyledItemDelegate):
    """
    set for a specific column, controls the rendering and editing of items in that column or row of a list or table
    see QAbstractItemView.setItemDelegateForRow/Column
    """
    def paint(self, painter, style, index):
        """
        Draw the item in the model
        :param painter:
        :param style:
        :param index:
        :return:
        """
        super(LayerWidgetDelegate, self).paint(painter, style, index)

    def sizeHint(self, QStyleOptionViewItem, QModelIndex):
        return QSize(100,36)

    # def paint(self, painter, option, index):
    #     '''
    #     Paint a checkbox without the label.
    #     from http://stackoverflow.com/questions/17748546/pyqt-column-of-checkboxes-in-a-qtableview
    #     '''
    #     checked = index.model().data(index, QtCore.Qt.DisplayRole) == 'True'
    #     check_box_style_option = QtGui.QStyleOptionButton()
    #
    #     if (index.flags() & QtCore.Qt.ItemIsEditable) > 0:
    #         check_box_style_option.state |= QtGui.QStyle.State_Enabled
    #     else:
    #         check_box_style_option.state |= QtGui.QStyle.State_ReadOnly
    #
    #     if checked:
    #         check_box_style_option.state |= QtGui.QStyle.State_On
    #     else:
    #         check_box_style_option.state |= QtGui.QStyle.State_Off
    #
    #     check_box_style_option.rect = self.getCheckBoxRect(option)
    #
    #     # this will not run - hasFlag does not exist
    #     #if not index.model().hasFlag(index, QtCore.Qt.ItemIsEditable):
    #         #check_box_style_option.state |= QtGui.QStyle.State_ReadOnly
    #
    #     check_box_style_option.state |= QtGui.QStyle.State_Enabled
    #
    #     QtGui.QApplication.style().drawControl(QtGui.QStyle.CE_CheckBox, check_box_style_option, painter)
    #
    # def editorEvent(self, event, model, option, index):
    #     '''
    #     Change the data in the model and the state of the checkbox
    #     if the user presses the left mousebutton or presses
    #     Key_Space or Key_Select and this cell is editable. Otherwise do nothing.
    #     '''
    #     print 'Check Box editor Event detected : '
    #     if not (index.flags() & QtCore.Qt.ItemIsEditable) > 0:
    #         return False
    #
    #     print 'Check Box edior Event detected : passed first check'
    #     # Do not change the checkbox-state
    #     if event.type() == QtCore.QEvent.MouseButtonRelease or event.type() == QtCore.QEvent.MouseButtonDblClick:
    #         if event.button() != QtCore.Qt.LeftButton or not self.getCheckBoxRect(option).contains(event.pos()):
    #             return False
    #         if event.type() == QtCore.QEvent.MouseButtonDblClick:
    #             return True
    #     elif event.type() == QtCore.QEvent.KeyPress:
    #         if event.key() != QtCore.Qt.Key_Space and event.key() != QtCore.Qt.Key_Select:
    #             return False
    #         else:
    #             return False
    #
    #     # Change the checkbox-state
    #     self.setModelData(None, model, index)
    #     return True


class LayerStackListViewModel(QAbstractListModel):
    """ behavior connecting list widget to layer stack (both ways)
        Each table view represents a different configured document layer stack "set" - user can select from at least four.
        Convey layer set information to/from the document to the respective table, including selection.
    """
    widget = None
    doc = None
    item_delegate = None

    def __init__(self, widgets, doc):
        """
        Connect one or more table views to the document via this model.
        :param widgets: list of TableViews to wire up
        :param doc: document to communicate with
        :return:
        """
        super(LayerStackListViewModel, self).__init__()
        self.widgets = list(widgets) # [weakref.ref(widget) for widget in widgets]
        self.doc = doc
        self.column = [self._visibilityData, self._nameData, self._animationData]
        self.item_delegate = LayerWidgetDelegate()

        doc.docDidChangeLayerOrder.connect(self.updateList)
        doc.docDidChangeLayer.connect(self.updateList)

        for widget in widgets:
            widget.clicked.connect(self.layer_clicked)
            widget.indexesMoved.connect(self.layers_moved)
            widget.customContextMenuRequested.connect(self.context_menu)
            widget.entered.connect(self.layer_entered)
            widget.pressed.connect(self.layer_pressed)
            widget.setModel(self)
            widget.setItemDelegate(self.item_delegate)


    # def columnCount(self, *args, **kwargs):
    #     return 1

    @property
    def listing(self):
        return [self.doc[dex] for dex in range(len(self.doc))]

    def rowCount(self, QModelIndex_parent=None, *args, **kwargs):
        LOG.info('{} layers'.format(len(self.doc)))
        return len(self.doc)

    # def columnCount(self, QModelIndex_parent=None, *args, **kwargs):
    #     return len(COLUMNS)

    def _visibilityData(self, row, listing):
        return True  # FIXME

    def _nameData(self, row, listing):
        return listing[row]['name']

    def _animationData(self, row, listing):
        return row+1  # FIXME

    def data(self, index, int_role=None):
        if not index.isValid():
            return None
        if int_role==Qt.ItemDataRole:
            return self.doc[index.row()] if index.row()<len(self.doc) else None
        elif int_role!=Qt.DisplayRole:
            return None
        # return "test"
        el = self.listing

        row = index.row()
        # col = index.column()
        col = 1
        LOG.debug('row,col {},{} is {}'.format(row, col, el[row]))
        return self.column[col](row, el)


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

