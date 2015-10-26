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
import pickle as pkl
import base64
from PyQt4.QtCore import QAbstractListModel, QAbstractTableModel, QVariant, Qt, QSize, QModelIndex, QPoint, QMimeData
from PyQt4.QtGui import QAbstractItemDelegate, QListView, QStyledItemDelegate, QAbstractItemView, QMenu, QStyleOptionViewItem, QItemSelection, QItemSelectionModel
from cspov.model.document import Document
from cspov.common import INFO, KIND
from cspov.view.Colormap import ALL_COLORMAPS, CATEGORIZED_COLORMAPS

LOG = logging.getLogger(__name__)

COLUMNS=('Visibility', 'Name', 'Enhancement')

class LayerWidgetDelegate(QStyledItemDelegate):
    """
    set for a specific column, controls the rendering and editing of items in that column or row of a list or table
    see QAbstractItemView.setItemDelegateForRow/Column
    """

    def sizeHint(self, option:QStyleOptionViewItem, index:QModelIndex):
        return QSize(100,36)

    # def paint(self, painter, option, index):
    #
    #     painter.save()
    #
    #     painter.setPen(QtGui.QPen(QtCore.Qt.NoPen))
    #
    #     if option.state & QtGui.QStyle.State_Selected:
    #         brush = QtGui.QBrush(QtGui.QColor("#66ff71"))
    #         painter.setBrush(brush)
    #
    #     else:
    #         brush = QtGui.QBrush(QtCore.Qt.white)
    #         painter.setBrush(brush)
    #
    #     painter.drawRect(option.rect)
    #
    #     painter.setPen(QtGui.QPen(QtCore.Qt.blue))
    #     value = index.data(QtCore.Qt.DisplayRole)
    #
    #     if value.isValid():
    #
    #         text = value.toString()
    #         align = QtCore.Qt.AlignCenter
    #         painter.drawText(option.rect, align, text)
    #
    #     #QtGui.QStyledItemDelegate.paint(self, painter, option, index)
    #     painter.restore()


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
        ref: http://duganchen.ca/a-pythonic-qt-list-model-implementation/
        http://doc.qt.io/qt-5/qabstractitemmodel.html#beginMoveRows
    """
    widgets = None
    doc = None
    item_delegate = None
    _mimetype = 'application/vnd.row.list'

    def __init__(self, widgets:list, doc:Document):
        """
        Connect one or more table views to the document via this model.
        :param widgets: list of TableViews to wire up
        :param doc: document to communicate with
        :return:
        """
        super(LayerStackListViewModel, self).__init__()
        self.widgets = list(widgets) # [weakref.ref(widget) for widget in widgets]
        self.doc = doc
        # self._column = [self._visibilityData, self._nameData]
        self.item_delegate = LayerWidgetDelegate()

        # for now, a copout by just having a refresh to the content when document changes
        doc.didReorderLayers.connect(self.refresh)
        doc.didRemoveLayers.connect(self.drop_layers_just_removed)
        doc.didChangeColormap.connect(self.refresh)
        doc.didChangeLayerVisibility.connect(self.refresh)
        doc.didChangeLayerName.connect(self.refresh)
        doc.didAddLayer.connect(self.doc_added_layer)
        doc.willPurgeLayer.connect(self.refresh)
        doc.didSwitchLayerSet.connect(self.refresh)
        doc.didReorderAnimation.connect(self.refresh)

        # self.setSupportedDragActions(Qt.MoveAction)

        for widget in widgets:
            self._init_widget(widget)

    def _init_widget(self, listbox:QListView):
        listbox.setModel(self)
        listbox.setItemDelegate(self.item_delegate)
        listbox.setContextMenuPolicy(Qt.CustomContextMenu)
        # listbox.customContextMenuRequested.connect(self.context_menu)
        listbox.customContextMenuRequested.connect(self.menu)
        listbox.setDragEnabled(True)
        listbox.setAcceptDrops(True)
        listbox.setDropIndicatorShown(True)
        listbox.setSelectionMode(listbox.ExtendedSelection)
        # listbox.indexesMoved.connect(FIXME)
        # listbox.setMovement(QListView.Snap)
        # listbox.setDragDropMode(QListView.InternalMove)
        listbox.setDragDropMode(QAbstractItemView.DragDrop)
        # listbox.setDefaultDropAction(Qt.MoveAction)
        # listbox.setDragDropOverwriteMode(False)
        # listbox.clicked.connect(self.layer_clicked)
        # listbox.entered.connect(self.layer_entered)
        # listbox.pressed.connect(self.layer_pressed)
        self.widgets.append(listbox)

    # def supportedDragActions(self):
    #     return Qt.MoveAction
    #
    def supportedDropActions(self):
        return Qt.MoveAction # | Qt.CopyAction

    @property
    def current_set_listbox(self):
        """
        We can have several list boxes, one for each layer_set in the document.
        Return whichever one is currently active.
        :return:
        """
        # FIXME this is fugly
        for widget in self.widgets:
            if widget.isVisible():
                return widget

    def doc_added_layer(self, new_order, info, content):
        dexes = [i for i,q in enumerate(new_order) if q==None]
        # for dex in dexes:
        #     self.beginInsertRows(QModelIndex(), dex, dex)
        #     self.endInsertRows()
        self.refresh()

    def refresh(self):
        # self.beginResetModel()
        # self.endResetModel()
        self.layoutAboutToBeChanged.emit()
        self.revert()
        self.layoutChanged.emit()

    def current_selected_uuids(self, lbox:QListView=None):
        lbox = self.current_set_listbox if lbox is None else lbox
        if lbox is None:
            LOG.error('not sure which list box is active! oh pooh.')
            return
        for q in lbox.selectedIndexes():
            yield self.doc.uuid_for_layer(q.row())

    def select(self, uuids, lbox:QListView=None, scroll_to_show_single=True):
        lbox = self.current_set_listbox if lbox is None else lbox
        lbox.clearSelection()
        if not uuids:
            return
        # FUTURE: this is quick and dirty
        rowdict = dict((u,i) for i,u in enumerate(self.doc.current_layer_order))
        items = QItemSelection()
        q = None
        for uuid in uuids:
            row = rowdict.get(uuid, None)
            if row is None:
                LOG.error('UUID {} cannot be selected in list view'.format(uuid))
                continue
            q = self.createIndex(row, 0)
            items.select(q, q)
            lbox.selectionModel().select(items, QItemSelectionModel.Select)
            # lbox.setCurrentIndex(q)
        if scroll_to_show_single and len(uuids)==1 and q is not None:
            lbox.scrollTo(q)

    def drop_layers_just_removed(self, layer_indices, uuid, row, count):
        """
        a layer was removed in the document, update the listview
        :param layer_indices: list of new layer indices
        :param uuid:
        :return:
        """
        self.refresh()
        # self.removeRows(row, count)

    def menu(self, pos:QPoint, *args):
        LOG.info('menu requested for layer list')
        menu = QMenu()
        actions = {}
        lbox = self.current_set_listbox
        # XXX: Normally we would create the menu and actions before hand but since we are checking the actions based
        # on selection we can't. Then we would use an ActionGroup and make it exclusive
        selected_uuids = list(self.current_selected_uuids(lbox))
        LOG.debug("selected UUID set is {0!r:s}".format(selected_uuids))
        current_colormaps = set(self.doc.colormap_for_uuids(selected_uuids))
        for cat, cat_colormaps in CATEGORIZED_COLORMAPS.items():
            submenu = QMenu(cat, parent=menu)
            for colormap in cat_colormaps.keys():
                action = submenu.addAction(colormap)
                actions[action] = colormap
                action.setCheckable(True)
                action.setChecked(colormap in current_colormaps)
            menu.addMenu(submenu)
        menu.addSeparator()
        flip_action = menu.addAction("Flip Color Limits")
        flip_action.setCheckable(True)
        flip_action.setChecked(any(self.doc.flipped_for_uuids(selected_uuids)))
        menu.addAction(flip_action)
        sel = menu.exec_(lbox.mapToGlobal(pos))
        if sel is flip_action:
            LOG.info("flipping color limits for sibling ids {0!r:s}".format(selected_uuids))
            self.doc.flip_climits_for_layers(uuids=selected_uuids)
        else:
            new_cmap = actions.get(sel, None)
            if new_cmap is not None:
                LOG.info("changing to colormap {0} for ids {1!r:s}".format(new_cmap, selected_uuids))
                self.doc.change_colormap_for_layers(name=new_cmap, uuids=selected_uuids)

    @property
    def listing(self):
        return [self.doc.get_info(dex) for dex in range(len(self.doc))]

    def dropMimeData(self, mime:QMimeData, action, row:int, column:int, parent:QModelIndex):
        LOG.debug('dropMimeData at row {}'.format(row))
        if action == Qt.IgnoreAction:
            return True

        if mime.hasFormat('text/uri-list'):
            if mime.hasUrls():
                LOG.debug('found urls in drop!')
                for qurl in mime.urls():
                    LOG.debug(qurl.path())
                    if qurl.isLocalFile():
                        path = qurl.path()
                        self.doc.open_file(path)
                return True
        elif mime.hasFormat(self._mimetype):
            # unpickle the presentation information and re-insert it
            # b = base64.decodebytes(mime.text())
            b = mime.data(self._mimetype)
            layer_set_len, insertion_info = pkl.loads(b)
            LOG.debug('dropped: {0!r:s}'.format(insertion_info))
            count = len(insertion_info)
            if row == -1:
                row = len(self.doc)  # append
            # self.insertRows(row, count)
            # for i, presentation in enumerate(l):
            #     self.setData(self.index(row+i, 0), presentation)
            order = list(range(layer_set_len))
            inserted_row_numbers = []
            # inserted_presentations = []
            # delete_these_rows = []
            insertion_point = row
            for old_row, presentation in insertion_info:
                del order[old_row]
                # if old_row<insertion_point:
                #     insertion_point -= 1
                inserted_row_numbers.append(old_row)
                # delete_these_rows.append(old_row if old_row<row else old_row+count)
                # inserted_presentations.append(presentation)
            order = order[:insertion_point] + inserted_row_numbers + order[insertion_point:]
            LOG.debug('new order after drop {0!r:s}'.format(order))
            self.doc.reorder_by_indices(order)
            # self.doc.insert_layer_prez(row, inserted_presentations)
            # LOG.debug('after insertion removing rows {0!r:s}'.format(delete_these_rows))
            # for exrow in delete_these_rows:
            #     self.doc.remove_layer_prez(exrow)
            # self.doc.didReorderLayers.emit(order)  # FUTURE: not our business to be emitting on behalf of the document
            assert(count==len(insertion_info))
            return True
        return False
        # return super(LayerStackListViewModel, self).dropMimeData(mime, action, row, column, parent)

    def mimeData(self, list_of_QModelIndex):
        l = []
        for index in list_of_QModelIndex:
            # create a list of prez tuples showing how layers are presented
            if index.isValid():
                l.append((index.row(), self.doc[index.row()]))
        p = pkl.dumps((len(self.doc.current_layer_set), l), pkl.HIGHEST_PROTOCOL)
        mime = QMimeData()
        # t = base64.encodebytes(p).decode('ascii')
        # LOG.debug('mimetext for drag is "{}"'.format(t))
        mime.setData(self._mimetype, p)
        LOG.debug('presenting mime data for {0!r:s}'.format(l))
        return mime

    def mimeTypes(self):
        return ['text/uri-list', self._mimetype]  # ref https://github.com/shotgunsoftware/pyqt-uploader/blob/master/uploader.py

    def flags(self, index):
        # flags = super(LayerStackListViewModel, self).flags(index)
        if index.isValid():
            flags = (Qt.ItemIsEnabled |
                     Qt.ItemIsSelectable |
                     Qt.ItemIsDragEnabled |
                     Qt.ItemIsUserCheckable |
                     Qt.ItemIsEditable)
        else:
            flags = Qt.ItemIsDropEnabled
        return flags

    def rowCount(self, QModelIndex_parent=None, *args, **kwargs):
        # LOG.debug('{} layers'.format(len(self.doc)))
        return len(self.doc)

    def data(self, index:QModelIndex, role:int=None):
        if not index.isValid():
            return None
        row = index.row()
        # col = index.column()
        el = self.listing
        if role == Qt.ItemDataRole or role == Qt.EditRole:
            return self.doc[index.row()] if index.row()<len(self.doc) else None
        elif role == Qt.CheckStateRole:
            check = Qt.Checked if self.doc.is_layer_visible(row) else Qt.Unchecked
            return check
        elif role == Qt.DisplayRole:
            lao = self.doc.layer_animation_order(row)
            name = el[row][INFO.NAME]
            # return  ('[-]  ' if lao is None else '[{}]'.format(lao+1)) + el[row]['name']
            return name
        return None

    def setData(self, index:QModelIndex, data, role:int=None):
        LOG.debug('setData {0!r:s}'.format(data))
        if not index.isValid():
            return False
        if role == Qt.EditRole:
            if isinstance(data, str):
                LOG.debug("changing row {0:d} name to {1!r:s}".format(index.row(), data))
                if not data:
                    LOG.warning("skipping rename to nothing")
                else:
                    self.doc.change_layer_name(index.row(), data)
                    self.dataChanged.emit(index, index)
                return True
            else:
                LOG.debug("data type is {0!r:s}".format(type(data)))
        elif role == Qt.CheckStateRole:
            newvalue = True if data==Qt.Checked else False
            LOG.debug('toggle layer visibility for row {} to {}'.format(index.row(), newvalue))
            self.doc.toggle_layer_visibility(index.row(), newvalue)
            self.dataChanged.emit(index, index)
            return True
        elif role==Qt.ItemDataRole:
            LOG.warning('attempting to change layer')
            # self.doc.replace_layer()
            # FIXME implement this
            self.dataChanged.emit(index, index)
            return True
        elif role==Qt.DisplayRole:
            if index.isValid():
                LOG.debug("changing row {} name to {0!r:s}".format(index.row(), data))
                self.doc.change_layer_name(index.row(), data)
                return True
        return False

    def insertRows(self, row, count, parent=QModelIndex()):
        self.beginInsertRows(QModelIndex(), row, row+count-1)
        LOG.debug(">>>> INSERT {} rows".format(count))
        # TODO: insert 'count' empty rows into document
        self.endInsertRows()
        return True

    def removeRows(self, row, count, QModelIndex_parent=None, *args, **kwargs):
        self.beginRemoveRows(QModelIndex(), row, row+count-1)
        LOG.debug(">>>> REMOVE {} rows at {}".format(count, row))
        # self.doc.remove_layer_prez(row, count)
        self.endRemoveRows()
        return True

    # def dragEnterEvent(self, event):
    #     if event.mimeData().hasUrls:
    #         event.accept()
    #
    #     else:
    #         event.ignore()
    #
    # def dragMoveEvent(self, event):
    #     if event.mimeData().hasUrls:
    #         event.setDropAction(QtCore.Qt.CopyAction)
    #         event.accept()
    #
    #     else:
    #         event.ignore()
    #
    # def dropEvent(self, event):
    #     if event.mimeData().hasUrls:
    #         event.setDropAction(QtCore.Qt.CopyAction)
    #         event.accept()
    #
    #         filePaths = [
    #             str(url.toLocalFile())
    #             for url in event.mimeData().urls()
    #         ]
    #
    #         self.dropped.emit(filePaths)
    #
    #     else:
    #         event.ignore()


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

        # self.widget().clear()
        # for x in self.doc.asListing():
        #     self.widget().addItem(x['name'])

