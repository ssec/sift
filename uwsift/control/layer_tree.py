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

__author__ = 'rayg'
__docformat__ = 'reStructuredText'

import logging
import pickle as pkl
import sys

from PyQt5.QtCore import (QAbstractItemModel, Qt, QSize, QModelIndex, QPoint, QMimeData,
                          pyqtSignal, QRect, QItemSelection, QItemSelectionModel)
from PyQt5.QtGui import QColor, QFont, QPen
from PyQt5.QtWidgets import (QTreeView, QStyledItemDelegate, QAbstractItemView,
                             QMenu, QStyle, QStyleOptionViewItem, QActionGroup, QAction)

from uwsift.common import Info, Kind
from uwsift.model.document import Document
from uwsift.view.colormap_dialogs import ChangeColormapDialog

LOG = logging.getLogger(__name__)

COLUMNS = ('Visibility', 'Name', 'Enhancement')

CELL_HEIGHT = 36 if 'darwin' in sys.platform else 48
CELL_WIDTH = 128 if 'darwin' in sys.platform else 160
LEFT_OFFSET = 28 if 'darwin' in sys.platform else 32
TOP_OFFSET = 3


class LayerWidgetDelegate(QStyledItemDelegate):
    """
    set for a specific column, controls the rendering and editing of items in that column or row of a list or table
    see QAbstractItemView.setItemDelegateForRow/Column
    """
    _doc: Document = None  # document we're representing

    # _doc: DocumentAsLayerStack = None  # document we're representing

    def layer_prez(self, index: int):
        cll = self._doc.current_layer_set
        return cll[index] if index < len(cll) and index >= 0 else None

    def __init__(self, doc: Document, *args, **kwargs):
        super(LayerWidgetDelegate, self).__init__(*args, **kwargs)
        self._doc = doc
        # self._doc = doc.as_layer_stack
        self.font = QFont('Andale Mono')
        self.font.setPointSizeF(12)

    def sizeHint(self, option: QStyleOptionViewItem, index: QModelIndex):
        # pz = self.layer_prez(index.row())
        # if pz.kind == Kind.RGB:
        #     LOG.debug('triple-sizing composite layer')
        #     return QSize(CELL_WIDTH, CELL_HEIGHT*3)
        # else:
        return QSize(CELL_WIDTH, CELL_HEIGHT)

    def displayText(self, *args, **kwargs):
        return None

    def paint(self, painter, option, index):
        """draw the individual lines in the layers control
        """
        painter.save()

        color = QColor(187, 213, 255, 255) if index.row() % 2 == 0 else QColor(177, 223, 255, 255)
        # color = QColor(187, 213, 255, 255)
        painter.setPen(QPen(color))
        painter.setFont(self.font)
        value = index.data(Qt.UserRole)
        text = index.data(Qt.DisplayRole)
        rect = option.rect

        # if we have a value, break out the animation order and other info
        animation_order = None
        if value:
            value, animation_order = value

        # if we have a point probe value, draw the filled bar to represent where it is in that layer's data range
        if value:
            value, bar, fmtd_str = value
            width = bar * float(rect.width())
            right = QRect(rect.left(), rect.top(), int(width), rect.height())
            painter.fillRect(right, color)

        super(LayerWidgetDelegate, self).paint(painter, option, index)

        # if this layer is selected, draw a colored rectangle to highlight it
        if option.state & QStyle.State_Selected and value:
            painter.fillRect(right, QColor(213, 187, 255, 96))

        # draw the name of the layer
        painter.setPen(QPen(Qt.black))
        painter.setFont(self.font)
        bounds = painter.drawText(rect.left() + LEFT_OFFSET,
                                  rect.top() + TOP_OFFSET,
                                  rect.width() - LEFT_OFFSET,
                                  int(CELL_HEIGHT / 2 - TOP_OFFSET),
                                  Qt.AlignLeft,
                                  text,
                                  )

        # also draw the animation order
        if animation_order is not None:
            painter.setPen(QPen(Qt.white))
            ao_rect = QRect(bounds.right(),
                            rect.top() + TOP_OFFSET,
                            rect.width() - bounds.right(),
                            int(CELL_HEIGHT / 2 - TOP_OFFSET),
                            )
            # draw the text once to get the bounding rectangle
            bounds = painter.drawText(ao_rect,
                                      Qt.AlignRight,
                                      str(animation_order + 1),
                                      )
            painter.fillRect(bounds, Qt.black)
            # draw the text a second time to make sure it appears in the rectangle
            painter.drawText(ao_rect,
                             Qt.AlignRight,
                             str(animation_order + 1),
                             )

        # if we have a point probe value, draw text with it's value
        if value:
            painter.setPen(Qt.darkBlue)
            theight = CELL_HEIGHT / 2
            top = rect.top() + rect.height() - theight
            if width < rect.width() / 3:  # place the text to the right of the bar instead of inside
                left = max(int(width), LEFT_OFFSET)
                right = rect.width()
                align = Qt.AlignLeft
            else:
                left = 0
                right = width
                align = Qt.AlignRight
            painter.drawText(left, top, right - left, theight, align, fmtd_str)

        painter.restore()

    # def paint(self, painter, option, index):
    #     '''
    #     Paint a checkbox without the label.
    #     from http://stackoverflow.com/questions/17748546/pyqt-column-of-checkboxes-in-a-qtableview
    #     '''
    #     checked = index.model().data(index, QtCore.Qt.DisplayRole) == 'True'
    #     check_box_style_option = QtGui.QStyleOptionButton()
    #
    #     if (index.Flags() & QtCore.Qt.ItemIsEditable) > 0:
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
    #     if not (index.Flags() & QtCore.Qt.ItemIsEditable) > 0:
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


class LayerStackTreeViewModel(QAbstractItemModel):
    """Behavior connecting list widget to layer stack (both ways)

    Each table view represents a different configured document layer stack "set" - user can select from at least four.
    Convey layer set information to/from the document to the respective table, including selection.

    References:

        - http://duganchen.ca/a-pythonic-qt-list-model-implementation/
        - http://doc.qt.io/qt-5/qabstractitemmodel.html#beginMoveRows
        - http://pyqt.sourceforge.net/Docs/PyQt4/qabstractitemmodel.html
        - http://doc.qt.io/qt-5/qtwidgets-itemviews-simpletreemodel-example.html

    """
    widgets = None
    doc = None
    item_delegate = None
    _last_equalizer_values = {}
    _mimetype = 'application/vnd.row.list'

    # signals
    uuidSelectionChanged = pyqtSignal(tuple)  # the list is a list of the currently selected UUIDs
    didRequestRGBCreation = pyqtSignal(dict)

    def __init__(self, widgets: list, doc: Document, parent=None):
        """
        Connect one or more table views to the document via this model.
        :param widgets: list of TableViews to wire up
        :param doc: document to communicate with
        :return:
        """
        super(LayerStackTreeViewModel, self).__init__(parent)

        self.widgets = []
        self.doc = doc
        # self._column = [self._visibilityData, self._nameData]
        self.item_delegate = LayerWidgetDelegate(doc)
        # FIXME: Reset colormap change dialog when layer set changes

        # for now, a copout by just having a refresh to the content when document changes
        doc.didReorderLayers.connect(self.refresh)
        doc.didRemoveLayers.connect(self.drop_layers_just_removed)
        doc.didChangeColormap.connect(self.refresh)
        # doc.didChangeColorLimits.connect(self.refresh)
        doc.didChangeLayerVisibility.connect(self.refresh)
        doc.didChangeLayerName.connect(self.refresh)
        doc.didAddBasicLayer.connect(self.doc_added_basic_layer)
        doc.didAddCompositeLayer.connect(self.refresh)
        doc.willPurgeLayer.connect(self.refresh)
        doc.didSwitchLayerSet.connect(self.refresh)
        doc.didReorderAnimation.connect(self.refresh)
        doc.didCalculateLayerEqualizerValues.connect(self.update_equalizer)

        # self.setSupportedDragActions(Qt.MoveAction)

        # set up each of the widgets
        for widget in widgets:
            self._init_widget(widget)

    # TODO, this wrapper is probably not needed, possibly remove later
    def add_widget(self, listbox: QTreeView):
        self._init_widget(listbox)

    def _init_widget(self, listbox: QTreeView):
        listbox.setModel(self)
        listbox.setItemDelegate(self.item_delegate)
        listbox.setContextMenuPolicy(Qt.CustomContextMenu)
        # listbox.customContextMenuRequested.connect(self.context_menu)
        listbox.customContextMenuRequested.connect(self.menu)
        listbox.setDragEnabled(True)
        listbox.setAcceptDrops(True)
        listbox.setDropIndicatorShown(True)
        listbox.setSelectionMode(listbox.ExtendedSelection)
        # listbox.setMovement(QTreeView.Snap)
        # listbox.setDragDropMode(QTreeView.InternalMove)
        listbox.setDragDropMode(QAbstractItemView.DragDrop)
        # listbox.setAlternatingRowColors(True)
        # listbox.setDefaultDropAction(Qt.MoveAction)
        # listbox.setDragDropOverwriteMode(True)
        # listbox.entered.connect(self.layer_entered)
        # listbox.setFont(QFont('Andale Mono', 13))

        # the various signals that may result from the user changing the selections
        # listbox.activated.connect(self.changedSelection)
        # listbox.clicked.connect(self.changedSelection)
        # listbox.doubleClicked.connect(self.changedSelection)
        # listbox.pressed.connect(self.changedSelection)
        listbox.selectionModel().selectionChanged.connect(self.changedSelection)

        self.widgets.append(listbox)

    # def supportedDragActions(self):
    #     return Qt.MoveAction
    #
    def supportedDropActions(self):
        return Qt.MoveAction | Qt.LinkAction

    def changedSelection(self, *args):
        """connected to the various listbox signals that represent the user changing selections
        """
        selected_uuids = tuple(self.current_selected_uuids(self.current_set_listbox))
        # FUTURE: this is needed in order to prevent selection display artifacts. why?
        self.current_set_listbox.update()
        self.uuidSelectionChanged.emit(selected_uuids)

    @property
    def current_set_listbox(self):
        """
        We can have several list boxes, one for each layer_set in the document.
        Return whichever one is currently active.
        :return:
        """
        # FUTURE this is brute force and could be tracked
        for widget in self.widgets:
            if widget.isVisible():
                return widget

    def doc_added_basic_layer(self, new_order, layer, presentation):
        # dexes = [i for i,q in enumerate(new_order) if q==None]
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

        # this is an ugly way to make sure the selection stays current
        try:
            self.changedSelection(None)
            self.current_set_listbox.update()
        except IndexError:
            pass

    def current_selected_uuids(self, lbox: QTreeView = None):
        lbox = self.current_set_listbox if lbox is None else lbox
        if lbox is None:
            LOG.error('not sure which list box is active! oh pooh.')
            return
        for q in lbox.selectedIndexes():
            yield self.doc.uuid_for_current_layer(q.row())

    def select(self, uuids, lbox: QTreeView = None, scroll_to_show_single=True):
        lbox = self.current_set_listbox if lbox is None else lbox
        lbox.clearSelection()
        if not uuids:
            return
        # FUTURE: this is quick and dirty
        rowdict = dict((u, i) for i, u in enumerate(self.doc.current_layer_uuid_order))
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
        if scroll_to_show_single and len(uuids) == 1 and q is not None:
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

    def update_equalizer(self, doc_values):
        """
        User has clicked on a point probe
        Document is conveniently providing us values for all the image layers
        Let's update the display to show them like a left-to-right equalizer
        :param doc_values: {uuid: value, value-relative-to-base, is-base:bool}, values can be NaN
        :return:
        """
        if not doc_values:
            # turn off all equalizer values
            self._last_equalizer_values = {}
        else:
            self._last_equalizer_values.update(doc_values)
        self.refresh()

    def change_layer_colormap_menu(self, menu, lbox: QTreeView, selected_uuids: list, *args):
        def _show_change_colormap_dialog(action):
            d = ChangeColormapDialog(self.doc, selected_uuids[0], parent=lbox)
            d.show()
            d.raise_()
            d.activateWindow()

        action = menu.addAction('Change Colormap...')
        return {action: _show_change_colormap_dialog}

    def composite_layer_menu(self, menu, lbox: QTreeView, selected_uuids: list, *args):
        """
        provide common options for RGB or other composite layers, eventually with option to go to a compositing dialog

        """
        actions = {}
        requests = {}
        if len(selected_uuids) > 3 or \
                any(self.doc[u][Info.KIND] not in [Kind.IMAGE, Kind.COMPOSITE] for u in selected_uuids):
            LOG.warning('Need 3 image layers to create a composite')
            return {}

        def _make_rgb_composite(action, requests=requests):
            request = requests.get(action, None)
            if request is not None:
                LOG.debug('RGB creation using {0!r:s}'.format(request))
                self.didRequestRGBCreation.emit(request)

        rgb_menu = QMenu("Create RGB From Selections...", menu)
        for rgb in sorted(set(x[:len(selected_uuids)] for x in ['RGB', 'RBG', 'GRB', 'GBR', 'BRG', 'BGR']),
                          reverse=True):
            # Only include the number of channels selected
            rgb = rgb[:len(selected_uuids)]
            action = rgb_menu.addAction(rgb)
            request = dict((channel.lower(), uuid) for (channel, uuid) in zip(rgb, selected_uuids))
            actions[action] = _make_rgb_composite
            requests[action] = request
        menu.addMenu(rgb_menu)
        return actions

    def change_layer_image_kind_menu(self, menu, lbox, selected_uuids, *args):
        current_kind = self.doc.prez_for_uuid(selected_uuids[0]).kind
        kind_menu = QMenu("Change Image Kind", menu)
        action_group = QActionGroup(menu, exclusive=True)
        actions = {}
        action_kinds = {}

        def _change_layers_image_kind(action, action_kinds=action_kinds):
            if not action.isChecked():
                # can't uncheck an image kind
                LOG.debug("Selected Kind action is not checked")
                return
            kind = action_kinds[action]
            return self.doc.change_layers_image_kind(selected_uuids, kind)

        for kind in [Kind.IMAGE, Kind.CONTOUR]:
            action = action_group.addAction(QAction(kind.name, menu, checkable=True))
            action_kinds[action] = kind
            action.setChecked(kind == current_kind)
            actions[action] = _change_layers_image_kind
            kind_menu.addAction(action)

        menu.addMenu(kind_menu)
        return actions

    def menu(self, pos: QPoint, *args):
        lbox = self.current_set_listbox
        selected_uuids = list(self.current_selected_uuids(lbox))
        LOG.debug("selected UUID set is {0!r:s}".format(selected_uuids))
        menu = QMenu()
        actions = {}
        if len(selected_uuids) == 1:
            if self.doc[selected_uuids[0]][Info.KIND] in [Kind.IMAGE, Kind.COMPOSITE, Kind.CONTOUR]:
                actions.update(self.change_layer_colormap_menu(menu, lbox, selected_uuids, *args))
            if self.doc[selected_uuids[0]][Info.KIND] in [Kind.CONTOUR]:
                actions.update(self.change_layer_image_kind_menu(menu, lbox, selected_uuids, *args))
        if 0 < len(selected_uuids) <= 3:
            if all(self.doc[u][Info.KIND] in [Kind.IMAGE, Kind.COMPOSITE]
                   for u in selected_uuids):
                actions.update(self.composite_layer_menu(
                    menu, lbox, selected_uuids, *args))

        if not actions:
            action = menu.addAction("No actions available for this layer")
            action.setEnabled(False)

        sel = menu.exec_(lbox.mapToGlobal(pos))
        if sel is None:
            return
        elif sel in actions:
            return actions[sel](sel)
        else:
            LOG.debug("Unimplemented menu option '{}'".format(sel.text()))

    def columnCount(self, QModelIndex_parent=None, *args, **kwargs):
        return 1

    #
    # def hasChildren(self, QModelIndex_parent=None, *args, **kwargs):
    #     return False  # FIXME

    def headerData(self, section: int, Qt_Orientation, role=None):
        pass

    @property
    def listing(self):
        return [self.doc.get_info(dex) for dex in range(len(self.doc))]

    def dropMimeData(self, mime: QMimeData, action, row: int, column: int, parent: QModelIndex):
        LOG.debug('dropMimeData at row {}'.format(row))
        if action == Qt.IgnoreAction:
            return True

        if mime.hasFormat('text/uri-list'):
            if mime.hasUrls():
                LOG.debug('found urls in drop!')
                paths = [qurl.path() for qurl in mime.urls() if qurl.isLocalFile()]
                self.doc.import_files(paths)  # FIXME: replace with a signal
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
                # FIXME: row=col=-1 implies drop-on-parent
                #  which may mean replace or may mean append for composite layers
            # self.insertRows(row, count)
            # for i, presentation in enumerate(l):
            #     self.setData(self.index(row+i, 0), presentation)
            order = list(range(layer_set_len))
            inserted_row_numbers = []
            # inserted_presentations = []
            # delete_these_rows = []
            insertion_point = row
            uuids = []
            for old_row, presentation in reversed(sorted(insertion_info)):
                del order[old_row]
                if old_row < insertion_point:
                    insertion_point -= 1
                inserted_row_numbers.insert(0, old_row)
                uuids.append(presentation.uuid)
                # delete_these_rows.append(old_row if old_row<row else old_row+count)
                # inserted_presentations.append(presentation)
            order = order[:insertion_point] + inserted_row_numbers + order[insertion_point:]
            LOG.debug('new order after drop {0!r:s}'.format(order))
            self.select([])
            self.doc.reorder_by_indices(order)
            # self.doc.insert_layer_prez(row, inserted_presentations)
            # LOG.debug('after insertion removing rows {0!r:s}'.format(delete_these_rows))
            # for exrow in delete_these_rows:
            #     self.doc.remove_layer_prez(exrow)
            # self.doc.didReorderLayers.emit(order)  # FUTURE: not our business to be emitting on behalf of the document
            assert (count == len(insertion_info))
            return True
        return False
        # return super(LayerStackListViewModel, self).dropMimeData(mime, action, row, column, parent)

    def mimeData(self, list_of_QModelIndex):
        valid_rows = []
        for index in list_of_QModelIndex:
            # create a list of Presentation tuples showing how layers are presented
            if index.isValid():
                valid_rows.append((index.row(), self.doc.current_layer_set[index.row()]))
        p = pkl.dumps((len(self.doc.current_layer_set), valid_rows), pkl.HIGHEST_PROTOCOL)
        mime = QMimeData()
        # t = base64.encodebytes(p).decode('ascii')
        # LOG.debug('mimetext for drag is "{}"'.format(t))
        mime.setData(self._mimetype, p)
        LOG.debug('presenting mime data for {0!r:s}'.format(valid_rows))
        return mime

    def mimeTypes(self):
        return ['text/uri-list',
                self._mimetype]  # ref https://github.com/shotgunsoftware/pyqt-uploader/blob/master/uploader.py

    # http://stackoverflow.com/questions/6942098/qt-qtreeview-only-allow-to-drop-on-an-existing-item
    # Reimplement the Flags method of the underlying model to return
    #     Qt::ItemIsDropEnabled only if passed index is valid.
    # When in between items, Flags() is called with an invalid index so I can decide not to accept the drop
    # For the inverse you can do this:
    #     if ( index.isValid() ) { return Qt::ItemIsSelectable | Qt::ItemIsEnabled | Qt::ItemIsDragEnabled; }
    # else { return Qt::ItemIsSelectable | Qt::ItemIsDragEnabled | Qt::ItemIsDropEnabled | Qt::ItemIsEnabled; }
    def flags(self, index):
        # Flags = super(LayerStackListViewModel, self).Flags(index)
        if index.isValid():
            flags = (Qt.ItemIsEnabled |
                     Qt.ItemIsSelectable |
                     Qt.ItemIsDragEnabled |
                     Qt.ItemIsUserCheckable |
                     Qt.ItemIsEditable)
        else:
            flags = Qt.ItemIsDropEnabled
        return flags

    def hasIndex(self, row, col, QModelIndex_parent=None, *args, **kwargs):
        if QModelIndex_parent.isValid() or col != 0:
            # then look up whether this layer has child layers, e.g. for RGB or algebraic
            # actually: current layer list ignores RGB/algebraic child layers
            return False
        return (row >= 0 and row < len(self.doc))

    def rowCount(self, QModelIndex_parent=None, *args, **kwargs):
        # LOG.debug('{} layers'.format(len(self.doc)))
        if QModelIndex_parent is None or QModelIndex_parent == QModelIndex():
            return len(self.doc)
        return 0

    def index(self, row, column, parent):
        if self.hasIndex(row, column, parent):
            return self.createIndex(row, column, parent)
        else:
            return QModelIndex()

    def parent(self, index=None):
        return QModelIndex()
        # FIXME
        # if not index.isValid():
        #     return QModelIndex()
        #
        # childItem = index.internalPointer()
        # if not childItem:
        #     return QModelIndex()
        #
        # parentItem = childItem.parent()
        #
        # if parentItem == self.rootItem:
        #     return QModelIndex()
        #
        # return self.createIndex(parentItem.row(), 0, parentItem)

    def data(self, index: QModelIndex, role: int = None):
        if not index.isValid():
            return None
        row = index.row()
        # LOG.debug("getting data for row %d" % row)
        # col = index.column()
        el = self.listing
        info = el[row] if row < len(self.doc) else None
        if not info:
            return None

        # pass auxiliary info about the layer through the Qt.UserRole for use when displaying
        eq_content = self._last_equalizer_values.get(info[Info.UUID], None)
        if role == Qt.UserRole:
            # get the animation order also
            animation_order = self.doc.layer_animation_order(row)
            return (eq_content, animation_order)

        elif role == Qt.EditRole:
            return self.doc.current_layer_set[index.row()] if index.row() < len(self.doc) else None
        elif role == Qt.CheckStateRole:
            check = Qt.Checked if self.doc.is_layer_visible(row) else Qt.Unchecked
            return check
        elif role == Qt.ToolTipRole:
            if not eq_content:
                return None
            value, normalized = eq_content[:2]
            return str(value)
        elif role == Qt.DisplayRole:
            # lao = self.doc.layer_animation_order(row)
            name = info[Info.DISPLAY_NAME]
            # return  ('[-]  ' if lao is None else '[{}]'.format(lao+1)) + el[row]['name']
            # if leroy:
            #     data = '[%.2f] ' % leroy[0]
            #     return data + name
            return name
        return None

    def setData(self, index: QModelIndex, data, role: int = None):
        LOG.debug('setData {0!r:s}'.format(data))
        if not index.isValid():
            return False
        if role == Qt.EditRole:
            if isinstance(data, str):
                LOG.debug("changing row {0:d} name to {1!r:s}".format(index.row(), data))
                self.doc.change_layer_name(index.row(), data)
                self.dataChanged.emit(index, index)
                return True
            else:
                LOG.debug("data type is {0!r:s}".format(type(data)))
        elif role == Qt.CheckStateRole:
            newvalue = True if data == Qt.Checked else False
            LOG.debug('toggle layer visibility for row {} to {}'.format(index.row(), newvalue))
            self.doc.toggle_layer_visibility(index.row(), newvalue)
            self.dataChanged.emit(index, index)
            return True
        elif role == Qt.ItemDataRole:
            LOG.warning('attempting to change layer')
            # self.doc.replace_layer()
            # FIXME implement this
            self.dataChanged.emit(index, index)
            return True
        elif role == Qt.DisplayRole:
            if index.isValid():
                LOG.debug("changing row {0} name to {1!r:s}".format(index.row(), data))
                self.doc.change_layer_name(index.row(), data)
                return True
        return False

    def insertRows(self, row, count, parent=None):
        if parent is None:
            parent = QModelIndex()
        self.beginInsertRows(QModelIndex(), row, row + count - 1)
        LOG.debug(">>>> INSERT {} rows".format(count))
        # TODO: insert 'count' empty rows into document
        self.endInsertRows()
        return True

    def removeRows(self, row, count, QModelIndex_parent=None, *args, **kwargs):
        self.beginRemoveRows(QModelIndex(), row, row + count - 1)
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

    # self.widget().clear()
    # for x in self.doc.asListing():
    #     self.widget().addItem(x['name'])
