import logging

from PyQt5 import QtGui, QtWidgets
from PyQt5.QtCore import QModelIndex, Qt, pyqtSignal
from PyQt5.QtWidgets import QAbstractItemView, QHeaderView, QMenu, QTreeView

from uwsift.common import Info
from uwsift.common import LayerModelColumns as LMC  # noqa
from uwsift.common import LayerVisibility, Platform
from uwsift.model.layer_item import LayerItem
from uwsift.util.widgets.pie_dial import PieDialDelegate
from uwsift.view.probes import DEFAULT_POINT_PROBE

LOG = logging.getLogger(__name__)


class LayerTreeView(QTreeView):
    layerSelectionChanged = pyqtSignal(tuple)
    selectedLayerForProbeChanged = pyqtSignal(str)

    def __init__(self, *args, **kwargs):
        super(LayerTreeView, self).__init__(*args, **kwargs)

        self.setHeaderHidden(False)
        self.header().setStretchLastSection(False)
        self.header().setSectionResizeMode(QHeaderView.ResizeToContents)

        self.setRootIsDecorated(False)

        self.setSelectionMode(QAbstractItemView.SingleSelection)

        # Strangely enough, the following 4 settings do not seem to have any
        # effect...
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDefaultDropAction(Qt.MoveAction)
        self.setDropIndicatorShown(True)
        self.setDragDropMode(QAbstractItemView.InternalMove)

        self.visibility_delegate = PieDialDelegate(parent=self)
        self.setItemDelegateForColumn(LMC.VISIBILITY, self.visibility_delegate)
        self.setEditTriggers(QAbstractItemView.AllEditTriggers)

        self.customContextMenuRequested.connect(self._open_layer_context_menu)

        # Set extra keyboard shortcuts
        self._decrease_opacity_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+Shift+Left"), self)
        self._decrease_opacity_shortcut.setContext(Qt.WidgetShortcut)
        self._decrease_opacity_shortcut.activated.connect(self._decrease_layer_opacity)

        self._increase_opacity_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+Shift+Right"), self)
        self._increase_opacity_shortcut.setContext(Qt.WidgetShortcut)
        self._increase_opacity_shortcut.activated.connect(self._increase_layer_opacity)

        self._move_selected_layer_up_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+Shift+Up"), self)
        self._move_selected_layer_up_shortcut.setContext(Qt.WidgetShortcut)
        self._move_selected_layer_up_shortcut.activated.connect(self._simulate_drag_and_drop_layer_up)

        self._move_selected_layer_down_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+Shift+Down"), self)
        self._move_selected_layer_down_shortcut.setContext(Qt.WidgetShortcut)
        self._move_selected_layer_down_shortcut.activated.connect(self._simulate_drag_and_drop_layer_down)

    def resizeColumnsToContents(self) -> None:  # noqa
        """Resize all columns to their current contents of the model."""
        # Note:  Currently this method is useless because of the
        # sectionResizeMode == ResizeToContents setting for the header (see
        # __init__()), but if users prefer the ability to manually change
        # column sizes, this can be useful to create an initial layout or
        # provide a way to reset column sizes.

        if not self.model():
            return
        for column in range(self.model().columnCount()):
            self.resizeColumnToContents(column)

    def rowsInserted(self, parent: QModelIndex, start: int, end: int) -> None:
        super(LayerTreeView, self).rowsInserted(parent, start, end)
        for idx in range(start, end + 1):
            model_idx = self.model().index(idx, 0, parent)

            if not self.isPersistentEditorOpen(model_idx):
                self.openPersistentEditor(model_idx)

            self.setCurrentIndex(model_idx)

    def _open_layer_context_menu(self, position):
        menu = QMenu()
        selection_model_idx = self.selectionModel().currentIndex()

        model_idx = self.model().index(selection_model_idx.row(), 0, QModelIndex())
        if not model_idx.isValid():
            raise ValueError(f"Entry at row {selection_model_idx.row()}" f" not in model")

        layer: LayerItem = self.model().layers[model_idx.row()]
        actions = {}
        if layer is not None:
            if layer.info.get(Info.PLATFORM) != Platform.SYSTEM:
                actions.update(self._delete_layer_menu(menu, selection_model_idx))

        if not actions:
            action = menu.addAction("No actions available for this layer")
            action.setEnabled(False)

        sel = menu.exec_(self.mapToGlobal(position))
        if sel is None:
            return
        elif sel in actions:
            return actions[sel](sel)
        else:
            LOG.debug("Unimplemented menu option '{}'".format(sel.text()))

    def _delete_layer_menu(self, menu: QMenu, selection_model_idx: QModelIndex):
        model = self.model()

        def _remove_layer(action):
            model.remove_layers([selection_model_idx])

        action = menu.addAction("Remove Layer")
        return {action: _remove_layer}

    def currentChanged(self, current: QModelIndex, previous: QModelIndex) -> None:
        # TODO: This is not Qt's default way of handling selections and the
        #  "current item", but for now - as we are not (yet) interested in
        #  multiple selections - it is sufficient. We may need to revise this
        #  and switch to using the QItemSelectionModel interface in the future.

        super(LayerTreeView, self).currentChanged(current, previous)
        if len(self.model().layers) > 0:
            selected_layer: tuple = (self.model().layers[current.row()],)
            self.layerSelectionChanged.emit(selected_layer)
            self.selectedLayerForProbeChanged.emit(DEFAULT_POINT_PROBE)

    def begin_layers_removal(self, *args, **kwargs):
        """
        Triggers the process that the model removes the current selected Rows out of itself.
        """
        self.model().remove_layers(self.selectionModel().selectedRows())

    def _decrease_layer_opacity(self):
        index = self.model().index(self.currentIndex().row(), LMC.VISIBILITY)
        if index.row() >= 0:
            layer: LayerItem = self.model().layers[index.row()]
            opacity = layer.opacity - 0.05

            if layer.opacity >= 0.0:
                if opacity < 0.0:
                    opacity = 0.0
                new_layer_visibility = LayerVisibility(layer.visible, opacity)
                self.model().setData(index, new_layer_visibility)

    def _increase_layer_opacity(self):
        index = self.model().index(self.currentIndex().row(), LMC.VISIBILITY)
        if index.row() >= 0:
            layer: LayerItem = self.model().layers[index.row()]
            opacity = layer.opacity + 0.05

            if layer.opacity <= 1.0:
                if opacity > 1.0:
                    opacity = 1.0
                new_layer_visibility = LayerVisibility(layer.visible, opacity)
                self.model().setData(index, new_layer_visibility)

    def _simulate_drag_and_drop_layer_up(self):
        index = self.currentIndex()
        mime_data = self.model().mimeData([index])
        parent_index = self.model().index(-1, -1)
        row = index.row() - 1 if index.row() > 0 else 0
        self.model().dropMimeData(mime_data, Qt.MoveAction, row, index.column(), parent_index)

    def _simulate_drag_and_drop_layer_down(self):
        index = self.currentIndex()
        mime_data = self.model().mimeData([index])
        parent_index = self.model().index(-1, -1)
        row = index.row() + 2 if index.row() + 2 < len(self.model().layers) else -1
        self.model().dropMimeData(mime_data, Qt.MoveAction, row, index.column(), parent_index)
