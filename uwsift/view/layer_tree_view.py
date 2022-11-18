import logging
from uuid import UUID

from PyQt5.QtCore import QModelIndex, Qt, pyqtSignal
from PyQt5.QtWidgets import QAbstractItemView, QHeaderView, QMenu, QTreeView

from uwsift.common import Info, Kind
from uwsift.common import LayerModelColumns as LMC  # noqa
from uwsift.common import Platform
from uwsift.model.layer_item import LayerItem
from uwsift.util.widgets.pie_dial import PieDialDelegate
from uwsift.view.colormap_dialogs import ChangeColormapDialog

LOG = logging.getLogger(__name__)


class LayerTreeView(QTreeView):
    layerSelectionChanged = pyqtSignal(tuple)

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

    def _open_layer_context_menu(self, position):

        menu = QMenu()
        selection_model_idx = self.selectionModel().currentIndex()

        model_idx = self.model().index(selection_model_idx.row(), 0, QModelIndex())
        if not model_idx.isValid():
            raise ValueError(f"Entry at row {selection_model_idx.row()}" f" not in model")

        layer: LayerItem = self.model().layers[model_idx.row()]
        actions = {}
        if layer is not None:
            if layer.kind in [Kind.IMAGE, Kind.COMPOSITE]:
                actions.update(self._change_layer_colormap_menu(menu, layer.uuid))
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

    def _change_layer_colormap_menu(self, menu: QMenu, selected_uuid: UUID):
        model = self.model()

        def _show_change_colormap_dialog(action):  # noqa
            d = ChangeColormapDialog(model, selected_uuid, parent=self)
            d.show()
            d.raise_()
            d.activateWindow()

        action = menu.addAction("Change Colormap...")
        return {action: _show_change_colormap_dialog}

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

    def begin_layers_removal(self, *args, **kwargs):
        """
        Triggers the process that the model removes the current selected Rows out of itself.
        """
        self.model().remove_layers(self.selectionModel().selectedRows())
