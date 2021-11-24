from PyQt5 import QtWidgets, QtCore, QtGui


class LayerTreeView(QtWidgets.QTreeView):
    def __init__(self, *args, **kwargs):
        super(LayerTreeView, self).__init__(*args, **kwargs)
        self.setDragEnabled(QtWidgets.QAbstractItemView.InternalMove)

    def rowsInserted(self, parent: QtCore.QModelIndex, start: int, end: int)\
            -> None:

        super(LayerTreeView, self).rowsInserted(parent, start, end)
        for idx in range(start, end + 1):
            model_idx = self.model().index(idx, 0, parent)

            if not self.isPersistentEditorOpen(model_idx):
                self.openPersistentEditor(model_idx)
