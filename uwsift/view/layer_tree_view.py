from PyQt5.QtCore import QModelIndex, Qt
from PyQt5.QtWidgets import QAbstractItemView, QHeaderView, QTreeView

from uwsift.common import LayerModelColumns as LMC  # noqa
from uwsift.util.widgets.pie_dial import PieDialDelegate


class LayerTreeView(QTreeView):
    def __init__(self, *args, **kwargs):
        super(LayerTreeView, self).__init__(*args, **kwargs)

        self.setHeaderHidden(False)

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
        self.setItemDelegateForColumn(LMC.VISIBILITY,
                                      self.visibility_delegate)
        self.setEditTriggers(QAbstractItemView.AllEditTriggers)

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

    def rowsInserted(self, parent: QModelIndex, start: int, end: int)\
            -> None:

        super(LayerTreeView, self).rowsInserted(parent, start, end)
        for idx in range(start, end + 1):
            model_idx = self.model().index(idx, 0, parent)

            if not self.isPersistentEditorOpen(model_idx):
                self.openPersistentEditor(model_idx)
