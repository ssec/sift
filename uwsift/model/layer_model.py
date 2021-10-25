import logging
from typing import List

from PyQt5.QtCore import (QAbstractItemModel, Qt, QModelIndex, pyqtSignal)

from uwsift.common import LAYER_TREE_VIEW_HEADER, Presentation, Info
from uwsift.model.layer_item import LayerItem
from uwsift.model.product_dataset import ProductDataset
from uwsift.workspace.workspace import frozendict

LOG = logging.getLogger(__name__)


class LayerModel(QAbstractItemModel):
    # ------------------- Creating layers and product datasets -----------------
    didCreateLayer = pyqtSignal(LayerItem)
    didAddProductDataset = pyqtSignal(LayerItem, ProductDataset)

    # ------------------ Changing properties of existing layers ----------------
    # didChangeColormap = pyqtSignal(dict)
    # didChangeColorLimits = pyqtSignal(dict)
    # didChangeGamma = pyqtSignal(dict)

    # didUpdateZOrder = pyqtSignal(list)
    # didChangeLayerName = pyqtSignal(UUID, str)  # layer uuid, new name

    # --------------- Adding layers derived from existing layers ---------------
    # didAddCompositeLayer = pyqtSignal(tuple, UUID, Presentation)

    # ----------------------- Removing existing layers -------------------------
    # didDeleteLayer = pyqtSignal(UUID)
    # didDeleteProductDataset = pyqtSignal(UUID)

    # --------------------------------------------------------------------------
    # didChangeImageKind = pyqtSignal(dict)

    def __init__(self, parent=None, policy=None):
        """
        Model for a "flat" layer tree (list/table of layers)
        (Note: For hierarchies the `parent` and `index` methods, among others,
        need rework).

        :param parent: Parent QObject of the LayerModel (Qt QObject hierarchy
                       argument)
        :param policy: Policy to match product datasets to layers and return a
                       matching layer.
        """

        super().__init__(parent)

        self._headers = LAYER_TREE_VIEW_HEADER

        if policy is None:
            self.policy = ProductFamilyKeyMappingPolicy(model=self)
        else:
            self.policy = policy(model=self)

        self.layers: List[LayerItem] = []

        self._supportedRoles = [Qt.DisplayRole]

    def data(self, index: QModelIndex, role: int = None):
        if not index.isValid():
            return None
        if role not in self._supportedRoles:
            return None

        layer: LayerItem = self.layers[index.row()]

        return layer.data(index.column())

    def flags(self, index):
        if index.isValid():
            flags = (Qt.ItemIsEnabled |
                     Qt.ItemIsSelectable)
        else:
            flags = Qt.NoItemFlags
        return flags

    def headerData(self, section: int, orientation, role=None):

        if (orientation == Qt.Horizontal) and (role == Qt.DisplayRole):
            return self._headers[section]
        return None

    def add_layer(self, layer: LayerItem) -> None:
        """
        Calling this method with a specified layer will insert that layer
        into the LayerModel's `layers` collection and emit the correct
        signals for QT to register the change that occurred to the model.

        :param layer: Layer to be inserted into LayerModel.
        """
        row = self.rowCount()
        count = 1
        self.beginInsertRows(QModelIndex(), row, row + count - 1)
        self.layers.append(layer)
        self.endInsertRows()

    def hasChildren(self, parent=QModelIndex()) -> bool:
        """
        For now the Layer model does not support layer hierarchies
        (group layers) thus only the root index can have children.

        :param parent: model index to query
        :return: true if parent is the root index and has at least one row and
                 column
        """
        # This needs modification if hierarchical layers are introduced.
        return not parent.isValid() and \
            (self.rowCount(parent) > 0) and (self.columnCount(parent) > 0)

    def columnCount(self, parent=None):
        return len(self._headers)

    def rowCount(self, parent=None):
        # This needs modification if hierarchical layers are introduced.
        return len(self.layers)

    def index(self, row: int, col: int, parent=QModelIndex()):
        if not self.hasIndex(row, col, parent):
            return QModelIndex()
        assert not parent.isValid()
        layer = self.layers[row]
        return self.createIndex(row, col, layer)

    def parent(self, index):
        # This needs modification if hierarchical layers are introduced.
        return QModelIndex()

    def order(self, layer: LayerItem) -> int:
        """
        Method to return the order of a specific layer within the model.
        Determined by its index in the model.

        :param layer: Layer whose oorder is queried.
        :return: Integer representing the order of queried layer.
        """
        return self.layers.index(layer)

    def _get_layer_for_dataset(self, info: frozendict,
                               presentation: Presentation) -> LayerItem:
        layer, grouping_key = self.policy.get_existing_layer_for_dataset(info)

        if not layer:
            # According to the policy, there is no layer for the dataset
            # represented by the given info yet, so a new one has to be
            # created. We use the grouping_key returned by the policy to
            # ensure that the new layer is compatible with the dataset in
            # terms of the policy.
            layer = LayerItem(self, info, presentation, grouping_key)
            self.add_layer(layer)
            self.didCreateLayer.emit(layer)

        return layer

    def add_dataset(self, info: frozendict, presentation: Presentation) \
            -> None:
        """
        Slot specifically to fill model from Document's
        `activate_product_uuid_as_new_layer`.
        For every loaded dataset Document emits `didAddDataset` signal which
        must be connected to this method.

        :param info: Dictionary of dataset metadata information.
        :param presentation: Presentation to be set for layer, when a new one
               has to be created to hold the dataset, ignored otherwise.
        """
        layer: LayerItem = self._get_layer_for_dataset(info, presentation)

        # Note: For now we do not set a presentation *per dataset*. In the
        #  future we may think about doing this, but there is not yet an
        #  according control flow that could be chosen by the user.
        product_dataset = layer.add_dataset(info)
        if product_dataset is not None:
            self.didAddProductDataset.emit(layer, product_dataset)


class ProductFamilyKeyMappingPolicy:
    def __init__(self, model: LayerModel):
        self.model = model

    def get_existing_layer_for_dataset(self, info: frozendict):
        """
        Returns layer within an instance of LayerModel according to a match
        between the `grouping_key` calculated from the given dataset metadata
        information and the `grouping_key` s within LayerModel's `layers`
        collection.

        :param info: Dataset metadata information

        :return: tuple with LayerItem with its `grouping_key` matching that
            of the passed dataset metadata information, if there is already one
            in the LayerModel, None otherwise. Second element of the tuple is
            the grouping key generated by the policy.
            You must use that key when creating a new layer for the dataset for
            the given info to make the policy work.
        """
        product_family_key = self.get_grouping_key(info)
        LOG.debug(f"Product Family Key:\n {product_family_key}")

        existing_product_family_keys = \
            [layer.grouping_key for layer in self.model.layers]

        if product_family_key in existing_product_family_keys:
            layer_idx = existing_product_family_keys.index(product_family_key)
            return self.model.layers[layer_idx], product_family_key

        return None, product_family_key

    @staticmethod
    def get_grouping_key(info):
        # This is, where layer grouping policies will differ:
        # This implementation returns the (legacy SIFT) product_family_key
        return info.get(Info.PLATFORM), \
               info.get(Info.INSTRUMENT), \
               info.get(Info.DATASET_NAME)
