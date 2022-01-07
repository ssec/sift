import logging
import struct
from typing import List, Optional
from uuid import UUID

from PyQt5.QtCore import (QAbstractItemModel, Qt, QModelIndex, pyqtSignal,
                          QMimeData)

from uwsift.common import LAYER_TREE_VIEW_HEADER, Presentation, Info, Kind, \
    LATLON_GRID_DATASET_NAME, BORDERS_DATASET_NAME, Platform, Instrument, \
    LayerModelColumns as LMC, LayerVisibility
from uwsift.model.layer_item import LayerItem
from uwsift.model.product_dataset import ProductDataset
from uwsift.workspace.workspace import frozendict

LOG = logging.getLogger(__name__)


class LayerModel(QAbstractItemModel):
    # ------------------- Creating layers and product datasets -----------------
    didCreateLayer = pyqtSignal(LayerItem)
    didAddImageDataset = pyqtSignal(LayerItem, ProductDataset)
    didAddLinesDataset = pyqtSignal(LayerItem, ProductDataset)
    didAddPointsDataset = pyqtSignal(LayerItem, ProductDataset)

    didAddSystemLayer = pyqtSignal(LayerItem)

    # ------------------ Changing properties of existing layers ----------------
    # didChangeColormap = pyqtSignal(dict)
    # didChangeColorLimits = pyqtSignal(dict)
    # didChangeGamma = pyqtSignal(dict)
    didChangeLayerVisible = pyqtSignal(UUID, bool)
    didChangeLayerOpacity = pyqtSignal(UUID, float)

    didUpdateLayers = pyqtSignal()
    didReorderLayers = pyqtSignal(list)
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

        self._supportedRoles = [Qt.DisplayRole, Qt.EditRole]

    def _init_system_layer(self, name):
        # The minimal 'dataset' information required by LayerItem
        # initialization:
        pseudo_info = frozendict({Info.KIND: Kind.LINES,
                                  Info.PLATFORM: Platform.SYSTEM,
                                  Info.INSTRUMENT: Instrument.GENERATED,
                                  "name": name})

        presentation = Presentation(uuid=None, kind=Kind.LINES)

        system_layer = LayerItem(self, pseudo_info, presentation)

        self.didCreateLayer.emit(system_layer)
        self._add_layer(system_layer)
        self.didAddSystemLayer.emit(system_layer)

    def init_system_layers(self):
        """ Create layers whose existence is controlled by the system, not
        by the user.

        Currently two system layers are set up, one for a latitude/longitude
        grid, the second for political borders.
        """
        for dataset_name in [LATLON_GRID_DATASET_NAME, BORDERS_DATASET_NAME]:
            self._init_system_layer(dataset_name)

    def get_dynamic_layers(self):
        return [layer for layer in self.layers if layer.dynamic]

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
                     Qt.ItemIsSelectable |
                     Qt.ItemIsDragEnabled)
        else:
            flags = Qt.ItemIsDropEnabled
        return flags

    def supportedDropActions(self):
        return Qt.MoveAction

    def headerData(self, section: int, orientation, role=None):

        if (orientation == Qt.Horizontal) and (role == Qt.DisplayRole):
            return self._headers[section]
        return None

    def _add_layer(self, layer: LayerItem) -> None:
        """
        Insert given layer into the LayerModel's `layers` list.

        :param layer: Layer to be inserted into LayerModel.

        The position of the new layer in the layer stack is determined by the
        kind of the layer: Layers of kinds LINES and POINTS are put in front,
        layers of other kinds are placed in front of the first existing layer,
        which is also neither of kind LINES nor POINTS.

        Emits the didReorderLayers() signal and (indirectly) other
        QAbstractItemModel signals notifying about the model change.
        """
        if layer in self.layers:
            raise ValueError(f"LayerItem {layer} is already in LayerModel.")

        # Determine the row to put the new layer into: layers of kinds POINTS
        # and LINES (for now) should be put on top of all existing layers (be
        # "favoured"), others should be on top of the existing layers of the
        # other kinds but below layers of a "favoured" kind.
        favoured_kinds = [
            Kind.LINES,
            Kind.POINTS,
        ]

        row = 0
        if layer.kind not in favoured_kinds:
            row = self.rowCount()  # fallback position: append
            for idx, existing_layer in enumerate(self.layers):
                if existing_layer.kind not in favoured_kinds:
                    # Found the first layer not of favoured kind: insert new
                    # layer right here putting it in front of the found one.
                    row = idx
                    break

        self.beginInsertRows(QModelIndex(), row, row)
        self.layers.insert(row, layer)
        self.endInsertRows()

        self._emit_didReorderLayers()

    def _emit_didReorderLayers(self):  # noqa
        uuids = [layer.uuid for layer in self.layers]
        self.didReorderLayers.emit(uuids)

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

    def setData(self, index: QModelIndex, data, role: int = Qt.EditRole):
        if not index.isValid():
            return False

        assert role == Qt.EditRole, \
            f"Unexpected role {role} for changing data."
        assert index.column() == LMC.VISIBILITY, \
            f"Attempt to edit immutable column {index.column()}."

        LOG.debug(f"Changing row {index.row()}, column {index.column()}"
                  f" to {data}.")

        layer = self.layers[index.row()]
        layer_visibility: LayerVisibility = data
        if layer.opacity != layer_visibility.opacity:
            layer.opacity = layer_visibility.opacity
            LOG.debug(f"Layer opacity changed to:"
                      f" {self.layers[index.row()].opacity}")
            self.didChangeLayerOpacity.emit(layer.uuid, layer.opacity)
        if layer.visible != layer_visibility.visible:
            layer.visible = layer_visibility.visible
            LOG.debug(f"Layer visible changed to:"
                      f" {self.layers[index.row()].visible}")
            self.didChangeLayerVisible.emit(layer.uuid, layer.visible)
        self.dataChanged.emit(index, index)
        return True

    def get_layer_by_uuid(self, uuid: UUID) -> Optional[LayerItem]:
        layers = [layer for layer in self.layers if layer.uuid == uuid]
        if len(layers) > 1:
            raise ValueError(f"Multiple Layers with UUID: {uuid} found"
                             f" with product_family_key:"
                             f" {self.product_family_key}!")
        elif len(layers) == 0:
            return None
        else:
            return layers[0]

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
            self.didCreateLayer.emit(layer)
            self._add_layer(layer)

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
            if product_dataset.kind == Kind.IMAGE:
                self.didAddImageDataset.emit(layer, product_dataset)
            elif product_dataset.kind == Kind.LINES:
                self.didAddLinesDataset.emit(layer, product_dataset)
            elif product_dataset.kind == Kind.POINTS:
                self.didAddPointsDataset.emit(layer, product_dataset)
            else:
                raise NotImplementedError(
                    f"Managing datasets of kind {product_dataset.kind}"
                    f" not (yet) supported.")

            self.didUpdateLayers.emit()

    def mimeTypes(self):
        return ['text/plain', 'text/xml']

    def mimeData(self, indexes):
        mime_data = QMimeData()
        rows = list(set([index.row() for index in indexes]))
        row_bytes = struct.pack("<I", rows[0])
        mime_data.setData('text/plain', row_bytes)
        return mime_data

    def dropMimeData(self, mime_data, action, row, column, parentIndex):
        if action == Qt.IgnoreAction:
            return True
        if action != Qt.MoveAction:
            return False

        source_row = struct.unpack("<I", mime_data.data('text/plain'))[0]

        if row != -1:  # we may also interpret this as put to the end
            target_row = row
        elif parentIndex.isValid():
            assert not parentIndex.isValid(), \
                "BUG: hierarchical layers not implemented," \
                " dropping on a parent must not yet occur!"
            # This case needs modification when hierarchical layers are
            # introduced.
            target_row = parentIndex.row()  # just to keep the linter calm
        else:
            target_row = self.rowCount(QModelIndex())

        move_is_possible = \
            self.beginMoveRows(QModelIndex(), source_row, source_row,
                               parentIndex, target_row)
        if not move_is_possible:
            return False

        # According to https://doc.qt.io/qt-5/qabstractitemmodel.html#beginMoveRows
        # now we can assert ...
        assert not source_row <= target_row <= source_row + 1

        if source_row < target_row:
            target_row -= 1
        self.layers.insert(target_row, self.layers.pop(source_row))
        self.endMoveRows()

        self._emit_didReorderLayers()
        self._refresh()

        return True

    def _refresh(self):
        self.layoutAboutToBeChanged.emit()
        self.revert()
        self.layoutChanged.emit()

    def on_didMatchTimes(self, t_matched_dict: dict):  # noqa
        for layer_uuid, active_dataset_uuids in t_matched_dict.items():
            layer = self.get_layer_by_uuid(layer_uuid)
            for product_dataset in layer.timeline.values():
                if product_dataset.uuid in active_dataset_uuids:
                    product_dataset.is_active = True
                else:
                    product_dataset.is_active = False
                self.didActivateProductDataset.emit(product_dataset.uuid,
                                                    product_dataset.is_active)


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
