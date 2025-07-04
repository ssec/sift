import logging

import numpy as np
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtCore import QModelIndex, QRect, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFontMetrics, QPainter
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QDesktopWidget,
    QHeaderView,
    QLabel,
    QMenu,
    QStyledItemDelegate,
    QTreeView,
)

from uwsift.common import FALLBACK_RANGE, Info
from uwsift.common import LayerModelColumns as LMC  # noqa
from uwsift.common import LayerVisibility, Platform
from uwsift.model.layer_item import LayerItem
from uwsift.util.widgets.pie_dial import PieDialDelegate
from uwsift.view.probes import DEFAULT_POINT_PROBE

LOG = logging.getLogger(__name__)


class EqualizerBarDelegate(QStyledItemDelegate):

    # If there is no valid_range available, we need to store and maintain the colour limits, since they are initially
    # valid (total) ranges but can get changed by the user. But since we also need to take care for reordering, we
    # also store the valid_range values to keep the reordering algorithm simple.
    _total_ranges: dict[int, tuple[float, float]] = {}
    EQUALIZER_BLUE = QColor(100, 200, 250)
    EQUALIZER_COL_EXEED_BLUE = QColor(180, 220, 255)

    def paint(self, painter, option, index):
        """Override from base class to realise the custom rendering of the bar."""
        dummy_index = QModelIndex()
        super().paint(painter, option, dummy_index)

        disp_text = index.data(Qt.DisplayRole)
        try:
            value = float(disp_text)
        except ValueError:
            value = None

        painter.save()

        idx = index.row()
        if value is not None:
            layer = index.model().layers[idx]
            unit_conv_funct = layer.info[Info.UNIT_CONVERSION][1]
            probe_val_range = FALLBACK_RANGE
            if layer.valid_range:
                # Simple case: the layer has a valid_range item that is fixed.
                probe_val_range = self._process_range(idx, layer.valid_range)
            else:
                # No valid_range member? Then we need to get that info via the colour limits.
                probe_val_range = self._total_ranges.get(idx, probe_val_range)
                _climits = layer.presentation.climits
                if _climits and any(isinstance(v, (int, float)) for v in _climits):
                    if np.isfinite(_climits).all():
                        # After the "health" checks of the colour limut values we store them after checking for min/max.
                        # Why check? The upper and lower limits come from the colour map and can get changed by the
                        # user. Since at this point we cannot be 100% sure when we do get the absolute limits,
                        # we always check for updates of the limits.
                        probe_val_range = self._process_range(idx, _climits)

            probe_val_range = (unit_conv_funct(probe_val_range[0]), unit_conv_funct(probe_val_range[1]))
            normed_probe_value = self._normalize_val(value, probe_val_range)
            if normed_probe_value != -1.0:
                bar_color = self.EQUALIZER_BLUE
            else:
                bar_color = self.EQUALIZER_COL_EXEED_BLUE
                normed_probe_value = 1.0

            if not np.isnan(normed_probe_value):
                bar_rect = QRect(
                    option.rect.left(),
                    option.rect.top() + 1,
                    int(option.rect.width() * normed_probe_value),
                    option.rect.height() - 2,
                )
                painter.setRenderHint(QPainter.Antialiasing)
                painter.fillRect(bar_rect, bar_color)

        else:  # Store dummies if there is no value to have the tree view structure in the total range list reflected
            if idx not in self._total_ranges:
                self._total_ranges[idx] = (np.inf, -np.inf)

        painter.setPen(Qt.black)
        painter.drawText(option.rect, Qt.AlignVCenter, f"{disp_text}")

        painter.restore()

    def _layers_reordered(self, sourceParent, sourceStart, sourceEnd, destinationParent, destinationRow):
        # Must not differ due to setSelectionMode(QTableWidget.SingleSelection)
        assert sourceStart == sourceEnd  # nosec B101
        highest_index = max(self._total_ranges.keys())
        if (sourceStart <= highest_index) and (destinationRow <= highest_index + 1):
            self._move_and_reindex(sourceStart, destinationRow)

    def _layers_removed(self, parent, first, last):
        self._reset()  # No management, just reset and let the list get built up again.

    def _layers_inserted(self, parent, first, last):
        self._reset()  # No management, just reset and let the list get built up again.

    def _move_and_reindex(self, from_idx: int, target_idx: int):
        # QT would never provide the same indices when reordering
        assert from_idx != target_idx  # nosec B101
        # Get ordered list of values
        items = [self._total_ranges[i] for i in sorted(self._total_ranges.keys())]
        # Move the item
        moved_item = items.pop(from_idx)
        # Insert at target position
        if from_idx > target_idx:
            insert_pos = target_idx
        else:
            insert_pos = target_idx - 1
        items.insert(insert_pos, moved_item)
        # Return reindexed dictionary
        self._total_ranges = {i: item for i, item in enumerate(items)}

    def _reset(self):
        self._total_ranges = {}

    def _process_range(self, idx: int, climit: tuple) -> tuple:
        min_val, max_val = climit
        # Swap if needed
        if min_val > max_val:
            min_val, max_val = max_val, min_val

        if idx not in self._total_ranges:
            self._total_ranges[idx] = (min_val, max_val)
        else:
            stored_range = self._total_ranges[idx]
            # If the upper max or lower min exceeds, we need to set this as the new
            # total range.
            if (min_val < stored_range[0]) or (stored_range[1] < max_val):
                min_val = float(min(min_val, stored_range[0]))
                max_val = float(max(max_val, stored_range[1]))
                self._total_ranges[idx] = (min_val, max_val)

        return self._total_ranges[idx]

    def _normalize_val(self, val: float, min_max: tuple) -> float:
        min_val, max_val = min_max
        if np.isclose(min_val, max_val, rtol=1e-9, atol=1e-12):
            return -1.0
        # Clamp v to the range
        v_clamped = max(min_val, min(val, max_val))
        # Normalize
        return (v_clamped - min_val) / (max_val - min_val)


class PersistentTooltip(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent, Qt.ToolTip)
        self.setWindowFlags(Qt.ToolTip)
        self.setStyleSheet("background-color: #ffffdd; border: 1px solid black; padding: 3px;")
        self.hide()

    def show_tooltip(self, text, global_mouse_pos):
        self.setText(text)
        self.adjustSize()

        screen = QDesktopWidget().screenGeometry(self)
        tooltip_size = self.size()

        offset_x = 20
        offset_y = 20

        # Default position: right & below the mouse
        pos_x = global_mouse_pos.x() + offset_x
        pos_y = global_mouse_pos.y() + offset_y

        # Check right boundary
        if pos_x + tooltip_size.width() > screen.right():
            pos_x = global_mouse_pos.x() - tooltip_size.width() - offset_x

        # Check top/bottom boundaries (just for safety)
        if pos_y + tooltip_size.height() > screen.bottom():
            pos_y = screen.bottom() - tooltip_size.height() - 5

        if pos_y < screen.top():
            pos_y = screen.top() + 5

        self.move(pos_x, pos_y)
        self.show()

    def hide_tooltip(self):
        self.hide()


class CustomHeaderView(QHeaderView):
    """Custom header viedw class to manage the column widths"""

    SEP_CLICK_TOL = 5

    def __init__(self, orientation, parent=None):
        super().__init__(orientation, parent)
        self._column_limits: dict[int, tuple] = {}

    def setColumnLimits(self, column, min_width, max_width):
        self._column_limits[column] = (min_width, max_width)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            col_index = self._get_separator_at_position(event.pos().x())
            if col_index >= 0:
                self._process_col_resize(col_index)
                return
        super().mouseDoubleClickEvent(event)

    def _adjust_all_columns(self):
        for col_index in range(self.count()):
            self._process_col_resize(col_index)

    def _eval_width(self, col_index, demanded_width):
        if col_index in self._column_limits:
            min_width = self._column_limits[col_index][0]
            max_width = self._column_limits[col_index][1]
            if demanded_width < min_width:
                return min_width
            elif max_width < demanded_width:
                return max_width
            else:
                return demanded_width
        return demanded_width

    def _process_col_resize(self, col_index):
        # Check if this column should resize to header text
        header_width = self._getHeaderTextWidth(col_index)
        # Calculate the ideal size for contents
        tree_view = self.parent()
        contents_width = tree_view.sizeHintForColumn(col_index)
        signif_width = max(header_width, contents_width)
        ideal_width = self._eval_width(col_index, signif_width)
        # Resize the section
        self.resizeSection(col_index, ideal_width)

    def _getHeaderTextWidth(self, col_index):
        if not self.model():
            return 0
        # Get the header text
        header_text = self.model().headerData(col_index, self.orientation(), Qt.DisplayRole)
        if not header_text:
            return 0
        # Get font metrics for the header font
        font_metrics = QFontMetrics(self.font())
        # Calculate text width with some padding
        text_width = font_metrics.width(str(header_text))
        # Add padding for margins, sort indicators, etc.
        padding = 10  # Adjust this value as needed
        return text_width + padding

    def _get_separator_at_position(self, x_pos):
        for i in range(self.count()):
            section_end = self.sectionPosition(i) + self.sectionSize(i)
            # Check if click is within tolerance of separator
            if abs(x_pos - section_end) <= self.SEP_CLICK_TOL:
                return i
        return -1


class LayerTreeView(QTreeView):
    layerSelectionChanged = pyqtSignal(tuple)
    layerSelectionChangedIndex = pyqtSignal(int)
    selectedLayerForProbeChanged = pyqtSignal(str)

    COL_WIDTH_MAX = {
        LMC.VISIBILITY: 20,
        LMC.SOURCE: 125,
        LMC.NAME: 125,
        LMC.WAVELENGTH: 60,
        LMC.PROBE_VALUE: 10000,
        LMC.PROBE_UNIT: 45,
    }

    def __init__(self, *args, **kwargs):
        super(LayerTreeView, self).__init__(*args, **kwargs)

        self._equalizer_bar_delegate = EqualizerBarDelegate()

        header = CustomHeaderView(Qt.Horizontal, self)
        self.setHeader(header)

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

        self.setMouseTracking(True)
        self.tooltip = PersistentTooltip(self)
        self._last_index = None
        self.setItemDelegateForColumn(LMC.PROBE_VALUE, self._equalizer_bar_delegate)

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

    def setModel(self, model):
        """Override from base class."""
        super().setModel(model)

        header = self.header()

        resizable_col = LMC.PROBE_VALUE

        for col, max_width in self.COL_WIDTH_MAX.items():
            col_size = header._getHeaderTextWidth(col)
            header.setColumnLimits(col, col_size, max_width)
            if col != resizable_col:
                header.setSectionResizeMode(col, header.Interactive)
            else:
                header.setSectionResizeMode(col, header.Stretch)

        self.setHeaderHidden(False)
        header.setStretchLastSection(False)
        header.setSectionsClickable(True)

        header.setMinimumSectionSize(20)
        header.setMaximumSectionSize(200)

        model.rowsMoved.connect(self._equalizer_bar_delegate._layers_reordered)
        model.rowsRemoved.connect(self._equalizer_bar_delegate._layers_removed)
        model.rowsInserted.connect(self._equalizer_bar_delegate._layers_inserted)

    def mouseMoveEvent(self, event):
        """Override from base class."""
        index = self.indexAt(event.pos())

        if index != self._last_index:
            self._last_index = index
            if not index.isValid():
                self.tooltip.hide_tooltip()
                return

            text = str(index.data(Qt.DisplayRole))
            option = self.viewOptions()
            option.rect = self.visualRect(index)

            font_metrics = QFontMetrics(option.font)
            text_width = font_metrics.width(f"{text}  ")  # Add two spaces to be on the safe side
            column_width = self.columnWidth(index.column())

            if text_width > column_width:
                global_pos = self.viewport().mapToGlobal(event.pos())
                self.tooltip.show_tooltip(text, global_pos)
            else:
                self.tooltip.hide_tooltip()

        super().mouseMoveEvent(event)

    def leaveEvent(self, event):
        """Override from base class."""
        self.tooltip.hide_tooltip()
        self._last_index = None
        super().leaveEvent(event)

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

        self._adjust_all_columns()

    def _adjust_all_columns(self):
        self.header()._adjust_all_columns()

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
            dynamic_layer_id = self.model().get_dynamic_layer_id(selected_layer[0])
            if dynamic_layer_id != -1:
                self.layerSelectionChangedIndex.emit(dynamic_layer_id)

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
