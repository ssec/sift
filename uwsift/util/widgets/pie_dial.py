import logging
import sys

from PyQt5 import QtCore, QtGui, QtWidgets

from uwsift.common import LayerVisibility

LOG = logging.getLogger(__name__)


class PieDial:
    """Widget for displaying visibility and opacity of a layer.

    The available look and behaviour of the QDial widget is not suitable for
    the purpose of displaying the state of visibility and opacity of a layer.
    Therefore, a separate widget system was developed.

    The opacity is indicated by the filled area of a pie:
        - If, for example, the pie is completely filled, then the opacity
          is 100%.
        - If the cake is e.g. only half filled, the opacity is 50%.

    Visibility is indicated by crossing out or not crossing out the circle.
    If the cake is crossed out, the layer is invisible and vice versa.

    The span angle value and the visibility value are changed via
    the PieDialEditor.

    The source code used as the basis for the custom dial widget is:

    https://pastebin.com/yN3B3fc2
    https://stackoverflow.com/questions/12011147/how-to-create-a-3-color-gradient-dial-indicator-the-one-that-shows-green-yellow
    """

    def __init__(
        self,
        visible: bool = True,
        opacity: float = 1.0,
        pie_brush=QtCore.Qt.blue,
        pie_pen=QtCore.Qt.NoPen,
        strike_out_brush=QtCore.Qt.red,
        strike_out_pen=QtCore.Qt.NoPen,
    ):
        self._opacity = opacity
        self._visible = visible

        self._pie_brush = pie_brush
        self._pie_pen = pie_pen
        self._strike_out_brush = strike_out_brush
        self._strike_out_pen = strike_out_pen

    def _configure_appearance_of_drawn_pie(self, painter: QtGui.QPainter, rect: QtCore.QRect):
        """

        :param painter:
        :param rect:
        """
        if isinstance(self._pie_brush, QtGui.QConicalGradient):
            self._pie_brush.setCenter(QtCore.QPointF(rect.center()))

        painter.setPen(self._pie_pen)
        painter.setBrush(self._pie_brush)

    def _configure_appearance_of_drawn_strike_out(self, painter: QtGui.QPainter):
        """

        :param painter:
        """
        painter.setPen(self._strike_out_pen)
        painter.setBrush(self._strike_out_brush)

    def _convert_current_value_to_degree(self) -> float:
        """Maps the current value of the widget to suitable degree value,
        so that this converted value can be used e.g. for span angle of the
        drawn pie by PyQt.

        :return: converted value which is mapped to suitable degree value
        """
        return -360 * self.opacity

    def paint(self, painter: QtGui.QPainter, rect: QtCore.QRect):
        """

        :param painter:
        :param rect:
        """
        painter.save()

        painter.setRenderHint(painter.Antialiasing, True)

        pie_rect = QtCore.QRect(rect)

        pie_rect_size_min = min(pie_rect.size().width(), pie_rect.size().height())

        pie_rect.setSize(QtCore.QSize(pie_rect_size_min - 2, pie_rect_size_min - 2))
        pie_rect.moveCenter(rect.center())

        self._configure_appearance_of_drawn_pie(painter, pie_rect)
        painter.drawPie(pie_rect, int(90.0 * 16), int(self._convert_current_value_to_degree() * 16))

        if not self._visible:
            blocked_rect = QtCore.QRect(pie_rect)
            current_size = blocked_rect.size()
            current_width = current_size.width()
            current_size.setWidth(int(current_width / 4))
            current_size.setHeight(int(current_size.height() * 1.5))
            blocked_rect.setSize(current_size)

            blocked_rect.moveCenter(QtCore.QPointF(pie_rect.center()).toPoint())

            center = blocked_rect.center()

            transform = (
                QtGui.QTransform().translate(center.x(), center.y()).rotate(45.0).translate(-center.x(), -center.y())
            )

            rotated_rect = transform.mapToPolygon(blocked_rect)

            self._configure_appearance_of_drawn_strike_out(painter)
            painter.drawPolygon(rotated_rect)

        painter.restore()

    @property
    def visible(self):
        return self._visible

    @visible.setter
    def visible(self, visible: bool):
        self._visible = visible

    @property
    def opacity(self):
        return self._opacity

    @opacity.setter
    def opacity(self, opacity: float):
        self._opacity = float(min(max(opacity, 0.0), 1.0))


class PieDialEditor(QtWidgets.QWidget):
    editingFinished = QtCore.pyqtSignal()

    def __init__(self, parent: QtWidgets.QWidget, pie_dial: PieDial):
        super(PieDialEditor, self).__init__(parent=parent)

        self.setAutoFillBackground(True)

        self.pie_dial = pie_dial

        self._layer_opacity_popup = SliderPopup(parent=self)
        self._layer_opacity_popup.get_slider().valueChanged.connect(self._set_opacity_in_percent)
        self._current_mouse_pos = QtCore.QPoint(0, 0)

    def _set_opacity_in_percent(self, val: int) -> None:
        self.pie_dial.opacity = val / 100
        self.editingFinished.emit()

    def set_pie_dial(self, pie_dial: PieDial):
        """

        :param pie_dial:
        """
        self.pie_dial = pie_dial

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        """

        :param event:
        """
        painter = QtGui.QPainter(self)
        if self.pie_dial:
            self.pie_dial.paint(painter, self.rect())

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        """

        :param event:
        """
        if event.button() == QtCore.Qt.RightButton and not self._layer_opacity_popup.active:
            global_pos = self.mapToGlobal(event.pos())
            self._layer_opacity_popup.show_at(global_pos, self.pie_dial.opacity)

        if event.button() == QtCore.Qt.LeftButton and not self._layer_opacity_popup.active:
            self.pie_dial.visible = not self.pie_dial.visible
            self.editingFinished.emit()

        # The event must not fall through to the parent ItemView, which might
        # react to it otherwise, which would feel strange, thus:
        event.accept()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        # The event must not fall through to the parent ItemView, which might
        # react to it otherwise, which would feel strange, thus:
        event.accept()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        # The event must not fall through to the parent ItemView, which might
        # react to it otherwise, which would feel strange.
        # TODO here the feature to directly modify visibility/opacity by mouse
        #  moving could be implemented.
        event.accept()

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        """

        :param event:
        """

        if event.angleDelta().y() > 0 and not self._layer_opacity_popup.active:
            self.pie_dial.opacity = self.pie_dial.opacity + 0.05
            self.editingFinished.emit()

        elif event.angleDelta().y() < 0 and not self._layer_opacity_popup.active:
            self.pie_dial.opacity = self.pie_dial.opacity - 0.05
            self.editingFinished.emit()

        # The event must not fall through to the parent ItemView, which would
        # scroll its view area otherwise, thus:
        event.accept()


class PieDialDelegate(QtWidgets.QStyledItemDelegate):
    """ """

    def __init__(self, *args, **kwargs):
        super(PieDialDelegate, self).__init__(*args, **kwargs)

    def createEditor(
        self, parent: QtWidgets.QWidget, option: QtWidgets.QStyleOptionViewItem, index: QtCore.QModelIndex
    ) -> QtWidgets.QWidget:
        """

        :param parent:
        :param option:
        :param index:
        :return:
        """
        editor = PieDialEditor(parent, PieDial())
        editor.editingFinished.connect(self._commit_and_close_editor)

        return editor

    def paint(self, painter: QtGui.QPainter, option: QtWidgets.QStyleOptionViewItem, index: QtCore.QModelIndex) -> None:
        layer_visibility: LayerVisibility = index.data()

        pie_dial = PieDial(visible=layer_visibility.visible, opacity=layer_visibility.opacity)
        pie_dial.paint(painter, option.rect)

    def _commit_and_close_editor(self):
        """ """
        editor = self.sender()
        self.commitData.emit(editor)

    def setEditorData(self, editor: QtWidgets.QWidget, index: QtCore.QModelIndex) -> None:
        """

        :param editor:
        :param index:
        """
        layer_visibility: LayerVisibility = index.data()

        if not editor.pie_dial:
            pie_dial = PieDial(visible=layer_visibility.visible, opacity=layer_visibility.opacity)
            editor.set_pie_dial(pie_dial)
        else:
            editor.pie_dial.visible = layer_visibility.visible
            editor.pie_dial.opacity = layer_visibility.opacity
            editor.update()

    def setModelData(
        self, editor: QtWidgets.QWidget, model: QtCore.QAbstractItemModel, index: QtCore.QModelIndex
    ) -> None:
        """

        :param editor:
        :param model:
        :param index:
        """
        layer_visibility = LayerVisibility(editor.pie_dial.visible, editor.pie_dial.opacity)

        if index.data() != layer_visibility:
            model.setData(index, layer_visibility)
        editor.update()

    def updateEditorGeometry(
        self, editor: QtWidgets.QWidget, option: QtWidgets.QStyleOptionViewItem, index: QtCore.QModelIndex
    ) -> None:
        """

        :param editor:
        :param option:
        :param index:
        """
        editor.setGeometry(option.rect)


class SliderPopup(QtWidgets.QWidget):
    """Custom slider widget that can change a value from 1 to 100 per cent and
    is displayed in a pop-up window. When the widget is displayed, the value to
    be changed must be converted if the value is not a percentage value.
    """

    def __init__(self, *args, **kwargs):
        super(SliderPopup, self).__init__(*args, **kwargs)

        self.setWindowFlags(QtCore.Qt.FramelessWindowHint | QtCore.Qt.Popup)
        self.setFocusPolicy(QtCore.Qt.ClickFocus)

        self._slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, parent=self)
        self._slider.setRange(0, 100)
        self._label = QtWidgets.QLabel("0 %")

        self._layout = QtWidgets.QHBoxLayout()
        self._layout.addWidget(self._slider)
        self._layout.addWidget(self._label)

        self.setLayout(self._layout)

        self._slider.valueChanged.connect(self._changed)

    def show_at(self, pos: QtCore.QPoint, val: float):
        """Show the window at the handed over position

        :param pos: the wished position where the window should be shown
        :param val: the value which the slider should show at the beginning
        """
        size = QtCore.QSize(180, 20)
        pos = QtCore.QPoint(pos.x(), pos.y())
        rect = QtCore.QRect(pos, size)
        self.setGeometry(rect)
        self.show()
        self._slider.setValue(self._convert(val))
        self._label.setText("{} %".format(self._slider.value()))

    @staticmethod
    def _convert(val: float) -> int:
        """Map the float value to percentage value.

        :param val: raw float value
        :return: converted value
        """
        return int(val * 100)

    def _changed(self, value: int):
        """Adapt the caption text of the window to the changed value.

        :param value: the current value which was changed
        """
        if not self.active:
            return
        self._label.setText("{} %".format(value))

    def get_slider(self) -> QtWidgets.QSlider:
        """Get the slider of the window"""
        return self._slider

    @property
    def active(self):
        """Get the current state if the window with the slider is active
        or not"""
        return self._slider.isActiveWindow()


def main():
    app = QtWidgets.QApplication(sys.argv)

    example = QtWidgets.QWidget()
    example.setGeometry(0, 0, 40, 40)

    layout = QtWidgets.QVBoxLayout(example)
    pie_dial_editor = PieDialEditor(example)
    pie_dial = PieDial(visible=True, opacity=1.0)
    pie_dial_editor.set_pie_dial(pie_dial)

    layout.addWidget(pie_dial_editor)

    example.show()
    example.raise_()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
