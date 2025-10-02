from decimal import ROUND_HALF_UP, Decimal

from PyQt5.QtCore import QEvent, QLocale, QSize, Qt, pyqtSignal
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QDial,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QSlider,
    QStyle,
    QTableWidget,
    QWidget,
    QWizardPage,
)


class QNoScrollComboBox(QComboBox):
    """Special subclass of QComboBox to stop it from taking focus on scroll over"""

    def __init__(self, *args, **kwargs):
        super(QNoScrollComboBox, self).__init__(*args, **kwargs)
        self.setFocusPolicy(Qt.StrongFocus)

    def wheelEvent(self, ev):
        # If we want it to scroll when it has focus then uncomment
        # Currently not desired for Projection combo box, but may
        # be desired for RGB layer selector
        ev.ignore()


class QNoScrollSlider(QSlider):
    def __init__(self, *args, **kwargs):
        super(QNoScrollSlider, self).__init__(*args, **kwargs)
        self.setFocusPolicy(Qt.StrongFocus)

    def wheelEvent(self, ev):
        if not self.hasFocus():
            ev.ignore()
        else:
            super(QNoScrollSlider, self).wheelEvent(ev)


class QNoScrollDoubleSpinBox(QDoubleSpinBox):
    def __init__(self, *args, **kwargs):
        super(QNoScrollDoubleSpinBox, self).__init__(*args, **kwargs)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setLocale(QLocale(QLocale.English, QLocale.UnitedStates))  # always use the "." separator

    def wheelEvent(self, ev):
        if not self.hasFocus():
            ev.ignore()
        else:
            super(QNoScrollDoubleSpinBox, self).wheelEvent(ev)


class QNoScrollWebView(QWebEngineView):
    def __init__(self, *args, **kwargs):
        super(QNoScrollWebView, self).__init__(*args, **kwargs)
        self.setFocusPolicy(Qt.StrongFocus)

    def wheelEvent(self, ev):
        if not self.hasFocus():
            ev.ignore()
        else:
            super(QNoScrollWebView, self).wheelEvent(ev)


class AnyWizardPage(QWizardPage):
    """QWizardPage where at least some list items have to be checked.

    This requires the user to connect the checked signals for items to
    this classes `isComplete` method.

    """

    child_types = (QListWidget, QTableWidget)

    def __init__(self, *args, **kwargs):
        self.important_children = kwargs.pop("important_children", [])
        self.sift_page_checked = True
        super(AnyWizardPage, self).__init__(*args, **kwargs)

    def isComplete(self):
        if not self.sift_page_checked:
            return False

        children = self.important_children or self.findChildren(self.child_types)
        for child_widget in children:
            if isinstance(child_widget, QListWidget):
                count = child_widget.count()
                get_item = child_widget.item
            else:
                count = child_widget.rowCount()
                get_item = lambda row: child_widget.item(row, 0)
            for item_idx in range(count):
                item = get_item(item_idx)
                # if the item is not checkable or it is and it is checked
                # then this widget is complete
                if not bool(item.flags() & Qt.ItemIsUserCheckable) or item.checkState():
                    break
            else:
                return False
        return True

    def completeChangedSlot(self, *args, **kwargs):
        self.completeChanged.emit()


class InitiallyIncompleteWizardPage(QWizardPage):
    """QWizardPage that is only complete once 'page_complete' is set to True"""

    def __init__(self, *args, **kwargs):
        self.page_complete = False
        super(InitiallyIncompleteWizardPage, self).__init__(*args, **kwargs)

    # override default behaviour by allowing the code to disable 'Next'/'Finish' buttons
    def isComplete(self):
        return self.page_complete

    def completeChangedSlot(self, *args, **kwargs):
        self.completeChanged.emit()


class QAdaptiveDoubleSpinBox(QDoubleSpinBox):
    """QDoubleSpinBox that sets/increments/decrements values with number of decimals significant to the user"""

    upArrowClicked = pyqtSignal()
    downArrowClicked = pyqtSignal()

    def __init__(self, *args, **kwargs):
        super(QAdaptiveDoubleSpinBox, self).__init__(*args, **kwargs)
        self._decimal_places_displayed = self.decimals()
        self.setLocale(QLocale(QLocale.English, QLocale.UnitedStates))  # always use the "." separator
        # install event filter to detect modifier keys. It needs to be on application level as the focus can be anywhere
        # before clicking on the inc/dec butons with a modifier.
        QApplication.instance().installEventFilter(self)
        # flag to track if Shift is pressed
        self._shift_pressed = False
        self.setToolTip("Hold Ctrl and/or Shift for larger increment/decrement")

    def eventFilter(self, obj, event):
        """Filter events to detect modifier keys"""
        if event.type() == QEvent.KeyPress:
            if event.key() == Qt.Key_Shift:
                self._shift_pressed = True
        elif event.type() == QEvent.KeyRelease:
            if event.key() == Qt.Key_Shift:
                self._shift_pressed = False
        return super().eventFilter(obj, event)

    def setValue(self, value, keep_displ_decimals=False):
        """Override of setValue, rounds and adjusts the value to the amount of user significant decimals"""
        value, decimal_places_displayed = self._round_and_count_decimals(value, self.decimals())
        if not keep_displ_decimals:
            self._decimal_places_displayed = decimal_places_displayed
        super().setValue(round(value, self._decimal_places_displayed))

    def textFromValue(self, value):
        """Override to set the effective amount of decimals the user intends to have"""
        tfv = super().textFromValue(value)
        if "." in tfv:  # do we have decimals at all?
            parts = tfv.split(".")
            if self._decimal_places_displayed > 0:  # Do we actually have effective decimals?
                tfv = f"{parts[0]}.{parts[1][: self._decimal_places_displayed]}"
            else:
                tfv = parts[0]
        return tfv

    def valueFromText(self, text):
        """Override to evaluate the effective amount of decimals the user intends to have"""
        vft = super().valueFromText(text)
        self._eval_num_decimals_displayed(text)
        return vft

    def stepBy(self, steps):
        """Override stepBy to provide custom increment/decrement behavior and notification about the button cklick"""
        if self._shift_pressed:
            # if shift is pressed, 'boost' by 100.
            steps *= 100
        if self._decimal_places_displayed > 0:
            # if we have decimals we need to take care that the inc/dec is on the last one
            step_size = 10**-self._decimal_places_displayed
            new_value = round(self.value() + (step_size * steps), self._decimal_places_displayed)
        else:
            new_value = round(self.value() + steps)

        self.setValue(new_value, True)

        if steps > 0:
            self.upArrowClicked.emit()
        elif steps < 0:
            self.downArrowClicked.emit()

    def _eval_num_decimals_displayed(self, value_as_text):
        """Get the amount of decimals the user has entered"""
        # count decimal places to determine step size
        if "." in value_as_text:
            decimal_str = value_as_text.split(".")[-1]
            # catch case when only the decimal separator is there but no decimal -> this is considered to be one decimal
            self._decimal_places_displayed = max(1, len(decimal_str))
        else:
            self._decimal_places_displayed = 0

    def _round_and_count_decimals(self, value, max_decimals) -> tuple[float, int]:
        """Rounds a float value if the number of decimals is larger than max_decimals"""
        # convert to Decimal from str to preserve what user likely intended
        dval = Decimal(str(value))
        # create the rounding quantizer, e.g., Decimal('0.00001') for 5 decimals
        quant = Decimal(f'1.{"0" * max_decimals}')
        # round with ROUND_HALF_UP
        rounded = dval.quantize(quant, rounding=ROUND_HALF_UP)
        # now count actual decimals
        # normalize removes trailing zeroes; exponent tells how many decimals
        exp = rounded.normalize().as_tuple().exponent
        if isinstance(exp, int):
            decimals = -exp if exp < 0 else 0
        else:
            decimals = 0

        return float(rounded), decimals


class AnimationSpeedWidget(QWidget):
    """Custom animation speed control."""

    def __init__(self, *args, **kwargs):
        super(AnimationSpeedWidget, self).__init__(*args, **kwargs)

        self._dial = AnimationSpeedWidget._AnimSpeedDial(parent=self)

        self._framerate_label = AnimationSpeedWidget._FPSLabel(self)
        self._framerate_label.update_fixed_width()
        self._framerate_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self._dial.setRange(self._raw_minimum, self._raw_maximum)
        self._dial.setWrapping(False)
        self._dial.setNotchesVisible(True)
        self._dial.setSingleStep(1)
        self._dial.valueChanged.connect(self._changed)
        self._display_fps(self._dial.value())
        self._value_change_cb = None
        self._layout = QHBoxLayout()

        self._layout.addWidget(self._dial)
        self._layout.addWidget(self._framerate_label)

        self.setLayout(self._layout)

    def _convert(self, val: int, reverse: bool = False) -> float:
        """Map 1..100 nonlinearly to 10ms .. 5000ms.
        Args:
            val: raw value to remap to milliseconds
            reverse: when True, reverse conversion milliseconds to ticks
        Returns: converted value
        """
        if reverse:  # convert milliseconds to fp10s
            fp10s = 10000.0 / float(val)
            return fp10s
        else:
            ms = 10000.0 / float(val)
            return ms

    def _changed(self, value):
        self._display_fps(value)
        val = self._convert(value)
        if self._value_change_cb:
            self._value_change_cb(val)

    def _display_fps(self, value):
        fps = float(value) / 10.0
        self._framerate_label.setText("{0:.1f} fps".format(fps))

    def set_value_changed_callback(self, callback):
        self._value_change_cb = callback

    def setSize(self, size: QSize):
        self._dial.setFixedSize(size)

    class _AnimSpeedDial(QDial):
        def __init__(self, parent=None, step=1):
            super().__init__(parent)
            self.step = step  # How much the dial changes per wheel step

        def wheelEvent(self, event):
            # event.angleDelta().y() is in 1/8 degrees, each notch = 120
            num_steps = event.angleDelta().y() // 120
            self.setValue(self.value() + num_steps * self.step)
            event.accept()

    class _FPSLabel(QLabel):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # optional: set initial fixed width based on max expected content
            self.update_fixed_width()

        def event(self, event):
            if event.type() == QEvent.FontChange:
                # system/application font changed
                self.update_fixed_width()
            return super().event(event)

        def update_fixed_width(self):
            self.setFixedWidth(self.fontMetrics().boundingRect("15.0 fps").width())

    _dial: _AnimSpeedDial
    _raw_minimum = 2
    _raw_maximum = 100  # frames per 10 seconds
