from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QComboBox, QSlider, QDoubleSpinBox, QWizardPage, QTableWidget, QListWidget
from PyQt5.QtWebEngineWidgets import QWebEngineView


class QNoScrollComboBox(QComboBox):
    """Special subclass of QComboBox to stop it from taking focus on scroll over
    """

    def __init__(self, *args, **kwargs):
        super(QNoScrollComboBox, self).__init__(*args, **kwargs)
        self.setFocusPolicy(Qt.StrongFocus)

    def wheelEvent(self, ev):
        # If we want it to scroll when it has focus then uncomment
        # Currently not desired for Projection combo box, but may
        # be desired for RGB layer selector
        # if not self.hasFocus():
        #    ev.ignore()
        # else:
        #    super(QNoScrollComboBox, self).wheelEvent(ev)
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
