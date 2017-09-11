from PyQt4.QtGui import QComboBox, QSlider, QDoubleSpinBox
from PyQt4.QtCore import Qt
from PyQt4.QtWebKit import QWebView


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
        #if not self.hasFocus():
        #    ev.ignore()
        #else:
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


class QNoScrollWebView(QWebView):
    def __init__(self, *args, **kwargs):
        super(QNoScrollWebView, self).__init__(*args, **kwargs)
        self.setFocusPolicy(Qt.StrongFocus)

    def wheelEvent(self, ev):
        if not self.hasFocus():
            ev.ignore()
        else:
            super(QNoScrollWebView, self).wheelEvent(ev)
