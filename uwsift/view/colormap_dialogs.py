import logging
from PyQt5 import QtCore, QtGui, QtWidgets
from functools import partial
from uuid import UUID

from uwsift.common import Info, Kind
from uwsift.ui.change_colormap_dialog_ui import Ui_changeColormapDialog

LOG = logging.getLogger(__name__)


class ChangeColormapDialog(QtWidgets.QDialog):
    userDidChangeColorLimits = QtCore.pyqtSignal(UUID, tuple)  # layer being changed, char from 'rgba', new-min, new-max

    def __init__(self, doc, uuid, parent=None):
        super(ChangeColormapDialog, self).__init__(parent)
        self.ui = Ui_changeColormapDialog()
        self.ui.setupUi(self)
        self.doc = doc
        self.uuid = uuid
        self._slider_steps = 100
        self.valid_min, self.valid_max = self.doc.valid_range_for_uuid(self.uuid)
        prez = self.doc.prez_for_uuid(self.uuid)
        layer = self.doc[self.uuid]
        conv = layer[Info.UNIT_CONVERSION]
        self.setWindowTitle(str(self.windowTitle()) + ": " + self.doc[self.uuid][Info.SHORT_NAME])
        self._initial_cmap = prez.colormap
        self._current_cmap = self._initial_cmap
        self._initial_clims = prez.climits
        self._current_clims = self._initial_clims
        self._initial_gamma = prez.gamma
        self._current_gamma = self._initial_gamma

        self._validator = QtGui.QDoubleValidator()
        self._init_cmap_combo()
        self._init_vmin_slider()
        self.ui.vmin_edit.setValidator(self._validator)
        self.ui.vmin_edit.setText('{:0.03f}'.format(conv[1](self._initial_clims[0])))
        self._init_vmax_slider()
        self.ui.vmax_edit.setValidator(self._validator)
        self.ui.vmax_edit.setText('{:0.03f}'.format(conv[1](self._initial_clims[1])))
        self.ui.gammaSpinBox.setValue(self._initial_gamma)

        self.ui.buttons.clicked.connect(self._clicked)
        close_button = self.ui.buttons.button(QtWidgets.QDialogButtonBox.Close)
        close_button.setAutoDefault(True)
        reset_button = self.ui.buttons.button(QtWidgets.QDialogButtonBox.Reset)
        reset_button.setAutoDefault(False)
        self.ui.buttons.accepted.disconnect()
        self.ui.buttons.rejected.disconnect()

        self.ui.cmap_combobox.currentIndexChanged.connect(self._cmap_changed)
        self.ui.vmin_slider.sliderReleased.connect(partial(self._slider_changed, is_max=False))
        self.ui.vmax_slider.sliderReleased.connect(partial(self._slider_changed, is_max=True))
        self.ui.vmin_edit.editingFinished.connect(partial(self._edit_changed, is_max=False))
        self.ui.vmax_edit.editingFinished.connect(partial(self._edit_changed, is_max=True))

        if layer[Info.KIND] in [Kind.CONTOUR]:
            self.ui.gammaSpinBox.setDisabled(True)
        else:
            self.ui.gammaSpinBox.valueChanged.connect(self._gamma_changed)

    def _clicked(self, button):
        r = self.ui.buttons.buttonRole(button)
        if r == self.ui.buttons.ResetRole:
            self.reset()
            return self.reject()
        return self.accept()

    def reset(self):
        # rejecting (Cancel button) means reset previous settings
        self.doc.change_colormap_for_layers(self._initial_cmap, (self.uuid,))
        self.doc.change_clims_for_siblings(self.uuid, self._initial_clims)
        self.doc.change_gamma_for_siblings(self.uuid, self._initial_gamma)

    def _cmap_changed(self, index):
        cmap_str = self.ui.cmap_combobox.itemData(index)
        self._current_cmap = str(cmap_str)
        self.doc.change_colormap_for_layers(self._current_cmap, (self.uuid,))

    def _set_new_clims(self, val, is_max):
        if is_max:
            self._current_clims = (self._current_clims[0], val)
        else:
            self._current_clims = (val, self._current_clims[1])
        self.doc.change_clims_for_siblings(self.uuid, self._current_clims)

    def _slider_changed(self, is_max=True):
        slider = self.ui.vmax_slider if is_max else self.ui.vmin_slider
        edit = self.ui.vmax_edit if is_max else self.ui.vmin_edit

        val = self._get_slider_value(slider.value())
        LOG.debug('slider %s %s => %f' % (self.uuid, 'max' if is_max else 'min', val))
        display_val = self.doc[self.uuid][Info.UNIT_CONVERSION][1](val)
        edit.setText('{:0.03f}'.format(display_val))
        return self._set_new_clims(val, is_max)

    def _edit_changed(self, is_max=True):
        slider = self.ui.vmax_slider if is_max else self.ui.vmin_slider
        edit = self.ui.vmax_edit if is_max else self.ui.vmin_edit

        vdis = float(edit.text())
        val = self.doc[self.uuid][Info.UNIT_CONVERSION][1](vdis, inverse=True)
        LOG.debug('line edit %s %s => %f => %f' % (self.uuid, 'max' if is_max else 'min', vdis, val))
        sv = self._create_slider_value(val)
        slider.setValue(sv)
        return self._set_new_clims(val, is_max)

    def _init_cmap_combo(self):
        # FIXME: We should do this by colormap category
        for idx, colormap in enumerate(self.doc.colormaps.keys()):
            self.ui.cmap_combobox.addItem(colormap, colormap)
            if colormap == self._initial_cmap:
                self.ui.cmap_combobox.setCurrentIndex(idx)

    def _get_slider_value(self, slider_val):
        return (slider_val / self._slider_steps) * (self.valid_max - self.valid_min) + self.valid_min

    def _create_slider_value(self, channel_val):
        return ((channel_val - self.valid_min) / (self.valid_max - self.valid_min)) * self._slider_steps

    def _init_vmin_slider(self):
        current_vmin = self._initial_clims[0]
        self.ui.vmin_slider.setRange(0, self._slider_steps)
        slider_val = self._create_slider_value(current_vmin)
        self.ui.vmin_slider.setSliderPosition(max(slider_val, 0))

    def _init_vmax_slider(self):
        current_vmax = self._initial_clims[1]
        self.ui.vmax_slider.setMaximum(32767)
        self.ui.vmax_slider.setRange(0, self._slider_steps)
        slider_val = self._create_slider_value(current_vmax)
        self.ui.vmax_slider.setSliderPosition(min(slider_val, 32767))

    def _gamma_changed(self, val):
        self._current_gamma = val
        self.doc.change_gamma_for_siblings(self.uuid, self._current_gamma)
