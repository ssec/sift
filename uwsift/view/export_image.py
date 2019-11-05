import logging
import os
import io
from PyQt5 import QtCore, QtGui, QtWidgets

import imageio
import numpy
import matplotlib as mpl
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont

from uwsift.common import Info
from uwsift.ui import export_image_dialog_ui
from uwsift.view.colormap import COLORMAP_MANAGER
from uwsift.util import get_package_data_dir, USER_DESKTOP_DIRECTORY

LOG = logging.getLogger(__name__)
DATA_DIR = get_package_data_dir()

NUM_TICKS = 8
TICK_SIZE = 14
FONT = 'arial'


def is_gif_filename(fn):
    return os.path.splitext(fn)[-1] in ['.gif']


def is_video_filename(fn):
    return os.path.splitext(fn)[-1] in ['.mp4', '.m4v', '.gif']


def get_imageio_format(fn):
    """Ask imageio if it knows what to do with this filename."""
    request = imageio.core.Request(fn, 'w?')
    return imageio.formats.search_write_format(request)


class ExportImageDialog(QtWidgets.QDialog):
    default_filename = 'sift_screenshot.png'

    def __init__(self, parent):
        super(ExportImageDialog, self).__init__(parent)

        self.ui = export_image_dialog_ui.Ui_ExportImageDialog()
        self.ui.setupUi(self)

        self.ui.animationGroupBox.setDisabled(True)
        self.ui.constantDelaySpin.setDisabled(True)
        self.ui.constantDelaySpin.setValue(100)
        self.ui.timeLapseRadio.setChecked(True)
        self.ui.timeLapseRadio.clicked.connect(self._delay_clicked)
        self.ui.constantDelayRadio.clicked.connect(self._delay_clicked)
        self._delay_clicked()

        self.ui.frameRangeFrom.setValidator(QtGui.QIntValidator(1, 1))
        self.ui.frameRangeTo.setValidator(QtGui.QIntValidator(1, 1))
        self.ui.saveAsLineEdit.textChanged.connect(self._validate_filename)
        self.ui.saveAsButton.clicked.connect(self._show_file_dialog)

        self._last_dir = USER_DESKTOP_DIRECTORY
        self.ui.saveAsLineEdit.setText(os.path.join(self._last_dir, self.default_filename))
        self._validate_filename()

        self.ui.includeFooterCheckbox.clicked.connect(self._footer_changed)
        self._footer_changed()

        self.ui.frameAllRadio.clicked.connect(self.change_frame_range)
        self.ui.frameCurrentRadio.clicked.connect(self.change_frame_range)
        self.ui.frameRangeRadio.clicked.connect(self.change_frame_range)
        self.change_frame_range()  # set default

    def set_total_frames(self, n):
        self.ui.frameRangeFrom.validator().setBottom(1)
        self.ui.frameRangeTo.validator().setBottom(2)
        self.ui.frameRangeFrom.validator().setTop(n - 1)
        self.ui.frameRangeTo.validator().setTop(n)
        if (self.ui.frameRangeFrom.text() == '' or
                int(self.ui.frameRangeFrom.text()) > n - 1):
            self.ui.frameRangeFrom.setText('1')
        if self.ui.frameRangeTo.text() in ['', '1']:
            self.ui.frameRangeTo.setText(str(n))

    def _delay_clicked(self):
        if self.ui.constantDelayRadio.isChecked():
            self.ui.constantDelaySpin.setDisabled(False)
        else:
            self.ui.constantDelaySpin.setDisabled(True)

    def _footer_changed(self):
        if self.ui.includeFooterCheckbox.isChecked():
            self.ui.footerFontSizeSpinBox.setDisabled(False)
        else:
            self.ui.footerFontSizeSpinBox.setDisabled(True)

    def _show_file_dialog(self):
        fn = QtWidgets.QFileDialog.getSaveFileName(
            self, caption=self.tr('Screenshot Filename'),
            directory=os.path.join(self._last_dir, self.default_filename),
            filter=self.tr('Image Files (*.png *.jpg *.gif *.mp4 *.m4v)'),
            options=QtWidgets.QFileDialog.DontConfirmOverwrite)[0]
        if fn:
            self.ui.saveAsLineEdit.setText(fn)
        # bring this dialog back in focus
        self.raise_()
        self.activateWindow()

    def _validate_filename(self):
        t = self.ui.saveAsLineEdit.text()
        bt = self.ui.buttonBox.button(QtWidgets.QDialogButtonBox.Save)
        if not t or os.path.splitext(t)[-1] not in ['.png', '.jpg', '.gif', '.mp4', '.m4v']:
            bt.setDisabled(True)
        else:
            self._last_dir = os.path.dirname(t)
            bt.setDisabled(False)
        self._check_animation_controls()

    def _is_gif_filename(self):
        fn = self.ui.saveAsLineEdit.text()
        return is_gif_filename(fn)

    def _is_video_filename(self):
        fn = self.ui.saveAsLineEdit.text()
        return is_video_filename(fn)

    def _get_append_direction(self):
        if self.ui.colorbarVerticalRadio.isChecked():
            return "vertical"
        elif self.ui.colorbarHorizontalRadio.isChecked():
            return "horizontal"
        else:
            return None

    def _check_animation_controls(self):
        is_gif = self._is_gif_filename()
        is_video = self._is_video_filename()
        is_frame_current = self.ui.frameCurrentRadio.isChecked()
        is_valid_frame_to = self.ui.frameRangeTo.validator().top() == 1
        disable = is_frame_current or is_valid_frame_to

        self.ui.animationGroupBox.setDisabled(disable or not is_gif)
        self.ui.frameDelayGroup.setDisabled(disable or not (is_gif or is_video))

    def change_frame_range(self):
        if self.ui.frameRangeRadio.isChecked():
            self.ui.frameRangeFrom.setDisabled(False)
            self.ui.frameRangeTo.setDisabled(False)
        else:
            self.ui.frameRangeFrom.setDisabled(True)
            self.ui.frameRangeTo.setDisabled(True)

        self._check_animation_controls()

    def get_frame_range(self):
        if self.ui.frameCurrentRadio.isChecked():
            frame = None
        elif self.ui.frameAllRadio.isChecked():
            frame = [None, None]
        elif self.ui.frameRangeRadio.isChecked():
            frame = [
                int(self.ui.frameRangeFrom.text()),
                int(self.ui.frameRangeTo.text())
            ]
        else:
            LOG.error("Unknown frame range selection")
            return
        return frame

    def get_info(self):
        if self.ui.timeLapseRadio.isChecked():
            fps = None
        elif self.ui.constantDelayRadio.isChecked():
            delay = self.ui.constantDelaySpin.value()
            fps = 1000 / delay
        elif self.ui.fpsDelayRadio.isChecked():
            fps = self.ui.fpsDelaySpin.value()

        # loop is actually an integer of number of times to loop (0 infinite)
        info = {
            'frame_range': self.get_frame_range(),
            'include_footer': self.ui.includeFooterCheckbox.isChecked(),
            # 'transparency': self.ui.transparentCheckbox.isChecked(),
            'loop': self.ui.loopRadio.isChecked(),
            'filename': self.ui.saveAsLineEdit.text(),
            'fps': fps,
            'font_size': self.ui.footerFontSizeSpinBox.value(),
            'colorbar': self._get_append_direction(),
        }
        return info

    def show(self):
        self._check_animation_controls()
        return super(ExportImageDialog, self).show()


class ExportImageHelper(QtCore.QObject):
    """Handle all the logic for creating screenshot images"""
    default_font = os.path.join(DATA_DIR, 'fonts', 'Andale Mono.ttf')

    def __init__(self, parent, doc, sgm):
        """Initialize helper with defaults and other object handles.

        Args:
            doc: Main ``Document`` object for frame metadata
            sgm: ``SceneGraphManager`` object to get image data
        """
        super(ExportImageHelper, self).__init__(parent)
        self.doc = doc
        self.sgm = sgm
        self._screenshot_dialog = None

    def take_screenshot(self):
        if not self._screenshot_dialog:
            self._screenshot_dialog = ExportImageDialog(self.parent())
            self._screenshot_dialog.accepted.connect(self._save_screenshot)
        self._screenshot_dialog.set_total_frames(max(self.sgm.layer_set.max_frame, 1))
        self._screenshot_dialog.show()

    def _add_screenshot_footer(self, im, banner_text, font_size=11):
        orig_w, orig_h = im.size
        font = ImageFont.truetype(self.default_font, font_size)
        banner_h = font_size
        new_im = Image.new(im.mode, (orig_w, orig_h + banner_h), "black")
        new_draw = ImageDraw.Draw(new_im)
        new_draw.rectangle([0, orig_h, orig_w, orig_h + banner_h], fill="#000000")
        # give one extra pixel on the left to make sure letters
        # don't get cut off
        new_draw.text([1, orig_h], banner_text, fill="#ffffff", font=font)
        txt_w, txt_h = new_draw.textsize("SIFT", font)
        new_draw.text([orig_w - txt_w, orig_h], "SIFT", fill="#ffffff", font=font)
        new_im.paste(im, (0, 0, orig_w, orig_h))
        return new_im

    def _create_colorbar(self, mode, u, size):
        mpl.rcParams['font.sans-serif'] = FONT
        mpl.rcParams.update({'font.size': TICK_SIZE})

        colors = COLORMAP_MANAGER[self.doc.colormap_for_uuid(u)]
        if self.doc.prez_for_uuid(u).colormap == 'Square Root (Vis Default)':
            colors = colors.map(numpy.linspace((0, 0, 0, 1), (1, 1, 1, 1), 256))
        else:
            colors = colors.colors.rgba

        dpi = self.sgm.main_canvas.dpi
        if mode == 'vertical':
            fig = plt.figure(figsize=(size[0] / dpi * .1, size[1] / dpi * 1.2), dpi=dpi)
            ax = fig.add_axes([0.3, 0.05, 0.2, 0.9])
        else:
            fig = plt.figure(figsize=(size[0] / dpi * 1.2, size[1] / dpi * .1), dpi=dpi)
            ax = fig.add_axes([0.05, 0.4, 0.9, 0.2])

        cmap = mpl.colors.ListedColormap(colors)
        vmin, vmax = self.doc.prez_for_uuid(u).climits
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        cbar = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation=mode)

        ticks = [str(self.doc[u][Info.UNIT_CONVERSION][2](self.doc[u][Info.UNIT_CONVERSION][1](t)))
                 for t in numpy.linspace(vmin, vmax, NUM_TICKS)]
        cbar.set_ticks(numpy.linspace(vmin, vmax, NUM_TICKS))
        cbar.set_ticklabels(ticks)

        return fig

    def _append_colorbar(self, mode, im, u):
        if mode is None or COLORMAP_MANAGER.get(self.doc.colormap_for_uuid(u)) is None:
            return im

        fig = self._create_colorbar(mode, u, im.size)

        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=self.sgm.main_canvas.dpi)
        buf.seek(0)
        fig_im = Image.open(buf)

        fig_im.thumbnail(im.size)
        orig_w, orig_h = im.size
        fig_w, fig_h = fig_im.size

        offset = 0
        if mode == 'vertical':
            new_im = Image.new(im.mode, (orig_w + fig_w, orig_h))
            for i in [im, fig_im]:
                new_im.paste(i, (offset, 0))
                offset += i.size[0]
            return new_im
        else:
            new_im = Image.new(im.mode, (orig_w, orig_h + fig_h))
            for i in [im, fig_im]:
                new_im.paste(i, (0, offset))
                offset += i.size[1]
            return new_im

    def _create_filenames(self, uuids, base_filename):
        if not uuids or uuids[0] is None:
            return [None], [base_filename]
        filenames = []
        # only use the first uuid to fill in filename information
        file_uuids = uuids[:1] if is_video_filename(base_filename) else uuids
        for u in file_uuids:
            layer_info = self.doc[u]
            fn = base_filename.format(
                start_time=layer_info[Info.SCHED_TIME],
                scene=Info.SCENE,
                instrument=Info.INSTRUMENT,
            )
            filenames.append(fn)

        # check for duplicate filenames
        if len(filenames) > 1 and all(filenames[0] == fn for fn in filenames):
            ext = os.path.splitext(filenames[0])[-1]
            filenames = [os.path.splitext(fn)[0] + "_{:03d}".format(i + 1) + ext for i, fn in enumerate(filenames)]

        return uuids, filenames

    def _overwrite_dialog(self):
        msg = QtWidgets.QMessageBox(self.parent())
        msg.setWindowTitle("Overwrite File(s)?")
        msg.setText("One or more files already exist.")
        msg.setInformativeText("Do you want to overwrite existing files?")
        msg.setStandardButtons(msg.Cancel)
        msg.setDefaultButton(msg.Cancel)
        msg.addButton("Overwrite All", msg.YesRole)
        # XXX: may raise "modalSession has been exited prematurely" for pyqt4 on mac
        ret = msg.exec_()
        if ret == msg.Cancel:
            # XXX: This could technically reach a recursion limit
            self.take_screenshot()
            return False
        return True

    def _get_animation_parameters(self, info, images):
        params = {}
        if info['fps'] is None:
            t = [self.doc[u][Info.SCHED_TIME] for u, im in images]
            t_diff = [max(1, (t[i] - t[i - 1]).total_seconds()) for i in range(1, len(t))]
            min_diff = float(min(t_diff))
            # imageio seems to be using duration in seconds
            # so use 1/10th of a second
            duration = [.1 * (this_diff / min_diff) for this_diff in t_diff]
            duration = [duration[0]] + duration
            if not info['loop']:
                duration = duration + duration[-2:0:-1]
            params['duration'] = duration
        else:
            params['fps'] = info['fps']

        if is_gif_filename(info['filename']):
            params['loop'] = 0  # infinite number of loops
        elif 'duration' in params:
            # not gif but were given "Time Lapse", can only have one FPS
            params['fps'] = int(1. / params.pop('duration')[0])

        return params

    def _convert_frame_range(self, frame_range):
        """Convert 1-based frame range to SGM's 0-based"""
        if frame_range is None:
            return None
        s, e = frame_range
        # user provided frames are 1-based, scene graph are 0-based
        if s is None:
            s = 1
        if e is None:
            e = max(self.sgm.layer_set.max_frame, 1)
        return s - 1, e - 1

    def _save_screenshot(self):
        info = self._screenshot_dialog.get_info()
        LOG.info("Exporting image with options: {}".format(info))
        info['frame_range'] = self._convert_frame_range(info['frame_range'])
        if info['frame_range']:
            s, e = info['frame_range']
            uuids = self.sgm.layer_set.frame_order[s: e + 1]
        else:
            uuids = [self.sgm.layer_set.top_layer_uuid()]
        uuids, filenames = self._create_filenames(uuids, info['filename'])

        # check for existing filenames
        if (any(os.path.isfile(fn) for fn in filenames) and
                not self._overwrite_dialog()):
            return

        # get canvas screenshot arrays (numpy arrays of canvas pixels)
        img_arrays = self.sgm.get_screenshot_array(info['frame_range'])
        if not img_arrays or len(uuids) != len(img_arrays):
            LOG.error("Number of frames does not equal number of UUIDs")
            return

        images = [(u, Image.fromarray(x)) for u, x in img_arrays]

        if info['colorbar'] is not None:
            images = [(u, self._append_colorbar(info['colorbar'], im, u)) for (u, im) in images]

        if info['include_footer']:
            banner_text = [self.doc[u][Info.DISPLAY_NAME] if u else "" for u, im in images]
            images = [(u, self._add_screenshot_footer(im, bt, font_size=info['font_size'])) for (u, im), bt in
                      zip(images, banner_text)]

        imageio_format = get_imageio_format(filenames[0])
        if imageio_format:
            format_name = imageio_format.name
        elif filenames[0].upper().endswith('.M4V'):
            format_name = 'MP4'
        else:
            raise ValueError("Not sure how to handle file with format: {}".format(filenames[0]))

        if is_video_filename(filenames[0]) and len(images) > 1:
            params = self._get_animation_parameters(info, images)
            if not info['loop'] and is_gif_filename(filenames[0]):
                # rocking animation
                # we want frames 0, 1, 2, 3, 2, 1
                # NOTE: this must be done *after* we get animation properties
                images = images + images[-2:0:-1]

            filenames = [(filenames[0], images)]
        else:
            params = {}
            filenames = list(zip(filenames, [[x] for x in images]))

        for filename, file_images in filenames:
            writer = imageio.get_writer(filename, format_name, **params)
            for u, x in file_images:
                writer.append_data(numpy.array(x))

        writer.close()
