from __future__ import annotations

import io
import logging
import os
from fractions import Fraction

import imageio.v3 as imageio
import matplotlib as mpl
import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from PyQt5 import QtCore, QtGui, QtWidgets

from uwsift.common import Info
from uwsift.model.layer_model import LayerModel
from uwsift.ui import export_image_dialog_ui
from uwsift.util import USER_DESKTOP_DIRECTORY, get_package_data_dir
from uwsift.view.colormap import COLORMAP_MANAGER

LOG = logging.getLogger(__name__)
DATA_DIR = get_package_data_dir()

NUM_TICKS = 8
TICK_SIZE = 14
FONT = "arial"
PYAV_ANIMATION_PARAMS = {
    "codec": "libx264",
    "plugin": "pyav",
    "in_pixel_format": "rgba",
}


def is_gif_filename(fn):
    return os.path.splitext(fn)[-1] in [".gif"]


def is_video_filename(fn):
    return os.path.splitext(fn)[-1] in [".mp4", ".m4v", ".gif"]


class ExportImageDialog(QtWidgets.QDialog):
    default_filename = "sift_screenshot.png"

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

        self.ui.frameAllRadio.clicked.connect(self._change_frame_range)
        self.ui.frameCurrentRadio.clicked.connect(self._change_frame_range)
        self.ui.frameRangeRadio.clicked.connect(self._change_frame_range)
        self._change_frame_range()  # set default

    def set_total_frames(self, n):
        self.ui.frameRangeFrom.validator().setBottom(1)
        self.ui.frameRangeTo.validator().setBottom(2)
        self.ui.frameRangeFrom.validator().setTop(n - 1)
        self.ui.frameRangeTo.validator().setTop(n)
        if self.ui.frameRangeFrom.text() == "" or int(self.ui.frameRangeFrom.text()) > n - 1:
            self.ui.frameRangeFrom.setText("1")
        if self.ui.frameRangeTo.text() in ["", "1"]:
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
            self,
            caption=self.tr("Screenshot Filename"),
            directory=os.path.join(self._last_dir, self.default_filename),
            filter=self.tr("Image Files (*.png *.jpg *.gif *.mp4 *.m4v)"),
            options=QtWidgets.QFileDialog.DontConfirmOverwrite,
        )[0]
        if fn:
            self.ui.saveAsLineEdit.setText(fn)
        # bring this dialog back in focus
        self.raise_()
        self.activateWindow()

    def _validate_filename(self):
        t = self.ui.saveAsLineEdit.text()
        bt = self.ui.buttonBox.button(QtWidgets.QDialogButtonBox.Save)
        if not t or os.path.splitext(t)[-1] not in [".png", ".jpg", ".gif", ".mp4", ".m4v"]:
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

    def _change_frame_range(self):
        if self.ui.frameRangeRadio.isChecked():
            self.ui.frameRangeFrom.setDisabled(False)
            self.ui.frameRangeTo.setDisabled(False)
        else:
            self.ui.frameRangeFrom.setDisabled(True)
            self.ui.frameRangeTo.setDisabled(True)

        self._check_animation_controls()

    def _get_frame_range(self):
        if self.ui.frameCurrentRadio.isChecked():
            frame = None
        elif self.ui.frameAllRadio.isChecked():
            frame = [None, None]
        elif self.ui.frameRangeRadio.isChecked():
            frame = [int(self.ui.frameRangeFrom.text()), int(self.ui.frameRangeTo.text())]
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
            "frame_range": self._get_frame_range(),
            "include_footer": self.ui.includeFooterCheckbox.isChecked(),
            # 'transparency': self.ui.transparentCheckbox.isChecked(),
            "loop": self.ui.loopRadio.isChecked(),
            "filename": self.ui.saveAsLineEdit.text(),
            "fps": fps,
            "font_size": self.ui.footerFontSizeSpinBox.value(),
            "colorbar": self._get_append_direction(),
        }
        return info

    def show(self):
        self._check_animation_controls()
        return super(ExportImageDialog, self).show()


class ExportImageHelper(QtCore.QObject):
    """Handle all the logic for creating screenshot images"""

    default_font = os.path.join(DATA_DIR, "fonts", "Andale Mono.ttf")

    def __init__(self, parent, sgm, model: LayerModel):
        """Initialize helper with defaults and other object handles.

        Args:
            doc: Main ``Document`` object for frame metadata
            sgm: ``SceneGraphManager`` object to get image data
        """
        super(ExportImageHelper, self).__init__(parent)
        self.sgm = sgm
        self.model = model
        self._screenshot_dialog = None

    def take_screenshot(self):
        if not self._screenshot_dialog:
            self._screenshot_dialog = ExportImageDialog(self.parent())
            self._screenshot_dialog.accepted.connect(self._save_screenshot)
        frame_count = self.sgm.animation_controller.get_frame_count()
        self._screenshot_dialog.set_total_frames(max(frame_count, 1))
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
        txt_w = new_draw.textlength("SIFT", font)
        new_draw.text([orig_w - txt_w, orig_h], "SIFT", fill="#ffffff", font=font)
        new_im.paste(im, (0, 0, orig_w, orig_h))
        return new_im

    def _create_colorbar(self, mode, u, size):
        mpl.rcParams["font.sans-serif"] = FONT
        mpl.rcParams.update({"font.size": TICK_SIZE})

        colormap = self.model.get_dataset_presentation_by_uuid(u).colormap
        colors = COLORMAP_MANAGER[colormap]

        if colormap == "Square Root (Vis Default)":
            colors = colors.map(np.linspace((0, 0, 0, 1), (1, 1, 1, 1), 256))
        else:
            colors = colors.colors.rgba

        dpi = self.sgm.main_canvas.dpi
        if mode == "vertical":
            fig = plt.figure(figsize=(size[0] / dpi * 0.1, size[1] / dpi * 1.2), dpi=dpi)
            ax = fig.add_axes([0.3, 0.05, 0.2, 0.9])
        else:
            fig = plt.figure(figsize=(size[0] / dpi * 1.2, size[1] / dpi * 0.1), dpi=dpi)
            ax = fig.add_axes([0.05, 0.4, 0.9, 0.2])

        cmap = mpl.colors.ListedColormap(colors)
        vmin, vmax = self.model.get_dataset_presentation_by_uuid(u).climits
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        cbar = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation=mode)

        ticks = [
            str(
                self.model.get_dataset_by_uuid(u).info.get(Info.UNIT_CONVERSION)[2](
                    self.model.get_dataset_by_uuid(u).info.get(Info.UNIT_CONVERSION)[1](t)
                )
            )
            for t in np.linspace(vmin, vmax, NUM_TICKS)
        ]
        cbar.set_ticks(np.linspace(vmin, vmax, NUM_TICKS))
        cbar.set_ticklabels(ticks)

        return fig

    def _append_colorbar(self, mode, im, u):
        if (
            mode is None
            or not self.model.get_dataset_presentation_by_uuid(u)
            or COLORMAP_MANAGER.get(self.model.get_dataset_presentation_by_uuid(u).colormap) is None
        ):
            return im

        fig = self._create_colorbar(mode, u, im.size)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=self.sgm.main_canvas.dpi)
        plt.close(fig)
        buf.seek(0)
        fig_im = Image.open(buf)

        fig_im.thumbnail(im.size)
        orig_w, orig_h = im.size
        fig_w, fig_h = fig_im.size

        offset = 0
        if mode == "vertical":
            new_im = Image.new(im.mode, (orig_w + fig_w, orig_h))
            for i in [im, fig_im]:
                new_im.paste(i, (offset, 0))
                offset += i.size[0]
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
            dataset_info = self.model.get_dataset_by_uuid(u).info
            fn = base_filename.format(
                start_time=dataset_info[Info.SCHED_TIME],
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
        if info["fps"] is None:
            params["duration"] = self._get_time_lapse_duration(images, info["loop"])
        else:
            params["fps"] = info["fps"]

        is_gif = is_gif_filename(info["filename"])
        if is_gif:
            params["loop"] = 0  # infinite number of loops
            if "fps" in params:
                # PIL duration in milliseconds
                params["duration"] = [1.0 / params.pop("fps") * 1000.0] * len(images)
        else:
            if "duration" in params:
                # not gif but were given "Time Lapse", can only have one FPS
                params["fps"] = 1.0 / params.pop("duration")[0]
            params.update(PYAV_ANIMATION_PARAMS)
            params["fps"] = Fraction(float(params["fps"])).limit_denominator(65535)
        return params

    def _get_time_lapse_duration(self, images, is_loop):
        if len(images) <= 1:
            return [1.0]  # arbitrary single frame duration
        t = [self.model.get_dataset_by_uuid(u).info.get(Info.SCHED_TIME) for u, im in images]
        t_diff = [max(1, (t[i] - t[i - 1]).total_seconds()) for i in range(1, len(t))]
        min_diff = float(min(t_diff))
        # imageio seems to be using duration in seconds
        # so use 1/10th of a second
        duration = [0.1 * (this_diff / min_diff) for this_diff in t_diff]
        duration = [duration[0]] + duration
        if not is_loop:
            duration = duration + duration[-2:0:-1]
        return duration

    def _convert_frame_range(self, frame_range):
        """Convert 1-based frame range to SGM's 0-based"""
        if frame_range is None:
            return None
        s, e = frame_range
        # user provided frames are 1-based, scene graph are 0-based
        if s is None:
            s = 1
        if e is None:
            e = max(self.sgm.animation_controller.get_frame_count(), 1)
        return s - 1, e - 1

    def _save_screenshot(self):
        info = self._screenshot_dialog.get_info()
        LOG.info("Exporting image with options: {}".format(info))
        info["frame_range"] = self._convert_frame_range(info["frame_range"])
        if info["frame_range"]:
            s, e = info["frame_range"]
        else:
            s = e = self.sgm.animation_controller.get_current_frame_index()

        uuids = self.sgm.animation_controller.get_frame_uuids()[s : e + 1]

        uuids, filenames = self._create_filenames(uuids, info["filename"])

        # check for existing filenames
        if any(os.path.isfile(fn) for fn in filenames) and not self._overwrite_dialog():
            return

        # get canvas screenshot arrays (numpy arrays of canvas pixels)
        img_arrays = self.sgm.get_screenshot_array(info["frame_range"])
        if not img_arrays or len(uuids) != len(img_arrays):
            LOG.error(
                f"Number of frames: {0 if not img_arrays else len(img_arrays)}"
                f" does not equal "
                f"number of UUIDs: {len(uuids)}"
            )
            return

        images = [(u, Image.fromarray(x)) for u, x in img_arrays]

        if info["colorbar"] is not None:
            images = [(u, self._append_colorbar(info["colorbar"], im, u)) for (u, im) in images]

        if info["include_footer"]:
            banner_text = [
                self.model.get_dataset_by_uuid(u).info.get(Info.DISPLAY_NAME) if u else "" for u, im in images
            ]
            images = [
                (u, self._add_screenshot_footer(im, bt, font_size=info["font_size"]))
                for (u, im), bt in zip(images, banner_text)
            ]

        if is_video_filename(filenames[0]):
            params = self._get_animation_parameters(info, images)
            if not info["loop"] and is_gif_filename(filenames[0]):
                # rocking animation
                # we want frames 0, 1, 2, 3, 2, 1
                # NOTE: this must be done *after* we get animation properties
                images = images + images[-2:0:-1]

            filenames = [(filenames[0], images)]
        else:
            params = {}
            filenames = list(zip(filenames, [[x] for x in images]))

        self._write_images(filenames, params)

    def _write_images(self, filenames, params):
        for filename, file_images in filenames:
            images_arrays = _image_to_frame_array(file_images, filename)
            try:
                imageio.imwrite(filename, images_arrays, **params)
            except IOError:
                msg = "Failed to write to file: {}".format(filename)
                LOG.error(msg)
                raise


def _image_to_frame_array(file_images: list[Image], filename: str) -> list[npt.NDArray[np.uint8]]:
    images_arrays = [np.array(image) for _, image in file_images]
    if not _supports_rgba(filename):
        images_arrays = [image_arr[:, :, :3] for image_arr in images_arrays]
    # make sure frames are divisible by 2 to make ffmpeg happy
    if is_video_filename(filename) and not is_gif_filename(filename):
        images_arrays = [_array_divisible_by_2(img_array) for img_array in images_arrays]
    return images_arrays


def _array_divisible_by_2(img_array: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    shape = img_array.shape
    shape_by_2 = tuple(dim_size - dim_size % 2 for dim_size in shape)
    if shape_by_2 == shape:
        return img_array
    return img_array[: shape_by_2[0], : shape_by_2[1], :]


def _supports_rgba(filename: str) -> bool:
    return not (filename.endswith(".jpeg") or filename.endswith(".jpg"))
