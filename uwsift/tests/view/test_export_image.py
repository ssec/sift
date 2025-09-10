from __future__ import annotations

import datetime
import os
from typing import Any, Optional
from unittest.mock import MagicMock, Mock, patch

import imageio.v3 as imageio
import numpy as np
import pytest
from matplotlib import pyplot as plt
from numpy.testing import assert_array_equal
from PIL import Image
from PyQt5.QtCore import Qt

from uwsift.common import Info
from uwsift.view import export_image


def _get_mock_model():
    """Mock LayerModel class for testing."""

    class MockPrez:
        def __init__(self):
            self.colormap = "Rainbow (IR Default)"
            self.climits = (0, 1)

    class MockDataset:
        def __init__(self, offset: int):
            self.info: dict[Info, Any] = {}
            self.info[Info.UNIT_CONVERSION] = ("unit", lambda t: t, lambda t: t)
            self.info[Info.SCHED_TIME] = datetime.datetime(2000, 1, 1, 0, 0, 0, 0) + datetime.timedelta(minutes=offset)
            self.info[Info.DISPLAY_NAME] = "name"

    class MockModel:
        def __init__(self):
            self.moc_prez = MockPrez()
            self._offset = 0
            self._offset_for_uuid = {}

        def get_dataset_presentation_by_uuid(self, u):
            return self.moc_prez

        def get_dataset_by_uuid(self, u):
            self._offset_for_uuid[u] = self._offset
            ds = MockDataset(self._offset)
            self._offset += 1
            return ds

    return MockModel()


def _get_mock_sd(
    frame_range: Optional[list[int]],
    filename: str,
    fps: Optional[float] = None,
):
    """Mock ScreenshotDialog class for testing."""

    class MockScreenshotDialog:
        def __init__(self):
            self.info = {
                "frame_range": frame_range,
                "include_footer": True,
                "filename": filename,
                "colorbar": "vertical",
                "font_size": 10,
                "loop": True,
                "fps": fps,
            }

        def get_info(self):
            return self.info

    return MockScreenshotDialog()


def _get_mock_sgm(frame_order):
    """Mock SceneGraphManager class for testing."""
    return _MockSGM(frame_order)


class _MockAnimationController:
    def __init__(self, frame_order):
        self._frame_order = frame_order

    def get_current_frame_index(self):
        # frame range of `False` (in these unit tests) means no data is loaded
        return max(self._frame_order or 1, 1) if self._frame_order is not False else 0

    def get_frame_uuids(self):
        if self._frame_order is False:
            # no data loaded, so no UUIDs
            return []
        if self._frame_order is None:
            # single "current layer" being shown
            # start fake UUIDs at 1 to avoid `if not uuid:` failures with 0
            return [1]
        # no need to get more UUIDs than what we're going to use in the test
        return list(range(1, max(self._frame_order) + 1))


class _MockCanvas:
    dpi = 100


class _MockSGM:
    fake_screenshot_shape = (5, 10, 4)

    def __init__(self, frame_order):
        self.animation_controller = _MockAnimationController(frame_order)
        self.main_canvas = _MockCanvas()
        self.rng = np.random.default_rng()
        self._frame_order = frame_order

    def get_screenshot_array(self, fr, _size):
        if self._frame_order is False and fr is None:
            # no data loaded
            return [("", self.rng.integers(0, 255, self.fake_screenshot_shape, dtype=np.uint8))]
        if fr is None:
            fr = (1, 1)
        frames = []
        for frame_idx in range(fr[0], fr[1] + 1):
            frames.append(
                (
                    str(frame_idx),
                    self.rng.integers(0, 255, self.fake_screenshot_shape, dtype=np.uint8),
                )
            )
        return frames


@pytest.mark.parametrize(
    "size,mode,exp",
    [
        ((100, 100), "vertical", [0.1, 1.2]),
        ((100, 100), "horizontal", [1.2, 0.1]),
    ],
)
def test_create_colorbar(size, mode, exp, monkeypatch, window):
    """Test colorbar is created correctly given dimensions and the colorbar append direction."""
    monkeypatch.setattr(window.export_image, "model", _get_mock_model())
    monkeypatch.setattr(window.export_image.sgm.main_canvas, "dpi", 100)

    res = window.export_image._create_colorbar(mode, None, size)

    assert_array_equal(res.get_size_inches(), exp)
    assert res.dpi == 100


@pytest.mark.parametrize(
    "mode,cbar_size,exp",
    [
        (None, (0, 0), (100, 100)),
        ("vertical", (10, 120), (108, 100)),
        ("horizontal", (110, 10), (100, 109)),
    ],
)
def test_append_colorbar(mode, cbar_size, exp, monkeypatch, window):
    """Test colorbar is appended to the appropriate location given the colorbar append direction."""
    monkeypatch.setattr(window.export_image, "model", _get_mock_model())
    monkeypatch.setattr(window.export_image.sgm.main_canvas, "dpi", 100)
    monkeypatch.setattr(window.export_image, "_create_colorbar", lambda x, y, z: plt.figure(figsize=cbar_size))

    im = Image.new("RGBA", (100, 100))
    res = window.export_image._append_colorbar(mode, im, None)

    assert res.size == exp


@pytest.mark.parametrize("size,fs,exp", [((100, 100), 10, (100, 110))])
def test_add_screenshot_footer(size, fs, exp, window):
    """Test screenshot footer is appended correctly."""
    im = Image.new("RGBA", size)
    res = window.export_image._add_screenshot_footer(im, "text", font_size=fs)
    assert res.size == exp


@pytest.mark.parametrize(
    "range,exp",
    [
        (None, None),
        ((None, 2), (0, 1)),
        ((1, None), (0, 0)),
        ((1, 5), (0, 4)),
    ],
)
def test_convert_frame_range(range, exp, window):
    """Test frame range is converted correctly."""
    res = window.export_image._convert_frame_range(range)
    assert res == exp


@pytest.mark.parametrize(
    "info,exp",
    [
        ({"fps": None, "filename": "test.gif", "loop": 0}, {"duration": [0.1, 0.1], "loop": 0}),
        (
            {"fps": None, "filename": "test.mp4", "loop": 0},
            {
                "fps": 10,
                "codec": "libx264",
                "plugin": "pyav",
                "in_pixel_format": "rgba",
            },
        ),
        ({"fps": 1, "filename": "test.gif"}, {"duration": [1000.0, 1000.0], "loop": 0}),
        (
            {"fps": 1, "filename": "test.mp4"},
            {
                "fps": 1,
                "codec": "libx264",
                "plugin": "pyav",
                "in_pixel_format": "rgba",
            },
        ),
    ],
)
def test_get_animation_parameters(info, exp, monkeypatch, window):
    """Test animation parameters are calculated correctly."""
    monkeypatch.setattr(window.export_image, "model", _get_mock_model())
    im = Image.new("RGBA", (100, 100))

    res = window.export_image._get_animation_parameters(info, [(0, im), (0, im)])
    assert res == exp


@pytest.mark.parametrize("fn,exp", [("test.gif", True), ("test.png", False)])
def test_is_gif_filename(fn, exp):
    """Test that gif file names are recognized."""
    res = export_image.is_gif_filename(fn)
    assert res == exp


@pytest.mark.parametrize("fn,exp", [("test.gif", True), ("test.m4v", True), ("test.mp4", True), ("test.png", False)])
def test_is_video_filename(fn, exp):
    """Test that video file names are recognized."""
    res = export_image.is_video_filename(fn)
    assert res == exp


@pytest.mark.parametrize(
    "uuids,base,exp",
    [
        ([0, 0], "test.gif", ([0, 0], ["test.gif"])),
        ([0, 0], "test.png", ([0, 0], ["test_001.png", "test_002.png"])),
        (None, "test.gif", ([None], ["test.gif"])),
    ],
)
def test_create_filenames(uuids, base, exp, monkeypatch, window):
    """Test file names are created correctly."""
    monkeypatch.setattr(window.export_image, "model", _get_mock_model())
    res = window.export_image._create_filenames(uuids, base)
    assert res == exp


@pytest.mark.parametrize(
    "fr,fn,overwrite",
    [
        ([1, 2], "test.gif", True),
        ([1, 2], "test.m4v", True),
        ([1, 2], "test.mp4", True),
        (None, "test.m4v", True),
        ([1, 2], "test.gif", False),
    ],
)
@pytest.mark.parametrize("fps", [None, 2.2])
def test_save_screenshot_animations(fr, fn, overwrite, fps, monkeypatch, window, tmp_path):
    """Test screenshot is saved correctly given the frame range and filename."""
    fn = tmp_path / fn
    if overwrite:
        fn.touch()
        monkeypatch.setattr(window.export_image, "_overwrite_dialog", lambda: overwrite)

    monkeypatch.setattr(window.export_image, "_screenshot_dialog", _get_mock_sd(fr or None, str(fn), fps))
    monkeypatch.setattr(window.export_image, "sgm", _get_mock_sgm(fr))
    monkeypatch.setattr(window.export_image, "model", _get_mock_model())

    window.export_image._save_screenshot()
    assert fn.is_file()
    if fn.suffix == ".gif":
        exp_frame_shape = (15, 18)
        read_kwargs = {"plugin": "pillow", "mode": "RGBA"}
    else:
        exp_frame_shape = (14, 18)
        read_kwargs = {"plugin": "pyav"}
    frames = imageio.imread(fn, **read_kwargs)
    assert frames.shape[0] == (len(fr) if fr else 1)
    assert frames[0].shape[:2] == exp_frame_shape


@pytest.mark.parametrize(
    "fr,fn,overwrite",
    [
        ([1, 2], "test_{start_time:%H%M%S}.jpg", True),
        ([1, 2], "test_{start_time:%H%M%S}.png", True),
        ([1, 2], "test_{start_time:%H%M%S}.png", False),
        (None, "test_XXXXXX.png", True),
        (None, "test_XXXXXX.png", False),
        (False, "test_XXXXXX.png", False),
    ],
)
def test_save_screenshot_images(fr, fn, overwrite, monkeypatch, window, tmp_path):
    """Test screenshot is saved correctly given the frame range and filename."""
    exp_num = 1 if not fr else len(fr)
    expected_files = [
        tmp_path / fn.format(start_time=datetime.datetime(2000, 1, 1, 0, offset, 0)) for offset in range(exp_num)
    ]
    if overwrite:
        for exp_file in expected_files:
            exp_file.touch()
        monkeypatch.setattr(window.export_image, "_overwrite_dialog", lambda: overwrite)

    monkeypatch.setattr(window.export_image, "_screenshot_dialog", _get_mock_sd(fr or None, str(tmp_path / fn)))
    monkeypatch.setattr(window.export_image, "sgm", _get_mock_sgm(fr))
    monkeypatch.setattr(window.export_image, "model", _get_mock_model())

    window.export_image._save_screenshot()
    assert len(list(tmp_path.iterdir())) == len(expected_files)
    for exp_file in expected_files:
        assert exp_file.is_file()
        img = imageio.imread(exp_file)
        assert img.shape[:2] == (15, 18)


def test_cmd_open_export_image_dialog(qtbot, window):
    """Test that the keyboard shortcut Ctrl/Cmd + I opens the export image menu."""
    qtbot.addWidget(window)
    qtbot.keyClick(window, Qt.Key_I, Qt.ControlModifier)

    def check_dialog():
        assert window.export_image._screenshot_dialog is not None

    qtbot.waitUntil(check_dialog)


def test_export_image_dialog_info_default(qtbot, window):
    """Assert changing no options results in the default screenshot settings."""
    window.export_image.take_screenshot()
    qtbot.waitUntil(lambda: window.export_image._screenshot_dialog is not None)

    res = window.export_image._screenshot_dialog.get_info()

    # only look at the filename
    res["filename"] = os.path.split(res["filename"])[-1]
    # since the sizes are system dependent, we ignore them.
    res.pop("size", None)

    exp = {
        "frame_range": None,
        "include_footer": True,
        "loop": True,
        "filename": export_image.ExportImageDialog.default_filename,
        "fps": None,
        "font_size": 11,
        "colorbar": None,
    }

    assert res == exp


def test_export_image_dialog_info(qtbot, window):
    """Test changing the options in the export image GUI."""
    window.export_image.take_screenshot()
    qtbot.waitUntil(lambda: window.export_image._screenshot_dialog is not None)

    qtbot.keyClick(window.export_image._screenshot_dialog.ui.saveAsLineEdit, Qt.Key_A, Qt.ControlModifier)
    qtbot.keyClick(window.export_image._screenshot_dialog.ui.saveAsLineEdit, Qt.Key_Backspace)
    qtbot.keyClicks(window.export_image._screenshot_dialog.ui.saveAsLineEdit, "test.png")
    qtbot.keyClick(window.export_image._screenshot_dialog.ui.footerFontSizeSpinBox, Qt.Key_A, Qt.ControlModifier)
    qtbot.keyClick(window.export_image._screenshot_dialog.ui.footerFontSizeSpinBox, Qt.Key_Backspace)
    qtbot.keyClicks(window.export_image._screenshot_dialog.ui.footerFontSizeSpinBox, "20")
    qtbot.mouseClick(window.export_image._screenshot_dialog.ui.frameRangeRadio, Qt.LeftButton)
    qtbot.mouseClick(window.export_image._screenshot_dialog.ui.colorbarVerticalRadio, Qt.LeftButton)
    qtbot.mouseClick(window.export_image._screenshot_dialog.ui.includeFooterCheckbox, Qt.LeftButton)

    res = window.export_image._screenshot_dialog.get_info()
    res["filename"] = os.path.split(res["filename"])[-1]
    # since the sizes are system dependent, we ignore them.
    res.pop("size", None)

    exp = {
        "frame_range": [1, 1],
        "include_footer": False,
        "loop": True,
        "filename": "test.png",
        "fps": None,
        "font_size": 20,
        "colorbar": "vertical",
    }

    assert res == exp


@pytest.mark.parametrize(
    "fn,exp",
    [
        ("test.tif", True),
        ("test.tiff", True),
        ("test.TIF", True),
        ("test.TIFF", True),
        ("test.png", False),
        ("test.jpg", False),
    ],
)
def test_is_tif_filename(fn, exp):
    """Test that TIF/TIFF file names are recognized correctly."""
    res = export_image.is_tif_filename(fn)
    assert res == exp


@pytest.mark.parametrize(
    "images,filename,expected_channels",
    [
        # Test RGBA image (4 channels)
        ([("uuid1", Image.new("RGBA", (10, 10), (255, 0, 0, 128)))], "test.tif", 4),
        # Test RGB image (3 channels)
        ([("uuid1", Image.new("RGB", (10, 10), (255, 0, 0)))], "test.tif", 3),
    ],
)
def test_image_to_frame_array_tif_formats(images, filename, expected_channels):
    """Test conversion of PIL Images to numpy arrays for TIF files."""
    result = export_image._image_to_frame_array(images, filename)

    assert len(result) == 1
    print(f"Shape: {result[0].shape}")

    assert result[0].shape[2] == expected_channels
    assert result[0].shape[:2] == (10, 10)


def test_image_to_frame_array_jpeg_removes_alpha():
    """Test that JPEG files have alpha channel removed."""
    images = [("uuid1", Image.new("RGBA", (10, 10), (255, 0, 0, 128)))]
    result = export_image._image_to_frame_array(images, "test.jpg")

    assert len(result) == 1
    assert result[0].shape[2] == 3  # RGB only, no alpha


def test_supports_rgba():
    """Test RGBA support detection for different file formats."""
    assert export_image._supports_rgba("test.png")
    assert export_image._supports_rgba("test.tif")
    assert export_image._supports_rgba("test.tiff")
    assert export_image._supports_rgba("test.gif")
    assert not export_image._supports_rgba("test.jpg")
    assert not export_image._supports_rgba("test.jpeg")


@patch("rasterio.open")
def test_write_tif_file_with_projection(mock_rasterio_open, window):
    """Test writing TIF file with projection information."""
    # Setup mock projection parameters
    mock_proj_params = {"crs": "EPSG:4326", "transform": [1.0, 0.0, -180.0, 0.0, -1.0, 90.0]}

    # Mock the SGM's collect_projection_infos method
    window.export_image.sgm.collect_projection_infos = Mock(return_value=mock_proj_params)

    # Create test image array
    test_array = np.random.randint(0, 255, (100, 200, 4), dtype=np.uint8)

    # Mock rasterio context manager
    mock_dst = MagicMock()
    mock_rasterio_open.return_value.__enter__.return_value = mock_dst

    # Call the method
    window.export_image._write_tif_file("test.tif", [test_array])

    # Verify rasterio.open was called with correct parameters
    mock_rasterio_open.assert_called_once_with(
        "test.tif",
        "w",
        driver="GTiff",
        height=100,
        width=200,
        count=4,
        dtype=test_array.dtype,
        crs="EPSG:4326",
        transform=[1.0, 0.0, -180.0, 0.0, -1.0, 90.0],
        compress="lzw",
    )

    # Verify all channels were written
    assert mock_dst.write.call_count == 4


@patch("rasterio.open")
def test_write_tif_file_without_projection(mock_rasterio_open, window):
    """Test writing TIF file without projection information."""
    # Mock SGM to return None (no projection info)
    window.export_image.sgm.collect_projection_infos = Mock(return_value=None)

    # Create test image array
    test_array = np.random.randint(0, 255, (50, 75, 3), dtype=np.uint8)

    # Mock rasterio context manager
    mock_dst = MagicMock()
    mock_rasterio_open.return_value.__enter__.return_value = mock_dst

    # Call the method
    window.export_image._write_tif_file("test.tif", [test_array])

    # Verify rasterio.open was called with None for crs and transform
    mock_rasterio_open.assert_called_once_with(
        "test.tif",
        "w",
        driver="GTiff",
        height=50,
        width=75,
        count=3,
        dtype=test_array.dtype,
        crs=None,
        transform=None,
        compress="lzw",
    )


def test_write_tif_file_invalid_image_count(window, caplog):
    """Test error handling when multiple images are provided for single TIF."""
    test_arrays = [
        np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8),
        np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8),
    ]

    window.export_image._write_tif_file("test.tif", test_arrays)

    # Check that error was logged
    assert "Invalid number of geotiff image arrays: 2" in caplog.text


@pytest.mark.parametrize(
    "fr,fn,overwrite",
    [
        ([1, 2], "test_{start_time:%H%M%S}.tif", True),
        ([1, 2], "test_{start_time:%H%M%S}.tiff", True),
        (None, "test_single.tif", True),
    ],
)
@patch("uwsift.view.export_image.ExportImageHelper._write_tif_file")
def test_save_screenshot_tif_files(mock_write_tif, fr, fn, overwrite, monkeypatch, window, tmp_path):
    """Test screenshot saving for TIF/TIFF files."""
    exp_num = 1 if not fr else len(fr)
    expected_files = [
        tmp_path / fn.format(start_time=datetime.datetime(2000, 1, 1, 0, offset, 0)) for offset in range(exp_num)
    ]

    if overwrite:
        for exp_file in expected_files:
            exp_file.touch()
        monkeypatch.setattr(window.export_image, "_overwrite_dialog", lambda: overwrite)

    monkeypatch.setattr(window.export_image, "_screenshot_dialog", _get_mock_sd(fr or None, str(tmp_path / fn)))
    monkeypatch.setattr(window.export_image, "sgm", _get_mock_sgm(fr))
    monkeypatch.setattr(window.export_image, "model", _get_mock_model())

    window.export_image._save_screenshot()

    # Verify _write_tif_file was called for each expected file
    assert mock_write_tif.call_count == len(expected_files)

    # Verify the calls were made with correct filenames
    for call, expected_file in zip(mock_write_tif.call_args_list, expected_files):
        filename, image_arrays = call[0]
        assert filename == str(expected_file)
        assert len(image_arrays) == 1  # Single image per TIF file


@patch("rasterio.open")
def test_write_images_calls_tif_writer(mock_rasterio_open, window):
    """Test that _write_images calls the TIF writer for TIF files."""
    # Mock rasterio
    mock_dst = MagicMock()
    mock_rasterio_open.return_value.__enter__.return_value = mock_dst

    # Mock SGM projection info
    window.export_image.sgm.collect_projection_infos = Mock(return_value=None)

    # Create test data
    test_image = Image.new("RGB", (10, 10), (255, 0, 0))
    filenames = [("test.tif", [("uuid1", test_image)])]
    params = {}

    # Call the method
    window.export_image._write_images(filenames, params)

    # Verify rasterio.open was called
    mock_rasterio_open.assert_called_once()
