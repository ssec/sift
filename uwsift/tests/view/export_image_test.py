from uwsift.__main__ import Main
from uwsift.util.default_paths import USER_CONFIG_DIR
import pytest
from numpy.testing import assert_array_equal
from PIL import Image
from matplotlib import pyplot as plt


window = Main(config_dir=USER_CONFIG_DIR)


def _get_mock_doc():
    class MockPrez():
        def __init__(self):
            self.colormap = 'Rainbow (IR Default)'
            self.climits = (0, 1)

    class MockDocBasicLayer():
        def __init__(self):
            self.attrs = {}
            self.attrs['unit_conversion'] = ('unit', lambda t: t, lambda t: t)

        def __getitem__(self, item):
            return self.attrs[item]

    class MockDoc():
        def __init__(self):
            self.layer = MockDocBasicLayer()
            self.prez = MockPrez()

        def __getitem__(self, item):
            return self.layer

        def colormap_for_uuid(self, u):
            return 'Rainbow (IR Default)'

        def prez_for_uuid(self, u):
            return MockPrez()

    return MockDoc()

@pytest.mark.parametrize("size,exp,mode", [
    ((100, 100), [0.1, 1.2], 'vertical'),
    ((100, 100), [1.2, 0.1], 'horizontal'),
    ((0, 0), [0, 0], 'vertical'),
])
def test_create_colorbar(size, exp, mode, monkeypatch):
    monkeypatch.setattr(window.export_image, 'doc', _get_mock_doc())
    monkeypatch.setattr(window.export_image.sgm.main_canvas, 'dpi', 100)

    res = window.export_image._create_colorbar(mode, None, size)

    assert_array_equal(res.get_size_inches(), exp)
    assert res.dpi == 100

@pytest.mark.parametrize("mode,size,cbar_size", [
    (None, (100, 100), (0, 0)),
    ('vertical', (108, 100), (10, 120)),
    ('horizontal', (100, 108), (120, 10)),
])
def test_append_colorbar(mode, size, cbar_size, monkeypatch):
    def mock_colorbar(m, u, s):
        fig = plt.figure(figsize=cbar_size)
        return fig

    monkeypatch.setattr(window.export_image, 'doc', _get_mock_doc())
    monkeypatch.setattr(window.export_image.sgm.main_canvas, 'dpi', 100)
    monkeypatch.setattr(window.export_image, '_create_colorbar', mock_colorbar)

    test_im = Image.new('RGBA', (100, 100))
    res = window.export_image._append_colorbar(mode, test_im, None)

    assert res.size == size
