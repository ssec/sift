import pytest
import numpy as np
from uwsift.view.tile_calculator import (TileCalculator,
                                         calc_pixel_size,
                                         get_reference_points,
                                         _calc_extent_component,
                                         clip,
                                         calc_view_extents,
                                         max_tiles_available,
                                         calc_tile_slice,
                                         visible_tiles,
                                         calc_tile_fraction,
                                         calc_stride,
                                         calc_overview_stride,
                                         calc_vertex_coordinates,
                                         calc_texture_coordinates)
from uwsift.common import Point, Box, ViewBox, Resolution


@pytest.fixture(params=[False, True], autouse=True)
def disable_jit(request, monkeypatch):
    """Runs the tests with jit enabled and disabled."""
    if request.param:
        monkeypatch.setattr('uwsift.view.tile_calculator.calc_tile_slice', calc_tile_slice.py_func)
        monkeypatch.setattr('uwsift.view.tile_calculator.get_reference_points', get_reference_points.py_func)
        monkeypatch.setattr('uwsift.view.tile_calculator.calc_pixel_size', calc_pixel_size.py_func)
        monkeypatch.setattr('uwsift.view.tile_calculator._calc_extent_component', _calc_extent_component.py_func)
        monkeypatch.setattr('uwsift.view.tile_calculator.clip', clip.py_func)
        monkeypatch.setattr('uwsift.view.tile_calculator.calc_view_extents', calc_view_extents.py_func)
        monkeypatch.setattr('uwsift.view.tile_calculator.max_tiles_available', max_tiles_available.py_func)
        monkeypatch.setattr('uwsift.view.tile_calculator.visible_tiles', visible_tiles.py_func)
        monkeypatch.setattr('uwsift.view.tile_calculator.calc_tile_fraction', calc_tile_fraction.py_func)
        monkeypatch.setattr('uwsift.view.tile_calculator.calc_stride', calc_stride.py_func)
        monkeypatch.setattr('uwsift.view.tile_calculator.calc_overview_stride', calc_overview_stride.py_func)
        monkeypatch.setattr('uwsift.view.tile_calculator.calc_vertex_coordinates', calc_vertex_coordinates.py_func)
        monkeypatch.setattr('uwsift.view.tile_calculator.calc_texture_coordinates', calc_texture_coordinates.py_func)


@pytest.mark.parametrize("tc_params,vg,etiles,stride,exp", [
    (["test", (500, 500), Point(500000, -500000), Resolution(200, 200), (50, 50)],
     ViewBox(200000, -300000, 500000, -6000, 500, 400), (1, 1, 1, 1), (2, 2),
     Box(bottom=3, left=7, top=-2, right=3))
])
def test_visible_tiles(tc_params, vg, etiles, stride, exp):
    """Test returned box of tiles to draw is correct given a visible world geometry and sampling."""
    tile_calc = TileCalculator(*tc_params)
    res = tile_calc.visible_tiles(vg, stride, etiles)
    assert res == exp


@pytest.mark.parametrize("cp,ip,cs,exp", [
    ([[10.0, 10.0], [20.0, 20.0]], [[10.0, 10.0], [20.0, 20.0]], (1, 1), (2.0, 2.0))
])
def test_calc_pixel_size(cp, ip, cs, exp):
    """Test calculated pixel size is correct given image data."""
    res = calc_pixel_size(np.array(cp), np.array(ip), cs)
    assert res == exp


@pytest.mark.parametrize("ic,iv,exp", [
    ([[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 1.0, 1.0]],
     [[1.0, 3.0, 7.0, 2.0], [3.0, 2.0, 8.0, 5.0]], (0, 1))
])
def test_get_reference_points(ic, iv, exp):
    """Test returned image reference point indexes are correct."""
    res = get_reference_points(np.array(ic), np.array(iv))
    assert res == exp


@pytest.mark.parametrize("ic,iv", [
    ([[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 1.0, 1.0]],
     [[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 1.0, 1.0]])
])
def test_get_reference_points_bad_points(ic, iv):
    """Test that error is thrown if given invalid mesh points."""
    with pytest.raises(ValueError):
        get_reference_points(np.array(ic), np.array(iv))


@pytest.mark.parametrize("cp,ip,num_p,mpp,exp", [
    (1.0, 500.0, 100, 3, (200, 500))
])
def test_calc_extent_component(cp, ip, num_p, mpp, exp):
    """Test bounding box extents are correct."""
    res = _calc_extent_component(cp, ip, num_p, mpp)
    assert res == exp


@pytest.mark.parametrize("v,n,x,exp", [
    (1.0, 2.0, 3.0, 2.0),
    (0.0, 0.0, 0.0, 0.0)
])
def test_clip(v, n, x, exp):
    """Test clipped value is correct."""
    res = clip(v, n, x)
    assert res == exp


@pytest.mark.parametrize("iebox,cp,ip,cs,dx,dy,exp", [
    (Box(100.0, 100.0, 200.0, 300.0), [1.0, 1.0], [500.0, 350.0], (100, 70), 3.0, 2.5,
     Box(bottom=175.0, left=200.0, top=200.0, right=300.0))
])
def test_calc_view_extents(iebox, cp, ip, cs, dx, dy, exp):
    """Test calculated viewing box for image is correct."""
    res = calc_view_extents(iebox, np.array(cp), np.array(ip), cs, dx, dy)
    assert res == exp


@pytest.mark.parametrize("iebox,cp,ip,cs,dx,dy", [
    (Box(0.0, 0.0, 0.0, 0.0), [1.0, 1.0], [500.0, 500.0], (100, 100), 3.0, 3.0)
])
def test_calc_view_extents_bad_box(iebox, cp, ip, cs, dx, dy):
    """Test that error is thrown given zero-sized box."""
    with pytest.raises(ValueError):
        calc_view_extents(iebox, np.array(cp), np.array(ip), cs, dx, dy)


@pytest.mark.parametrize("ims,ts,s,exp", [
    (Point(20, 20), Point(2, 2), Point(2, 2), (5.0, 5.0))
])
def test_max_tiles_available(ims, ts, s, exp):
    """Test the max number of tiles available is returned given image shape, tile shape, and stride."""
    res = max_tiles_available(ims[0], ims[1], ts[0], ts[1], s[0], s[1])
    assert res == exp


@pytest.mark.parametrize("tc_params,tiy,tix,s,exp", [
    (["test", (500, 500), Point(500000, -500000), Resolution(200, 200), (50, 50)],
     10, 10, (2, 2), (slice(600, 650, 1), slice(600, 650, 1))),
    (["test", (500, 500), Point(500000, -500000), Resolution(200, 200), (500, 500)],
     0, 0, (2, 2), (slice(0, 375, 1), slice(0, 375, 1)))
])
def test_calc_tile_slice(tc_params, tiy, tix, s, exp, monkeypatch):
    """Test appropriate slice is returned given image data."""
    tile_calc = TileCalculator(*tc_params)
    res = tile_calc.calc_tile_slice(tiy, tix, s)
    assert res == exp


@pytest.mark.parametrize("tc_params,tiy,tix,s,exp", [
    (["test", (500, 500), Point(500000, -500000), Resolution(200, 200), (50, 50)],
     -5, -5, (2, 2), (Resolution(-2.0, -2.0), Resolution(3.0, 3.0))),
    (["test", (500, 500), Point(500000, -500000), Resolution(200, 200), (50, 50)],
     1, 1, (2, 2), (Resolution(1.0, 1.0), Resolution(0.0, 0.0))),
    (["test", (500, 500), Point(500000, -500000), Resolution(200, 200), (50, 50)],
     5, 5, (2, 2), (Resolution(-2.0, -2.0), Resolution(0.0, 0.0))),
])
def test_calc_tile_fraction(tc_params, tiy, tix, s, exp):
    """Test calculated fractional components of the specified tile are correct."""
    tile_calc = TileCalculator(*tc_params)
    res = tile_calc.calc_tile_fraction(tiy, tix, s)
    assert res == exp


@pytest.mark.parametrize("tc_params,v,t,exp", [
    (["test", (500, 500), Point(500000, -500000), Resolution(200, 200), (50, 50)],
     ViewBox(dx=1, dy=1, bottom=1, top=1, right=1, left=1), None, Point(y=1, x=1)),
    (["test", (500, 500), Point(500000, -500000), Resolution(200, 200), (50, 50)],
     Resolution(dx=1, dy=1), None, Point(y=1, x=1)),
    (["test", (500, 500), Point(500000, -500000), Resolution(200, 200), (50, 50)],
     ViewBox(dx=1, dy=1, bottom=1, top=1, right=1, left=1), Resolution(100, 100), Point(y=1, x=1)),
])
def test_calc_stride(tc_params, v, t, exp):
    """Test calculated stride value is correct given world geometry and sampling."""
    tile_calc = TileCalculator(*tc_params)
    res = tile_calc.calc_stride(v, t)
    assert res == exp


@pytest.mark.parametrize("tc_params,ims,exp", [
    (["test", (500, 500), Point(500000, -500000), Resolution(200, 200), (50, 50)],
     None, (slice(0, 500, 10), slice(0, 500, 10))),
    (["test", (500, 500), Point(500000, -500000), Resolution(200, 200), (50, 50)],
     (100, 100), (slice(0, 100, 2), slice(0, 100, 2)))
])
def test_calc_overview_stride(tc_params, ims, exp):
    """Test calculated stride is correct given a valid image."""
    tile_calc = TileCalculator(*tc_params)
    res = tile_calc.calc_overview_stride(ims)
    assert res == exp


@pytest.mark.parametrize("tc_params,tiy,tix,sy,sx,fr,ofr,tl,exp", [
    (["test", (500, 500), Point(500000, -500000), Resolution(200, 200), (50, 50)],
     10, 10, 2, 2, Resolution(dy=2, dx=2), Resolution(dy=2, dx=2), 1,
     np.array([[-220000, 220000.],
               [-180000, 220000.],
               [-180000, 180000.],
               [-220000, 220000.],
               [-180000, 180000.],
               [-220000, 180000.]]))
])
def test_calc_vertex_coordinates(tc_params, tiy, tix, sy, sx, fr, ofr, tl, exp):
    """Test vertex coordinates for a given tile are correct."""
    tile_calc = TileCalculator(*tc_params)
    res = tile_calc.calc_vertex_coordinates(tiy, tix, sy, sx, fr, ofr, tl)
    assert np.array_equal(res, exp)


@pytest.mark.parametrize("tc_params,ti,fr,ofr,tl,exp", [
    (["test", (500, 500), Point(500000, -500000), Resolution(200, 200), (50, 50)],
     10, Resolution(dy=2, dx=2), Resolution(dy=2, dx=2), 1,
     np.array([[0.625, 0.],
               [0.75, 0.],
               [0.75, 1.],
               [0.625, 0.],
               [0.75, 1.],
               [0.625, 1.]]))
])
def test_calc_texture_coordinates(tc_params, ti, fr, ofr, tl, exp):
    """Test texture coordinates for a given tile are correct."""
    tile_calc = TileCalculator(*tc_params)
    res = tile_calc.calc_texture_coordinates(ti, fr, ofr, tl)
    assert np.array_equal(res, exp)
