import pytest
import numpy as np
from uwsift.view.tile_calculator import (TileCalculator,
                                         calc_pixel_size,
                                         get_reference_points,
                                         _calc_extent_component,
                                         clip,
                                         calc_view_extents,
                                         max_tiles_available)
from uwsift.common import Point, Box, ViewBox, Resolution


@pytest.mark.parametrize("tc_params,vg,etiles,stride,tiles,exp", [
    (["test", (500, 500), Point(500000, -500000), Resolution(200, 200), (50, 50)],
     ViewBox(200000, -300000, 500000, -6000, 500, 400), (1, 1, 1, 1), (2, 2), (5.0, 5.0),
     Box(bottom=3, left=7, top=-2, right=3))
])
def test_visible_tiles(tc_params, vg, etiles, stride, tiles, exp):
    tile_calc = TileCalculator(*tc_params)
    res = tile_calc.visible_tiles(vg, stride, etiles)
    assert res == exp


@pytest.mark.parametrize("cp,ip,cs,exp", [
    ([[10.0, 10.0], [20.0, 20.0]], [[10.0, 10.0], [20.0, 20.0]], (1, 1), (2.0, 2.0))
])
def test_calc_pixel_size(cp, ip, cs, exp):
    res = calc_pixel_size(np.array(cp), np.array(ip), cs)
    assert res == exp


@pytest.mark.parametrize("ic,iv,exp", [
    ([[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 1.0, 1.0]],
     [[1.0, 3.0, 7.0, 2.0], [3.0, 2.0, 8.0, 5.0]], (0, 1))
])
def test_get_reference_points(ic, iv, exp):
    res = get_reference_points(np.array(ic), np.array(iv))
    assert res == exp


@pytest.mark.parametrize("ic,iv", [
    ([[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 1.0, 1.0]],
     [[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 1.0, 1.0]])
])
def test_get_reference_points_bad_args(ic, iv):
    with pytest.raises(ValueError):
        get_reference_points(np.array(ic), np.array(iv))
        assert False


@pytest.mark.parametrize("cp,ip,num_p,mpp,exp", [
    (1.0, 500.0, 100, 3, (200, 500))
])
def test_calc_extent_component(cp, ip, num_p, mpp, exp):
    res = _calc_extent_component(cp, ip, num_p, mpp)
    assert res == exp


@pytest.mark.parametrize("v,n,x,exp", [
    (1.0, 2.0, 3.0, 2.0),
    (0.0, 0.0, 0.0, 0.0)
])
def test_clip(v, n, x, exp):
    res = clip(v, n, x)
    assert res == exp


@pytest.mark.parametrize("iebox,cp,ip,cs,dx,dy,exp", [
    (Box(100.0, 100.0, 200.0, 300.0), [1.0, 1.0], [500.0, 350.0], (100, 70), 3.0, 2.5,
     Box(bottom=175.0, left=200.0, top=200.0, right=300.0))
])
def test_calc_view_extents(iebox, cp, ip, cs, dx, dy, exp):
    res = calc_view_extents(iebox, np.array(cp), np.array(ip), cs, dx, dy)
    assert res == exp


@pytest.mark.parametrize("iebox,cp,ip,cs,dx,dy", [
    (Box(0.0, 0.0, 0.0, 0.0), [1.0, 1.0], [500.0, 500.0], (100, 100), 3.0, 3.0)
])
def test_calc_view_extents_bad_args(iebox, cp, ip, cs, dx, dy):
    with pytest.raises(ValueError):
        calc_view_extents(iebox, np.array(cp), np.array(ip), cs, dx, dy)
        assert False


@pytest.mark.parametrize("ims,ts,s,exp", [
    (Point(20, 20), Point(2, 2), Point(2, 2), (5.0, 5.0))
])
def test_max_tiles_available(ims, ts, s, exp):
    res = max_tiles_available(ims[0], ims[1], ts[0], ts[1], s[0], s[1])
    assert res == exp


@pytest.mark.parametrize("tc_params,tiy,tix,s,exp", [
    (["test", (500, 500), Point(500000, -500000), Resolution(200, 200), (50, 50)],
     10, 10, (2, 2), (slice(600, 650, 1), slice(600, 650, 1))),
    (["test", (500, 500), Point(500000, -500000), Resolution(200, 200), (500, 500)],
     0.5, 0.5, (2, 2), (slice(0, 375, 1), slice(0, 375, 1)))
])
def test_calc_tile_slice(tc_params, tiy, tix, s, exp):
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
    tile_calc = TileCalculator(*tc_params)
    res = tile_calc.calc_stride(v, t)
    assert res == exp


@pytest.mark.parametrize("tc_params,ims,exp", [
    (["test", (500, 500), Point(500000, -500000), Resolution(200, 200), (50, 50)],
     None, (slice(0, 500, 10), slice(0, 500, 10))),
    (["test", (500, 500), Point(500000, -500000), Resolution(200, 200), (50, 50)],
     (100, 100), (slice(0, 100, 2), slice(0, 100, 2)))
])
def test_calc_overview_strde(tc_params, ims, exp):
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
    tile_calc = TileCalculator(*tc_params)
    res = tile_calc.calc_texture_coordinates(ti, fr, ofr, tl)
    assert np.array_equal(res, exp)
