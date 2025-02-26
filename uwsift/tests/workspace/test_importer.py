#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for the importer functions and classes."""

import os
from datetime import datetime
from unittest.mock import patch

import dask.array as da
import numpy as np
import pytest
import xarray as xr
import yaml
from pyresample.geometry import AreaDefinition
from satpy import Scene
from satpy.tests.utils import make_dataid

from uwsift.common import Info, Instrument, Platform
from uwsift.model.area_definitions_manager import AreaDefinitionsManager
from uwsift.workspace.importer import SatpyImporter, available_satpy_readers


def test_available_satpy_readers_defaults():
    """Test default behavior of the satpy reader function."""
    readers = available_satpy_readers()
    assert isinstance(readers, list)
    assert len(readers) != 0
    assert isinstance(readers[0], str)  # names


def test_available_satpy_readers_as_dict():
    """Test asking for reader info as dictionaries."""
    readers = available_satpy_readers(as_dict=True)
    assert isinstance(readers, list)
    assert len(readers) != 0
    assert isinstance(readers[0], dict)


def test_available_satpy_readers_no_cache(tmpdir, monkeypatch):
    """Test loading the satpy readers when we know the cache is missing."""
    p = tmpdir.join("satpy_available_readers.yaml")
    monkeypatch.setattr("uwsift.workspace.importer.SATPY_READER_CACHE_FILE", str(p))
    readers = available_satpy_readers()
    assert isinstance(readers, list)
    assert len(readers) != 0
    assert isinstance(readers[0], str)


def test_available_satpy_readers_empty_cache(tmpdir, monkeypatch):
    """Test loading the satpy readers when the cache exists but is empty."""
    p = tmpdir.join("satpy_available_readers.yaml")
    with open(p, "w") as cfile:
        yaml.dump({}, cfile)
    monkeypatch.setattr("uwsift.workspace.importer.SATPY_READER_CACHE_FILE", str(p))
    readers = available_satpy_readers()
    assert isinstance(readers, list)
    assert len(readers) != 0
    assert isinstance(readers[0], str)


def test_available_satpy_readers_known_cache(tmpdir, monkeypatch):
    """Test loading the satpy readers when the cache exists."""

    # TODO: The next line of code skips the test.
    #  If this test is no longer to be skipped, this line must be removed.
    #  Adjustments may have to be made to make the test work correctly again.
    pytest.skip("Satpy available readers caching is deactivated at the moment.")

    p = tmpdir.join("satpy_available_readers.yaml")
    with open(p, "w") as cfile:
        yaml.dump({}, cfile)
    monkeypatch.setattr("uwsift.workspace.importer.SATPY_READER_CACHE_FILE", str(p))
    monkeypatch.setattr("uwsift.workspace.importer._SATPY_READERS", None)
    # create the cache
    _ = available_satpy_readers()
    mod_time1 = os.stat(p).st_mtime

    # load from the cache
    monkeypatch.setattr("uwsift.workspace.importer._SATPY_READERS", None)
    _ = available_satpy_readers()
    mod_time2 = os.stat(p).st_mtime
    assert mod_time1 == mod_time2, "Cache was not reused"

    # force recreation of the cache
    readers = available_satpy_readers(force_cache_refresh=True)
    mod_time3 = os.stat(p).st_mtime
    assert mod_time3 != mod_time1, "Cache was reused when it should have been recreated"
    assert isinstance(readers, list)
    assert len(readers) != 0
    assert isinstance(readers[0], str)


def _get_data_array_generator(data_arrs):
    """Help mimic what a real Satpy Scene would do."""
    yield from data_arrs


def _get_fake_g16_abi_c01_scene(mocker):
    attrs = {
        "name": "C01",
        "wavelength": (1.0, 2.0, 3.0),
        "area": AreaDefinition(
            "test",
            "test",
            "test",
            {
                "proj": "geos",
                "sweep": "x",
                "lon_0": -75,
                "h": 35786023,
                "ellps": "GRS80",
                "units": "m",
            },
            5,
            5,
            (-5434894.885056, -5434894.885056, 5434894.885056, 5434894.885056),
        ),
        "start_time": datetime(2018, 9, 10, 17, 0, 31, 100000),
        "end_time": datetime(2018, 9, 10, 17, 11, 7, 800000),
        "standard_name": "toa_bidirectional_reflectance",
        "sensor": "abi",
        "platform_name": "GOES-16",
        "platform_shortname": "G16",
    }
    data_arr = xr.DataArray(da.from_array(np.empty((5, 5), dtype=np.float64), chunks="auto"), attrs=attrs)
    scn = Scene()
    scn["C01"] = data_arr
    scn.load = mocker.MagicMock()  # don't do anything on load
    return scn


def _get_fake_g18_abi_c01_scene(mocker):
    scn = _get_fake_g16_abi_c01_scene(mocker)
    scn["C01"].attrs["platform_name"] = "GOES-18"
    scn["C01"].attrs["platform_shortname"] = "G18"
    return scn


def _get_fake_g18_abi_c01_scene_no_pname(mocker):
    # old versions of satpy didn't assign a proper platform_name
    scn = _get_fake_g18_abi_c01_scene(mocker)
    scn["C01"].attrs["platform_name"] = None
    return scn


@pytest.mark.parametrize(
    ["get_scene", "exp_platform"],
    [
        (_get_fake_g16_abi_c01_scene, Platform.GOES_16),
        (_get_fake_g18_abi_c01_scene, Platform.GOES_18),
        (_get_fake_g18_abi_c01_scene_no_pname, Platform.GOES_18),
    ],
)
def test_satpy_importer_basic(get_scene, exp_platform, tmpdir, monkeypatch, mocker):
    """Basic import test using Satpy."""
    db_sess = mocker.MagicMock()
    scn = get_scene(mocker)
    imp = SatpyImporter(
        ["/test/file.nc"], tmpdir, db_sess, scene=scn, reader="abi_l1b", dataset_ids=[make_dataid(name="C01")]
    )
    imp.merge_resources()
    assert imp.num_products == 1
    products = list(imp.merge_products())
    assert len(products) == 1
    assert products[0].info[Info.CENTRAL_WAVELENGTH] == 2.0
    assert products[0].info[Info.STANDARD_NAME] == "toa_bidirectional_reflectance"
    assert products[0].info[Info.PLATFORM] == exp_platform
    assert products[0].info[Info.INSTRUMENT] == Instrument.ABI


# Test if the resampling method is used with a custom resampling and does calculations
def test_satpy_importer_resampling(tmpdir, monkeypatch, mocker):
    """Basic import test using Satpy."""
    db_sess = mocker.MagicMock()
    scn = _get_fake_g16_abi_c01_scene(mocker)
    reader_name = "fci_l1c_nc"
    imp = SatpyImporter(
        ["/test/file.nc"], tmpdir, db_sess, scene=scn, reader=reader_name, dataset_ids=[make_dataid(name="vis_06")]
    )
    resampling_info = {
        "resampler": "nearest",
        "area_id": "mtg_fci_fdss_1km",
        "projection": "+ellps=WGS84 +h=35786400 +lon_0=0 +no_defs +proj=geos +type=crs +units=m +x_0=0 +y_0=0",
        "radius_of_influence": 5000,
        # This is not yet an arbitrary shape, expect that values going into resample do not deviate from initial
        # when custom is set to "True"
        "shape": (11136, 11136),
        "custom": False,
    }
    imp.resampling_info = resampling_info

    # The result of Scene.finest_area needs to be corrected for our needs
    def finest_area_sideeffect(*args, **kwargs):
        orig_fa = orig_finest_area(*args, **kwargs)
        orig_fa.area_id = resampling_info["area_id"]
        return orig_fa

    area_def_resampled = None

    # Catch the Scene.resample call and harvest the arguments provided by the caller
    def resample_sideeffect(*args, **kwargs):
        nonlocal area_def_resampled
        area_def_resampled = args[0]
        return scn  # just return back the original scene

    # Setup patch for Scene.finest_area
    orig_finest_area = scn.finest_area
    finest_area_patcher = patch.object(Scene, "finest_area")
    mock_finest_area = finest_area_patcher.start()
    mock_finest_area.side_effect = finest_area_sideeffect

    # Setup patch for Scene.resample
    resample_patcher = patch.object(Scene, "resample")
    mock_resample = resample_patcher.start()
    mock_resample.side_effect = resample_sideeffect

    # Call _preprocess_products_with_resampling with a shape not being "custom"
    imp._preprocess_products_with_resampling()
    assert not mock_resample.called  # Scene.resample should not have been called

    # Grab the original area definition values before resampling
    area_def = AreaDefinitionsManager.area_def_by_id(resampling_info["area_id"])
    prev_ad_p_size_x = area_def.pixel_size_x
    prev_ad_p_size_y = area_def.pixel_size_y
    prev_ad_p_upper_left = area_def.pixel_upper_left
    prev_ad_p_offset_x = area_def.pixel_offset_x
    prev_ad_p_offset_y = area_def.pixel_offset_y
    # Now Scene.resample must be called but the area definition values provided must not have changes
    resampling_info["custom"] = True
    imp._preprocess_products_with_resampling()
    # Check that the are definition values have NOT been recalculated as the shape itself is still not custom
    assert prev_ad_p_size_x == area_def_resampled.pixel_size_x
    assert prev_ad_p_size_y == area_def_resampled.pixel_size_y
    assert prev_ad_p_upper_left[0] == area_def_resampled.pixel_upper_left[0]
    assert prev_ad_p_upper_left[1] == area_def_resampled.pixel_upper_left[1]
    assert prev_ad_p_offset_x == area_def_resampled.pixel_offset_x
    assert prev_ad_p_offset_y == area_def_resampled.pixel_offset_y

    # Some arbitrary shape, expect that resample is now being called with recalculated definition values
    resampling_info["shape"] = (5000, 5000)
    imp._preprocess_products_with_resampling()
    # Check that the are definition values have been recalculated
    assert prev_ad_p_size_x != area_def_resampled.pixel_size_x
    assert prev_ad_p_size_y != area_def_resampled.pixel_size_y
    assert prev_ad_p_upper_left[0] != area_def_resampled.pixel_upper_left[0]
    assert prev_ad_p_upper_left[1] != area_def_resampled.pixel_upper_left[1]
    assert prev_ad_p_offset_x != area_def_resampled.pixel_offset_x
    assert prev_ad_p_offset_y != area_def_resampled.pixel_offset_y

    # Clean up
    finest_area_patcher.stop()
    resample_patcher.stop()
