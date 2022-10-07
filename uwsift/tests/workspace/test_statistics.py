#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for the statistics functions and classes.

TODO:
 - The tests are implemented with a lot of code duplication. This could be
   cleaned up.
 - Several more details could be checked to make sure the dictionary returned by
   dataset_statistical_analysis() comply to the agreed interface.
"""

import dask.array as da
import numpy as np
import xarray as xr

from uwsift.workspace.statistics import dataset_statistical_analysis


def test_categorial_data_statistics_dict():
    """Test statistics dictionary for categorial data"""
    arr = np.array(
        [
            [0, 0, 0, 0, 0],
            [1, 1, 1, 2, 2],
        ]
    )
    attrs = {
        "flag_values": [0, 1, 2],
    }
    data_arr = xr.DataArray(da.from_array(arr, chunks="auto"), attrs=attrs)

    stats_dict = dataset_statistical_analysis(data_arr)
    assert type(stats_dict) is dict
    assert stats_dict["header"] == ["value", "meaning", "count / -", "fraction / %"]
    assert type(stats_dict["stats"]) is list
    assert stats_dict["stats"] == [
        [0, "n/a", 5, 50.0],
        [1, "n/a", 3, 30.0],
        [2, "n/a", 2, 20.0],
    ]
    assert type(stats_dict["stats"][0][2]) is int
    assert isinstance(stats_dict["stats"][0][3], float)


def test_bit_encoded_data_statistics_dict():
    """Test code path for bit-encoded data"""
    attrs = {
        "flag_masks": [1, 2, 12],
    }
    data_arr = xr.DataArray(da.from_array(np.empty((5, 5), dtype=np.float64), chunks="auto"), attrs=attrs)

    stats_dict = dataset_statistical_analysis(data_arr)
    assert stats_dict == {}


def test_general_algebraics_data_statistics_dict():
    """Test statistics dictionary algebraics data not computed as difference."""
    attrs = {
        "algebraic": "a*b",
    }
    data_arr = xr.DataArray(da.from_array(np.empty((5, 5), dtype=np.float64), chunks="auto"), attrs=attrs)

    stats_dict = dataset_statistical_analysis(data_arr)
    assert isinstance(stats_dict, dict)
    assert "header" not in stats_dict
    assert type(stats_dict["stats"]) is dict
    assert "mad" not in stats_dict["stats"]
    stats_values_iter = iter(stats_dict["stats"].values())
    assert type(next(stats_values_iter)[0]) is int
    assert isinstance(next(stats_values_iter)[0], float)


def test_difference_algebraics_data_statistics_dict():
    """Test statistics dictionary for algebraics data computed as difference."""
    attrs = {
        "algebraic": "x-y",
    }
    data_arr = xr.DataArray(da.from_array(np.empty((5, 5), dtype=np.float64), chunks="auto"), attrs=attrs)

    stats_dict = dataset_statistical_analysis(data_arr)
    assert type(stats_dict) is dict
    assert "header" not in stats_dict
    assert type(stats_dict["stats"]) is dict
    assert "mad" in stats_dict["stats"]
    stats_values_iter = iter(stats_dict["stats"].values())
    assert type(next(stats_values_iter)[0]) is int
    assert isinstance(next(stats_values_iter)[0], float)


def test_implicit_categorial_data_statistics_dict():
    """Test statistics dictionary for categorial data, which has no explicitly defined flags."""
    arr = np.array(
        [
            [0, 0, 0, 0, 0],
            [1, 1, 1, 2, 2],
        ]
    )
    data_arr = xr.DataArray(da.from_array(arr, chunks="auto"))
    assert data_arr.dtype.kind == "i"

    stats_dict = dataset_statistical_analysis(data_arr)
    assert type(stats_dict) is dict
    assert stats_dict["header"] == ["value", "meaning", "count / -", "fraction / %"]
    assert type(stats_dict["stats"]) is list
    assert stats_dict["stats"] == [
        [0, "n/a", 5, 50.0],
        [1, "n/a", 3, 30.0],
        [2, "n/a", 2, 20.0],
    ]
    assert type(stats_dict["stats"][0][2]) is int
    assert isinstance(stats_dict["stats"][0][3], float)


def test_continuous_data_statistics_dict():
    """Test statistics dictionary for continuous data"""
    data_arr = xr.DataArray(da.from_array(np.empty((5, 5), dtype=np.float64), chunks="auto"))

    stats_dict = dataset_statistical_analysis(data_arr)
    assert type(stats_dict) is dict
    assert "header" not in stats_dict
    assert type(stats_dict["stats"]) is dict
    stats_values_iter = iter(stats_dict["stats"].values())
    assert type(next(stats_values_iter)[0]) is int
    assert isinstance(next(stats_values_iter)[0], float)
