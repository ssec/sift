#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for algebraic layer operations."""

from datetime import datetime

import numpy as np

from uwsift.common import Info, Instrument, Platform


def _get_data_array_generator(data_arrs):
    """Help mimic what a real Satpy Scene would do."""
    yield from data_arrs


def test_satpy_importer_basic(tmpdir, monkeypatch, mocker):
    """Basic import test using Satpy."""
    from uuid import uuid1 as uuidgen

    from uwsift.workspace import CachingWorkspace

    ws = CachingWorkspace(str(tmpdir))
    c01_attrs = {
        Info.SHORT_NAME: "C01",
        Info.DATASET_NAME: "C01",
        Info.CENTRAL_WAVELENGTH: 2.0,
        Info.UUID: uuidgen(),
    }
    c03_attrs = {
        Info.SHORT_NAME: "C03",
        Info.DATASET_NAME: "C03",
        Info.CENTRAL_WAVELENGTH: 4.0,
        Info.UUID: uuidgen(),
    }
    for ds in [c01_attrs, c03_attrs]:
        ds[Info.ORIGIN_X] = -5434894.885056
        ds[Info.ORIGIN_Y] = 5434894.885056
        ds[Info.CELL_HEIGHT] = 1000.0
        ds[Info.CELL_WIDTH] = 1000.0
        ds[Info.STANDARD_NAME] = "toa_bidirectional_reflectance"
        ds[Info.VALID_RANGE] = (0, 120)
        ds[Info.SCHED_TIME] = ds[Info.OBS_TIME] = datetime(2018, 9, 10, 17, 0, 31, 100000)
        ds[Info.SHAPE] = (2, 2)
        ds[Info.PROJ] = "+proj=merc"
        ds[Info.FAMILY] = "family"
        ds[Info.CATEGORY] = "category"
        ds[Info.SERIAL] = "serial"
        ds[Info.PLATFORM] = Platform.GOES_16
        ds[Info.INSTRUMENT] = Instrument.ABI
        ds[Info.GRID_ORIGIN] = "SE"
        ds[Info.GRID_FIRST_INDEX_X] = 1
        ds[Info.GRID_FIRST_INDEX_Y] = 1

    ops = "z = x - y"
    ns = {"y": c03_attrs[Info.UUID], "x": c01_attrs[Info.UUID]}

    def get_metadata(u):
        return c01_attrs if u == c01_attrs[Info.UUID] else c03_attrs

    def get_content(u):
        return np.ones((2, 2)) if u == c01_attrs[Info.UUID] else np.zeros((2, 2))

    monkeypatch.setattr(ws, "get_metadata", get_metadata)
    monkeypatch.setattr(ws, "get_content", get_content)
    uuid, info, data = ws.create_algebraic_composite(ops, ns, info={Info.SHORT_NAME: "new", Info.DATASET_NAME: "new"})

    np.testing.assert_equal(data, 1)
    assert info.get(Info.STANDARD_NAME) == "unknown"
