#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Wrappers around Satpy to handle changes between versions."""

try:
    from satpy import DataID

    def get_id_value(id_obj, key, default=None):
        return id_obj.get(key, default)

    def get_id_items(id_obj):
        return id_obj.items()

    def id_from_attrs(attrs):
        return attrs["_satpy_id"]

except ImportError:
    import warnings

    warnings.warn(
        "Satpy <0.23.0 will not be supported in future versions. " "Please update your version of Satpy.",
        DeprecationWarning,
        stacklevel=2,
    )
    from satpy import DatasetID as DataID

    def get_id_value(id_obj, key, default=None):
        return getattr(id_obj, key, default)

    def get_id_items(id_obj):
        return id_obj._asdict().items()

    def id_from_attrs(attrs):
        return DataID.from_dict(attrs)
