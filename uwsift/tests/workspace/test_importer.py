#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for the importer functions and classes."""

import os
import yaml
from uwsift.workspace.importer import available_satpy_readers


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
    p = tmpdir.join('satpy_available_readers.yaml')
    monkeypatch.setattr('uwsift.workspace.importer.SATPY_READER_CACHE_FILE', str(p))
    readers = available_satpy_readers()
    assert isinstance(readers, list)
    assert len(readers) != 0
    assert isinstance(readers[0], str)


def test_available_satpy_readers_empty_cache(tmpdir, monkeypatch):
    """Test loading the satpy readers when the cache exists but is empty."""
    p = tmpdir.join('satpy_available_readers.yaml')
    with open(p, 'w') as cfile:
        yaml.dump({}, cfile)
    monkeypatch.setattr('uwsift.workspace.importer.SATPY_READER_CACHE_FILE', str(p))
    readers = available_satpy_readers()
    assert isinstance(readers, list)
    assert len(readers) != 0
    assert isinstance(readers[0], str)


def test_available_satpy_readers_known_cache(tmpdir, monkeypatch):
    """Test loading the satpy readers when the cache exists."""
    p = tmpdir.join('satpy_available_readers.yaml')
    with open(p, 'w') as cfile:
        yaml.dump({}, cfile)
    monkeypatch.setattr('uwsift.workspace.importer.SATPY_READER_CACHE_FILE', str(p))
    monkeypatch.setattr('uwsift.workspace.importer._SATPY_READERS', None)
    # create the cache
    _ = available_satpy_readers()
    mod_time1 = os.stat(p).st_mtime

    # load from the cache
    monkeypatch.setattr('uwsift.workspace.importer._SATPY_READERS', None)
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
