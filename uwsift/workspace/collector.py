#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PURPOSE
Collector is a zookeeper of products, which populates and revises the workspace metadatabase
Collector uses Hunters to find individual formats/conventions/products
Products live in Resources (typically files)
Collector skims files without reading data
Collector populates the metadatabase with information about available products
More than one Product may be in a Resource

Collector also knows which Importer can bring Content from the Resource into the Workspace

REFERENCES

REQUIRES

:author: R.K.Garcia <rkgarcia@wisc.edu>
:copyright: 2017 by University of Wisconsin Regents, see AUTHORS for more details
:license: GPLv3, see LICENSE for more details
"""
import logging
import os
import sys
import unittest
from datetime import datetime
from typing import List, Iterable, Mapping

from PyQt5.QtCore import QObject

from uwsift import config
from satpy.readers import group_files
from uwsift.queue import TASK_DOING, TASK_PROGRESS
from .workspace import Workspace
from .importer import available_satpy_readers
from ..common import Info

LOG = logging.getLogger(__name__)


class _workspace_test_proxy(object):
    def __init__(self):
        self.cwd = '/tmp' if os.path.isdir("/tmp") else os.getcwd()

    def collect_product_metadata_for_paths(self, paths):
        LOG.debug("import metadata for files: {}".format(repr(paths)))
        for path in paths:
            yield 1, {Info.PATHNAME: path}

    class _emitsy(object):
        def emit(self, stuff):
            print("==> " + repr(stuff))

    didUpdateProductsMetadata = _emitsy()


class ResourceSearchPathCollector(QObject):
    """Given a set of search paths,
    awaken for new files available within the directories,
    update the metadatabase for new resources,
    and mark for purge any files no longer available.
    """
    _ws: Workspace = None
    _paths: List[str] = None
    _dir_mtimes: Mapping[str, datetime] = None
    _timestamp_path: str = None  # path which tracks the last time we skimmed the paths
    _is_posix: bool = None
    _scheduled_dirs: List[str] = None
    _scheduled_files: List[str] = None

    @property
    def paths(self):
        return list(self._paths)

    @paths.setter
    def paths(self, new_paths):
        nu = set(new_paths)
        ol = set(self._paths)
        removed = ol - nu
        added = nu - ol
        self._scheduled_dirs = []
        self._scheduled_files = []
        self._paths = list(new_paths)
        self._flush_dirs(removed)
        self._schedule_walk_dirs(added)
        LOG.debug('old search directories removed: {}'.format(':'.join(sorted(removed))))
        LOG.debug('new search directories added: {}'.format(':'.join(sorted(added))))

    def _flush_dirs(self, dirs: Iterable[str]):
        pass

    def _schedule_walk_dirs(self, dirs: Iterable[str]):
        self._scheduled_dirs += list(dirs)

    @property
    def has_pending_files(self):
        return len(self._scheduled_files)

    def __bool__(self):
        return len(self._paths) > 0

    def _skim(self, last_checked: int = 0, dirs: Iterable[str] = None):
        """skim directories for new mtimes
        """
        skipped_dirs = 0
        for rawpath in (dirs or self._paths):
            path = os.path.abspath(rawpath)
            if not os.path.isdir(path):
                LOG.warning("{} is not a directory".format(path))
                continue
            for dirpath, dirnames, filenames in os.walk(path):
                if self._is_posix and (os.stat(dirpath).st_mtime < last_checked):
                    skipped_dirs += 1
                    continue
                for filename in filenames:
                    if filename.startswith('.'):
                        continue  # dammit Apple, ._*.nc files ...
                    filepath = os.path.join(dirpath, filename)
                    if os.path.isfile(filepath) and (os.stat(filepath).st_mtime >= last_checked):
                        yield filepath
        LOG.debug("skipped files in {} dirs due to POSIX directory mtime".format(skipped_dirs))

    def _touch(self):
        mtime = 0
        if os.path.isfile(self._timestamp_path):
            mtime = os.stat(self._timestamp_path).st_mtime
        else:
            with open(self._timestamp_path, 'wb') as fp:
                fp.close()
        os.utime(self._timestamp_path)
        return mtime

    def __init__(self, ws: [Workspace, _workspace_test_proxy]):
        super(ResourceSearchPathCollector, self).__init__()
        self._ws = ws
        self._paths = []
        self._dir_mtimes = {}
        self._scheduled_files = []
        self._timestamp_path = os.path.join(ws.cwd, '.last_collection_check')
        self._is_posix = sys.platform in {'linux', 'darwin'}
        self.satpy_readers = config.get('data_reading.readers')
        if not self.satpy_readers:
            self.satpy_readers = available_satpy_readers()

    def look_for_new_files(self):
        if len(self._scheduled_dirs):
            new_dirs, self._scheduled_dirs = self._scheduled_dirs, []
            LOG.debug('giving special attention to new search paths {}'.format(':'.join(new_dirs)))
            new_files = list(self._skim(0, new_dirs))
            LOG.debug('found {} files in new search paths'.format(len(new_files)))
            self._scheduled_files += new_files
        when = self._touch()
        new_files = list(self._skim(when))
        if new_files:
            LOG.info('found {} additional files to skim metadata for, for a total of {}'.format(len(new_files), len(
                self._scheduled_files)))
            self._scheduled_files += new_files

    def bgnd_look_for_new_files(self):
        LOG.debug("searching for files in search path {}".format(':'.join(self._paths)))
        yield {TASK_DOING: 'skimming', TASK_PROGRESS: 0.5}
        self.look_for_new_files()
        yield {TASK_DOING: 'skimming', TASK_PROGRESS: 1.0}

    def bgnd_merge_new_file_metadata_into_mdb(self):
        todo, self._scheduled_files = self._scheduled_files, []
        ntodo = len(todo)
        LOG.debug('collecting metadata from {} potential new files'.format(ntodo))
        yield {TASK_DOING: 'collecting metadata 0/{}'.format(ntodo), TASK_PROGRESS: 0.0}
        changed_uuids = set()
        readers_and_files = group_files(todo, reader='abi_l1b')
        num_seen = 0
        for reader_and_files in readers_and_files:
            for reader_name, filenames in reader_and_files.items():
                product_infos = self._ws.collect_product_metadata_for_paths(filenames, reader=reader_name)
                for num_prods, product_info in product_infos:
                    changed_uuids.add(product_info[Info.UUID])
                    num_seen += len(filenames)
                    status = {TASK_DOING: 'collecting metadata {}/{}'.format(num_seen, ntodo),
                              TASK_PROGRESS: float(num_seen) / ntodo + 1}
                    yield status
        yield {TASK_DOING: 'collecting metadata done', TASK_PROGRESS: 1.0}
        if changed_uuids:
            LOG.debug('{} changed UUIDs, signaling product updates'.format(len(changed_uuids)))
            # FUTURE: decide whether signals for metadatabase should belong to metadatabase
            self._ws.didUpdateProductsMetadata.emit(changed_uuids)


def _debug(type, value, tb):
    "enable with sys.excepthook = debug"
    if not sys.stdin.isatty():
        sys.__excepthook__(type, value, tb)
    else:
        import traceback
        import pdb
        traceback.print_exception(type, value, tb)
        # …then start the debugger in post-mortem mode.
        pdb.post_mortem(tb)  # more “modern”


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="PURPOSE",
        epilog="",
        fromfile_prefix_chars='@')
    parser.add_argument('-v', '--verbose', dest='verbosity', action="count", default=0,
                        help='each occurrence increases verbosity 1 level through ERROR-WARNING-Info-DEBUG')
    parser.add_argument('-d', '--debug', dest='debug', action='store_true',
                        help="enable interactive PDB debugger on exception")
    parser.add_argument('inputs', nargs='*',
                        help="input files to process")
    args = parser.parse_args()

    levels = [logging.ERROR, logging.WARN, logging.INFO, logging.DEBUG]
    logging.basicConfig(level=levels[min(3, args.verbosity)], datefmt='%Y-%m-%dT%H:%M:%S',
                        format='%(levelname)s %(asctime)s %(module)s:%(funcName)s:L%(lineno)d %(message)s')

    if args.debug:
        sys.excepthook = _debug

    if not args.inputs:
        unittest.main()
        return 0

    ws = _workspace_test_proxy()
    collector = ResourceSearchPathCollector(ws)
    collector.paths = list(args.inputs)

    from time import sleep
    for i in range(3):
        if i > 0:
            sleep(5)
        LOG.info("poll #{}".format(i + 1))
        collector.look_for_new_files()
        if collector.has_pending_files:
            for progress in collector.bgnd_merge_new_file_metadata_into_mdb():
                LOG.debug(repr(progress))

    return 0


if __name__ == '__main__':
    sys.exit(main())
