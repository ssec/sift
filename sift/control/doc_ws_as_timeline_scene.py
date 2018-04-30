#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Implement QGraphicsScene for Document and Workspace, as a delegate to TimelineView

This module is the boundary where Metadatabase/Workspace Product/Content information is translated to Track/Frame.


:author: R.K.Garcia <rkgarcia@wisc.edu>
:copyright: 2017-2018 by University of Wisconsin Regents, see AUTHORS for more details
:license: GPLv3, see LICENSE for more details
"""
import os, sys
import logging, unittest
from uuid import UUID
from typing import Tuple, Optional, Mapping, List, Any, Callable, Set
from PyQt4.QtGui import QMenu

from sift.view.TimelineItems import QTrackItem, QFrameItem
from sift.view.TimelineScene import QFramesInTracksScene
from sift.workspace import Workspace
from sift.model.document import Document, DocumentAsTrackStack
from sift.workspace.metadatabase import Metadatabase


LOG = logging.getLogger(__name__)

def mdb_products_as_tracks(mdb: Metadatabase) -> dict:
    """Yield dictionary
    """
    with mdb as S:




class SiftDocumentAsFramesInTracks(QFramesInTracksScene):
    """ represent SIFT Workspace and Document as frames in tracks
    preferably, we use doc.as_track_stack high level interface to query doc+mdb+ws
    """
    _doc: Document = None
    _ws: Workspace = None
    _mdb: Metadatabase = None

    @property
    def _dats(self):
        """Document as track stack
        """
        return self._doc.as_track_stack

    def __init__(self, doc: Document, mdb: Metadatabase, ws: Workspace, *args, **kwargs):
        """
        Args:
            ws (Workspace): owns cached and computed data
            mdb (Metadatabase): owns definitive metadata
            doc (Document): owns user selections and constructions
        """
        super(SiftDocumentAsFramesInTracks, self).__init__(*args, **kwargs)
        self._ws, self._mdb, self._doc = ws, mdb, doc
        self._connect_signals()

    def _connect_signals(self):
        """Connect document, workspace, signals in order to invalidate and update scene representation
        """
        pass

    def get(self, uuid: UUID) -> [QTrackItem, QFrameItem, None]:
        z = self._track_items.get(uuid, None)
        if z is not None:
            return z
        z = self._frame_items.get(uuid, None)
        return z

    def may_rearrange_track_z_order(self, track_uuid_list: List[UUID]) -> Optional[Callable[[bool], None]]:
        """Determine whether tracks can be rearranged and provide a commit/abort function if so
        Optionally: reflect any such changes on other parts of the application

        Args:
            track_uuid_list: new track UUID arrangement in top to bottom order

        Returns: None if rearrange is not permitted,
        else a callable that can be used to commit the change in the document
        callable(True) commits change; callable(False) aborts change
        Only one callable is valid at a time
        """
        LOG.warning("using base class may_rearrange_track_z_order which does nothing")
        return lambda commit: None

    def tracks_in_same_family(self, track: UUID) -> Set[UUID]:
        """inform the view on which tracks are closely related to the given track
        typically this is used to stylistically highlight related tracks during a drag operation
        """
        LOG.warning("using base class tracks_in_same_family which does nothing")
        return set()

    def may_reassign_color_map(self, from_track: UUID, to_track: UUID) -> Optional[Callable[[bool], None]]:
        """User is dragging a color map around, determine if drop is permitted and provide a commit/abort function if so
        """
        LOG.warning("using base class may_reassign_color_map which does nothing")
        return lambda b: None

    def menu_for_track(self, track_uuid: UUID, frame_uuid: UUID = None) -> Optional[QMenu]:
        """Generate QMenu to use as context menu for a given track, optionally with frame if mouse was over that frame"""
        LOG.warning("using base class menu_for_track which does nothing")

        return None

    def update(self, changed_track_uuids: [Set, None] = None, changed_frame_uuids: [Set, None] = None) -> int:
        """Populate or update scene, returning number of items changed in scene
        Does not add new items for tracks and frames already present
        Parameters serve only as hints
        """
        LOG.warning("using base class update which does nothing")
        return 0



# def _debug(type, value, tb):
#     "enable with sys.excepthook = debug"
#     if not sys.stdin.isatty():
#         sys.__excepthook__(type, value, tb)
#     else:
#         import traceback, pdb
#         traceback.print_exception(type, value, tb)
#         # …then start the debugger in post-mortem mode.
#         pdb.post_mortem(tb)  # more “modern”
#
#
# def main():
#     import argparse
#     parser = argparse.ArgumentParser(
#         description="PURPOSE",
#         epilog="",
#         fromfile_prefix_chars='@')
#     parser.add_argument('-v', '--verbose', dest='verbosity', action="count", default=0,
#                         help='each occurrence increases verbosity 1 level through ERROR-WARNING-INFO-DEBUG')
#     parser.add_argument('-d', '--debug', dest='debug', action='store_true',
#                         help="enable interactive PDB debugger on exception")
#     parser.add_argument('inputs', nargs='*',
#                         help="input files to process")
#     args = parser.parse_args()
#
#     levels = [logging.ERROR, logging.WARN, logging.INFO, logging.DEBUG]
#     logging.basicConfig(level=levels[min(3, args.verbosity)])
#
#     if args.debug:
#         sys.excepthook = _debug
#
#     if not args.inputs:
#         unittest.main()
#         return 0
#
#     for pn in args.inputs:
#         pass
#
#     return 0


if __name__ == '__main__':
    sys.exit(main())
