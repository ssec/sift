#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Implement QGraphicsScene for Document and Workspace, as a delegate to TimelineView

This QGraphicsScene represents document.as_track_stack, passing operations back to the document

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
from sift.model.document import Document, DocumentAsTrackStack, FrameInfo, TrackInfo
from sift.workspace.metadatabase import Metadatabase


LOG = logging.getLogger(__name__)


class SiftDocumentAsFramesInTracks(QFramesInTracksScene):
    """ represent SIFT Workspace and Document as frames in tracks
    preferably, we use doc.as_track_stack high level interface to query doc+mdb+ws
    """
    _doc: DocumentAsTrackStack = None

    def __init__(self, doc: Document, *args, **kwargs):
        """
        Args:
            ws (Workspace): owns cached and computed data
            mdb (Metadatabase): owns definitive metadata
            doc (Document): owns user selections and constructions
        """
        super(SiftDocumentAsFramesInTracks, self).__init__(*args, **kwargs)
        self._doc = doc.as_track_stack  # we should be limiting our interaction to this context
        self._connect_signals(doc)  # but the main doc is still the signaling hub

    def _sync_track(self, qti: QTrackItem, z:int, trk: TrackInfo):
        qti.z, old_z = z, qti.z
        qti.state = trk.state

    def _create_track(self, z: int, trk: TrackInfo) -> QTrackItem:
        qti = QTrackItem(self, self.coords, trk.track, z, trk.primary, trk.secondary)
        return qti

    def _create_frame(self, qti: QTrackItem, frm: FrameInfo) -> QFrameItem:
        qfi = QFrameItem(qti, self.coords, frm.uuid, frm.when.s, frm.when.d, frm.state, frm.primary, frm.secondary)
        return qfi

    def _sync_frame(self, qfi: QFrameItem, frm: FrameInfo):
        qfi.state = frm.state

    def _sync_tracks_frames(self):
        """populate QTrackItems and QFrameItems, filling any gaps and removing as needed
        """
        new_tracks = []
        new_frames = []
        for z, trk in self._doc.enumerate_tracks_frames():
            qti = self._track_items.get(trk.track)
            if qti is not None:
                self._sync_track(qti, z, trk)
            else:
                qti = self._create_track(z, trk)
                new_tracks.append(qti)
            for frm in trk.frames:
                qfi = self._frame_items.get(frm.uuid)
                if qfi is not None:
                    self._sync_frame(qfi, frm)
                else:
                    new_frames.append(self._create_frame(qti, frm))
        self.propagate_max_z()
        for track in new_tracks:
            track.update_pos_bounds()
            track.update_frame_positions()

    def _invalidate(self):
        """document state has changed, re-consult document and update our display
        """
        self._sync_tracks_frames()

    def _connect_signals(self, doc:Document):
        """Connect document, workspace, signals in order to invalidate and update scene representation
        """
        # FUTURE: more fine-grained response than just invalidating and redrawing
        doc.didAddBasicLayer.connect(self._invalidate)
        doc.didAddCompositeLayer.connect(self._invalidate)
        doc.didChangeLayerVisibility.connect(self._invalidate)
        doc.didChangeLayerName.connect(self._invalidate)
        doc.didReorderLayers.connect(self._invalidate)
        doc.didChangeComposition.connect(self._invalidate)

    def get(self, uuid: UUID) -> [QTrackItem, QFrameItem, None]:
        z = self._track_items.get(uuid, None)
        if z is not None:
            return z
        z = self._frame_items.get(uuid, None)
        return z

    def may_rearrange_track_z_order(self, track_list: List[str]) -> Optional[Callable[[bool], None]]:
        """Determine whether tracks can be rearranged and provide a commit/abort function if so
        Optionally: reflect any such changes on other parts of the application

        Args:
            track_list: new track name arrangement in top to bottom order

        Returns: None if rearrange is not permitted,
        else a callable that can be used to commit the change in the document
        callable(True) commits change; callable(False) aborts change
        Only one callable is valid at a time
        """
        LOG.warning("using base class may_rearrange_track_z_order which does nothing")
        return lambda commit: None

    def tracks_in_same_family(self, track: str) -> Set[str]:
        """inform the view on which tracks are closely related to the given track
        typically this is used to stylistically highlight related active tracks during a drag or presentation editing operation
        """
        LOG.warning("using base class tracks_in_same_family which does nothing")
        return set(self._doc.tracks_in_family(track, only_active=True))

    def may_reassign_color_map(self, from_track: UUID, to_track: UUID) -> Optional[Callable[[bool], None]]:
        """User is dragging a color map around, determine if drop is permitted and provide a commit/abort function if so
        """
        LOG.warning("using base class may_reassign_color_map which does nothing")
        return lambda b: None

    def menu_for_track(self, track: UUID, frame_uuid: UUID = None) -> Optional[QMenu]:
        """Generate QMenu to use as context menu for a given track, optionally with frame if mouse was over that frame"""
        LOG.warning("using base class menu_for_track which does nothing")

        return None

    def update(self, changed_tracks: [Set, None] = None, changed_frame_uuids: [Set, None] = None) -> int:
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
