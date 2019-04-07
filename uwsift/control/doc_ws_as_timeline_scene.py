#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Implement QGraphicsScene for Document and Workspace, as a delegate to TimelineView

This QGraphicsScene represents document.as_track_stack, passing operations back to the document

:author: R.K.Garcia <rkgarcia@wisc.edu>
:copyright: 2017-2018 by University of Wisconsin Regents, see AUTHORS for more details
:license: GPLv3, see LICENSE for more details
"""
import logging
from typing import Tuple, Optional, Mapping, List, Any, Callable, Set, Iterable
from uuid import UUID

from PyQt5.QtWidgets import QMenu

from uwsift.common import Span, State, Flags
from uwsift.model.document import Document, DocumentAsTrackStack, FrameInfo, TrackInfo
from uwsift.view.timeline.common import VisualState
from uwsift.view.timeline.items import QTrackItem, QFrameItem
from uwsift.view.timeline.scene import QFramesInTracksScene
from uwsift.workspace import Workspace

LOG = logging.getLogger(__name__)

# since timeline view is interface-independent of uwsift.common, we translate
DOC_STATE_TO_VISUAL_STATE = {
    State.POTENTIAL: set(),
    State.ARRIVING: {VisualState.BUSY},
    State.CACHED: {VisualState.AVAILABLE},
    State.ATTACHED: {VisualState.READY},
    State.ONSCREEN: {VisualState.HIGHLIGHT},
    State.UNKNOWN: {VisualState.ERROR},
    State.DANGLING: {VisualState.WARNING},
}


def _translate_to_visual_state(s: Flags) -> Flags:
    f = Flags()
    for x in s:
        f.update(DOC_STATE_TO_VISUAL_STATE.get(x) or set())
    return f


class SiftDocumentAsFramesInTracks(QFramesInTracksScene):
    """ represent SIFT Workspace and Document as frames in tracks
    Timeline QGraphicsScene uses doc.as_track_stack high level interface to query doc+mdb+ws
    Allow GUI manipulation to be passed thru to the document state and vice versa,
    including handling inquiries about the legality of operations like drag-n-drops
    """
    _doc: DocumentAsTrackStack = None

    def __init__(self, doc: Document, ws: Workspace, *args, **kwargs):
        """

        Args:
            doc (Document): owns user selections and constructions
            ws (Workspace): owns cached and computed data

        """
        super(SiftDocumentAsFramesInTracks, self).__init__(*args, **kwargs)
        self._doc = doc.as_track_stack  # we should be limiting our interaction to this context
        self._connect_signals(doc, ws)  # but the main doc is still the signaling hub

    @property
    def timeline_span(self) -> Span:
        return self._doc.timeline_span

    def _sync_track(self, qti: QTrackItem, z: int, trk: TrackInfo):
        qti.z = z
        qti.state = trk.state
        # qti.update()

    def _create_track(self, z: int, trk: TrackInfo) -> QTrackItem:
        qti = QTrackItem(self, self.coords, trk.track, z, trk.primary, trk.secondary)
        return qti

    def _create_frame(self, qti: QTrackItem, frm: FrameInfo) -> QFrameItem:
        qfi = QFrameItem(qti, self.coords, frm.uuid, frm.when.s, frm.when.d, _translate_to_visual_state(frm.state),
                         frm.primary, frm.secondary)
        return qfi

    def _sync_frame(self, qfi: QFrameItem, frm: FrameInfo):
        qfi.state = _translate_to_visual_state(frm.state)
        # qfi.update()

    def _purge_orphan_tracks_frames(self, tracks: Iterable[str], frames: Iterable[UUID]):
        """Remove QTrackItem and QFrameItem instances that no longer correspond to document content
        """
        LOG.debug("purging {} orphan tracks and {} orphan frames from timeline scene".format(len(tracks), len(frames)))
        for frid in frames:
            self.removeItem()
        self._frame_items = dict((k, v) for (k, v) in self._frame_items.items() if k not in frames)
        self._track_items = dict((k, v) for (k, v) in self._track_items.items() if k not in tracks)

    def _sync_tracks_frames(self):
        """populate QTrackItems and QFrameItems, filling any gaps and removing as needed
        """
        new_tracks = []
        new_frames = []
        orphan_tracks = set(self._track_items.keys())
        orphan_frames = set(self._frame_items.keys())
        LOG.debug(
            "current timeline scene population: {} frames in {} tracks".format(len(orphan_frames), len(orphan_tracks)))
        iters = 0
        for z, trk in self._doc.enumerate_tracks_frames():
            qti = self._track_items.get(trk.track)
            if qti is not None:
                self._sync_track(qti, z, trk)
                orphan_tracks.remove(trk.track)
            else:
                qti = self._create_track(z, trk)
                new_tracks.append(qti)
            _first = True
            LOG.debug("track {} z={} has {} frames".format(trk.track, z, len(trk.frames)))
            for frm in trk.frames:
                iters += 1
                if _first:  # debug
                    _first = False
                    LOG.debug("track {} frame {}".format(trk.track, frm))
                qfi = self._frame_items.get(frm.uuid)
                if qfi is not None:
                    self._sync_frame(qfi, frm)
                    try:
                        orphan_frames.remove(qfi.uuid)
                    except KeyError:
                        LOG.warning("frame {} <{}> found but not originally "
                                    "present in collection {}".format(frm.ident, frm.uuid, orphan_frames))
                else:
                    new_frames.append(self._create_frame(qti, frm))
        LOG.debug("added {} tracks and {} frames to timeline scene after {} iterations".format(len(new_tracks),
                                                                                               len(new_frames), iters))
        self._purge_orphan_tracks_frames(orphan_tracks, orphan_frames)
        self.propagate_max_z()
        for track in new_tracks:
            track.update_pos_bounds()
            track.update_frame_positions()
        super(SiftDocumentAsFramesInTracks, self).update()

    def _invalidate(self, *args, **kwargs):
        """document state has changed, re-consult document and update our display
        """
        self._sync_tracks_frames()

    def _sync_and_update_frame(self, uuid: UUID, frm: Optional[FrameInfo] = None):
        qfi = self._frame_items.get(uuid)
        if qfi is not None:
            if frm is None:
                frm = self._doc.frame_info_for_product(uuid=uuid)
            self._sync_frame(qfi, frm)
            qfi.update()
            return qfi
        else:  # FUTURE: create the frame and if necessary the track
            return None

    def sync_available_tracks(self):
        self._doc.sync_available_tracks()

    def _update_product_name(self, uuid: UUID, name: str):
        self._sync_and_update_frame(uuid)

    def _update_visibility_for_products(self, uuid_vis: Mapping[UUID, bool]):
        # set the corresponding display state for these products
        all_frame_items = [self._sync_and_update_frame(uuid) for uuid in uuid_vis.keys()]
        if None in all_frame_items:
            LOG.debug("a frame did not exist to update, have to refresh everything")
            self._invalidate()
        # LOG.warning("UNIMPLEMENTED: update display state for {} layers".format(len(uuid_vis)))

    # def _reorder_tracks_given_layer_order(self, new_order: tuple):
    #     # user dragged layer list around;
    #     # .as_layer_list should have updated the document track order for us a didReorderTracks
    #     self._invalidate()

    def _connect_signals(self, doc: Document, ws: Workspace):
        """Connect document, workspace, signals in order to invalidate and update scene representation
        """

        # FUTURE: more fine-grained response than just invalidating and redrawing
        def refresh_with_new_product(order, uuid, presentation, ts=self):
            LOG.debug("new layer added to document, refreshing timeline for product {}".format(str(uuid)))
            if ts._sync_and_update_frame(uuid) is None:
                LOG.info("no corresponding frame glyph, re-syncing timeline")
                ts.sync_available_tracks()
                ts.sync_items()

        doc.didAddBasicLayer.connect(refresh_with_new_product)
        doc.didAddCompositeLayer.connect(refresh_with_new_product)

        doc.didChangeLayerVisibility.connect(self._update_visibility_for_products)

        def refresh_product_new_name(uuid, name, ts=self):
            if ts._sync_and_update_frame(uuid) is None:
                LOG.warning("no corresponding frame glyph after rename??; re-syncing timeline")
                ts.sync_available_tracks()
                ts.sync_items()

        doc.didChangeLayerName.connect(refresh_product_new_name)

        # def refresh_track_order(self, added_tracks, removed_tracks, ts=self):
        #     pass
        # doc.didReorderTracks.connect(refresh_track_order)

        # doc.didReorderLayers.connect(self._reorder_tracks_given_layer_order)
        # doc.didChangeComposition.connect(self._invalidate)

        def refresh(changed_uuids, ts=self, *args, **kwargs):
            LOG.debug("updating timeline for {} changed products".format(len(changed_uuids)))
            ts.sync_available_tracks()
            ts.sync_items(changed_frame_uuids=changed_uuids)

        ws.didUpdateProductsMetadata.connect(refresh)

        def refresh_product(uuid, state, *args, **kwargs):
            LOG.debug('updating frame state {} in timeline'.format(str(uuid)))
            self._sync_and_update_frame(uuid)

        ws.didChangeProductState.connect(refresh_product)

    # def get(self, uuid: UUID) -> [QTrackItem, QFrameItem, None]:
    #     z = self._track_items.get(uuid, None)
    #     if z is not None:
    #         return z
    #     z = self._frame_items.get(uuid, None)
    #     return z

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
        """Inform the view on which tracks are closely related to the given track.

        Typically this is used to stylistically highlight related active tracks during a drag or presentation editing
        operation.

        """
        return set(self._doc.tracks_in_family(track, only_active=True))

    def may_reassign_color_map(self, from_track: str, to_track: str) -> Optional[Callable[[bool], None]]:
        """User is dragging a color map around, determine if drop is permitted and provide a commit/abort function if so
        """
        LOG.warning("using base class may_reassign_color_map which does nothing")
        return lambda b: None

    def menu_for_track(self, track: str, frame: Optional[UUID] = None) -> Optional[
            Tuple[QMenu, Mapping[Any, Callable]]]:
        """Generate QMenu to use as context menu for a given track, optionally with frame if mouse was over that frame
        """
        LOG.debug("generating menu with track {} and frame {}".format(track, frame))
        menu = QMenu()
        actions = {}
        if frame:
            sfx = ' frame'

            def _activate(*args, **kwargs):
                self._doc.activate_frames(frame)

            def _deactivate(*args, **kwargs):
                self._doc.deactivate_frames(frame)

        else:
            # FUTURE: frames in track is too dangerous for now
            return None

            sfx = ' frames in track'

            def _activate(*args, **kwargs):
                self._doc.activate_track(track)

            def _deactivate(*args, **kwargs):
                self._doc.deactivate_track(track)

        actions[menu.addAction("Activate" + sfx)] = _activate
        actions[menu.addAction("Deactivate" + sfx)] = _deactivate
        return menu, actions

    def sync_items(self, changed_tracks: Optional[Iterable[str]] = None,
                   changed_frame_uuids: Optional[Iterable[UUID]] = None):
        """Populate or update scene, returning number of items changed in scene
        Does not add new items for tracks and frames already present
        Parameters serve only as hints
        """
        acted = False
        with self._doc.mdb:  # optional peformance optimization to prevent session flipping
            if changed_frame_uuids is not None:
                changed_frame_uuids = list(changed_frame_uuids)
                all_frame_items = [self._sync_and_update_frame(uuid) for uuid in changed_frame_uuids]
                if None in all_frame_items:
                    LOG.debug("new frames, resorting to invalidate")
                    self._invalidate()
                    return
                LOG.debug("done updating {} products in timeline".format(len(changed_frame_uuids)))
                acted = True
        if changed_tracks is not None:
            LOG.warning("NOT IMPLEMENTED: selectively updating {} tracks in timeline".format(len(list(changed_tracks))))
        if not acted:
            self._invalidate()
        self.update()

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
#                         help='each occurrence increases verbosity 1 level through ERROR-WARNING-Info-DEBUG')
#     parser.add_argument('-d', '--debug', dest='debug', action='store_true',
#                         help="enable interactive PDB debugger on exception")
#     parser.add_argument('inputs', nargs='*',
#                         help="input files to process")
#     args = parser.parse_args()
#
#     levels = [logging.ERROR, logging.WARN, logging.Info, logging.DEBUG]
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


# if __name__ == '__main__':
#     sys.exit(main())
