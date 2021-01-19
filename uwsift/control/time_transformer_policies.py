from datetime import datetime
import logging

from typing import List, Optional

from PyQt5.QtCore import QObject, pyqtSignal

from uwsift.model.document import DataLayer

import numpy as np

LOG = logging.getLogger(__name__)


class WrappingDrivingPolicy(QObject):
    """
        Time translation policy that when called returns a simulated data time
        by taking a timestamp from a designated data layer (the driving layer).
        Each successive call will return the driving layer's next timestamp. If there are
        no more timestamps in the driving layer it starts over from the first timestamp.
    """

    didUpdatePolicy = pyqtSignal(DataLayer)

    def __init__(self, collection):
        super().__init__()
        self._collection = collection
        self._driving_idx = 0
        self._timeline = None
        self._timeline_length = None
        self._driving_layer = None
        self._driving_layer_pfkey = None

    def on_collection_update(self):
        """
        Slot connected to Collection's didUpdateCollection signal. Takes the first loaded data layer
        if the old driving layer is not in the collection anymore.
        """
        if self._driving_layer_pfkey not in self._collection.product_family_keys:
            self.driving_layer = list(self._collection.data_layers.values())[0]
        else:
            self.driving_layer = self._collection.data_layers[self._driving_layer_pfkey]

    def change_timebase(self, data_layer):
        self.driving_layer = data_layer

    @property
    def driving_layer(self):
        return self._driving_layer

    @driving_layer.setter
    def driving_layer(self, data_layer: DataLayer):
        if not data_layer:
            raise ValueError("Driving layer needs to be a valid Product Family Key.")
        self._driving_layer_pfkey = data_layer.product_family_key
        if not self._driving_layer:
            try:
                self._collection.data_layers[data_layer.product_family_key]
            except KeyError:
                raise KeyError(
                    f"Driving layer {data_layer.product_family_key} "
                    f"not found in current collection.")
            self.timeline = list(self._collection.data_layers[self._driving_layer_pfkey]
                                 .timeline.keys())
            self._driving_layer = data_layer
            self._driving_idx = 0
        else:
            # Store timestamp of old timeline to retrieve analogous timestamp of new timeline.
            curr_tstamp = list(self._driving_layer.timeline.keys())[self._driving_idx]
            self.timeline = list(data_layer.timeline.keys())
            self._driving_layer = data_layer
            nearest_past_idx = self._find_nearest_past(curr_tstamp)
            if nearest_past_idx:
                self._driving_idx = nearest_past_idx
            else:
                self._driving_idx = 0
        self.didUpdatePolicy.emit(data_layer)

    @property
    def timeline(self):
        return self._timeline

    @timeline.setter
    def timeline(self, timeline: List[datetime]):
        self._timeline = list(self._collection.data_layers[self._driving_layer_pfkey]
                              .timeline.keys())
        self._timeline_length = len(self._timeline)

    @property
    def driving_layer_pfkey(self):
        return self._driving_layer_pfkey

    def _find_nearest_past(self, tstamp: datetime) -> Optional[int]:
        """
        Upon driving layer change find the nearest past tstamp in the new driving
        layer and return its index.
        """
        old_tstamp_np = np.asarray([tstamp])
        other_timeline_np = np.asarray(self.timeline)
        past_idcs = other_timeline_np <= old_tstamp_np
        distances = np.abs(other_timeline_np[past_idcs] - old_tstamp_np)
        if distances.size > 0:
            return np.argmin(distances)
        else:
            return None

    def curr_t_sim(self):
        if not self.timeline:
            return None
        else:
            return self.timeline[self._driving_idx]

    def curr_timeline_index(self):
        return self._driving_idx

    def jump_to_t_sim(self, index: int) -> datetime:
        """
            Returns t_sim by looking up the driving layer's timestamp at the provided index
            location. Raises an exception if being called with an invalid index.
        :param index: integer location in the timeline to jump to.
        :return: datetime object located at the provided index.
        """
        try:
            self._driving_idx = index
            t_sim = self.timeline[self._driving_idx]
            test = 5
        except Exception as e:
            LOG.error(f"Invalid index passed to driving layer timeline: "
                      f"index={index}\n", exc_info=True)
            raise e
        return t_sim

    def compute_t_sim(self, tick_time: int, backwards: bool = False) -> datetime:
        """
        Returns timestamp t_sim by:
            1) backwards set to False and index is None:
                incrementing the index of the current driving layer and returning the timestamp
                of the driving layer timeline at the index location
            2) backwards set to True and index is None:
                decrementing the index of the current driving layer...(see 1)
            3) backwards set to False but providing setting the index kwarg to a valid integer:
                looking up the driving layer's timestamp at the provided index location
        """
        if backwards:
            self._driving_idx = \
                (self._driving_idx + (self._timeline_length - 1)) % self._timeline_length
            t_sim = self.timeline[self._driving_idx]
        else:
            self._driving_idx += 1
            if self._driving_idx >= self._timeline_length:
                self._driving_idx = 0
            t_sim = self.timeline[self._driving_idx]
        return t_sim


