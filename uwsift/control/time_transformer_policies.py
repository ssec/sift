from datetime import datetime
from typing import Tuple, List

from uwsift.model.document import DataLayer

import numpy as np


class WrappingDrivingPolicy:
    """
        Time translation policy that when called returns a simulated data time
        by taking a timestamp from a designated data layer (the driving layer).
        Each successive call will return the driving layer's next timestamp. If there are
        no more timestamps in the driving layer it starts over from the first timestamp.
    """
    def __init__(self, collection):
        self._collection = collection
        self._driving_idx = 0
        self._timeline = None
        self._timeline_length = None
        self._driving_layer = None
        self._driving_layer_pfkey = None
        # Set first loaded layer as driving layer
        self.driving_layer = list(self._collection.data_layers.values())[0]

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
            self._driving_idx = 0
        else:
            # Store timestamp of old timeline to retrieve analogous timestamp of new timeline.
            curr_tstamp = list(self._driving_layer.timeline.keys())[self._driving_idx]
            self.timeline = list(self._driving_layer.timeline.keys())
            self._driving_idx = self._find_nearest_past(curr_tstamp)

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

    def _find_nearest_past(self, tstamp: datetime) -> int:
        """
        Upon driving layer change find the nearest past tstamp in the new driving
        layer and return its index.
        """
        old_tstamp_np = np.asarray([tstamp])
        other_timeline_np = np.asarray(self.timeline)
        past_idcs = other_timeline_np <= old_tstamp_np
        distances = np.abs(other_timeline_np[past_idcs] - old_tstamp_np)
        return np.argmin(distances)

    def curr_t_sim(self):
        if not self.timeline:
            return None
        else:
            return self.timeline[self._driving_idx]

    def compute_t_sim(self, tick_time: int, backwards=False) -> datetime:
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


