from datetime import datetime
from typing import Tuple

import numpy as np


class WrappingDrivingPolicy:
    """
        Time translation policy that when called returns a simulated data time
        by taking a timestamp from a designated data layer (the driving layer).
        Each successive call will return the driving layer's next timestamp. If there are
        no more timestamps in the driving layer it starts over from the first timestamp.
    """
    def __init__(self, collection, driving_layer_pfkey):
        self._collection = collection
        self._driving_idx = 0
        self._driving_layer_pfkey = None
        self.timeline = None
        self.set_driving_layer(driving_layer_pfkey)

    def set_driving_layer(self, driving_layer_pfkey: Tuple) -> None:
        """
        Sets a new driving layer by product family key.
        Throws KeyError if that Product Family Key is not found in collection.
        """
        if driving_layer_pfkey is None:
            raise ValueError("Driving layer needs to be a valid Product Family Key.")
        try:
            self._collection.data_layers[driving_layer_pfkey]
        except KeyError:
            raise KeyError(f"Driving layer {driving_layer_pfkey} not found in current collection.")
        self._driving_layer_pfkey = driving_layer_pfkey
        if self.timeline is None:
            self.timeline = list(self._collection.data_layers[self._driving_layer_pfkey]
                                 .timeline.keys())
            self._driving_idx = 0
        else:
            # Store timestamp of old timeline to retrieve analogous timestamp of new timeline.
            curr_tstamp = self.timeline[self._driving_idx]
            self.timeline = list(self._collection.data_layers[self._driving_layer_pfkey]
                                 .timeline.keys())
            self._driving_idx = self._find_nearest_past(curr_tstamp)

    def _find_nearest_past(self, tstamp: datetime) -> int:
        """
        Upon driving layer change find the nearest past tstamp in the new driving
        layer and return its index.
        """
        old_tstamp_np = np.asarray([tstamp])
        other_timeline_np = np.asarray(self.timeline)
        past_idcs = other_timeline_np <= old_tstamp_np
        distances = np.abs(other_timeline_np[past_idcs] - old_tstamp_np)
        return np.argmin(distances)[0]

    def compute_t_sim(self, tick_time: int) -> datetime:
        if self._driving_idx >= len(self.timeline):
            self._driving_idx = 0
        t_sim = self.timeline[self._driving_idx]
        self._driving_idx += 1
        return t_sim


