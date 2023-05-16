from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from uwsift.common import DEFAULT_TIME_FORMAT

LOG = logging.getLogger(__name__)

if TYPE_CHECKING:
    from uwsift.control.time_transformer_policies import WrappingDrivingPolicy


class TimeTransformer:
    """
    Tick-to-Timestep Translator in concept.
        - get simulated time t_sim from driving layer/timeline
            at each tick: - check for curr. driving layer
                          - get t_sim from driving layer
        - sends t_sim to DisplayManager
    """

    def __init__(self, translation_policy: WrappingDrivingPolicy):
        self._translation_policy = translation_policy

        self.curr_tick_time: float = -1.0
        self.prev_tick_time: float = -1.0
        self._tick_delta: float = -1.0
        self._tick_accum: float = 0.0
        self.t_sim = self._translation_policy.curr_t_sim()
        self.timeline_index = self._translation_policy.curr_timeline_index()

    def create_formatted_time_stamp(self, fmt=DEFAULT_TIME_FORMAT):
        return self.t_sim.strftime(fmt)

    def change_timebase(self, layer):
        self._translation_policy.change_timebase(layer)
        self.update_current_timebase()

    def jump(self, index):
        self.t_sim = self._translation_policy.jump_to_t_sim(index=index)
        self.timeline_index = self._translation_policy.curr_timeline_index()

    def step(self, backwards=False):
        # tick times in milliseconds since Epoch
        self.curr_tick_time = time.time_ns() // 1000000
        self.t_sim = self._translation_policy.compute_t_sim(self.curr_tick_time, backwards=backwards)
        self.timeline_index = self._translation_policy.curr_timeline_index()

    def update_current_timebase(self):
        """Update timebase parameters to pick up according changes from the time translation policy.

        Implementation details: This method is only public to be callable from TimeManager, no other use intended.
        """
        self.t_sim = self._translation_policy.curr_t_sim()
        self.timeline_index = self._translation_policy.curr_timeline_index()
