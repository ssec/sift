import logging
import time

LOG = logging.getLogger(__name__)


class TimeTransformer:
    """
    Tick-to-Timestep Translator in concept.
        - get simulated time t_sim from driving layer/timeline
            at each tick: - check for curr. driving layer
                          - get t_sim from driving layer
        - sends t_sim to DisplayManager
    """
    def __init__(self, translation_policy):
        # Translation policy must always return a t_sim from its compute_t_sim method
        self._translation_policy = translation_policy

        self.curr_tick_time: float = -1.0
        self.prev_tick_time: float = -1.0
        self._tick_delta: float = -1.0
        self._tick_accum: float = 0.
        self.t_sim = None

    def tick(self, backwards=False):
        # tick times in milliseconds since Epoch
        self.curr_tick_time = time.time_ns()//1000000
        self.t_sim = self._translation_policy.compute_t_sim(self.curr_tick_time,
                                                            backwards=backwards)
