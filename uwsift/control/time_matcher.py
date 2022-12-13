from datetime import datetime
from typing import Callable, List, Optional


class TimeMatcher:
    def __init__(self, policy: Callable):
        self.policy = policy

    def match(self, timeline: List[datetime], t_sim: datetime, policy: Optional[Callable] = None):
        if not isinstance(t_sim, datetime):
            raise ValueError("Queried Time not an instance of a datetime object.")
        if policy is None:
            policy = self.policy
        matched = policy(timeline, t_sim)
        return matched
