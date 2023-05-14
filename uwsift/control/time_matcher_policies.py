from datetime import datetime
from typing import List, Optional

import numpy as np


# Example time Matching policies
def find_nearest(ref: List[datetime], query: datetime) -> Optional[datetime]:
    ref_np = np.asarray(ref)
    query_np = np.asarray(query)
    distances = np.abs(ref_np - query_np)
    return ref_np[np.argmin(distances)]


def find_nearest_past(ref: List[datetime], query: datetime) -> Optional[datetime]:
    query_np = np.asarray(query)
    ref_np = np.array([t for t in ref])
    past_idcs = ref_np <= query_np
    if np.any(past_idcs):
        distances = np.abs(ref_np[past_idcs] - query_np)
        return ref_np[np.argmin(distances)]
    else:
        return None
