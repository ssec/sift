import numpy as np


# Example time Matching policies
def find_nearest(ref, query):
    ref_np = np.asarray(ref)
    query_np = np.asarray(query)
    distances = np.abs(ref_np - query_np)
    return ref_np[np.argmin(distances)]


def find_nearest_past(ref, query):
    query_np = np.asarray(query)
    ref_np = np.array([t for t in ref])
    past_idcs = ref_np <= query_np
    distances = np.abs(ref_np[past_idcs] - query_np)
    return ref_np[np.argmin(distances)]
