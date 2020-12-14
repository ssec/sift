import logging
from typing import List

from uwsift.control.time_matcher import TimeMatcher
from uwsift.control.time_matcher_policies import find_nearest_past
from uwsift.control.time_transformer import TimeTransformer
from uwsift.control.time_transformer_policies import WrappingDrivingPolicy
from uwsift.model.document import DataLayer, DataLayerCollection

LOG = logging.getLogger(__name__)


class TimeManager:
    """
        Actions upon tick event:
            - Time Manager gets t_sim from t2t_translator
            - forwards it to Display Layers
            - Display Layers each give their timeline and t_sim to TimeMatcher
            - TimeMatcher returns t_matched for every non-driving layer timeline
            - each Display Layer requests the image corresponding to the matched timestamp
              from collection
            - Image is displayed
    """
    def __init__(self, collection: List[DataLayer], animation_speed,
                 matching_policy=find_nearest_past):

        self._collection = collection
        self._animation_speed = animation_speed
        self._time_transformer = None
        if self._collection is not None:
            self._time_transformer = TimeTransformer(self._collection, self._animation_speed)
        self._time_matcher = TimeMatcher(matching_policy)

    @property
    def collection(self):
        return self._collection

    @collection.setter
    def collection(self, coll: DataLayerCollection):
        self._collection = coll
        driving_layer_pfkey = list(coll.data_layers.keys())[0]
        policy = WrappingDrivingPolicy(self._collection,
                                       driving_layer_pfkey)
        self._time_transformer = TimeTransformer(policy)

    def tick(self, backwards=False):
        self._time_transformer.tick(backwards=backwards)
        t_sim = self._time_transformer.t_sim
        self.update_collection_state(t_sim)

    def update_collection_state(self, t_sim):
        """
            Iterate over data layers in collection to match times to t_sim and set the t_matched
            state of all data_layers to their t_matched.
        """
        for pfkey, data_layer in self._collection.data_layers.items():
            timeline = list(data_layer.timeline.keys())
            data_layer.t_matched = self._time_matcher.match(timeline, t_sim)
