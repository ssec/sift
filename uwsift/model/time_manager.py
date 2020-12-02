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
    # TODO(mk): persist TimeTransformer and Policy across collection changes / in general
    # TODO(mk): implement policy property of time manager OR pass ref to policy to time transformer
    #           and do not save it as TimeManager's state

    @property
    def collection(self):
        return self._collection

    @collection.setter
    def collection(self, coll: DataLayerCollection):
        self._collection = coll
        # TODO(mk): persist state and keep old driving layer if it still exists in new collection
        #           do this in collection property setter of policy
        #           pick first layer of user selection from FileWizard as driving layer if no
        #           previous driving layer existed
        #           give GUI capability for user to change driving layer -> lower left of
        #           Timeline QQuickWidget opens ComboBox, 'Most Frequent' as one of the entries?
        policy = WrappingDrivingPolicy(self._collection)
        # WrappingDrivingPolicy.make_new_policy(policy, collection)
        # TODO(mk): parsing driving_layer_pfkey in layerToDisplay's setter may be brittle code for
        #           other types of satellite data
        self._time_transformer = TimeTransformer(policy)

    def tick(self, backwards=False):
        self._time_transformer.tick(backwards=backwards)
        t_sim = self._time_transformer.t_sim
        self.update_collection_state(t_sim)

    def create_formatted_t_sim(self):
        """
        Used for updating the animation label during animation.
        """
        return self._time_transformer.create_formatted_time_stamp()

    def update_collection_state(self, t_sim):
        """
            Iterate over data layers in collection to match times to t_sim and set the t_matched
            state of all data_layers to their t_matched.
        """
        for pfkey, data_layer in self._collection.data_layers.items():
            timeline = list(data_layer.timeline.keys())
            data_layer.t_matched = self._time_matcher.match(timeline, t_sim)
