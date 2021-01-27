import logging
from typing import List

from PyQt5.QtCore import QStringListModel

from uwsift.control.time_matcher import TimeMatcher
from uwsift.control.time_matcher_policies import find_nearest_past
from uwsift.control.time_transformer import TimeTransformer
from uwsift.control.time_transformer_policies import WrappingDrivingPolicy
from uwsift.control.qml_utils import QmlLayerManager, QmlTimelineManager, MyTestModel2
from uwsift.model.document import DataLayer, DataLayerCollection
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pytz

LOG = logging.getLogger(__name__)


class TimeManager:
    # TODO(mk): make this class abstract and subclass,
    #           as soon as non driving layer policies are necessary?
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
    def __init__(self, collection: DataLayerCollection, animation_speed,
                 matching_policy=find_nearest_past):

        self._collection = collection
        self.qml_root_object = None
        self.qml_backend = None
        self.qml_layer_manager = QmlLayerManager()
        self.qml_timestamp_manager = QmlTimelineManager()
        test_dts = list(map(lambda dt: dt.replace(tzinfo=pytz.UTC).strftime("%H:%M"),
                            [datetime.now() + relativedelta(hours=i) for i in range(5)]))
        self.qml_test_model = MyTestModel2(vals=test_dts)
        # TODO(mk): remove the below as soon as the 2 above are working
        self.timeStampQStringModel = QStringListModel()

        self._animation_speed = animation_speed
        self._time_transformer = None
        self._init_collection(collection)
        self._animation_speed = animation_speed

        self._time_matcher = TimeMatcher(matching_policy)

    def _init_collection(self, collection: DataLayerCollection):
        self._collection = collection
        # TODO(mk): Give GUI capability for user to change driving layer -> lower left of
        #           Timeline QQuickWidget opens ComboBox, 'Most Frequent' as one of the entries?
        policy = WrappingDrivingPolicy(self._collection)
        self._collection.didUpdateCollection.connect(policy.on_collection_update)
        self._time_transformer = TimeTransformer(policy)

    @property
    def collection(self):
        return self._collection

    def tick(self, backwards=False):
        self._time_transformer.tick(backwards=backwards)
        t_sim = self._time_transformer.t_sim

        # TODO(mk): if TimeManager is subclassed the behavior below must be adapted:
        #           it may no longer be desirable to show t_sim as the current time step
        #self.qml_layer_manager.testModel.setStringList([t_sim.strftime("%Y-%m-%d %H:%M%Z")])
        self.qml_layer_manager.dateToDisplay = t_sim
        self.qml_layer_manager.test += 1
        # TODO(mk): below just show tick updates work but QML skips one update
        if True:
            print(f"T SIM: {t_sim}")
            print(f"QML D2Disp: {self.qml_layer_manager.dateToDisplay}")
            print(f"QML TEST {self.qml_layer_manager.test}")
        # self.timeStampQStringModel.dataChanged().emit()
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
