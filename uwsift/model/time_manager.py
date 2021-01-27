import logging
from typing import List

from PyQt5.QtCore import QObject, QStringListModel, QDateTime

from uwsift.control.time_matcher import TimeMatcher
from uwsift.control.time_matcher_policies import find_nearest_past
from uwsift.control.time_transformer import TimeTransformer
from uwsift.control.time_transformer_policies import WrappingDrivingPolicy
from uwsift.control.qml_utils import QmlLayerManager, QmlTimelineManager, MyTestModel2, \
                                     LayerModel, QmlBackend
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
        self.qml_engine = None
        self._qml_backend = None
        self.qml_layer_manager: QmlLayerManager = QmlLayerManager()

        self.qml_timeline_manager = QmlTimelineManager()

        # test_dts = list(map(lambda dt: dt.replace(tzinfo=pytz.UTC).strftime("%H:%M"),
        #                     [datetime.now() + relativedelta(hours=i) for i in range(5)]))
        dummy_dt = datetime.now()
        dummy_dt = datetime(dummy_dt.year, dummy_dt.month, dummy_dt.day, dummy_dt.hour)
        test_qdts = list(map(lambda dt: QDateTime(dt),
                             [dummy_dt + relativedelta(hours=i) for i in range(5)]))
        # self.qml_test_model = MyTestModel2(vals=test_dts)
        self.qml_test_model = MyTestModel2(vals=test_qdts)
        # TODO(mk): remove the below as soon as the 2 above are working
        self.timeStampQStringModel = QStringListModel()
        self._animation_speed = animation_speed
        self._time_transformer = None
        self._init_collection(collection)
        self._animation_speed = animation_speed

        self._time_matcher = TimeMatcher(matching_policy)

    @property
    def qml_backend(self) -> QmlBackend:
        return self._qml_backend

    @qml_backend.setter
    def qml_backend(self, backend):
        self._qml_backend = backend
        #self._qml_backend.didJumpInTimeline.connect(self.jump)

    def _init_collection(self, collection: DataLayerCollection):
        self._collection = collection
        # Expose collection's convenience functions to QmlLayerManager
        if self.qml_layer_manager is not None:
            self.qml_layer_manager.convenience_functions = self._collection.convenience_functions
        policy = WrappingDrivingPolicy(self._collection)
        self._collection.didUpdateCollection.connect(policy.on_collection_update)
        self._collection.didUpdateCollection.connect(self.update_qml_collection_representation)
        policy.didUpdatePolicy.connect(self.update_qml_timeline)
        self._time_transformer = TimeTransformer(policy)

    @property
    def collection(self):
        return self._collection

    def jump(self, index):
        self._time_transformer.jump(index)
        t_sim = self._time_transformer.t_sim
        t_idx = self._time_transformer.timeline_index
        self.tick_qml_state(t_sim, t_idx)
        self.update_collection_state(t_sim)

    def tick(self, backwards=False):
        self._time_transformer.tick(backwards=backwards)
        t_sim = self._time_transformer.t_sim
        t_idx = self._time_transformer.timeline_index
        self.tick_qml_state(t_sim, t_idx)
        ############################################################################################
        # TODO(mk): remove this once everything works
        if True:
            print(f"T SIM: {t_sim}")
            print(f"QML D2Disp: {self.qml_layer_manager.dateToDisplay}")
            print(f"QML TEST {self.qml_layer_manager.test}")
        ############################################################################################
        self.update_collection_state(t_sim)

    def update_qml_timeline(self, data_layer: DataLayer):
        # TODO(mk): pass data_layer as argument?
        """
        Slot that updates and refreshes QML timeline state using a DataLayer that is either:
            a) a driving layer or some other form of high priority data layer
            b) a 'synthetic' data layer, only created to reflect the best fitting
                timeline/layer info for the current policy -> this may be policy-dependant
            # TODO(mk): the policy should not be responsible for UI, another policy or an object
                        that ingests a policy and handles UI based on that?
        """
        self.qml_layer_manager.dateToDisplay = list(data_layer.timeline.keys())[0]

        #self.qml_layer_manager.layerToDisplay = data_layer.product_family_key
        self.qml_timeline_manager.update(data_layer.timeline)
        self.qml_engine.clearComponentCache()


        # TODO(mk): make datetime formatter configurable?
        #tstamps = list(map(lambda dt: dt.strftime("%H:%M"), data_layer.timeline.keys()))
        test_qdts = list(map(lambda dt: QDateTime(dt), data_layer.timeline.keys()))
        self.qml_test_model.vals = test_qdts
        self.qml_backend.refresh_timeline()

    def update_qml_collection_representation(self):
        """
        Slot connected to didUpdateCollection signal, responsible for managing the data layer
        combo box contents
        """
        data_layers_str = []
        self.qml_layer_manager.layerModel.clear()
        for i, pfkey in enumerate(self._collection.data_layers):
            dl_str = self.qml_layer_manager.format_product_family_key(pfkey)
            data_layers_str.append(dl_str)
        self.qml_layer_manager.layerModel.push(data_layers_str)

    def tick_qml_state(self, t_sim, timeline_idx):
        # TODO(mk): if TimeManager is subclassed the behavior below must be adapted:
        #           it may no longer be desirable to show t_sim as the current time step
        self.qml_layer_manager.dateToDisplay = t_sim
        self.qml_timeline_manager.timelineIndex = timeline_idx
        self.qml_backend.notify_tidx_changed(timeline_idx)
        print("TICKED QML STATE")
        self.qml_layer_manager.test += 1

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

    def on_timebase_change(self, index):
        """
        Slot to trigger timebase change by looking up data layer at specified index.
         Then calls time transformer to execute change of the timebase.
        :param index: DataLayer index obtained by either: clicking an item in the ComboBox or
                      by clicking a convenience function in the convenience function popup menu
        """
        data_layer = self._collection.get_data_layer_by_index(index)
        if data_layer:
            self._time_transformer.change_timebase(data_layer)
            self.qml_backend.refresh_timeline()
