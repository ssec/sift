import logging
from datetime import datetime
from typing import Callable, List, Optional
from uuid import UUID

from PyQt5.QtCore import QDateTime, QObject, pyqtSignal

from uwsift.control.qml_utils import QmlBackend, QmlLayerManager, TimebaseModel
from uwsift.control.time_matcher import TimeMatcher
from uwsift.control.time_matcher_policies import find_nearest_past
from uwsift.control.time_transformer import TimeTransformer
from uwsift.control.time_transformer_policies import WrappingDrivingPolicy
from uwsift.model.layer_item import LayerItem
from uwsift.model.layer_model import LayerModel
from uwsift.model.product_dataset import ProductDataset

LOG = logging.getLogger(__name__)


class TimeManager(QObject):
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

    didMatchTimes = pyqtSignal(dict)

    def __init__(self, animation_speed: float, matching_policy: Callable = find_nearest_past) -> None:
        super().__init__()
        self._animation_speed = animation_speed
        self._time_matcher = TimeMatcher(matching_policy)

        self._layer_model: Optional[LayerModel] = None

        self.qml_root_object = None
        self.qml_engine = None
        self._qml_backend = None
        self.qml_layer_manager: QmlLayerManager = QmlLayerManager()
        self.current_timebase_uuid = None

        self.qml_timestamps_model = TimebaseModel(timestamps=None)
        self._time_transformer: Optional[TimeTransformer] = None

    @property
    def qml_backend(self) -> QmlBackend:
        if self._qml_backend is None:
            raise RuntimeError("Trying to access time manager QML backend before it is initialized")
        return self._qml_backend

    @qml_backend.setter
    def qml_backend(self, backend):
        self._qml_backend = backend

    def connect_to_model(self, layer_model: LayerModel):
        self._layer_model = layer_model
        # FIXME: Access to private member
        self.qml_layer_manager._layer_model = layer_model

        policy = WrappingDrivingPolicy(self._layer_model.layers)
        layer_model.didUpdateLayers.connect(policy.on_layers_update)
        layer_model.didUpdateLayers.connect(self.update_qml_layer_model)
        layer_model.didUpdateLayers.connect(self.sync_to_time_transformer)
        layer_model.didChangeRecipeLayerNames.connect(self.update_qml_layer_model)
        layer_model.didReorderLayers.connect(self._update_layer_order)

        self.didMatchTimes.connect(self._layer_model.on_didMatchTimes)

        policy.didUpdatePolicy.connect(self.update_qml_timeline)
        self._time_transformer = TimeTransformer(policy)

    def tick(self, event):
        """Proxy function for `TimeManager.step()`.

        TimeManager cannot directly
        receive a signal from the animation timer signal because the latter
        passes an `event` that `step()` cannot deal with. Thus connect to
        this method to actually trigger `step()`.

        :param event: Event passed by `AnimationController.animation_timer` on expiry, simply dropped.
        """
        self.step()

    def step(self, backwards: bool = False):
        """Advance in time, either forwards or backwards, by one time step.

        :param backwards: Flag which sets advancement either to `forwards` or `backwards`.
        """
        assert self._time_transformer is not None  # nosec B101 # suppress mypy [union-attr]
        self._time_transformer.step(backwards=backwards)
        self.sync_to_time_transformer()

    def jump(self, index):
        self._time_transformer.jump(index)
        self.sync_to_time_transformer()

    def sync_to_time_transformer(self):  # noqa D102 MAKE_PRIVATE
        t_sim = self._time_transformer.t_sim
        t_idx = self._time_transformer.timeline_index

        t_matched_dict = self._match_times(t_sim)
        self.didMatchTimes.emit(t_matched_dict)

        self.tick_qml_state(t_sim, t_idx)

    def get_current_timebase_timeline(self):  # noqa D102 MAKE_PRIVATE
        timebase_layer = self._layer_model.get_layer_by_uuid(self.current_timebase_uuid)
        return timebase_layer.timeline if timebase_layer else {}

    def get_current_timebase_dataset_count(self):
        return len(self.get_current_timebase_timeline())

    def get_current_timebase_timeline_index(self):
        return self._time_transformer.timeline_index

    def get_current_timebase_current_dataset_uuid(self) -> Optional[UUID]:
        current_dataset = self.get_current_timebase_current_dataset()
        return None if not current_dataset else current_dataset.uuid

    def get_current_timebase_datasets(self) -> List[ProductDataset]:  # noqa D102 MAKE_PRIVATE
        timeline = self.get_current_timebase_timeline()
        timeline_datasets = list(timeline.values())
        return timeline_datasets

    def get_current_timebase_current_dataset(self):  # noqa D102 MAKE_PRIVATE
        i = self.get_current_timebase_timeline_index()
        try:
            return self.get_current_timebase_datasets()[i]
        except IndexError:
            return None

    def get_current_timebase_dataset_uuids(self) -> List[UUID]:
        return [ds.uuid for ds in self.get_current_timebase_datasets()]

    def _match_times(self, t_sim: datetime) -> dict:
        """
        Match time steps of available data in LayerModel's dynamic layers to
        `t_sim` of i.e.: a driving layer.

        A mapping of one layer to multiple soon-to-be visible ProductDatasets is
        made possible to support products (i.e.: Lightning) where multiple
        ProductDatasets may accumulate and must thus be made visible to the
        user.

        :param t_sim: Datetime of current active time step of time base.
        :return: Dictionary of possibly multiple tuples of
        (layer_uuid -> [product_dataset_uuid0,..,product_dataset_uuidN]),
        describing all ProductDatasets within a layer that are to be set
        visible.
        """
        assert self._layer_model is not None  # nosec B101 # suppress mypy [union-attr]
        t_matched_dict = {}
        for layer in self._layer_model.get_dynamic_layers():
            t_matched = self._time_matcher.match(layer.timeline, t_sim)
            if t_matched:
                t_matched_dict[layer.uuid] = [layer.timeline[t_matched].uuid]
            else:
                t_matched_dict[layer.uuid] = [None]
        return t_matched_dict

    def update_qml_timeline(self, layer: LayerItem):
        """Slot that updates and refreshes QML timeline state using a DataLayer.

        DataLayer is either:
            a) a driving layer or some other form of high priority data layer
            b) a 'synthetic' data layer, only created to reflect the best fitting
                timeline/layer info for the current policy -> this may be policy-dependant

        # TODO(mk): the policy should not be responsible for UI, another policy or an object
                    that ingests a policy and handles UI based on that?
        """
        assert self._time_transformer is not None  # nosec B101 # suppress mypy [union-attr]
        if self.qml_engine is None:
            raise RuntimeError("Can't update timeline until QML Engine has been assigned.")
        self.qml_engine.clearComponentCache()
        if not layer or not layer.dynamic:
            self.qml_timestamps_model.clear()
            self.qml_backend.clear_timeline()
            self._time_transformer.update_current_timebase()
        else:
            new_timestamp_qdts = list(map(lambda dt: QDateTime(dt), layer.timeline.keys()))

            if not self._time_transformer.t_sim:
                self.qml_timestamps_model.currentTimestamp = list(layer.timeline.keys())[0]  # type: ignore
            else:
                self.qml_timestamps_model.currentTimestamp = self._time_transformer.t_sim  # type: ignore
            self.qml_timestamps_model.timestamps = new_timestamp_qdts
        self.qml_backend.refresh_timeline()

    def update_qml_layer_model(self):
        """
        Slot connected to didUpdateCollection signal, responsible for
        managing the data layer combo box contents
        """
        dynamic_layers_descriptors = []
        # In case the current timebase layer isn't found again (it may have
        # been removed), select the first, therefore initialize to 0:
        new_index_of_current_timebase = 0
        for idx, layer in enumerate(self._layer_model.get_dynamic_layers()):
            dynamic_layers_descriptors.append(layer.descriptor)
            if layer.uuid == self.current_timebase_uuid:
                new_index_of_current_timebase = idx

        self.qml_layer_manager._qml_layer_model.layer_strings = dynamic_layers_descriptors

        self.qml_backend.didChangeTimebase.emit(new_index_of_current_timebase)

    def _update_layer_order(self):
        dynamic_layers = self._layer_model.get_dynamic_layers()
        dynamic_layers_descriptors = [layer.descriptor for layer in dynamic_layers]

        self.qml_layer_manager._qml_layer_model.layer_strings = dynamic_layers_descriptors

    def tick_qml_state(self, t_sim, timeline_idx):  # noqa D102 MAKE_PRIVATE
        # TODO(mk): if TimeManager is subclassed the behavior below must be adapted:
        #           it may no longer be desirable to show t_sim as the current time step
        self.qml_timestamps_model.currentTimestamp = self._time_transformer.t_sim
        self.qml_backend.doNotifyTimelineIndexChanged.emit(timeline_idx)

    def create_formatted_t_sim(self):
        """
        Used for updating the animation label during animation.
        """
        return self._time_transformer.create_formatted_time_stamp()

    def on_timebase_change(self, index):
        """
        Slot to trigger timebase change by looking up data layer at specified
        index. Then calls time transformer to execute change of the timebase.

        :param index: DataLayer index obtained by either: clicking an item in
                      the ComboBox or by clicking a convenience function in the
                      convenience function popup menu
        """

        dynamic_layers = self._layer_model.get_dynamic_layers()

        if not dynamic_layers:
            # FIXME: reset to initial state when last dynamic layer has been
            #  removed (as soon as layer removal becomes possible)
            self.update_qml_timeline(None)
            return

        assert 0 <= index < len(dynamic_layers)  # nosec B101

        layer = self._layer_model.get_dynamic_layers()[index]
        self.current_timebase_uuid = layer.uuid
        self._time_transformer.change_timebase(layer)
        self.update_qml_timeline(layer)
        self.sync_to_time_transformer()
