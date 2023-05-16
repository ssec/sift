import logging
from copy import deepcopy
from typing import Callable, Dict, List, Optional, Tuple
from uuid import UUID

from vispy import app

from uwsift import config
from uwsift.model.catalogue import Catalogue
from uwsift.queue import TASK_DOING, TASK_PROGRESS, TheQueue

LOG = logging.getLogger(__name__)


class StartTimeGranuleUpdatePolicy:
    def __init__(self, query_catalogue_for_satpy_importer_args: Callable):
        self._query_catalogue_for_satpy_importer_args = query_catalogue_for_satpy_importer_args
        self._last_scene_files = None

    # Check scenes list returned from catalogue
    # Compare last found scene with scene loaded in the previous update,
    # return None, if nothing has changed
    def update(self, current_constraints: Dict) -> Optional[Tuple]:
        """
        :param current_constraints: dictionary as provided by
                catalogue_settings.yaml.

        :returns: None if no new satpy scenes are available from the configured
                  search dir or a tuple consisting of the reader: List and
                  importer_kwargs: Dict needed to load data into SIFT.
        """
        reader_scenes_ds_ids, readers = self._query_catalogue_for_satpy_importer_args(current_constraints)
        if not reader_scenes_ds_ids or not readers:
            return None

        sorted_scenes_dict = sorted(reader_scenes_ds_ids["scenes"].items(), key=lambda item: item[1].start_time)
        if not sorted_scenes_dict:
            return None
        most_recent_scene_item = sorted_scenes_dict.pop()

        if most_recent_scene_item[0] != self._last_scene_files:
            self._last_scene_files = deepcopy(most_recent_scene_item[0])

            importer_kwargs = {
                "reader": reader_scenes_ds_ids["reader"],
                "scenes": {most_recent_scene_item[0]: most_recent_scene_item[1]},
                "dataset_ids": reader_scenes_ds_ids["dataset_ids"],
            }
            readers_to_use = [readers[0]]

            return readers_to_use, importer_kwargs
        return None


class AutoUpdateManager:
    def __init__(self, window, minimum_interval, search_path=None):
        # "Static"
        self.search_path = search_path
        self.reader = None
        self.filter_patterns = None
        self.group_keys = None
        self.products = None
        self._window = window

        # "Dynamically patched"
        self.filter = None  # update datetime to "now"
        self.timer = None
        # State
        self._files_loaded = None
        self._old_uuids = []

        self._init_catalogue()

        def update_in_background(event):
            TheQueue.add("auto update", self.update(), None)

        # Set up auto update mode timer, with minimum waiting time between
        # update cycles.
        # Minimum is exceeded if data loading time exceeds minimum_wait time.
        # Timer is paused until end of data loading to account for this.
        self.timer = app.Timer(minimum_interval, connect=update_in_background)
        self.timer.start()

        self._auto_update_policy = StartTimeGranuleUpdatePolicy(self._query_for_satpy_importer_kwargs_and_readers)

    def _query_for_satpy_importer_kwargs_and_readers(self, current_constraints):
        return Catalogue.query_for_satpy_importer_kwargs_and_readers(
            self.reader, self.search_path, self.filter_patterns, self.group_keys, current_constraints, self.products
        )

    def _init_catalogue(self):
        catalogue_config = config.get("catalogue", None)
        try:
            first_query = catalogue_config[0]
        except (TypeError, IndexError):
            msg = "No catalogue query found in configuration."
            raise RuntimeError(msg)

        (
            self.reader,
            self.search_path,
            self.filter_patterns,
            self.group_keys,
            self.filter,
            self.products,
        ) = Catalogue.extract_query_parameters(first_query)

    def on_loading_done(self, uuids: List[UUID]):
        # The time-consuming stuff is done, let's already start the next round
        self.timer.start()

        # Only upon completion of data loading allow for removal of old data.
        uuids_to_keep = set(uuids)
        uuids_to_remove = list(set(self._old_uuids) - uuids_to_keep)
        self._window.layer_model.remove_datasets_from_all_layers(uuids_to_remove)
        self._old_uuids.clear()

    def update(self):
        """
        Called by self.timer's tick
        """
        yield {TASK_DOING: "Run auto update", TASK_PROGRESS: 0.0}
        # remove all but the "newest" (with respect to 'start_time') scene from
        # importer_kwargs and according reader entries in readers_to_use
        readers_importer_tup = self._auto_update_policy.update(self.filter)

        if readers_importer_tup is not None:
            files_to_load, importer_kwargs = readers_importer_tup
            self._old_uuids = self._window.document.get_uuids()
            # Stop timer to prohibit slow data loading to transition directly to
            # deletion of just loaded or still loading data
            self.timer.stop()
            self._window.open_paths(files_to_load, **importer_kwargs)
        yield {TASK_DOING: "Auto update finished", TASK_PROGRESS: 1.0}
