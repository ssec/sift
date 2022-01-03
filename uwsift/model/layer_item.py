import logging
from types import MappingProxyType
from typing import Optional, Tuple
from uuid import uuid1

from uwsift import config
from uwsift.common import Presentation, N_A, Info, LayerModelColumns as LMC, \
    LayerVisibility
from uwsift.model.composite_recipes import Recipe
from uwsift.model.product_dataset import ProductDataset
from uwsift.workspace.workspace import frozendict

LOG = logging.getLogger(__name__)


class LayerItem:
    def __init__(self, model, info: frozendict, presentation: Presentation,
                 grouping_key=None,
                 recipe: Optional[Recipe] = None, parent=None):
        """
        Class to represent layers as items of LayerModel's `layers` collection.

        :param model: LayerModel holding this LayerItem.
        :param info: dataset metadata information
        :param presentation: Presentation object, used to represent information
        about visual appearance of a layer.
        :param recipe: If this is defined, the layer represents either a
        CompositeLayer or an AlgebraicLayer.
        :param parent: Parent of this layer, in case of a layer hierarchy within
        a LayerModel (NOTE: hierarchical layers are left for future development)
        """
        self.uuid = uuid1()
        if not model:
            raise ValueError(f"No valid model set for Layer: {self.uuid}")
        self.model = model

        self._parent = parent

        self._timeline = {}
        self._presentation = presentation
        self._kind = info[Info.KIND]

        self.recipe = recipe

        self.grouping_key = grouping_key
        self._invariable_display_data = \
            self._generate_invariable_display_data(info)

    @property
    def kind(self):
        return self._kind

    @property
    def order(self):
        return self.model.order(self)

    @property
    def name(self):
        return self._invariable_display_data[LMC.NAME]

    @property
    def descriptor(self):
        return f"{self._invariable_display_data[LMC.SOURCE]} " \
               f"{self._invariable_display_data[LMC.NAME]} " \
               f"{self._invariable_display_data[LMC.WAVELENGTH]}"

    @staticmethod
    def _generate_invariable_display_data(info) -> dict:
        name, wavelength, unit = LayerItem._get_dataset_info_labels(info)

        platform = info.get(Info.PLATFORM)
        instrument = info.get(Info.INSTRUMENT)
        return {
            LMC.SOURCE: f"{platform.value} {instrument.value}",
            LMC.NAME: name,
            LMC.WAVELENGTH: wavelength,
            LMC.PROBE_UNIT: unit
        }

    def data(self, column: int):
        if column in self._invariable_display_data:
            return self._invariable_display_data.get(column)
        if column == LMC.VISIBILITY:
            return LayerVisibility(self.visible, self.opacity)
        if column == LMC.PROBE_VALUE:
            return N_A  # point probing not yet implemented
        return None

    @property
    def visible(self):
        return self._presentation.visible

    @visible.setter
    def visible(self, visible):
        self._presentation.visible = visible

    @property
    def opacity(self):
        return self._presentation.opacity

    @opacity.setter
    def opacity(self, opacity):
        self._presentation.opacity = opacity

    @property
    def presentation(self):
        return self._presentation

    @presentation.setter
    def presentation(self, presentation):
        self._presentation = presentation

    @staticmethod
    def _get_dataset_info_labels(info: frozendict) -> Tuple[str, str, str]:
        """
        NOTE: Original code from Johan Strandgren (EUMETSAT) (see
        https://gitlab.eumetsat.int/strandgren/sift_dev/-/blob/master/layer_manager_naming_decision_tree.py),
        adopted and adapted.

        MAINTENANCE: The implementation is a little tricky, don't bother
        optimising redundancies.

        :param info: Info dictionary belonging to a dataset prototypical of
               a layer
        :return: 3 strings with the texts to display for name, wavelength, unit
        """
        try:
            ds_name = info["standard_name"]
        except KeyError:
            try:
                ds_name = info["name"]
            except KeyError:
                ds_name = N_A

        ds_name_from_config = config.get(f"standard_names.{ds_name}",
                                         default=None)

        if not ds_name_from_config:
            ds_name = ds_name.replace('_', ' ')
        else:
            ds_name = ds_name_from_config

        try:
            ds_wl = f"{info['wavelength'].central} {info['wavelength'].unit}"
        except KeyError:
            ds_wl = N_A

        try:
            ds_unit = info["units"]
        except KeyError:
            ds_unit = N_A

        return ds_name or N_A, ds_wl or N_A, ds_unit or N_A

    @property
    def timeline(self):
        return MappingProxyType(self._timeline)

    @property
    def dynamic(self):
        return len(self._timeline) != 0

    def add_dataset(self, info: frozendict,
                    presentation: Optional[Presentation] = None) \
            -> Optional[ProductDataset]:
        """
        Add ProductDataset to Layer. If a Presentation is passed
        it overwrites the Presentation of the layer for the given dataset.

        :param info: Mapping providing metadata for ProductDataset instantiation
        :param presentation: Mapping with visualisation configuration for the
               dataset to add.
        :return: Newly created ProductDataset if a dataset
                 with the same uuid or for the same scheduling does not already
                 exist in the layer
        """
        dataset_uuid = info[Info.UUID]
        sched_time = info[Info.SCHED_TIME]
        if self.has_in_timeline(dataset_uuid):
            LOG.debug(f"The given dataset for {sched_time}"
                      f" is already in layer '{self.descriptor}',"
                      f" nothing to do.")
            return None

        if sched_time in self._timeline:
            # TODO: Consider: Would it be better to overwrite the exiting entry?
            LOG.warning(f"Other dataset for {sched_time}"
                        f" is already in layer '{self.descriptor}',"
                        f" ignoring the new one.")
            return None

        product_dataset = ProductDataset(self.uuid, info, presentation)
        self._timeline[sched_time] = product_dataset
        return product_dataset

    def has_in_timeline(self, dataset_uuid) -> bool:
        return dataset_uuid in [pds.uuid for pds in self._timeline.values()]
