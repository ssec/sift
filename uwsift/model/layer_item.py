import logging
from datetime import datetime
from types import MappingProxyType
from typing import List, Optional, Tuple
from uuid import uuid1, UUID

from uwsift import config
from uwsift.common import Presentation, N_A, Info, LayerModelColumns as LMC, \
    LayerVisibility
from uwsift.model.composite_recipes import Recipe
from uwsift.model.document import units_conversion
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
        self.info = self.extract_layer_info(info)

        self._probe_value: Optional[float] = None

        self.recipe = recipe

        self.grouping_key = grouping_key

        self._invariable_display_data = None
        self.update_invariable_display_data()

    @staticmethod
    def extract_layer_info(info: frozendict) -> frozendict:

        layer_info = {}
        unit_conversion = units_conversion(info)
        layer_info[Info.UNIT_CONVERSION] = unit_conversion

        for key in [
            Info.CLIM,
            Info.CENTRAL_WAVELENGTH,
            Info.INSTRUMENT,
            Info.KIND,
            Info.PLATFORM,
            Info.SHORT_NAME,
            Info.UNITS,
            Info.VALID_RANGE,
            # for _get_dataset_info_labels()
            "name",
            "standard_name",
            "wavelength"
        ]:
            if key in info:
                layer_info[key] = info[key]

        return frozendict(layer_info)

    @property
    def kind(self):
        return self.info[Info.KIND]

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

    def update_invariable_display_data(self) -> None:
        if self.recipe:
            self._invariable_display_data = {
                LMC.SOURCE: self.recipe.kind(),
                LMC.NAME: self.recipe.name,
                LMC.WAVELENGTH: N_A,
                LMC.PROBE_UNIT: N_A
            }
            return

        name, wavelength, unit = LayerItem._get_dataset_info_labels(self.info)

        platform = self.info.get(Info.PLATFORM)
        instrument = self.info.get(Info.INSTRUMENT)
        self._invariable_display_data = {
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
            if self.probe_value is None:
                return N_A
            uc_info = self.info[Info.UNIT_CONVERSION]
            uc_probe_value = uc_info[1](self.probe_value)
            # TODO: We cannot used uc_info[2](uc_probe_value) to get the
            #  formatted string (as in Main.update_point_probe_text()) because
            #  it would contain the unit. Thus for now we have to do our own
            #  formatting:
            return f"{uc_probe_value:.02f}"
        return None

    @property
    def probe_value(self):
        return self._probe_value

    @probe_value.setter
    def probe_value(self, probe_value: Optional[float]):
        self._probe_value = probe_value

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
            ds_unit = info[Info.UNIT_CONVERSION][0]
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
        self._sort_timeline()
        return product_dataset

    def has_in_timeline(self, dataset_uuid) -> bool:
        return dataset_uuid in [pds.uuid for pds in self._timeline.values()]

    def _sort_timeline(self):
        self._timeline = {kv[0]: kv[1] for kv in sorted(self._timeline.items(),
                                                        key=lambda kv: kv[0])}

    def get_datasets_uuids(self) -> List[UUID]:
        return [pd.uuid for pd in self.timeline.values()]

    def get_active_product_datasets(self) -> List[ProductDataset]:
        return [pd for pd in self.timeline.values() if pd.is_active]

    def get_first_active_product_dataset(self) -> Optional[ProductDataset]:
        active_product_datasets = self.get_active_product_datasets()
        num_active_product_datasets = len(active_product_datasets)
        # TODO: For now we do not support multiple active datasets per layer
        #  but this is likely to change in the future (e.g. for Lightning
        #  products). Let's make sure that any change in this regard doesn't
        #  go unnoticed here, thus:
        assert num_active_product_datasets <= 1

        return None if num_active_product_datasets == 0 \
            else active_product_datasets[0]

    def add_multichannel_dataset(
            self, presentation: Optional[Presentation],
            sched_time: datetime,
            input_datasets_uuids: Optional[List[UUID]],
            input_datasets_infos: Optional[List[frozendict]]
    ) -> ProductDataset:
        """Add multichannel ProductDataset to Layer. If a Presentation is passed
        it overwrites the Presentation of the layer for the given dataset.

        :param presentation: Mapping with visualisation configuration for the
               dataset to add.
        :param sched_time:
        :param input_datasets_uuids:
        :param input_datasets_infos: List of mapping providing metadata for
               ProductDatasets
        :return: Newly created  multichannel ProductDataset if a dataset
                 with the same scheduled time does not already
                 exist in the layer
        """
        if sched_time in self._timeline:
            LOG.warning(f"Other  multichannel dataset for {sched_time}"
                        f" is already in layer '{self.descriptor}',"
                        f" ignoring the new one.")
            return None

        product_dataset = ProductDataset.get_rgb_multichannel_product_dataset(
            self.uuid, presentation, input_datasets_uuids, self.kind,
            sched_time, input_datasets_infos
        )
        self._timeline[sched_time] = product_dataset
        self._sort_timeline()
        return product_dataset

    def remove_dataset(self, sched_time):
        """Remove a dataset for given datetime from layer

        Gracefully ignores if no dataset with the given shed_time exists in
        the layer.
        """
        self._timeline.pop(sched_time, None)

    def add_algebraic_dataset(self, presentation: Optional[Presentation],
                              info: frozendict, sched_time: datetime,
                              input_datasets_uuids: Optional[List[UUID]]
                              ):
        if sched_time in self._timeline:
            LOG.warning(f"Other  multichannel dataset for {sched_time}"
                        f" is already in layer '{self.descriptor}',"
                        f" ignoring the new one.")
            return None

        product_dataset = ProductDataset.get_algebraic_dataset(
            self.uuid, info, presentation, input_datasets_uuids
        )

        self._timeline[sched_time] = product_dataset
        self._sort_timeline()
        return product_dataset
