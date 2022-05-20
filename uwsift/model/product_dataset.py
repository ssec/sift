import logging
from typing import Optional, List
from uuid import UUID, uuid1 as uuidgen

from uwsift.common import Info, Presentation, Kind
from uwsift.workspace.workspace import frozendict

LOG = logging.getLogger(__name__)


class ProductDataset:
    def __init__(self, layer_uuid: UUID, info: frozendict,
                 presentation: Presentation,
                 input_datasets_uuids: Optional[List[UUID]] = None):
        self.info = info
        self.presentation = presentation
        self.input_datasets_uuids = input_datasets_uuids
        self.is_active = False

        self.layer_uuid = layer_uuid

    @property
    def uuid(self):
        return self.info[Info.UUID]

    @property
    def kind(self):
        return self.info[Info.KIND]

    @classmethod
    def get_rgb_multichannel_product_dataset(
            cls, layer_uuid: UUID, presentation: Presentation,
            input_datasets_uuids: List[Optional[UUID]], kind: Kind,
            scheduled_time, input_datasets_infos: List[frozendict]
    ) -> Optional['ProductDataset']:

        if not any(input_datasets_uuids):
            LOG.debug("Could not create a multichannel ProductDataset, when no input ProductDatasets exist.")
            return None

        dataset_info = {
            Info.UUID: uuidgen(),
            Info.KIND: kind,
            Info.SCHED_TIME: scheduled_time
        }

        dataset_info = cls._update_info_with_input_infos(
            dataset_info, input_datasets_infos
        )

        if not dataset_info:
            return None

        dataset = cls(layer_uuid=layer_uuid, info=dataset_info,
                      presentation=presentation,
                      input_datasets_uuids=input_datasets_uuids)

        return dataset

    @staticmethod
    def _update_info_with_input_infos(
            dataset_info, input_datasets_infos
    ) -> dict:
        keys_to_compare = [Info.ORIGIN_X, Info.ORIGIN_Y, Info.CELL_WIDTH,
                           Info.CELL_HEIGHT, Info.PROJ]
        infos_to_compare = [info for info in input_datasets_infos if info]
        if len(infos_to_compare) == 0:
            LOG.debug("Could not create a multichannel ProductDataset, without any given datasets infos")
            return None
        for key in keys_to_compare:
            for idx in range(len(infos_to_compare) - 1):
                info_one = infos_to_compare[idx].get(key)
                info_two = infos_to_compare[-1].get(key)

                if info_one != info_two:
                    LOG.debug(f"Could not create a multichannel"
                              f" ProductDataset when input ProductDatasets"
                              f" have different values for {key}")
                    return None
        color_limits = tuple([info.get(Info.CLIM) if info else (None, None)
                              for info in input_datasets_infos])
        dataset_info.update(
            {
                Info.ORIGIN_X: infos_to_compare[-1].get(Info.ORIGIN_X),
                Info.ORIGIN_Y: infos_to_compare[-1].get(Info.ORIGIN_Y),
                Info.CELL_WIDTH: infos_to_compare[-1].get(Info.CELL_WIDTH),
                Info.CELL_HEIGHT: infos_to_compare[-1].get(Info.CELL_HEIGHT),
                Info.PROJ: infos_to_compare[-1].get(Info.PROJ),
                Info.CLIM: color_limits,
            }
        )

        return dataset_info

    def update_multichannel_dataset_info(self, input_datasets_infos):
        if not self.input_datasets_uuids:
            LOG.debug("Can't update multichannel dataset info if the dataset is not a multichannel dataset.")
            return None

        if not any(input_datasets_infos):
            LOG.debug("Can't update multichannel dataset info if no valid input datasets infos available.")
            return None

        dataset_info = {
            Info.UUID: self.uuid,
            Info.KIND: self.kind,
            Info.SCHED_TIME: self.info.get(Info.SCHED_TIME)
        }

        dataset_info = self._update_info_with_input_infos(dataset_info,
                                                          input_datasets_infos)

        self.info = dataset_info

    @classmethod
    def get_algebraic_dataset(
            cls, layer_uuid: UUID, info: frozendict,
            presentation: Presentation,
            input_datasets_uuids: List[Optional[UUID]]
    ):
        if not any(input_datasets_uuids):
            LOG.debug("Could not create a algebraic ProductDataset, when no input ProductDatasets exist.")
            return None

        dataset = cls(layer_uuid=layer_uuid, info=info,
                      presentation=presentation,
                      input_datasets_uuids=input_datasets_uuids)

        return dataset
