import logging
from typing import Optional
from uuid import UUID

from uwsift.common import Info, Presentation
from uwsift.workspace.workspace import frozendict

LOG = logging.getLogger(__name__)


class ProductDataset:
    def __init__(self, layer_uuid: UUID, info: frozendict,
                 presentation: Optional[Presentation] = None):
        self.info = info
        self.presentation = presentation
        self.is_active = False

        self.layer_uuid = layer_uuid

    @property
    def uuid(self):
        return self.info[Info.UUID]

    @property
    def kind(self):
        return self.info[Info.KIND]
