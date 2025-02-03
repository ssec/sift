#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
uwsift.model.document
---------------------

The document is an interface to further process some user interactions and delegate the import of new content to the
workspace. It also contains all metadata information of all loaded records.

The document handles the following tasks:
    - import new files
    - instruct the workspace to import new content
    - create a Presentation using metadata information
    - manage the currently active area definition used to present the data
    - manage ser color maps

The communication between the document and other parts of the application are done with signal/slot connections.

Document has zero or more Colormaps, determining how they're presented

The document does not own data (content). It only owns metadata (info).

All entities in the Document have a UUID that is their identity throughout their lifecycle,
and is often used as shorthand between subsystems. Document rarely deals directly with content.

:author: R.K.Garcia <rayg@ssec.wisc.edu> and others
:copyright: 2015 by University of Wisconsin Regents, see AUTHORS for more details
:license: GPLv3, see LICENSE for more details
"""
from __future__ import annotations

__author__ = "rayg"
__docformat__ = "reStructuredText"

import json
import logging
import os
import typing as typ
from uuid import UUID

from PyQt5.QtCore import QObject, pyqtSignal

from uwsift.common import Info, Kind, Presentation
from uwsift.model.area_definitions_manager import AreaDefinitionsManager
from uwsift.queue import TASK_DOING, TASK_PROGRESS, TaskQueue
from uwsift.util.common import get_initial_gamma, units_conversion
from uwsift.util.default_paths import DOCUMENT_SETTINGS_DIR
from uwsift.view.colormap import (
    COLORMAP_MANAGER,
    SITE_CATEGORY,
    USER_CATEGORY,
    PyQtGraphColormap,
)
from uwsift.workspace import BaseWorkspace, CachingWorkspace, SimpleWorkspace
from uwsift.workspace.metadatabase import Product

LOG = logging.getLogger(__name__)

###################################################################################################################


class Document(QObject):  # base class is rightmost, mixins left of that
    """Storage for dataset info and user information.

    This is the low-level "internal" interface that acts as a signaling hub.
    Direct access to the document is being deprecated.
    Most direct access patterns should be migrated to using a contextual view of the document,
    in order to reduce abstraction leakage and permit the document storage to evolve.
    """

    # signals
    didAddDataset = pyqtSignal(dict, Presentation)
    didUpdateBasicDataset = pyqtSignal(UUID, Kind)
    didChangeProjection = pyqtSignal(str)  # name of projection (area definition)
    didReorderTracks = pyqtSignal(set, set)  # added track names, removed track names
    didUpdateUserColormap = pyqtSignal(str)  # name of colormap which has an update

    def __init__(
        self,
        workspace: BaseWorkspace,
        queue: TaskQueue,
        config_dir=DOCUMENT_SETTINGS_DIR,
        **kwargs,
    ):
        super(Document, self).__init__(**kwargs)
        self.config_dir = config_dir
        self.queue = queue
        if not os.path.isdir(self.config_dir):
            LOG.debug("Creating settings directory {}".format(self.config_dir))
            os.makedirs(self.config_dir)

        self._workspace = workspace
        self._info_by_uuid: typ.Dict[UUID, dict] = {}  # dict(uuid:frozendict)

        self.colormaps = COLORMAP_MANAGER
        self.default_area_def_name = AreaDefinitionsManager.default_area_def_name()
        self.current_area_def_name = self.default_area_def_name

        # Create directory if it does not exist
        cmap_base_dir = os.path.join(self.config_dir, "colormaps")
        read_cmap_dir = os.path.join(cmap_base_dir, "site")  # read-only
        write_cmap_dir = os.path.join(cmap_base_dir, "user")  # writeable
        self.read_cmap_dir = read_cmap_dir
        self.write_cmap_dir = write_cmap_dir
        importable_cmap_cats = [(True, SITE_CATEGORY, read_cmap_dir), (False, USER_CATEGORY, write_cmap_dir)]
        for read_only, cmap_cat, cmap_dir in importable_cmap_cats:
            if not os.path.exists(cmap_dir):
                os.makedirs(cmap_dir)
            else:
                self.colormaps.import_colormaps(cmap_dir, read_only=read_only, category=cmap_cat)

    def find_colormap(self, colormap):
        if isinstance(colormap, str) and colormap in self.colormaps:
            colormap = self.colormaps[colormap]
        return colormap

    def area_definition(self, area_definition_name=None):
        return AreaDefinitionsManager.area_def_by_name(area_definition_name or self.current_area_def_name)

    def change_projection(self, area_def_name=None):
        if area_def_name is None:
            area_def_name = self.default_area_def_name
        assert area_def_name in AreaDefinitionsManager.available_area_def_names()  # nosec B101
        if area_def_name != self.current_area_def_name:
            LOG.debug(
                f"Changing projection (area definition) from" f" '{self.current_area_def_name}' to '{area_def_name}'"
            )
            self.current_area_def_name = area_def_name
            self.didChangeProjection.emit(self.current_area_def_name)

    def update_user_colormap(self, colormap, name):
        # Update new gradient into save location
        try:
            filepath = self.write_cmap_dir
            cmap_file = open(os.path.join(filepath, name + ".json"), "w")
            cmap_file.write(json.dumps(colormap, indent=2, sort_keys=True))
            cmap_file.close()
        except IOError:
            LOG.error("Error saving gradient: {}".format(name), exc_info=True)

        cmap = PyQtGraphColormap(colormap)
        self.colormaps[name] = cmap

        # Update live map
        self.didUpdateUserColormap.emit(name)

    def remove_user_colormap(self, name):
        try:
            os.remove(os.path.join(self.config_dir, "colormaps", "user", name + ".json"))
        except OSError:
            pass

        del self.colormaps[name]

    def current_projection_index(self):
        return list(AreaDefinitionsManager.available_area_def_names()).index(self.current_area_def_name)

    def change_projection_index(self, idx):
        return self.change_projection(tuple(AreaDefinitionsManager.available_area_def_names())[idx])

    def _insert_dataset_with_info(self, info: dict, cmap=None, style=None, insert_before=0):
        """
        insert a dataset into the presentations but do not signal
        :return: new Presentation tuple, new reordered indices tuple
        """
        if cmap is None:
            cmap = info.get(Info.COLORMAP)
        if style is None:
            style = info.get(Info.STYLE)
        gamma = get_initial_gamma(info)

        climits = self._workspace.get_range_for_dataset_no_fail(info)

        p = Presentation(
            uuid=info[Info.UUID],
            kind=info[Info.KIND],
            visible=True,
            colormap=cmap,
            style=style,
            climits=climits,
            gamma=gamma,
            opacity=1.0,
        )

        return p

    def activate_product_uuid_as_new_dataset(self, uuid: UUID, insert_before=0, **importer_kwargs):
        if uuid in self._info_by_uuid:
            LOG.debug("dataset already loaded: {}".format(uuid))
            self._workspace.import_product_content(uuid, **importer_kwargs)
            return

        # FUTURE: Load this async, the slots for the below signal need to be OK
        # with that
        self._workspace.import_product_content(uuid, **importer_kwargs)
        # updated metadata with content information (most importantly navigation
        # information)
        frozen_info = self._workspace.get_info(uuid)
        assert frozen_info is not None  # nosec B101
        info = dict(frozen_info)  # make a copy to which stuff can be added
        LOG.debug("cell_width: {}".format(repr(info[Info.CELL_WIDTH])))

        LOG.debug("new dataset info: {}".format(repr(info)))
        self._info_by_uuid[uuid] = info
        if Info.UNIT_CONVERSION not in info:
            info[Info.UNIT_CONVERSION] = units_conversion(info)
        if Info.FAMILY not in info:
            info[Info.FAMILY] = self._family_for_product_or_info(info)
        presentation = self._insert_dataset_with_info(info, insert_before=insert_before)

        # signal updates from the document
        self.didAddDataset.emit(info, presentation)

    def _family_for_product_or_info(self, uuid_or_info):
        if isinstance(uuid_or_info, UUID):
            if isinstance(self._workspace, CachingWorkspace):
                with self._workspace.metadatabase as s:
                    fam = s.query(Product.family).filter_by(uuid_str=str(uuid_or_info)).first()
            if isinstance(self._workspace, SimpleWorkspace):
                fam = self._workspace.get_info(uuid_or_info)[Info.FAMILY]
            if fam:
                return fam[0]
            uuid_or_info = self[uuid_or_info]
        if Info.FAMILY in uuid_or_info:
            LOG.debug("using pre-existing family {}".format(uuid_or_info[Info.FAMILY]))
            return uuid_or_info[Info.FAMILY]
        # kind:pointofreference:measurement:wavelength
        kind = uuid_or_info[Info.KIND]
        refpoint = "unknown"  # FUTURE: geo/leo
        measurement = uuid_or_info.get(Info.STANDARD_NAME)
        if uuid_or_info.get("recipe"):
            # RGB
            subcat = uuid_or_info["recipe"].name
        elif uuid_or_info.get(Info.CENTRAL_WAVELENGTH):
            # basic band
            subcat = uuid_or_info[Info.CENTRAL_WAVELENGTH]
        else:
            # higher level product or algebraic dataset
            subcat = uuid_or_info[Info.DATASET_NAME]
        return "{}:{}:{}:{}".format(kind.name, refpoint, measurement, subcat)

    def import_files(self, paths, insert_before=0, **importer_kwargs) -> typ.Generator[dict, None, None]:
        """Load product metadata and content from provided file paths.

        :param paths: paths to open
        :param insert_before: where to insert them in layer manager
        :return:

        """

        # NOTE: if the importer argument 'merge_with_existing' is not set it
        # defaults to True here.
        # TODO(AR) make 'merge_with_existing' an explicit argument to this
        #  method.
        do_merge_with_existing = importer_kwargs.get("merge_with_existing", True) and not importer_kwargs.get(
            "resampling_info"
        )
        # Ensure that the result of the test just performed is consistently
        # passed on to further import steps via importer_kwargs:
        importer_kwargs["merge_with_existing"] = do_merge_with_existing

        # Load all the metadata so we can sort the files
        # assume metadata collection is in the most user-friendly order
        infos = self._workspace.collect_product_metadata_for_paths(paths, **importer_kwargs)
        uuids = []
        merge_target_uuids = {}  # map new files uuids to merge target uuids
        total_products = 0
        for dex, (num_prods, info) in enumerate(infos):
            uuid = info[Info.UUID]
            merge_target_uuid = uuid
            if do_merge_with_existing:
                # real_paths because for satpy imports the methods paths parameter actually
                # contains the reader names
                real_paths = info["paths"]
                merge_target = self._workspace.find_merge_target(uuid, real_paths, info)
                if merge_target:
                    merge_target_uuid = merge_target.uuid

            yield {
                TASK_DOING: "Collecting metadata {}/{}".format(dex + 1, num_prods),
                TASK_PROGRESS: float(dex + 1) / float(num_prods),
                "uuid": merge_target_uuid,
                "num_products": num_prods,
            }
            # redundant but also more explicit than depending on num_prods
            total_products = num_prods
            uuids.append(uuid)
            merge_target_uuids[uuid] = merge_target_uuid

        if not total_products:
            raise ValueError("no products available in {}".format(paths))

        # collect product and resource information but don't yet import content
        for dex, uuid in enumerate(uuids):
            merge_target_uuid = merge_target_uuids[uuid]
            if do_merge_with_existing and uuid != merge_target_uuid:  # merge products
                active_content_data = self._workspace.import_product_content(
                    uuid, merge_target_uuid=merge_target_uuid, **importer_kwargs
                )
                # active_content_data is none if all segments are already loaded
                # and there is nothing new to import
                if active_content_data:
                    dataset_info = self[merge_target_uuid]
                    self.didUpdateBasicDataset.emit(merge_target_uuid, dataset_info[Info.KIND])
            elif uuid in self._info_by_uuid:
                LOG.warning("dataset with UUID {} already in document?".format(uuid))
                self._workspace.get_content(uuid)
            else:
                self.activate_product_uuid_as_new_dataset(uuid, insert_before=insert_before, **importer_kwargs)

            yield {
                TASK_DOING: "Loading content {}/{}".format(dex + 1, total_products),
                TASK_PROGRESS: float(dex + 1) / float(total_products),
                "uuid": merge_target_uuid,
                "num_products": total_products,
            }

    def sort_product_uuids(self, uuids: typ.Iterable[UUID]) -> typ.List[UUID]:
        assert isinstance(self._workspace, CachingWorkspace)  # nosec B101
        uuidset = set(str(x) for x in uuids)
        if not uuidset:
            return []
        with self._workspace.metadatabase as S:
            zult = [
                (x.uuid, x.ident)
                for x in S.query(Product)
                .filter(Product.uuid_str.in_(uuidset))
                .order_by(Product.family, Product.category, Product.serial)
                .all()
            ]
        LOG.debug("sorted products: {}".format(repr(zult)))
        return [u for u, _ in zult]

    # TODO: add a document style guide which says how different bands from different instruments are displayed

    def get_uuids(self):
        return list(self._info_by_uuid.keys())

    def __getitem__(self, dataset_uuid):
        """
        return dataset info with the given UUID
        """
        if dataset_uuid is None:
            raise KeyError("Key 'None' does not exist in document or workspace")
        elif not isinstance(dataset_uuid, UUID):
            raise ValueError("document[UUID] required, %r was used" % type(dataset_uuid))

        if dataset_uuid in self._info_by_uuid:
            return self._info_by_uuid[dataset_uuid]

        # check the workspace for information
        try:
            LOG.debug("Checking workspace for information on inactive product")
            info = self._workspace.get_info(dataset_uuid)
        except KeyError:
            info = None

        if info is None:
            raise KeyError("Key '{}' does not exist in document or workspace".format(dataset_uuid))
        return info

    def remove_dataset_info(self, uuid: UUID):
        """Remove the info of a dataset because it is no longer needed

        :param uuid: UUID of the dataset which is removed
        """
        LOG.debug(f"Remove dataset info of  uuid {uuid}")
        self._info_by_uuid.pop(uuid, None)
