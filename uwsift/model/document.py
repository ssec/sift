#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
uwsift.model.document
---------------------

Core (low-level) document model for SIFT.
The core is sometimes accessed via Facets, which are like database views for a specific group of use cases

The document model is a metadata representation which permits the workspace to be constructed and managed.

Document is primarily a composition of layers.
Layers come in several flavors:

 - Image : a float32 field shown as tiles containing strides and/or alternate LODs, having a colormap
 - Outline : typically a geographic political map
 - Shape : a highlighted region selected by the user: point, line (great circle), polygon
 - Combination : calculated from two or more image layers, e.g. RGBA combination of images
                 combinations may be limited to areas specified by region layers.

Future Work:

 - Volume : a 3D dataset, potentially sparse
 - DenseVolume
 - SparseVolume : (x,y,z) point cloud

Layers are represented in 1 or more LayerSets, which are alternate configurations of the display.
Users may wish to display the same data in several different ways for illustration purposes.
Only one LayerSet is used on a given Map window at a time.

Layers have presentation settings that vary with LayerSet:

 - z_order: bottom to top in the map display
 - visible: whether or not it's being drawn on the map
 - a_order: animation order, when the animation button is hit
 - colormap: how the data is converted to pixels

Document has zero or more Probes. Layers also come in multiple
flavors that may be be attached to plugins or helper applications.

 - Scatter: (layerA, layerB, region) -> xy plot
 - Slice: (volume, line) -> curtain plot
 - Profile: (volume, point) -> profile plot

Document has zero or more Colormaps, determining how they're presented

The document does not own data (content). It only owns metadata (info).
At most, document holds coarse overview data content for preview purposes.

All entities in the Document have a UUID that is their identity throughout their lifecycle,
and is often used as shorthand between subsystems. Document rarely deals directly with content.

:author: R.K.Garcia <rayg@ssec.wisc.edu>
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
from uwsift.util.common import units_conversion
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
    """Storage for layer and user information.

    Document is a set of tracks in a Z order, with Z>=0 for "active" tracks the user is working with
    Tracks with Z-order <0 are inactive, but may be displayed in the timeline as potentials for the
    user to drag to active
    Document has a playhead, a playback time range, an active timeline display range
    Tracks and frames (aka Products) can have state information set

    This is the low-level "internal" interface that acts as a signaling hub.
    Direct access to the document is being deprecated.
    Most direct access patterns should be migrated to using a contextual view of the document,
    in order to reduce abstraction leakage and permit the document storage to evolve.
    """

    config_dir: str = None
    queue: TaskQueue = None
    _workspace: BaseWorkspace = None

    # DEPRECATION in progress: layer sets
    """
    Document has one or more LayerSets choosable by the user (one at a time) as currentLayerSet
    LayerSets configure animation order, visibility, enhancements and linear combinations
    LayerSets can be cloned from the prior active LayerSet when unconfigured
    Document has Probes, which operate on the currentLayerSet
    Probes have spatial areas (point probes, shaped areas)
    Probe areas are translated into localized data masks against the workspace raw data content
    """
    _info_by_uuid = None  # dict(uuid:Doc____Layer)

    # signals
    # Clarification: Layer interfaces migrate to layer meaning "current active products under the playhead"
    # new order list with None for new layer; info-dictionary, overview-content-ndarray
    didAddDataset = pyqtSignal(dict, Presentation)
    didUpdateBasicDataset = pyqtSignal(UUID, Kind)
    didChangeLayerVisibility = pyqtSignal(dict)  # {UUID: new-visibility, ...} for changed layers
    didChangeColormap = pyqtSignal(dict)  # dict of {uuid: colormap-name-or-UUID, ...} for all changed layers
    didChangeProjection = pyqtSignal(str)  # name of projection (area definition)
    didReorderTracks = pyqtSignal(set, set)  # added track names, removed track names
    didUpdateUserColormap = pyqtSignal(str)  # name of colormap which has an update

    def __init__(
        self,
        workspace: BaseWorkspace,
        queue,
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
        self._info_by_uuid = {}

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
        assert area_def_name in AreaDefinitionsManager.available_area_def_names()
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

    def _insert_layer_with_info(self, info: dict, cmap=None, style=None, insert_before=0):
        """
        insert a layer into the presentations but do not signal
        :return: new Presentation tuple, new reordered indices tuple
        """
        if cmap is None:
            cmap = info.get(Info.COLORMAP)
        if style is None:
            style = info.get(Info.STYLE)
        gamma = 1.0
        if info.get(Info.KIND) == Kind.RGB:
            gamma = (1.0,) * 3
        elif hasattr(info, "layers"):
            gamma = (1.0,) * len(info.layers)

        p = Presentation(
            uuid=info[Info.UUID],
            kind=info[Info.KIND],
            visible=True,
            a_order=None,
            colormap=cmap,
            style=style,
            climits=info[Info.CLIM],
            gamma=gamma,
            opacity=1.0,
        )

        return p

    def activate_product_uuid_as_new_dataset(self, uuid: UUID, insert_before=0, **importer_kwargs):
        if uuid in self._info_by_uuid:
            LOG.debug("Layer already loaded: {}".format(uuid))
            self._workspace.import_product_content(uuid, **importer_kwargs)
            return

        # FUTURE: Load this async, the slots for the below signal need to be OK
        # with that
        self._workspace.import_product_content(uuid, **importer_kwargs)
        # updated metadata with content information (most importantly navigation
        # information)
        info = dict(self._workspace.get_info(uuid))
        assert info is not None
        LOG.debug("cell_width: {}".format(repr(info[Info.CELL_WIDTH])))

        LOG.debug("new layer info: {}".format(repr(info)))
        self._info_by_uuid[uuid] = info
        if Info.UNIT_CONVERSION not in info:
            info[Info.UNIT_CONVERSION] = units_conversion(info)
        if Info.FAMILY not in info:
            info[Info.FAMILY] = self.family_for_product_or_layer(info)
        presentation = self._insert_layer_with_info(info, insert_before=insert_before)

        # signal updates from the document
        self.didAddDataset.emit(info, presentation)

    def family_for_product_or_layer(self, uuid_or_layer):
        if isinstance(uuid_or_layer, UUID):
            if isinstance(self._workspace, CachingWorkspace):
                with self._workspace.metadatabase as s:
                    fam = s.query(Product.family).filter_by(uuid_str=str(uuid_or_layer)).first()
            if isinstance(self._workspace, SimpleWorkspace):
                fam = self._workspace.get_info(uuid_or_layer)[Info.FAMILY]
            if fam:
                return fam[0]
            uuid_or_layer = self[uuid_or_layer]
        if Info.FAMILY in uuid_or_layer:
            LOG.debug("using pre-existing family {}".format(uuid_or_layer[Info.FAMILY]))
            return uuid_or_layer[Info.FAMILY]
        # kind:pointofreference:measurement:wavelength
        kind = uuid_or_layer[Info.KIND]
        refpoint = "unknown"  # FUTURE: geo/leo
        measurement = uuid_or_layer.get(Info.STANDARD_NAME)
        if uuid_or_layer.get("recipe"):
            # RGB
            subcat = uuid_or_layer["recipe"].name
        elif uuid_or_layer.get(Info.CENTRAL_WAVELENGTH):
            # basic band
            subcat = uuid_or_layer[Info.CENTRAL_WAVELENGTH]
        else:
            # higher level product or algebraic layer
            subcat = uuid_or_layer[Info.DATASET_NAME]
        return "{}:{}:{}:{}".format(kind.name, refpoint, measurement, subcat)

    def import_files(self, paths, insert_before=0, **importer_kwargs) -> dict:
        """Load product metadata and content from provided file paths.

        :param paths: paths to open
        :param insert_before: where to insert them in layer list
        :return:

        """

        # NOTE: if the importer argument 'merge_with_existing' is not set it
        # defaults to True here.
        # TODO(AR) make 'merge_with_existing' an explicit argument to this
        #  method.
        do_merge_with_existing = importer_kwargs.get("merge_with_existing", True) and not importer_kwargs.get(
            "resampling_info"
        )
        # Load all the metadata so we can sort the files
        # assume metadata collection is in the most user-friendly order
        infos = self._workspace.collect_product_metadata_for_paths(paths, **importer_kwargs)
        uuids = []
        merge_target_uuids = {}  # map new files uuids to merge target uuids
        total_products = 0
        for dex, (num_prods, info) in enumerate(infos):
            assert info is not None

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

        if isinstance(self._workspace, CachingWorkspace):
            # reverse list since we always insert a top layer
            uuids = list(reversed(self.sort_product_uuids(uuids)))

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
                    dataset = self[merge_target_uuid]
                    self.didUpdateBasicDataset.emit(merge_target_uuid, dataset[Info.KIND])
            elif uuid in self._info_by_uuid:
                LOG.warning("layer with UUID {} already in document?".format(uuid))
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

    def get_algebraic_namespace(self, uuid):
        return self._workspace.get_algebraic_namespace(uuid)

    def __getitem__(self, layer_uuid):
        """
        return layer with the given UUID
        """
        if layer_uuid is None:
            raise KeyError("Key 'None' does not exist in document or workspace")
        elif not isinstance(layer_uuid, UUID):
            raise ValueError("document[UUID] required, %r was used" % type(layer_uuid))

        if layer_uuid in self._info_by_uuid:
            return self._info_by_uuid[layer_uuid]

        # check the workspace for information
        try:
            LOG.debug("Checking workspace for information on inactive product")
            info = self._workspace.get_info(layer_uuid)
        except KeyError:
            info = None

        if info is None:
            raise KeyError("Key '{}' does not exist in document or workspace".format(layer_uuid))
        return info

    def remove_dataset_info(self, uuid: UUID):
        """Remove the info of a dataset because it is no longer needed

        :param uuid: UUID of the dataset which is removed
        """
        LOG.debug(f"Remove dataset info of  uuid {uuid}")
        self._info_by_uuid.pop(uuid, None)
