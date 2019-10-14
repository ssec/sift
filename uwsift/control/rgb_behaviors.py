#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Behavior objects dealing with RGB image layers."""

import logging

from PyQt5.QtCore import QObject

from uwsift.common import Info, Kind
from uwsift.control.layer_tree import LayerStackTreeViewModel
# type hints:
from uwsift.model.document import Document
from uwsift.view.rgb_config import RGBLayerConfigPane

LOG = logging.getLogger(__name__)


class UserModifiesRGBLayers(QObject):
    """Behavior to handle RGB creation and modification actions.

    Tasks:

        1. User requests to create an RGB recipe, the layer list's current
           selections are used as default component layers.
        2. User selects an RGB layer in the layer list and the RGB pane is
           set to configure that RGB layer.
        3. User modifies the RGB configuration in the RGB pane triggering
           changes in the Document's representation of the RGB layer.
        4. A new layer family is discovered in the Document (a layer for a
           previously unloaded family is loaded) and the RGB pane is told
           about this new family.
        5. User selects layers in layer list, right clicks and in the context
           menu selects to create an RGB layer from the selections.

    """

    def __init__(self, document: Document, rgb_pane: RGBLayerConfigPane,
                 layer_list_model: LayerStackTreeViewModel, parent=None):
        super().__init__(parent)
        self.doc = document
        self.rgb_pane = rgb_pane
        self.layer_list_model = layer_list_model
        self._connect_signals()

    def _connect_signals(self):
        # Task 1
        # Added by Main UI: create_rgb
        # Task 2
        self.layer_list_model.uuidSelectionChanged.connect(self._selection_did_change)
        # Task 3
        self.rgb_pane.didChangeRGBComponentSelection.connect(self._component_changed)
        self.rgb_pane.didChangeRGBComponentLimits.connect(self._limits_changed)
        self.rgb_pane.didChangeRGBComponentGamma.connect(self._gamma_changed)
        # Task 4
        self.doc.didAddFamily.connect(self._family_added)
        self.doc.didRemoveFamily.connect(self._family_removed)
        # Task 5
        self.layer_list_model.didRequestRGBCreation.connect(self._create_rgb_from_uuids)

    def _create_rgb_from_uuids(self, recipe_dict):
        families = []
        for color in 'rgb':
            if color in recipe_dict:
                families.append(self.doc[recipe_dict[color]][Info.FAMILY])
            else:
                families.append(None)
        self.create_rgb(families=families)

    def create_rgb(self, action=None, families=[]):
        if len(families) == 0:
            # get the layers to composite from current selection
            uuids = [u for u in self.layer_list_model.current_selected_uuids() if
                     self.doc[u][Info.KIND] in [Kind.IMAGE, Kind.COMPOSITE]]
            if not uuids:
                LOG.warning("No layers available to make RGB")
                return
            families = [self.doc[u][Info.FAMILY] for u in uuids]
        if len(families) < 3:  # pad with None
            families = families + ([None] * (3 - len(families)))
        # Don't use non-basic layers as starting points for the new composite
        families = [f for f in families if
                    f is None or self.doc.family_info(f)[Info.KIND] in [Kind.IMAGE, Kind.COMPOSITE]]
        layer = next(self.doc.create_rgb_composite(families[0],
                                                   families[1],
                                                   families[2]))

        if layer is not None:
            self.layer_list_model.select([layer.uuid])

    def _selection_did_change(self, uuids=None):
        if uuids is not None and len(uuids) == 1:
            # get the recipe for the RGB layer if that's what this is
            layer = self.doc[uuids[0]]
            if layer[Info.KIND] == Kind.RGB:
                recipe = layer.get('recipe')
                self.rgb_pane.selection_did_change(recipe)
                return
        # disable the rgb pane
        self.rgb_pane.selection_did_change(None)

    def _component_changed(self, recipe, component, new_family):
        self.doc.change_rgb_recipe_components(recipe,
                                              **{component: new_family})

    def _limits_changed(self, recipe, new_limits):
        self.doc.change_rgb_recipe_prez(recipe, climits=new_limits)

    def _gamma_changed(self, recipe, new_gammas):
        self.doc.change_rgb_recipe_prez(recipe, gamma=new_gammas)

    def _family_added(self, family, family_info):
        # can't use RGB layers as components of an RGB
        if family_info[Info.KIND] not in [Kind.RGB]:
            self.rgb_pane.family_added(family, family_info)

    def _family_removed(self, family):
        self.rgb_pane.family_removed(family)
