#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Behavior objects dealing with RGB image layers."""

import logging
from PyQt4.QtCore import QObject
from sift.common import INFO, KIND

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

    """
    def __init__(self, document, rgb_pane, layer_list, parent=None):
        super().__init__(parent)
        self.doc = document
        self.rgb_pane = rgb_pane
        self.layer_list = layer_list
        self._connect_signals()

    def _connect_signals(self):
        # Task 1
        # Added by Main UI: create_rgb
        # Task 2
        self.layer_list.set_behaviors.uuidSelectionChanged.connect(self._selection_did_change)
        # Task 3
        self.rgb_pane.didChangeRGBComponentSelection.connect(self._component_changed)
        self.rgb_pane.didChangeRGBComponentLimits.connect(self._limits_changed)
        self.rgb_pane.didChangeRGBComponentGamma.connect(self._gamma_changed)
        # Task 4
        self.doc.didAddFamily.connect(self._family_added)

    def create_rgb(self, action=None, families=[]):
        layer_list_model = self.layer_list.getLayerStackListViewModel()
        if len(families) == 0:
            # get the layers to composite from current selection
            uuids = list(layer_list_model.current_selected_uuids())
            families = [self.doc[u][INFO.FAMILY] for u in uuids]
        if len(families) < 3:  # pad with None
            families = families + ([None] * (3 - len(families)))
        # Don't use non-basic layers as starting points for the new composite
        families = [f for f in families if f is None or self.doc.family_info(f)[INFO.KIND] in [KIND.IMAGE, KIND.COMPOSITE]]
        layer = next(self.doc.create_rgb_composite(families[0],
                                                   families[1],
                                                   families[2]))
        if layer is not None:
            layer_list_model.select([layer.uuid])

    def _selection_did_change(self, uuids=None):
        if uuids is not None and len(uuids) == 1:
            # get the recipe for the RGB layer if that's what this is
            layer = self.doc[uuids[0]]
            if layer[INFO.KIND] == KIND.RGB:
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
        if family_info[INFO.KIND] not in [KIND.RGB]:
            self.rgb_pane.family_added(family, family_info)

