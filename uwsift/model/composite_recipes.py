#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Composite recipe utilities and classes.

Composites in SIFT can be generated in two main ways:

- Algebraic layers: Combine one or more layers in to a new single band
                    layer by performing arithmetic between the input layers.
                    These composites are typically calculated once, can't be
                    modified, and are cached on disk.
- RGB layers: Combine 1-3 layers in to a red, green, blue channel image
              to produce a colorful RGB image. These composites are typically
              generated on-the-fly by the GPU by providing all inputs as
              textures. These composites are typically not cached on disk.

This module deals with the on-the-fly type composites like RGB layers. Since
these composites are not cached, the recipes to make them must be stored so
they can be recreated in the future.

"""

import dataclasses
import logging
import os
import uuid
from abc import abstractmethod
from dataclasses import dataclass
from glob import glob
from typing import Mapping, Optional, Tuple
from uuid import uuid1 as uuidgen

import yaml
from PyQt5.QtCore import QObject, pyqtSignal

from uwsift.util.default_paths import DOCUMENT_SETTINGS_DIR

LOG = logging.getLogger(__name__)

CHANNEL_RED = 0
CHANNEL_GREEN = 1
CHANNEL_BLUE = 2
CHANNEL_ALPHA = 3

RGBA2IDX: Mapping[str, int] = dict(r=CHANNEL_RED, g=CHANNEL_GREEN, b=CHANNEL_BLUE, a=CHANNEL_ALPHA)

IDX2RGBA: Mapping[int, str] = dict([(0, "r"), (1, "g"), (2, "b"), (3, "a")])

CHANNEL_X = 0
CHANNEL_Y = 1
CHANNEL_Z = 2

XYZ2IDX: Mapping[str, int] = dict(x=CHANNEL_X, y=CHANNEL_Y, z=CHANNEL_Z)

IDX2XYZ: Mapping[int, str] = dict([(0, "x"), (1, "y"), (2, "z")])

DIFF_OP_NAME = "Difference"
NDI_OP_NAME = "Normalized Difference Index"
CUSTOM_OP_NAME = "Custom..."
PRESET_OPERATIONS = {
    DIFF_OP_NAME: ("result = x - y", 2),
    NDI_OP_NAME: ("result = (x - y) / (x + y)", 2),
}


@dataclass
class Recipe:
    """
    Recipe base class. All recipes belong to a Layer and store information
    which input Layers provide the image data that is used to generate the
    images of their Layer.
    """

    name: str
    input_layer_ids: list = dataclasses.field(default_factory=list)
    read_only: bool = False

    def __post_init__(self) -> None:
        self.__id: uuid.UUID = uuidgen()

    @property
    def id(self):
        return self.__id

    def to_dict(self):
        """Convert to YAML-compatible dict."""
        return dataclasses.asdict(self)

    def copy(self, new_name):
        """Get a copy of this recipe with a new name"""
        return dataclasses.replace(self, name=new_name)

    @classmethod
    @abstractmethod
    def kind(cls):
        pass


@dataclass
class CompositeRecipe(Recipe):
    """
    Recipe class responsible for storing the combination of 1-3 layers as red,
    green and blue channel image to produce a colorful RGB image. These
    composites are typically generated on-the-fly by the GPU by providing
    all inputs as textures.

    Do not instantiate this class directly but use `CompositeRecipe.from_rgb()`.
    """

    color_limits: list = dataclasses.field(default_factory=list)
    gammas: list = dataclasses.field(default_factory=list)

    def __post_init__(self):
        super().__post_init__()

        def _normalize_list(x, default=None):
            return [x[idx] if x and len(x) > idx and x[idx] else default for idx in range(3)]

        self.input_layer_ids = _normalize_list(self.input_layer_ids)
        self.color_limits = _normalize_list(self.color_limits, (None, None))
        self.gammas = _normalize_list(self.gammas, 1.0)

    @classmethod
    def from_rgb(cls, name, r=None, g=None, b=None, color_limits=None, gammas=None):
        return cls(name, input_layer_ids=[r, g, b], color_limits=color_limits, gammas=gammas)

    def _channel_info(self, idx):
        """
        Get the control parameters for one of the composite channels as a dict.

        :param idx: Index of the channel (0 = red, 1 = green, 2 = blue).
        :return: Info dict for the channel
        """
        return {
            "name": self.input_layer_ids[idx],
            "color_limits": self.color_limits[idx],
            "gamma": self.gammas[idx],
        }

    def set_default_color_limits(self, r=None, g=None, b=None):
        """Set color limits based on dependency limits"""
        if self.read_only:
            raise RuntimeError("Composite recipe is read only")
        for idx, comp in enumerate([r, g, b]):
            if comp is None:
                # component was not updated
                continue
            if self.input_layer_ids[idx] is None:
                # our component is None
                self.color_limits[idx] = (None, None)
            else:
                self.color_limits[idx] = comp

    @property
    def red(self):
        """Get the control parameters for the red channel as a dict."""
        return self._channel_info(CHANNEL_RED)

    @property
    def green(self):
        """Get the control parameters for the green channel as a dict."""
        return self._channel_info(CHANNEL_GREEN)

    @property
    def blue(self):
        """Get the control parameters for the blue channel as a dict."""
        return self._channel_info(CHANNEL_BLUE)

    @classmethod
    def kind(cls):
        return "RGB Composite"


@dataclass
class AlgebraicRecipe(Recipe):
    operation_kind: str = dataclasses.field(default_factory=str)
    operation_formula: str = dataclasses.field(default_factory=str)

    def __post_init__(self):
        super().__post_init__()

        self.__modified = True

        def _normalize_list(x, default=None):
            return [x[idx] if x and len(x) > idx and x[idx] else default for idx in range(3)]

        self.input_layer_ids = _normalize_list(self.input_layer_ids)

        if self.operation_kind not in [DIFF_OP_NAME, NDI_OP_NAME, CUSTOM_OP_NAME]:
            self.operation_kind = DIFF_OP_NAME

        if self.operation_formula is None:
            self.operation_formula = PRESET_OPERATIONS.get(self.operation_kind, PRESET_OPERATIONS.get(DIFF_OP_NAME))

    @classmethod
    def from_algebraic(cls, name, x=None, y=None, z=None, operation_kind=None, operation_formula=None):
        return cls(name, input_layer_ids=[x, y, z], operation_kind=operation_kind, operation_formula=operation_formula)

    @classmethod
    def kind(cls):
        return "Algebraic"

    @property
    def modified(self) -> bool:
        return self.__modified

    @modified.setter
    def modified(self, status: bool):
        self.__modified = status  # noqa


class RecipeManager(QObject):
    # RGB Composites
    didCreateRGBCompositeRecipe = pyqtSignal(CompositeRecipe)
    didUpdateRGBInputLayers = pyqtSignal(CompositeRecipe)
    didUpdateRGBColorLimits = pyqtSignal(CompositeRecipe)
    didUpdateRGBGamma = pyqtSignal(CompositeRecipe)
    # Algebraics
    didCreateAlgebraicRecipe = pyqtSignal(AlgebraicRecipe)
    didUpdateAlgebraicInputLayers = pyqtSignal(AlgebraicRecipe)
    # Common
    didUpdateRecipeName = pyqtSignal(Recipe)

    def __init__(self, parent=None, config_dir=None):
        super(RecipeManager, self).__init__(parent)
        if config_dir is None:
            config_dir = DOCUMENT_SETTINGS_DIR

        recipe_dir = os.path.join(config_dir, "composite_recipes")
        if not os.path.isdir(recipe_dir):
            LOG.debug("creating new composite recipes directory at {}".format(recipe_dir))
            os.makedirs(recipe_dir)
        self.recipe_dir = recipe_dir
        # recipe_name -> recipe object
        self.recipes = {}
        # recipe_name -> (filename, recipe object)
        self._stored_recipes = {}
        self.load_available_recipes()

    def load_available_recipes(self):
        """Load recipes from stored config files"""
        for pathname in glob(os.path.join(self.recipe_dir, "*")):
            if not (pathname.endswith(".yml") or pathname.endswith(".yaml")):
                continue
            recipe = self.open_recipe(pathname)
            self._stored_recipes[recipe.name] = (pathname, recipe)

    def _add_recipe(self, recipe):
        self.recipes[recipe.id] = recipe

    def create_rgb_recipe(self, layers):
        """Create an RGB recipe and triggers a signal that a rgb composite
        layer can be created.

        :param layers: The layers which will be used to create a rgb composite
        """

        recipe_name = CompositeRecipe.kind()
        recipe = CompositeRecipe.from_rgb(
            recipe_name,
            r=None if layers[0] is None else layers[0].uuid,
            g=None if layers[1] is None else layers[1].uuid,
            b=None if layers[2] is None else layers[2].uuid,
        )
        self._add_recipe(recipe)

        self.didCreateRGBCompositeRecipe.emit(recipe)

    def update_rgb_recipe_input_layers(
        self,
        recipe: CompositeRecipe,
        channel: str,
        layer_uuid: Optional[uuid.UUID],
        clims: Tuple[Optional[float], Optional[float]],
        gamma: float,
    ):
        """Update the input layers in the recipe for a specific channel.
        With this change, the color limits and the gamma value of this specific
        channel has to be changed, too.
        """
        channel_idx = RGBA2IDX.get(channel)

        assert channel_idx is not None, f"Given channel '{channel}' is invalid"  # nosec B101

        recipe.input_layer_ids[channel_idx] = layer_uuid
        recipe.color_limits[channel_idx] = clims
        recipe.gammas[channel_idx] = gamma

        self.recipes[recipe.id] = recipe
        self.didUpdateRGBInputLayers.emit(recipe)

    def update_rgb_recipe_gammas(self, recipe: CompositeRecipe, channel: str, gamma: float):
        """Update the gamma value of the given channel"""
        channel_idx = RGBA2IDX.get(channel)

        assert channel_idx is not None, f"Given channel '{channel}' is invalid"  # nosec B101

        recipe.gammas[channel_idx] = gamma

        self.recipes[recipe.id] = recipe
        self.didUpdateRGBGamma.emit(recipe)

    def update_rgb_recipe_color_limits(self, recipe: CompositeRecipe, channel: str, clim: Tuple[float, float]):
        """Update the color limit value of the given channel"""
        channel_idx = RGBA2IDX.get(channel)

        assert channel_idx is not None, f"Given channel '{channel}' is invalid"  # nosec B101

        recipe.color_limits[channel_idx] = clim

        self.recipes[recipe.id] = recipe
        self.didUpdateRGBColorLimits.emit(recipe)

    def update_recipe_name(self, recipe: CompositeRecipe, name: str):
        recipe.name = name

        self.recipes[recipe.id] = recipe
        self.didUpdateRecipeName.emit(recipe)

    def create_algebraic_recipe(self, layers):
        recipe_name = AlgebraicRecipe.kind()
        recipe = AlgebraicRecipe.from_algebraic(
            recipe_name,
            x=None if layers[0] is None else layers[0].uuid,
            y=None if layers[1] is None else layers[1].uuid,
            z=None if layers[2] is None else layers[2].uuid,
            operation_kind=DIFF_OP_NAME,
        )
        self._add_recipe(recipe)

        self.didCreateAlgebraicRecipe.emit(recipe)

    def update_algebraic_recipe_operation_kind(self, recipe: AlgebraicRecipe, operation_kind: str):
        recipe.operation_kind = operation_kind
        recipe.modified = True
        self.recipes[recipe.id] = recipe

    def update_algebraic_recipe_operation_formula(self, recipe: AlgebraicRecipe, operation_formula: str):
        recipe.operation_formula = operation_formula
        recipe.modified = True
        self.recipes[recipe.id] = recipe

    def update_algebraic_recipe_input_layers(
        self, recipe: AlgebraicRecipe, channel: str, layer_uuid: Optional[uuid.UUID]
    ):
        channel_idx = XYZ2IDX.get(channel)

        assert channel_idx is not None, f"Given channel '{channel}' is invalid"  # nosec B101

        recipe.input_layer_ids[channel_idx] = layer_uuid
        recipe.modified = True
        self.recipes[recipe.id] = recipe

    def remove_layer_as_recipe_input(self, layer_uuid: uuid.UUID):
        """
        Remove a layer from all recipes in which it is used as input layer.

        Must be called before the layer given by the layer_uuid can be removed from the system.

        :param layer_uuid: UUID of the layer to be removed from all recipes
        """
        for recipe in self.recipes.values():
            if layer_uuid in recipe.input_layer_ids:
                idx = recipe.input_layer_ids.index(layer_uuid)
                if isinstance(recipe, CompositeRecipe):
                    channel = IDX2RGBA[idx]
                    self.update_rgb_recipe_input_layers(recipe, channel, None, (None, None), 1.0)
                if isinstance(recipe, AlgebraicRecipe):
                    channel = IDX2XYZ[idx]
                    self.update_algebraic_recipe_input_layers(recipe, channel, None)
                    self.didUpdateAlgebraicInputLayers.emit(recipe)

    def __getitem__(self, recipe_id):
        return self.recipes[recipe_id]

    def __delitem__(self, recipe_id):
        del self.recipes[recipe_id]

    def save_recipe(self, recipe, filename=None, overwrite=False):  # noqa D102. Unused, consider removing
        if not filename:
            filename = recipe.name + ".yaml"
        pathname = os.path.join(self.recipe_dir, filename)
        if os.path.isfile(pathname) and not overwrite:
            raise FileExistsError("Recipe file '{}' already exists".format(pathname))
        yaml.dump(recipe.to_dict(), pathname)

    def open_recipe(self, pathname):
        """Open a recipe file and return a `CompositeRecipe` object.

        Args:
            pathname (str): Full path to a recipe YAML document

        Raises:
            ValueError: if any error occurs reading and loading the recipe

        """
        LOG.debug("Loading composite recipes from {}".format(pathname))
        try:
            for recipe_content in yaml.safe_load_all(pathname):
                name = recipe_content["name"]
                input_layer_ids = [
                    recipe_content["red"]["name"],
                    recipe_content["green"]["name"],
                    recipe_content["blue"]["name"],
                ]
                color_limits = [
                    recipe_content["red"]["color_limit"],
                    recipe_content["green"]["color_limit"],
                    recipe_content["blue"]["color_limit"],
                ]
                gammas = [
                    recipe_content["red"]["gamma"],
                    recipe_content["green"]["gamma"],
                    recipe_content["blue"]["gamma"],
                ]
                recipe = CompositeRecipe(
                    name=name,
                    input_layer_ids=input_layer_ids,
                    color_limits=color_limits,
                    gammas=gammas,
                    read_only=True,
                )
                yield recipe
        except yaml.YAMLError:
            LOG.error("Bad YAML in '{}'".format(pathname))
            raise ValueError("Could not open recipe")
        except (ValueError, KeyError, TypeError):
            LOG.error("Could not add recipes from '{}'".format(pathname))
            raise ValueError("Could not open recipe")
