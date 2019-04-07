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

import logging
import os
from collections import namedtuple
from glob import glob

import yaml

from uwsift.util.default_paths import DOCUMENT_SETTINGS_DIR

LOG = logging.getLogger(__name__)

_CompositeRecipe = namedtuple('CompositeRecipe', ['name', 'input_ids', 'color_limits', 'gammas', 'read_only'])


class CompositeRecipe(_CompositeRecipe):
    def __new__(cls, name, input_ids=None, color_limits=None, gammas=None, read_only=False):
        def _normalize_list(x, default=None):
            return [x[idx] if x and len(x) > idx and x[idx] else default for idx in range(3)]

        input_ids = _normalize_list(input_ids)
        color_limits = _normalize_list(color_limits, (None, None))
        gammas = _normalize_list(gammas, 1.0)
        return super(CompositeRecipe, cls).__new__(cls, name, input_ids, color_limits, gammas, read_only)

    @classmethod
    def from_rgb(cls, name, r=None, g=None, b=None, color_limits=None, gammas=None):
        return cls(name, input_ids=[r, g, b], color_limits=color_limits, gammas=gammas)

    def _channel_info(self, idx):
        return {
            'name': self.input_ids[idx],
            'color_limits': self.color_limits[idx],
            'gamma': self.gammas[idx],
        }

    def set_default_color_limits(self, r=None, g=None, b=None):
        """Set color limits based on dependency limits"""
        if self.read_only:
            raise RuntimeError("Composite recipe is read only")
        for idx, comp in enumerate([r, g, b]):
            if comp is None:
                # component was not updated
                continue
            if self.input_ids[idx] is None:
                # our component is None
                self.color_limits[idx] = (None, None)
            else:
                self.color_limits[idx] = comp

    @property
    def red(self):
        return self._channel_info(0)

    @property
    def green(self):
        return self._channel_info(1)

    @property
    def blue(self):
        return self._channel_info(2)

    def to_dict(self):
        """Convert to YAML-compatible dict."""
        return {
            'name': self.name,
            'red': self.red,
            'green': self.green,
            'blue': self.blue,
        }

    def copy(self, new_name):
        """Get a copy of this recipe with a new name"""
        return self._replace(name=new_name)


class RecipeManager(object):
    def __init__(self, config_dir=None):
        if config_dir is None:
            config_dir = DOCUMENT_SETTINGS_DIR

        recipe_dir = os.path.join(config_dir, 'composite_recipes')
        if not os.path.isdir(recipe_dir):
            LOG.info("creating new composite recipes directory at {}".format(recipe_dir))
            os.makedirs(recipe_dir)
        self.recipe_dir = recipe_dir
        # recipe_name -> recipe object
        self.recipes = {}
        # recipe_name -> (filename, recipe object)
        self._stored_recipes = {}
        self.load_available_recipes()

    def load_available_recipes(self):
        """Load recipes from stored config files"""
        for pathname in glob(os.path.join(self.recipe_dir, '*')):
            if not (pathname.endswith('.yml') or pathname.endswith('.yaml')):
                continue
            recipe = self.open_recipe(pathname)
            self._stored_recipes[recipe.name] = (pathname, recipe)

    def add_recipe(self, recipe):
        self.recipes[recipe.name] = recipe

    def __getitem__(self, recipe_name):
        return self.recipes[recipe_name]

    def __delitem__(self, recipe_name):
        del self.recipes[recipe_name]

    def save_recipe(self, recipe, filename=None, overwrite=False):
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
                name = recipe_content['name']
                input_ids = [
                    recipe_content['red']['name'],
                    recipe_content['green']['name'],
                    recipe_content['blue']['name'],
                ]
                color_limits = [
                    recipe_content['red']['color_limit'],
                    recipe_content['green']['color_limit'],
                    recipe_content['blue']['color_limit'],
                ]
                gammas = [
                    recipe_content['red']['gamma'],
                    recipe_content['green']['gamma'],
                    recipe_content['blue']['gamma'],
                ]
                recipe = CompositeRecipe(
                    name=name,
                    input_ids=input_ids,
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
