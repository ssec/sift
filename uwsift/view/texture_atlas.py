#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
.py
~~~

PURPOSE
gloo.Program wrappers for different purposes such as tile drawing

REFERENCES


REQUIRES


:author: R.K.Garcia <rayg@ssec.wisc.edu>
:copyright: 2015 by University of Wisconsin Regents, see AUTHORS for more details
:license: GPLv3, see LICENSE for more details
"""

import logging
import os

import numpy as np
from vispy.gloo import Texture2D

from uwsift.common import DEFAULT_TILE_HEIGHT, DEFAULT_TILE_WIDTH

DEBUG_IMAGE_TILE = bool(os.environ.get("SIFT_DEBUG_TILES", False))

__author__ = 'rayg'
__docformat__ = 'reStructuredText'

LOG = logging.getLogger(__name__)


class TextureAtlas2D(Texture2D):
    """A 2D Texture Array structure implemented as a 2D Texture Atlas.
    """

    def __init__(self, texture_shape, tile_shape=(DEFAULT_TILE_HEIGHT, DEFAULT_TILE_WIDTH),
                 format=None, resizable=True,
                 interpolation=None, wrapping=None,
                 internalformat=None, resizeable=None):
        assert len(texture_shape) == 2
        # Number of tiles in each direction (y, x)
        self.texture_shape = texture_shape
        # Number of rows and columns for each tile
        self.tile_shape = tile_shape
        # Number of rows and columns to hold all of these tiles in one texture
        shape = (self.texture_shape[0] * self.tile_shape[0], self.texture_shape[1] * self.tile_shape[1])
        self.texture_size = shape
        self._fill_array = np.tile(np.nan, self.tile_shape).astype(np.float32)
        # will add self.shape:
        super(TextureAtlas2D, self).__init__(None, format, resizable, interpolation,
                                             wrapping, shape, internalformat, resizeable)

    def _tex_offset(self, idx):
        """Return the X, Y texture index offset for the 1D tile index.

        This class presents a 1D indexing scheme, but internally can hold multiple tiles in both X and Y direction.
        """
        row = int(idx / self.texture_shape[1])
        col = idx % self.texture_shape[1]
        return row * self.tile_shape[0], col * self.tile_shape[1]

    def set_tile_data(self, tile_idx, data, copy=False):
        """Write a single tile of data into the texture.
        """
        offset = self._tex_offset(tile_idx)
        if data is None:
            # Special "fill" parameter
            data = self._fill_array
        else:
            # FIXME: Doesn't this always return the shape of the input data?
            tile_offset = (min(self.tile_shape[0], data.shape[0]),
                           min(self.tile_shape[1], data.shape[1]))
            if tile_offset[0] < self.tile_shape[0] or tile_offset[1] < self.tile_shape[1]:
                # FIXME: This should be handled by the caller to expand the array to be NaN filled and aligned
                # Assign a fill value, make sure to copy the data so that we don't overwrite the original
                data_orig = data
                data = np.zeros(self.tile_shape, dtype=data.dtype)
                # data = data.copy()
                data[:] = np.nan
                data[:tile_offset[0], :tile_offset[1]] = data_orig[:tile_offset[0], :tile_offset[1]]
        if DEBUG_IMAGE_TILE:
            data[:5, :] = 1000.
            data[-5:, :] = 1000.
            data[:, :5] = 1000.
            data[:, -5:] = 1000.
        super(TextureAtlas2D, self).set_data(data, offset=offset, copy=copy)
