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
import warnings

import numpy as np
from vispy.visuals._scalable_textures import GPUScaledTexture2D

from uwsift.common import DEFAULT_TILE_HEIGHT, DEFAULT_TILE_WIDTH

DEBUG_IMAGE_TILE = bool(os.environ.get("SIFT_DEBUG_TILES", False))

__author__ = "rayg"
__docformat__ = "reStructuredText"

LOG = logging.getLogger(__name__)


class TextureAtlas2D(GPUScaledTexture2D):
    """A 2D Texture Array structure implemented as a 2D Texture Atlas."""

    def __init__(self, texture_shape, tile_shape=(DEFAULT_TILE_HEIGHT, DEFAULT_TILE_WIDTH), **texture_kwargs):
        # Number of tiles in each direction (y, x)
        self.texture_shape = self._check_texture_shape(texture_shape)
        # Number of rows and columns for each tile
        self.tile_shape = tile_shape
        # Number of rows and columns to hold all of these tiles in one texture
        shape = (self.texture_shape[0] * self.tile_shape[0], self.texture_shape[1] * self.tile_shape[1])
        if len(tile_shape) == 3:
            shape = (shape[0], shape[1], tile_shape[2])
        self.texture_size = shape
        self._fill_array = np.tile(np.float32(np.nan), self.tile_shape)
        # create a representative array so the texture can be initialized properly with the right dtype
        rep_arr = (
            np.zeros((10, 10, tile_shape[2]), dtype=np.float32)
            if len(tile_shape) == 3
            else np.zeros((10, 10), dtype=np.float32)
        )
        # will add self.shape:
        super(TextureAtlas2D, self).__init__(data=rep_arr, **texture_kwargs)
        # GPUScaledTexture2D always uses a "representative" size
        # we need to force the shape to our final size so we can start setting tiles right away
        self._resize(shape)

    def _check_texture_shape(self, texture_shape):
        if isinstance(texture_shape, tuple):
            if len(texture_shape) != 2:
                raise ValueError("A shape tuple must be two elements.")
            texture_shape = texture_shape
        else:
            texture_shape = texture_shape.shape
        return texture_shape

    def _tex_offset(self, idx):
        """Return the X, Y texture index offset for the 1D tile index.

        This class presents a 1D indexing scheme, but internally can hold multiple tiles in both X and Y direction.
        """
        row = int(idx / self.texture_shape[1])
        col = idx % self.texture_shape[1]
        return row * self.tile_shape[0], col * self.tile_shape[1]

    def set_tile_data(self, tile_idx, data, copy=False):
        """Write a single tile of data into the texture."""
        offset = self._tex_offset(tile_idx)
        if data is None:
            # Special "fill" parameter
            data = self._fill_array
        else:
            # FIXME: Doesn't this always return the shape of the input data?
            tile_offset = (min(self.tile_shape[0], data.shape[0]), min(self.tile_shape[1], data.shape[1]))
            if tile_offset[0] < self.tile_shape[0] or tile_offset[1] < self.tile_shape[1]:
                # FIXME: This should be handled by the caller to expand the array to be NaN filled and aligned
                # Assign a fill value, make sure to copy the data so that we don't overwrite the original
                data_orig = data
                data = np.zeros(self.tile_shape, dtype=data.dtype)
                # data = data.copy()
                data[:] = np.nan
                data[: tile_offset[0], : tile_offset[1]] = data_orig[: tile_offset[0], : tile_offset[1]]
        if DEBUG_IMAGE_TILE:
            data[:5, :] = 1000.0
            data[-5:, :] = 1000.0
            data[:, :5] = 1000.0
            data[:, -5:] = 1000.0
        super(TextureAtlas2D, self).scale_and_set_data(data, offset=offset, copy=copy)


class MultiChannelGPUScaledTexture2D:
    """Wrapper class around individual textures.

    This helper class allows for easier handling of multiple textures that
    represent individual R, G, and B channels of an image.

    """

    _singular_texture_class = GPUScaledTexture2D
    _ndim = 2

    def __init__(self, data, **texture_kwargs):
        # data to sent to texture when not being used
        self._fill_arr = np.full((10, 10), np.float32(np.nan), dtype=np.float32)

        self.num_channels = len(data)
        data = [x if x is not None else self._fill_arr for x in data]
        self._textures = self._create_textures(self.num_channels, data, **texture_kwargs)

    def _create_textures(self, num_channels, data, **texture_kwargs):
        return [self._singular_texture_class(data[i], **texture_kwargs) for i in range(num_channels)]

    @property
    def textures(self):
        return self._textures

    @property
    def clim(self):
        """Get color limits used when rendering the image (cmin, cmax)."""
        return tuple(t.clim for t in self._textures)

    def set_clim(self, clim):
        if isinstance(clim, str) or len(clim) == 2:
            clim = [clim] * self.num_channels

        need_tex_upload = False
        for tex, single_clim in zip(self._textures, clim):
            if single_clim is None or single_clim[0] is None:
                single_clim = (0, 0)  # let VisPy decide what to do with unusable clims
            if tex.set_clim(single_clim):
                need_tex_upload = True
        return need_tex_upload

    @property
    def clim_normalized(self):
        return tuple(tex.clim_normalized for tex in self._textures)

    @property
    def internalformat(self):
        return self._textures[0].internalformat

    @internalformat.setter
    def internalformat(self, value):
        for tex in self._textures:
            tex.internalformat = value

    @property
    def interpolation(self):
        return self._textures[0].interpolation

    @interpolation.setter
    def interpolation(self, value):
        for _ in self._textures:
            self._texture.interpolation = value

    def check_data_format(self, data_arrays):
        if len(data_arrays) != self.num_channels:
            raise ValueError(f"Expected {self.num_channels} number of channels, got {len(data_arrays)}.")
        for tex, data in zip(self._textures, data_arrays):
            if data is not None:
                tex.check_data_format(data)

    def scale_and_set_data(self, data, offset=None, copy=False):
        """Scale and set data for one or all sub-textures.

        Parameters
        ----------
        data : list | ndarray
            Texture data in the form of a numpy array or as a list of numpy
            arrays. If a list is provided then it must be the same length as
            ``num_channels`` for this texture. If a numpy array is provided
            then ``offset`` should also be provided with the first value
            representing which sub-texture to update. For example,
            ``offset=(1, 0, 0)`` would update the entire the second (index 1)
            sub-texture with an offset of ``(0, 0)``. The list can also contain
            ``None`` to not update the sub-texture at that index.
        offset: tuple | None
            Offset into the texture where to write the provided data. If
            ``None`` then data will be written with no offset (0). If
            provided as a 2-element tuple then that offset will be used
            for all sub-textures. If a 3-element tuple then the first offset
            index represents the sub-texture to update.

        """
        is_multi = isinstance(data, (list, tuple))
        index_provided = offset is not None and len(offset) == self._ndim + 1
        if not is_multi and not index_provided:
            raise ValueError(
                "Setting texture data for a single sub-texture "
                "requires 'offset' to be passed with the first "
                "element specifying the sub-texture index."
            )
        elif is_multi and index_provided:
            warnings.warn(
                "Multiple texture arrays were passed, but so was "
                "sub-texture index in 'offset'. Ignoring that index.",
                UserWarning,
                stacklevel=4,
            )
            offset = offset[1:]
        if is_multi and len(data) != self.num_channels:
            raise ValueError(
                "Multiple provided arrays must match number of channels. "
                f"Got {len(data)}, expected {self.num_channels}."
            )

        if offset is not None and len(offset) == self._ndim + 1:
            tex_indexes = offset[:1]
            offset = offset[1:]
            data = [data]
        else:
            tex_indexes = range(self.num_channels)

        for tex_idx, _data in zip(tex_indexes, data):
            if _data is None:
                _data = self._fill_arr
            self._textures[tex_idx].scale_and_set_data(_data, offset=offset, copy=copy)


class MultiChannelTextureAtlas2D(MultiChannelGPUScaledTexture2D):
    """Helper texture for working with RGB images in SIFT."""

    _singular_texture_class = TextureAtlas2D

    def set_tile_data(self, tile_idx, data_arrays, copy=False):
        for idx, data in enumerate(data_arrays):
            self._textures[idx].set_tile_data(tile_idx, data, copy=copy)
