#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
shapes.py
~~~~~~~~~

PURPOSE
Shape datasets which can be represented in the workspace as data content masks

REFERENCES


REQUIRES


:author: R.K.Garcia <rayg@ssec.wisc.edu>
:copyright: 2014 by University of Wisconsin Regents, see AUTHORS for more details
:license: GPLv3, see LICENSE for more details
"""
__author__ = "rayg"
__docformat__ = "reStructuredText"

import logging

import numpy as np
import shapely.geometry.polygon as sgp
from rasterio import Affine
from rasterio.features import rasterize

LOG = logging.getLogger(__name__)


def content_within_shape(content: np.ndarray, trans: Affine, shape: sgp.LinearRing):
    """

    :param content: data being displayed on the screen
    :param trans: affine transform between content array indices and screen coordinates
    :param shape: LinearRing in screen coordinates (e.g. mercator meters)
    :return: masked_content:masked_array, (y_index_offset:int, x_index_offset:int)
        containing minified masked content array

    """
    # Get the bounds so we can limit how big our rasterize boolean array actually is
    inv_trans = ~trans
    # convert bounding box to content coordinates
    # (0, 0) image index is upper-left origin of data (needs more work if otherwise)
    nx, ny, mx, my = shape.bounds  # minx,miny,maxx,maxy
    nx, my = inv_trans * (nx, my)
    mx, ny = inv_trans * (mx, ny)
    nx, my = int(nx), int(my)
    mx, ny = int(np.ceil(mx)), int(np.ceil(ny))

    # subset the content (ny is the higher *index*, my is the lower *index*)
    w = (mx - nx) + 1
    h = (ny - my) + 1

    # Make our linear ring a properly oriented shapely polygon
    shape = sgp.Polygon(shape)
    shape = sgp.orient(shape)
    # create a transform that is shifted to where the polygon is
    offset_trans = trans * Affine.translation(nx, my)

    # Get boolean mask for where the polygon is and get an index mask of those positions
    index_mask = np.nonzero(
        rasterize([shape], out_shape=(h, w), transform=offset_trans, default_value=1).astype(np.bool_)
    )
    # translate the mask indexes back to the original data array coordinates (original index mask is read-only)
    index_mask = (index_mask[0] + my, index_mask[1] + nx)
    return index_mask, content[index_mask]
