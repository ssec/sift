#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
shapes.py
~~~~~~~~~

PURPOSE
Shape layers which can be represented in the workspace as data content masks

REFERENCES


REQUIRES


:author: R.K.Garcia <rayg@ssec.wisc.edu>
:copyright: 2014 by University of Wisconsin Regents, see AUTHORS for more details
:license: GPLv3, see LICENSE for more details
"""
__author__ = 'rayg'
__docformat__ = 'reStructuredText'

import os, sys
import logging, unittest, argparse
from functools import partial
import shapely.geometry.polygon as sgp, shapely.ops as sops, shapely.geometry as sgeo
# import shapefile as shf
import numpy as np
import pyproj as prj
from numba import jit

LOG = logging.getLogger(__name__)

def convert_shape_to_proj(to_proj:prj.Proj, shape:sgp.LinearRing, shape_proj:prj.Proj):
    # ref http://toblerity.org/shapely/manual.html#shapely.ops.transform
    # ref http://all-geo.org/volcan01010/2012/11/change-coordinates-with-pyproj/
    # inverse-project the ring
    if to_proj is shape_proj:
        return shape
    project = partial(
        prj.transform,
        shape_proj,
        to_proj)
    newshape = sops.transform(project, shape)
    return newshape

@jit
def mask_inside_index_shape(xoff, yoff, width, height, shape):
    """
    for an index shape, return a mask of coordinates inside the shape
    :param xoff: x offset between mask and shape
    :param yoff: y offset
    :param xmax: maximum x index 0-(xmax-1)
    :param ymax: max y index
    :param shape:
    :return:
    """
    mask = np.zeros((height,width), dtype=np.bool_)
    for y in range(height):
        for x in range(width):
            p = sgeo.Point(x+xoff,y+yoff)
            if p.within(shape):
                mask[y,x] = True
    return mask

def content_within_shape(content:np.ndarray, display_xform:prj.Proj, shape:sgp.LinearRing):
    """

    :param content: data being displayed on the screen
    :param display_proj: transform between content array indices and screen coordinates
    :param shape: LinearRing in screen coordinates (e.g. mercator meters)
    :return: masked_content:masked_array, (y_index_offset:int, x_index_offset:int) containing minified masked content array
    """
    # invert shape to content indices
    invproject = partial(
        display_xform,
        inverse=True
    )
    # FIXME: deal with dateline crossing
    # convert shape from display coords to content array indices
    index_shape = sops.transform(invproject, shape)

    # now convert bounding box to content coordinates
    nx,ny,mx,my = shape.bounds  # minx,miny,maxx,maxy
    nx, ny = display_xform(nx,ny, inverse=True)
    mx, my = display_xform(mx,my, inverse=True)
    nx, ny = int(nx), int(ny)
    mx, my = int(np.ceil(mx)), int(np.ceil(my))

    # subset the content
    w = (mx-nx)+1
    h = (my-ny)+1

    # generate index arrays
    mask = mask_inside_index_shape(nx, ny, w, h, index_shape)

    masked_content = np.ma.masked_array(content[ny:ny+h, nx:nx+w], mask=mask)
    return masked_content


def original_data_within_shape(raw_data:np.ndarray,
                               content_proj:prj.Proj,
                               display_xform:prj.Proj,
                               shape:sgp.LinearRing):
    """
    :param raw_data: ndarray, unprojected data (e.g. lat/lon instead of XY space)
    :param content_proj: projection applied to raw_data to make projected content array
    :param display_xform: transformation applied to content to make screen coordinates
    :param shape: shape we're working with, in display coordinates
    :return:
    """
    #
    #
    raise NotImplementedError('incomplete implementation')
    # convert shape to coordinate system of content
    if shape_proj is not None:
        shape = convert_shape_to_proj(content_proj, shape, shape_proj)

    # generate bounding box for the shape
    nx,ny,mx,my = shape.bounds  # minx,miny,maxx,maxy

    # reverse the bounding box corners projection to array indices as subbox iy,ix,w,h
    nx,ny = content_proj(nx,ny, inverse=True)
    mx,my = content_proj(mx,my, inverse=True)

    # generate "subbox" projected x,y coordinate arrays for that bounded area of the content


    # test inside/outside the shape for all the coordinates inside the subbox



def main():
    parser = argparse.ArgumentParser(
        description="PURPOSE",
        epilog="",
        fromfile_prefix_chars='@')
    parser.add_argument('-v', '--verbose', dest='verbosity', action="count", default=0,
                        help='each occurrence increases verbosity 1 level through ERROR-WARNING-INFO-DEBUG')
    # http://docs.python.org/2.7/library/argparse.html#nargs
    # parser.add_argument('--stuff', nargs='5', dest='my_stuff',
    #                    help="one or more random things")
    parser.add_argument('pos_args', nargs='*',
                        help="positional arguments don't have the '-' prefix")
    args = parser.parse_args()

    levels = [logging.ERROR, logging.WARN, logging.INFO, logging.DEBUG]
    logging.basicConfig(level=levels[min(3, args.verbosity)])

    if not args.pos_args:
        unittest.main()
        return 0

    for pn in args.pos_args:
        pass

    return 0


if __name__ == '__main__':
    sys.exit(main())
