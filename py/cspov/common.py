#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
.py
~~~

PURPOSE


REFERENCES


REQUIRES


:author: R.K.Garcia <rayg@ssec.wisc.edu>
:copyright: 2014 by University of Wisconsin Regents, see AUTHORS for more details
:license: GPLv3, see LICENSE for more details
"""
from collections import namedtuple
import numpy as np

__author__ = 'rayg'
__docformat__ = 'reStructuredText'

import os, sys
import logging, unittest, argparse
from numba import autojit

LOG = logging.getLogger(__name__)


# http://home.online.no/~sigurdhu/WGS84_Eng.html

DEFAULT_TILE_HEIGHT = 512
DEFAULT_TILE_WIDTH = 512

R_EQ = 6378.1370  # km
R_POL = 6356.7523142  # km
C_EQ = 40075.0170  # linear km
C_POL = 40007.8630  # linear km

MAX_EXCURSION_Y = C_POL/4.0
MAX_EXCURSION_X = C_EQ/2.0

box = namedtuple('box', ('b', 'l', 't', 'r'))  # bottom, left, top, right
rez = namedtuple('rez', ('dy', 'dx'))
pnt = namedtuple('pnt', ('y', 'x'))
geo = namedtuple('geo', ('n', 'e'))  # lat N, lon E
vue = namedtuple('vue', ('b', 'l', 't', 'r', 'dy', 'dx'))

WORLD_EXTENT_BOX = box(b=-MAX_EXCURSION_Y, l=-MAX_EXCURSION_X, t=MAX_EXCURSION_Y, r=MAX_EXCURSION_X)




class MercatorTileCalc(object):
    """
    common calculations for mercator tile groups in an array or file
    tiles are identified by (iy,ix) zero-based indicators
    """
    OVERSAMPLED=1
    UNDERSAMPLED=-1
    WELLSAMPLED=0

    name = None
    pixel_shape = None
    pixel_rez = None
    zero_point = None
    tile_shape = None
    # derived
    extents_box = None  # word coordinates that this image and its tiles corresponds to
    tiles_avail = None  # (ny,nx) available tile count for this image

    def __init__(self, name, pixel_shape, zero_point, pixel_rez, tile_shape=(DEFAULT_TILE_HEIGHT, DEFAULT_TILE_WIDTH)):
        """
        name: the 'name' of the tile, typically the path of the file it represents
        pixel_shape: (h:int,w:int) in pixels
        zero_point: (y:float,x:float) in pixels that represents world coords 0N,0E eqm, even if outside the image and even if fractional
        pixel_rez: (dy:float,dx:float) in world coords per pixel ascending from corner [0,0], as measured near zero_point
        tile_shape: the pixel dimensions (h:int, w:int) of the GPU tiling we want to use

        Tiling is aligned to pixels, not world
        World coordinates are eqm such that 0,0 matches 0°N 0°E, going north/south +-90° and west/east +-180°
        Data coordinates are pixels with b l or b r corner being 0,0
        """
        super(MercatorTileCalc, self).__init__()
        self.name = name
        self.pixel_shape = pixel_shape
        self.zero_point = zero_point
        self.pixel_rez = pixel_rez
        self.tile_shape = tile_shape

        assert(pixel_rez.dy > 0.0)        # FIXME: what if pixel_rez.dy < 0? can we handle this reliably?
        assert(pixel_rez.dx > 0.0)

        h,w = pixel_shape
        zy,zx = zero_point
        # below < 0, above >0
        # h = above - below
        # zy + above = h
        # below = -zy
        pxbelow = float(-zy)
        pxabove = float(h) - float(zy)
        # r > 0, l < 0
        # w = r - l
        # zx + r = w
        # l = -zx
        pxright = float(w) - float(zx)
        pxleft = float(-zx)

        self.extents_box = box(
            b = pxbelow * pixel_rez.dy,
            t = pxabove * pixel_rez.dy,
            l = pxleft * pixel_rez.dx,
            r = pxright * pixel_rez.dx
        )

        self.tiles_avail = (h/tile_shape[0], w/tile_shape[1])

        # FIXME: deal with the AHI seeing the dateline and not the prime meridian!
        # FIXME: for now, require image size to be a multiple of tile size, else we have to deal with partial tiles!
        assert(h % tile_shape[0]==0)
        assert(w % tile_shape[1]==0)

    @autojit
    def visible_tiles(self, visible_geom, extra_tiles_box=box(0,0,0,0)):
        """
        given a visible world geometry and sampling, return (sampling-state, [box-of-tiles-to-draw])
        sampling state is WELLSAMPLED/OVERSAMPLED/UNDERSAMPLED
        returned box should be iterated per standard start:stop style
        tiles are specified as (iy,ix) integer pairs
        extra_box value says how many extra tiles to include around each edge
        """
        V = visible_geom
        X = extra_tiles_box  # FUTURE: extra_geom_box specifies in world coordinates instead of tile count
        E = self.extents_box
        Z = self.pixel_rez

        # convert world coords to pixel coords
        py0, px0 = self.extents_box.b, self.extents_box.l

        # pixel view b
        pv = box(
            b = (V.b - E.b)/Z.dy,
            l = (V.l - E.l)/Z.dx,
            t = (V.t - E.b)/Z.dy,
            r = (V.r - E.l)/Z.dx
        )

        # number of tiles wide and high we'll absolutely need
        th,tw = self.tile_shape
        nth = int(np.ceil((pv.t - pv.b) / th))
        ntw = int(np.ceil((pv.r - pv.l) / tw))

        # first tile we'll need is (tiy0,tix0)
        tiy0 = int(np.floor(pv.b / th))
        tix0 = int(np.floor(pv.l / tw))

        # now add the extras
        if X.b>0:
            tiy0 -= int(X.b)
            nth += int(X.b)
        if X.l>0:
            tix0 -= int(X.l)
            ntw += int(X.l)
        if X.t>0:
            nth += int(X.t)
        if X.r>0:
            ntw += int(X.r)

        # truncate to the available tiles
        if tix0<0:
            ntw += tix0
            tix0 = 0
        if tiy0<0:
            nth += tiy0
            tiy0 = 0

        ath,atw = self.tiles_avail
        xth = ath - (tiy0 + nth)
        if xth < 0:  # then we're asking for tiles that don't exist
            nth += xth  # trim it back
        xtw = atw - (tix0 + ntw)
        if xtw < 0:  # likewise with tiles wide
            ntw += xtw

        # FIXME: compare visible dx/dy versus tile dx/dy to determine over/undersampledness
        overunder = self.WELLSAMPLED

        tilebox = box(
            b = int(tiy0),
            l = int(tix0),
            t = int(tiy0 + nth),
            r = int(tix0 + ntw)
        )

        return overunder, tilebox

    @autojit
    def tile_world_box(self, tiy, tix, ny=1, nx=1):
        """
        return world coordinate box a given tile fills
        """
        LOG.debug('{}y, {}x'.format(tiy,tix))
        eb,el = self.extents_box.b, self.extents_box.l
        dy,dx = self.pixel_rez
        th,tw = map(float, self.tile_shape)

        b = eb + dy*(th*tiy)
        t = eb + dy*(th*(tiy+ny))
        l = el + dx*(tw*tix)
        r = el + dx*(tw*(tix+nx))

        return box(b=b,l=l,t=t,r=r)


    def tile_pixels(self, data, tiy, tix):
        """
        extract pixel data for a given tile
        """
        return data[
               tiy*self.tile_shape[0]:(tiy+1)*self.tile_shape[0],
               tix*self.tile_shape[1]:(tix+1)*self.tile_shape[1]
               ]


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
