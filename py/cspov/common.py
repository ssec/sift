#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
.py
~~~

PURPOSE
Support calculations, namedtuples and constants used throughout the library and application.

REFERENCES


REQUIRES
numpy
numba

:author: R.K.Garcia <rayg@ssec.wisc.edu>
:copyright: 2015 by University of Wisconsin Regents, see AUTHORS for more details
:license: GPLv3, see LICENSE for more details
"""
from collections import namedtuple
import numpy as np
from enum import Enum

__author__ = 'rayg'
__docformat__ = 'reStructuredText'

import os, sys
import logging, unittest, argparse
from numba import jit
from pyproj import Proj

LOG = logging.getLogger(__name__)


# http://home.online.no/~sigurdhu/WGS84_Eng.html

DEFAULT_TILE_HEIGHT = 512
DEFAULT_TILE_WIDTH = 512
DEFAULT_TEXTURE_HEIGHT=2
DEFAULT_TEXTURE_WIDTH=16
DEFAULT_ANIMATION_DELAY=1.0/10.0
# The values below are taken from the test geotiffs that are projected to the `DEFAULT_PROJECTION` below.
# These units are in meters in mercator projection space
DEFAULT_X_PIXEL_SIZE = 4891.969810251281160
DEFAULT_Y_PIXEL_SIZE = -7566.684931505724307
DEFAULT_ORIGIN_X = -20037508.342789247632027
DEFAULT_ORIGIN_Y = 15496570.739723727107048

PREFERRED_SCREEN_TO_TEXTURE_RATIO = 0.5  # screenpx:texturepx that we want to keep, ideally, by striding
DEFAULT_PROJECTION = "+proj=merc +datum=WGS84 +ellps=WGS84"
DEFAULT_PROJ_OBJ = p = Proj(DEFAULT_PROJECTION)
C_EQ = p(180, 0)[0] - p(-180, 0)[0]
C_POL = p(0, 89.9)[1] - p(0, -89.9)[1]
MAX_EXCURSION_Y = C_POL/2.0
MAX_EXCURSION_X = C_EQ/2.0

#R_EQ = 6378.1370  # km
#R_POL = 6356.7523142  # km
#C_EQ = 40075.0170  # linear km
#C_POL = 40007.8630  # linear km

# MAX_EXCURSION_Y = C_POL/4.0
# MAX_EXCURSION_X = C_EQ/2.0

box = namedtuple('box', ('b', 'l', 't', 'r'))  # bottom, left, top, right
rez = namedtuple('rez', ('dy', 'dx'))  # world km / pixel distance
pnt = namedtuple('pnt', ('y', 'x'))
geo = namedtuple('geo', ('n', 'e'))  # lat N, lon E
vue = namedtuple('vue', ('b', 'l', 't', 'r', 'dy', 'dx'))  # combination of box + rez

WORLD_EXTENT_BOX = box(b=-MAX_EXCURSION_Y, l=-MAX_EXCURSION_X, t=MAX_EXCURSION_Y, r=MAX_EXCURSION_X)


class KIND(Enum):
    """kind of entities we're working with
    """
    UNKNOWN = 0
    IMAGE = 1
    OUTLINE = 2
    SHAPE = 3
    COMBINATION = 4

class INFO(Enum):
    """
    Standard keys for info dictionaries
    """

    PATHNAME = 'pathname'  # full path to the data file
    NAME = 'name'  # logical name of the file (possibly human assigned)
    KIND = 'kind'  # KIND enumeration on what kind of layer this makes
    UUID = 'uuid'  # UUID assigned on import, which follows the layer around the system
    ORIGIN_X = 'origin_x'
    ORIGIN_Y = 'origin_y'
    CELL_WIDTH = 'cell_width'
    CELL_HEIGHT = 'cell_height'
    PROJ = 'proj4_string'
    CLIM = 'clim'  # (min,max) color map limits
    SHAPE = 'shape' # (rows, columns) or (rows, columns, levels) data shape
    COLORMAP = 'colormap'  # name or UUID of a color map
    DISPLAY_TIME = 'display_time'  # typically from guidebook, used for labeling animation frame

class MercatorTileCalc(object):
    """
    common calculations for mercator tile groups in an array or file
    tiles are identified by (iy,ix) zero-based indicators
    """
    OVERSAMPLED='oversampled'
    UNDERSAMPLED='undersampled'
    WELLSAMPLED='wellsampled'

    name = None
    image_shape = None
    pixel_rez = None
    zero_point = None
    tile_shape = None
    # derived
    image_extents_box = None  # word coordinates that this image and its tiles corresponds to
    tiles_avail = None  # (ny,nx) available tile count for this image

    def __init__(self, name, image_shape, ul_origin, pixel_rez,
                 tile_shape=(DEFAULT_TILE_HEIGHT, DEFAULT_TILE_WIDTH),
                 texture_shape=(DEFAULT_TEXTURE_HEIGHT, DEFAULT_TEXTURE_WIDTH),
                 projection=DEFAULT_PROJECTION,
                 wrap_lon=False):
        """
        name: the 'name' of the tile, typically the path of the file it represents
        image_shape: (h:int,w:int) in pixels
        ul_origin: (y:float,x:float) in world coords specifies upper-left coordinate of the image
        pixel_rez: (dy:float,dx:float) in world coords per pixel ascending from corner [0,0], as measured near zero_point
        tile_shape: the pixel dimensions (h:int, w:int) of the GPU tiling we want to use
        texture_shape: the size of the texture being used (h:int, w:int) in number of tiles

        Tiling is aligned to pixels, not world
        World coordinates are eqm such that 0,0 matches 0°N 0°E, going north/south +-90° and west/east +-180°
        Data coordinates are pixels with b l or b r corner being 0,0
        """
        super(MercatorTileCalc, self).__init__()
        self.name = name
        self.image_shape = image_shape
        self.ul_origin = ul_origin
        self.pixel_rez = pixel_rez
        self.tile_shape = tile_shape
        # in units of tiles:
        self.texture_shape = texture_shape
        # in units of data elements (float32):
        self.texture_size = (self.texture_shape[0] * self.tile_shape[0], self.texture_shape[1] * self.tile_shape[1])
        self.image_tiles_avail = (self.image_shape[0] / self.tile_shape[0], self.image_shape[1] / self.tile_shape[1])
        self.wrap_lon = wrap_lon

        p = Proj(projection)
        # Note: this logic probably only works for mercator or other cylindrical projections
        self.world_extents_box = box(
            b=p(0, -89.9)[1],
            t=p(0, 89.9)[1],
            l=p(-180, 0)[0],
            r=p(180, 0)[0],
        )
        self.image_extents_box = box(
            b=self.ul_origin[0] - self.image_shape[0] * self.pixel_rez.dy,
            t=self.ul_origin[0],
            l=self.ul_origin[1],
            r=self.ul_origin[1] + self.image_shape[1] * self.pixel_rez.dx,
        )

    @jit
    def visible_tiles(self, visible_geom, stride=1, extra_tiles_box=box(0,0,0,0)):
        """
        given a visible world geometry and sampling, return (sampling-state, [box-of-tiles-to-draw])
        sampling state is WELLSAMPLED/OVERSAMPLED/UNDERSAMPLED
        returned box should be iterated per standard start:stop style
        tiles are specified as (iy,ix) integer pairs
        extra_box value says how many extra tiles to include around each edge
        """
        V = visible_geom
        X = extra_tiles_box  # FUTURE: extra_geom_box specifies in world coordinates instead of tile count
        E = self.image_extents_box
        Z = self.pixel_rez

        if self.wrap_lon:
            # FIXME: Technically this doesn't handle reusing the tiles properly, this assumes that the tiles align perfectly
            # make the image look like it covers twice as much of the earth
            # XXX: this only works for global images (not subimages)
            E = box(b=E.b, l=E.l, t=E.t, r=E.r + (E.r - E.l))  # copy the original instance variable

        # convert world coords to pixel coords
        # py0, px0 = self.extents_box.b, self.extents_box.l

        # pixel view b
        pv = box(
            b = (V.b - E.t)/-(Z.dy * stride),
            l = (V.l - E.l)/(Z.dx * stride),
            t = (V.t - E.t)/-(Z.dy * stride),
            r = (V.r - E.l)/(Z.dx * stride)
        )

        # number of tiles wide and high we'll absolutely need
        th,tw = self.tile_shape
        nth = int(np.ceil((pv.b - pv.t) / th)) + 1  # FIXME: is the +1 correct?
        ntw = int(np.ceil((pv.r - pv.l) / tw)) + 1

        # first tile we'll need is (tiy0,tix0)
        tiy0 = int(np.floor(pv.t / th))
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

        # Total number of tiles in this image at this stride
        ath, atw = self.max_tiles_available(stride, self.wrap_lon)
        xth = ath - (tiy0 + nth)
        if xth < 0:  # then we're asking for tiles that don't exist
            nth += xth  # trim it back
        xtw = atw - (tix0 + ntw)
        if xtw < 0:  # likewise with tiles wide
            ntw += xtw

        # FIXME: use vue() instead of box() to represent visible geometry,
        #        so we can estimate sampledness and decide when to re-render
        overunder = None
        # if not isinstance(visible_geom, vue):
        #     overunder = None
        # else:  # use dy/dx to calculate texture pixels to screen pixels ratio
        #     overunder = self.calc_sampling(visible_geom, Z)

        tilebox = box(
            b = int(tiy0 + nth),
            l = int(tix0),
            t = int(tiy0),
            r = int(tix0 + ntw)
        )

        return overunder, tilebox

    @jit
    def max_tiles_available(self, stride, wrap_lon=False):
        ath = np.ceil((self.image_shape[0] / float(stride)) / self.tile_shape[0])
        atw = np.ceil((self.image_shape[1] / float(stride)) / self.tile_shape[1])
        if wrap_lon:
            atw *= 2
        return ath, atw

    @jit
    def calc_sampling(self, visible, stride, texture=None):
        """
        estimate whether we're oversampled, undersampled or well-sampled
        visible.dy, .dx: d(world distance)/d(screen pixels)
        texture.dy, .dx: d(world distance)/d(texture pixels)
        texture pixels / screen pixels = visible / texture
        1:1 is optimal, 2:1 is oversampled, 1:2 is undersampled
        """
        texture = texture or self.pixel_rez
        tsy = visible.dy / (texture.dy * float(stride))
        tsx = visible.dx / (texture.dx * float(stride))
        if min(tsy,tsx) <= 0.5:
            LOG.debug('undersampled tsy,tsx = {0:.2f},{1:.2f}'.format(tsy,tsx))
            return self.UNDERSAMPLED
        if max(tsy,tsx) >= 2.0:
            LOG.debug('oversampled tsy,tsx = {0:.2f},{1:.2f}'.format(tsy,tsx))
            return self.OVERSAMPLED
        return self.WELLSAMPLED

    @jit
    def calc_stride(self, visible, texture=None):
        """
        given world geometry and sampling as a vue or rez tuple
        calculate a conservative stride value for rendering a set of tiles
        :param visible: vue or rez with world pixels per screen pixel
        :param texture: vue or rez with texture resolution as world pixels per screen pixel
        """
        # screen dy,dx in world distance per pixel
        # world distance per pixel for our data
        # compute texture pixels per screen pixels
        texture = texture or self.pixel_rez
        tsy = max(1, np.ceil(visible.dy * PREFERRED_SCREEN_TO_TEXTURE_RATIO / texture.dy))
        tsx = max(1, np.ceil(visible.dx * PREFERRED_SCREEN_TO_TEXTURE_RATIO / texture.dx))
        ts = min(tsy,tsx)
        stride = int(ts)
        return stride

    @jit
    def overview_stride(self):
        # FUTURE: Come up with a fancier way of doing overviews like averaging each strided section, if needed
        tsy = max(1, np.floor(self.image_shape[0] / self.tile_shape[0]))
        tsx = max(1, np.floor(self.image_shape[1] / self.tile_shape[1]))
        y_slice = slice(0, self.image_shape[0], tsy)
        x_slice = slice(0, self.image_shape[1], tsx)
        return y_slice, x_slice

    @jit
    def tile_world_box(self, tiy, tix, ny=1, nx=1):
        """
        return world coordinate box a given tile fills
        """
        LOG.debug('{}y, {}x'.format(tiy,tix))
        eb,el = self.image_extents_box.b, self.image_extents_box.l
        dy,dx = self.pixel_rez
        if dy<0:
            LOG.warning('unexpected dy={}'.format(dy))
        if dx<0:
            LOG.warning('unexpected dx={}'.format(dx))
        th,tw = map(float, self.tile_shape)

        b = eb + dy*(th*tiy)
        t = eb + dy*(th*(tiy+ny))
        l = el + dx*(tw*tix)
        r = el + dx*(tw*(tix+nx))

        if l>=r:
            LOG.warning('left > right')

        return box(b=b,l=l,t=t,r=r)

    @jit
    def tile_slices(self, tiy, tix, stride):
        y_slice = slice(tiy*self.tile_shape[0]*stride, (tiy+1)*self.tile_shape[0]*stride, stride)
        x_slice = slice(tix*self.tile_shape[1]*stride, (tix+1)*self.tile_shape[1]*stride, stride)
        return y_slice, x_slice

    def tile_pixels(self, data, tiy, tix, stride):
        """
        extract pixel data for a given tile
        """
        return data[
               tiy*self.tile_shape[0]:(tiy+1)*self.tile_shape[0]:stride,
               tix*self.tile_shape[1]:(tix+1)*self.tile_shape[1]:stride
               ]

    @jit
    def fractional_wrapped_tile(self, stride):
        """The amount of tile that overlaps at the antimeridian and should be removed from the wrapped tiles.
        """
        # Index of the first
        tix = self.max_tiles_available(stride)[1]
        tile_start_idx = tix * self.tile_shape[1]
        return (tile_start_idx - int(self.image_shape[1] / stride)) / self.tile_shape[1]

    @jit
    def calc_vertex_coordinates(self, tiy, tix, stridey, stridex):
        quad = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0],
                         [0, 0, 0], [1, 1, 0], [0, 1, 0]],
                        dtype=np.float32)
        tile_width = self.pixel_rez.dx * self.tile_shape[1] * stridex
        tile_height = self.pixel_rez.dy * self.tile_shape[0] * stridey
        max_tiles = self.max_tiles_available(stridex)
        virt_tix = tix % max_tiles[1]
        # which image of the repeating/wrapping images are we in
        image_idx = int(tix / max_tiles[1])
        # one whole image in the X direction is this many meters:
        image_origin_x = self.ul_origin.x + self.pixel_rez.dx * self.image_shape[1] * image_idx
        quad[:, 0] *= tile_width
        quad[:, 0] += image_origin_x + (tile_width * virt_tix)
        quad[:, 1] *= -tile_height  # Origin is upper-left so image goes down
        quad[:, 1] += self.ul_origin.y - tile_height * tiy
        quad = quad.reshape(6, 3)
        return quad[:, :2]

    @jit
    def calc_texture_coordinates(self, ttile_idx):
        """Get texture coordinates for one tile as a quad.

        :param ttile_idx: int, texture 1D index that maps to some internal texture tile location
        """
        tiy = int(ttile_idx / self.texture_shape[1])
        tix = ttile_idx % self.texture_shape[1]
        # start with basic quad describing the entire texture
        quad = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0],
                         [0, 0, 0], [1, 1, 0], [0, 1, 0]],
                        dtype=np.float32)
        # Now scale and translate the coordinates so they only apply to one tile in the texture
        one_tile_tex_width = 1.0 / self.texture_size[1] * self.tile_shape[1]
        one_tile_tex_height = 1.0 / self.texture_size[0] * self.tile_shape[0]
        quad[:, 0] *= one_tile_tex_width
        quad[:, 0] += one_tile_tex_width * tix
        quad[:, 1] *= one_tile_tex_height
        quad[:, 1] += one_tile_tex_height * tiy
        quad = quad.reshape(6, 3)
        quad = np.ascontiguousarray(quad[:, :2])
        return quad


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
