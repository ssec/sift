#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LayerRep.py
~~~~~~~~~~~

PURPOSE
Layer representation - the "physical" realization of content to draw on the map.
A layer representation can have multiple levels of detail

A factory will convert URIs into LayerReps
LayerReps are managed by document, and handed off to the MapWidget as part of a LayerDrawingPlan

REFERENCES


REQUIRES


:author: R.K.Garcia <rayg@ssec.wisc.edu>
:copyright: 2014 by University of Wisconsin Regents, see AUTHORS for more details
:license: GPLv3, see LICENSE for more details
"""
import os
import sys
import logging
import unittest
import argparse

import scipy.misc as spm
from PyQt4.QtCore import QObject, pyqtSignal

from cspov.common import pnt, rez, MAX_EXCURSION_Y, MAX_EXCURSION_X, MercatorTileCalc, WORLD_EXTENT_BOX, \
    DEFAULT_TILE_HEIGHT, DEFAULT_TILE_WIDTH, vue
from cspov.view.Program import GlooRGBTile

__author__ = 'rayg'
__docformat__ = 'reStructuredText'

LOG = logging.getLogger(__name__)


class LayerRep(QObject):
    """
    A Layer Representation on the View side of the fence
    - has one or more representations available to immediately draw
    - may want to schedule the rendering of other representations during idle time, to get ideal view
    - may have a backing science representation which is pure science data instead of pixel values or RGBA maps
    - typically will cache a "coarsest" single-tile representation for zoom-out events (preferred for fast=True paint calls)
    - can have probes attached which operate primarily on the science representation
    """
    propertyDidChange = pyqtSignal(dict)
    _z = 0.0
    _alpha = 1.0
    _name = 'unnnamed'

    def __init__(self):
        super(LayerRep, self).__init__()

    def get_z(self):
        return self._z

    def set_z(self, new_z):
        self._z = new_z
        self.propertyDidChange.emit({'z': new_z})

    z = property(get_z, set_z)

    def get_alpha(self):
        return self._alpha

    def set_alpha(self, new_alpha):
        self._alpha = new_alpha
        self.propertyDidChange.emit({'alpha': new_alpha})

    alpha = property(get_alpha, set_alpha)

    def get_name(self):
        return self._name

    def set_name(self, new_name):
        self._name = new_name
        self.propertyDidChange.emit({'name': new_name})

    name = property(get_name, set_name)

    def paint(self, geom, mvp, fast=False, **kwargs):
        """
        draw the most appropriate representation for this layer, given world geometry represented and projection matrix
        if a better representation could be rendered for later draws, return False and render() will be queued for later idle time
        fast flag requests that low-cost rendering be used
        """
        return True

    def render(self, geom, *more_geom):
        """
        cache a rendering (typically a draw-list with textures) that best handles the extents and sampling requested
        if more than one view is active, more geometry may be provided for other views
        return False if resources were too limited and a purge is needed among the layer stack
        :param geom: screen geometry as a vue tuple, world coordinates with d(world)/d(pixel) dy and dx
        """
        return True

    def purge(self, geom, *more_geom):
        """
        release any cached representations that we haven't used lately, leaving at most 1
        :return: True if any GL resources were released
        """
        return False

    def probe_point_xy(self, x, y):
        """
        return a value array for the requested point as specified in mercator-meters
        """
        raise NotImplementedError()

    def probe_point_geo(self, lat, lon):
        """
        """
        raise NotImplementedError()

    def probe_shape(self, geo_shape):
        """
        given a shapely description of an area, return a masked array of data
        """
        raise NotImplementedError()


class MercatorTiffTileLayer(LayerRep):
    """
    A layer with a Mercator TIFF image of world extent
    """
    def __init__(self, pathname, extent=WORLD_EXTENT_BOX):
        self.pathname = pathname


class BackgroundRGBWorldTiles(LayerRep):
    """
    Tile an RGB image representing the full -180..180 longitude, -90..90 latitude
    """
    image = None
    shape = None
    calc = None
    tiles = None  # dictionary of {(y,x): GlooRgbTile, ...}
    _stride = 1

    def set_z(self, z):
        super(BackgroundRGBWorldTiles, self).set_z(z)
        for tile in self.tiles.values():
            tile.z = z

    def set_alpha(self, alpha):
        for tile in self.tiles.values():
            tile.alpha = alpha
        super(BackgroundRGBWorldTiles, self).set_alpha(alpha)

    def __init__(self, model, view, filename=None, world_box=None, tile_shape=None):
        super(BackgroundRGBWorldTiles, self).__init__()
        self.image = spm.imread(filename or 'cspov/data/shadedrelief.jpg')  # FIXME package resource
        self.image = self.image[::-1]  # flip so 0,0 is bottom left instead of top left
        if filename is None:
            tile_shape = (1080,1080)  # FIXME make tile shape smarter
            self.name = 'shadedrelief'
        else:
            self.name = os.path.split(filename)[-1]
        tile_shape = tile_shape or (DEFAULT_TILE_HEIGHT,DEFAULT_TILE_WIDTH)
        self.world_box = world_box or WORLD_EXTENT_BOX
        self.shape = (h,w) = tuple(self.image.shape[:2])
        zero_point = pnt(float(h)/2, float(w)/2)
        pixel_rez = rez(MAX_EXCURSION_Y*2/float(h), MAX_EXCURSION_X*2/float(w))
        self.calc = MercatorTileCalc('bgnd', self.shape, zero_point, pixel_rez, tile_shape)
        self.tiles = {}
        self.model = model
        self.view = view
        # start with a very coarse representation that fits in minimal tiles
        tinyworld = vue(*WORLD_EXTENT_BOX,
                        dy=MAX_EXCURSION_Y*2/DEFAULT_TILE_HEIGHT,
                        dx=MAX_EXCURSION_X*2/DEFAULT_TILE_WIDTH)
        self._generate_tiles(self.calc.calc_stride(tinyworld))
        # self._generate_tiles()

    def paint(self, visible_geom, mvp, fast=False, **kwargs):
        """
        draw the most appropriate representation for this layer
        if a better representation could be rendered for later draws, return False and render() will be queued for later idle time
        fast flag requests that low-cost rendering be used
        """
        for tile in self.tiles.values():  # FIXME: draw only the tiles that are visible in the geom
            # LOG.debug('draw tile {0!r:s}'.format(tile))
            m,v,p = mvp
            tile.set_mvp(m,v,p)
            tile.draw()
        preferred_stride = self.calc.calc_stride(visible_geom)
        return True if preferred_stride != self._stride else False

        # sampling = self.calc.calc_sampling(visible_geom, self._stride)
        # # LOG.debug('sampling is {}'.format(sampling))
        # return False if (sampling is self.calc.WELLSAMPLED) else True

    def render(self, geom, *more_geom):
        "render at a suitable sampling for the screen geometry"
        stride = self.calc.calc_stride(geom)
        self._generate_tiles(stride)

    def _generate_tiles(self, stride=None):
        h,w = self.image.shape[:2]
        _, tilebox = self.calc.visible_tiles(WORLD_EXTENT_BOX)
        # LOG.info(tilebox)
        # for tiy in range(int((tilebox.b+tilebox.t)/2), tilebox.t):  DEBUG
        #     for tix in range(int((tilebox.l+tilebox.r)/2), tilebox.r):
        if stride is not None:
            self._stride = stride
        for tiy in range(tilebox.b, tilebox.t):
            for tix in range(tilebox.l, tilebox.r):
                tilegeom = self.calc.tile_world_box(tiy,tix)
                # if (tilegeom.r+tilegeom.l) < 0 or (tilegeom.b+tilegeom.t) < 0: continue ## DEBUG
                LOG.debug('y:{0} x:{1} geom:{2!r:s}'.format(tiy,tix,tilegeom))
                subim = self.calc.tile_pixels(self.image, tiy, tix, self._stride)
                self.tiles[(tiy,tix)] = t = GlooRGBTile(tilegeom, subim)
                t.set_mvp(model=self.model, view=self.view)





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

