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
import os, sys
import logging, unittest, argparse
from OpenGL.GL import *
from PIL import Image
import numpy as np

import scipy.misc as spm
from cspov.common import pnt, rez, MAX_EXCURSION_Y, MAX_EXCURSION_X, MercatorTileCalc, WORLD_EXTENT_BOX, \
    DEFAULT_TILE_HEIGHT, DEFAULT_TILE_WIDTH, box
from cspov.view.Program import GlooRGBTile

__author__ = 'rayg'
__docformat__ = 'reStructuredText'


LOG = logging.getLogger(__name__)



class Layer(object):
    """
    A Layer
    - has one or more representations available to immediately draw
    - may want to schedule the rendering of other representations during idle time, to get ideal view
    - may have a backing science representation which is pure science data instead of pixel values or RGBA maps
    - typically will cache a "coarsest" single-tile representation for zoom-out events (preferred for fast=True paint calls)
    - can have probes attached which operate primarily on the science representation
    """
    def paint(self, geom, proj, fast=False, **kwargs):
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
        """
        return True

    def purge(self, geom, *more_geom):
        """
        release any cached representations that we haven't used lately, leaving at most 1
        return True if any GL resources were released
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


TEST_COLORS = [
    (1.,1.,1.),
    (0.1,0.1,0.1),
    (0.,0.,1.),
    (0.,1.,0.),
    (0.,1.,1.),
    (1.,0.,0.),
    (1.,0.,1.),
    (1.,1.,0.),
]


class MercatorTiffTileLayer(Layer):
    """
    A layer with a Mercator TIFF image of world extent
    """
    def __init__(self, pathname, extent=WORLD_EXTENT_BOX):
        self.pathname = pathname


class BackgroundRGBWorldTiles(Layer):
    """
    Tile an RGB image representing the full -180..180 longitude, -90..90 latitude
    """
    image = None
    shape = None
    calc = None
    tiles = None  # dictionary of {(y,x): GlooRgbTile, ...}

    def __init__(self, model, view, filename=None, world_box=None, tile_shape=None):
        self.image = spm.imread(filename or 'cspov/data/shadedrelief.jpg')  # FIXME package resource
        if filename is None:
            tile_shape = (1080,1080)  # FIXME make tile shape smarter
        self.world_box = world_box or WORLD_EXTENT_BOX
        self.shape = (h,w) = tuple(self.image.shape[:2])
        zero_point = pnt(float(h)/2, float(w)/2)
        pixel_rez = rez(180.0/float(h), 360.0/float(w))
        self.calc = MercatorTileCalc('bgns', self.shape, zero_point, pixel_rez, tile_shape)
        self.tiles = {}
        self.model = model
        self.view = view
        self._generate_tiles()

    def paint(self, geom, proj, fast=False, **kwargs):
        """
        draw the most appropriate representation for this layer
        if a better representation could be rendered for later draws, return False and render() will be queued for later idle time
        fast flag requests that low-cost rendering be used
        """
        # tile = self.tiles[(2,2)]
        # tile.set_mvp(projection=proj)
        # tile.draw()
        # return True

        for tile in self.tiles.values():  # FIXME: draw only the tiles that are visible in the geom
            # LOG.debug('draw tile {0!r:s}'.format(tile))
            tile.set_mvp(projection=proj)
            tile.draw()
        return True

    def _generate_tiles(self):
        h,w = self.image.shape[:2]
        _, tilebox = self.calc.visible_tiles(WORLD_EXTENT_BOX)
        LOG.info(tilebox)
        for tiy in range(tilebox.b, tilebox.t):
            for tix in range(tilebox.l, tilebox.r):
                tilegeom = self.calc.tile_world_box(tiy,tix)
                subim = self.calc.tile_pixels(self.image, tiy, tix)
                self.tiles[(tiy,tix)] = t = GlooRGBTile(tilegeom, subim)
                t.set_mvp(model=self.model, view=self.view)




class FlatFileTileSet(object):
    """
    A lazy-loaded image which can be mapped to a texture buffer and drawn on a polygon
    Represents a single x-y coordinate range at a single level of detail
    Will map data into GL texture memory for rendering

    Tiles have
    - extents in the mercator-meters space
    - a texture map
    - later: a shader which may be shared
    """
    data = None
    path = None
    _active = None     # {(y,x): (texture-id, texture-box), ...}
    _calc = None   # calculator

    def __init__(self, path, element_dtype, shape, zero_point, pixel_rez, tile_shape=(
    DEFAULT_TILE_HEIGHT, DEFAULT_TILE_WIDTH)):
        """
        map the file as read-only
        chop it into data tiles
        tiles are set up such that tile (0,0) starts with zero_point and goes up and to the r

        """
        super(FlatFileTileSet, self).__init__()
        self.data = np.memmap(path, element_dtype, 'r', shape=shape)
        self._calc = MercatorTileCalc(name=path, pixel_shape=shape, zero_point=zero_point, pixel_rez=pixel_rez, tile_shape=tile_shape)
        self.path = path
        self._active = {}


    def __getitem__(self, tileyx):
        try:
            return self._active[tuple(tileyx)]
        except KeyError as unavailable:
            return self.activate(tileyx)


    def activate(self, tile_yx, ufunc=None):
        """
        load a texture map from this data, optionally filtering data with a numpy ufunc
        return (texture-id, box)
        """
        # FUTURE: implement using glGenBuffers() and use a shader to render
        # We want to push buffers of data into GL, and use GLSL to determine the color/transparency of the data.
        # See VisPy for tools and techniques.
        # For now, alpha test with plain old textures

        texid = glGenTextures(1)

        # offset within the array
        th,tw = self._calc.tile_shape
        # tile_y, tile_x = tile_yx
        # ys = (tile_y*th)*self._calc.zero_point[0]
        # xs = (tile_x*tw)*self._calc.zero_point[1]
        # ye = ys + th
        # xe = xs + tw
        # assert(xe<=self.data.shape[1])
        # assert(ye<=self.data.shape[0])
        # tiledata = self.data[ys:ye, xs:xe]
        # prepare a tile of data for OpenGL
        tiledata = self._calc.tile_pixels(self.data, *tile_yx)
        tile = np.require(tiledata, dtype=np.float32, requirements=['C_CONTIGUOUS', 'ALIGNED'])
        del tiledata

        # FIXME: temporarily require that textures aren't odd sizes
        txbox = box(b=0, l=0, t=th, r=tw)

        glBindTexture(GL_TEXTURE_BUFFER, texid)
        LOD = 0
        glTexSubImage2D(GL_TEXTURE_2D, LOD, 0,0, tw,th, GL_RED, GL_FLOAT, tile.data)

        nfo = (texid, txbox)
        self._active[tile_yx] = nfo
        return nfo


        # # start working with this buffer
        # glBindBuffer(GL_COPY_WRITE_BUFFER, buffer_id)
        # # allocate memory for it
        # glBufferData(GL_COPY_WRITE_BUFFER, , size, )
        # # borrow a writable pointer
        # pgl = glMapBuffer(, )
        # # make a numpy ndarray wrapper that targets that memory location
        #
        # # copy the data from the memory mapped file
        # # if there's not enough data in the file, fill with NaNs first
        # if npslice.shape != npgl.shape:
        #     npgl[:] = np.nan(0)
        #
        # if not transform_func:
        #     # straight copy
        #     np.copyto(npgl, npslice, casting='unsafe')
        # else:
        #     # copy through transform function
        #
        # # let GL push it to the GPU
        #glUnmapBuffer(GL_COPY_WRITE_BUFFER)
        # also see glTexBuffer()


    def deactivate(self, tile_y, tile_x):
        """
        release a given
        """
        k = (tile_y, tile_x)
        t = self._active[k]
        del self._active[k]

        glDeleteTextures([t])


class TextureTileLayer(Layer):
    """
    A layer represented as an array of quads with textures on them
    """
    pass


class ColormapTiles(TextureTileLayer):
    """
    Mercator-projected geographic layer with one or more levels of detail.
    Coarsest level of detail (1° typically) is always expected to be available for fast zoom.

    """
    _tileset = None
    _image = None
    _shape = None

    def __init__(self, path, zero_point=None, pixel_rez=None):
        """

        """
        super(ColormapTiles, self).__init__()
        if path.lower().endswith('.json'):
            from json import load
            with open(path, 'rt') as fp:
                info = load(fp)
                globals().update(info)  # FIXME: security
        self._image = im = Image.open(path)
        # image is top-left to bottom-right, remember this when loading textures
        im.load()
        l,t,r,b = im.getbbox()
        w = r-l
        h = b-t
        self._shape = (h,w)



    def paint(self, geom, fast=False):
        return True

    def render(self, geom, *more_geom):
        """
        in offline GL, make a drawlist that nicely handles the world geometry requested
        """

        return True

    def purge(self, geom, *more_geom):
        return False

    def probe_point_xy(self, x, y):
        raise NotImplementedError()

    def probe_point_geo(self, lat, lon):
        raise NotImplementedError()

    def probe_shape(self, geo_shape):
        raise NotImplementedError()


    # def tileseq_in_area(self, index_bltr):
    #     """
    #     yield the sequence of tiles in a given rectangular tileseq_in_area
    #     """
    #     pass
    #
    # def tileseq_visible(self, data_bltr):
    #     """
    #     given data coordinates, determine which tiles are on the canvas
    #     """
    #     pass
    #


class TestLayer(Layer):
    def paint(self, *args):
        glColor3f(0.0, 0.0, 1.0)
        glRectf(-5, -5, 5, 5)
        glColor3f(1.0, 0.0, 0.0)
        glBegin(GL_LINES)
        glVertex3f(0, 0, 0)
        glVertex3f(20, 20, 0)



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

