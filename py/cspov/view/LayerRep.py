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
import shapefile

from vispy.scene.visuals import create_visual_node
from vispy.visuals import LineVisual, ImageVisual, CompoundVisual
import numpy as np
from datetime import datetime
from pyproj import Proj

# from cspov.common import pnt, rez, MAX_EXCURSION_Y, MAX_EXCURSION_X, MercatorTileCalc, WORLD_EXTENT_BOX, \
#     DEFAULT_TILE_HEIGHT, DEFAULT_TILE_WIDTH, vue
from cspov.common import (DEFAULT_X_PIXEL_SIZE,
                          DEFAULT_Y_PIXEL_SIZE,
                          DEFAULT_ORIGIN_X,
                          DEFAULT_ORIGIN_Y,
                          DEFAULT_PROJECTION,
                          WORLD_EXTENT_BOX,
                          C_EQ,
                          )
from cspov.view.Program import GlooRGBImageTile

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


class GeolocatedImageVisual(ImageVisual):
    """Visual class for image data that is geolocated and should be referenced that way when displayed.

    Note: VisPy separates the Visual from the Node used in the SceneGraph by dynamically creating
    the `vispy.scene.visuals.Image` class. Use the GeolocatedImage class with scenes.
    """
    def __init__(self,
                 origin_x=DEFAULT_ORIGIN_X, origin_y=DEFAULT_ORIGIN_Y,
                 cell_width=DEFAULT_X_PIXEL_SIZE, cell_height=DEFAULT_Y_PIXEL_SIZE,
                 double=False, **kwargs):
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.cell_width = cell_width
        self.cell_height = cell_height
        self.double = double
        super(GeolocatedImageVisual, self).__init__(**kwargs)

    @classmethod
    def from_geotiff(cls, geotiff_filepath, **kwargs):
        import gdal
        gtiff = gdal.Open(geotiff_filepath)
        ox, cw, _, oy, _, ch = gtiff.GetGeoTransform()
        img_data = gtiff.GetRasterBand(1).ReadAsArray()
        return cls(data=img_data, origin_x=ox, origin_y=oy, cell_width=cw, cell_height=ch, **kwargs)

    def _build_vertex_data(self):
        """Rebuild the vertex buffers used for rendering the image when using
        the subdivide method.

        CSPOV Note: Copied from 0.5.0dev original ImageVisual class
        """
        grid = self._grid
        w = 1.0 / grid[1]
        h = 1.0 / grid[0]

        quad = np.array([[0, 0, 0], [w, 0, 0], [w, h, 0],
                         [0, 0, 0], [w, h, 0], [0, h, 0]],
                        dtype=np.float32)
        quads = np.empty((grid[1], grid[0], 6, 3), dtype=np.float32)
        quads[:] = quad

        mgrid = np.mgrid[0.:grid[1], 0.:grid[0]].transpose(1, 2, 0)
        mgrid = mgrid[:, :, np.newaxis, :]
        mgrid[..., 0] *= w
        mgrid[..., 1] *= h

        quads[..., :2] += mgrid
        tex_coords = quads.reshape(grid[1]*grid[0]*6, 3)
        tex_coords = np.ascontiguousarray(tex_coords[:, :2])
        vertices = tex_coords * self.size

        # FUTURE: This should probably be done as a transform
        vertices = vertices.astype('float32')
        vertices[:, 0] *= self.cell_width
        vertices[:, 0] += self.origin_x
        vertices[:, 1] *= self.cell_height
        vertices[:, 1] += self.origin_y
        if self.double:
            orig_points = vertices.shape[0]
            vertices = np.concatenate((vertices, vertices), axis=0)
            tex_coords = np.concatenate((tex_coords, tex_coords), axis=0)
            vertices[orig_points:, 0] += C_EQ
        self._subdiv_position.set_data(vertices.astype('float32'))
        self._subdiv_texcoord.set_data(tex_coords.astype('float32'))


GeolocatedImage = create_visual_node(GeolocatedImageVisual)


class TiledGeolocatedImageVisual(CompoundVisual):
    """Geolocated image that is specially tiled to save on GPU memory usage.

    Currently this visual holds on to the original data on the CPU side forever. The
    provided data is assumed to be a memory mapped file. In future iterations of this
    class it will provide a method to set the data for the lower resolutions based on
    events/signals.

    Tile sizes (pixels) are based on the original full resolution of the data in meters
    since this data is known to be geolocated. For example, input data for image 1
    could be of size 10x10 at 1km resolution and image 2 could be at the same 1km
    spatial/projected resolution but be 20x20 pixels.
    At lower resolution display (Z greater than image Z) we would want to display the
    same number of pixels for each image regardless of full resolution number of pixels.
    """
    def __init__(self, data, tile_width=10000000.0, tile_height=16000000.0, **kwargs):
        # Note: tile_width=5000000.0, tile_height=8000000.0 give ~1000 pixels per tile for test images
        # TODO: Put all logic about "what tile information should I use" in to a separate class
        # FIXME: Hardcoded for now
        self._subimages = []
        # Make a logical assertion that a tile must be at least 10 pixels
        assert(tile_width >= 10 * kwargs["cell_width"])
        assert(tile_height >= 10 * kwargs["cell_height"])
        tile_x_step = int(abs(tile_width // kwargs["cell_width"]) + 0.5)
        tile_y_step = int(abs(tile_height // kwargs["cell_height"]) + 0.5)
        y_offsets = range(0, data.shape[0], tile_x_step)
        dy = y_offsets[1] - y_offsets[0]
        x_offsets = range(0, data.shape[1], tile_y_step)
        dx = x_offsets[1] - x_offsets[0]
        print(list(x_offsets), list(y_offsets))
        print(len(x_offsets), len(y_offsets))
        for y_offset in y_offsets:
            for x_offset in x_offsets:
                kws = kwargs.copy()
                kws["origin_x"] = kws["origin_x"] + x_offset * kws["cell_width"]
                kws["origin_y"] = kws["origin_y"] + y_offset * kws["cell_height"]
                image = GeolocatedImageVisual(data=data[y_offset: y_offset + dy, x_offset: x_offset + dx], **kws)
                self._subimages.append(image)
        super(TiledGeolocatedImageVisual, self).__init__(self._subimages)

    def _generate_tiles(self):
        pass

    @classmethod
    def from_geotiff(cls, geotiff_filepath, **kwargs):
        import gdal
        gtiff = gdal.Open(geotiff_filepath)
        ox, cw, _, oy, _, ch = gtiff.GetGeoTransform()
        img_data = gtiff.GetRasterBand(1).ReadAsArray()
        return cls(data=img_data, origin_x=ox, origin_y=oy, cell_width=cw, cell_height=ch, **kwargs)


TiledGeolocatedImage = create_visual_node(TiledGeolocatedImageVisual)


# FIXME: replace this with a LayerRepFactory which tracks all the GPU resources that have been dedicated?
class OldTiledImageFile(LayerRep):
    """
    Tile an RGB or float32 image representing the full -180..180 longitude, -90..90 latitude
    """
    image = None
    shape = None
    calc = None
    tiles = None  # dictionary of {(y,x): GlooRgbTile, ...}
    _stride = 1
    _tile_kwargs = {}

    def set_z(self, z):
        super(TiledImageFile, self).set_z(z)
        for tile in self.tiles.values():
            tile.z = z

    def set_alpha(self, alpha):
        for tile in self.tiles.values():
            tile.alpha = alpha
        super(TiledImageFile, self).set_alpha(alpha)

    def __init__(self, filename=None, world_box=None, tile_shape=None, tile_class=GlooRGBImageTile, **kwargs):
        super(TiledImageFile, self).__init__()
        self._tile_class = tile_class
        self._tile_kwargs = dict(kwargs)  # FUTURE this is too cryptic, initially used to propagate range default
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
        _, tilebox = self.calc.visible_tiles(visible_geom)
        # LOG.info(tilebox)
        # print(repr(tilebox))
        for iy in range(tilebox.b, tilebox.t):
            for ix in range(tilebox.l, tilebox.r):
                tile = self.tiles[(iy,ix)]
                # LOG.debug('draw tile {0!r:s}'.format(tile))
                # m,v,p = mvp
                tile.set_mvp(*mvp)
                tile.draw()
        preferred_stride = self.calc.calc_stride(visible_geom)
        return True if preferred_stride != self._stride else False

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
                self.tiles[(tiy,tix)] = t = self._tile_class(tilegeom, subim, **self._tile_kwargs)
                # t.set_mvp(model=self.model, view=self.view)


class ShapefileLinesVisual(CompoundVisual):
    def __init__(self, filepath, projection=DEFAULT_PROJECTION, double=False, **kwargs):
        LOG.debug("Using border shapefile '%s'", filepath)
        self.sf = shapefile.Reader(filepath)
        # FUTURE: Proj stuff should be done in GLSL for better speeds and flexibility with swapping projection (may require something in addition to transform)
        self.proj = Proj(projection)

        print("Loading boundaries: ", datetime.utcnow().isoformat(" "))
        # Prepare the arrays
        total_points = 0
        total_parts = 0
        for idx, one_shape in enumerate(self.sf.iterShapes()):
            total_points += len(one_shape.points)
            total_parts += len(one_shape.parts)
        vertex_buffer = np.empty((total_points * 2 - total_parts * 2, 2), dtype=np.float32)
        prev_idx = 0
        for idx, one_shape in enumerate(self.sf.iterShapes()):
            # end_idx = prev_idx + len(one_shape.points) * 2 - len(one_shape.parts) * 2
            # vertex_buffer[prev_idx:end_idx:2] = one_shape.points[:-1]
            # for part_idx in one_shape.parts:
            for part_start, part_end in zip(one_shape.parts, list(one_shape.parts[1:]) + [len(one_shape.points)]):
                end_idx = prev_idx + (part_end - part_start) * 2 - 2
                vertex_buffer[prev_idx:end_idx:2] = one_shape.points[part_start:part_end-1]
                vertex_buffer[prev_idx + 1:end_idx:2] = one_shape.points[part_start+1:part_end]
                prev_idx = end_idx

        # Clip lats to +/- 89.9 otherwise PROJ.4 on mercator projection will fail
        np.clip(vertex_buffer[:, 1], -89.9, 89.9, out=vertex_buffer[:, 1])
        vertex_buffer[:, 0], vertex_buffer[:, 1] = self.proj(vertex_buffer[:, 0], vertex_buffer[:, 1])
        if double:
            LOG.debug("Adding 180 to 540 double of shapefile")
            orig_points = vertex_buffer.shape[0]
            vertex_buffer = np.concatenate((vertex_buffer, vertex_buffer), axis=0)
            vertex_buffer[orig_points:, 0] += C_EQ

        kwargs.setdefault("color", (0.0, 0.0, 1.0, 1.0))
        kwargs.setdefault("width", 1)
        self._border_lines = LineVisual(vertex_buffer, connect="segments", **kwargs)
        print("Done loading boundaries: ", datetime.utcnow().isoformat(" "))

        super(ShapefileLinesVisual, self).__init__((self._border_lines,))


ShapefileLines = create_visual_node(ShapefileLinesVisual)


class NEShapefileLinesVisual(ShapefileLinesVisual):
    """Layer class for handling shapefiles from Natural Earth.

    http://www.naturalearthdata.com/

    There should be no difference in the format of the file, but some
    assumptions can be made with data from Natural Earth about filenaming,
    data resolution, fields and other record information that is normally
    included in most Natural Earth files.
    """
    pass


NEShapefileLines = create_visual_node(NEShapefileLinesVisual)


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

