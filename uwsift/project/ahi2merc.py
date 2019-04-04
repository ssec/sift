#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Script to convert AHI Geostationary NetCDF files to Mercator geotiffs at the same resolution.

:author: David Hoese <david.hoese@ssec.wisc.edu>
:copyright: 2015 by University of Wisconsin Regents, see AUTHORS for more details
:license: GPLv3, see LICENSE for more details
"""
__author__ = 'davidh'

import logging
import os
import subprocess
import sys
from glob import glob

import numpy as np
import osr
from osgeo import gdal
from pyproj import Proj

from uwsift.project.ahi2gtiff import create_ahi_geotiff, ahi_image_info, ahi_image_data

LOG = logging.getLogger(__name__)
GTIFF_DRIVER = gdal.GetDriverByName("GTIFF")
DEFAULT_PROJ_STR = "+proj=merc +datum=WGS84 +ellps=WGS84 +no_defs"


def run_gdalwarp(input_file, output_file, *args):
    args = list(args) + [input_file, output_file]
    LOG.info("Running gdalwarp with: gdalwarp %s", " ".join(args))
    subprocess.call(["gdalwarp"] + args)


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Convert AHI Geos NetCDF files to mercator geotiffs at the same resolution")
    parser.add_argument("--merc-ext", default=".merc.tif",
                        help="Extension for new mercator files (replace '.tif' with '.merc.tif' by default)")
    parser.add_argument("--input-pattern", default="????/*.nc",
                        help="Input pattern used search for NetCDF files in 'input_dir'")
    parser.add_argument('-v', '--verbose', dest='verbosity', action="count", default=0,
                        help='each occurrence increases verbosity 1 level through '
                             'ERROR-WARNING-Info-DEBUG (default Info)')

    # http://www.gdal.org/frmt_gtiff.html
    parser.add_argument('--compress', default=None,
                        help="Type of compression for geotiffs (passed to GDAL GeoTIFF Driver)")
    parser.add_argument('--predictor', default=None, type=int,
                        help="Set predictor for geotiff compression (LZW or DEFLATE)")
    parser.add_argument('--tiled', action='store_true',
                        help="Create tiled geotiffs")
    parser.add_argument('--blockxsize', default=None, type=int,
                        help="Set tile block X size")
    parser.add_argument('--blockysize', default=None, type=int,
                        help="Set tile block Y size")
    parser.add_argument('--extents', default=[np.nan, np.nan, np.nan, np.nan], nargs=4, type=float,
                        help="Set mercator bounds in lat/lon space (lon_min lat_min lon_max lat_max)")
    parser.add_argument('--nodata', default=None, type=float,
                        help="Set the nodata value for the geotiffs that are created")

    parser.add_argument("input_dir",
                        help="Input directory to search for the 'input_pattern' specified")
    parser.add_argument("output_dir",
                        help="Output directory to place new mercator files "
                             "(input_pattern structure is reflected in output dir)")
    parser.add_argument("gdalwarp_args", nargs="*",
                        help="arguments that are passed directly to gdalwarp")
    args = parser.parse_args()

    levels = [logging.ERROR, logging.WARN, logging.INFO, logging.DEBUG]
    logging.basicConfig(level=levels[min(3, args.verbosity)])

    idir = args.input_dir
    odir = args.output_dir
    ipat = args.input_pattern

    for nc_file in glob(os.path.join(idir, ipat)):
        if not os.path.isfile(nc_file):
            LOG.error("Not a file '%s'" % (nc_file,))
            continue

        # Come up with an intermediate geotiff name
        geos_file = nc_file.replace(idir, odir, 1).replace(".nc", ".tif")
        opath = os.path.dirname(geos_file)
        if not os.path.exists(opath):
            LOG.info("Creating output directory: %s", opath)
            os.makedirs(opath, exist_ok=True)

        # Come up with an output mercator filename
        merc_file = geos_file.replace(".tif", args.merc_ext)
        if os.path.isfile(merc_file):
            LOG.warning("Output file already exists, will delete to start over: %s" % (merc_file,))
            try:
                os.remove(merc_file)
            except Exception:
                LOG.error("Could not remove previous file: %s" % (merc_file,))
                continue

        try:
            src_info = ahi_image_info(nc_file)
            if not os.path.exists(geos_file):
                src_data = ahi_image_data(nc_file)
                # print("### Resource Data: Min (%f) | Max (%f)" % (src_data.min(), src_data.max()))
                create_ahi_geotiff(src_info, src_data, geos_file,
                                   compress=args.compress,
                                   predictor=args.predictor,
                                   tiled=args.tiled,
                                   blockxsize=args.blockxsize,
                                   blockysize=args.blockysize,
                                   nodata=args.nodata)
            else:
                LOG.debug("GEOS Projection GeoTIFF already exists, won't recreate...")
            lon_west, lon_east = src_info["lon_extents"]
        except RuntimeError:
            LOG.error("Could not create geotiff for '%s'" % (nc_file,))
            LOG.debug("Exception Information: ", exc_info=True)
            continue

        # Get information about the geotiff
        gtiff = gdal.Open(geos_file)
        srs = osr.SpatialReference()
        srs.ImportFromWkt(gtiff.GetProjection())
        proj = srs.ExportToProj4()
        if "+proj=geos" not in proj:
            LOG.warning("Tried to process non-geos geotiff: %s" % (geos_file,))
        ox, cw, _, oy, _, ch = gtiff.GetGeoTransform()

        # Run gdalwarp
        LOG.info("Running gdalwarp on '%s' to create '%s'", geos_file, merc_file)
        proj = DEFAULT_PROJ_STR
        proj = proj + " +over" if lon_east >= 180 else proj
        # Include the '+over' parameter so longitudes are wrapper around the antimeridian
        src_proj = Proj(proj)

        # use image bounds
        if np.isnan(args.extents[0]):
            args.extents[0] = lon_west
        if np.isnan(args.extents[1]):
            args.extents[1] = -80
        if np.isnan(args.extents[2]):
            args.extents[2] = lon_east
        if np.isnan(args.extents[3]):
            args.extents[3] = 80

        x_min, y_min = src_proj(args.extents[0], args.extents[1])
        x_max, y_max = src_proj(args.extents[2], args.extents[3])
        x_extent = (x_min, x_max)
        y_extent = (y_min, y_max)
        LOG.debug("Using extents (%f : %f : %f : %f)", x_extent[0], y_extent[0], x_extent[1], y_extent[1])

        gdalwarp_args = args.gdalwarp_args + [
            # "-multi",
            "-t_srs", proj,
            "-tr", str(cw), str(ch),
            "-te",
            "{:0.03f}".format(x_extent[0]),
            "{:0.03f}".format(y_extent[0]),
            "{:0.03f}".format(x_extent[1]),
            "{:0.03f}".format(y_extent[1]),
        ]
        if args.nodata is not None:
            gdalwarp_args.extend([
                "-srcnodata", str(args.nodata),
                "-dstnodata", str(args.nodata),
            ])
        if args.compress is not None:
            gdalwarp_args.extend(["-co", "COMPRESS=%s" % (args.compress,)])
            if args.predictor is not None:
                gdalwarp_args.extend(["-co", "PREDICTOR=%d" % (args.predictor,)])
        if args.tiled:
            gdalwarp_args.extend(["-co", "TILED=YES"])
        if args.blockxsize is not None:
            gdalwarp_args.extend(["-co", "BLOCKXSIZE=%d" % (args.blockxsize,)])
        if args.blockysize is not None:
            gdalwarp_args.extend(["-co", "BLOCKYSIZE=%d" % (args.blockysize,)])
        run_gdalwarp(geos_file, merc_file, *gdalwarp_args)


if __name__ == "__main__":
    sys.exit(main())
