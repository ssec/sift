#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Script to convert AHI Geostationary NetCDF files to Mercator geotiffs at the same resolution.

:author: David Hoese <david.hoese@ssec.wisc.edu>
:copyright: 2015 by University of Wisconsin Regents, see AUTHORS for more details
:license: GPLv3, see LICENSE for more details
"""
__author__ = 'davidh'

import os
import sys
import logging

from glob import glob
from osgeo import gdal
import osr
import subprocess

from cspov.project.ahi2gtiff import ahi2gtiff

LOG = logging.getLogger(__name__)
GTIFF_DRIVER = gdal.GetDriverByName("GTIFF")
DEFAULT_PROJ_STR = "+proj=merc +datum=WGS84 +ellps=WGS84 +no_defs"


def run_gdalwarp(input_file, output_file, *args):
    subprocess.call(["gdalwarp"] + list(args) + [input_file, output_file])


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Convert AHI Geos NetCDF files to mercator geotiffs at the same resolution")
    parser.add_argument("--merc-ext", default=".merc.tif",
                        help="Extension for new mercator files (replace '.tif' with '.merc.tif' by default)")
    parser.add_argument("--input-pattern", default="????/*.nc",
                        help="Input pattern used search for NetCDF files in 'input_dir'")
    parser.add_argument('-v', '--verbose', dest='verbosity', action="count", default=0,
                        help='each occurrence increases verbosity 1 level through ERROR-WARNING-INFO-DEBUG (default INFO)')
    parser.add_argument("input_dir",
                        help="Input directory to search for the 'input_pattern' specified")
    parser.add_argument("output_dir",
                        help="Output directory to place new mercator files (input_pattern structure is reflected in output dir)")
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

        geos_file = nc_file.replace(idir, odir).replace(".nc", ".tif")
        opath = os.path.dirname(geos_file)
        if not os.path.exists(opath):
            LOG.info("Creating output directory: %s", opath)
            os.makedirs(opath)
        ahi2gtiff(nc_file, geos_file)

        # Get information about the geotiff
        gtiff = gdal.Open(geos_file)
        srs = osr.SpatialReference()
        srs.ImportFromWkt(gtiff.GetProjection())
        proj = srs.ExportToProj4()
        if "+proj=geos" not in proj:
            LOG.warning("Tried to process non-geos geotiff: %s" % (geos_file,))
        ox, cw, _, oy, _, ch = gtiff.GetGeoTransform()

        # Come up with an output filename
        merc_file = geos_file.replace(".tif", args.merc_ext)
        if os.path.isfile(merc_file):
            LOG.warning("Output file already exists: %s" % (merc_file,))
            continue

        # Run gdalwarp
        LOG.info("Running gdalwarp on '%s' to create '%s'", geos_file, merc_file)
        gdalwarp_args = args.gdalwarp_args + [
            "-multi",
            "-t_srs", DEFAULT_PROJ_STR,
            "-tr", str(cw), str(ch),
            "-te_srs", "+proj=latlong +datum=WGS84 +ellps=WGS84",
            "-te", "-180", "-80", "180", "80",
        ]
        run_gdalwarp(geos_file, merc_file, *gdalwarp_args)

if __name__ == "__main__":
    sys.exit(main())
