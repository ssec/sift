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
from netCDF4 import Dataset
from osgeo import gdal
from pyproj import Proj

from uwsift.project.ahi2gtiff import create_ahi_geotiff, AHI_NADIR_RES

LOG = logging.getLogger(__name__)
GTIFF_DRIVER = gdal.GetDriverByName("GTIFF")
DEFAULT_PROJ_STR = "+proj=merc +datum=WGS84 +ellps=WGS84 +no_defs"
DEFAULT_DATASETS = [
    "eps_cmask_ahi_cloud_mask",
    "enterprise_cldphase_10_11_13_14_15_cloud_phase",
    "enterprise_cldphase_10_11_13_14_15_cloud_type",
    "ACHA_mode_8_cloud_top_height",
]
LEVEL_1_DATASETS = [
    'himawari_8_ahi_channel_1_reflectance',
    'himawari_8_ahi_channel_2_reflectance',
    'himawari_8_ahi_channel_3_reflectance',
    'himawari_8_ahi_channel_4_reflectance',
    'himawari_8_ahi_channel_5_reflectance',
    'himawari_8_ahi_channel_6_reflectance',
    'himawari_8_ahi_channel_7_brightness_temperature',
    # 'himawari_8_ahi_channel_7_emissivity',
    # 'himawari_8_ahi_channel_7_reflectance',
    'himawari_8_ahi_channel_8_brightness_temperature',
    'himawari_8_ahi_channel_9_brightness_temperature',
    'himawari_8_ahi_channel_10_brightness_temperature',
    'himawari_8_ahi_channel_11_brightness_temperature',
    'himawari_8_ahi_channel_12_brightness_temperature',
    'himawari_8_ahi_channel_13_brightness_temperature',
    'himawari_8_ahi_channel_14_brightness_temperature',
    'himawari_8_ahi_channel_15_brightness_temperature',
    'himawari_8_ahi_channel_16_brightness_temperature',
]


def run_gdalwarp(input_file, output_file, *args):
    args = list(args) + [input_file, output_file]
    LOG.info("Running gdalwarp with: gdalwarp %s", " ".join(args))
    subprocess.call(["gdalwarp"] + args)


def ahi_image_info(input_filename):
    nc = Dataset(input_filename, "r")

    # projection_info = nc.variables["Projection"].__dict__.copy()
    # projection_info["semi_major_axis"] *= 1000.0  # is in kilometers, need meters
    # projection_info["perspective_point_height"] *= 1000.0  # is in kilometers, need meters
    # projection_info["flattening"] = 1.0 / projection_info["inverse_flattening"]

    # Notes on Sweep: https://trac.osgeo.org/proj/wiki/proj%3Dgeos
    # If AHI behaves like GOES then sweep should be Y
    # +over is needed so that we do 0-360 lon/lats and x/y
    # input_proj_str = "+proj=geos +over "
    # input_proj_str += "+lon_0={longitude_of_projection_origin:0.3f} "
    # input_proj_str += "+lat_0={latitude_of_projection_origin:0.3f} "
    # input_proj_str += "+a={semi_major_axis:0.3f} "
    # input_proj_str += "+f={flattening} "
    # input_proj_str += "+h={perspective_point_height} "
    # input_proj_str += "+sweep={sweep_angle_axis}"
    # input_proj_str = input_proj_str.format(**projection_info)
    input_proj_str = "+proj=geos +over +lon_0=140.7 +h=35785863 +x_0=0 +y_0=0 " \
                     "+a=6378137 +b=6356752.299581327 +units=m +no_defs"
    LOG.debug("AHI Fixed Grid Projection String: %s", input_proj_str)

    # Calculate upper-left corner (origin) using center point as a reference
    shape = (len(nc.dimensions["lines"]), len(nc.dimensions["elements"]))
    pixel_size_x = AHI_NADIR_RES[shape[0]]
    pixel_size_y = -AHI_NADIR_RES[shape[1]]

    # Assume that center pixel is at 0, 0 projection space
    origin_x = -pixel_size_x * (shape[1] / 2.0)
    origin_y = -pixel_size_y * (shape[0] / 2.0)
    LOG.debug("Origin X: %f\tOrigin Y: %f", origin_x, origin_y)

    # AHI NetCDF files have "proper" 0-360 longitude so lon_min is west, lon_max is east
    # Find the left side of the image and the right side of the image
    p = Proj(input_proj_str)
    x = np.empty(shape, dtype=np.float32)
    x[:] = origin_x + (pixel_size_x * np.arange(shape[1], dtype=np.float32))
    y = np.empty(shape, dtype=np.float32)
    y[:] = origin_y + (pixel_size_y * np.arange(shape[0], dtype=np.float32)[:, None])
    lon, lat = p(x, y, inverse=True)
    lon[lon == 1e30] = np.nan
    lat[lat == 1e30] = np.nan
    lon_min = np.nanmin(lon)
    lon_max = np.nanmax(lon)
    lat_south = np.nanmin(lat)
    lat_north = np.nanmax(lat)
    LOG.info("Longitude Minimum: %f; Maximum: %f" % (lon_min, lon_max))
    LOG.info("Latitude Minimum: %f; Maximum: %f" % (lat_south, lat_north))
    # this logic only works if lons are 0-360 (see +over attribute of above PROJ.4)
    if lon_max >= 180 or (lon_max - lon_min) < 180:
        # If longitudes are 0-360 then coordinates are as expected
        lon_west = lon_min
        lon_east = lon_max
    else:
        # If we are wrapping around the antimeridian
        lon_west = lon_max
        lon_east = lon_min

    info = {
        "proj": input_proj_str,
        "origin_x": origin_x,
        "origin_y": origin_y,
        "cell_width": pixel_size_x,
        "cell_height": pixel_size_y,
        "width": shape[1],
        "height": shape[0],
        "lon_extents": (lon_west, lon_east),
        "lat_extents": (lat_south, lat_north),
    }

    return info


VAR_NAME_STANDARD_NAME = {
    'cloud_top_height': 'height_at_cloud_top',
    'cloud_phase': 'thermodynamic_phase_of_cloud_water_particles_at_cloud_top',
    'cloud_type': 'cloud_type',
    'cloud_mask': 'cloud_mask',
    'brightness_temperature': 'toa_brightness_temperature',
    'reflectance': 'toa_bidirectional_reflectance',
}


def ahi_dataset_metadata(input_filename, dataset_name):
    nc = Dataset(input_filename, "r")
    input_var = nc.variables[dataset_name]
    # __dict__ gets us all the attributes but is likely OrderedDict
    # make it a normal dict
    metadata = dict(input_var.__dict__)
    metadata.setdefault("name", dataset_name)
    metadata.setdefault("platform", nc.Spacecraft_Name.title())
    metadata.setdefault("sensor", nc.Sensor_Name)
    # sensor is wrong in the file
    if metadata["sensor"] == "himawari8":
        metadata["sensor"] = "AHI"

    # if we have valid_min and valid_max then get the scaled values
    valid_min, valid_max = metadata.pop("valid_range",
                                        (metadata.pop("valid_min", None), metadata.pop("valid_max", None)))
    scale_factor = metadata.pop("scale_factor", 1)
    add_offset = metadata.pop("add_offset", 0)
    if valid_min is not None and valid_max is not None:
        metadata["valid_min"] = valid_min * scale_factor + add_offset
        metadata["valid_max"] = valid_max * scale_factor + add_offset

    # mimic satpy metadata
    metadata["start_time"] = getattr(nc, "Image_Date_Time")
    metadata["end_time"] = getattr(nc, "Image_Date_Time")
    if "standard_name" not in metadata:
        for k, v in VAR_NAME_STANDARD_NAME.items():
            if dataset_name.endswith(k):
                metadata["standard_name"] = v
                break

    # get rid of keys that don't make sense once the data is in the geotiff
    for k in ["scale_factor", "add_offset", "_FillValue"]:
        metadata.pop(k, None)
    return metadata


def ahi_image_data(input_filename, dataset):
    nc = Dataset(input_filename, "r")
    input_data = nc.variables[dataset][:]
    input_data = input_data.astype(np.float32)
    if hasattr(input_data, "mask"):
        return input_data.filled(np.nan)
    else:
        return input_data


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Convert geocat HDF4 files to mercator geotiffs at the same resolution")
    parser.add_argument("--merc-ext", default=".{dataset}.merc.tif",
                        help="Extension for new mercator files (replace '.tif' with '.merc.tif' by default)")
    parser.add_argument("--input-pattern", default="????/*.nc",
                        help="Input pattern used search for NetCDF files in 'input_dir'")
    parser.add_argument('-v', '--verbose', dest='verbosity', action="count", default=0,
                        help='each occurrence increases verbosity 1 level through'
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
    parser.add_argument('--datasets', default=DEFAULT_DATASETS, nargs="+",
                        help="Specify variables to load from input file")
    parser.add_argument('--level1-channels', dest="datasets", action='store_const', const=LEVEL_1_DATASETS,
                        help="Use the default Level 1 dataset names instead of `--datasets`")
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

        src_info = ahi_image_info(nc_file)

        for dataset in args.datasets:
            LOG.info("Creating geotiff for dataset '%s'", dataset)
            # Come up with an intermediate geotiff name
            in_ext = os.path.splitext(nc_file)[-1]
            geos_ext = ".{dataset}.tif".format(dataset=dataset)
            geos_file = nc_file.replace(idir, odir, 1).replace(in_ext, geos_ext)
            opath = os.path.dirname(geos_file)
            if not os.path.exists(opath):
                LOG.info("Creating output directory: %s", opath)
                os.makedirs(opath, exist_ok=True)

            # Come up with an output mercator filename
            merc_file = geos_file.replace(geos_ext, args.merc_ext.format(dataset=dataset))
            if os.path.isfile(merc_file):
                LOG.warning("Output file already exists, will delete to start over: %s" % (merc_file,))
                try:
                    os.remove(merc_file)
                except Exception:
                    LOG.error("Could not remove previous file: %s" % (merc_file,))
                    continue

            try:
                if not os.path.exists(geos_file):
                    src_data = ahi_image_data(nc_file, dataset=dataset)
                    src_meta = ahi_dataset_metadata(nc_file, dataset_name=dataset)
                    # print("### Source Data: Min (%f) | Max (%f)" % (src_data.min(), src_data.max()))
                    create_ahi_geotiff(src_info, src_meta, src_data, geos_file,
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
            # lon east/west should be 0-360 right now
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
            LOG.debug("Using degrees extents (%f : %f : %f : %f)", *args.extents)
            LOG.debug("Using extents (%f : %f : %f : %f)", x_extent[0], y_extent[0], x_extent[1], y_extent[1])
            # import ipdb; ipdb.set_trace()

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
