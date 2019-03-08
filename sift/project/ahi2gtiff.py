#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Script to convert AHI NetCDF files in to a geotiff image.

:author: David Hoese <david.hoese@ssec.wisc.edu>
:copyright: 2015 by University of Wisconsin Regents, see AUTHORS for more details
:license: GPLv3, see LICENSE for more details
"""

import logging
import os
import sys

import numpy as np
import osr
from netCDF4 import Dataset
from osgeo import gdal
from pyproj import Proj

LOG = logging.getLogger(__name__)
GTIFF_DRIVER = gdal.GetDriverByName("GTIFF")

AHI_NADIR_RES = {
    5500: 2000,
    11000: 1000,
    22000: 500,
}


def _proj4_to_srs(proj4_str):
    """Helper function to convert a proj4 string
    into a GDAL compatible srs.  Mainly a function
    so if used multiple times it only has to be changed
    once for special cases.
    """
    try:
        srs = osr.SpatialReference()
        # GDAL doesn't like unicode
        result = srs.ImportFromProj4(str(proj4_str))
    except Exception:
        LOG.error("Could not convert Proj4 string '%s' into a GDAL SRS" % (proj4_str,))
        LOG.debug("Exception: ", exc_info=True)
        raise

    if result != 0:
        LOG.error("Could not convert Proj4 string '%s' into a GDAL SRS" % (proj4_str,))
        raise ValueError("Could not convert Proj4 string '%s' into a GDAL SRS" % (proj4_str,))

    return srs


def create_geotiff(data, output_filename, proj4_str, geotransform, etype=gdal.GDT_Byte,
                   compress=None, predictor=None, tile=False,
                   blockxsize=None, blockysize=None,
                   quicklook=False, gcps=None,
                   nodata=np.nan, meta=None, **kwargs):
    """Function that creates a geotiff from the information provided.
    """
    LOG.info("Creating geotiff '%s'" % (output_filename,))

    if etype != gdal.GDT_Float32 and nodata is not None and np.isnan(nodata):
        nodata = 0

    # Find the number of bands provided
    if isinstance(data, (list, tuple)):
        num_bands = len(data)
    elif len(data.shape) == 2:
        num_bands = 1
    else:
        num_bands = data.shape[0]

    # We only know how to handle gray scale, RGB, and RGBA
    if num_bands not in [1, 3, 4]:
        msg = "Geotiff backend doesn't know how to handle data of shape '%r'" % (data.shape,)
        LOG.error(msg)
        raise ValueError(msg)

    options = []
    if num_bands == 1:
        options.append("PHOTOMETRIC=MINISBLACK")
    elif num_bands == 3:
        options.append("PHOTOMETRIC=RGB")
    elif num_bands == 4:
        options.append("PHOTOMETRIC=RGB")

    if compress is not None and compress != "NONE":
        options.append("COMPRESS=%s" % (compress,))
        if predictor is not None:
            options.append("PREDICTOR=%d" % (predictor,))
    if tile:
        options.append("TILED=YES")
    if blockxsize is not None:
        options.append("BLOCKXSIZE=%d" % (blockxsize,))
    if blockysize is not None:
        options.append("BLOCKYSIZE=%d" % (blockysize,))

    # Creating the file will truncate any pre-existing file
    LOG.debug("Creation Geotiff with options %r", options)
    if num_bands == 1:
        gtiff = GTIFF_DRIVER.Create(output_filename, data.shape[1], data.shape[0],
                                    bands=num_bands, eType=etype, options=options)
    else:
        gtiff = GTIFF_DRIVER.Create(output_filename, data[0].shape[1], data[0].shape[0],
                                    bands=num_bands, eType=etype, options=options)

    if gcps:
        new_gcps = [gdal.GCP(x, y, 0, col, row) for x, y, row, col in gcps]
        gtiff.setGCPs(new_gcps)
    else:
        gtiff.SetGeoTransform(geotransform)

    srs = _proj4_to_srs(proj4_str)
    gtiff.SetProjection(srs.ExportToWkt())

    for idx in range(num_bands):
        gtiff_band = gtiff.GetRasterBand(idx + 1)

        if num_bands == 1:
            band_data = data
        else:
            band_data = data[idx]

        # if log_level <= logging.DEBUG:
        #     LOG.debug("Data min: %f, max: %f" % (band_data.min(), band_data.max()))

        # Write the data
        if gtiff_band.WriteArray(band_data) != 0:
            LOG.error("Could not write band 1 data to geotiff '%s'" % (output_filename,))
            raise ValueError("Could not write band 1 data to geotiff '%s'" % (output_filename,))

        # Set No Data value
        if nodata is not None:
            gtiff_band.SetNoDataValue(nodata)

    if meta is not None:
        for k, v in meta.items():
            if isinstance(v, (list, tuple, set, np.ndarray)):
                v = ",".join((str(x) for x in v))
            else:
                v = str(v)
            gtiff.SetMetadataItem(k, v)

    if quicklook:
        png_filename = output_filename.replace(os.path.splitext(output_filename)[1], ".png")
        png_driver = gdal.GetDriverByName("PNG")
        png_driver.CreateCopy(png_filename, gtiff)

        # Garbage collection/destructor should close the file properly


def ahi_image_data(input_filename):
    nc = Dataset(input_filename, "r")

    if "albedo" in nc.variables:
        input_data = nc.variables["albedo"][:]
    else:
        input_data = nc.variables["brightness_temp"][:]
    input_data = input_data.astype(np.float32)  # make sure everything is 32-bit floats
    return input_data.filled(np.nan)


def ahi_image_info(input_filename):
    nc = Dataset(input_filename, "r")

    projection_info = nc.variables["Projection"].__dict__.copy()
    projection_info["semi_major_axis"] *= 1000.0  # is in kilometers, need meters
    projection_info["perspective_point_height"] *= 1000.0  # is in kilometers, need meters
    projection_info["flattening"] = 1.0 / projection_info["inverse_flattening"]
    # projection_info["semi_minor_axis"] = projection_info["semi_major_axis"] * (1 - projection_info["flattening"])
    # Shouldn't be needed: 6356752.29960574448

    # Notes on Sweep: https://trac.osgeo.org/proj/wiki/proj%3Dgeos
    # If AHI behaves like GOES then sweep should be Y
    # +over is needed so that we do 0-360 lon/lats and x/y
    input_proj_str = "+proj=geos +over "
    input_proj_str += "+lon_0={longitude_of_projection_origin:0.3f} "
    input_proj_str += "+lat_0={latitude_of_projection_origin:0.3f} "
    input_proj_str += "+a={semi_major_axis:0.3f} "
    input_proj_str += "+f={flattening} "
    # input_proj_str += "+b={semi_minor_axis:0.3f} "
    input_proj_str += "+h={perspective_point_height} "
    input_proj_str += "+sweep={sweep_angle_axis}"
    input_proj_str = input_proj_str.format(**projection_info)
    LOG.debug("AHI Fixed Grid Projection String: %s", input_proj_str)

    # Calculate upper-left corner (origin) using center point as a reference
    shape = (len(nc.dimensions["y"]), len(nc.dimensions["x"]))
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


def ahi2gtiff(input_filename, output_filename):
    LOG.debug("Opening input netcdf4 file: %s", input_filename)
    data = ahi_image_data(input_filename)
    info = ahi_image_info(input_filename)
    return create_ahi_geotiff(info, data, output_filename)


def create_ahi_geotiff(info, meta, data, output_filename, **kwargs):
    etype = gdal.GDT_Byte if data.dtype == np.uint8 else gdal.GDT_Float32
    # origin_x, cell_width, rotation_x, origin_y, rotation_y, cell_height
    geotransform = (info["origin_x"], info["cell_width"], 0, info["origin_y"], 0, info["cell_height"])
    create_geotiff(data, output_filename, info["proj"], geotransform, etype=etype, meta=meta, **kwargs)


def main():
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Convert AHI NetCDF files to Geotiff images")
    parser.add_argument("input_filename",
                        help="Input AHI NetCDF file")
    parser.add_argument("-o", "--output", dest="output_filename", default=None,
                        help="Output geotiff filename")
    parser.add_argument('-v', '--verbose', dest='verbosity', action="count", default=0,
                        help='each occurrence increases verbosity 1 level through '
                             'ERROR-WARNING-Info-DEBUG (default Info)')
    args = parser.parse_args()

    levels = [logging.ERROR, logging.WARN, logging.INFO, logging.DEBUG]
    logging.basicConfig(level=levels[min(3, args.verbosity)])

    if args.output_filename is None:
        stem = os.path.splitext(args.input_filename)[0]
        args.output_filename = stem + ".tif"

    ahi2gtiff(args.input_filename, args.output_filename)


if __name__ == "__main__":
    sys.exit(main())
