#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Script to convert AHI NetCDF files in to a geotiff image.

:author: David Hoese <david.hoese@ssec.wisc.edu>
:copyright: 2015 by University of Wisconsin Regents, see AUTHORS for more details
:license: GPLv3, see LICENSE for more details
"""

import os
import sys
import logging

from netCDF4 import Dataset
from osgeo import gdal
import osr
from pyproj import Proj
import numpy as np

LOG = logging.getLogger(__name__)
GTIFF_DRIVER = gdal.GetDriverByName("GTIFF")
DEFAULT_PROJ_STR = "+proj=merc +datum=WGS84 +ellps=WGS84 +no_defs"


AHI_NADIR_RES = {
    5500: 2000,
}


### Scaling Functions Copied from Polar2Grid ###

def linear_flexible_scale(img, min_out, max_out, min_in=None, max_in=None, flip=False, **kwargs):
    """Flexible linear scaling by specifying what you want output, not the parameters of the linear equation.

    This scaling function stops humans from doing math...let the computers do it.

    - If you aren't sure what the valid limits of your data are, only specify
        the min and max output values. The input minimum and maximum will be
        computed. Note that this could add a considerable amount of time to
        the calculation.
    - If you know the limits, specify the output and input ranges.
    - If the data needs to be clipped to the output range, specify 1 or 0 for
        the "clip" keyword. Note that most backends will do this to fit the
        data type of the output format.
    """
    LOG.debug("Running 'linear_flexible_scale' with (min_out: %f, max_out: %f)..." % (min_out, max_out))

    # Assume masked arrays for ahi2gtiff.py:
    min_in = img.min() if min_in is None else min_in
    max_in = img.max() if max_in is None else max_in
    if min_in == max_in:
        # Data doesn't differ...at all
        LOG.warning("Data does not differ (min/max are the same), can not scale properly")
        max_in = min_in + 1.0
    LOG.debug("Input minimum: %f, Input maximum: %f" % (min_in, max_in))

    if flip:
        m = (min_out - max_out) / (max_in - min_in)
        b = max_out - m * min_in
    else:
        m = (max_out - min_out) / (max_in - min_in)
        b = min_out - m * min_in
    LOG.debug("Linear parameters: m=%f, b=%f", m, b)

    if m != 1:
        np.multiply(img, m, img)
    if b != 0:
        np.add(img, b, img)

    return img


def sqrt_scale(img, min_out, max_out, inner_mult=None, outer_mult=None, min_in=0.0, max_in=1.0, **kwargs):
    """Square root enhancement

    Note that any values below zero are clipped to zero before calculations.

    Default behavior (for regular 8-bit scaling):
        new_data = sqrt(data * 100.0) * 25.5
    """
    LOG.debug("Running 'sqrt_scale'...")
    if min_out != 0 and min_in != 0:
        raise RuntimeError("'sqrt_scale' does not support a `min_out` or `min_in` not equal to 0")
    inner_mult = inner_mult if inner_mult is not None else (100.0 / max_in)
    outer_mult = outer_mult if outer_mult is not None else max_out / np.sqrt(inner_mult * max_in)
    LOG.debug("Sqrt scaling using 'inner_mult'=%f and 'outer_mult'=%f", inner_mult, outer_mult)
    img[img < 0] = 0  # because < 0 cant be sqrted

    if inner_mult != 1:
        np.multiply(img, inner_mult, img)

    np.sqrt(img, out=img)

    if outer_mult != 1:
        np.multiply(img, outer_mult, img)

    np.round(img, out=img)

    return img

def brightness_temperature_scale(img, threshold, min_in, max_in, min_out, max_out,
                                 threshold_out=None, units="kelvin", **kwargs):
    """Brightness temperature scaling is a piecewise function with two linear sub-functions.

    Temperatures less than `threshold` are scaled linearly from `threshold_out` to `max_out`. Temperatures greater than
    or equal to `threshold` are scaled linearly from `min_out` to `threshold_out`.

    In previous versions, this function took the now calculated linear parameters, ``m`` and ``b`` for
    each sub-function. For historical documentation here is how these were converted to the current method:

        equation 1: middle_file_value = low_max - (low_mult * threshold)
        equation 2: max_out = low_max - (low_mult * min_temp)
        equation #1 - #2: threshold_out - max_out = low_mult * threshold + low_mult * min_temp = low_mult (threshold + min_temp) => low_mult = (threshold_out - max_out) / (min_temp - threshold)
        equation 3: middle_file_value = high_max - (high_mult * threshold)
        equation 4: min_out = high_max - (high_mult * max_temp)
        equation #3 - #4: (middle_file_value - min_out) = high_mult * (max_in - threshold)

    :param units: If 'celsius', convert 'in' parameters from kelvin to degrees celsius before performing calculations.

    """
    LOG.debug("Running 'bt_scale'...")
    if units == "celsius":
        min_in -= 273.15
        max_in -= 273.15
        threshold -= 273.15
    threshold_out = threshold_out if threshold_out is not None else (176 / 255.0) * max_out
    low_factor = (threshold_out - max_out) / (min_in - threshold)
    low_offset = max_out + (low_factor * min_in)
    high_factor = (threshold_out - min_out) / (max_in - threshold)
    high_offset = min_out + (high_factor * max_in)
    LOG.debug("BT scale: threshold_out=%f; low_factor=%f; low_offset=%f; high_factor=%f; high_offset=%f",
              threshold_out, low_factor, low_offset, high_factor, high_offset)

    high_idx = img >= threshold
    low_idx = img < threshold
    img[high_idx] = high_offset - (high_factor * img[high_idx])
    img[low_idx] = low_offset - (low_factor * img[low_idx])
    return img

### End of Scaling Copy ###


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


def create_geotiff(data, output_filename, proj4_str, geotransform, etype=gdal.GDT_UInt16, compress=None,
                   quicklook=False, gcps=None, **kwargs):
    """Function that creates a geotiff from the information provided.
    """
    log_level = logging.getLogger('').handlers[0].level or 0
    LOG.info("Creating geotiff '%s'" % (output_filename,))

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

        if log_level <= logging.DEBUG:
            LOG.debug("Data min: %f, max: %f" % (band_data.min(), band_data.max()))

        # Write the data
        if gtiff_band.WriteArray(band_data) != 0:
            LOG.error("Could not write band 1 data to geotiff '%s'" % (output_filename,))
            raise ValueError("Could not write band 1 data to geotiff '%s'" % (output_filename,))

    if quicklook:
        png_filename = output_filename.replace(os.path.splitext(output_filename)[1], ".png")
        png_driver = gdal.GetDriverByName("PNG")
        png_driver.CreateCopy(png_filename, gtiff)

        # Garbage collection/destructor should close the file properly


def main():
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Convert AHI NetCDF files to Geotiff images")
    parser.add_argument("input_filename",
                        help="Input AHI NetCDF file")
    parser.add_argument("-o", "--output", dest="output_filename", default=None,
                        help="Output geotiff filename")
    parser.add_argument('-v', '--verbose', dest='verbosity', action="count", default=0,
                        help='each occurrence increases verbosity 1 level through ERROR-WARNING-INFO-DEBUG (default INFO)')
    args = parser.parse_args()

    levels = [logging.ERROR, logging.WARN, logging.INFO, logging.DEBUG]
    logging.basicConfig(level=levels[min(3, args.verbosity)])

    if args.output_filename is None:
        stem = os.path.splitext(args.input_filename)[0]
        args.output_filename = stem + ".tif"

    LOG.debug("Opening input netcdf4 file: %s", args.input_filename)
    nc = Dataset(args.input_filename, "r")

    if "albedo" in nc.variables:
        input_data = nc.variables["albedo"][:]
        # input_data = linear_flexible_scale(input_data, 0, 255)
        # input_data = sqrt_scale(input_data, 0, 255)
    else:
        input_data = nc.variables["brightness_temp"][:]
        # input_data = brightness_temperature_scale(input_data, 242.0, 163.0, 330.0, 0, 255)
    # np.clip(input_data, 0, 255, out=input_data)
    # input_data = input_data.astype(np.uint8)
    input_data = input_data.astype(np.float32)  # make sure everything is 32-bit floats

    projection_info = nc.variables["Projection"].__dict__.copy()
    projection_info["semi_major_axis"] *= 1000.0  # is in kilometers, need meters
    projection_info["perspective_point_height"] *= 1000.0  # is in kilometers, need meters
    projection_info["flattening"] = 1.0 / projection_info["inverse_flattening"]
    # projection_info["semi_minor_axis"] = projection_info["semi_major_axis"] * (1 - projection_info["flattening"])
    # Shouldn't be needed: 6356752.29960574448

    # Notes on Sweep: https://trac.osgeo.org/proj/wiki/proj%3Dgeos
    # If AHI behaves like GOES then sweep should be Y
    input_proj_str = "+proj=geos "
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
    shape = nc.variables["longitude"].shape
    # center_x_idx = shape[1] / 2
    # center_y_idx = shape[0] / 2
    pixel_size_x = AHI_NADIR_RES[shape[0]]
    pixel_size_y = -AHI_NADIR_RES[shape[1]]
    # center_lon = nc.variables["longitude"][center_x_idx, center_y_idx]
    # center_lat = nc.variables["latitude"][center_x_idx, center_y_idx]
    # p = Proj(input_proj_str)
    # center_x, center_y = p(center_lon, center_lat)
    # origin_x = center_x - pixel_size_x * center_x_idx
    # origin_y = center_y - pixel_size_y * center_y_idx
    # LOG.debug("Center Lon: %f\tCenter Lat: %f", center_lon, center_lat)
    # LOG.debug("Center X: %f\tCenter Y: %f", center_x, center_y)

    # Assume that center pixel is at 0, 0 projection space
    origin_x = -pixel_size_x * (shape[1] / 2.0)
    origin_y = -pixel_size_y * (shape[0] / 2.0)
    LOG.debug("Origin X: %f\tOrigin Y: %f", origin_x, origin_y)

    # origin_x, cell_width, rotation_x, origin_y, rotation_y, cell_height
    geotransform = (origin_x, pixel_size_x, 0, origin_y, 0, pixel_size_y)
    # gcps = [(center_lon, center_lat, center_x_idx, center_y_idx),]

    etype = gdal.GDT_Byte if input_data.dtype == np.uint8 else gdal.GDT_Float32
    create_geotiff(input_data, args.output_filename, input_proj_str, geotransform, etype=etype)

if __name__ == "__main__":
    sys.exit(main())
