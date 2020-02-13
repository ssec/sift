from enum import Enum
import logging
from uwsift import config

LOG = logging.getLogger(__name__)


# Data mapping: parameters to describe how data is oriented when passed to SIFT so that it can be
# transformed to a normalized SIFT-internal representation. Based on:
# https://apps.ecmwf.int/codes/grib/format/grib1/flag/8/ and eccodes implementation:
# /usr/share/eccodes/definitions/grib2/template.3.scanning_mode.def
#
# Examples: abi_l1b:         all: False (default)
#           seviri_l1b_hrit: both scan directions: True, j-consecutive: False (default)
#
class DataMapping(Enum):
    # i direction: west to east along a parallel, or left to right along an X-axis
    I_SCANS_NEGATIVELY = "iScansNegatively"
    # j direction: south to north along a meridian, or bottom to top along a Y-axis
    J_SCANS_POSITIVELY = "jScansPositively"
    #
    J_POINTS_ARE_CONSECUTIVE = "jPointsAreConsecutive"


# Key in .yaml files where data mapping information is stored for each reader
DATA_MAPPING_CONFIG_KEY = 'data_mapping'


def get_flipping_parameters(reader):
    """Read current data mapping configuration for provided 'reader' and translate to actions
    required to transform data and area to a normalized SIFT-internal representation"""

    flip_left_right = False
    flip_up_down = False
    swap_axes = False

    base_path = DATA_MAPPING_CONFIG_KEY + '.' + reader + '.'

    flip_left_right = config.get(base_path + DataMapping.I_SCANS_NEGATIVELY.value, False)
    flip_up_down = config.get(base_path + DataMapping.J_SCANS_POSITIVELY.value, False)

    # swap_axes = config.get(base_path + DataMapping.J_POINTS_ARE_CONSECUTIVE.value, False)
    if config.get(base_path + DataMapping.J_POINTS_ARE_CONSECUTIVE.value, None) is not None:
        LOG.warning(f"Data mapping: '{DataMapping.J_POINTS_ARE_CONSECUTIVE.value}' is "
                    "not yet supported.")

    return flip_left_right, flip_up_down, swap_axes


def flip_data_for_sift(original_data, reader):
    """Flip data according to data mapping configuration of the reader"""

    import dask.array as da

    data = original_data
    flip_left_right, flip_up_down, swap_axes = get_flipping_parameters(reader)

    if flip_left_right:
        data = da.fliplr(data)
    if flip_up_down:
        data = da.flipud(data)
    # if swap_axes:
    #     data = da.swapaxes(data, 0, 1)  # TODO: check if this implementation would be correct

    return data


def flip_area_for_sift(original_area, reader):
    """Flip area according to data mapping configuration of the reader"""

    from pyresample.geometry import AreaDefinition
    if not isinstance(original_area, AreaDefinition):
        raise NotImplementedError("Only AreaDefinition datasets can be loaded at this time.")

    original_extent = list(original_area.area_extent)
    flipped_extent = original_extent.copy()
    flip_left_right, flip_up_down, swap_axes = get_flipping_parameters(reader)

    if flip_left_right:
        flipped_extent[0] = original_extent[2]
        flipped_extent[2] = original_extent[0]
    if flip_up_down:
        flipped_extent[1] = original_extent[3]
        flipped_extent[3] = original_extent[1]
    # if swap_axes:
    #    pass  # TODO: check if this implementation would be correct

    override_kwargs = {
        'area_extent': tuple(flipped_extent)
    }

    # Copy existing area but update all properties related to the area_extent
    return original_area.copy(**override_kwargs)
