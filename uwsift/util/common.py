from __future__ import annotations

import logging
import math
from datetime import datetime
from decimal import Decimal
from typing import List, Optional, Set

import numpy as np
import satpy.readers.hrit_base
from satpy import DataID, Scene

from uwsift import config as config
from uwsift.common import INVALID_COLOR_LIMITS, NAN, Info, Kind

LOG = logging.getLogger(__name__)


def is_datetime_format(format_str):
    if not format_str:
        return False
    format_result = datetime.today().strftime(format_str)
    return format_str != format_result


def is_same_proj(proj_a: str, proj_b: str) -> bool:
    """
    Compare the given proj strings and consider them the same if numeric
    parameters do not differ in the first 9 significant digits.

    The order of proj parameters and possible differences in case are ignored.
    """
    if proj_a == proj_b:
        return True

    proj_a_args = proj_a.casefold().split()
    proj_b_args = proj_b.casefold().split()
    if len(proj_a_args) != len(proj_b_args):
        return False

    proj_a_args.sort()
    proj_b_args.sort()
    for arg_a, arg_b in zip(proj_a_args, proj_b_args):
        if arg_a == arg_b:
            continue

        arg_a_parameter, arg_a_value = arg_a.split("=", 1)
        arg_b_parameter, arg_b_value = arg_a.split("=", 1)
        if arg_a_parameter != arg_b_parameter:
            return False

        if arg_a_value == arg_b_value:
            continue

        try:
            a = float(arg_a_value)
            b = float(arg_b_value)
        except ValueError:
            return False

        if not math.isclose(a, b, rel_tol=1e-09):
            return False

    return True


def get_reader_kwargs_dict(reader_names):
    """Extract the reader_kwargs from the config for multiple readers."""
    reader_kwargs = {}
    for reader_name in reader_names:
        reader_kwargs.update({reader_name: config.get(f"data_reading.{reader_name}.reader_kwargs", {})})
    return reader_kwargs


def create_scenes(scenes: dict, file_groups: dict) -> List[DataID]:
    """Create Scene objects for the selected files."""
    all_available_products: Set[DataID] = set()
    for group_id, file_group in file_groups.items():
        scn = scenes.get(group_id)
        if scn is None:
            # need to create the Scene for the first time
            # file_group includes what reader to use
            # NOTE: We only allow a single reader at a time
            reader_kwargs = get_reader_kwargs_dict(file_group.keys())
            try:
                scn = Scene(filenames=file_group, reader_kwargs=reader_kwargs)
            except ValueError as e:
                LOG.warning(e)
                continue
            scenes[group_id] = scn

            if _scene_contains_compressed_seviri_hrit_files(scn):
                # To decompress compressed SEVIRI HRIT files, an external
                # decompression tool (xRITDecompress) must be executable by the
                # Satpy HRIT reader (its file path is communicated via the
                # XRIT_DECOMPRESS_PATH environment variable).
                # Even though the creation of the scene was successful, even if
                # the decompression tool is missing, in this case the loading
                # of the scene would fail.
                # To enable the caller of create_scenes() to fail early and
                # inform the user about the problem we check beforehand if the
                # decompression tool is available: if not, the following call
                # will throw an exception which is intentionally not caught
                # here.
                satpy.readers.hrit_base.get_xritdecompress_cmd()

        all_available_products.update(scn.available_dataset_ids(composites=True))

    # update the widgets
    return sorted(all_available_products)


def _scene_contains_compressed_seviri_hrit_files(scn):
    # TODO: Scene may not provide information about reader in the
    #  future - here the "protected" variable '_readers' is used as
    #  workaround already
    for r in scn._readers.values():
        # only perform check when using a relevant reader, so that this is not
        # triggered mistakenly when another reader uses the same metadata key
        # 'compression_flag_for_data' but does not need xRITDecompress to
        # uncompress
        if r.name in ["seviri_l1b_hrit"]:
            for fh in r.file_handlers.values():
                for fh2 in fh:
                    if fh2.mda.get("compression_flag_for_data"):
                        return True
    return False


# TODO: use cf-units or cfunits or SIUnits or other package or: put all these
#  units and their accepted spellings into a dictionary. Also try to stay in
#  sync with Satpy, but note, that Satpy doesn't seem to have a well designed
#  units handling concept either
#
# First entry is the standard symbol used for display
Unit_Strings_Kelvin = ["K", "Kelvin", "kelvin"]
Unit_Strings_degC = ["°C", "C", "degrees_Celsius"]

# TODO: this may be taken from the units config too in the future?
Temperature_Quantities = ["brightness_temperature", "toa_brightness_temperature"]


def unit_symbol(unit):
    # FUTURE: Use cfunits or cf_units package
    # cf_units gives the wrong symbol for Celsius
    if unit == "1":
        return ""
    elif unit == "%":
        return "%"
    elif unit in Unit_Strings_degC:
        return Unit_Strings_degC[0]
    elif unit in Unit_Strings_Kelvin:
        return Unit_Strings_Kelvin[0]
    else:
        return unit or ""


def _unit_format_func(info, units):
    units = unit_symbol(units)
    standard_name = info.get(Info.STANDARD_NAME)

    if (standard_name is not None) and (standard_name in Temperature_Quantities):
        # BT data limits, Kelvin to degC
        def _format_unit(val, numeric=True, include_units=True):
            unit_str = units if include_units else ""
            if np.isnan(val):
                return f"{NAN}{unit_str}"
            else:
                return f"{val:.02f}{unit_str}"

    elif "flag_values" in info:
        _format_unit = _unit_format_func_for_flags(info)

    else:
        # default formatting string
        def _format_unit(val, numeric=True, include_units=True):
            unit_str = units if include_units else ""
            if np.isnan(val):
                return f"{NAN}{unit_str}"
            else:
                return f"{val:.03f}{unit_str}"

    return _format_unit


def _unit_format_func_for_flags(info):
    # flag values don't have units
    if "flag_meanings" in info:
        flag_masks = info["flag_masks"] if "flag_masks" in info else [-1] * len(info["flag_values"])
        flag_info = tuple(zip(info["flag_meanings"], info["flag_values"], flag_masks))

        def _format_unit(val, numeric=True, include_units=True, flag_info=flag_info):
            if np.isnan(val):
                return NAN

            val = int(val)
            if numeric:
                return f"{val}"

            meanings = []
            for fmean, fval, fmask in flag_info:
                if (val & fmask) == fval:
                    meanings.append(fmean)
            return f"{val} ({', '.join(meanings)})"

    else:

        def _format_unit(val, numeric=True, include_units=True, flag_info=None):
            if np.isnan(val):
                return NAN
            else:
                return f"{val}"

    return _format_unit


def preferred_units(info) -> str:
    """
    Return unit string (i.e.: Kelvin) for a Product currently being loaded.
    :param info: metadata information describing the product currently being added.
    :return: String describing the preferred unit for the product described in info.
    """
    # FUTURE: Use cfunits or cf_units package

    default_temperature_unit = Unit_Strings_Kelvin[0]
    lookup_name = info.get(Info.STANDARD_NAME)  # may return 'None', that's OK
    if lookup_name == "toa_bidirectional_reflectance":
        return "1"
    elif lookup_name in Temperature_Quantities:
        calibration_to_temperature_unit_map = config.get("units.temperature", default=None)
        if calibration_to_temperature_unit_map is None:
            LOG.info(f"No configuration for unit of temperature found..." f" Reverting to {default_temperature_unit}.")
            return default_temperature_unit
        elif lookup_name in calibration_to_temperature_unit_map:
            temperature_unit = calibration_to_temperature_unit_map[lookup_name]
        else:
            lookup_name = "all"  # For the warning message below
            temperature_unit = calibration_to_temperature_unit_map.get(lookup_name, default_temperature_unit)
        if temperature_unit not in Unit_Strings_Kelvin and temperature_unit not in Unit_Strings_degC:
            LOG.warning(
                f"Unit '{temperature_unit}' as configured for"
                f" '{lookup_name}' is not a known temperature unit."
                f" Reverting to {default_temperature_unit}."
            )
            return default_temperature_unit
        return temperature_unit

    else:
        return info.get(Info.UNITS, None)


def units_conversion(info):
    """return UTF8 unit string, lambda v,inverse=False: convert-raw-data-to-unit,
    format string for converted value with unit."""
    # the dataset might be in one unit, but the user may want something else
    # FUTURE: Use cfunits or cf_units package
    punits = preferred_units(info)

    # Conversion functions
    # FUTURE: Use cfunits or cf_units package
    if info.get(Info.UNITS) in Unit_Strings_Kelvin and punits in Unit_Strings_degC:

        def conv_func(x, inverse=False):
            return x - 273.15 if not inverse else x + 273.15

    elif info.get(Info.UNITS) == "%" and punits == "1":

        def conv_func(x, inverse=False):
            return x / 100.0 if not inverse else x * 100.0

    else:

        def conv_func(x, inverse=False):
            return x

    # Format strings
    format_func = _unit_format_func(info, punits)
    return punits, conv_func, format_func


def format_clims(clims, unit_conv):
    if clims == INVALID_COLOR_LIMITS:
        return "N/A"

    display_clims = unit_conv[1](np.array(clims))
    min_str = unit_conv[2](display_clims[0], include_units=False)
    max_str = unit_conv[2](display_clims[1])
    clims_str = f"{min_str} ~ {max_str}"
    return clims_str


def format_wavelength(wl):
    """Return wavelength formatted as string with unit.

    A wavelength below 4.1 micrometers is formatted with two, above with one decimal.
    """
    if wl < 4.1:
        wl_str = f"{wl:0.02f} µm"
    else:
        wl_str = f"{wl:0.01f} µm"
    return wl_str


def format_resolution(resolution: float) -> str:
    """Return string representing the given resolution (interpreted in meters)
    in the unit km for values greater than 1 km, otherwise in m, including the
    unit symbol."""
    if isinstance(resolution, np.number):
        # Decimal requires a python numeric value
        resolution = resolution.item()
    resolution_dec = Decimal(resolution)
    if resolution_dec < 1000:
        return str(resolution_dec.quantize(Decimal("1."))) + " m"
    else:
        resolution_dec /= Decimal(1000)
        return str(resolution_dec.quantize(Decimal(".1"))) + " km"


def normalize_longitude(lon: float) -> float:
    """Return longitude centered around prime meridian, i.e. in the range ]-180.0, 180.0]"""
    # Fiddling with '-' is required to include 180.0 and exclude -180.0 from the range,
    # the code would be neater the other way around: (lon + 180.0) % 360.0 - 180.0
    return -((-lon + 180.0) % 360.0 - 180.0)


def get_initial_gamma(info):
    """Determine initial gamma value based on 'info'"""
    gamma = 1.0
    if info.get(Info.KIND) == Kind.RGB:
        gamma = (1.0,) * 3
    elif hasattr(info, "layers"):
        gamma = (1.0,) * len(info.layers)
    return gamma


def range_hull(range_1: tuple, range_2: tuple) -> tuple:
    """Get a range which contains the given ranges.

    If the given ranges don't overlap the resulting range also contains the range between them. Thus, the result
    can be more than the union of the given ranges.

    Here a 'range' is defined by a pair of numbers a, b. The range is considered empty, if a > b."""
    return min(range_1[0], range_2[0]), max(range_1[1], range_2[1])


def range_hull_no_fail(range_1: Optional[tuple], range_2: Optional[tuple], fallback: tuple) -> tuple:
    """Get a range which contains the given ranges, safely.

    Same as 'range_hull()' but if the range can not be computed from 'range_1' and 'range_2' then the given
    'fallback' is returned. This happens if any of 'range_1' and 'range_2' is None or (None, None).
    """

    try:
        # 'range_1' or 'range_2' can be None or (None, None), but this method exists solely to deal with that case.
        # That's why we silence mypy by ignoring its findings here:
        lower = min(range_1[0], range_2[0])  # type: ignore
        upper = max(range_1[1], range_2[1])  # type: ignore
    except TypeError:
        return fallback

    return lower, upper
