import logging
import re
from typing import Dict

from uwsift import config
from uwsift.common import Info
from uwsift.view.colormap import COLORMAP_MANAGER
from uwsift.workspace.guidebook import Guidebook

LOG = logging.getLogger(__name__)


def get_default_colormap(layer_info: dict,
                         guidebook: Guidebook) -> str:
    """
    Return the name of a colormap configured in 'default_colormaps' for
    the standard name taken from the given ``layer_info``.

    If no according colormap is configured or the configuration yields an
    unknown colormap name (in that case a warning is logged), the colormap
    provided by the `guidebook` is returned.
    """

    layer_standard_name = layer_info[Info.STANDARD_NAME]
    if layer_standard_name is None:
        LOG.debug("Cannot determine default colormap from configuration "
                  "for layer which does not have a standard name.")
    else:
        colormap_name = config.get('.'.join(["default_colormaps",
                                            layer_standard_name]),
                                   None)
        if colormap_name in COLORMAP_MANAGER:
            LOG.debug(f"Returning color map '{colormap_name}' as configured for"
                      f" layer standard name '{layer_standard_name}'.")
            return colormap_name

        if colormap_name:
            LOG.warning(f"Unknown color map '{colormap_name}' configured for"
                        f" layer standard name '{layer_standard_name}'. "
                        f" Falling back to internal Guidebook mapping.")

    return guidebook.default_colormap(layer_info)


# FIXME move this to a better location
DEFAULT_POINT_STYLE_UNKNOWN = "unknown"


def get_default_point_style_name(layer_info: dict) -> str:
    # FIXME actually to have this consistent with IMAGE layer default colormap
    #  selection the name chosen should be layer_info[Info.STANDARD_NAME] only
    #  as done by the Guidebook, but the STANDARD_NAME is not set for points
    #  data (yet).
    #  It needs to be clarified which approach is the right one - either making
    #  sure the STANDARD_NAME is always set or implementing a common fallback
    #  strategy.
    layer_identifying_name = layer_info.get(Info.STANDARD_NAME, None)
    if not layer_identifying_name:
        layer_identifying_name = layer_info.get(Info.SHORT_NAME, None)
    if not layer_identifying_name:
        layer_identifying_name = layer_info.get(Info.LONG_NAME, None)
    if not layer_identifying_name:
        layer_identifying_name = layer_info.get("name", None)
    if not layer_identifying_name:
        LOG.warning(f"Layer has no name."
                    f" Cannot determine a default point style."
                    f" Layer info: {layer_info}")
        return DEFAULT_POINT_STYLE_UNKNOWN

    point_style_name = \
        config.get(f"default_point_styles.{layer_identifying_name}", None)
    if not point_style_name:
        LOG.warning(f"No default point style configured for layer"
                    f" '{layer_identifying_name}'")
        return DEFAULT_POINT_STYLE_UNKNOWN
    # Check, whether there is a style defined for the given name
    point_style = get_point_style_by_name(point_style_name)
    if not point_style:
        LOG.warning(f"Unknown point style '{point_style_name}' configured for"
                    f" layer '{layer_identifying_name}'.")
        return DEFAULT_POINT_STYLE_UNKNOWN
    return point_style_name


def get_point_style_by_name(point_style_name: str) -> Dict[str, str]:
    point_style = config.get(f"point_styles.{point_style_name}", {})
    return point_style


# FIXME Move all these constants and functions to a styling related module - or,
#  preferable: get Vispy guys to implement a SVG/HTML styling compatible
#  interface

# Matches a CSS <length> (https://drafts.csswg.org/css-values-3/#length-value)
# but (for now) only for units 'px' and '%' (case insensitively)
STYLE_LENGTH_REGEX = re.compile(r"^(\d+)(?i:(px|%))")

STYLE_KEYWORD_SYMBOL = "symbol"
STYLE_KEYWORD_SIZE = "size"
STYLE_KEYWORD_STROKE = "stroke"
STYLE_KEYWORD_STROKE_WIDTH = "stroke-width"
STYLE_KEYWORD_FILL = "fill"

COLOR_TRANSPARENT = "#00000000"

def map_point_style_to_marker_kwargs(point_style: dict):
    """
    Map a dictionary containing style settings to the Vispy Markers interface.

    The mappings (tries to) implement a subset of SVG/HTML styling
    specifications.

    TODO:
      Verify specification conformance.

    REFERENCES:
      * https://www.w3.org/TR/SVG/styling.html
      * https://www.w3.org/TR/SVG/painting.html

    :param point_style: dictionary of SVG/HTML style settings for the markers
    :return: kwargs to be used when calling a Markers constructor
    """
    kwargs = {
        "symbol": point_style.get(STYLE_KEYWORD_SYMBOL, "cross")
    }

    size_value = point_style.get(STYLE_KEYWORD_SIZE, "9px")
    size_match = STYLE_LENGTH_REGEX.search(size_value)
    if size_match:
        kwargs["size"] = float(size_match.group(1))
        if size_match.group(2) == "%":
            kwargs["scaling"] = True
    else:
        raise ValueError(f"Unrecognized setting for 'size': {size_value}")

    # TODO
    #  If markers without outline (stroke) are possible, the default should
    #  become None here and for that default kwargs should be constructed
    #  resulting in no outline
    stroke_value = point_style.get(STYLE_KEYWORD_STROKE, "white")
    kwargs["edge_color"] = stroke_value

    stroke_width_value = point_style.get(STYLE_KEYWORD_STROKE_WIDTH, "1px")
    stroke_width_match = STYLE_LENGTH_REGEX.search(stroke_width_value)
    if stroke_width_match:
        if stroke_width_match.group(2) == "%":
            kwargs["edge_width_rel"] = float(stroke_width_match.group(1))
            kwargs["edge_width"] = None  # overwrite default value of MarkerVisual.set_data()
        else:
            kwargs["edge_width"] = float(stroke_width_match.group(1))
    else:
        raise ValueError(f"Unrecognized value for 'border_width':"
                         f" {stroke_width_value}")

    # TODO
    #  If markers without fill are possible, the default should become None here
    #  and for that default kwargs should be constructed resulting in no fill
    fill_value = point_style.get(STYLE_KEYWORD_FILL, COLOR_TRANSPARENT)
    kwargs["face_color"] = fill_value

    return kwargs
