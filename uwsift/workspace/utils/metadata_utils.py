import logging

from uwsift.common import Info
from uwsift.view.colormap import COLORMAP_MANAGER
from uwsift.workspace.guidebook import Guidebook
from uwsift import config

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


