Configuring a Colormap and its Colorlimits Based on the Dataset Standard Name
-----------------------------------------------------------------------------

SIFT automatically associates a colormap and a value range to a newly
loaded dataset based on the standard name of each dataset via its internal
*Guidebook*. The colormap is applied to the values in the range, values
outside this range get the first respectively last color of the colormap.

This association can be overwritten by configuring a colormap and range
(optional) as follows::

    default_colormaps:
        STANDARD_NAME_1:
            colormap: COLORMAP_NAME_1
            range: RANGE_1
        STANDARD_NAME_2:
            colormap: COLORMAP_NAME_2
        ...

**Example** ::

    default_colormaps:
         toa_bidirectional_reflectance:
            colormap: IR Color Clouds Summer
            range: [-1.2, 120]
         toa_brightness_temperature:
            colormap: Square Root (Vis Default)


If a given ``COLORMAP_NAME_<N>`` is not registered internally a warning is
logged and the colormap is chosen by the Guidebook as fallback. The same
happens for all standard names which are not listed in the configuration.

.. note ::

    When file based caching is active changing the associated default colormap or clims
    by configuration will not affect data which is already cached from an
    earlier run of SIFT because the colormap is stored as meta-data in the
    cache.  In such a case you may destroy the cache (usually at
    ``~/.cache/SIFT``) or manually choose the wanted colormap after loading the
    data.
