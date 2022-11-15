Configuring a Colormap Based on the Dataset Standard Name
-------------------------------------------------------

MTG-SIFT automatically associates a colormap to a newly loaded dataset based on
the standard name of each dataset via its internal *Guidebook*. This
association can be overwritten by configuring a different colormaps as follows::

    default_colormaps:
        STANDARD_NAME_1: COLORMAP_NAME_1
        STANDARD_NAME_2: COLORMAP_NAME_2
        ...

**Example** ::

    default_colormaps:
         toa_bidirectional_reflectance: IR Color Clouds Summer
         toa_brightness_temperature: Square Root (Vis Default)


If a given ``COLORMAP_NAME_<N>`` is not registered internally a warning is
logged and the colormap is chosen by the Guidebook as fallback. The same
happens for all standard names which are not listed in the configuration.

.. note ::

    When file based caching is active changing the associated default colormap
    by configuration will not affect data which is already cached from an
    earlier run of MTG-SIFT because the colormap is stored as meta-data in the
    cache.  In such a case you may destroy the cache (usually at
    ``~/.cache/SIFT``) or manually choose the wanted colormap after loading the
    data.
