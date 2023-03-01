.. role:: yaml(code)

Configuring Area Definitions
----------------------------

SIFT allows only map projections which are provided by configuration via
Satpy. All or some of them must be "activated" for use by configuring
``area_definitions`` as follows::

    area_definitions:
      [DISPLAY_NAME_1]: [AREA_ID_1]
      ...
      [DISPLAY_NAME_N]: [AREA_ID_N]

were ``DISPLAY_NAME_i`` is the name to be used in the GUI to refer to the area
definition with the according ID ``AREA_ID_i`` which must be provided by
according Satpy configuration (unknown area ids are skipped with a warning
log message).

The area definitions appear in the *Projection:* picklist in the same order they
are listed in the ``area_definitions`` configuration, the first entry is
selected at application start.

One additional area definition is appended as fallback by SIFT if no area
definition or none with a pseudo lat/lon projection (Plate Carree) is
found. This is to make sure there is always projection showing the whole world
selectable in the application, which is useful for examining data of yet unknown
area.

**Example**::

  area_definitions:
    MSG SEVIRI FES 3km: msg_seviri_fes_3km
    MSG SEVIRI RSS 1km: msg_seviri_rss_1km
    MSG SEVIRI IODC 3km: msg_seviri_iodc_3km
    MTG FCI FDSS 2km: mtg_fci_fdss_2km
