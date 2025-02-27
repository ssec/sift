.. role:: yaml(code)

Configuring Area Definitions
----------------------------

SIFT allows only map projections which are provided by configuration via
Satpy. All or some of them must be "activated" for use by configuring
``area_definitions`` as follows::

    area_definitions:
      [DISPLAY_NAME_AREA_GROUP_1]:
        {
          [DISPLAY_RESOLUTION_IDENTIFIER_1_1] : [AREA_ID_1_1],
          [DISPLAY_RESOLUTION_IDENTIFIER_1_2] : [AREA_ID_1_2],
          ...
        }
      ...
      [DISPLAY_NAME_AREA_GROUP_N]:
        {
          [DISPLAY_RESOLUTION_IDENTIFIER_N_1] : [AREA_ID_N_1],
          [DISPLAY_RESOLUTION_IDENTIFIER_N_2] : [AREA_ID_N_2],
          ...
        }

where ``DISPLAY_NAME_AREA_GROUP_i`` is the name to be used in the GUI to refer to the area group. Each
``DISPLAY_NAME_AREA_GROUP_i`` represents a group of areas that share the same projection but have
different resolutions. Therefore, within a single area group, there are one or more areas organized
in a key-value structure. The key denotes ``DISPLAY_RESOLUTION_IDENTIFIER_i_j``, which is used in the GUI
to represent resolutions for the selected area group. The value represents ``AREA_ID_i_j``, by which a
specific area definition is reached. Area ID ``AREA_ID_i_j`` must be provided by according Satpy
configuration (unknown area ids are skipped with a warning log message).

The area groups appear in the *Projection:* picklist in the same order they are listed in the
``area_definitions`` configuration, with the first entry selected at application start. The resolution
identifiers appear in the *Resolution:* picklist in the Open File Wizard window based on the selected
projection (area group). They are listed in the same order as within a single area group in the
``area_definitions`` configuration, and by default, the first resolution identifier entry is selected.
In this way, the user has the possibility to select the projection and the resolution and based on that the area
ID is determined.

One additional area definition is appended as a fallback by SIFT if no area
definition or none with a pseudo lat/lon projection (Plate Carree) is
found. This is to make sure there is always a projection showing the whole world
selectable in the application, which is useful for examining data of a yet unknown
area.

**Example**::

  area_definitions:
    MTG FCI FDSS:
      {
        1km: mtg_fci_fdss_1km,
        2km: mtg_fci_fdss_2km,
        500m: mtg_fci_fdss_500m,
        32km: mtg_fci_fdss_32km
      }

    MSG SEVIRI FES:
      {
        3km: msg_seviri_fes_3km,
        1km: msg_seviri_fes_1km,
        9km: msg_seviri_fes_9km
      }
