Configuring Readers
===================

General Configuration
---------------------

The readers offered by the Open File Wizard to be used for loading are a subset
of the readers provided by Satpy and - if configured accordingly via
``satpy_extra_config_path`` - in a directory of additional readers.
The list of these readers must be configured, e.g. as follows::

  data_reading:
    readers:
      - fci_l1c_nc
      - seviri_l1b_hrit
      - seviri_l1b_native
      - avhrr_l1b_eps

This is enough to make the readers available, but the Open File Wizard works
better with additional configuration.

Per-Reader Configuration
------------------------

General Configuration
^^^^^^^^^^^^^^^^^^^^^

For each reader it is recommended to
configure:

- ``filter_patterns``, which help to partition the names of files in the chosen
  input directory into their components and
- ``group_keys``, which define, which components of the file names are used to
  group files together into one dataset

These settings are configured in the same way as in the Satpy reader
configuration YAML files.

If special reader keyword arguments need to be passed to the reader during the
Satpy Scene initialisation, these can be configured by adding a
``reader_kwargs`` entry to the reader configuration, containing the
kwargs key-value pairs.

Grid Numbering Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In addition for GEOS files the grid layout can be configured to enable correct
index display according to their PUGs. In the ``grid`` sub-config three
parameters can be set:

- ``origin``, to define, which corner the indexing starts from. Valid settings
  are ``"SE"``, ``"SW"``, ``"NE"`` and ``"NW"``. If not set it defaults to ``"NW"``
- ``first_index_x`` and ``first_index_y`` to define the value indexing starts
  from for columns and rows. Usually this value is 0 or 1 and the same for both
  parameters and defaults to 0 if not given.

GEO Segment Merging Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For segmented file formats (such as MSG SEVIRI HRIT and MTG FCI NetCDF), it is
usually preferable to merge segments that are loaded in separate load operations
(Open File Wizard calls) but belong to the same disk into one dataset, rather than creating a
separate dataset for each load operation. This is especially important when
incrementally loading stripes in auto-update mode.

To control this behaviour the ``data_reading``-setting ``merge_with_existing``
can be configured as either ``False`` (the default) or ``True``.

.. note:: Currently, dataset merging does not work together with the caching
          database, so make sure you set ``storage.use_inventory_db:
          False``, otherwise caching will take precedence and disable merging.

.. note:: Currently, dataset merging does not work well together with the
          adaptively retiling image display. Consider switching it off by
          setting ``display.use_tiled_geolocated_images: False``, otherwise
          for some zoom levels the data segments loaded later may not be
          visible.

Example Configuration For Image Readers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As example for a complete configuration for a reader, this is one for SEVIRI
Level 1B in HRIT format::

  data_reading:
    seviri_l1b_hrit:
      group_keys: ['start_time', 'platform_shortname', 'service']
      filter_patterns: ['{rate:1s}-000-{hrit_format:_<6s}-{platform_shortname:4s}_{service:_<7s}-{channel:_<6s}___-{segment:_<6s}___-{start_time:%Y%m%d%H%M}-{c:1s}_']
      kind: IMAGE
      grid:
        origin: "SE"
        first_index_x: 1
        first_index_y: 1

      reader_kwargs:
         fill_hrv: True

Point and Lines Readers Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The visualisation of point and lines can be activated for the according readers
using the kind ``POINTS`` or ``LINES``. These kinds also support the
``style_attributes`` option. It is used to control which product is used for a
certain style attribute. Currently only the attribute ``fill`` is supported,
which influences the colour of ``POINTS``.

Therefore, in the following example from the ``gld360_ualf2`` reader, the colour
of the marker of each point is controlled by the value of the product
``peak_current`` or ``altitude``, if one of them is selected in the file wizard
for loading (the colour map to be used must be configured for the chosen product in the
``default_colormaps`` list)::

    data_reading:
      gld360_ualf2:
        ...
        kind: POINTS
        style_attributes:
          fill: ['peak_current', 'altitude']

For the correct configuration for displaying data of kind ``LINES`` the
following must be done: While the location of the start point of a line is
always given by the fields ``lons`` and ``lats`` guaranteed to be provided by
the reader, the end points of the lines may come in different fields depending
on the loaded file format. Therefore, the coordinates of the end point must be
named in the configuration ``coordinates_end``, which has to be a list of two
field names, as for example for the ``fci_l1_geoobs_lmk_nav_err`` reader::

    data_reading:
      fci_l1_geoobs_lmk_nav_err:
        ...
        kind: LINES
        coordinates_end: ['longitude_reference', 'latitude_reference']

Readers that can return both image and point data, e.g. the LI L2 reader, can
be configured to support both kinds at the same time using the ``DYNAMIC`` kind.
SIFT will then make the assumption that if the loaded dataset is 1-D and has a
pyresample ``SwathDefinition``, it represents point data.

.. note:: Currently, the ``DYNAMIC`` kind supports only the image/point ambiguity.
