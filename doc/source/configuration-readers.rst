Configuring Readers
-------------------

The readers offered by the Open File Wizard to be used for loading are a subset
of the readers provided by Satpy and - if configured accordingly via
``satpy_extra_readers_import_path`` - in a directory of additional readers.
The list of these readers must be configured, e.g. as follows::

  data_reading:
    readers:
      - fci_l1c_nc
      - seviri_l1b_hrit
      - seviri_l1b_native
      - avhrr_l1b_eps

This is enough to make the readers available, but the Open File Wizard works
better with additional configuration. For each reader it is recommended to
configure:

- ``filter_patterns``, which help to partition the names of files in the chosen
  input directory into their components and
- ``group_keys``, which define, which components of the file names are used to
  group files together into one dataset

These settings are configured in the same way as in the Satpy reader
configuration YAML files.
  
In addition for GEOS files the grid layout can be configured to enable correct
index display according to their PUGs. In the ``grid`` sub-config three
parameters can be set:

- ``origin``, to define, which corner the indexing starts from. Valid settings
  are ``"SE"``, ``"SW"``, ``"NE"`` and ``"NW"``. If not set it defaults to ``"NW"``
- ``first_index_x`` and ``first_index_y`` to define the value indexing starts
  from for columns and rows. Usually this value is 0 or 1 and the same for both
  parameters and defaults to 0 if not given.

For segmented file formats (such as MSG SEVIRI HRIT and MTG FCI NetCDF), it is
usually preferable to merge segments that are loaded in separate load operations
but belong to the same data set into one data set, rather than creating a
separate data set for each load operation. This is especially important when
incrementally loading stripes in auto update mode.

To control this behaviour the ``data_reading``-setting ``merge_with_existing``
can be configured as either ``True`` (the default) or ``False``.

.. note:: Currently, dataset merging does not work together with the caching
          database, so make sure you set ``storage.use_inventory_db:
          False``, otherwise caching will take precedence and disable merging.

.. note:: Currently, dataset merging does not work well together with the
	  adaptively retiling image display. Consider switching it off by
	  setting ``display.use_tiled_geolocated_images: False``, otherwise
	  for some zoom levels the data segments loaded later may not be
	  visible.

As example for a complete configuration for a reader this is one for SEVIRI
Level 1B in HRIT format::

  data_reading:
    merge_with_existing: True
    seviri_l1b_hrit:
      group_keys: ['start_time', 'platform_shortname', 'service']
      filter_patterns: ['{rate:1s}-000-{hrit_format:_<6s}-{platform_shortname:4s}_{service:_<7s}-{channel:_<6s}___-{segment:_<6s}___-{start_time:%Y%m%d%H%M}-{c:1s}_']

      grid:
        origin: "SE"
        first_index_x: 1
        first_index_y: 1

