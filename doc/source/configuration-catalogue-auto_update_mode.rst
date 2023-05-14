Configuring Catalogue and Auto Update Mode
==========================================

.. _auto_update_catalogue_config:

Catalogue Query Configuration
-----------------------------

The Catalogue configuration defines a list of "queries".

An example configuration looks a follows::

  catalogue:
    - reader: <string>
      search_path: <string>
      constraints:
        <replacement_field_1>: <replacement_field_1_setting>
        <replacement_field_2>:
          - <replacement_field_2_setting_1>
          - <replacement_field_2_setting_2>
      products:
        <source>:
          - <product_1>
          - <product_2>


Each query must define the ``reader`` to use, the ``search_path`` where to look
for files, optional ``constraints``, and a mandatory definition of the desired
``products``.

The configration for the given ``reader`` (usually in ``readers/<reader>.yaml``)
is evaluated to get *filter_patterns*, which are processed and then used to
match files in the given ``search_path``.

The ``constraints`` are applied to reduce the result list (for
example to see only files where ``platform_shortname`` is ``MSG4``). The given
constraint items correspond to the replacement fields of the *filter_patterns*
(different *filter_patterns* may have arbitrarily different constraints).


Very important constraint options are those for defining restrictions for the
data time(s). For at most one datetime replacement field from the
*filter_patterns* a constraint can be given (only the first is evaluated, the
others are ignored).  This type of constraint is detected, when an according
explicit ``type`` is defined for them; two of these explicit constraint types
are available (for now):

* ``type: datetime``

  a fixed filter based on the different parts of the data time can be defined,
  e.g. data from all 1st days of each month in 2019 at 12:00

* ``type: recent_datetime``

  a range of time steps relative to the current time ("now") can be defined,
  e.g. all data for the current hour and the two before with the value
  ``[0, -1, -2]``


Finally with filename based filtering defined it must be configured, which
actual ``products`` should be loaded/generated from the actual selection.  Each
*source* (a channel or dataset name as defined for the file type) must be given
with a (possibly empty) list of derived product names - if the list is empty,
the original dataset name is taken as product name.

Note, that the order of items in a query is free, but the order of the top
level items is recommended as shown here.

**Example:**

The following defines a catalogue query suitable for loading the MTG FCI FDHSI
product ``brightness temperature`` from the source channel ``ir_105`` for the
current and previous hour from the configured search path
``/path/to/fci/data``. This configuration is suitable for the *Auto Update
Mode*::

    catalogue:

      - reader: 'fci_l1c_fdhsi'
        search_path: '/path/to/fci/data/'

        constraints:
          spacecraft_id: 1
          data_source: FCI
          processing_level: 1C
          start_time:
            type: recent_datetime
            H: [0, -1]

        products:
          ir_105: [brightness_temperature]

Another example shows a catalogue query suitable for loading SEVIRI channel
``IR_108`` products ``brightness temperature`` and ``radiance`` for data times
2019-10-21T12:00 UTC until 2019-10-21T13:00 UTC (exclusive) from the configured
search path ``/path/to/seviri/data/``. This query is not suitable for the Auto
Update Mode since it defines a fixed time span for the data::

    catalogue:

      - reader: 'seviri_l1b_hrit'
        search_path: '/path/to/seviri/data/'

        constraints:
          platform_shortname:
            - MSG4
          channel:
            - ______
            - IR_108
          start_time:
            type: datetime
            Y: 2019
            m: 10
            d: 21
            H: 12

        products:
          IR_108: [brightness_temperature, radiance]

Note, that to catch the EPI and PRO files of the SEVIRI HRIT format the item
``______`` must be given for the replacement field option ``channel``: EPI and
PRO files have this at the *channel* part of their filenames.

.. _auto_update_mode_activation:

Activation of Auto Update Mode
------------------------------

To activate the auto update mode the following entry must be available in the
configuration settings::

   auto_update:
     active: [boolean]
     interval: [float]

The option ``interval`` defines the time span between consecutive update cycles
in seconds. It sets the duration to wait after the loading of a dataset has been
finished before the next check for updates is performed. As long as no new data
is found, this check is repeated every ``interval`` seconds.

For this to work a suitable Catalogue query configuration is required as
described in the next section.
