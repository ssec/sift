Configuration
-------------

Many aspects of MTG-SIFT can or must be configured.

The software comes with a set of sample configuration in separate files in a
directory structure below the directory ``resources`` This structure mirrors the
structure where the according configurations have to be located for actual use:
The contents of the tree ``/config/SIFT/settings/config/``,
``/config/SIFT/settings/config/readers`` can be copied (and adjusted of course)
to the user's directories ``~/.config/SIFT/settings/config/``, and
``~/.config/SIFT/settings/config/readers`` respectively. Instead of having
several single files the YAML files on the same directory level can be merged
together into one.

MTG-SIFT uses the Donfig config library, for addtional details on how to write
configurations have a look here: https://donfig.readthedocs.io/en/latest/.

.. toctree::
   :maxdepth: 2

   Logging <configuration-logging.rst>
   Catalogue and Auto Update Mode <configuration-catalogue-auto_update_mode.rst>
   Watchdog <configuration-watchdog.rst>
   Default Colormaps <configuration-default_colormaps.rst>
   Points Styles <configuration-points_styles.rst>
   Display <configuration-display.rst>
   External Satpy <configuration-external_satpy.rst>
   Readers <configuration-readers.rst>
   Latlon Grid Resolution <configuration-latlon-grid-resolution.rst>
   Projections / Area Definitions <configuration-area_definitions.rst>
   Units <configuration-units.rst>
   Storage <configuration-storage.rst>
