Configuration
-------------

Many aspects of SIFT can or must be configured.

The software comes with a default - *system* - configuration split in several
separate files in a directory structure below the directory
``etc/``::

    etc/
    └── SIFT/
        └── config/
            ├── *.yaml
            └── readers/
                └── *.yaml

Configurations specific to single Satpy readers are in separate files in a
sub-directory 'readers'.

The configuration in this directory tree is not intended to be edited by the
user, as any changes would be lost when the software is updated.
Instead, to configure the software differently from the default, the user can
place additional configuration files below the into an analogous directory
tree::

    <USER_SIFT_CONFIG_DIR>/
    └── config/
        ├── *.yaml
        └── readers/
            └── *.yaml


where USER_SIFT_CONFIG_DIR is the the standard operation system configuration
path for the application SIFT
(Windows: ``C:\Users\<user>\AppData\Roaming\CIMSS-SSEC\SIFT\``,
Linux: ``~/.config/SIFT/``).

The *user* configuration can be empty or only be partial: it is merged with the
*system* configuration, adding or overwriting settings of the latter.
It is not necessary to keep the configuration in individual files, it can be
put into larger files as you wish.

SIFT uses the Donfig config library, for additional details on how to write
configurations have a look here: https://donfig.readthedocs.io/en/latest/.

.. toctree::
   :maxdepth: 2

   Logging <configuration-logging.rst>
   Catalogue and Auto Update Mode <configuration-catalogue-auto_update_mode.rst>
   Watchdog <configuration-watchdog.rst>
   Default Colormaps <configuration-default_colormaps.rst>
   Points Styles <configuration-points_styles.rst>
   Display <configuration-display.rst>
   Layer Manager Display Names <configuration-standard_names.rst>
   External Satpy <configuration-external_satpy.rst>
   Readers <configuration-readers.rst>
   Latlon Grid Resolution <configuration-latlon-grid-resolution.rst>
   Projections / Area Definitions <configuration-area_definitions.rst>
   Units <configuration-units.rst>
   Storage <configuration-storage.rst>
