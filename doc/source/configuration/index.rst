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
(Windows: ``C:\Users\<user>\AppData\Roaming\SIFT\``,
Linux: ``~/.config/SIFT/``).

The *user* configuration can be empty or only be partial: it is merged with the
*system* configuration, adding or overwriting settings of the latter.
It is not necessary to keep the configuration in individual files, it can be
put into larger files as you wish.

SIFT uses the Donfig config library, for additional details on how to write
configurations have a look here: https://donfig.readthedocs.io/en/latest/.

.. toctree::
   :maxdepth: 2

   Logging <logging>
   Catalogue and Auto Update Mode <catalogue-auto_update_mode>
   Watchdog <watchdog>
   Default Colormaps <default_colormaps>
   Points Styles <points_styles>
   Display <display>
   Layer Manager Display Names <standard_names>
   External Satpy <external_satpy>
   Readers <readers>
   Latlon Grid Resolution <latlon-grid-resolution>
   Projections / Area Definitions <area_definitions>
   Units <units>
   Storage <storage>
