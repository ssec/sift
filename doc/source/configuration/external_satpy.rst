Configuring External Satpy Components
-------------------------------------

Replacing Satpy by External Installation
========================================

SIFT can be instructed to import Satpy modules from another location than
from the site packages of the active Python environment when the following
setting points to an appropriate package directory::

   satpy_import_path: [directory path]

For example you can use your development version of Satpy cloned directly from
GitHub to ``/home/me/development/satpy`` by configuring::

   satpy_import_path: "/home/me/development/satpy/satpy"

or setting the according environment variable before starting SIFT::

   export UWSIFT_SATPY_IMPORT_PATH="/home/me/development/satpy/satpy"

It is your responsibility to make sure the setting points to a suitable Satpy
package: If the given path doesn't point to a Python package directory or not to
one providing Satpy, the application may exit immediately throwing Exceptions.

Using Extra Satpy component configuration
=========================================

SIFT can use external :ref:`Satpy component configuration <satpy:component_configuration>` folder,
that hosts extra ``readers``, ``composites``, ``enhancements`` and ``areas`` definitions.
To use external satpy component configuration it is necessary to define
``satpy_extra_config_path`` in `external_satpy.yaml <https://github.com/ssec/sift/blob/master/uwsift/etc/SIFT/config/external_satpy.yaml>`_::

    satpy_extra_config_path: [directory path]

or it can be defined via environmet variable ``SATPY_CONFIG_PATH`` as described `here <https://satpy.readthedocs.io/en/stable/config.html#config-path-setting>`_.

Example of external readers configuration
`````````````````````````````````````````

Several data formats which are or will be produced by EUMETSAT need special
readers which are not (yet) part of the official Satpy distribution. EUMETSAT
maintains a Git repository ``satpy/local_readers`` on their `GitLab
<https://gitlab.eumetsat.int/satpy/local_readers>`_ providing these special
readers. To use these readers it is neccsary to put them into folder: ``satpy_extra_config_path/readers``.

Furthermore the desired readers need to be added to the configuration
``data_reading.readers`` and their reader specific configuration as well (see
**TODO**).

For example assuming that the repository has been cloned as follows::

    git clone https://gitlab.eumetsat.int/satpy/local_readers.git /path/to/satpy_extra_config_path/readers

the readers for the *FCI L1 Landmark Locations Catalogue*, *FCI L1 GEOOBS
Landmarks* (landmark locations) and *FCI L1 GEOOBS Landmark Matching Results*
(landmark navigation error) can be made available in SIFT with::

    satpy_extra_config_path: /path/to/satpy_extra_config_path

    data_reading:
      readers:
        ...
        - fci_l1_cat_lmk_loc
        - fci_l1_geoobs_lmk_loc
        - fci_l1_geoobs_lmk_nav_err
        ...

and adding according reader detail configuration files
``~/.config/SIFT/settings/config/readers/fci_l1_cat_lmk_loc.yaml``,
``~/.config/SIFT/settings/config/readers/fci_l1_geoobs_lmk_loc.yaml`` and
``~/.config/SIFT/settings/config/readers/fci_l1_geoobs_lmk_nav_err.yaml``.
