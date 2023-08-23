Configuring External Satpy Components
=====================================

SIFT can use external :ref:`Satpy component configuration <satpy:component_configuration>` folder,
that hosts extra ``readers``, ``composites``, ``enhancements`` and ``areas`` definitions.
To use the external satpy component configuration it is necessary to define either
``satpy_extra_config_path`` in the personal user configs (e.g. inside a file called `external_satpy.yaml`)::

    satpy_extra_config_path: [directory path]

or the environment variable ``SATPY_CONFIG_PATH`` as described `here <https://satpy.readthedocs.io/en/stable/config.html#config-path-setting>`_.

Example of external readers configuration
-----------------------------------------

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
