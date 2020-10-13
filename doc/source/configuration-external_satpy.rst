Configuring External Satpy Components
-------------------------------------

Replacing Satpy by External Installation
========================================

MTG-SIFT can be instructed to import Satpy modules from another location than
from the site packages of the active Python environment when the following
setting points to an appropriate package directory::
   
   satpy_import_path: [directory path]

For example you can use your development version of Satpy cloned directly from
GitHub to ``/home/me/development/satpy`` by configuring::
   
   satpy_import_path: "/home/me/development/satpy/satpy"

or setting the according environment variable before starting MTG-SIFT::
   
   export UWSIFT_SATPY_IMPORT_PATH="/home/me/development/satpy/satpy"

It is your responsibility to make sure the setting points to a suitable Satpy
package: If the given path doesn't point to a Python package directory or not to
one providing Satpy, the application may exit immediately throwing Exceptions.

Using FCI L1 GEOOBS Reader
==========================

To be able to load FCI L1 GEOOBS landmark matching results data a special reader
(developed by Andrea Meraner, EUMETSAT) is necessary and must be configured via
the setting::

    satpy_fci_l1_geoobs_import_path: [directory path]

When set properly, e.g. to the root directory
``/home/me/development/fci_l1_geoobs_satpy_reader`` created as clone from
``gitlab.eumetsat.int:Meraner/fci_l1_geoobs_satpy_reader.git``, the reader
becomes available for selection in MTG-SIFT, thus e.g. with::

  satpy_fci_l1_geoobs_import_path: /home/me/development/fci_l1_geoobs_satpy_reader
