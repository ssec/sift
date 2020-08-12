
Replacing Satpy by External Installation
----------------------------------------

MTG-SIFT can be instructed to import Satpy modules from another location than
from the site packages of the active Python environment when the following
setting points to an appropriate package directory::
   
   satpy_import_path: "[directory path]"

For example you can use your development version of Satpy cloned directly
from GitHub to ``/home/me/development/satpy`` by configuring::
   
   satpy_import_path: "/home/me/development/satpy/satpy"

or setting the according environment variable before starting MTG-SIFT::
   
   export UWSIFT_SATPY_IMPORT_PATH="/home/me/development/satpy/satpy"

It is your responsibility to make sure the setting points to a suitable Satpy
package: If the given path doesn't point to a Python package directory or not to
one providing Satpy, the application may exit immediately throwing Exceptions.

