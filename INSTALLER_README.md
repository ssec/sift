# SIFT

Satellite Information Familiarization Tool (SIFT) was designed by the SSEC to
support scientists during forecaster training events. It provides a graphical
interface for visualization and basic analysis of geostationary satellite data.

SIFT is built on open source technologies like Python, OpenGL, and PyQt5. It
can be run from Mac, Windows, and Linux. The SIFT application is provided as
a python library called "uwsift". It can also be installed as a standalone
application.

SIFT's main website is http://sift.ssec.wisc.edu/.

The Git repository where you can find SIFT's source code, issue tracker, and
other documentation is on GitHub: https://github.com/ssec/sift

The project wiki with some in-depth usage and installation instructions can
also be found on GitHub: https://github.com/ssec/sift/wiki

Developer documentation can be found on https://sift.readthedocs.io/en/latest/.

## Data Access and Reading

SIFT uses the open source python library Satpy to read input data. By using
Satpy SIFT is able to read many satellite instrument file formats, but may not
be able to display or understand all data formats that Satpy can read. SIFT
defaults to a limited set of readers for loading satellite instrument data.
This set of readers includes but is not limited to:

* GOES-R ABI Level 1b
* Himawari AHI HRIT
* Himawari AHI HSD
* GEO-KOMPSAT-2 AMI Level 1b

Other readers can be accessed from SIFT but this is considered an advanced
usage right now.

## Installation

See the SIFT GitHub page (linked above) for more information on the various
installation methods and download options for SIFT.

## Usage

SIFT can be installed as a python package or from an all-in-one installer
which bundles all dependencies for easy installation. The following
instructions describe how to run SIFT from the all-in-one installer on the
three supported platforms.

### Running on Windows

Once installed through the downloaded installation wizard ``.exe``, SIFT
can be started from the "SIFT" shortcut in the start menu. By default SIFT
caches files in a "workspace" located at the user's
`/Users/<User>/AppData/Local/CIMSS-SSEC/SIFT/Cache/workspace` directory.
Configuration files for the application are stored in the user's
`/Users/<User>/AppData/Roaming/CIMSS-SSEC/SIFT/settings` directory.

### Running on Linux

The downloaded tarball `.tar.gz` can be extracted by running:

    tar -xf SIFT_X.Y.Z.tar.gz

SIFT can then be started by running the `SIFT/SIFT` executable. Run
`SIFT/SIFT -h` for available command line options.

If SIFT will not start please ensure that the `LD_LIBRARY_PATH` environment
variable is not set.

SIFT will cache files in a `~/.cache/SIFT` directory and configuration
files in a `~/.config/SIFT` directory.

### Running on Mac

The downloaded DMG file can be extracted opened by double clicking on it.
The available `.app` should then be moved to the appropriate `Applications`
folder. Due to Apple developer application signing limitations, the `.app`
must first be opened by right clicking and clicking "Open". After SIFT is
opened for the first time double clicking the `.app` icon from `Applications`
will execute SIFT as usual.

SIFT will cache files in a `~/Library/Caches/SIFT` directory and configuration
files in a `~/Library/Application Support/SIFT` directory.
