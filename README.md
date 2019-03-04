SIFT
====

Satellite Information Familiarization Tool (SIFT) was designed by the SSEC to
support scientists during forecaster training events. It provides a graphical
interface for visualization and basic analysis of geostationary satellite data.

SIFT is built on open source technologies like Python, OpenGL, and PyQt4. It
can be run from Mac, Windows, and Linux.

The main website is http://sift.ssec.wisc.edu/.

The Git repository is available on GitHub: https://github.com/ssec/sift

The project wiki with some in-depth usage and installation instructions can
also be found on GitHub: https://github.com/ssec/sift/wiki

Developer documentation can be found on https://sift.readthedocs.io/en/latest/.


Data Access
-----------

SIFT currently accepts a limited number of input formats. It is able to load
NetCDF4 L1B files for the GOES-16 ABI instrument. It will accept more input
files in the future. Please contact Jordan Gerth, Ray Garcia, or David Hoese
to get access to this early release data set.

Installation
------------

SIFT installers and bundles are available on the SIFT FTP location:

ftp://ftp.ssec.wisc.edu/pub/sift/dist
    
The Windows installers end in `.exe`, Linux with `.tar.gz`, and Mac OSX with
`.dmg`. See the sections below for details on installing SIFT for each
operating system.

### Run on Windows

After executing the downloaded `.exe` installer follow the installation
wizard to install SIFT. SIFT can then be run from the "SIFT" shortcut
in the start menu. By default SIFT caches files in a "Workspace" located
at the user's `Documents/sift_workspace`. The installation wizard allows
you to customize this location.

### Run on Linux

The downloaded tarball `.tar.gz` can be extracted by running:

    tar -xf SIFT_X.Y.Z.tar.gz
    
SIFT can then be run by executing the `SIFT/SIFT`. Run `SIFT/SIFT -h`
for available command line options.

If SIFT will not start please ensure that the `LD_LIBRARY_PATH` environment
variable is not set.

### Run on Mac

The downloaded DMG file can be extracted opened by double clicking on it.
The available `.app` should then be moved to the appropriate `Applications`
folder. Due to Apple developer application signing limitations, the `.app`
must first be opened by right clicking and clicking "Open". After SIFT is
opened for the first time double clicking the `.app` icon from `Applications`
will execute SIFT as usual.

### Installing with Conda

SIFT can also be installed with the Anaconda/Conda package manager. Python
3.6 is currently the only supported python environment. It can be installed by
running:

    conda install -c http://larch.ssec.wisc.edu/channels/sift sift
    
And then run with:

    python -m sift
    
The `-h` flag can be added for documentation on additional command line
options.

### Building and releasing

For instructions on how SIFT is built and packaged see the
[releasing instructions](RELEASING.md). Note that these instructions
are mainly for SIFT developers and may require technical understanding of
SIFT and the libraries it depends on.