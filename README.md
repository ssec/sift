# SIFT

[![Coverage Status](https://coveralls.io/repos/github/ssec/sift/badge.svg)](https://coveralls.io/github/ssec/sift)
[![PyPI version](https://badge.fury.io/py/uwsift.svg)](https://badge.fury.io/py/uwsift)
![CI](https://github.com/ssec/sift/actions/workflows/ci.yaml/badge.svg)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2587907.svg)](https://doi.org/10.5281/zenodo.2587907)
[![Gitter chat](https://badges.gitter.im/gitterHQ/gitter.png)](https://gitter.im/gitterHQ/gitter)


Satellite Information Familiarization Tool (SIFT) was designed by the Space
Science and Engineering Center (SSEC) at the University of Wisconsin - Madison
to support scientists during forecaster training events. It provides a
graphical interface for visualization and basic analysis of geostationary
satellite data.

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

SIFT can be installed as an all-in-one bundled application or the python
library "uwsift" can be installed in a traditional python environment.

Detailed installation instructions can be found on the
[GitHub Wiki](https://github.com/ssec/sift/wiki/Installation-Guide).

## Contributors

SIFT is an open source project welcoming all contributions. See the
[Contributing Guide](https://github.com/ssec/sift/wiki/Contributing)
for more information on how you can help.

### Building and releasing

For instructions on how SIFT is built and packaged see the
[releasing instructions](RELEASING.md). Note that these instructions
are mainly for SIFT developers and may require technical understanding of
SIFT and the libraries it depends on.
