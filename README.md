# SIFT

[![Coverage Status](https://coveralls.io/repos/github/ssec/sift/badge.svg)](https://coveralls.io/github/ssec/sift)
[![PyPI version](https://badge.fury.io/py/uwsift.svg)](https://badge.fury.io/py/uwsift)
![CI](https://github.com/ssec/sift/actions/workflows/ci.yaml/badge.svg)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2587907.svg)](https://doi.org/10.5281/zenodo.2587907)
[![Gitter chat](https://badges.gitter.im/gitterHQ/gitter.png)](https://gitter.im/gitterHQ/gitter)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/ssec/sift/master.svg)](https://results.pre-commit.ci/latest/github/ssec/sift/master)


SIFT (Satellite Information Familiarization Tool) is a visualization tool
for satellite data. It provides a graphical interface that can be used for
e.g. fast visualization, scientific data analysis, training, cal/val activities
and operations.

SIFT is built on open source technologies like Python, OpenGL, PyQt5, and
makes use of the [Pytroll framework](https://pytroll.github.io/) for reading
and processing the input data.
It can be run from Mac, Windows, and Linux. The SIFT application is provided as
a python library called "uwsift". It can also be installed as a standalone
application.

SIFT's main website is http://sift.ssec.wisc.edu/.

The Git repository where you can find SIFT's source code, issue tracker, and
other documentation is on GitHub: https://github.com/ssec/sift

The project wiki with some in-depth usage and installation instructions can
also be found on GitHub: https://github.com/ssec/sift/wiki

Developer and configuration documentation can be found on
https://sift.readthedocs.io/en/latest/.

The recording of a SIFT Short Course organised by EUMETSAT can be found [here](https://classroom.eumetsat.int/course/view.php?id=478).

## What's new in SIFT 2.0

Many new features have been added starting from the version 2.0 of SIFT, including:
- reading of data from both geostationary (GEO) as well as low-Earth-orbit (LEO)
  satellite instruments
- visualization of point data (e.g. lightning)
- support for composite (RGB) visualization
- an improved timeline manager
- integration of a statistics module
- full resampling functionalities using Pyresample
- an automatic update/monitoring mode
- partial redesign of the UI/UX
- ... many more small but useful features!

Note that SIFT v2.0 is still in Beta phase (see [Releases](https://github.com/ssec/sift/releases)). Until a full release is reached, the
packaged builds are available in the [experimental ftp folder](https://bin.ssec.wisc.edu/pub/sift/dist/experimental/).
See the Installatio section below for more information.

## History

SIFT was originally created and designed at [SSEC/CIMSS at the University of
Wisconsin - Madison](https://cimss.ssec.wisc.edu/) as a training tool for US
NWS forecasters. Later, [EUMETSAT, European Organization for the Exploitation
of Meteorological Satellites](https://www.eumetsat.int/),
joined the project contributing many new features and refactoring various
portions of the project to support instrument calibration/validation workflows
as well as additional scientific analysis. CIMSS and EUMETSAT now work on the
project together as well as accepting contributions from users outside these
groups.

EUMETSAT contributions, leading up to SIFT 2.0, were carried out by
[ask – Innovative Visualisierungslösungen GmbH](https://askvisual.de/).

## Data Access and Reading

SIFT uses the open source python library Satpy to read input data. By using
Satpy, SIFT is able to read many satellite instrument file formats,
especially in the meteorology domain. The full list of available Satpy readers
can be found in
[Satpy's documentation](https://satpy.readthedocs.io/en/stable/index.html#id1).
Note however that SIFT may not be able to display or understand all data formats
that Satpy can read.
SIFT defaults to a limited set of readers; head to the
[configuration documentation](https://sift.readthedocs.io/en/latest/configuration/index.html)
for customizing your SIFT.

## Installation

SIFT can be installed as an all-in-one bundled application or the python
library "uwsift" can be installed in a traditional python environment.

Detailed installation instructions can be found on the
[installation documentation](https://sift.readthedocs.io/en/latest/installation.html).

## Contributors

SIFT is an open source project welcoming all contributions. See the
[Contributing Guide](https://sift.readthedocs.io/en/latest/dev_guide/contributing.html)
for more information on how you can help.

### Building and releasing

For instructions on how SIFT is built and packaged see the
[releasing instructions](RELEASING.md). Note that these instructions
are mainly for SIFT developers and may require technical understanding of
SIFT and the libraries it depends on.
