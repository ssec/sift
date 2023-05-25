Installation
============

SIFT is provided either as Conda packages or as a compressed all-in-one
bundle (tarball/zip) of all dependencies with a single start script.
Unless otherwise documented below, it is recommended that the software
bundle be used for any novice users as it requires the least amount of
prior execution of conda or command line tools. Conda-based installation
is best for testers, developers, and those wishing to customize versions
of dependencies used by SIFT.

System Requirements
-------------------

SIFT works on Windows, Mac, and Linux, but depends on many complex
libraries which may make installation difficult on some platforms or systems.
It also makes heavy use of GPU (video card) processing via the PyOpenGL and VisPy
libraries. In general it is best to have a local physical system to run SIFT as opposed
to a virtual machine or remote server. Virtual machines and remote servers
commonly have limited ways of displaying OpenGL visualizations to another
machine.

While the below system resources should work they should be considered the
lower end of machines for running SIFT. In general the better GPU and faster
memory (RAM) the better experience you'll have with SIFT.

* Windows 10+ / Mac OS X >11.0 / Linux >= Rocky Linux 8
* 8GB RAM
* Disk space (preferrably on a SSD/NVMe drive) with 20GB+ available
* GPU with 2GB VRAM and OpenGL 3+ support
* Data files to be loaded can require several GB of disk space

Ultimately operating system support is limited to those supported by
conda-forge as this is the primary package repository where dependencies
are pulled from.

.. _bundle-install:

Application and Bundle
----------------------

Windows Bundle
^^^^^^^^^^^^^^

SIFT is distributed as a compressed `.zip` archive with all dependencies
provided internally. The bundle (.zip) can be downloaded from our FTP server
`here <https://bin.ssec.wisc.edu/pub/sift/dist/>`_. Once downloaded and the zip
file extracted, the file `SIFT.bat` can be double clicked to start SIFT.

By default SIFT caches files in a "workspace" located
at the user's
``\Users\<User>\AppData\Local\SIFT\Cache\workspace`` directory.
Configuration files for the application are stored in the user's
``\Users\<User>\AppData\Roaming\SIFT\config`` directory.

Linux Bundle
^^^^^^^^^^^^

SIFT is available as an all-in-one tarball (``.tar.gz``) for Rocky Linux 8+
systems. It can be downloaded from our FTP server
`here <https://bin.ssec.wisc.edu/pub/sift/dist/>`_. Once downloaded the files
must be extracted:

.. code-block:: bash

   tar -xzf SIFT_X.Y.Z.tar.gz

This will create a ``SIFT_X.Y.Z`` directory. SIFT can then be run by doing:

.. code-block:: bash

   SIFT_X.Y.Z/SIFT.sh

Adding a `-h` will show the available command line options, but the defaults should work in most cases.

SIFT will cache files in a ``~/.cache/SIFT`` directory and configuration
files in a ``~/.config/SIFT`` directory.

Mac/OSX Bundle
^^^^^^^^^^^^^^

SIFT is available as an all-in-one ``.tar.gz`` bundle. It can be downloaded
from our FTP server
`here <https://bin.ssec.wisc.edu/pub/sift/dist/>`_.
Once downloaded, double clicking on the `.tar.gz` file in Finder should extract
the files and create a new folder named "SIFT_X.Y.Z" where "X.Y.Z" is the
version of SIFT that was downloaded. Double click this new folder to open it
and then double click "SIFT.command" to start SIFT.

SIFT will cache files in a ``~/Library/Caches/SIFT`` directory and configuration
files in a ``~/Library/Application Support/SIFT`` directory.

.. _install-conda-packages:

How to Install SIFT from Conda Packages
-------------------------------------------

To get SIFT in a Conda environment you can choose from two packages.

The first one - *uwsift* - installs the software ready to be run. It is
intended for "end" users of the software who are not interested in developing
it.

The other package - *uwsift-devel-deps* - actually doesn't even provide
SIFT but only makes sure, that the dependencies necessary to develop and
package it are installed. SIFT itself must be provided as source tree
e.g. by cloning from its Git repository or by extracting it from a tarball.

There is a third Conda package - *uwsift-deps*. It is not meant to be
installed directly but it is pulled automatically when one of the other ones
is installed to provide their common dependencies.

Common Preparations
^^^^^^^^^^^^^^^^^^^

It is best to keep Conda environments intended for just using SIFT
separate from ones for developing it. In detail, you should not install
*uwsift* but only *uwsift-devel-deps* into an development environment, since
otherwise the installed SIFT software may interfere with the version from
the sources. And vice versa.

This said, let's assume that ``MY_ENV`` denotes the respective environment
and ``SIFT_CHANNEL`` a Conda channel, where the SIFT packages can be
found [#f1]_, the following common steps should be performed to prepare a clean
environment for the desired task::

  %> conda create --name MY_ENV --channel conda-forge --strict-channel-priority python=3.10
  %> conda activate MY_ENV
  (MY_ENV)%> conda config --env --add channels conda-forge
  (MY_ENV)%> conda config --env --add channels SIFT_CHANNEL
  (MY_ENV)%> conda config --set channel_priority strict

.. rubric:: Footnotes

.. [#f1] You need to ask for the URL or name of this ``SIFT_CHANNEL``. If you
	 build packages yourself, the local build directory can be used as
	 this channel, by default it is ``~/conda-channels/uwsift/`` (see
	 :ref:`conda-packaging`)

For Users
^^^^^^^^^

Install the package *uwsift* into an environment called e.g. ``work`` and
prepared as described above::

  (work)%> conda install uwsift

Now you can start SIFT like so::

  (work)%> python -m uwsift

.. _install-conda-uwsift-devel:

For Developers
^^^^^^^^^^^^^^

Set up the Conda environment as above - let's call it ``devel`` - and then
install all dependencies for developing SIFT as follows::

  (devel)%> conda install uwsift-devel-deps

PIP-install SIFT in editable mode by run the following in the root
directory of the SIFT sources::

  (devel)%> pip install --editable .

Now you can run SIFT from the current sources with all your changes to the
source code being active immediately just like so::

  (devel)%> python -m uwsift
