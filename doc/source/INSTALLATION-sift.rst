Installation options for SIFT
=================================

SIFT is provided either as Conda packages or as executable created with
PyInstaller.

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
+++++++++++++++++++

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

Installation for using SIFT
+++++++++++++++++++++++++++++++

Install the package *uwsift* into an environment called e.g. ``work`` and
prepared as described above::

  (work)%> conda install uwsift

Now you can start SIFT like so::

  (work)%> python -m uwsift

.. _install-conda-uwsift-devel:

Installation for developing SIFT
+++++++++++++++++++++++++++++++++++++

Set up the Conda environment as above - let's call it ``devel`` - and then
install all dependencies for developing SIFT as follows::

  (devel)%> conda install uwsift-devel-deps

PIP-install SIFT in editable mode by run the following in the root
directory of the SIFT sources::

  (devel)%> pip install --editable .

Now you can run SIFT from the current sources with all your changes to the
source code being active immediately just like so::

  (devel)%> python -m uwsift

How to Install SIFT from PyInstaller Packages
-------------------------------------------------

The SIFT packages created with PyInstaller are "portable software", i.e.,
they neither need to be installed nor do they require administration
privileges to be run. Depending on how your SIFT packager provides the
software you may get it either as one single executable file *sift*
(*sift.exe* for Windows) or as a directory *sift* (you may need to
unpack it from an archive), which contains an executable *sift*
(*sift.exe* for Windows) as well as all dependencies (libraries,
configuration, databases).

Note that the single executable file variant has significant slower startup
since each time it is run the contained dependencies are extracted into a
temporary directory.
