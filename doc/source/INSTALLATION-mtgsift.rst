Installation options for MTG-SIFT
=================================

MTG-SIFT is provided either as Conda packages or as executable created with
PyInstaller.

.. _install-conda-packages:

How to Install MTG-SIFT from Conda Packages
-------------------------------------------

To get MTG-SIFT in a Conda environment you can choose from two packages.

The first one - *mtgsift* - installs the software ready to be run. It is
intended for "end" users of the software who are not interested in developing
it.

The other package - *mtgsift-devel-deps* - actually doesn't even provide
MTG-SIFT but only makes sure, that the dependencies necessary to develop and
package it are installed. MTG-SIFT itself must be provided as source tree
e.g. by cloning from its Git repository or by extracting it from a tarball.

There is a third Conda package - *mtgsift-deps*. It is not meant to be
installed directly but it is pulled automatically when one of the other ones
is installed to provide their common dependencies.

Common Preparations
+++++++++++++++++++

It is best to keep Conda environments intended for just using MTG-SIFT
separate from ones for developing it. In detail, you should not install
*mtgsift* but only *mtgsift-devel-deps* into an development environment, since
otherwise the installed MTG-SIFT software may interfere with the version from
the sources. And vice versa.

This said, let's assume that ``MY_ENV`` denotes the respective environment
and ``MTGSIFT_CHANNEL`` a Conda channel, where the MTG-SIFT packages can be
found [#f1]_, the following common steps should be performed to prepare a clean
environment for the desired task::

  %> conda create --name MY_ENV --channel conda-forge --strict-channel-priority python=3.10
  %> conda activate MY_ENV
  (MY_ENV)%> conda config --env --add channels conda-forge
  (MY_ENV)%> conda config --env --add channels MTGSIFT_CHANNEL
  (MY_ENV)%> conda config --set channel_priority strict

.. rubric:: Footnotes

.. [#f1] You need to ask for the URL or name of this ``MTGSIFT_CHANNEL``. If you
	 build packages yourself, the local build directory can be used as
	 this channel, by default it is ``~/conda-channels/mtgvis/`` (see
	 :ref:`conda-packaging`)

Installation for using MTG-SIFT
+++++++++++++++++++++++++++++++

Install the package *mtgsift* into an environment called e.g. ``work`` and
prepared as described above::

  (work)%> conda install mtgsift

Now you can start MTG-SIFT like so::

  (work)%> python -m uwsift

.. _install-conda-mtgsift-devel:

Installation for developing MTG-SIFT
+++++++++++++++++++++++++++++++++++++

Set up the Conda environment as above - let's call it ``devel`` - and then
install all dependencies for developing MTG-SIFT as follows::

  (devel)%> conda install mtgsift-devel-deps

PIP-install MTG-SIFT in editable mode by run the following in the root
directory of the MTG-SIFT sources::

  (devel)%> pip install --editable .

Now you can run MTG-SIFT from the current sources with all your changes to the
source code being active immediately just like so::

  (devel)%> python -m uwsift

How to Install MTG-SIFT from PyInstaller Packages
-------------------------------------------------

The MTG-SIFT packages created with PyInstaller are "portable software", i.e.,
they neither need to be installed nor do they require administration
privileges to be run. Depending on how your MTG-SIFT packager provides the
software you may get it either as one single executable file *mtgsift*
(*mtgsift.exe* for Windows) or as a directory *mtgsift* (you may need to
unpack it from an archive), which contains an executable *mtgsift*
(*mtgsift.exe* for Windows) as well as all dependencies (libraries,
configuration, databases).

Note that the single executable file variant has significant slower startup
since each time it is run the contained dependencies are extracted into a
temporary directory.
