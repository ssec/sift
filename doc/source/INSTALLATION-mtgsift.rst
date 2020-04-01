Installation options for MTG-Sift
=================================

MTG-Sift is provided either as Conda packages or as executable created with
PyInstaller.

.. _install-conda-packages:

How to Install MTG-Sift from Conda Packages
-------------------------------------------

To get MTG-Sift in a Conda environment you can choose from two packages.

The fist one - *mtgsift* - installs the software ready to be run. It is
intended for "end" users of the software who are not interested in developing
it.

The other package - *mtgsift-devel-deps* - actually doesn't even provide
MTG-Sift but only makes sure, that the dependencies necessary to develop and
package it are installed. MTG-Sift itself must be provided as source tree
e.g. by cloning from its Git repository or by extracting it from a tarball.

There is a third Conda package - *mtgsift-deps*. It is not meant to be
installed directly but it is pulled automatically when one of the other ones
is installed to provide their common dependencies.

Common Preparations
+++++++++++++++++++

It is best to keep Conda environments intended for just using MTG-Sift
separate from ones for developing it. In detail, you should not install
*mtgsift* but only *mtgsift-devel-deps* into an development environment, since
otherwise the installed MTG-Sift software may interfere with the version from
the sources. And vice versa.

This said, let's assume that ``MY_ENV`` denotes the respective environment
and ``MTGSIFT_CHANNEL`` a Conda channel, where the MTG-Sift packages can be
found [#f1]_, the following common steps should be performed to prepare a clean
environment for the desired task::

  %> conda create --name MY_ENV --channel conda-forge --strict-channel-priority python=3.7
  %> conda activate MY_ENV
  (MY_ENV)%> conda config --env --add channels conda-forge
  (MY_ENV)%> conda config --env --add channels MTGSIFT_CHANNEL
  (MY_ENV)%> conda config --set channel_priority strict
   
.. rubric:: Footnotes
	    
.. [#f1] You need to ask for the URL or name of this ``MTGSIFT_CHANNEL``. If you
	 build packages yourself, the local build directory can be used as
	 this channel, by default it is ``~/conda-channels/mtgvis/`` (see
	 :ref:`conda-packaging`)

Installation for using MTG-Sift
+++++++++++++++++++++++++++++++

Install the package *mtgsift* into an environment called e.g. ``work`` and
prepared as described above::

  (work)%> conda install mtgsift

Now you can start MTG-Sift like so::
 
  (work)%> python -m uwsift

.. _install-conda-mtgsift-devel:
  
Installation for developing MTG-Sift
+++++++++++++++++++++++++++++++++++++

Set up the Conda environment as above - let's call it ``devel`` - and then
install all dependencies for developing MTG-Sift as follows::
  
  (devel)%> conda install mtgsift-devel-deps

PIP-install MTG-Sift in editable mode by run the following in the root
directory of the MTG-Sift sources::
  
  (devel)%> pip install --editable .

Now you can run MTG-Sift from the current sources with all your changes to the
source code being active immediately just like so::
  
  (devel)%> python -m uwsift

How to Install MTG-Sift from PyInstaller Packages
-------------------------------------------------

The MTG-Sift packages created with PyInstaller are "portable software", i.e.,
they neither need to be installed nor do they require administration
privileges to be run. Depending on how your MTG-Sift packager provides the
software you may get it either as one single executable file *mtgsift*
(*mtgsift.exe* for Windows) or as a directory *mtgsift* (you may need to
unpack it from an archive), which contains an executable *mtgsift*
(*mtgsift.exe* for Windows) as well as all dependencies (libraries,
configuration, databases).

Note that the single executable file variant has significant slower startup
since each time it is run the contained dependencies are extracted into a
temporary directory.


