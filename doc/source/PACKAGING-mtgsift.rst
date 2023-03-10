How to get started using the CMake build system
===============================================

The following documentation gives a step-by-step introduction how to initially
set up a SIFT development environment from scratch using the CMake-based
build system.

The description is platform independent [#f1]_ but based on the usual Linux
workflows. Experienced users may change single steps to better fit their
workflows of course.

It is assumed that the reader knows how to install and initialize a Conda
environment (refer to :ref:`install-conda-packages` otherwise) and how to get
the SIFT sources from the Git repository or from an archive.

Bootstrapping
-------------

This procedure is (only) necessary as long as there is no Conda package
*uwsift-devel-deps* available for the current platform. When the package is
available you may skip this part and jump instead to
:ref:`install-conda-uwsift-devel` and then continue with
:ref:`conda-and-pyinstaller-packaging`.

Prerequisites
+++++++++++++

Only a few preparatory steps need to be done outside the build system.

It is assumed that a `Conda <https://docs.conda.io/projects/conda/en/stable/>`_
system is installed and properly initialized for your shell [#f2]_. Because it
uses the quicker
`Mamba <https://mamba.readthedocs.io/en/latest/index.html>`_
and sets up environments with the `conda-forge <https://conda-forge.org/>`_
repository as default channel the recommended system is
`Mambaforge <https://github.com/conda-forge/miniforge#mambaforge>`_,
but also `Miniforge3 <https://github.com/conda-forge/miniforge#miniforge3>`_,
`Miniconda3 <https://docs.conda.io/en/latest/miniconda.html#latest-miniconda-installer-links>`_
or `Anaconda3 <https://www.anaconda.com/products/distribution>`_ should work.

Install *CMake* and *Make* [#f3]_ into your *base* environment.

On *Linux* this step may be optional - if any of these tools is already
available in your shell in a suitably recent version (CMake: version 3.14 at
least) there is no need to install it into the *base* environment.

On *Windows* you should perform this step in any case, unless you know how to
perform the following steps analogously using a different build system
combination than CMake with "Unix Makefiles".

::

  %> conda install --name base cmake make

CMake-guided System Setup
+++++++++++++++++++++++++

Activate the *base* environment::

  %> conda activate base

In the top-level directory of the SIFT sources run *cmake* with a generator
for the chosen build system::

  (base)%> cmake -G "Unix Makefiles" .

Then follow the advice given by the messages showing up. When doing so, a new
Conda environment *devel-<python version>* (by default *devel-3.10*) [#f4]_ will
be created and populated with all packages necessary for SIFT development
and packaging::

  (base)%> make devel-bootstrap

Please also execute the steps printed out at the end of building this
meta-target *devel-bootstrap* [#f5]_

.. _conda-and-pyinstaller-packaging:

Building Conda and PyInstaller Packages
---------------------------------------

If not already done, execute the steps::

  (base)%> conda activate devel-3.10
  (devel-3.10)%> conda config --env --add channels conda-forge
  (devel-3.10)%> conda config --env --set channel_priority strict
  (devel-3.10)%> cmake -G "Unix Makefiles" .

The system is now ready to create Conda and PyInstaller packages for
SIFT. The according targets can be found in the target lists printed by
running::

  (devel-3.10)%> make help
   The following are some of the valid targets for this Makefile:
   [...]
   ... conda-packages
   ... pyinstaller-package
   [...]

Each of these *package-target*\ s can be build by running::

   (devel-3.10)%> make <package-target>

.. _conda-packaging:

The Target *conda-packages*
+++++++++++++++++++++++++++

The target *conda-packages* creates three Conda packages: *uwsift*,
*uwsift-devel-deps* and *uwsift-deps*.

Only *uwsift* and *uwsift-devel-deps* are meant to be directly installed
from an appropriate Conda channel *sift-channel* [#f6]_. However, the two
packages should not be installed into the same Conda environment together.

The third one is a meta-package which only pulls common dependencies for two
others and is automatically installed when any of them is installed. Please
refer to :ref:`install-conda-packages` regarding how to install and use them.

The three Conda packages are created in a directory, by default
``~/conda-channels/uwsift/`` [#f7]_. This directory is also indexed as a Conda
repository thus it can be used as a local Conda channel in each Conda
environment from which the path is accessible::

  (MY_ENV)%> conda config --add channels ~/conda-channels/uwsift/

.. _pyinstaller-packaging:

The target *pyinstaller-package*
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Using this target you can create an executable which allow to run SIFT
without any additional installation of dependencies. All dependencies are
provided.



.. rubric:: Footnotes

.. [#f1] Tested on Linux and Windows at the time of writing.
.. [#f2] Please refer to the according documentation.
.. [#f3] You may use another build tool supported by CMake as e.g. *Ninja* or
         *MSBuild.exe*.
.. [#f4] The name of the environment can be changed in the CMake configuring
         step.
.. [#f5] Refer to the steps printed by CMake since they may differ from those
	 listed in this document.
.. [#f6] How to provide and populate such a Conda channel is not part of this
         documentation.
.. [#f7] The path of the Conda packages directory of the environment can be
         changed in the CMake configuring step.
