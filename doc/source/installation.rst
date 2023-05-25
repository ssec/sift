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

Python Package Installation
---------------------------

SIFT can be installed and run in a python environment with the same
functionality as the application and bundle installations. SIFT can be installed
with pip (PyPI) and with conda via the conda-forge channel. The SIFT team
recommends using the conda installation method due to some of the more
complex dependencies that SIFT has.

Installing with Conda
^^^^^^^^^^^^^^^^^^^^^

SIFT is made available as a conda package. This first requires installing a
conda distribution (Anaconda, miniconda, miniforge, or mambaforge). To do this,
first download Miniforge for Python 3 for your platform from the
[Miniforge portion](https://github.com/conda-forge/miniforge#miniforge3) of the
Miniforge download page and then follow the installation instructions on the
download page.

Miniforge is a version of Miniconda that comes
pre-configured with the conda-forge channel (where most SIFT dependencies
come from). Mambaforge is also available which comes with the alternative
``mamba`` command. Mamba should behave similar to ``conda`` but with better
performance. If you're unsure, use the conda-based miniforge installer.
If you do choose ``mamba``, use the ``mamba`` command inplace of any ``conda``
commands below.

You don't need admin privileges to install Miniforge. After installing it,
create a conda environment specifically for SIFT.
Starting with version 1.1, SIFT can be installed directly from the conda-forge
conda channel. It is recommended that a separate conda environment be made
specifically for working with SIFT. SIFT can be installed during environment
creation by doing:

.. code-block:: bash

   conda create -n sift_env -c conda-forge --strict-channel-priority python uwsift

Where ``sift_env`` is whatever you want to name your environment. You can then
activate your environment by running:

.. code-block:: bash

   conda activate sift_env

See the "Running from the python package" section below to learn how to run
SIFT.

Alternatively, if you'd like to install SIFT in an existing conda environment, first activate
your conda environment and then run:

.. code-block:: bash

   conda install -c conda-forge uwsift

Installing with pip
^^^^^^^^^^^^^^^^^^^

Starting with version 1.1, SIFT can be installed with ``pip`` in a normal python
environment. To install it run:

.. code-block:: bash

   pip install uwsift

Running from the python package
-------------------------------

To run the normal SIFT GUI run the following from the command line:

.. code-block:: bash

   python -m uwsift

Note that if running from a conda environment, the environment *must* be
activated before running the above command.

Append the `-h` flag to the above call to see the available command line
options. The python library will cache data and store application settings
in the same locations that the application installations do (see above).

.. _install-conda-uwsift-devel:

For Developers
^^^^^^^^^^^^^^

Check the :ref:`dedicated developer installation documentation <dev_install>`.
