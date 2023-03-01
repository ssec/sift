Auto Update Mode
================

SIFT can be operated in *Auto Update Mode* for operational monitoring of
incoming data.

In that mode the graphical user interfaces shows three time displays instead of
the controls for animations:

- *Date Time*: displays the (start) time of the currently displayed data
- *Import Time*: displays, when the currently displayed data was imported
- *Current Time*: displays the current time like a wall clock

Both *Date Time* and *Import Time* are coloured green as long as the according
times are sufficiently recent but change the colour to red when there is
significant delay.

To run SIFT in *Auto Update Mode* according settings must be added to the
application configuration, see :ref:`auto_update_mode_activation`.

Since in auto update mode no files are load interactively a *Catalogue* must
be configured which configures files according to which file name patterns
should be loaded and where to find them. For details please see
:ref:`auto_update_catalogue_config`.

Disk Space Management Tools
---------------------------

SIFT creates temporary files during ingestion of new data. Some of these
files are written by libraries SIFT depends on, e.g. intermediate files are
created by Satpy when it decompresses data stored in compressed file
formats. But also SIFT itself creates intermediate files which are only used
once and are not of permanent value.

Especially when SIFT runs in *Auto Update Mode* and repeatedly loads new
data these files could fill up the file system. Two tools are provided to manage
this:

* The *Disk Management* tool to identify which files are written by SIFT

* The *Storage Agent* which actually cleans up configured directories which have
  been identified as containing intermediate files from SIFT

The recommended approach is to use the *Disk Management* tool to analyse into
which directories SIFT writes temporary files and use this information to
configure the *Storage Agent* to watch these directories and purge obsolete
temporary files from them. The analysis step has to be done only during
development of the software or initially after installing or updating the
software to figure out, whether additional directories are affected.

Disk Management
~~~~~~~~~~~~~~~

``disk_management.py`` allows to collect a list of files accessed by a
program. To analyse the files opened by the SIFT process started via
commandline

.. code-block:: bash

    $ python -m uwsift

the tool has to be called as follows:

.. code-block:: bash

    $ python uwsift/util/disk_management.py --cmdline "python -m uwsift"

Alternatively the argument ``--pid`` can be used to trace a specific PID:

.. code-block:: bash

    $ python uwsift/util/disk_management.py --pid <pid>

Example Output
..............


.. code-block::

    Observing the following processes:
        65762 -> python -m uwsift

    Searching for open files...............................................

    READ:
        /home/user/Dokumente/MSG3_RSS_-201910211200/MSG3_RSS_1/H-000-MSG3__-MSG3_RSS____-HRV______-000016___-201910211200-__
        /home/user/Dokumente/eumSIFT/uwsift/data/ne_50m_admin_0_countries/ne_50m_admin_0_countries.dbf
        /home/user/Dokumente/eumSIFT/uwsift/data/ne_50m_admin_0_countries/ne_50m_admin_0_countries.shp
        /home/user/Dokumente/eumSIFT/uwsift/data/ne_50m_admin_0_countries/ne_50m_admin_0_countries.shx
        /home/user/Dokumente/eumSIFT/uwsift/data/ne_50m_admin_1_states_provinces_lakes/ne_50m_admin_1_states_provinces_lakes.dbf
        /home/user/Dokumente/eumSIFT/uwsift/data/ne_50m_admin_1_states_provinces_lakes/ne_50m_admin_1_states_provinces_lakes.shp
        /home/user/Dokumente/eumSIFT/uwsift/data/ne_50m_admin_1_states_provinces_lakes/ne_50m_admin_1_states_provinces_lakes.shx
        /usr/share/icons/Adwaita/icon-theme.cache
        /usr/share/icons/hicolor/icon-theme.cache
        /usr/share/mime/mime.cache
        /var/local/miniconda3/envs/devel-default/lib/libopenblasp-r0.3.9.so
        /var/local/miniconda3/envs/devel-default/lib/python3.7/site-packages/PyQt5/QtGui.so
        /var/local/miniconda3/envs/devel-default/lib/python3.7/site-packages/pandas/_libs/tslibs/timestamps.cpython-37m-x86_64-linux-gnu.so
        /var/local/miniconda3/envs/devel-default/lib/python3.7/site-packages/satpy/etc/composites/seviri.yaml
        /var/local/miniconda3/envs/devel-default/lib/python3.7/site-packages/satpy/etc/composites/visir.yaml
        /var/local/miniconda3/envs/devel-default/lib/python3.7/site-packages/satpy/etc/readers/abi_l1b_scmi.yaml
        /var/local/miniconda3/envs/devel-default/lib/python3.7/site-packages/satpy/etc/readers/abi_l2_nc.yaml
        /var/local/miniconda3/envs/devel-default/lib/python3.7/site-packages/satpy/etc/readers/avhrr_l1b_gaclac.yaml
        /var/local/miniconda3/envs/devel-default/lib/python3.7/site-packages/satpy/etc/readers/hsaf_grib.yaml
        /var/local/miniconda3/envs/devel-default/lib/python3.7/site-packages/satpy/etc/readers/modis_l1b.yaml
        /var/local/miniconda3/envs/devel-default/lib/python3.7/site-packages/satpy/etc/readers/modis_l2.yaml
        /var/local/miniconda3/envs/devel-default/lib/python3.7/site-packages/satpy/etc/readers/nwcsaf-geo.yaml
        /var/local/miniconda3/envs/devel-default/lib/python3.7/site-packages/satpy/etc/readers/seviri_l1b_hrit.yaml
        /var/local/miniconda3/envs/devel-default/lib/python3.7/site-packages/satpy/etc/readers/seviri_l1b_native.yaml
        /var/local/miniconda3/envs/devel-default/lib/python3.7/site-packages/satpy/etc/readers/seviri_l2_bufr.yaml
        /var/local/miniconda3/envs/devel-default/lib/python3.7/site-packages/satpy/etc/readers/slstr_l1b.yaml
        /var/local/miniconda3/envs/devel-default/lib/python3.7/site-packages/scipy/ndimage/__pycache__/interpolation.cpython-37.pyc
        /var/local/miniconda3/envs/devel-default/lib/python3.7/site-packages/scipy/optimize/_bglu_dense.cpython-37m-x86_64-linux-gnu.so
        /var/local/miniconda3/envs/devel-default/plugins/platforms/libqoffscreen.so
        /var/local/miniconda3/envs/devel-default/resources/icudtl.dat
        /var/local/miniconda3/envs/devel-default/resources/qtwebengine_devtools_resources.pak
        /var/local/miniconda3/envs/devel-default/resources/qtwebengine_resources.pak
        /var/local/miniconda3/envs/devel-default/resources/qtwebengine_resources_100p.pak
        /var/local/miniconda3/envs/devel-default/resources/qtwebengine_resources_200p.pak
        /var/local/miniconda3/envs/devel-default/share/proj/proj.db
        /var/local/miniconda3/envs/devel-default/translations/qtwebengine_locales/de.pak

    READ + WRITE:
        /home/user/.cache/SIFT/workspace/_inventory.db
        /home/user/.cache/SIFT/workspace/_inventory.db-journal
        /home/user/.cache/SIFT/workspace/data_cache/62b993fc-dcb2-11ea-8b91-eca86b8d05fb.image
        /home/user/.local/share/QtWebEngine/Default/GPUCache/data_0
        /home/user/.local/share/QtWebEngine/Default/GPUCache/data_1
        /home/user/.local/share/QtWebEngine/Default/GPUCache/data_2
        /home/user/.local/share/QtWebEngine/Default/GPUCache/data_3
        /home/user/.local/share/QtWebEngine/Default/GPUCache/index
        /home/user/.local/share/QtWebEngine/Default/Visited Links
        /home/user/.nv/GLCache/e213ecd26c5b62b33e76a1434cd31a0e/fdca7a61d748231c/42fc55430588c083.bin
        /home/user/.nv/GLCache/e213ecd26c5b62b33e76a1434cd31a0e/fdca7a61d748231c/42fc55430588c083.toc

Storage Agent
~~~~~~~~~~~~~

The Storage Agent can be used to cleanup directories e.g. from files generated
for caching.  The agent is started without command line options, since it reads
all its settings from the configuration. The configuration for the storage agent
is part of the ``storage`` configuration (see Storage
<configuration-storage.rst>).

.. code-block:: bash

    ./storage_agent.py

After reading in its configuration the agent observes and purges the configured
directories in given intervals until terminated: It attempts to delete each file
in the observed directories whose age is larger than the configured lifetime.
The file age is counted from the last time it was modified.

If a file can't be removed, the Storage Agent will notify about this and ignore
the file and therefore won't try to delete it again. The notification may by a
simple log message to the console or additionally an event raised to the
EUMETSAT's GEMS monitoring system when a ``notification_cmd`` is properly
configured.
