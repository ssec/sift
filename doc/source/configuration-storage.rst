.. role:: yaml(code)

Storage Configuration
=====================

Activate File Based Inventory Database and Caching
--------------------------------------------------

SIFT can either run with a file system based inventory database or without
it.  The first operation mode is useful if certain data files are loaded
repeatedly while the latter is preferable when the system operates automatically
and usually loads each file only once.

The options to control the behaviour are in the ``storage`` group::

    storage:
        use_inventory_db: [boolean]
        cleanup_file_cache:  [boolean]

The option ``use_inventory_db`` controls whether the inventory database is
used. If so, in the `File` menu two items - `Open from Cache` and `Open Recent`
- are available, which help loading recently loaded data again.

The second option ``cleanup_file_cache`` controls, whether intermediate files
used internally are removed as early as possible to keep the disk space usage
low. This option has only an effect when ``use_inventory_db`` is ``False``,
otherwise they are `not` housekept anyways.

**Examples**

For interactive sessions this configuration is most user-friendly::

    storage:
        use_inventory_db: True

In automated environments the following configuration is recommended (which is
the default)::

    storage:
        use_inventory_db: False
        cleanup_file_cache: True

Observing Directories with the Storage Agent
--------------------------------------------

The settings below ``storage.agent`` are read by the *Storage Agent*::

    storage:
        agent:
            notification_cmd: [path to executable]
            # interval: [number]
            files_lifetime: [number]
            directories:
              - [directory path 1]
              - [directory path 2]
              - ...

All time related settings are in seconds. The ``files_lifetime`` setting defines
the age of files with respect to their last modification in the given
``directories`` after which they are to be deleted. When given, ``interval`` is
the time the storage agent waits, before it does its next check. It defaults to
the ``files_lifetime`` or 60 seconds, whatever is lower.  Finally if the
``notification_cmd`` is configured it will be called additionally to console
logging to inform the GEMS monitoring system about events.

The paths given for ``directories`` may contain a placeholder in the form
``$$CACHE_DIR$$``. When used, this part is expanded to the default cache
directory for the application according to the XDG standard
(``$$CACHE_DIR$$`` expands to ``~/.cache/SIFT`` on Linux systems).


**Example**

.. code-block:: yaml

    storage:
        agent:
            notification_cmd: /opt/eum/bin/raiseEvent.sh
            # interval: 60
            files_lifetime: 1200
            directories:
                - "$$CACHE_DIR$$/workspace/data_cache"
                - "$$CACHE_DIR$$/workspace/temp"
