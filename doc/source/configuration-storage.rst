.. role:: yaml(code)

The following documentation gives an overview how the configuration files for
a certain feature which is configurable have to look like.

Activate File Based Inventory Database and Caching
--------------------------------------------------

MTG-SIFT can either run with a file system based inventory database or without
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

    caching:
        file_caching_active: True

In automated environments the following configuration is recommended (which is
the default)::

    caching:
        file_caching_active: False
        cleanup_file_cache:  True
