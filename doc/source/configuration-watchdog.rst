Configuring Watchdog Functionality
----------------------------------

The *Watchdog* is a script (``uwsift/util/watchdog.py``) running separately from
MTG-SIFT and has the responsibility to assess, whether MTG-SIFT running as
monitoring tool (with ``auto_update.active: True``) is working correctly and to
"bark" otherwise by calling an adaptor script *raiseEvent.sh* to notify
GEMS. The location of this script has to be configured as ``notification_cmd``.

The Watchdog does not directly interact with the running MTG-SIFT instance but
monitors a file to be configured as ``heartbeat_file``, which MTG-SIFT updates
with the data timestamp (i.e. the ``start_time`` is written into the file) every
time it loads new data. From this information and the filesystem change time of
the heartbeat file the Watchdog can determine, when the monitoring system is not
alive anymore and/or it does not succeed to ingest up to date satellite
data. With the frequency configured by ``heartbeat_check_interval`` the Watchdog
reads the file and compares the time information against the current time and
gives alarm, when the data timestamp stored is older than
``max_tolerable_dataset_age`` and/or the last time the heartbeat file was
updated is longer ago than the ``max_tolerable_idle_time``. These three time
span related configurations are in seconds.

A complete watchdog configuration looks as follows::

  watchdog:
    heartbeat_file: "$$CACHE_DIR$$/heartbeat.txt"
    notification_cmd: /path/to/raiseEvent.sh
    heartbeat_check_interval: 30
    max_tolerable_dataset_age: 120
    max_tolerable_idle_time: 60

Note the part ``$$CACHE_DIR$$`` of the path for the heartbeat file. When used,
this part is expanded to the default cache directory for the application
according to the XDG standard (MTG-SIFT calls itself still ``SIFT``, thus for
now ``$$CACHE_DIR$$`` expands to ``~/.cache/SIFT`` on Linux systems). A normal
absolute file path works too.




