.. role:: yaml(code)

Configuring Logging
-------------------

Logging can be configured to write either to the console or to a file.
Furthermore the global log level can be set as well as individual levels for
each logger.

All logging configuration is done below the item ``logging``.

Logging to file is activated, when an *absolute* [#abspath]_ file path is given
for the keyword ``filename``::

    logging:
      filename: [absolute file path]

If the configured file is not writable, logging falls back to the console.

A global log level can be set as follows::

    logging:
      loggers:
        all:
	  level: [log level]

where ``[log level]`` must be one of ``CRITICAL``, ``ERROR``, ``WARNING``,
``INFO``, ``DEBUG`` or ``NOTSET``. The default log level is ``WARNING``.

Additionally the log level of specifig loggers can be overwritten by listing
their names below the keyword ``loggers`` and adding a ``level`` setting
analogously.

**Example** ::

    logging:
      filename: /tmp/sift.log
      loggers:
        all:
          level: DEBUG
	vispy:
	  level: INFO

With this configuration all log messages go to the file ``/tmp/sift.log`` and
all modules output messages of all levels except for the module ``vispy`` which
only outputs messages of level ``INFO`` and higher.


.. rubric:: Footnotes

.. [#abspath] Accepting a relative path could lead to stray log files in every
	      directory which is current when SIFT is started.
