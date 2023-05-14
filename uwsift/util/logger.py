import logging
import sys
from pathlib import PurePath
from typing import List, Optional

from PyQt5.QtWidgets import QMessageBox

from uwsift import config

LOG = logging.getLogger(__name__)


def configure_loggers() -> None:
    """Configure all loggers with a certain log level, format and handler.

    While all loggers are configured to have the same format and handler,
    the log level can be configured explicitly for "all" loggers and/or for
    individual ones. If not configured, the log level is left as is, i.e.,
    loggers of third party libraries keep their default log level.
    If a `logging.filename` is given, logging goes to that file (if it is
    writable), to console otherwise.
    """

    formatter: logging.Formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(module)s:" "%(funcName)s:L%(lineno)d %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    config_key: str = "logging.filename"
    file_path: Optional[str] = config.get(config_key, None)
    handler = __configure_handler(file_path)

    handler.setFormatter(formatter)

    loggers_all_level: Optional[str] = __configure_root_logger(handler)
    __configure_available_loggers(loggers_all_level)


def __configure_handler(file_path: Optional[str]):
    """
    If an absolute `file_path` to a writable file is given, return a FileHandler
    writing to that file, otherwise a StreamHandler to log to the console.
    Warn by popping up a message box if a given `file_path` is not absolute or
    doesn't point to a not writable file. `file_path` can be `None`.
    """
    handler: logging.Handler = logging.StreamHandler(sys.stderr)

    if file_path is None:
        return handler

    if not PurePath(file_path).is_absolute():
        err_msg = (
            f"Logging to file '{file_path}' is not possible."
            f"\nThe given path is not absolute."
            f"\nLogging to console instead."
        )
        QMessageBox.warning(None, "Error in Logging Configuration", err_msg)
        return handler

    try:
        handler = logging.FileHandler(file_path)
    except OSError as e:
        err_msg = f"Logging to file '{file_path}' is not possible." f"\n{e}" f"\nLogging to console instead."
        QMessageBox.warning(None, "Error in Logging Configuration", err_msg)

    return handler


def __configure_available_loggers(logger_all_level: Optional[str]) -> None:
    """
    Remove handlers from all loggers to ensure that all write to either the
    console or a log file by propagating to the root logger.
    Furthermore, if found in the configuration set an individual log level or the
    one configured for "all" loggers.
    """
    # The following will collect all loggers except for the root logger. This is
    # exactly what's intended.
    loggers: List[logging.Logger] = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger in loggers:
        # No individual handlers, only the root logger shall have handlers.
        # This avoids any duplicated log messages.
        logger.handlers.clear()
        logger.propagate = True

        config_key: str = "logging.loggers." + logger.name + ".level"
        individual_level = config.get(config_key, logger_all_level)

        if individual_level is None:
            # Do nothing since no level was configured for the current logger
            # nor given for "all" loggers
            continue

        try:
            logger.setLevel(individual_level)
        except ValueError:
            LOG.warning(f"Logger '{logger.name}' configured with invalid" f" log level '{individual_level}'. Ignoring.")


def __configure_root_logger(handler: logging.Handler) -> Optional[str]:
    """
    Set the given handler as the only one of the root logger.
    If a log level is configured by 'logger.all.level' set the level of the root
    logger to it and return that log level, `None` otherwise.
    """

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)

    config_key: str = "logging.loggers.all.level"
    loggers_all_level: Optional[str] = config.get(config_key, None)
    if loggers_all_level is None:
        # Do nothing since no level was configured for "all" loggers
        return None

    try:
        root_logger.setLevel(loggers_all_level)
    except ValueError:
        LOG.warning(f"Configured '{config_key}: {loggers_all_level}'" f" isn't a valid log level. Ignoring.")
        return None

    return loggers_all_level
