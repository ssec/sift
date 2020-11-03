#!/usr/bin/env python

import logging
import os
import shlex
import subprocess
from datetime import datetime, timezone
from socket import gethostname
from time import sleep
from typing import List

import appdirs
from donfig import Config

LOG = logging.getLogger(__name__)

APPLICATION_DIR = "SIFT"
APPLICATION_AUTHOR = "CIMSS-SSEC"

WATCHDOG_DATETIME_FORMAT_STORE = "%Y-%m-%d %H:%M:%S %z"


class Watchdog:
    def __init__(self, config_dirs: List[str], cache_dir: str):
        self.hostname = gethostname()
        config = Config('uwsift', paths=config_dirs)

        heartbeat_file = config.get("watchdog.heartbeat_file", None)
        if heartbeat_file is None:
            raise RuntimeError("Can't find `heartbeat_file`"
                               " in the watchdog config")
        self.heartbeat_file = heartbeat_file.replace("$$CACHE_DIR$$", cache_dir)

        self.notification_cmd = None
        notification_cmd = config.get("watchdog.notification_cmd", None)
        if not notification_cmd:
            LOG.warning("Can't send notifications"
                        " because `notification_cmd` isn't configured")
        else:
            self.notification_cmd = shlex.quote(notification_cmd)

        heartbeat_check_interval = config.get(
            "watchdog.heartbeat_check_interval", None)
        if heartbeat_check_interval is None:
            raise RuntimeError("Can't find `heartbeat_check_interval`"
                               " in the watchdog config")
        self.heartbeat_check_interval = float(heartbeat_check_interval)

        max_tolerable_dataset_age = config.get(
            "watchdog.max_tolerable_dataset_age", None)
        if max_tolerable_dataset_age is None:
            raise RuntimeError("Can't find `max_tolerable_dataset_age`"
                               " in the watchdog config")
        self.max_tolerable_dataset_age = float(max_tolerable_dataset_age)

        max_tolerable_idle_time = config.get(
            "watchdog.max_tolerable_idle_time", None)
        if max_tolerable_dataset_age is None:
            raise RuntimeError("Can't find `max_tolerable_idle_time`"
                               " in the  watchdog config")
        self.max_tolerable_idle_time = float(max_tolerable_idle_time)

    def _read_watchdog_file(self) -> datetime:
        with open(self.heartbeat_file) as file:
            content = file.read()

        return datetime.strptime(content.rstrip("\n"),
                                 WATCHDOG_DATETIME_FORMAT_STORE)

    def _notify(self, level: int, text: str):
        if not self.notification_cmd:
            LOG.log(level, text)
        else:
            machine = shlex.quote(self.hostname)
            process_name = shlex.quote(f"{APPLICATION_DIR}-watchdog")
            severity = shlex.quote(logging.getLevelName(level))
            text = shlex.quote(text)
            cmd = (f"{self.notification_cmd}"
                   f" {machine} {process_name} {severity} {text}")

            try:
                subprocess.run(cmd, shell=True, check=True)
            except subprocess.CalledProcessError as err:
                LOG.error(f"Can't run the notification command: {err}")
                LOG.log(level, text)

    def run(self):
        while True:
            sleep(self.heartbeat_check_interval)

            try:
                latest_dataset_time = self._read_watchdog_file()
            except ValueError as err:
                self._notify(logging.ERROR, f"Can't parse the watchdog file:"
                                            f" {err}")
                continue
            except FileNotFoundError:
                # application might not have been started yet or
                # the application is running but no data was loaded yet
                self._notify(logging.INFO, f"Heartbeat file doesn't exist:"
                                           f" {self.heartbeat_file}")
                continue

            # The last update time is implicitly stored as the file modification
            # time. It is possible to approximate this time by checking whether
            # the file content changes, but if the user loads the same dataset
            # again, then this update won't be detected.
            modification_time = datetime.fromtimestamp(
                os.path.getmtime(self.heartbeat_file), tz=timezone.utc)

            now_utc = datetime.now(tz=timezone.utc)
            idle_time = now_utc - modification_time
            if idle_time.total_seconds() > self.max_tolerable_idle_time:
                self._notify(logging.WARNING, f"Dataset was not updated"
                                              f" since {modification_time}")
            else:
                self._notify(logging.INFO,
                             f"Dataset was updated {idle_time.total_seconds()}"
                             f" seconds ago at: {modification_time}")

            dataset_age = now_utc - latest_dataset_time
            if dataset_age.total_seconds() > self.max_tolerable_dataset_age:
                overdue_time = \
                    dataset_age.total_seconds() - self.max_tolerable_dataset_age
                self._notify(logging.WARNING,
                             f"Current dataset scheduled time for observation"
                             f" ('start_time'): {latest_dataset_time} -"
                             f" Next dataset is overdue by "
                             f"{overdue_time:.1f} seconds.")
            else:
                self._notify(logging.INFO,
                             f"Current dataset scheduled time for observation"
                             f" ('start_time'): {latest_dataset_time} - OK")


if __name__ == "__main__":
    user_cache_dir = appdirs.user_cache_dir(APPLICATION_DIR, APPLICATION_AUTHOR)
    user_config_dir = appdirs.user_config_dir(APPLICATION_DIR,
                                              APPLICATION_AUTHOR, roaming=True)
    config_dir = os.path.join(user_config_dir, "settings", "config")

    try:
        Watchdog([config_dir], user_cache_dir).run()
    except KeyboardInterrupt:
        pass
