#!/usr/bin/env python

import logging
import os
import shlex
import signal
import subprocess
from datetime import datetime, timezone, timedelta
from socket import gethostname
from time import sleep
from typing import List, Tuple

import appdirs
from donfig import Config
from psutil import Process, NoSuchProcess

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

        restart_interval = int(config.get("watchdog.auto_restart_interval", 0))
        if restart_interval == 0:
            LOG.warning("Auto Restart is disabled")
            self.restart_interval = None
        else:
            self.restart_interval = timedelta(seconds=restart_interval)

        ask_again_interval = int(config.get(
            "watchdog.auto_restart_ask_again_interval", 0))
        if ask_again_interval == 0:
            LOG.warning("Auto Restart will ask the user only once")
            self.ask_again_interval = None
        else:
            self.ask_again_interval = timedelta(seconds=ask_again_interval)

    def _read_watchdog_file(self) -> Tuple[int, datetime]:
        with open(self.heartbeat_file) as file:
            content = file.read()

        pid, timestamp = content.splitlines()
        return int(pid), datetime.strptime(timestamp,
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
        old_pid = None
        application_start_time = None
        sent_restart_request = False

        process_cwd = None
        process_cmdline = None

        while True:
            if self.restart_interval is None:
                sleep(self.heartbeat_check_interval)
            else:
                sleep(min(self.heartbeat_check_interval,
                          self.restart_interval.seconds))

            try:
                pid, latest_dataset_time = self._read_watchdog_file()
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

            if application_start_time is None:
                application_start_time = now_utc

            if old_pid is None:
                old_pid = pid
            elif old_pid != pid:
                self._notify(logging.INFO, f"Application was restarted: {pid}")
                application_start_time = now_utc
                old_pid = pid

            try:
                process = Process(pid)
                process_cwd = process.cwd()
                process_cmdline = process.cmdline()
            except NoSuchProcess:
                if process_cwd is None or process_cmdline is None:
                    self._notify(logging.ERROR, "Can't restart the application "
                                 "because the current working directory and "
                                 "command line could not be retrieved")
                    continue

                # this doesn't race because the uwsift process died
                # prevent auto restart from spawning multiple subprocesses
                os.remove(self.heartbeat_file)

                # don't wait for the subprocess to finish
                # the subprocess will be terminated when the watchdog exits
                process = subprocess.Popen(process_cmdline, cwd=process_cwd)
                application_start_time = now_utc
                old_pid = process.pid
                continue

            if self.restart_interval is not None and not sent_restart_request:
                runtime = now_utc - application_start_time
                if runtime > self.restart_interval:
                    os.kill(pid, signal.SIGHUP)
                    self._notify(logging.INFO, f"Sent restart request to {pid}")
                    if self.ask_again_interval is None:
                        # send the restart request only once
                        sent_restart_request = True
                    else:
                        application_start_time += self.ask_again_interval


if __name__ == "__main__":
    user_cache_dir = appdirs.user_cache_dir(APPLICATION_DIR, APPLICATION_AUTHOR)
    user_config_dir = appdirs.user_config_dir(APPLICATION_DIR,
                                              APPLICATION_AUTHOR, roaming=True)
    config_dir = os.path.join(user_config_dir, "settings", "config")

    try:
        Watchdog([config_dir], user_cache_dir).run()
    except KeyboardInterrupt:
        pass
