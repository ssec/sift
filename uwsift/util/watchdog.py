#!/usr/bin/env python

import logging
import os
import shlex
import signal
import subprocess  # nosec: B404
from datetime import datetime, timedelta, timezone
from socket import gethostname
from time import sleep
from typing import List, Optional, Tuple

import appdirs
from donfig import Config
from psutil import NoSuchProcess, Process

from uwsift.util.default_paths import APPLICATION_AUTHOR, APPLICATION_NAME

LOG = logging.getLogger(__name__)


# This constant must be kept in sync with the definition in the following
# file: uwsift/__main__.py
WATCHDOG_DATETIME_FORMAT_STORE = "%Y-%m-%d %H:%M:%S %z"


def get_config_value(config: Config, key: str) -> str:
    """
    Wrapper for the `get` method from the donfig library, which provides
    a more friendly error message if the key doesn't exist.

    :param config: Config object from donfig
    :param key: key to the config value
    :return: str or dict with the config value
    """
    try:
        return config.get(key)
    except KeyError:
        raise KeyError(f"Can't find `{key}` in the watchdog config")


class Watchdog:
    ask_again_interval: Optional[timedelta] = None
    restart_interval: Optional[timedelta] = None
    allowed_mem_usage: Optional[int] = None
    notification_cmd: Optional[str] = None

    def __init__(self, config_dirs: List[str], cache_dir: str):
        """
        Create a new Watchdog object.

        :param config_dirs: List of search paths for the watchdog YAML
             configuration files
        :param cache_dir: Path to the SIFT caching directory, which is
             used as the default location of the heartbeat file.
        """
        self.hostname = gethostname()
        config = Config("uwsift", paths=config_dirs)

        heartbeat_file = get_config_value(config, "watchdog.heartbeat_file")
        self.heartbeat_file = heartbeat_file.replace("$$CACHE_DIR$$", cache_dir)

        notification_cmd = config.get("watchdog.notification_cmd", None)
        if not notification_cmd:
            LOG.warning("Can't send notifications" " because `notification_cmd` isn't configured")
        else:
            self.notification_cmd = shlex.quote(notification_cmd)

        self.heartbeat_check_interval = float(get_config_value(config, "watchdog.heartbeat_check_interval"))
        self.max_tolerable_dataset_age = float(get_config_value(config, "watchdog.max_tolerable_dataset_age"))
        self.max_tolerable_idle_time = float(get_config_value(config, "watchdog.max_tolerable_idle_time"))

        restart_interval = int(config.get("watchdog.auto_restart_interval", 0))
        if restart_interval == 0:
            LOG.warning("Auto Restart is disabled")
        else:
            self.restart_interval = timedelta(seconds=restart_interval)

        ask_again_interval = int(config.get("watchdog.auto_restart_ask_again_interval", 0))
        if ask_again_interval == 0:
            LOG.warning("Auto Restart will ask the user only once")
        else:
            self.ask_again_interval = timedelta(seconds=ask_again_interval)

        allowed_max_mem = config.get("watchdog.max_memory_consumption", None)
        if allowed_max_mem:
            self.allowed_mem_usage = self._parse_byte_count(allowed_max_mem)
        else:
            LOG.warning("Memory consumption won't be checked")

        # Variable runtime state
        self.old_pid = None
        self.application_start_time = None
        self.sent_restart_request = False

    @staticmethod
    def _parse_byte_count(byte_count_str: str) -> int:
        """
        Parses the str representation of a byte count and converts the units
        `M` (*Mebibytes*) and `G` (*Gibibytes*) into the appropriate byte count.

        **Note:** To be compatible with *systemd* the units are interpreted with
        the base 1024 (not 1000) although they are called *Megabytes* and
        *Gigabytes* there, see e.g.
        https://www.freedesktop.org/software/systemd/man/systemd.resource-control.html#MemoryHigh=bytes

        :param byte_count: str of the byte count with unit suffix
        :return: number of bytes as int
        """
        if not byte_count_str:
            raise ValueError("expected a unit like `M` or `G`")
        byte_count, unit = int(byte_count_str[:-1]), byte_count_str[-1]

        MEBIBYTE_BYTES = 1024**2
        GIBIBYTE_BYTES = 1024**3

        if unit == "M":
            return byte_count * MEBIBYTE_BYTES
        elif unit == "G":
            return byte_count * GIBIBYTE_BYTES
        else:
            raise ValueError(f"byte count contains unknown unit: {unit}")

    def _read_watchdog_file(self) -> Tuple[int, datetime]:
        """
        Open the watchdog file and parse it.

        :return: tuple of the PID as int and datetime of the dataset creation
        """
        with open(self.heartbeat_file) as file:
            content = file.read()

        pid, timestamp = content.splitlines()
        return int(pid), datetime.strptime(timestamp, WATCHDOG_DATETIME_FORMAT_STORE)

    def _notify(self, level: int, text: str):
        """
        If the notification_cmd was defined in the config, then invoke the
        command using the subprocess API. Otherwise print text with the
        specified level using the logging API.

        :param level: int as defined in the logging package
        :param text: message to log
        """
        if not self.notification_cmd:
            LOG.log(level, text)
        else:
            machine = shlex.quote(self.hostname)
            process_name = shlex.quote(f"{APPLICATION_NAME}-watchdog")
            severity = shlex.quote(logging.getLevelName(level))
            text = shlex.quote(text)
            cmd = [self.notification_cmd, machine, process_name, severity, text]

            try:
                subprocess.run(cmd, shell=True, check=True)  # nosec: B602
            except subprocess.CalledProcessError as err:
                LOG.error(f"Can't run the notification command: {err}")
                LOG.log(level, text)

    def _get_process_tree(self, pid: int) -> List[Process]:
        """
        Traverse the process tree recursively and get all child
        processes of the specified process.

        :param pid: PID of process which may have child processes
        :return: list of specified process with all child processes
        """
        try:
            process = Process(pid)
        except NoSuchProcess:
            return []
        processes = [process]
        for child_process in process.children():
            processes.extend(self._get_process_tree(child_process))
        return processes

    def _get_memory_consumption(self, pid: int) -> int:
        """
        Calculates the memory consumption in bytes of the specified
        process and all its child processes. Shared memory will only
        be counted once.

        :param pid: PID of the process
        :return: memory consumption in bytes or 0 if the process isn't
            alive any more
        """
        pss_sum = 0
        for process in self._get_process_tree(pid):
            try:
                mem_info = process.memory_full_info()
                pss_sum += mem_info.pss
            except NoSuchProcess:
                pass
        return pss_sum

    def _restart_application(self, pid: int):
        """
        Issue a restart request using the SIGUSR1 signal. The process
        may choose to ignore this request.

        Don't use SIGTERM, because ``systemctl --user stop uwsift`` should
        terminate SIFT without asking the user. However the application
        should be able to save its state, so don't use SIGKILL either.

        This function doesn't use the command ``systemctl restart uwsift``,
        because the application may choose to ignore the restart request.
        Systemd first sends SIGTERM and then SIGKILL if the application is
        still alive. The wait time between SIGTERM and SIGKILL could be
        configured by ``TimeoutStopSec=infinity`` in order to accommodate this
        use case, but by doing so the ``systemctl stop uwsift`` command can't
        reliably terminate the program in case of a hang.

        :param pid: PID of SIFT as int
        """
        try:
            os.kill(pid, signal.SIGUSR1)
            self._notify(logging.INFO, f"Sent restart request to {pid}")
        except ProcessLookupError:
            self._notify(logging.WARNING, f"Can't issue restart request because" f" the PID {pid} doesn't exist")

    def run(self):
        """
        Run the watchdog in blocking mode. The watchdog will read the
        heartbeat file periodically and check whether the timestamp of
        the dataset creation is too old. Additional it will ensure that
        the application is restarted from time to time and that the
        memory consumption isn't too high.
        """

        while True:
            if self.restart_interval is None:
                sleep(self.heartbeat_check_interval)
            else:
                sleep(min(self.heartbeat_check_interval, self.restart_interval.seconds))

            try:
                pid, latest_dataset_time = self._read_watchdog_file()
            except ValueError as err:
                self._notify(logging.ERROR, f"Can't parse the watchdog file:" f" {err}")
                continue
            except FileNotFoundError:
                # application might not have been started yet or
                # the application is running but no data was loaded yet
                self._notify(logging.INFO, f"Heartbeat file doesn't exist:" f" {self.heartbeat_file}")
                continue

            now_utc = datetime.now(tz=timezone.utc)

            self._check_idle_time(now_utc)

            self._check_dataset_age(latest_dataset_time, now_utc)

            if self.application_start_time is None:
                self.application_start_time = now_utc

            self._check_application_restarted(now_utc, pid)

            self._check_restart_application_on_memory_shortage(pid)

            self._check_restart_application_on_runtime_exceeded(now_utc, pid)

    def _check_restart_application_on_runtime_exceeded(self, now_utc, pid):
        if self.restart_interval is not None and not self.sent_restart_request:
            runtime = now_utc - self.application_start_time
            if runtime > self.restart_interval:
                self._restart_application(pid)
                if self.ask_again_interval is None:
                    # send the restart request only once
                    self.sent_restart_request = True
                else:
                    self.application_start_time += self.ask_again_interval

    def _check_restart_application_on_memory_shortage(self, pid):
        if self.allowed_mem_usage:
            mem_usage = self._get_memory_consumption(pid)
            if mem_usage > self.allowed_mem_usage:
                LOG.warning(f"program uses too much memory: {mem_usage}")
                self._restart_application(pid)

    def _check_application_restarted(self, now_utc, pid):
        if self.old_pid is None:
            self.old_pid = pid
        elif self.old_pid != pid:
            self._notify(logging.INFO, f"Application was restarted: {pid}")
            self.application_start_time = now_utc
            self.old_pid = pid

    def _check_dataset_age(self, latest_dataset_time, now_utc):
        dataset_age = now_utc - latest_dataset_time
        if dataset_age.total_seconds() > self.max_tolerable_dataset_age:
            overdue_time = dataset_age.total_seconds() - self.max_tolerable_dataset_age
            self._notify(
                logging.WARNING,
                f"Current dataset scheduled time for observation"
                f" ('start_time'): {latest_dataset_time} -"
                f" Next dataset is overdue by "
                f"{overdue_time:.1f} seconds.",
            )
        else:
            self._notify(
                logging.INFO,
                f"Current dataset scheduled time for observation" f" ('start_time'): {latest_dataset_time} - OK",
            )

    def _check_idle_time(self, now_utc):
        # The last update time is implicitly stored as the file modification
        # time. It is possible to approximate this time by checking whether
        # the file content changes, but if the user loads the same dataset
        # again, then this update won't be detected.
        modification_time = datetime.fromtimestamp(os.path.getmtime(self.heartbeat_file), tz=timezone.utc)
        idle_time = now_utc - modification_time
        if idle_time.total_seconds() > self.max_tolerable_idle_time:
            self._notify(logging.WARNING, f"Dataset was not updated" f" since {modification_time}")
        else:
            self._notify(
                logging.INFO,
                f"Dataset was updated {idle_time.total_seconds()}" f" seconds ago at: {modification_time}",
            )


if __name__ == "__main__":
    user_cache_dir = appdirs.user_cache_dir(APPLICATION_NAME, APPLICATION_AUTHOR)
    user_config_dir = appdirs.user_config_dir(APPLICATION_NAME, APPLICATION_AUTHOR, roaming=True)
    config_dir = os.path.join(user_config_dir, "settings", "config")

    try:
        Watchdog([config_dir], user_cache_dir).run()
    except KeyboardInterrupt:
        pass
