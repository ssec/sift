#!/usr/bin/env python
import logging
import os
import random
import shlex
import string
import subprocess  # nosec: B404
import time
from datetime import datetime, timedelta
from itertools import chain
from socket import gethostname
from typing import Dict, List, Optional, Set

import appdirs
from donfig import Config

from uwsift.util.default_paths import APPLICATION_AUTHOR, APPLICATION_NAME

LOG = logging.getLogger(__name__)


class FileMetadata:
    """
    FileMetadata stores the path, size and last data modification time
    of an entry from the filesystem. This class is an implementation
    detail of StorageAgent and not exposed as part of the public API
    surface.

    :param path: absolute path to the filesystem entry
    :param size: size in bytes
    :param mtime: last data modification time in seconds
    """

    def __init__(self, path: str, size: int, mtime: float):
        self.path = path
        self.size = size
        self.last_data_modification = datetime.fromtimestamp(mtime)

    def __repr__(self) -> str:
        return (
            f"FileMetadata {{ path: {self.path}, size: {self.size}, "
            f"last data modification: {self.last_data_modification} }}"
        )


class StorageAgent:
    """
    The StoregeAgent can be used to cleanup temporary files, which are not
    deleted up by the application itself. Only files and directories which
    were not used for some time are deleted by this agent.

    :param files_lifetime: number of seconds after which the entry is deleted
    :param notification_cmd: command to send log messages to a reporting system
    :raise ValueError: files_lifetime is negative or zero
    """

    dir_paths: List[str] = []
    _fs_entries: Dict[str, FileMetadata] = {}
    _ignored_entries: Set[str] = set()

    def __init__(self, files_lifetime: int, notification_cmd: Optional[str]):
        if files_lifetime < 1:
            raise ValueError("files_lifetime must not be negative or zero" " but is {files_lifetime}.")
        self.hostname = gethostname()
        self.files_lifetime = timedelta(seconds=files_lifetime)

        self.notification_cmd = None
        if notification_cmd:
            self.notification_cmd = shlex.quote(notification_cmd)

    def _notify(self, level: int, text: str) -> None:
        """
        Send the logging message to the reporting system if the notification_cmd
        isn't None. The following strings are replaced in the notification_cmd:
        `$$MACHINE$$`, `$$PROCESS_NAME$$`, `$$SEVERITY$$` and `$$TEXT$$`.

        :param level: logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR
        :param text: logging message
        """
        if not self.notification_cmd:
            LOG.log(level, text)
        else:
            machine = shlex.quote(self.hostname)
            process_name = shlex.quote(f"{APPLICATION_NAME}-storage-agent")
            severity = shlex.quote(logging.getLevelName(level))
            text = shlex.quote(text)
            cmd = f"{self.notification_cmd}" f" {machine} {process_name} {severity} {text}"

            try:
                subprocess.run(cmd, shell=True, check=True)  # nosec: 602
            except subprocess.CalledProcessError as err:
                LOG.error(f"Can't run the notification command: {err}")
                LOG.log(level, text)

    def _check_write_access(self, dir_path: str, attempts: int = 5) -> bool:
        """
        Check if the current process has write access to a specific directory.
        Try to create a file with random name and write a few bytes to it.
        If the file already exists, then this process will be repeated.

        :param dir_path: absolute path to a directory
        :param attempts: number of attempts if the random file name already exists
        :return: true if the current process has write access
        """
        for _ in range(attempts):
            # don't use upper and lower case because NTFS/Windows is case insensitive
            char_set = string.ascii_lowercase + string.digits
            random_name = "".join(random.choices(char_set, k=25))  # nosec: B311
            path = os.path.join(dir_path, random_name)

            try:
                with open(path, "wb") as file:
                    file.write(b"Hello World!\n")
            except FileExistsError:
                continue
            except OSError:
                return False

            return True

        self._notify(logging.WARNING, f"encountered a file name clash {attempts} times")
        return False

    def register_directory(self, dir_path: str) -> bool:
        """
        Checks if the directory exists and if this process has write access.
        An IOError will be raised when either condition is violated.
        This can be bypassed by appending the dir_path to self.dir_paths.

        :param dir_path: absolute path to a directory
        :return: False if the directory doesn't exist or has read-only access
        """
        if self._check_write_access(dir_path):
            self.dir_paths.append(dir_path)
            return True
        else:
            self._notify(logging.ERROR, f"directory does not exist or has read-only access: {dir_path}")
            return False

    def _list_filesystem_entries(self, root_dir_path: str) -> List[FileMetadata]:
        """
        Crawl the specified directory and retrieve the metadata for each
        entry if it is not part of the list _ignored_entries.

        :param root_dir_path: absolute path to a directory
        :return: a list of FileMetadata objects
        """
        entries = []

        # Files need to be removed before directories so we visit them from
        # bottom to top. Don't follow symlinks because files outside of the
        # root_dir_path should not be analyzed. The files will be deleted after
        # a certain time and files outside of the temporary directory should
        # not be affected by this tool.
        for root, dirs, files in os.walk(root_dir_path, topdown=False, followlinks=False):
            # return files before dirs in order to prevent a call to rmdir with a non empty directory
            for entry in chain(files, dirs):
                entry_path = os.path.join(root, entry)
                if entry_path not in self._ignored_entries:
                    stat = os.stat(entry_path, follow_symlinks=False)
                    entries.append(FileMetadata(entry_path, stat.st_size, stat.st_mtime))
        return entries

    def _check_for_deletable_entries(self) -> List[FileMetadata]:
        """
        Check all registered directories for new, changed or deteleted
        entries. For each entry a message will be printed if the verbose mode is
        enabled. If a entry changes, then its files_lifetime will be reset. If
        the files_lifetime of an entry reaches zero, then it will be included in
        the returned list.

        :return: list of deletable filesystem entries
        """
        deletable_entries = []
        now = datetime.now()

        checked_paths = set()
        for dir_path in self.dir_paths:
            for entry in self._list_filesystem_entries(dir_path):
                checked_paths.add(entry.path)
                deadline = entry.last_data_modification + self.files_lifetime

                old_entry = self._fs_entries.get(entry.path)
                if old_entry is None:
                    self._fs_entries[entry.path] = entry
                    self._notify(logging.DEBUG, f"[FOUND] {entry.path} -> will be deleted at {deadline}")
                # don't check the size because last_data_modification changes too
                elif old_entry.last_data_modification != entry.last_data_modification:
                    self._fs_entries[entry.path] = entry
                    self._notify(logging.DEBUG, f"[MODIFIED] {entry.path} -> will be deleted at {deadline}")

                if now > deadline:
                    del self._fs_entries[entry.path]
                    deletable_entries.append(entry)

        deleted_entries = []
        for path in self._fs_entries.keys():
            if path not in checked_paths:
                deleted_entries.append(path)
                self._notify(logging.DEBUG, f"[DELETED BY USER] {path}")

        for deleted_entry in deleted_entries:
            del self._fs_entries[deleted_entry]

        return deletable_entries

    def run(self, interval: Optional[int]) -> None:
        """
        Start the scanning of the processes specified by the method
        register_directory. This method will block indefinitely and
        can only be exited using a KeyboardInterrupt Exception.

        :param interval: time between the checks in seconds
        """
        if not self.dir_paths:
            self._notify(logging.WARNING, "no directory was registered")
            return

        if not interval:
            # sleep longer if verbose mode is disabled
            interval = min(int(self.files_lifetime.total_seconds()), 60)
            print(f"Directories will be checked every {interval} seconds")

        if interval <= 0:
            raise ValueError("interval must not be negative or zero" " but is {interval}.")

        while True:
            for deletable_entry in self._check_for_deletable_entries():
                try:
                    if os.path.isdir(deletable_entry.path):
                        os.rmdir(deletable_entry.path)
                    else:
                        os.remove(deletable_entry.path)

                    self._notify(logging.INFO, f"[REMOVED] {deletable_entry.path} (Size: {deletable_entry.size} bytes)")
                except FileNotFoundError:
                    pass
                except OSError as e:
                    self._notify(logging.WARNING, f"entry could not be removed: {e}")
                    # don't try again if the entry can't be removed
                    self._ignored_entries.add(deletable_entry.path)

            time.sleep(float(interval))


if __name__ == "__main__":
    user_cache_dir = appdirs.user_cache_dir(APPLICATION_NAME, APPLICATION_AUTHOR)
    user_config_dir = appdirs.user_config_dir(APPLICATION_NAME, APPLICATION_AUTHOR, roaming=True)
    config_dir = os.path.join(user_config_dir, "settings", "config")

    config = Config("uwsift", paths=[config_dir])

    files_lifetime: int = int(config.get("storage.agent.files_lifetime", -1))
    if files_lifetime < 0:  #
        raise RuntimeError("Config option `files_lifetime` is required")

    notification_cmd = config.get("storage.agent.notification_cmd", None)
    if not notification_cmd:
        LOG.warning("Can't send notifications" " because `notification_cmd` isn't configured")
        notification_cmd = None

    interval = config.get("storage.agent.interval", None)

    agent = StorageAgent(files_lifetime, notification_cmd)
    for path in config.get("storage.agent.directories", []):
        agent.register_directory(path.replace("$$CACHE_DIR$$", user_cache_dir))

    try:
        agent.run(interval)
    except KeyboardInterrupt:
        pass
