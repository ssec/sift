#!/usr/bin/env python
from typing import Dict, List, Optional, Set
from itertools import chain
from datetime import datetime, timedelta
import string
import random
import time
import os


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
        return f"FileMetadata {{ path: {self.path}, size: {self.size}, " \
               f"last data modification: {self.last_data_modification} }}"


class StorageAgent:
    """
    The StoregeAgent can be used to cleanup temporary files, which are not
    deleted up by the application itself. Only files and directories which
    were not used for some time are deleted by this agent.

    :param lifetime: number of seconds after which the entry is deleted
    :param verbose: true if the verbose output should be enabled
    :raise ValueError: lifetime is negative or zero
    """
    dir_paths: List[str] = []
    _fs_entries: Dict[str, FileMetadata] = {}
    _ignored_entries: Set[str] = set()

    def __init__(self, lifetime: int, verbose: bool = False):
        if lifetime < 1:
            raise ValueError("lifetime can't be negative or zero")
        self.lifetime = timedelta(seconds=lifetime)
        self.verbose = verbose

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
            random_name = "".join(random.choices(char_set, k=25))
            path = os.path.join(dir_path, random_name)

            try:
                with open(path, "wb") as file:
                    file.write(b"Hello World!\n")
            except FileExistsError:
                continue
            except (OSError, IOError):
                return False

            return True

        if self.verbose:
            print(f"encountered a file name clash {attempts} times")

        return False

    def register_directory(self, dir_path: str) -> None:
        """
        Checks if the directory exists and if this process has write access.
        An IOError will be raised when either condition is violated.
        This can be bypassed by appending the dir_path to self.dir_paths.

        :param dir_path: absolute path to a directory
        :raise IOError: directory doesn't exist or has read-only access
        """
        if self._check_write_access(dir_path):
            self.dir_paths.append(dir_path)
        else:
            raise IOError(f"directory does not exist or has read-only access: {dir_path}")

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
        entries. For each entry a message will be printed if the verbose mode
        is enabled. If a entry changes, then its lifetime will be reset. If
        the lifetime of an entry reaches zero, then it will be included in the
        returned list.

        :return: list of deletable filesystem entries
        """
        deletable_entries = []
        now = datetime.now()

        checked_paths = set()
        for dir_path in self.dir_paths:
            for entry in self._list_filesystem_entries(dir_path):
                checked_paths.add(entry.path)
                deadline = entry.last_data_modification + self.lifetime

                old_entry = self._fs_entries.get(entry.path)
                if old_entry is None:
                    self._fs_entries[entry.path] = entry
                    if self.verbose:
                        print(f"[FOUND] {entry.path} -> will be deleted at {deadline}")
                # don't check the size because last_data_modification changes too
                elif old_entry.last_data_modification != entry.last_data_modification:
                    self._fs_entries[entry.path] = entry
                    if self.verbose:
                        print(f"[MODIFIED] {entry.path} -> will be deleted at {deadline}")

                if now > deadline:
                    del self._fs_entries[entry.path]
                    deletable_entries.append(entry)

        deleted_entries = []
        for path in self._fs_entries.keys():
            if path not in checked_paths:
                deleted_entries.append(path)
                if self.verbose:
                    print(f"[DELETED BY USER] {path}")

        for deleted_entry in deleted_entries:
            del self._fs_entries[deleted_entry]

        return deletable_entries

    def start(self, interval: Optional[int] = None) -> None:
        """
        Start the scanning of the processes specified by the method
        register_directory. This method will block indefinitely and
        can only be exited using a KeyboardInterrupt Exception.

        :param interval: time between the checks in seconds
        """
        if interval is None:
            # sleep longer if verbose mode is disabled
            interval = min(int(self.lifetime.total_seconds()), 5 if self.verbose else 600)
            print(f"Directories will be checked every {interval} seconds")

        while True:
            for deletable_entry in self._check_for_deletable_entries():
                try:
                    if os.path.isdir(deletable_entry.path):
                        os.rmdir(deletable_entry.path)
                    else:
                        os.remove(deletable_entry.path)

                    print(f"[REMOVED] {deletable_entry.path} (Size: {deletable_entry.size} bytes)")
                except FileNotFoundError:
                    pass
                except OSError as e:
                    print(f"WARNING: entry could not be removed: {e}")
                    # don't try again if the entry can't be removed
                    self._ignored_entries.add(deletable_entry.path)

            time.sleep(float(interval))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Storage Agent")
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose",
                        help="enable verbose output")
    parser.add_argument("-l", "--lifetime", type=int, required=True,
                        help="set the lifetime in seconds for the file entries")
    parser.add_argument("paths", nargs="+",
                        help="paths to the observed directories")
    args = parser.parse_args()

    agent = StorageAgent(args.lifetime, args.verbose)
    for arg_path in args.paths:
        agent.register_directory(arg_path)

    try:
        agent.start()
    except KeyboardInterrupt:
        pass
