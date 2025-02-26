#!/usr/bin/env python
import atexit
from time import sleep
from typing import Dict, List, Optional

from psutil import NoSuchProcess, Process, process_iter


class DiskManagement:
    """
    DiskManagement can be used to locate temporary files, which are not
    cleaned up by the application itself. When the program terminates
    a detailed report will be printed to stdout containing all open
    files grouped by their access mode.

    :param pid: Process Identifier of the process to be traced
    :param cmdline: command line of the process to be traced
    :raise ValueError: both pid and cmdline are None,
         cmdline of process does not match with specified cmdline,
         process with the specified command line could not be found
    """

    open_files: Dict[str, str] = {}

    def __init__(self, pid: Optional[int] = None, cmdline: Optional[str] = None):
        atexit.register(self._print_open_files)

        if pid is not None:
            process = Process(pid)
            if cmdline is not None:
                process_cmdline = " ".join(process.cmdline())
                if process_cmdline != cmdline:
                    raise ValueError(f"PID {pid} has wrong cmdline: {cmdline} -> {process_cmdline}")
        elif cmdline is not None:
            process = self._find_process(cmdline)
            if process is None:
                raise ValueError(f"process could not be found: {cmdline}")
        else:
            raise ValueError("pid or cmdline must be given")

        self.processes = self._get_process_tree(process)
        print("Observing the following processes:")
        for process in self.processes:
            process_cmdline = " ".join(process.cmdline())
            print(f"\t{process.pid} -> {process_cmdline}")

    @staticmethod
    def _find_process(cmdline: str, timeout: float = 10.0, interval: float = 0.2) -> Optional[Process]:
        """
        Find a process with the specified command line. The command line consists of
        the executable name concatenated with the arguments using spaces.

        :param cmdline: command line containing the executable name and the arguments
        :param timeout: timeout in seconds after which None will be returned
        :param interval: time in seconds between checks
        :return: Process if it was found
        """
        while timeout > 0:
            for process in process_iter(["cmdline"]):
                process_cmdline = " ".join(process.cmdline())
                if process_cmdline == cmdline:
                    return process
            timeout -= interval
            sleep(interval)
        return None

    def _get_process_tree(self, process: Process) -> List[Process]:
        """
        Traverse the process tree recursively and get all child
        processes of the specified process.

        :param process: process which may have child processes
        :return: list of specified process with all child processes
        """
        processes = [process]
        for child_process in process.children():
            processes.extend(self._get_process_tree(child_process))
        return processes

    def collect_open_files(self, interval: float = 0.2) -> None:
        """
        Collect a list of open files from all registered processes.
        If a file is opened in read only mode, then the access mode
        will be "r". If a file is opened in read/write mode, then the
        access mode will be "r+". Also, a file can be reopened with
        write privileges. Additionally, a message with the number of
        new files will be printed to stdout.

        :param interval: time between checks in seconds
        """
        print("\nSearching for open files", end="", flush=True)
        while len(self.processes) > 0:
            new_file_counter = 0
            dead_pids = []

            for process in self.processes:
                try:
                    open_files = process.open_files()
                except (NoSuchProcess, PermissionError):
                    dead_pids.append(process.pid)
                    continue

                for open_file in open_files:
                    old_access_mode = self.open_files.get(open_file.path)
                    if old_access_mode is None:
                        new_file_counter += 1
                    access_mode = "r+" if old_access_mode == "r+" else open_file.mode
                    self.open_files[open_file.path] = access_mode

            self.processes = list(filter(lambda p: p.pid not in dead_pids, self.processes))

            # indicate progress to the user by printing a dot for each new file
            for _ in range(new_file_counter):
                print(".", end="", flush=True)

            sleep(interval)

    def _print_open_files(self) -> None:
        """
        Print all open files and group them into files with only
        read access and files with read/write access.

        :raise RuntimeError: entry has an invalid read/write mode
        """
        read_files = []
        write_files = []
        for path, mode in self.open_files.items():
            if mode == "r+":
                write_files.append(path)
            elif mode == "r":
                read_files.append(path)
            else:
                raise RuntimeError(f"unknown mode `{mode}`: {path}")

        if len(read_files) > 0:
            print("\nREAD:")
            for path in sorted(read_files):
                print(f"\t{path}")

        if len(write_files) > 0:
            print("\nREAD + WRITE:")
            for path in sorted(write_files):
                print(f"\t{path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Disk Management")
    parser.add_argument("--pid", type=int, help="PID of the process, which should be traced")
    parser.add_argument("--cmdline", default="python -m uwsift", help="find the process using the cmdline")
    args = parser.parse_args()

    try:
        disk_management = DiskManagement(pid=args.pid, cmdline=args.cmdline)
        disk_management.collect_open_files()
    except KeyboardInterrupt:
        pass
