#!/usr/bin/env python
from time import perf_counter, sleep
from typing import List, Optional

from psutil import NoSuchProcess, Process, process_iter


class PsProfiler:
    file = None

    def __init__(self, cmdline: str):
        self.main_process = self._find_process(cmdline)
        if self.main_process is None:
            raise RuntimeError(f"can't find a process with the following cmdline: {cmdline}")
        print(f"found process: {self.main_process.pid}")

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

    def run(self, out_file: str, interval: float = 0.2):
        assert self.main_process is not None  # nosec B101 # suppress mypy [union-attr]
        start_time = perf_counter()
        self.file = open(out_file, "w")
        self.file.write("time,pid,uss,pss,lib,shared,vms,rss\n")
        while True:
            pss_sum = 0
            time_delta = perf_counter() - start_time

            try:
                processes = self._get_process_tree(self.main_process)
            except NoSuchProcess:
                try:
                    self.main_process.status()
                    continue
                except NoSuchProcess:
                    self.file.close()
                    self.file = None
                    return

            pids = []
            for process in processes:
                try:
                    mem_info = process.memory_full_info()
                except NoSuchProcess:
                    continue

                self.file.write(
                    f"{time_delta:.4f},{process.pid},{mem_info.uss},"
                    f"{mem_info.pss},{mem_info.lib},{mem_info.shared},"
                    f"{mem_info.vms},{mem_info.rss}\n"
                )
                pss_sum += mem_info.pss
                pids.append(process.pid)
            self.file.flush()
            print(f"Memory consumption of {pids}: {pss_sum}")
            sleep(interval)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("PsProfiler")
    parser.add_argument("--cmdline", default="python -m uwsift")
    parser.add_argument("--outfile", required=True)
    args = parser.parse_args()

    ps_profiler = None
    try:
        ps_profiler = PsProfiler(args.cmdline)
        ps_profiler.run(args.outfile)
    except KeyboardInterrupt:
        if ps_profiler is not None and ps_profiler.file is not None:
            ps_profiler.file.close()
