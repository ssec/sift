#!/usr/bin/env python

import logging
import os
import pickle  # nosec: B403
import re
import sys
import tracemalloc
from collections import OrderedDict, defaultdict
from typing import Generator, List, Optional

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from uwsift.util.heap_profiler import format_byte_count

LOG = logging.getLogger(__name__)


class HeapAnalyzer:
    _allocations: defaultdict = defaultdict(list)
    _combined_snapshot_count = 0

    @staticmethod
    def load_snapshots(snapshot_directory: str) -> Generator[List[tracemalloc.Statistic], None, None]:
        file_name_regex = re.compile(r"^([0-9]+)\.snapshot$")

        filtered_files = {}
        for file in os.listdir(snapshot_directory):
            if file.endswith(".snapshot"):
                captures = file_name_regex.search(file)
                if captures is None:
                    continue
                idx = int(captures.group(1))
                filtered_files[idx] = file

        print(f"Found {len(filtered_files)} snapshot files")

        for idx in range(len(filtered_files)):
            snapshot_path = os.path.join(snapshot_directory, filtered_files[idx])
            print(f"Loading snapshot: {snapshot_path}")

            snapshot = tracemalloc.Snapshot.load(snapshot_path)
            yield snapshot.statistics("lineno")

    @staticmethod
    def get_python_stdlib_path() -> Optional[str]:
        dir_regex = re.compile(os.path.join(sys.exec_prefix, "lib", "python3.[0-9]+"))
        for path in sys.path:
            if dir_regex.match(path):
                return path + os.path.sep
        else:
            LOG.warning("can't find the path to the Python standard library")
            return None

    @staticmethod
    def get_conda_packages_path(stdlib_directory: str) -> Optional[str]:
        conda_directory = os.path.join(stdlib_directory, "site-packages")
        if os.path.isdir(conda_directory):
            return conda_directory
        else:
            LOG.warning("can't find the path to the Conda packages")
            return None

    @staticmethod
    def get_uwsift_project_path() -> Optional[str]:
        util_directory = os.path.dirname(__file__)
        uwsift_directory = os.path.dirname(util_directory)
        if os.path.isdir(uwsift_directory):
            return uwsift_directory
        else:
            LOG.warning("can't find the `uwsift` project directory")
            return None

    def combine_snapshot(self, snapshot: List[tracemalloc.Statistic]):
        self._combined_snapshot_count += 1

        for statistic in snapshot:
            frame = statistic.traceback[0]
            location = (frame.filename, frame.lineno)
            self._allocations[location].append({"count": statistic.count, "size": statistic.size})

        for location, allocs in self._allocations.items():
            if len(allocs) < self._combined_snapshot_count:
                needed_zeros = self._combined_snapshot_count - len(allocs)
                self._allocations[location].extend([{"count": 0, "size": 0}] * needed_zeros)

    def dump_combined_snapshots(self, path: str):
        with open(path, "wb") as file:
            pickle.dump(self._allocations, file)

    def load_combined_snapshots(self, path: str):
        with open(path, "rb") as file:
            self._allocations = pickle.load(file)  # nosec: B301

        for _, allocs in self._allocations.items():
            if self._combined_snapshot_count == 0:
                self._combined_snapshot_count = len(allocs)
            else:
                # the lists must have the same length for the stacked plot
                assert self._combined_snapshot_count == len(allocs)  # nosec B101

    def _get_sorted_allocations(self, sort_key: str = "size") -> OrderedDict:
        if sort_key not in ["size", "count"]:
            raise ValueError("sort key must be size or count")

        sorted_list_of_allocs = sorted(
            self._allocations.items(), reverse=True, key=lambda item: sum(alloc[sort_key] for alloc in item[1])
        )

        sorted_dict = OrderedDict()
        for location, allocs in sorted_list_of_allocs:
            sorted_dict[location] = allocs
        return sorted_dict

    def get_top_stats(self, limit: int = 25) -> Generator[str, None, None]:
        sorted_allocations = self._get_sorted_allocations()
        for idx in range(self._combined_snapshot_count):
            output = ""
            total_bytes = 0
            other_bytes = 0
            other_alloc_count = 0
            for rank, (location, allocs) in enumerate(sorted_allocations.items()):
                alloc = allocs[idx]
                total_bytes += alloc["size"]
                if rank < limit:
                    file_name, line_number = location
                    byte_count = format_byte_count(alloc["size"])
                    output += f"#{rank + 1}: {file_name}:{line_number}: {alloc['count']} allocations -> {byte_count}\n"
                else:
                    other_bytes += alloc["size"]
                    other_alloc_count += alloc["count"]

            other_bytes = format_byte_count(other_bytes)
            other_location_count = len(sorted_allocations) - limit
            output += f"{other_location_count} other locations: {other_alloc_count} -> {other_bytes}\n"

            total_bytes = format_byte_count(total_bytes)
            output += f"==> Total allocated size: {total_bytes}\n"
            yield output

    def create_plot(self, top_sort: str = "size", limit: int = 25):
        sorted_allocations = self._get_sorted_allocations()
        other_allocations = [0] * self._combined_snapshot_count
        x_axis = range(self._combined_snapshot_count)
        stacked_y_axis = []
        labels = []

        uwsift_project_path = self.get_uwsift_project_path()
        python_stdlib_path = self.get_python_stdlib_path()
        if python_stdlib_path is not None:
            conda_packages_path = self.get_conda_packages_path(python_stdlib_path)
        else:
            conda_packages_path = None

        for rank, ((file_name, line_number), allocs) in enumerate(sorted_allocations.items()):
            if rank < limit:
                # use "relative" paths because the absolute paths are too long
                if conda_packages_path and file_name.startswith(conda_packages_path):
                    file_name = "<conda>/%s" % file_name[len(conda_packages_path) :]
                elif python_stdlib_path and file_name.startswith(python_stdlib_path):
                    file_name = "<python-stdlib>/%s" % file_name[len(python_stdlib_path) :]
                elif uwsift_project_path and file_name.startswith(uwsift_project_path):
                    file_name = "<uwsift>/%s" % file_name[len(uwsift_project_path) :]

                labels.append(f"{file_name}:{line_number}")
                stacked_y_axis.append([alloc[top_sort] for alloc in allocs])
            else:
                for idx, alloc in enumerate(allocs):
                    other_allocations[idx] += alloc[top_sort]

        stacked_y_axis.append(other_allocations)
        labels.append("other allocations")

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.stackplot(x_axis, *stacked_y_axis, labels=labels)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: format_byte_count(int(x))))
        ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=3, borderaxespad=0, frameon=False)
        ax.grid(True, "major", "y")
        plt.subplots_adjust(top=0.75)
        plt.xlabel("Time")
        plt.ylabel("RAM Usage")
        plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze snapshots from the Heap Profiler")
    parser.add_argument("--snapshot-dir", help="path to the snapshot directory")
    parser.add_argument("--combine", help="combine all snapshots and save them at the given path")
    parser.add_argument("--load", help="path to the combined snapshot")
    parser.add_argument("--plot", action="store_true", help="draw a plot using matplotlib")
    parser.add_argument("--text", action="store_true", help="print the data to stdout")
    args = parser.parse_args()

    analyzer = HeapAnalyzer()

    if args.combine:
        if not args.snapshot_dir:
            parser.error("--snapshot-dir must be passed")

        for data in analyzer.load_snapshots(args.snapshot_dir):
            analyzer.combine_snapshot(data)

        analyzer.dump_combined_snapshots(args.combine)
    elif args.load:
        analyzer.load_combined_snapshots(args.load)
        if args.plot:
            analyzer.create_plot()
        elif args.text:
            for top_stats in analyzer.get_top_stats():
                print(top_stats)
        else:
            parser.error("--plot or --text must be used")
    else:
        parser.error("--combine or --load must be used")
