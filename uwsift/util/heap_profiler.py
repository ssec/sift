#!/usr/bin/env python

import logging
import os
import tracemalloc
from collections import OrderedDict
from datetime import datetime
from threading import Event, Thread

try:
    import psutil
except ImportError:
    psutil = None

LOG = logging.getLogger(__name__)


def format_byte_count(byte_count: int):
    # min_byte_count can't be 0, because it's impossible to divide by 0
    if byte_count == 0:
        return "0 B"

    prefix = ""
    if byte_count < 0:
        prefix = "-"
        byte_count = abs(byte_count)

    symbols = OrderedDict()
    for idx, symbol in enumerate(["B", "KiB", "MiB", "GiB", "TiB", "PiB"]):
        symbols[symbol] = 1 << idx * 10
    for symbol, min_byte_count in reversed(symbols.items()):
        if byte_count >= min_byte_count:
            value = byte_count / min_byte_count
            return f"{prefix}{value:.2f} {symbol}"
    raise RuntimeError(f"can't display the byte count: {byte_count}")


class RepeatableTimer(Thread):
    finished = Event()

    def __init__(self, sec_interval: float, callback, *, args=None, kwargs=None):
        Thread.__init__(self)
        self.sec_interval = sec_interval
        self.callback = callback
        self.args = args if args is not None else []
        self.kwargs = kwargs if kwargs is not None else {}

    def cancel(self):
        self.finished.set()

    def run(self):
        while not self.finished.is_set():
            self.finished.wait(self.sec_interval)
            self.callback(*self.args, **self.kwargs)


class HeapProfiler:
    snapshot_idx = 0

    def __init__(self, sec_interval: float):
        if psutil is None:
            raise ImportError("Missing 'psutil' dependency which is required for heap profiling.")
        self.timer = RepeatableTimer(sec_interval, self._record_heap_usage)

        directory_name = datetime.now().strftime("%Y%m%d_%H-%M_uwsift_heap_profile")
        self.directory_path = os.path.join(os.getcwd(), directory_name)
        os.mkdir(self.directory_path)
        LOG.info(f"The snapshots will be saved in the following directory: {self.directory_path}")

    def start(self):
        tracemalloc.start()
        self.timer.start()

    def cancel(self):
        self.timer.cancel()

    def _record_heap_usage(self):
        snapshot = tracemalloc.take_snapshot()
        snapshot.dump(os.path.join(self.directory_path, f"{self.snapshot_idx}.snapshot"))

        allocated_bytes = 0
        allocation_count = 0
        for statistic in snapshot.statistics("lineno"):
            allocated_bytes += statistic.size
            allocation_count += statistic.count

        allocated_bytes = format_byte_count(allocated_bytes)
        physical_ram_usage = psutil.virtual_memory()[2]
        cpu_usage = psutil.cpu_percent()
        pid = os.getpid()
        LOG.info(
            f"[PID {pid}] Python: {allocation_count} allocations -> {allocated_bytes} "
            + f"(System usage: CPU {cpu_usage}% - Physical memory {physical_ram_usage}%)"
        )

        self.snapshot_idx += 1
