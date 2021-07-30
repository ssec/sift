#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
queue.py
~~~~~~~~

PURPOSE
Global background task queue for loading, rendering, et cetera.
Use TheQueue.add() to create background behavior.
Note that Qt4 facilities other than signals should not be used on the task queue!

REFERENCES


REQUIRES


:author: R.K.Garcia <rayg@ssec.wisc.edu>
:copyright: 2014 by University of Wisconsin Regents, see AUTHORS for more details
:license: GPLv3, see LICENSE for more details
"""
__author__ = 'rayg'
__docformat__ = 'reStructuredText'

import logging
from collections import OrderedDict

from PyQt5.QtCore import QObject, pyqtSignal, QThread

LOG = logging.getLogger(__name__)

# keys for status dictionaries
TASK_DOING = ("activity", str)
TASK_PROGRESS = ("progress", float)  # 0.0 - 1.0 progress

# singleton instance used by clients
TheQueue = None


class Worker(QThread):
    """
    Worker thread use by TaskQueue
    """
    queue = None
    depth = 0

    # worker id, sequence of dictionaries listing update information to be propagated to view
    workerDidMakeProgress = pyqtSignal(int, list)
    # task-key, ok: False if exception occurred else True
    workerDidCompleteTask = pyqtSignal(str, bool)

    def __init__(self, myid: int):
        super(Worker, self).__init__()
        self.queue = OrderedDict()
        self.depth = 0
        self.id = myid

    def add(self, key, task_iterable):
        # FUTURE: replace queued task if key matches
        self.queue[key] = task_iterable
        self.depth = len(self.queue)
        self.start()

    def _did_progress(self, task_status):
        """
        Summarize the task queue, including progress, and send it out as a signal
        :param task_status:
        :return:
        """
        # FIXME: this should have entries for the upcoming stuff as well so we can have an Activity panel
        info = [task_status] if task_status else []
        self.workerDidMakeProgress.emit(self.id, info)

    def run(self):
        while len(self.queue) > 0:
            key, task = self.queue.popitem(last=False)
            # LOG.debug('starting background work on {}'.format(key))
            ok = True
            try:
                for status in task:
                    self._did_progress(status)
            except Exception:
                # LOG.error("Background task failed")
                LOG.error("Background task exception: ", exc_info=True)
                ok = False
            self.workerDidCompleteTask.emit(key, ok)
        self.depth = 0
        self._did_progress(None)


class TaskQueue(QObject):
    """
    Global background task queue for loading, rendering, et cetera.
    Includes state updates and GUI links.
    Eventually will include thread pools and multiprocess pools.
    Two threads for interactive tasks (high priority), one thread for background tasks (low priority): 0, 1, 2
    """
    process_pool = None  # process pool for background activity
    workers = None  # thread pool for background activity
    _interactive_round_robin = 0  # swaps between 0 and 1 for interactive tasks
    _last_status = None  # list of last status reports for different workers
    _completion_futures = None  # dictionary of id(task) : completion(bool)

    didMakeProgress = pyqtSignal(list)  # sequence of dictionaries listing update information to be propagated to view

    # started : inherited
    # finished : inherited
    # terminated : inherited

    def __init__(self, process_pool=None, worker_count=3):
        super(TaskQueue, self).__init__()
        self._interactive_round_robin = 0
        self.process_pool = process_pool
        self._completion_futures = {}
        self._last_status = []
        self.workers = []
        for id in range(3):
            worker = Worker(id)
            worker.workerDidMakeProgress.connect(self._did_progress)
            worker.workerDidCompleteTask.connect(self._did_complete_task)
            self.workers.append(worker)
            self._last_status.append(None)

        global TheQueue
        assert (TheQueue is None)
        TheQueue = self

    @property
    def depth(self):
        return sum([x.depth for x in self.workers])

    @property
    def remaining(self):
        return sum([len(x.queue) for x in self.workers])

    def add(self, key, task_iterable, description, interactive=False, and_then=None, use_process_pool=False,
            use_thread_pool=False):
        """Add an iterable task which will yield progress information dictionaries.

        Expect behavior like this::

         for task in queue:
            for status_info in task:
                update_display(status_info)
            pop_display(final_status_info)

        Args:
            key (str): unique key for task. Queuing the same key will result in the old task being removed
                and the new one deferred to the end
            task_iterable (iter): callable resulting in an iterable, or an iterable itself to be run on the background

        """
        if interactive:
            wdex = self._interactive_round_robin
            self._interactive_round_robin += 1
            self._interactive_round_robin %= 2
        else:
            wdex = 2
        if callable(and_then):
            self._completion_futures[key] = and_then
        self.workers[wdex].add(key, task_iterable)

    def _did_progress(self, worker_id, worker_status):
        """
        Summarize the task queue, including progress, and send it out as a signal
        :param worker_status: list of active items, or at least the thing it's working on now
        :return:
        """
        # FIXME: this should consolidate entries for the upcoming stuff as well so we can have an Activity panel

        self._last_status[worker_id] = worker_status

        # report on the lowest worker number that's active; (0,1 interactive; 2 background)
        # yes, this will be redundant
        # FUTURE make this a more useful signal content, rather than relying on progress_ratio back-query
        for wdex, status in enumerate(self._last_status):
            if self.workers[wdex].isRunning() and status is not None:
                self.didMakeProgress.emit(status)
                return

        # otherwise this is a notification that we're finally at full idle
        self.didMakeProgress.emit([{TASK_DOING: '', TASK_PROGRESS: 0.0}])
        # FUTURE: consider one progress bar per worker

    def _did_complete_task(self, task_key: str, succeeded: bool):
        # LOG.debug("background task complete!")
        todo = self._completion_futures.pop(task_key, None)
        if callable(todo):
            LOG.debug("completed task {}, and_then we do this...".format(succeeded))
            todo(succeeded)
        # else:
        #     LOG.debug("nothing further to do <{}>".format(repr(todo)))

    def progress_ratio(self, current_progress=None):
        depth = self.depth
        if depth == 0:
            return 0.0
        elif depth == 1 and current_progress is not None:
            # show something other than 50% if there is only 1 job
            return current_progress
        else:
            depth, remaining = self.depth, self.remaining
            return float(depth - remaining) / depth


def test_task():
    for dex in range(10):
        yield {TASK_DOING: 'test task', TASK_PROGRESS: float(dex) / 10.0}
        TheQueue.sleep(1)
    yield {TASK_DOING: 'test task', TASK_PROGRESS: 1.0}
