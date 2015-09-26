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

import os, sys
import logging, unittest, argparse
from PyQt4.QtCore import QObject, pyqtSignal, QThread

LOG = logging.getLogger(__name__)

# keys for status dictionaries
TASK_DOING = ("activity", str)
TASK_PROGRESS = ("progress", float) # 0.0 - 1.0 progress


# singleton instance used by clients
TheQueue = None


class TaskQueue(QThread):
    """
    Global background task queue for loading, rendering, et cetera.
    Includes state updates and GUI links.
    Eventually will include thread pools and multiprocess pools.
    """
    process_pool = None  # process pool for background activity
    thread_pool = None  # thread pool for background activity
    queue = None

    didMakeProgress = pyqtSignal(list)  # sequence of dictionaries listing update information to be propagated to view
    # started : inherited
    # finished : inherited
    # terminated : inherited

    def __init__(self, process_pool=None, thread_pool=None):
        super(TaskQueue, self).__init__()
        self.process_pool = process_pool
        self.thread_pool = thread_pool
        self.queue = []
        assert(TheQueue is None)
        global TheQueue
        TheQueue = self

    def add(self, key, task_iterable, description, use_process_pool=False, use_thread_pool=False):
        """
        Add an iterable task which will yield progress information dictionaries.

        Expect behavior like this:
         for task in queue:
            for status_info in task:
                update_display(status_info)
            pop_display(final_status_info)

        :param key: unique key for task; queuing the same key will result in the old task being removed and the new one deferred to the end
        :param value: iterable task object
        :return:
        """
        self.queue.append(task_iterable)
        self.start()  # safe if already running

    def _did_progress(self, task_status):
        """
        Summarize the task queue, including progress, and send it out as a signal
        :param task_status:
        :return:
        """
        # FIXME: this should have entries for the upcoming stuff as well so we can have an Activity panel
        info = [task_status]
        self.didMakeProgress.emit(info)

    def run(self):
        while len(self.queue)>0:
            task = self.queue.pop(0)
            for status in task:
                self._did_progress(status)


def test_task():
    for dex in range(10):
        yield {TASK_DOING: 'test task', TASK_PROGRESS: float(dex)/10.0}
        TheQueue.sleep(1)
    yield {TASK_DOING: 'test task', TASK_PROGRESS: 1.0}




def main():
    parser = argparse.ArgumentParser(
        description="PURPOSE",
        epilog="",
        fromfile_prefix_chars='@')
    parser.add_argument('-v', '--verbose', dest='verbosity', action="count", default=0,
                        help='each occurrence increases verbosity 1 level through ERROR-WARNING-INFO-DEBUG')
    # http://docs.python.org/2.7/library/argparse.html#nargs
    # parser.add_argument('--stuff', nargs='5', dest='my_stuff',
    #                    help="one or more random things")
    parser.add_argument('pos_args', nargs='*',
                        help="positional arguments don't have the '-' prefix")
    args = parser.parse_args()

    levels = [logging.ERROR, logging.WARN, logging.INFO, logging.DEBUG]
    logging.basicConfig(level=levels[min(3, args.verbosity)])

    if not args.pos_args:
        unittest.main()
        return 0

    for pn in args.pos_args:
        pass

    return 0


if __name__ == '__main__':
    sys.exit(main())
