#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
.py
~~~

PURPOSE


REFERENCES


REQUIRES


:author: R.K.Garcia <rayg@ssec.wisc.edu>
:copyright: 2014 by University of Wisconsin Regents, see AUTHORS for more details
:license: GPLv3, see LICENSE for more details
"""
__author__ = 'rayg'
__docformat__ = 'reStructuredText'

import logging

from PyQt5.QtCore import QObject

# from PyQt4.QtGui import QAction

LOG = logging.getLogger(__name__)
FORMAT_GUIDEBOOK = {}


class UserAddsFileToDoc(QObject):
    """
    Manage an open dialog and queue up adding files to the document
    """
    _main = None
    _open_dialog = None
    _task_queue = None

    def __init__(self, main_window, open_dialog, task_queue):
        super(UserAddsFileToDoc, self).__init__()
        self._main = main_window
        self._task_queue = task_queue
        self._open_dialog = open_dialog

    def open_files(self):
        """
        activate dialog
        do any previewing appropriate
        when ready, add file to document
        load coarse representation immediately
        queue up background refinement
        :return:
        """
        pass

    def __call__(self, files_to_open):
        """
        given information from dialog box,
        for each file
            identify file type using guidebook

        :param files_to_open:
        :return:
        """
        pass
