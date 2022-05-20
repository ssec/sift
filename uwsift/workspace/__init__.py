#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__init__.py
~~~~~~~~~~~

PURPOSE
Workspace
- owns a reasonably large and fast chunk of disk
- provides memory maps for large datasets
- allows data to be shared with plugins and helpers and other applications


REFERENCES


REQUIRES


:author: R.K.Garcia <rayg@ssec.wisc.edu>
:copyright: 2014 by University of Wisconsin Regents, see AUTHORS for more details
:license: GPLv3, see LICENSE for more details
"""

from .caching_workspace import CachingWorkspace  # noqa: F401
from .simple_workspace import SimpleWorkspace  # noqa: F401
from .workspace import BaseWorkspace  # noqa: F401
