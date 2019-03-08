#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
organize_data_topics.py
~~~~~~~~~~~~~~~~~~~~~~~

PURPOSE
Mirror the default AHI data directory structure in to a directory structure organized in to topics.

Note: Hardlinks are required for this and not softlinks because cwrsync for Windows does not properly support softlinks
from basic testing.

Data for SIFT is organized as `/odyssey/isis/tmp/davidh/sift_data/ahi/YYYY_MM_DD_JJJ/HHMM/*B[01-16]*.tif`.
This script will organize them in to `/odyssey/isis/tmp/davidh/sift_data/ahi/<topic>/*B[01-16]*.tif`. All of the files
for a specific case will be in the same directory.

REFERENCES


REQUIRES


:author: David Hoese <david.hoese@ssec.wisc.edu>
:copyright: 2014 by University of Wisconsin Regents, see AUTHORS for more details
:license: GPLv3, see LICENSE for more details
"""
__docformat__ = 'reStructuredText'
__author__ = 'davidh'

import logging
import os
import re
import sys
from collections import namedtuple
from datetime import datetime, timedelta
from glob import glob

LOG = logging.getLogger(__name__)

FILENAME_RE = r'HS_H08_(?P<date>\d{8})_(?P<time>\d{4})_(?P<band>B\d{2})_FLDK_(?P<res>R\d+)\.(?P<ext>.+)'
fn_re = re.compile(FILENAME_RE)

DT_FORMAT = "%Y%m%d_%H%M"
CASE_NAME_FORMAT = "{start}_{end}_{delta:02d}"

DataCase = namedtuple("DataCase", ["topic_title", "start", "end", "delta", "bands"])

### Guam Cases ###
guam_cases = {}
# Kathy's Cases
guam_cases["Introduction"] = []
guam_cases["Introduction"].append(DataCase("Introduction",
                                           datetime(2015, 7, 17, 21, 0, 0),
                                           datetime(2015, 7, 18, 20, 0, 0),
                                           timedelta(minutes=60),
                                           "all"))
guam_cases["Introduction"].append(DataCase("Introduction",
                                           datetime(2015, 7, 18, 1, 0, 0),
                                           datetime(2015, 7, 18, 3, 20, 0),
                                           timedelta(minutes=10),
                                           "all"))
guam_cases["Introduction"].append(DataCase("Introduction",
                                           datetime(2015, 7, 18, 14, 0, 0),
                                           datetime(2015, 7, 18, 16, 0, 0),
                                           timedelta(minutes=10),
                                           "all"))
guam_cases["Introduction"].append(DataCase("Introduction",
                                           datetime(2016, 3, 9, 0, 0, 0),
                                           datetime(2016, 3, 9, 4, 0, 0),
                                           timedelta(minutes=60),
                                           "all"))
guam_cases["Introduction"].append(DataCase("Introduction",
                                           datetime(2016, 3, 9, 1, 30, 0),
                                           datetime(2016, 3, 9, 4, 0, 0),
                                           timedelta(minutes=10),
                                           "all"))

# Scott's Cases
guam_cases["Water Vapor"] = []
guam_cases["Water Vapor"].append(DataCase("Water Vapor",
                                          datetime(2015, 10, 7, 0, 0, 0),
                                          datetime(2015, 10, 8, 0, 0, 0),
                                          timedelta(minutes=30),
                                          "all"))
guam_cases["Water Vapor"].append(DataCase("Water Vapor",
                                          datetime(2016, 2, 19, 19, 0, 0),
                                          datetime(2016, 2, 20, 5, 0, 0),
                                          timedelta(minutes=60),
                                          "all"))

# Tim's Cases
guam_cases["Weighting Functions"] = []
guam_cases["Weighting Functions"].append(DataCase("Weighting Functions",
                                                  datetime(2015, 9, 20, 2, 30, 0),
                                                  datetime(2015, 9, 20, 2, 30, 0),
                                                  timedelta(minutes=0),
                                                  "all"))
guam_cases["Weighting Functions"].append(DataCase("Weighting Functions",
                                                  datetime(2015, 9, 20, 0, 0, 0),
                                                  datetime(2015, 9, 20, 6, 0, 0),
                                                  timedelta(minutes=60),
                                                  "all"))
guam_cases["Weighting Functions"].append(DataCase("Weighting Functions",
                                                  datetime(2015, 9, 20, 1, 30, 0),
                                                  datetime(2015, 9, 20, 2, 30, 0),
                                                  timedelta(minutes=10),
                                                  "all"))
guam_cases["Weighting Functions"].append(DataCase("Weighting Functions",
                                                  datetime(2015, 9, 20, 1, 0, 0),
                                                  datetime(2015, 9, 20, 3, 0, 0),
                                                  timedelta(minutes=10),
                                                  "all"))

# Jordan's Cases
guam_cases["Extra"] = []
guam_cases["Extra"].append(DataCase("Extra",
                                    datetime(2015, 8, 17, 12, 0, 0),
                                    datetime(2015, 8, 18, 12, 0, 0),
                                    timedelta(minutes=60),
                                    "all"))
guam_cases["Extra"].append(DataCase("Extra",
                                    datetime(2015, 8, 17, 22, 0, 0),
                                    datetime(2015, 8, 18, 1, 0, 0),
                                    timedelta(minutes=10),
                                    "all"))
guam_cases["Extra"].append(DataCase("Extra",
                                    datetime(2015, 8, 24, 15, 0, 0),
                                    datetime(2015, 8, 15, 21, 0, 0),
                                    timedelta(minutes=60),
                                    "all"))
guam_cases["Extra"].append(DataCase("Extra",
                                    datetime(2015, 8, 25, 2, 0, 0),
                                    datetime(2015, 8, 25, 5, 0, 0),
                                    timedelta(minutes=10),
                                    "all"))


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Regenerate or generate mirrored AHI data structure")
    parser.add_argument("base_ahi_dir", default="/odyssey/isis/tmp/davidh/sift_data/ahi",
                        help="Base AHI directory for the geotiff data files "
                             "(next child directory is the full dated directory)")
    parser.add_argument('-v', '--verbose', dest='verbosity', action="count",
                        default=int(os.environ.get("VERBOSITY", 2)),
                        help='each occurrence increases verbosity 1 level through '
                             'ERROR-WARNING-Info-DEBUG (default Info)')
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing hardlinks")
    args = parser.parse_args()

    levels = [logging.ERROR, logging.WARN, logging.INFO, logging.DEBUG]
    level = levels[min(3, args.verbosity)]
    logging.basicConfig(level=level)

    if not os.path.isdir(args.base_ahi_dir):
        raise NotADirectoryError("Directory does not exist: %s" % (args.base_ahi_dir,))

    os.chdir(args.base_ahi_dir)

    for section_name, cases in guam_cases.items():
        for case in cases:
            start_str = case.start.strftime(DT_FORMAT)
            end_str = case.end.strftime(DT_FORMAT)
            # Note this only uses the minutes!
            case_name = CASE_NAME_FORMAT.format(start=start_str, end=end_str,
                                                delta=int(case.delta.total_seconds() / 60.0))
            case_dir = os.path.join(args.base_ahi_dir, section_name, case_name)
            if not os.path.isdir(case_dir):
                LOG.info("Creating case directory: %s", case_dir)
                os.makedirs(case_dir)
            else:
                LOG.error("Case directory already exists: %s", case_dir)
                continue

            t = case.start
            while t <= case.end:
                glob_pattern = t.strftime("%Y_%m_%d_%j/%H%M/*_%Y%m%d_%H%M_B??_*.merc.tif")
                t = t + case.delta

                matches = glob(glob_pattern)
                if len(matches) == 0:
                    LOG.error("Zero files found matching pattern: %s", glob_pattern)
                    continue
                for input_pathname in matches:
                    fn = os.path.basename(input_pathname)
                    link_path = os.path.join(case_dir, fn)
                    if os.path.exists(link_path) and not args.overwrite:
                        LOG.debug("Link '%s' already exists, skipping...", link_path)
                        continue
                    LOG.info("Creating hardlink '%s' -> '%s'", link_path, input_pathname)
                    os.link(input_pathname, link_path)
                if int(case.delta.total_seconds()) == 0:
                    LOG.debug("Only one file needed to meet delta of 0")
                    break
            LOG.info("done mirroring files")


if __name__ == "__main__":
    sys.exit(main())
