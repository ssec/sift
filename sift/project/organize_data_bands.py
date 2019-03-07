#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
organize_data_bands.py
~~~~~~~~~~~~~~~~~~~~~~

PURPOSE
Mirror the default AHI data directory structure with band being the primary key using hardlinks.

Note: Hardlinks are required for this and not softlinks because cwrsync for Windows does not properly support softlinks
from basic testing.

Data for SIFT is organized as `/odyssey/isis/tmp/davidh/sift_data/ahi/YYYY_MM_DD_JJJ/HHMM/*B[01-16]*.tif`. This
doesn't make it easy to select a band for multiple time steps so this script will create hardlinks so the structure is:
`/odyssey/isis/tmp/davidh/sift_data/ahi/B[01-16]/*B[01-16]*.tif`

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
from glob import glob

LOG = logging.getLogger(__name__)

FILENAME_RE = r'HS_H08_(?P<date>\d{8})_(?P<time>\d{4})_(?P<band>B\d{2})_FLDK_(?P<res>R\d+)\.(?P<ext>.+)'
fn_re = re.compile(FILENAME_RE)


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

    for tif_file in glob(os.path.join("????_??_??_???", "????", "*.merc.tif")):
        dated_dir, time_dir, tif_fn = tif_file.split(os.sep)
        m = fn_re.match(tif_fn)
        if m is None:
            LOG.warning("Filename '%s' does not match regular expression", tif_file)
            continue
        nfo = m.groupdict()
        link_path = os.path.join(nfo["band"], tif_fn)
        if os.path.exists(link_path) and not args.overwrite:
            LOG.debug("Link '%s' already exists, skipping...", link_path)
            continue
        link_dir = os.path.dirname(link_path)
        if not os.path.isdir(link_dir):
            LOG.info("Creating directory for link: %s", link_dir)
            os.makedirs(link_dir)
        LOG.info("Creating hardlink '%s' -> '%s'", link_path, tif_file)
        os.link(tif_file, link_path)
    LOG.info("Done mirroring files")


if __name__ == "__main__":
    sys.exit(main())
