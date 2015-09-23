#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
.py
~~~

PURPOSE
Implement Workspace, a singleton object which manages large amounts of data
- background loading, up to and including reprojection
- providing memory-compatible, stride-able arrays
- accepting data from external sources written in arbitrary languages

REFERENCES


REQUIRES


:author: R.K.Garcia <rayg@ssec.wisc.edu>
:copyright: 2014 by University of Wisconsin Regents, see AUTHORS for more details
:license: GPLv3, see LICENSE for more details
"""
__author__ = 'rayg'
__docformat__ = 'reStructuredText'

import os, sys, re
import logging, unittest, argparse
from PyQt4.QtCore import QObject, pyqtSignal
import gdal
from collections import namedtuple


LOG = logging.getLogger(__name__)

import_progress = namedtuple('import_progress', ['uuid', 'stages', 'current_stage', 'completion', 'stage_desc', 'dataset_info', 'data'])
# stages:int, number of stages this import requires
# current_stage:int, 0..stages-1 , which stage we're on
# completion:float, 0..1 how far we are along on this stage
# stage_desc:tuple(str), brief description of each of the stages we'll be doing


class WorkspaceImporter(object):
    """
    Instances of this class are typically singletons owned by Workspace.
    They're used to perform background activity for importing large input files.
    """
    def __init__(self, **kwargs):
        super(WorkspaceImporter, self).__init__()
    
    def is_relevant(self, source_uri=None, source_path=None):
        """
        return True if this importer is capable of reading this URI.
        """
        return False

    def __call__(self, dest_cwd, dest_uuid, source_uri=None, source_path=None, process_pool=None, **kwargs):
        """
        Yield a series of import_status tuples updating status of the import.
        :param dest_cwd: destination directory to place flat files into
        :param dest_uuid: uuid key to use in reference to this dataset at all LODs - may or may not be used in file naming, but should be included in datasetinfo
        :param source_uri: uri to load from
        :param source_path: path to load from (alternative to source_uri)
        :return: sequence of import_progress, the first and last of which must include data,
                 inbetween updates typically will release data when stages complete and have None for dataset_info and data fields
        """
        raise NotImplementedError('subclass must implement')


class GeoTiffImporter(WorkspaceImporter):
    """
    GeoTIFF data importer
    """
    def __init__(self, **kwargs):
        super(GeoTiffImporter, self).__init__()

    def is_relevant(self, uri):
        return True if (uri.lower().endswith('.tif') or uri.lower().endswith('.tiff')) else False

    def __call__(self, dest_cwd, dest_uuid, source_uri=None, source_path=None, process_pool=None, **kwargs):
        # yield successive levels of detail as we load
        if source_uri is not None:
            raise NotImplementedError("GeoTiffImporter cannot read from URIs yet")
        d = {}
        gtiff = gdal.Open(source_path)

        # FIXME: consider yielding status at this point so our progress bar starts moving

        ox, cw, _, oy, _, ch = gtiff.GetGeoTransform()
        d["origin_x"] = ox
        d["origin_y"] = oy
        d["cell_width"] = cw
        d["cell_height"] = ch
        # FUTURE: Should the Workspace normalize all input data or should the Image Layer handle any projection?
        srs = gdal.osr.SpatialReference()
        srs.ImportFromWkt(gtiff.GetProjection())
        d["proj"] = srs.ExportToProj4()

        d["name"] = os.path.split(source_path)[-1]
        d["filepath"] = source_path
        item = re.findall(r'_(B\d\d)_', source_path)[-1]  # FIXME: this should be a guidebook
        # Valid min and max for colormap use
        if item in ["B01", "B02", "B03", "B04", "B05", "B06"]:
            # Reflectance/visible data limits
            # FIXME: Are these correct?
            d["clim"] = (0.0, 1.0)
        else:
            # BT data limits
            # FIXME: Are these correct?
            d["clim"] = (200.0, 350.0)

        img_data = gtiff.GetRasterBand(1).ReadAsArray()

        # Full resolution shape
        # d["shape"] = self.get_dataset_data(item, time_step).shape
        d['shape'] = img_data.shape

        # normally we would place a numpy.memmap in the workspace with the content of the geotiff raster band/s here

        # single stage import with all the data for this simple case
        zult = import_progress(uuid=dest_uuid,
                               stages=1,
                               current_stage=0,
                               completion=1.0,
                               stage_desc="loading geotiff",
                               dataset_info=d,
                               data=img_data)
        yield zult
        # further yields would logically add levels of detail with their own sampling values
        # FIXME: provide example of multiple LOD loading and how datasetinfo dictionary/dictionaries look in that case
        # note that once the coarse data is yielded, we may be operating in another thread - think about that for now?






class Workspace(QObject):
    """
    Workspace is a singleton object which works with Datasets shall:
    - own a working directory full of recently used datasets
    - provide DatasetInfo dictionaries for shorthand use between application subsystems
    - unpack data in "packing crate" formats like NetCDF into memory-compatible flat files
    - efficiently create on-demand subsections and strides of raster data as numpy arrays
    - incrementally cache often-used subsections and strides ("image pyramid") using appropriate tools like gdal
    - notify subscribers of changes to datasets (Qt signal/slot pub-sub)
    - during idle, clean out unused/idle data content, given DatasetInfo contents provides enough metadata to recreate
    - interface to external data processing or loading plug-ins and notify application of new-dataset-in-workspace
    """
    cwd = None  # directory we work in
    _own_cwd = None  # whether or not we created the cwd - which is also whether or not we're allowed to destroy it
    _pool = None  # process pool that importers can use for background activities, if any
    _importers = None  # list of importers to consult when asked to start an import
    _info = None
    _data = None

    # signals
    didStartImport = pyqtSignal(dict)  # a dataset started importing; generated after overview level of detail is available
    didMakeImportProgress = pyqtSignal(dict)
    didImportLevelOfDetail = pyqtSignal(dict)  # partial completion of a dataset import
    didFinishImport = pyqtSignal(dict)  # all loading activities for a dataset have completed
    didDiscoverExternalDataset = pyqtSignal(dict)  # a new dataset was added to the workspace from an external agent

    IMPORT_CLASSES = [ GeoTiffImporter ]


    def __init__(self, directory_path=None, process_pool=None):
        """
        Initialize a new or attach an existing workspace, creating any necessary bookkeeping.
        """
        super(Workspace, self).__init__()
        self.cwd = directory_path = os.path.abspath(directory_path)
        if not os.path.isdir(directory_path):
            os.makedirs(directory_path)
            self._own_cwd = True
        else:
            self._own_cwd = False
        self._data = {}
        self._info = {}


    def idle(self):
        """
        Called periodically when application is idle. Does a clean-up tasks and returns True if more needs to be done later.
        Time constrained to ~0.1s.
        :return: True/False, whether or not more clean-up needs to be scheduled.
        """
        return False

    def import_uri(self, uri):
        """
        Start loading URI data into the workspace asynchronously.
        When enough of the data is available to produce and overview,
        return a DatasetInfo dictionary which can be used by client as a token to grab data.
        :param uri:
        :return:
        """

    def import_file(self, pathname):
        """
        Start loading URI data into the workspace asynchronously.

        :param pathname:
        :return:
        """
        gen = None
        uuid = pathname  # FIXME
        for imp in self._importers:
            if imp.is_relevant(source_path=pathname):
                gen = imp(self.cwd, uuid, source_path=pathname)  # FIXME: use a real unique id to track this dataset
                break
        if gen is None:
            raise IOError("unable to import {}".format(pathname))

        # FIXME: for now, just iterate the incremental load. later we want to add this to TheQueue and update the UI as we get more data loaded
        for update in gen:
            if update.data is not None:
                info = self._info[uuid] = update.dataset_info
                data = self._data[uuid] = update.data
                LOG.debug(repr(update))
        return info, data





    def remove(self, dsi):
        """
        Formally detach a dataset, removing its content from the workspace fully by the time that idle() has nothing more to do.
        :param dsi: datasetinfo dictionary
        :return: None
        """

    def __getitem__(self, datasetinfo):
        """
        return a dataset or dataset proxy capable of generating a numpy array when sliced
        :param datasetinfo: metadata on the dataset
        :return: sliceable object returning numpy arrays
        """
        pass


    def asProbeDataSource(self, **kwargs):
        """
        Delegate used to match masks to data content.
        :param kwargs:
        :return: delegate object used by probe objects to access workspace content
        """


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
