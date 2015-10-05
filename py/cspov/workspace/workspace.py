#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
workspace.py
~~~~~~~~~~~~

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
import gdal, osr
import numpy as np
from collections import namedtuple
from uuid import UUID, uuid1 as uuidgen
from PyQt4.QtCore import QObject, pyqtSignal

LOG = logging.getLogger(__name__)

import_progress = namedtuple('import_progress', ['uuid', 'stages', 'current_stage', 'completion', 'stage_desc', 'dataset_info', 'data'])
# stages:int, number of stages this import requires
# current_stage:int, 0..stages-1 , which stage we're on
# completion:float, 0..1 how far we are along on this stage
# stage_desc:tuple(str), brief description of each of the stages we'll be doing


# first instance is main singleton instance; don't preclude the possibility of importing from another workspace later on
TheWorkspace = None


class WorkspaceImporter(object):
    """
    Instances of this class are typically singletons owned by Workspace.
    They're used to perform background activity for importing large input files.
    """
    def __init__(self, **kwargs):
        super(WorkspaceImporter, self).__init__()
    
    def is_relevant(self, source_path=None, source_uri=None):
        """
        return True if this importer is capable of reading this URI.
        """
        return False

    def __call__(self, dest_workspace, dest_wd, dest_uuid, source_path=None, source_uri=None, **kwargs):
        """
        Yield a series of import_status tuples updating status of the import.
        Typically this is going to run on TheQueue when possible.
        :param dest_cwd: destination directory to place flat files into, may be anywhere inside workspace.cwd
        :param dest_uuid: uuid key to use in reference to this dataset at all LODs - may/not be used in file naming, but should be included in datasetinfo
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

    def is_relevant(self, source_path=None, source_uri=None):
        source = source_path or source_uri
        return True if (source.lower().endswith('.tif') or source.lower().endswith('.tiff')) else False

    def __call__(self, dest_workspace, dest_wd, dest_uuid, source_path=None, source_uri=None, **kwargs):
        # yield successive levels of detail as we load
        if source_uri is not None:
            raise NotImplementedError("GeoTiffImporter cannot read from URIs yet")
        d = {}
        gtiff = gdal.Open(source_path)

        # FIXME: consider yielding status at this point so our progress bar starts moving

        ox, cw, _, oy, _, ch = gtiff.GetGeoTransform()
        d["uuid"] = dest_uuid
        d["origin_x"] = ox
        d["origin_y"] = oy
        d["cell_width"] = cw
        d["cell_height"] = ch
        # FUTURE: Should the Workspace normalize all input data or should the Image Layer handle any projection?
        srs = osr.SpatialReference()
        srs.ImportFromWkt(gtiff.GetProjection())
        d["proj"] = srs.ExportToProj4()

        d["name"] = os.path.split(source_path)[-1]
        d["filepath"] = source_path
        item = re.findall(r'_(B\d\d)_', source_path)[-1]  # FIXME: this should be a guidebook
        # Valid min and max for colormap use
        margin = 0.005
        if item in ["B01", "B02", "B03", "B04", "B05", "B06"]:
            # Reflectance/visible data limits
            # FIXME: Are these correct?
            d["clim"] = (0.0, 1.0-margin)
        else:
            # BT data limits
            # FIXME: Are these correct?
            d["clim"] = (200.0, 350.0)

        # FIXME: read this into a numpy.memmap backed by disk in the workspace
        img_data = gtiff.GetRasterBand(1).ReadAsArray()
        img_data = np.require(img_data, dtype=np.float32, requirements=['C'])  # FIXME: is this necessary/correct?

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
    -- datasetinfo dictionaries are ordinary python dictionaries containing ['uuid'], projection metadata, LOD info
    - identify datasets primarily with a UUID object which tracks the dataset and its various representations through the system
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
    didUpdateDataset = pyqtSignal(dict)  # partial completion of a dataset import, new datasetinfo dict is released
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
            self._init_create_workspace()
        else:
            self._own_cwd = False
            self._init_inventory_existing_datasets()
        self._data = {}
        self._info = {}
        self._importers = [x() for x in self.IMPORT_CLASSES]
        if TheWorkspace is None:
            global TheWorkspace  # singleton
            TheWorkspace = self

    def _init_inventory_existing_datasets(self):
        """
        Do an inventory of an pre-existing workspace
        :return:
        """
        pass

    def idle(self):
        """
        Called periodically when application is idle. Does a clean-up tasks and returns True if more needs to be done later.
        Time constrained to ~0.1s.
        :return: True/False, whether or not more clean-up needs to be scheduled.
        """
        return False

    def import_image(self, source_path=None, source_uri=None):
        """
        Start loading URI data into the workspace asynchronously.

        :param source_path:
        :return:
        """
        if source_uri is not None and source_path is None:
            raise NotImplementedError('URI load not yet supported')

        gen = None
        # FIXME: check if the data is already in the workspace
        uuid = uuidgen()
        for imp in self._importers:
            if imp.is_relevant(source_path=source_path):
                gen = imp(self, self.cwd, uuid, source_path=source_path)
                break
        if gen is None:
            raise IOError("unable to import {}".format(source_path))

        # FIXME: for now, just iterate the incremental load. later we want to add this to TheQueue and update the UI as we get more data loaded
        for update in gen:
            if update.data is not None:
                info = self._info[uuid] = update.dataset_info
                data = self._data[uuid] = update.data
                LOG.debug(repr(update))
        return uuid, info, data

    def remove(self, dsi):
        """
        Formally detach a dataset, removing its content from the workspace fully by the time that idle() has nothing more to do.
        :param dsi: datasetinfo dictionary
        :return: None
        """

    def get_info(self, dsi_or_uuid, lod=None):
        """
        :param dsi_or_uuid: existing datasetinfo dictionary, or its UUID
        :param lod: desired level of detail to focus
        :return:
        """
        if isinstance(dsi_or_uuid, str):
            dsi_or_uuid = UUID(dsi_or_uuid)
        return self._info[dsi_or_uuid]

    def get_content(self, dsi_or_uuid, lod=None):
        """
        :param dsi_or_uuid: existing datasetinfo dictionary, or its UUID
        :param lod: desired level of detail to focus
        :return:
        """
        if isinstance(dsi_or_uuid, str):
            dsi_or_uuid = UUID(dsi_or_uuid)
        return self._data[dsi_or_uuid]

    def _position_to_index(self, dsi_or_uuid, xy_pos):
        info = self.get_info(dsi_or_uuid)
        x = xy_pos[0]
        y = xy_pos[1]
        col = (x - info["origin_x"]) / info["cell_width"]
        row = (y - info["origin_y"]) / info["cell_height"]
        return np.round(row), np.round(col)

    def get_content_point(self, dsi_or_uuid, xy_pos):
        row, col = self._position_to_index(dsi_or_uuid, xy_pos)
        data = self.get_content(dsi_or_uuid)
        if not ((0 <= col < data.shape[1]) and (0 <= row < data.shape[0])):
            raise ValueError("X/Y position is outside of image with UUID: %s", dsi_or_uuid)
        return data[row, col]

    def get_content_polygon(self, dsi_or_uuid, points):
        data = self.get_content(dsi_or_uuid)
        xmin = data.shape[1]
        xmax = 0
        ymin = data.shape[0]
        ymax = 0
        for point in points:
            row, col = self._position_to_index(dsi_or_uuid, point)
            if row < ymin:
                ymin = row
            elif row > ymax:
                ymax = row
            if col < xmin:
                xmin = col
            elif col > xmax:
                xmax = col
        return data[ymin:ymax, xmin:xmax]

    def __getitem__(self, datasetinfo_or_uuid):
        """
        return science content proxy capable of generating a numpy array when sliced
        :param datasetinfo_or_uuid: metadata or key for the dataset
        :return: sliceable object returning numpy arrays
        """
        pass

    def asProbeDataSource(self, **kwargs):
        """
        Produce delegate used to match masks to data content.
        :param kwargs:
        :return: delegate object used by probe objects to access workspace content
        """
        return self  # FUTURE: revise this once we have more of an interface specification

    def asLayerDataSource(self, uuid=None, **kwargs):
        """
        produce layer data source delegate to be handed to a LayerRep
        :param kwargs:
        :return:
        """
        return self  # FUTURE: revise this once we have more of an interface specification


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
