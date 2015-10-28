#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ProbeGraphs.py
~~~
This module holds the code which deals with the controller and view for the area probe graphs.

:author: Eva Schiffer <evas@ssec.wisc.edu>
:copyright: 2015 by University of Wisconsin Regents, see AUTHORS for more details
:license: GPLv3, see LICENSE for more details
"""
__author__ = 'evas'
__docformat__ = 'reStructuredText'

from vispy import app
try:
    app_object = app.use_app('pyqt4')
except Exception:
    app_object = app.use_app('pyside')
QtCore = app_object.backend_module.QtCore
QtGui = app_object.backend_module.QtGui
from PyQt4.QtCore import QObject, pyqtSignal

# a useful constant
from cspov.common import INFO

import logging
import numpy

# http://stackoverflow.com/questions/12459811/how-to-embed-matplotib-in-pyqt-for-dummies
# see also: http://matplotlib.org/users/navigation_toolbar.html
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.colors import LogNorm

LOG = logging.getLogger(__name__)

class ProbeGraphManager (QObject) :
    """The ProbeGraphManager manages the many tabs of the Area Probe Graphs.
    """

    # signals
    didChangeTab = pyqtSignal(list,)  # list of probe areas to show

    graphs = None
    selected_graph_index = -1
    workspace = None
    document = None
    tab_widget_object = None
    max_tab_letter = None

    def __init__(self, tab_widget, workspace, document) :
        """Setup our tab widget with an appropriate graph object in the first tab.

        FUTURE, once we are saving our graph configurations, load those instead of setting up this default.
        """

        super(ProbeGraphManager, self).__init__(tab_widget)

        # hang on to the workspace and document
        self.workspace = workspace
        self.document  = document

        # hang on to the tab widget
        self.tab_widget_object = tab_widget
        if self.tab_widget_object.count() != 1 :
            LOG.info("Unexpected number of tabs in the QTabWidget used for the Area Probe Graphs.")

        # set up the first tab
        self.graphs = [ ]
        self.selected_graph_index = 0
        self.max_tab_letter = 'A'
        self.set_up_tab(self.selected_graph_index, do_increment_tab_letter=False)

        # hook things up so we know when the selected tab changes
        self.tab_widget_object.connect(self.tab_widget_object,
                                       QtCore.SIGNAL('currentChanged(int)'),
                                       self.handle_tab_change)

        # hook up the various document signals that would mean we need to reload things
        self.document.didReorderLayers.connect(self.handleLayersChanged)
        self.document.didChangeLayerName.connect(self.handleLayersChanged)
        self.document.didAddLayer.connect(self.handleLayersChanged)
        self.document.willPurgeLayer.connect(self.handleLayersChanged)
        self.document.didSwitchLayerSet.connect(self.handleLayersChanged)

    def set_up_tab (self, tab_index, do_increment_tab_letter=True) :
        """Create a new tab at tab_index and add it to the list of graphs
        """

        # increment our tab label letter if desired
        if do_increment_tab_letter :
            self.max_tab_letter = chr(ord(self.max_tab_letter) + 1) # this will get strange after Z!

        # create our tab
        temp_widget = QtGui.QWidget()
        self.tab_widget_object.insertTab(tab_index, temp_widget, self.max_tab_letter)

        # create the associated graph display object
        self.graphs.append(ProbeGraphDisplay(temp_widget, self.workspace, self.max_tab_letter))

        # go to the tab we just created
        self.tab_widget_object.setCurrentIndex(tab_index)

        # load up the layers for this new tab
        uuid_list = self.document.current_layer_order
        self.graphs[tab_index].set_possible_layers(uuid_list, do_rebuild_plot=True)

    def handleLayersChanged (self) :
        """Used when the document signals that something about the layers has changed
        """

        # reload the layer list for the existing graphs
        uuid_list = self.document.current_layer_order
        for graphObj in self.graphs :
            doRebuild = graphObj is self.graphs[self.selected_graph_index]
            graphObj.set_possible_layers(uuid_list, do_rebuild_plot=doRebuild)

    def currentPolygonChanged (self, polygonPoints) :
        """Update the current polygon in the selected graph and rebuild it's plot

        FUTURE, once the polygon is a layer, this signal will be unnecessary
        """

        return self.graphs[self.selected_graph_index].setPolygon (polygonPoints)

    def handle_tab_change (self, ) :
        """deal with the fact that the tab changed in the tab widget
        """

        newTabIndex = self.tab_widget_object.currentIndex()

        # if this is the last tab, make a new tab and switch to that
        if newTabIndex == (self.tab_widget_object.count() - 1) :
            LOG.info ("Creating new area probe graph tab.")

            self.set_up_tab(newTabIndex)

        # otherwise, just update our current index and make sure the graph is fresh
        else :
            self.selected_graph_index = newTabIndex
            self.graphs[self.selected_graph_index].rebuildPlot()

        currentName = self.graphs[self.selected_graph_index].getName()
        self.didChangeTab.emit([currentName])

class ProbeGraphDisplay (object) :
    """The ProbeGraphDisplay controls one tab of the Area Probe Graphs.
    The ProbeGraphDisplay handles generating a displaying a single graph.
    """

    # the most data we are willing to plot in a scatter plot
    # this limit was determined experimentally on Eva's laptop for glance, may need to revisit this
    MAX_SCATTER_PLOT_DATA = 1e7

    # the default number of bins for the histogram and density scatter plot
    DEFAULT_NUM_BINS = 100

    # the display name of the probe, should be unique across all probes
    myName          = None

    # plotting related controls
    figure          = None
    canvas          = None
    toolbar         = None
    yCheckBox       = None
    xDropDown       = None
    yDropDown       = None

    # internal objects to reference for info and data
    polygon         = None
    workspace       = None

    # internal values that control the behavior of plotting and controls
    xSelectedUUID   = None
    ySelectedUUID   = None
    uuidMap         = None  # this is needed because the drop downs can't properly handle objects as ids
    _stale          = True  # whether or not the plot needs to be redrawn

    def __init__(self, qt_parent, workspace, name_str):
        """build the graph tab controls
        :return:
        """

        # hang on to our name
        self.myName = name_str

        # save the workspace for use later
        self.workspace = workspace

        # a figure instance to plot on
        self.figure = Figure(figsize=(3,3), dpi=72)
        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        self.canvas = FigureCanvas(self.figure, )
        # make sure our figure is clear
        self.clearPlot()

        # make a matplotlib toolbar to attach to the graph
        self.toolbar = NavigationToolbar(self.canvas, qt_parent)

        # create our selection controls

        # the label for the x selection
        xLabel= QtGui.QLabel("X layer:")

        # the check box that turns on and off comparing to a y layer
        self.yCheckBox = QtGui.QCheckBox("vs Y layer:")
        self.yCheckBox.setToolTip("Plot X layer data vs Y layer when this is checked.")
        self.yCheckBox.stateChanged.connect(self.vsChecked)

        # the drop down for selecting the x layer
        self.xDropDown = QtGui.QComboBox(qt_parent)
        self.xDropDown.setToolTip("The X layer data to use for plotting.")
        self.xDropDown.activated.connect(self.xSelected)

        # the drop down for selecting the y layer
        self.yDropDown = QtGui.QComboBox(qt_parent)
        self.yDropDown.setDisabled(True)
        self.yDropDown.setToolTip("The Y layer data to use for plotting.")
        self.yDropDown.activated.connect(self.ySelected)

        # set the layout
        # Note: add in a grid is (widget, row#, col#) or (widget, row#, col#, row_span, col_span)
        layout = QtGui.QGridLayout()
        layout.addWidget(self.toolbar,   1, 1, 1, 3)
        layout.addWidget(self.canvas,    2, 1, 1, 3)
        layout.addWidget(xLabel,         3, 1)
        layout.addWidget(self.xDropDown, 3, 2, 1, 2)
        layout.addWidget(self.yCheckBox, 4, 1)
        layout.addWidget(self.yDropDown, 4, 2, 1, 2)
        qt_parent.setLayout(layout)

    def set_possible_layers (self, uuid_list, do_rebuild_plot=False) :
        """Given a list of layer UUIDs, set the names and UUIDs in the drop downs
        """

        # make a uuid map because the mapping in a combo box doesn't work with objects
        self.uuidMap = { }

        # clear out the current lists
        self.xDropDown.clear()
        self.yDropDown.clear()

        # fill up our lists of layers
        for uuid in uuid_list :

            layer_name = self.workspace.get_info(uuid)[INFO.NAME]
            uuid_string = str(uuid)
            self.xDropDown.addItem(layer_name, uuid_string)
            self.yDropDown.addItem(layer_name, uuid_string)

            self.uuidMap[uuid_string] = uuid

        # if possible, set the selections back to the way they were
        need_rebuild = False
        xIndex = self.xDropDown.findData(str(self.xSelectedUUID))
        if xIndex >= 0 :
            # Selection didn't change
            self.xDropDown.setCurrentIndex(xIndex)
        elif self.xDropDown.count() > 0 :
            # Setting to a new layer
            need_rebuild = True
            self.xSelectedUUID = self.uuidMap[self.xDropDown.itemData(0)]
            self.xDropDown.setCurrentIndex(0)
        else :
            # we had something selected but now there is nothing new to select
            need_rebuild = need_rebuild or self.xSelectedUUID is not None
            self.xSelectedUUID = None
        yIndex = self.yDropDown.findData(str(self.ySelectedUUID))
        if yIndex >= 0 :
            # Selection didn't change
            self.yDropDown.setCurrentIndex(yIndex)
        elif self.yDropDown.count() > 0 :
            # Setting to a new layer
            need_rebuild = need_rebuild or self.yCheckBox.isChecked()
            self.ySelectedUUID = self.uuidMap[self.yDropDown.itemData(0)]
            self.yDropDown.setCurrentIndex(0)
        else :
            # we had something selected but now there is nothing new to select
            need_rebuild = need_rebuild or self.ySelectedUUID is not None
            self.ySelectedUUID = None

        # refresh the plot
        self._stale = need_rebuild
        if do_rebuild_plot:
            # Rebuild the plot (stale is used to determine if actual rebuild is needed)
            self.rebuildPlot()

    def xSelected (self) :
        """The user selected something in the X layer list.
        """

        oldXStr = str(self.xSelectedUUID)
        newXStr = self.xDropDown.itemData(self.xDropDown.currentIndex())
        self.xSelectedUUID = self.uuidMap[newXStr]

        # regenerate the plot
        if oldXStr != newXStr :
            self._stale = True
            self.rebuildPlot()

    def ySelected (self) :
        """The user selected something in the Y layer list.
        """

        oldYStr = str(self.ySelectedUUID)
        newYStr = self.yDropDown.itemData(self.yDropDown.currentIndex())
        self.ySelectedUUID = self.uuidMap[newYStr]

        # regenerate the plot
        if (oldYStr != newYStr) and self.yCheckBox.isChecked() :
            self._stale = True
            self.rebuildPlot()

    def vsChecked (self) :
        """The vs check box was checked!
        """

        # look at the state of the vs box and enable/disable the y drop down accordingly
        doPlotVS = self.yCheckBox.isChecked()
        self.yDropDown.setDisabled(not doPlotVS)

        # regenerate the plot
        self._stale = True
        self.rebuildPlot()

    def setPolygon (self, polygonPoints) :
        """Set the polygon selection for this graph
        """

        self.polygon = polygonPoints

        # regenerate the plot
        self._stale = True
        self.rebuildPlot()

        # return our name to be used for the polygon name
        return self.myName

    def getName (self) :
        """Accessor method for the graph's name
        """
        return self.myName

    def rebuildPlot (self, ) :
        """Given what we currently know about the selection area and selected bands, rebuild our plot

        Note: This should be called only when the selections change in some way.
        """
        if not self._stale:
            LOG.debug("Plot doesn't need to be rebuilt")
            return

        # should be be plotting vs Y?
        doPlotVS = self.yCheckBox.isChecked()

        # if we are plotting only x and we have a selected x and a polygon
        if not doPlotVS and self.xSelectedUUID is not None and self.polygon is not None :

            # get the data and info we need for this plot
            data_polygon = self.workspace.get_content_polygon(self.xSelectedUUID, self.polygon)
            title = self.workspace.get_info(self.xSelectedUUID)[INFO.NAME]

            # plot a histogram
            self.plotHistogram (data_polygon.flatten(), title)

        # if we are plotting x vs y and have x, y, and a polygon
        elif doPlotVS and self.xSelectedUUID is not None and self.ySelectedUUID is not None and self.polygon is not None :

            # get the data and info we need for this plot
            data1 = self.workspace.get_content_polygon(self.xSelectedUUID, self.polygon)
            name1 = self.workspace.get_info(self.xSelectedUUID)[INFO.NAME]
            data2 = self.workspace.get_content_polygon(self.ySelectedUUID, self.polygon)
            name2 = self.workspace.get_info(self.ySelectedUUID)[INFO.NAME]

            # we can only scatter plot if both data sets have the same resolution
            if data1.size != data2.size :
                LOG.info("Unable to plot bands of different resolutions in the Area Probe Graph.")
                self.clearPlot()
            else:
                # plot a scatter plot
                self.plotScatterplot (data1.flatten(), name1, data2.flatten(), name2)

        # if we have some combination of selections we don't understand, clear the figure
        else :

            self.clearPlot()

        self._stale = False


    def plotHistogram (self, data, title, numBins=100) :
        """Make a histogram using the given data and label it with the given title
        """
        self.figure.clf()
        axes = self.figure.add_subplot(111)
        axes.hist(data, bins=self.DEFAULT_NUM_BINS)
        axes.set_title(title)
        self.canvas.draw()

    def plotScatterplot (self, dataX, nameX, dataY, nameY) :
        """Make a scatter plot of the x and y data
        """

        # we should have the same size data here
        assert(dataX.size == dataY.size)

        if dataX.size > self.MAX_SCATTER_PLOT_DATA :
            LOG.info("Too much data in selected region to generate scatter plot.")
            self.clearPlot()
            #self.plotDensityScatterplot(dataX, nameX, dataY, nameY)

        else :
            self.figure.clf()
            axes = self.figure.add_subplot(111)
            axes.scatter(dataX.flatten(), dataY.flatten(), color='b', s=1, alpha=0.5)
            axes.set_xlabel(nameX)
            axes.set_ylabel(nameY)
            axes.set_title(nameX + " vs " + nameY)
            self._draw_xy_line(axes)

            self.canvas.draw()

    # TODO, come back to this when we are properly backgrounding our plots
    def plotDensityScatterplot (self, dataX, nameX, dataY, nameY) :
        """Make a density scatter plot for the given data
        """

        # flatten our data
        dataX = dataX.flatten()
        dataY = dataY.flatten()

        # clear the figure and make a new subplot
        self.figure.clf()
        axes = self.figure.add_subplot(111)

        # figure out the range of the data
        min_value = min(numpy.min(dataX), numpy.min(dataY))
        max_value = max(numpy.max(dataX), numpy.max(dataY))
        # bounds should be defined in the form [[xmin, xmax], [ymin, ymax]]
        bounds = [[min_value, max_value], [min_value, max_value]]

        # make the binned density map for this data set
        density_map, _, _ = numpy.histogram2d(dataX, dataY, bins=self.DEFAULT_NUM_BINS, range=bounds)
        # mask out zero counts; flip because y goes the opposite direction in an imshow graph
        density_map = numpy.flipud(numpy.transpose(numpy.ma.masked_array(density_map, mask=density_map == 0)))

        # display the density map data
        axes.imshow(density_map, extent=[min_value, max_value, min_value, max_value],
                    interpolation='nearest', norm=LogNorm())

        # TODO make a colorbar
        #colorbar = self.figure.colorbar()
        #colorbar.set_label('log(count of data points)')

        # set the various text labels
        axes.set_xlabel(nameX)
        axes.set_ylabel(nameY)
        axes.set_title(nameX + " vs " + nameY)

        # draw the x vs y line
        self._draw_xy_line(axes)

    def clearPlot (self) :
        """Clear our plot
        """

        self.figure.clf()
        self.canvas.draw()

    def _draw_xy_line (self, axes) :

        # get the bounds for our calculations and so we can reset the viewing window later
        x_bounds = axes.get_xbound()
        y_bounds = axes.get_ybound()

        # figure out the size of the ranges
        xrange = x_bounds[1] - x_bounds[0]
        yrange = y_bounds[1] - y_bounds[0]

        # draw the x=y line
        perfect = [max(x_bounds[0], y_bounds[0]), min(x_bounds[1], y_bounds[1])]
        axes.plot(perfect, perfect, '--', color='k', label='X = Y')

        # reset the bounds
        axes.set_xbound(x_bounds)
        axes.set_ybound(y_bounds)
