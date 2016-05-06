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

import PyQt4.QtCore as QtCore
import PyQt4.QtGui as QtGui
from PyQt4.QtCore import QObject, pyqtSignal

# a useful constant
from cspov.common import INFO
from cspov.queue import TASK_PROGRESS, TASK_DOING

import logging
import numpy

# http://stackoverflow.com/questions/12459811/how-to-embed-matplotib-in-pyqt-for-dummies
# see also: http://matplotlib.org/users/navigation_toolbar.html
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.colors import LogNorm

# Stuff for custom toolbars
try:
    import six
    from matplotlib.backends.qt_compat import QtWidgets
    import matplotlib.backends.qt_editor.figureoptions as figureoptions
except ImportError:
    figureoptions = None

LOG = logging.getLogger(__name__)
DEFAULT_POINT_PROBE = "default_probe_name"


class NavigationToolbar(NavigationToolbar):
    """Custom matplotlib toolbar
    """
    def edit_parameters(self):
        allaxes = self.canvas.figure.get_axes()
        if len(allaxes) == 1:
            axes = allaxes[0]
        else:
            titles = []
            for axes in allaxes:
                title = axes.get_title()
                ylabel = axes.get_ylabel()
                label = axes.get_label()
                if title:
                    fmt = "%(title)s"
                    # if ylabel:
                    #     fmt += ": %(ylabel)s"
                    # fmt += " (%(axes_repr)s)"
                elif ylabel:
                    fmt = "%(axes_repr)s (%(ylabel)s)"
                elif label:
                    fmt = "%(axes_repr)s (%(label)s)"
                else:
                    fmt = "%(axes_repr)s"
                titles.append(fmt % dict(title=title,
                                         ylabel=ylabel, label=label,
                                         axes_repr=repr(axes)))
            if len(titles) == 2 and "Colorbar" in titles:
                other_idx = titles.index("Colorbar") - 1
                axes = allaxes[other_idx]
            else:
                item, ok = QtWidgets.QInputDialog.getItem(
                    self.parent, 'Customize', 'Select axes:', titles, 0, False)
                if ok:
                    axes = allaxes[titles.index(six.text_type(item))]
                else:
                    return

        figureoptions.figure_edit(axes, self)

class ProbeGraphManager (QObject) :
    """The ProbeGraphManager manages the many tabs of the Area Probe Graphs.
    """

    # signals
    didChangeTab = pyqtSignal(list,)  # list of probe areas to show
    didClonePolygon = pyqtSignal(str, str)
    drawChildGraph = pyqtSignal(str,)
    pointProbeChanged = pyqtSignal(str, bool, tuple)

    graphs = None
    selected_graph_index = -1
    workspace = None
    queue = None
    document = None
    tab_widget_object = None
    max_tab_letter = None

    def __init__(self, tab_widget, workspace, document, queue) :
        """Setup our tab widget with an appropriate graph object in the first tab.

        FUTURE, once we are saving our graph configurations, load those instead of setting up this default.
        """

        super(ProbeGraphManager, self).__init__(tab_widget)

        # hang on to the workspace and document
        self.workspace = workspace
        self.document  = document
        self.queue = queue

        # hang on to the tab widget
        self.tab_widget_object = tab_widget
        if self.tab_widget_object.count() != 1 :
            LOG.info("Unexpected number of tabs in the QTabWidget used for the Area Probe Graphs.")
        # hold on to point probe locations (point probes are shared across tabs)
        self.point_probes = {}

        # set up the first tab
        self.graphs = [ ]
        self.selected_graph_index = 0
        self.max_tab_letter = 'A'
        self.set_up_tab(self.selected_graph_index, do_increment_tab_letter=False)

        # hook things up so we know when the selected tab changes
        self.tab_widget_object.connect(self.tab_widget_object,
                                       QtCore.SIGNAL('currentChanged(int)'),
                                       self.handle_tab_change)
        self.drawChildGraph.connect(self.draw_child)

        # hook up the various document signals that would mean we need to reload things
        self.document.didReorderLayers.connect(self.handleLayersChanged)
        self.document.didChangeLayerName.connect(self.handleLayersChanged)
        self.document.didAddLayer.connect(self.handleLayersChanged)
        self.document.willPurgeLayer.connect(self.handleLayersChanged)
        self.document.didSwitchLayerSet.connect(self.handleLayersChanged)

    def draw_child(self, child_name):
        for child in self.graphs:
            if child.myName == child_name:
                child.draw()
                break

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
        graph = ProbeGraphDisplay(self, temp_widget, self.workspace, self.queue, self.document, self.max_tab_letter)
        self.graphs.append(graph)

        # load up the layers for this new tab
        uuid_list = self.document.current_layer_order
        graph.set_possible_layers(uuid_list)

        # clone the previous tab
        if self.selected_graph_index != tab_index:
            # if we aren't setting up the initial tab, clone the current tab
            current_graph = self.graphs[self.selected_graph_index]
            graph.set_default_layer_selections(current_graph.xSelectedUUID, current_graph.ySelectedUUID)
            # give it a copy of the current polygon
            graph.setPolygon(current_graph.polygon[:] if current_graph.polygon is not None else None)
            graph.checked = current_graph.checked
            point_status, point_xy = self.point_probes[DEFAULT_POINT_PROBE]
            point_xy = point_xy if point_status else None
            graph.setPoint(point_xy, rebuild=False)

        # Create the initial plot
        graph.rebuildPlot()

        # go to the tab we just created
        self.tab_widget_object.setCurrentIndex(tab_index)

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

    def update_point_probe(self, probe_name, xy_pos=None, state=None):
        if xy_pos is None and state is None:
            # they didn't ask to change anything
            return
        if probe_name not in self.point_probes:
            # new point
            if xy_pos is None:
                raise ValueError("Point probe '{}' does not exist".format(probe_name))
            # if this is a new point probe, then it must be enabled
            state = True if state is None else state
        else:
            old_state, old_xy_pos = self.point_probes[probe_name]
            if xy_pos is None:
                # state is what is changing
                xy_pos = old_xy_pos
            elif state is None:
                # they are updating the position only
                # we have to turn the probe back on
                state = True

            if old_state == state and old_xy_pos == xy_pos:
                # nothing has changed so no need to tell anyone
                return
            if old_state != state:
                LOG.info("Changing point probe '{}' state to '{}'".format(probe_name, "on" if state else "off"))
            if old_xy_pos != xy_pos:
                LOG.info("Changing point probe '{}' position to '{}'".format(probe_name, xy_pos))

        self.point_probes[probe_name] = [state, xy_pos]
        self.pointProbeChanged.emit(probe_name, state, xy_pos)

    def update_point_probe_graph(self, probe_name, state, xy_pos):
        # need to set the point for all graphs because the point probe
        # is used across all plots
        for idx, graph in enumerate(self.graphs):
            rebuild = idx == self.selected_graph_index
            if state:
                graph.setPoint(xy_pos, rebuild=rebuild)
            elif state is not None:
                # if it is False/"off"
                graph.setPoint(None, rebuild=rebuild)

    def current_point_probe_status(self, probe_name):
        if probe_name not in self.point_probes:
            raise KeyError("Probe '{}' does not exist".format(probe_name))
        return self.point_probes[probe_name]

    def toggle_point_probe(self, probe_name, state=None):
        if probe_name not in self.point_probes:
            LOG.info("No point probe to toggle")
            return

        old_state = self.point_probes[probe_name][0]
        state = state if state is not None else not old_state
        self.update_point_probe(probe_name, state=state)

    def set_default_layer_selections(self, *uuids):
        """Set the UUIDs for the current graph if it doesn't have a polygon
        """
        return self.graphs[self.selected_graph_index].set_default_layer_selections(*uuids)

    def handle_tab_change (self, ) :
        """deal with the fact that the tab changed in the tab widget
        """

        newTabIndex = self.tab_widget_object.currentIndex()

        # if this is the last tab, make a new tab and switch to that
        if newTabIndex == (self.tab_widget_object.count() - 1) :
            LOG.info ("Creating new area probe graph tab.")

            old_name = self.graphs[self.selected_graph_index].getName()
            self.set_up_tab(newTabIndex)

            # notify everyone that we cloned a polygon (if we did)
            if self.graphs[self.selected_graph_index].polygon is not None:
                new_name = self.graphs[-1].getName()
                self.didClonePolygon.emit(old_name, new_name)

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
    point           = None
    manager         = None
    workspace       = None
    queue           = None
    document        = None

    # internal values that control the behavior of plotting and controls
    xSelectedUUID   = None
    ySelectedUUID   = None
    uuidMap         = None  # this is needed because the drop downs can't properly handle objects as ids
    _stale          = True  # whether or not the plot needs to be redrawn

    def __init__(self, manager, qt_parent, workspace, queue, document, name_str):
        """build the graph tab controls
        :return:
        """

        # hang on to our name
        self.myName = name_str

        # save the workspace and queue for use later
        self.manager = manager
        self.workspace = workspace
        self.queue = queue
        self.document = document

        # a figure instance to plot on
        self.figure = Figure(figsize=(3,3), dpi=72)
        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        self.canvas = FigureCanvas(self.figure, )
        self.canvas.setMinimumSize(100, 100)
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

    def set_default_layer_selections(self, *layer_uuids):
        # only set the defaults if we don't have a polygon yet
        if self.polygon is not None:
            return
        if len(layer_uuids) > 2:
            raise ValueError("Probe graphs can handle a maximum of 2 layers (got %d)", len(layer_uuids))

        if len(layer_uuids) >= 1:
            xIndex = self.xDropDown.findData(str(layer_uuids[0]))
            if xIndex >= 0:
                self.xDropDown.setCurrentIndex(xIndex)
                self.xSelectedUUID = layer_uuids[0]
            else:
                LOG.error("Tried to set probe graph to non-existent layer: %s", layer_uuids[0])

        if len(layer_uuids) >= 2:
            yIndex = self.xDropDown.findData(str(layer_uuids[1]))
            if yIndex >= 0:
                self.yDropDown.setCurrentIndex(yIndex)
                self.ySelectedUUID = layer_uuids[1]
            else:
                LOG.error("Tried to set probe graph to non-existent layer: %s", layer_uuids[1])

    @property
    def checked(self):
        return self.yCheckBox.isChecked()

    @checked.setter
    def checked(self, is_checked):
        return self.yCheckBox.setChecked(is_checked)

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

    def setPoint(self, coordinates, rebuild=True):
        self.point = coordinates
        self._stale = True
        # sometimes we set the point to be redrawn later
        if rebuild:
            self.rebuildPlot()

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
        task_name = "%s_%s_region_plotting" % (self.xSelectedUUID, self.ySelectedUUID)
        self.queue.add(task_name, self._rebuild_plot_task(self.xSelectedUUID, self.ySelectedUUID, self.polygon, self.point, plot_versus=doPlotVS), "Creating plot for region probe data", interactive=True)
        # Assume that the task gets resolved otherwise we might try to draw multiple times
        self._stale = False

    def _rebuild_plot_task(self, x_uuid, y_uuid, polygon, point_xy, plot_versus=False):

        # if we are plotting only x and we have a selected x and a polygon
        if not plot_versus and x_uuid is not None and polygon is not None :
            yield {TASK_DOING: 'Probe Plot: Collecting polygon data...', TASK_PROGRESS: 0.0}

            # get the data and info we need for this plot
            data_polygon = self.workspace.get_content_polygon(x_uuid, polygon)
            fmt, units, data_polygon = self.document.convert_units(x_uuid, data_polygon)
            title = self.workspace.get_info(x_uuid)[INFO.NAME]

            # get point probe value
            if point_xy:
                x_point = self.workspace.get_content_point(x_uuid, point_xy)
                format_str, unit_str, x_point = self.document.convert_units(x_uuid, x_point)
            else:
                x_point = None

            # plot a histogram
            yield {TASK_DOING: 'Probe Plot: Creating histogram plot', TASK_PROGRESS: 0.25}
            self.plotHistogram (data_polygon, title, x_point)

        # if we are plotting x vs y and have x, y, and a polygon
        elif plot_versus and x_uuid is not None and y_uuid is not None and polygon is not None :
            yield {TASK_DOING: 'Probe Plot: Collecting polygon data (layer 1)...', TASK_PROGRESS: 0.0}

            # get the data and info we need for this plot
            name1 = self.workspace.get_info(x_uuid)[INFO.NAME]
            name2 = self.workspace.get_info(y_uuid)[INFO.NAME]
            hires_uuid = self.workspace.lowest_resolution_uuid(x_uuid, y_uuid)
            hires_coord_mask, hires_data = self.workspace.get_coordinate_mask_polygon(hires_uuid, polygon)
            _, _, hires_data = self.document.convert_units(hires_uuid, hires_data)
            yield {TASK_DOING: 'Probe Plot: Collecting polygon data (layer 2)...', TASK_PROGRESS: 0.15}
            if hires_uuid is x_uuid:
                # the hires data was from the X UUID
                data1 = hires_data
                data2 = self.workspace.get_content_coordinate_mask(y_uuid, hires_coord_mask)
                _, _, data2 = self.document.convert_units(y_uuid, data2)
            else:
                # the hires data was from the Y UUID
                data2 = hires_data
                data1 = self.workspace.get_content_coordinate_mask(x_uuid, hires_coord_mask)
                _, _, data1 = self.document.convert_units(x_uuid, data1)
            yield {TASK_DOING: 'Probe Plot: Creating scatter plot...', TASK_PROGRESS: 0.25}

            if point_xy:
                x_point = self.workspace.get_content_point(x_uuid, point_xy)
                format_str, unit_str, x_point = self.document.convert_units(x_uuid, x_point)
                y_point = self.workspace.get_content_point(y_uuid, point_xy)
                format_str, unit_str, y_point = self.document.convert_units(x_uuid, y_point)
            else:
                x_point = None
                y_point = None

            # plot a scatter plot
            # self.plotScatterplot (data1, name1, data2, name2)
            self.plotDensityScatterplot (data1, name1, data2, name2, x_point, y_point)

        # if we have some combination of selections we don't understand, clear the figure
        else :
            yield {TASK_DOING: 'Probe Plot: Clearing plot figure...', TASK_PROGRESS: 0.0}
            self.clearPlot()

        yield {TASK_DOING: 'Probe Plot: Drawing plot...', TASK_PROGRESS: 0.95}
        self.manager.drawChildGraph.emit(self.myName)
        yield {TASK_DOING: 'Probe Plot: Done', TASK_PROGRESS: 1.0}

    def draw(self):
        self.canvas.draw()

    def plotHistogram (self, data, title, x_point, numBins=100) :
        """Make a histogram using the given data and label it with the given title
        """
        self.figure.clf()
        axes = self.figure.add_subplot(111)
        bars = axes.hist(data, bins=self.DEFAULT_NUM_BINS)
        if x_point is not None:
            # go through each rectangle object and make the one that contains x_point 'red'
            # default color is blue so red should stand out
            for bar in bars[2][::-1]:
                if bar.xy[0] <= x_point:
                    bar.set_color('red')
                    break
        axes.set_title(title)

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
            axes.scatter(dataX, dataY, color='b', s=1, alpha=0.5)
            axes.set_xlabel(nameX)
            axes.set_ylabel(nameY)
            axes.set_title(nameX + " vs " + nameY)
            self._draw_xy_line(axes)

    def plotDensityScatterplot (self, dataX, nameX, dataY, nameY, pointX, pointY) :
        """Make a density scatter plot for the given data
        """

        # clear the figure and make a new subplot
        self.figure.clf()
        axes = self.figure.add_subplot(111)

        # figure out the range of the data
        # you might not be comparing the same units
        xmin_value = numpy.min(dataX)
        xmax_value = numpy.max(dataX)
        ymin_value = numpy.min(dataY)
        ymax_value = numpy.max(dataY)
        # bounds should be defined in the form [[xmin, xmax], [ymin, ymax]]
        bounds = [[xmin_value, xmax_value], [ymin_value, ymax_value]]

        # make the binned density map for this data set
        density_map, _, _ = numpy.histogram2d(dataX, dataY, bins=self.DEFAULT_NUM_BINS, range=bounds)
        # mask out zero counts; flip because y goes the opposite direction in an imshow graph
        density_map = numpy.flipud(numpy.transpose(numpy.ma.masked_array(density_map, mask=density_map == 0)))

        # display the density map data
        img = axes.imshow(density_map, extent=[xmin_value, xmax_value, ymin_value, ymax_value], aspect='auto',
                          interpolation='nearest', norm=LogNorm())
        if pointX is not None:
            axes.set_autoscale_on(False)
            axes.plot(pointX, pointY, marker='o',
                      markerfacecolor='white', markeredgecolor='black',
                      markersize=10, markeredgewidth=1.)
            axes.set_autoscale_on(True)

        colorbar = self.figure.colorbar(img)
        colorbar.ax.set_title("Colorbar")  # for the 'Customize' menu in the MPL toolbar
        colorbar.set_label('log(count of data points)')

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
