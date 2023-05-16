#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
probes.py
~~~
This module holds the code which deals with the controller and view for the area probe graphs.

:author: Eva Schiffer <evas@ssec.wisc.edu>
:copyright: 2015 by University of Wisconsin Regents, see AUTHORS for more details
:license: GPLv3, see LICENSE for more details
"""
__author__ = "evas"
__docformat__ = "reStructuredText"

import logging

import numpy as np

# http://stackoverflow.com/questions/12459811/how-to-embed-matplotib-in-pyqt-for-dummies
# see also: http://matplotlib.org/users/navigation_toolbar.html
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure
from PyQt5 import QtWidgets
from PyQt5.QtCore import QObject, Qt, pyqtSignal

# a useful constant
from uwsift.common import Info
from uwsift.model.layer_model import LayerModel
from uwsift.queue import TASK_DOING, TASK_PROGRESS

# Stuff for custom toolbars
try:
    import matplotlib.backends.qt_editor.figureoptions as figureoptions
except ImportError:
    figureoptions = None

LOG = logging.getLogger(__name__)
DEFAULT_POINT_PROBE = "default_probe_name"


class CustomNavigationToolbar(NavigationToolbar):
    """Custom matplotlib toolbar."""

    def __init__(self, *args, **kwargs):
        self.__include_colorbar = kwargs.get("include_colorbar", False)
        super(CustomNavigationToolbar, self).__init__(*args, **kwargs)

    def edit_parameters(self):
        allaxes = self.canvas.figure.get_axes()
        if not allaxes:
            QtWidgets.QMessageBox.warning(self.parent, "Error", "There are no axes to edit.")
            return
        elif len(allaxes) == 1:
            (axes,) = allaxes
        else:
            titles = []
            not_colorbar_idx = -1
            for idx, axes in enumerate(allaxes):
                if any(x.colorbar for x in axes.images):
                    not_colorbar_idx = idx
                name = (
                    axes.get_title()
                    or " - ".join(filter(None, [axes.get_xlabel(), axes.get_ylabel()]))
                    or "<anonymous {} (id: {:#x})>".format(type(axes).__name__, id(axes))
                )
                titles.append(name)

            if len(titles) == 2 and not_colorbar_idx != -1 and not self.__include_colorbar:
                axes = allaxes[not_colorbar_idx]
            else:
                item, ok = QtWidgets.QInputDialog.getItem(self.parent, "Customize", "Select axes:", titles, 0, False)
                if ok:
                    axes = allaxes[titles.index(str(item))]
                else:
                    return

        figureoptions.figure_edit(axes, self)


class ProbeGraphManager(QObject):
    """The ProbeGraphManager manages the many tabs of the Area Probe Graphs."""

    # signals
    didChangeTab = pyqtSignal(
        tuple,
    )  # list of probe areas to show
    didClonePolygon = pyqtSignal(str, str)
    drawChildGraph = pyqtSignal(
        str,
    )
    pointProbeChanged = pyqtSignal(str, bool, tuple)

    def __init__(self, tab_widget, auto_update_checkbox, update_button, workspace, layer_model: LayerModel, queue):
        """Set up our tab widget with an appropriate graph object in the first
        tab.

        FUTURE, once we are saving our graph configurations, load those instead
        of setting up this default.

        :param auto_update_checkbox: the QCheckBox defined in the pov_main.ui
               file. It's logic - to switch on/off automatic update of graphs
               is managed here

        :param update_button: the QButton defined in the pov_main.ui
               file to trigger manual graph updates in case auto_update_checkbox
               is off.
        """

        super(ProbeGraphManager, self).__init__(tab_widget)

        # hang on to the workspace
        self.workspace = workspace
        self.layer_model = layer_model
        self.queue = queue

        # hang on to the tab widget
        self.tab_widget_object = tab_widget
        self.new_tab_button = QtWidgets.QToolButton()
        self.new_tab_button.setText("+")
        self.tab_widget_object.setCornerWidget(self.new_tab_button, corner=Qt.TopLeftCorner)
        self.tab_widget_object.clear()  # Delete all tabs that may have been created in the Designer
        self.auto_update_checkbox = auto_update_checkbox
        self.update_button = update_button
        # hold on to point probe locations (point probes are shared across tabs)
        self.point_probes: dict = {}

        # set up the first tab
        self.graphs: list = []
        self.selected_graph_index = -1
        self.max_tab_letter = "A"

        # hook things up so we know when the selected tab changes
        self.tab_widget_object.currentChanged[int].connect(self._handle_tab_change)
        self.drawChildGraph.connect(self._draw_child)

        # hook up signals relating to changes in the number of tabs
        self.new_tab_button.clicked.connect(self._add_tab)

        # hook up auto update vs manual update changes
        self.update_button.clicked.connect(self.handleActiveProductDatasetsChanged)
        self.update_button.clicked.connect(self._update_default_point_probe_graph)
        self.auto_update_checkbox.stateChanged.connect(self._on_auto_update_checkbox_state_changed)
        self.auto_update_checkbox.setCheckState(Qt.Unchecked)

    def _draw_child(self, child_name):
        for child in self.graphs:
            if child.myName == child_name:
                child._draw()
                break

    def set_up_tab(self, tab_index, do_increment_tab_letter=True):
        """Create a new tab at tab_index and add it to the list of graphs"""

        # increment our tab label letter if desired
        if do_increment_tab_letter:
            self.max_tab_letter = chr(ord(self.max_tab_letter) + 1)  # this will get strange after Z!

        # create our tab
        temp_widget = QtWidgets.QWidget()
        self.tab_widget_object.insertTab(tab_index, temp_widget, self.max_tab_letter)

        # create the associated graph display object
        graph = ProbeGraphDisplay(self, temp_widget, self.workspace, self.layer_model, self.queue, self.max_tab_letter)
        self.graphs.append(graph)

        # load up the layers for this new tab
        graph.set_possible_layers()

        # clone the previous tab
        if self.selected_graph_index != tab_index:
            # if we aren't setting up the initial tab, clone the current tab
            current_graph = self.graphs[self.selected_graph_index]
            graph.set_default_layer_selections([current_graph.xSelectedUUID, current_graph.ySelectedUUID])
            # give it a copy of the current full_data_selection or polygon
            if current_graph.full_data_selection:
                graph.setRegion(select_full_data=graph.full_data_selection)
            else:
                graph.setRegion(polygon_points=current_graph.polygon[:] if current_graph.polygon is not None else None)
            graph.checked = current_graph.checked
            point_status, point_xy = self.point_probes.get(DEFAULT_POINT_PROBE, (None, None))
            point_xy = point_xy if point_status else None
            graph.setPoint(point_xy, rebuild=False)

        # Create the initial plot
        graph.rebuildPlot()

        # go to the tab we just created
        self.tab_widget_object.setCurrentIndex(tab_index)

    def handleActiveProductDatasetsChanged(self):
        """Used when the layer model signals that something about the layers
        has changed
        """

        # reload the layer list for the existing graphs
        for graphObj in self.graphs:
            doRebuild = graphObj is self.graphs[self.selected_graph_index]
            graphObj.set_possible_layers(do_rebuild_plot=doRebuild)  # FIXME

    def current_graph_set_region(self, polygon_points=None, select_full_data=False):
        """Update the current region in the selected graph and rebuild its plot

        :return: Name of the current probe graph ('A', 'B', ...)

        Probably outdated comment (TODO):
        FUTURE, once the polygon is a layer, this signal will be unnecessary
        """

        return self.graphs[self.selected_graph_index].setRegion(
            polygon_points=polygon_points, select_full_data=select_full_data
        )

    def current_graph_has_polygon(self) -> bool:
        return self.graphs[self.selected_graph_index].polygon is not None

    def update_point_probe(self, probe_name=DEFAULT_POINT_PROBE, xy_pos=None, state=None):
        if xy_pos is None and state is None:
            if probe_name not in self.point_probes:
                # nothing to do
                return
            # they didn't ask to change anything
            # but they may want to refresh stuff
            state, xy_pos = self.point_probes[probe_name]
        elif probe_name not in self.point_probes:
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

    def _update_default_point_probe_graph(self):
        probe_name = DEFAULT_POINT_PROBE
        point_probe = self.point_probes.get(probe_name, [None, None])
        self._update_point_probe_graph(probe_name, *point_probe)

    def _update_point_probe_graph(self, probe_name, state, xy_pos):
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
            return False, None
        return self.point_probes[probe_name]

    def toggle_point_probe(self, probe_name, state=None):
        if probe_name not in self.point_probes:
            LOG.info("No point probe to toggle")
            return

        old_state = self.point_probes[probe_name][0]
        state = state if state is not None else not old_state
        self.update_point_probe(probe_name, state=state)

    def set_default_layer_selections(self, layer_uuids):
        """Set the UUIDs for the current graph if it doesn't have a polygon"""
        return self.graphs[self.selected_graph_index].set_default_layer_selections(layer_uuids)

    def on_region_probe_tool_selected(self):
        if len(self.graphs) > 0:
            return
        # There is no graph tab yet, we must create one
        self.set_up_tab(self.tab_widget_object.count(), do_increment_tab_letter=False)

        current_name = self.graphs[self.selected_graph_index].getName()
        self.didChangeTab.emit((current_name,))

    def _add_tab(self):
        LOG.info("Creating new area probe graph tab.")

        old_name = self.graphs[self.selected_graph_index].getName()
        self.set_up_tab(self.tab_widget_object.count())

        # notify everyone that we cloned a polygon (if we did)
        if self.graphs[self.selected_graph_index].polygon is not None:
            new_name = self.graphs[-1].getName()
            self.didClonePolygon.emit(old_name, new_name)

        current_name = self.graphs[self.selected_graph_index].getName()
        self.didChangeTab.emit((current_name,))

    def _handle_tab_change(self):
        """Deal with the fact that the tab changed in the tab widget"""

        new_tab_index = self.tab_widget_object.currentIndex()

        self.selected_graph_index = new_tab_index
        self.graphs[self.selected_graph_index].rebuildPlot()

        current_name = self.graphs[self.selected_graph_index].getName()
        self.didChangeTab.emit((current_name,))

    def _on_auto_update_checkbox_state_changed(self, state):
        if self.auto_update_checkbox.isChecked():
            self.update_button.setEnabled(False)
            self.layer_model.didFinishActivateProductDatasets.connect(self.handleActiveProductDatasetsChanged)
            self.pointProbeChanged.connect(self._update_point_probe_graph)
        else:
            self.layer_model.didFinishActivateProductDatasets.disconnect(self.handleActiveProductDatasetsChanged)
            self.pointProbeChanged.disconnect(self._update_point_probe_graph)
            self.update_button.setEnabled(True)


class ProbeGraphDisplay(object):
    """The ProbeGraphDisplay controls one tab of the Area Probe Graphs.
    The ProbeGraphDisplay handles generating a displaying a single graph.
    """

    # the most data we are willing to plot in a scatter plot
    # this limit was determined experimentally on Eva's laptop for glance, may need to revisit this
    MAX_SCATTER_PLOT_DATA = 1e7

    # the default number of bins for the histogram and density scatter plot
    DEFAULT_NUM_BINS = 100

    def __init__(self, manager, qt_parent, workspace, layer_model: LayerModel, queue, name_str):
        """build the graph tab controls
        :param layer_model:
        :return:
        """

        # hang on to our name
        self.myName = name_str

        # plotting related controls
        self.toolbar = None
        self.yCheckBox = None
        self.xDropDown = None
        self.yDropDown = None

        # internal objects to reference for info and data
        self.polygon = None
        self.point = None
        self.full_data_selection = False

        # save the workspace and queue for use later
        self.manager = manager
        self.workspace = workspace
        self.layer_model = layer_model
        self.queue = queue

        # internal values that control the behavior of plotting and controls
        self.xSelectedUUID = None
        self.ySelectedUUID = None

        self.xCurrentDatasetUUID = None
        self.yCurrentDatasetUUID = None

        self.uuidMap = None  # this is needed because the drop downs can't properly handle objects as ids
        self._stale = True  # whether or not the plot needs to be redrawn

        # a figure instance to plot on
        self.figure = Figure(figsize=(3, 3), dpi=72)
        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumSize(100, 100)
        # make sure our figure is clear
        self.clearPlot()

        # make a matplotlib toolbar to attach to the graph
        self.toolbar = CustomNavigationToolbar(self.canvas, qt_parent)

        # create our selection controls

        # the label for the x selection
        xLabel = QtWidgets.QLabel("X layer:")

        # the check box that turns on and off comparing to a y layer
        self.yCheckBox = QtWidgets.QCheckBox("vs Y layer:")
        self.yCheckBox.setToolTip("Plot X layer data vs Y layer when this is checked.")
        self.yCheckBox.stateChanged.connect(self.vsChecked)

        # the drop down for selecting the x layer
        self.xDropDown = QtWidgets.QComboBox(qt_parent)
        self.xDropDown.setToolTip("The X layer data to use for plotting.")
        self.xDropDown.activated.connect(self.xSelected)

        # the drop down for selecting the y layer
        self.yDropDown = QtWidgets.QComboBox(qt_parent)
        self.yDropDown.setDisabled(True)
        self.yDropDown.setToolTip("The Y layer data to use for plotting.")
        self.yDropDown.activated.connect(self.ySelected)

        # set the layout
        # Note: add in a grid is (widget, row#, col#) or (widget, row#, col#, row_span, col_span)
        layout = QtWidgets.QGridLayout()
        layout.addWidget(self.toolbar, 1, 1, 1, 3)
        layout.addWidget(self.canvas, 2, 1, 1, 3)
        layout.addWidget(xLabel, 3, 1)
        layout.addWidget(self.xDropDown, 3, 2, 1, 2)
        layout.addWidget(self.yCheckBox, 4, 1)
        layout.addWidget(self.yDropDown, 4, 2, 1, 2)
        qt_parent.setLayout(layout)

    def set_possible_layers(self, do_rebuild_plot=False):
        """Given a list of layer UUIDs, set the names and UUIDs in the drop downs"""

        # make a uuid map because the mapping in a combo box doesn't work with objects
        self.uuidMap = {}

        # clear out the current lists
        self.xDropDown.clear()
        self.yDropDown.clear()

        # fill up our lists of layers
        for layer in self.layer_model.get_probeable_layers():
            uuid_string = str(layer.uuid)
            self.xDropDown.addItem(layer.descriptor, uuid_string)
            self.yDropDown.addItem(layer.descriptor, uuid_string)

            self.uuidMap[uuid_string] = layer.uuid

        # if possible, set the selections back to the way they were
        need_rebuild = False
        x_index = self.xDropDown.findData(str(self.xSelectedUUID))
        if x_index >= 0:
            # Selection didn't change
            self.xDropDown.setCurrentIndex(x_index)
        elif self.xDropDown.count() > 0:
            # Setting to a new layer
            need_rebuild = True
            self.xSelectedUUID = self.uuidMap[self.xDropDown.itemData(0)]
            self.xDropDown.setCurrentIndex(0)
        else:
            # we had something selected but now there is nothing new to select
            need_rebuild = need_rebuild or self.xSelectedUUID is not None
            self.xSelectedUUID = None
        y_index = self.yDropDown.findData(str(self.ySelectedUUID))
        if y_index >= 0:
            # Selection didn't change
            self.yDropDown.setCurrentIndex(y_index)
        elif self.yDropDown.count() > 0:
            # Setting to a new layer
            need_rebuild = need_rebuild or self.yCheckBox.isChecked()
            self.ySelectedUUID = self.uuidMap[self.yDropDown.itemData(0)]
            self.yDropDown.setCurrentIndex(0)
        else:
            # we had something selected but now there is nothing new to select
            need_rebuild = need_rebuild or self.ySelectedUUID is not None
            self.ySelectedUUID = None

        need_rebuild |= self._check_active_datasets_changed()

        # refresh the plot
        self._stale = need_rebuild
        if do_rebuild_plot:
            # Rebuild the plot (stale is used to determine if actual rebuild
            # is needed)
            self.rebuildPlot()

    def _check_active_datasets_changed(self):
        # check whether active datasets have changed. If so, update stored
        # dataset uuids and indicate that the graph needs to be rebuilt.
        need_rebuild = False
        x_layer = self.layer_model.get_layer_by_uuid(self.xSelectedUUID)
        x_active_product_dataset = None if not x_layer else x_layer.get_first_active_product_dataset()
        if not x_active_product_dataset:
            if self.xCurrentDatasetUUID is not None:
                need_rebuild = True
                self.xCurrentDatasetUUID = None
        elif x_active_product_dataset.uuid != self.xCurrentDatasetUUID:
            need_rebuild = True
            self.xCurrentDatasetUUID = x_active_product_dataset.uuid
        y_layer = self.layer_model.get_layer_by_uuid(self.ySelectedUUID)
        y_active_product_dataset = None if not y_layer else y_layer.get_first_active_product_dataset()
        if not y_active_product_dataset:
            if self.yCurrentDatasetUUID is not None:
                need_rebuild |= self.yCheckBox.isChecked()
                self.yCurrentDatasetUUID = None
        elif y_active_product_dataset.uuid != self.yCurrentDatasetUUID:
            need_rebuild |= self.yCheckBox.isChecked()
            self.yCurrentDatasetUUID = y_active_product_dataset.uuid
        return need_rebuild

    def set_default_layer_selections(self, layer_uuids):
        # only set the defaults if we don't have a polygon yet
        if self.polygon is not None:
            return

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

    def xSelected(self):
        """The user selected something in the X layer list."""
        oldXStr = str(self.xSelectedUUID)
        newXStr = self.xDropDown.itemData(self.xDropDown.currentIndex())
        self.xSelectedUUID = self.uuidMap[newXStr]

        # regenerate the plot
        if oldXStr != newXStr:
            self._stale = True
            self.rebuildPlot()

    def ySelected(self):
        """The user selected something in the Y layer list."""
        oldYStr = str(self.ySelectedUUID)
        newYStr = self.yDropDown.itemData(self.yDropDown.currentIndex())
        self.ySelectedUUID = self.uuidMap[newYStr]

        # regenerate the plot
        if (oldYStr != newYStr) and self.yCheckBox.isChecked():
            self._stale = True
            self.rebuildPlot()

    def vsChecked(self):
        """The vs check box was checked!"""
        # look at the state of the vs box and enable/disable the y drop down accordingly
        doPlotVS = self.yCheckBox.isChecked()
        self.yDropDown.setDisabled(not doPlotVS)

        # regenerate the plot
        self._stale = True
        self.rebuildPlot()

    def setRegion(self, polygon_points=None, select_full_data=False):
        """Set the region for this graph as polygon selection or full data."""

        assert polygon_points is None or not select_full_data, (  # nosec B101
            "Must not give both 'polygon_points' and True for 'select_full_data':"
            " Defining region by polygon and as full data are mutually exclusive."
        )

        # Even with assertions switched off we will get a valid state here: a
        # polygonal region wins over the full data selection: the first one will
        # have a visual echo, the second one not, so this is more likely to give
        # a consistent state.
        self.polygon = polygon_points
        self.full_data_selection = False if self.polygon is not None else select_full_data

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

    def getName(self):
        """Accessor method for the graph's name."""
        return self.myName

    def rebuildPlot(self):
        """Given what we currently know about the selection area and selected bands, rebuild our plot

        Note: This should be called only when the selections change in some way.
        """
        if not self._stale:
            LOG.debug("Plot doesn't need to be rebuilt")
            return

        # should be plotting vs Y?
        doPlotVS = self.yCheckBox.isChecked()
        task_name = "%s_%s_region_plotting" % (self.xSelectedUUID, self.ySelectedUUID)
        task_description = (
            "Creating plot for full data" if self.full_data_selection else "Creating plot for region probe data"
        )
        self.queue.add(
            task_name,
            self._rebuild_plot_task(
                self.xSelectedUUID,
                self.ySelectedUUID,
                self.polygon,
                self.point,
                plot_versus=doPlotVS,
                plot_full_data=self.full_data_selection,
            ),
            task_description,
            interactive=True,
        )
        # Assume that the task gets resolved otherwise we might try to draw multiple times
        self._stale = False

    def _rebuild_plot_task(  # noqa: C901
        self, x_layer_uuid, y_layer_uuid, polygon, point_xy, plot_versus=False, plot_full_data=True
    ):
        data_source_description = "full data" if plot_full_data else "polygon data"

        x_layer = self.layer_model.get_layer_by_uuid(x_layer_uuid)
        x_active_product_dataset = None if not x_layer else x_layer.get_first_active_product_dataset()
        x_uuid = None if not x_active_product_dataset else x_active_product_dataset.uuid

        y_layer = self.layer_model.get_layer_by_uuid(y_layer_uuid)
        y_active_product_dataset = None if not y_layer else y_layer.get_first_active_product_dataset()
        y_uuid = None if not y_active_product_dataset else y_active_product_dataset.uuid

        # if we are plotting only x and we have a selected x and a polygon
        have_x_layer = x_layer_uuid is not None
        have_y_layer = y_layer_uuid is not None
        should_plot = polygon is not None or plot_full_data
        if not plot_versus and have_x_layer and should_plot:
            yield {TASK_DOING: f"Probe Plot: Collecting {data_source_description}...", TASK_PROGRESS: 0.0}

            # get the data and info we need for this plot
            if x_active_product_dataset:
                if plot_full_data:
                    data_polygon = self.workspace.get_content(x_active_product_dataset.uuid)
                else:
                    data_polygon = self.workspace.get_content_polygon(x_active_product_dataset.uuid, polygon)
            else:
                data_polygon = np.array([])

            x_conv_func = x_layer.info[Info.UNIT_CONVERSION][1]
            data_polygon = x_conv_func(data_polygon)
            time = x_active_product_dataset.info[Info.DISPLAY_TIME]
            title = f"{time}"
            x_axis_label = x_layer.descriptor
            y_axis_label = "Count of data points"

            # get point probe value
            if x_active_product_dataset and point_xy:
                x_point = self.workspace.get_content_point(x_active_product_dataset.uuid, point_xy)
                x_point = x_conv_func(x_point)
            else:
                x_point = None

            # plot a histogram
            yield {TASK_DOING: "Probe Plot: Creating histogram plot", TASK_PROGRESS: 0.25}
            self.plotHistogram(data_polygon, title, x_point, x_axis_label, y_axis_label)

        # if we are plotting x vs y and have x, y, and a polygon
        elif plot_versus and have_x_layer and have_y_layer and should_plot:
            yield {TASK_DOING: f"Probe Plot: Collecting {data_source_description} (layer 1)...", TASK_PROGRESS: 0.0}

            name1 = x_layer.descriptor
            name2 = y_layer.descriptor
            if not x_active_product_dataset or not y_active_product_dataset:
                x_point = None
                y_point = None
                time1 = None
                time2 = None
                data1 = np.array([0])
                data2 = np.array([0])
            else:
                # get the data and info we need for this plot
                x_info = x_active_product_dataset.info
                y_info = y_active_product_dataset.info
                time1 = x_info[Info.DISPLAY_TIME]
                time2 = y_info[Info.DISPLAY_TIME]
                hires_uuid = self.workspace.lowest_resolution_uuid(x_uuid, y_uuid)
                # hires_coord_mask are the lat/lon coordinates of each of the
                # pixels in hires_data. The coordinates are (lat, lon) to resemble
                # the (Y, X) indexing of numpy arrays
                if plot_full_data:
                    hires_coord_mask = None
                    hires_data = self.workspace.get_content(hires_uuid)
                else:
                    hires_coord_mask, hires_data = self.workspace.get_coordinate_mask_polygon(hires_uuid, polygon)

                x_conv_func = x_layer.info[Info.UNIT_CONVERSION][1]
                y_conv_func = y_layer.info[Info.UNIT_CONVERSION][1]
                yield {
                    TASK_DOING: f"Probe Plot: Collecting {data_source_description} (layer 2)...",
                    TASK_PROGRESS: 0.15,
                }
                if hires_uuid == x_uuid:
                    # the hires data was from the X UUID
                    data1 = x_conv_func(hires_data)
                    if plot_full_data:
                        data2 = self.workspace.get_content(y_uuid)
                    else:
                        data2 = self.workspace.get_content_coordinate_mask(y_uuid, hires_coord_mask)
                    data2 = y_conv_func(data2)
                else:
                    # the hires data was from the Y UUID
                    data2 = y_conv_func(hires_data)
                    if plot_full_data:
                        data1 = self.workspace.get_content(x_uuid)
                    else:
                        data1 = self.workspace.get_content_coordinate_mask(x_uuid, hires_coord_mask)
                    data1 = x_conv_func(data1)
                yield {TASK_DOING: "Probe Plot: Creating scatter plot...", TASK_PROGRESS: 0.25}

                if point_xy:
                    x_point = self.workspace.get_content_point(x_uuid, point_xy)
                    x_point = x_conv_func(x_point)
                    y_point = self.workspace.get_content_point(y_uuid, point_xy)
                    y_point = y_conv_func(y_point)
                else:
                    x_point = None
                    y_point = None

                # plot a scatter plot
                good_mask = ~(np.isnan(data1) | np.isnan(data2))
                data1 = data1[good_mask]
                data2 = data2[good_mask]

            self.plotDensityScatterplot(data1, name1, time1, data2, name2, time2, x_point, y_point)

        # if we have some combination of selections we don't understand, clear the figure
        else:
            yield {TASK_DOING: "Probe Plot: Clearing plot figure...", TASK_PROGRESS: 0.0}
            self.clearPlot()

        yield {TASK_DOING: "Probe Plot: Drawing plot...", TASK_PROGRESS: 0.95}
        self.manager.drawChildGraph.emit(self.myName)
        yield {TASK_DOING: "Probe Plot: Done", TASK_PROGRESS: 1.0}

    def _draw(self):
        self.canvas.draw()

    def plotHistogram(self, data, title, x_point, x_label, y_label, numBins=100):
        """Make a histogram using the given data and label it with the given title"""
        self.figure.clf()
        axes = self.figure.add_subplot(111)
        bars = axes.hist(data[~np.isnan(data)], bins=self.DEFAULT_NUM_BINS)
        if x_point is not None:
            # go through each rectangle object and make the one that contains x_point 'red'
            # default color is blue so red should stand out
            for bar in bars[2][::-1]:
                if bar.xy[0] <= x_point:
                    bar.set_color("red")
                    break
        axes.set_title(title)
        axes.set_xlabel(x_label)
        axes.set_ylabel(y_label)

    def plotDensityScatterplot(self, dataX, nameX, timeX, dataY, nameY, timeY, pointX, pointY):
        """Make a density scatter plot for the given data
        :param timeX:
        :param timeY:
        """

        # clear the figure and make a new subplot
        self.figure.clf()
        axes = self.figure.add_subplot(111)

        # figure out the range of the data
        # you might not be comparing the same units
        xmin_value = np.min(dataX)
        xmax_value = np.max(dataX)
        ymin_value = np.min(dataY)
        ymax_value = np.max(dataY)
        # bounds should be defined in the form [[xmin, xmax], [ymin, ymax]]
        bounds = [[xmin_value, xmax_value], [ymin_value, ymax_value]]

        # make the binned density map for this data set
        density_map, _, _ = np.histogram2d(dataX, dataY, bins=self.DEFAULT_NUM_BINS, range=bounds)
        # mask out zero counts; flip because y goes the opposite direction in an imshow graph
        density_map = np.flipud(np.transpose(np.ma.masked_array(density_map, mask=density_map == 0)))

        # display the density map data
        img = axes.imshow(
            density_map,
            extent=[xmin_value, xmax_value, ymin_value, ymax_value],
            aspect="auto",
            interpolation="nearest",
            norm=LogNorm(),
        )
        if pointX is not None:
            axes.set_autoscale_on(False)
            axes.plot(
                pointX,
                pointY,
                marker="o",
                markerfacecolor="white",
                markeredgecolor="black",
                markersize=10,
                markeredgewidth=1.0,
            )
            axes.set_autoscale_on(True)

        colorbar = self.figure.colorbar(img)
        colorbar.set_label("log(count of data points)")

        # set the various text labels
        axes.set_xlabel(f"{nameX}")
        axes.set_ylabel(f"{nameY}")
        axes.set_title(timeX)

        # draw the x vs y line
        self._draw_xy_line(axes)

    def clearPlot(self):
        """Clear our plot"""
        self.full_data_selection = False
        self.figure.clf()

    def _draw_xy_line(self, axes):
        # get the bounds for our calculations and so we can reset the viewing window later
        x_bounds = axes.get_xbound()
        y_bounds = axes.get_ybound()

        # draw the x=y line
        perfect = [max(x_bounds[0], y_bounds[0]), min(x_bounds[1], y_bounds[1])]
        axes.plot(perfect, perfect, "--", color="k", label="X = Y")

        # reset the bounds
        axes.set_xbound(x_bounds)
        axes.set_ybound(y_bounds)
