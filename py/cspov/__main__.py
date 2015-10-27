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


from vispy import app
try:
    app_object = app.use_app('pyqt4')
except Exception:
    app_object = app.use_app('pyside')
QtCore = app_object.backend_module.QtCore
QtGui = app_object.backend_module.QtGui
from PyQt4.QtCore import QObject, pyqtSignal

from cspov.control.layer_list import LayerStackListViewModel
from cspov.model import Document
from cspov.view.SceneGraphManager import SceneGraphManager
from cspov.queue import TaskQueue, test_task, TASK_PROGRESS, TASK_DOING
from cspov.workspace import Workspace
from cspov.view.Colormap import ALL_COLORMAPS

from functools import partial

# this is generated with pyuic4 pov_main.ui >pov_main_ui.py
from cspov.ui.pov_main_ui import Ui_MainWindow
import cspov.view.Colormap
from cspov.common import INFO, KIND

import os
import logging
import numpy

# http://stackoverflow.com/questions/12459811/how-to-embed-matplotib-in-pyqt-for-dummies
# see also: http://matplotlib.org/users/navigation_toolbar.html
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.colors import LogNorm

LOG = logging.getLogger(__name__)
PROGRESS_BAR_MAX = 1000


def test_layers_from_directory(ws, doc, layer_tiff_glob, range_txt=None):
    """
    TIFF_GLOB='/Users/keoni/Data/CSPOV/2015_07_14_195/00?0/HS*_B03_*merc.tif' VERBOSITY=3 python -m cspov
    :param model:
    :param view:
    :param layer_tiff_glob:
    :return:
    """
    from glob import glob
    range = None
    if range_txt:
        import re
        range = tuple(map(float, re.findall(r'[\.0-9]+', range_txt)))
    for tif in glob(layer_tiff_glob):
        # doc.addFullGlobMercatorColormappedFloatImageLayer(tif, range=range)
        # uuid, info, overview_data = ws.import_image(tif)
        uuid, info, overview_data = doc.open_file(tif)
        LOG.info('loaded uuid {} from {}'.format(uuid, tif))
        yield uuid, info, overview_data


def test_layers(ws, doc, glob_pattern=None):
    if glob_pattern:
        return test_layers_from_directory(ws, doc, glob_pattern, os.environ.get('RANGE', None))
    LOG.warning("No image glob pattern provided")
    return []

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
        tempCount = self.tab_widget_object.count()
        if tempCount != 1 :
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
        xIndex = self.xDropDown.findData(str(self.xSelectedUUID))
        if xIndex >= 0 :
            self.xDropDown.setCurrentIndex(xIndex)
        elif self.xDropDown.count() > 0 :
            self.xSelectedUUID = self.uuidMap[self.xDropDown.itemData(0)]
            self.xDropDown.setCurrentIndex(0)
        else :
            self.xSelectedUUID = None
        yIndex = self.yDropDown.findData(str(self.ySelectedUUID))
        if yIndex >= 0 :
            self.yDropDown.setCurrentIndex(yIndex)
        elif self.yDropDown.count() > 0 :
            self.ySelectedUUID = self.uuidMap[self.yDropDown.itemData(0)]
            self.yDropDown.setCurrentIndex(0)
        else :
            self.ySelectedUUID = None

        # refresh the plot
        if do_rebuild_plot :
            self.rebuildPlot()

    def xSelected (self) :
        """The user selected something in the X layer list.
        """

        oldXStr = str(self.xSelectedUUID)
        newXStr = self.xDropDown.itemData(self.xDropDown.currentIndex())
        self.xSelectedUUID = self.uuidMap[newXStr]

        # regenerate the plot
        if oldXStr != newXStr :
            self.rebuildPlot()

    def ySelected (self) :
        """The user selected something in the Y layer list.
        """

        oldYStr = str(self.ySelectedUUID)
        newYStr = self.yDropDown.itemData(self.yDropDown.currentIndex())
        self.ySelectedUUID = self.uuidMap[newYStr]

        # regenerate the plot
        if (oldYStr != newYStr) and self.yCheckBox.isChecked() :
            self.rebuildPlot()

    def vsChecked (self) :
        """The vs check box was checked!
        """

        # look at the state of the vs box and enable/disable the y drop down accordingly
        doPlotVS = self.yCheckBox.isChecked()
        self.yDropDown.setDisabled(not doPlotVS)

        # regenerate the plot
        self.rebuildPlot()

    def setPolygon (self, polygonPoints) :
        """Set the polygon selection for this graph
        """

        self.polygon = polygonPoints

        # regenerate the plot
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


class AnimationSpeedPopupWindow(QtGui.QWidget):
    _slider = None
    _active = False

    def __init__(self, slot, *args, **kwargs):
        super(AnimationSpeedPopupWindow, self).__init__(*args, **kwargs)
        from PyQt4.QtCore import Qt
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Popup)
        self.setFocusPolicy(Qt.ClickFocus)
        self.setToolTip('Set animation speed')
        self._slider = QtGui.QSlider(parent=self)
        # n, x = self._convert(10, reverse=True), self._convert(5000, reverse=True)
        n, x = 2, 150  # frames per 10 seconds
        self._slider.setRange(n, x) #
        # self._slider.setSingleStep(1)
        # self._slider.setInvertedAppearance(True)
        self._slot = slot
        self._slider.valueChanged.connect(self._changed)
        self._layout = QtGui.QHBoxLayout()
        self._layout.addWidget(self._slider)
        self.setLayout(self._layout)

    def _convert(self, val, reverse=False):
        """
        map 1..100 nonlinearly to 10ms .. 5000ms
        :param val: ticks to convert to milliseconds
        :param reverse: when true, reverse conversion
        :return:
        """
        if reverse: # convert milliseconds to fp10s
            fp10s = 10000.0 / float(val)
            return fp10s
        else:
            ms = 10000.0 / float(val)
            return ms

    def _changed(self, value):
        if not self._active:
            return
        fps = float(value) / 10.0
        self.setToolTip('{0:.1f} fps'.format(fps))
        val = self._convert(value)
        self._slot(val)

    def show_at(self, pos, val):
        from PyQt4.QtCore import QRect, QPoint, QSize
        sz = QSize(40, 180)
        pt = QPoint(pos.x() - 20, pos.y() - 160)
        rect = QRect(pt, sz)
        self.setGeometry(rect)
        self.show()
        self._slider.setValue(int(self._convert(val, reverse=True)))
        self._active = True

    def focusOutEvent(self, *args, **kwargs):
        self.hide()
        self._active = False


class Main(QtGui.QMainWindow):
    _last_open_dir = None  # directory to open files in
    _animation_speed_popup = None  # window we'll show temporarily with animation speed popup

    def open_files(self):
        files = QtGui.QFileDialog.getOpenFileNames(self,
                                                   "Select one or more files to open",
                                                   self._last_open_dir or os.getenv("HOME"),
                                                   'Mercator GeoTIFF (*.tiff *.tif)')
        for pathname in files:
            self.document.open_file(pathname)
            self._last_open_dir = os.path.split(pathname)[0]

    def dropEvent(self, event):
        LOG.debug('drop event on mainwindow')
        mime = event.mimeData()
        if mime.hasUrls:
            event.setDropAction(QtCore.Qt.CopyAction)
            event.accept()
            for url in mime.urls():
                path = str(url.toLocalFile())
                LOG.info('about to open {}'.format(path))
                self.document.open_file(path)
        else:
            event.ignore()

    def change_tool(self, name="pz_camera"):
        buttons = [self.ui.panZoomToolButton, self.ui.pointSelectButton, self.ui.regionSelectButton]
        names = [self.scene_manager.pz_camera.name, self.scene_manager.point_probe_camera.name, self.scene_manager.polygon_probe_camera.name]
        names = dict((name,value) for (value,name) in enumerate(names))
        dex = names[name]
        for q,b in enumerate(buttons):
            b.setDown(dex==q)
        self.scene_manager.change_camera(dex)

    def update_progress_bar(self, status_info, *args, **kwargs):
        active = status_info[0]
        LOG.debug('{0!r:s}'.format(status_info))
        # val = active[TASK_PROGRESS]
        txt = active[TASK_DOING]
        val = self.queue.progress_ratio()
        self.ui.progressBar.setValue(int(val*PROGRESS_BAR_MAX))
        self.ui.progressText.setText(txt)
        #LOG.warning('progress bar updated to {}'.format(val))

    def update_frame_slider(self, frame_info):
        """
        animation is in progress or completed
        update the animation slider and label to show what's going on
        :param frame_info: tuple, ultimately from scenegraphmanager.layer_set callback into sgm
        :return:
        """
        frame_index, frame_count, animating, uuid = frame_info[:4]
        self.ui.animationSlider.setRange(0, frame_count-1)
        self.ui.animationSlider.setValue(frame_index or 0)
        # LOG.debug('did update animation slider {} {}'.format(frame_index, frame_count))
        self.ui.animPlayPause.setDown(animating)
        self.ui.animationSlider.repaint()
        self.ui.animationLabel.setText(self.document.time_label_for_uuid(uuid))

    def update_frame_time_to_top_visible(self):
        # FUTURE: don't address layer set directly
        self.ui.animationLabel.setText(self.document.time_label_for_uuid(self.scene_manager.layer_set.top_layer_uuid()))

    def remove_layer(self, *args, **kwargs):
        uuids = self.behaviorLayersList.current_selected_uuids()
        for uuid in uuids:
            LOG.debug('removing layer {}'.format(uuid))
            self.document.remove_layer_prez(uuid)

    def animation_slider_jump_frame(self, event, *args, **kwargs):
        "user has moved frame slider, update the display"
        frame = self.ui.animationSlider.value()
        self.scene_manager.set_frame_number(frame)
        # TODO: update layer list to reflect what layers are visible/hidden?

    def _next_last_time_visibility(self, direction=0, *args, **kwargs):
        LOG.info('time incr {}'.format(direction))
        # TODO: if this frame is part of the animation sequence, update the slider as well!
        uuids = self.behaviorLayersList.current_selected_uuids()
        if not uuids:
            self.ui.cursorProbeText.setText('No layer selected?')
            pass # FIXME: notify user
        new_focus = None
        for uuid in uuids:
            new_focus = self.document.next_last_step(uuid, direction, bandwise=False)
        return new_focus

    def update_slider_if_frame_is_in_animation(self, uuid):
        # FUTURE: this could be a cheaper operation but it's probably fine since it's input-driven
        cao = self.document.current_animation_order
        try:
            dex = cao.index(uuid)
        except ValueError as not_present:
            return
        frame_change_tuple = (dex, len(cao), False, uuid)
        self.update_frame_slider(frame_change_tuple)

    def next_last_time(self, direction=0, *args, **kwargs):
        self.scene_manager.layer_set.animating = False
        new_focus = self._next_last_time_visibility(direction=direction)
        self.behaviorLayersList.select([new_focus])
        # if this part of the animation cycle, update the animation slider and displayed time as well
        self.update_slider_if_frame_is_in_animation(new_focus)
        # FIXME: force animation off
        return new_focus
        # self.document.animate_siblings_of_layer(new_focus)
        self.update_frame_time_to_top_visible()

    def next_last_band(self, direction=0, *args, **kwargs):
        LOG.info('band incr {}'.format(direction))
        uuids = self.behaviorLayersList.current_selected_uuids()
        new_focus = None
        if not uuids:
            pass # FIXME: notify user
        for uuid in uuids:
            new_focus = self.document.next_last_step(uuid, direction, bandwise=True)
        if new_focus is not None:
            self.behaviorLayersList.select([new_focus])
            self.update_frame_time_to_top_visible()

    def change_animation_to_current_selection_siblings(self, *args, **kwargs):
        uuid = self._next_last_time_visibility(direction=0)
        # calculate the new animation sequence by consulting the guidebook
        uuids = self.document.animate_siblings_of_layer(uuid)
        if uuids:
            self.ui.cursorProbeText.setText("Frame order updated")
            self.behaviorLayersList.select(uuids)
        else:
            self.ui.cursorProbeText.setText("Layer with time steps needed")
        LOG.info('using siblings of {} for animation loop'.format(uuids[0]))

    def set_animation_speed(self, milliseconds):
        LOG.info('animation speed set to {}ms'.format(milliseconds))
        # FUTURE: propagate this into the document?
        self.scene_manager.layer_set.animation_speed = milliseconds

    def show_animation_speed_slider(self, pos:QtCore.QPoint, *args):
        LOG.info('menu requested for animation control')
        gpos = self.ui.animPlayPause.mapToGlobal(pos)

        if self._animation_speed_popup is None:
            self._animation_speed_popup = popup = AnimationSpeedPopupWindow(slot=self.set_animation_speed, parent=None)
        else:
            popup = self._animation_speed_popup
        if not popup.isVisible():
            popup.show_at(gpos, self.scene_manager.layer_set.animation_speed)

    def toggle_visibility_on_selected_layers(self, *args, **kwargs):
        uuids = self.behaviorLayersList.current_selected_uuids()
        self.document.toggle_layer_visibility(uuids)
        self.update_frame_time_to_top_visible()

    def toggle_animation(self, event, *args, **kwargs):
        self.scene_manager.layer_set.toggle_animation(*args, **kwargs)

    # def accept_new_layer(self, new_order, info, overview_content):
    #     LOG.debug('accepting new layer order {0!r:s}'.format(new_order))
    #     if info[INFO.KIND] == KIND.IMAGE:
    #         LOG.info("rebuilding animation based on newly loaded image layer")
    #         self.document.animate_using_layer(info[INFO.UUID])
    #         self.animation_slider_jump_frame(None)
    #         self.behaviorLayersList.select([info[INFO.UUID]])

    def __init__(self, workspace_dir=None, workspace_size=None, glob_pattern=None, border_shapefile=None, center=None):
        super(Main, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        # refer to objectName'd entities as self.ui.objectName
        self.setAcceptDrops(True)

        self.queue = TaskQueue()
        self.ui.progressBar.setRange(0, PROGRESS_BAR_MAX)
        self.queue.didMakeProgress.connect(self.update_progress_bar)

        # create document
        self.workspace = Workspace(workspace_dir, max_size_gb=workspace_size)
        self.document = doc = Document(self.workspace)
        self.scene_manager = SceneGraphManager(doc, self.workspace, self.queue,
                                               glob_pattern=glob_pattern,
                                               border_shapefile=border_shapefile,
                                               center=center,
                                               parent=self)
        self.ui.mainWidgets.addTab(self.scene_manager.main_canvas.native, 'Mercator')

        self.scene_manager.didChangeFrame.connect(self.update_frame_slider)
        self.ui.animPlayPause.clicked.connect(self.toggle_animation)
        self.ui.animPlayPause.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.ui.animPlayPause.customContextMenuRequested.connect(self.show_animation_speed_slider)
        # def next_frame(*args, **kwargs):
        #     self.scene_manager.animating = False
        #     self.scene_manager.layer_set.next_frame()
        # self.ui.animForward.clicked.connect(next_frame)
        # def prev_frame(*args, **kwargs):
        #     self.scene_manager.animating = False
        #     self.scene_manager.layer_set.next_frame(frame_number=-1)
        # self.ui.animBack.clicked.connect(prev_frame)

        # allow animation slider to set animation frame being displayed:
        self.ui.animationSlider.valueChanged.connect(self.animation_slider_jump_frame)

        # allow animation, once stopped, to propagate visibility to the document and layerlist:
        self.scene_manager.didChangeLayerVisibility.connect(self.document.animation_changed_visibility)

        # disable close button on panes
        for pane in [self.ui.areaProbePane, self.ui.layersPane]:
            pane.setFeatures(QtGui.QDockWidget.DockWidgetFloatable |
                             QtGui.QDockWidget.DockWidgetMovable)

        for uuid, ds_info, full_data in test_layers(self.workspace, self.document, glob_pattern=glob_pattern):
            # this now fires off a document modification cascade resulting in a new layer going up
            pass

        # Interaction Setup
        self.setup_key_releases()
        self.scheduler = QtCore.QTimer(parent=self)
        self.scheduler.setInterval(200.0)
        self.scheduler.timeout.connect(partial(self.scene_manager.on_view_change, self.scheduler))
        def start_wrapper(timer, event):
            """Simple wrapper around a timers start method so we can accept but ignore the event provided
            """
            timer.start()
        self.scene_manager.main_canvas.transforms.changed.connect(partial(start_wrapper, self.scheduler))

        # convey action between document and layer list view
        self.behaviorLayersList = LayerStackListViewModel([self.ui.layerSet1Table, self.ui.layerSet2Table, self.ui.layerSet3Table, self.ui.layerSet4Table], doc)

        # coordinate what gets done when a layer is added by document
        self.document.didAddLayer.connect(self.update_frame_time_to_top_visible)

        def update_probe_point(uuid, xy_pos):
            data_point = self.workspace.get_content_point(uuid, xy_pos)
            self.ui.cursorProbeText.setText("Point Probe: {:.03f}".format(float(data_point)))
        self.scene_manager.newProbePoint.connect(update_probe_point)

        def update_probe_polygon(uuid, points, layerlist=self.behaviorLayersList):

            selected_uuids = list(layerlist.current_selected_uuids())
            LOG.debug("selected UUID set is {0!r:s}".format(selected_uuids))

            # if the layer list doesn't have any selected UUIDs, use the one passed in
            if len(selected_uuids) <= 0:
                selected_uuids = [uuid]

            # if we have more than two uuids, just plot the very first one
            elif len(selected_uuids) > 2 :
                selected_uuids = selected_uuids[0:1]

            # now we must have 1 or 2 UUIDs in our list

            # TODO, when the plots manage their own layer selection, change this call
            # FUTURE, once the polygon is a layer, this will need to change
            # update our current plot with the new polygon
            polygon_name = self.graphManager.currentPolygonChanged (polygonPoints=points)

            # do whatever other updates the scene manager needs
            self.scene_manager.on_new_polygon(polygon_name, points)

        self.scene_manager.newProbePolygon.connect(update_probe_polygon)

        self.ui.mainWidgets.removeTab(0)
        self.ui.mainWidgets.removeTab(0)

        # self.queue.add('test', test_task(), 'test000')
        # self.ui.layers
        print(self.scene_manager.main_view.describe_tree(with_transform=True))
        self.document.didChangeColormap.connect(self.scene_manager.change_layers_colormap)
        self.document.didChangeColorLimits.connect(self.scene_manager.change_layers_color_limits)

        self.ui.panZoomToolButton.clicked.connect(partial(self.change_tool, name=self.scene_manager.pz_camera.name))
        self.ui.pointSelectButton.clicked.connect(partial(self.change_tool, name=self.scene_manager.point_probe_camera.name))
        self.ui.regionSelectButton.clicked.connect(partial(self.change_tool, name=self.scene_manager.polygon_probe_camera.name))
        self.change_tool()

        self.setup_menu()
        self.graphManager = ProbeGraphManager(self.ui.probeTabWidget, self.workspace, self.document)
        self.graphManager.didChangeTab.connect(self.scene_manager.show_only_polygons)

    def closeEvent(self, event, *args, **kwargs):
        LOG.debug('main window closing')
        self.workspace.close()

    def toggle_animation(self, action:QtGui.QAction=None, *args):
        new_state = self.scene_manager.layer_set.toggle_animation()
        self.ui.animPlayPause.setChecked(new_state)

    def setup_menu(self):
        open_action = QtGui.QAction("&Open", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_files)

        exit_action = QtGui.QAction("&Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(QtGui.qApp.quit)

        menubar = self.ui.menubar
        file_menu = menubar.addMenu('&File')
        file_menu.addAction(open_action)
        file_menu.addAction(exit_action)

        next_time = QtGui.QAction("Next Time", self)
        next_time.setShortcut(QtCore.Qt.Key_Right)
        next_slot = partial(self.next_last_time, direction=1)
        next_time.triggered.connect(next_slot)
        self.ui.animForward.clicked.connect(next_slot)

        focus_current = QtGui.QAction("Focus Current Timestep", self)
        focus_current.setShortcut('.')
        focus_current.triggered.connect(partial(self.next_last_band, direction=0))

        prev_time = QtGui.QAction("Previous Time", self)
        prev_time.setShortcut(QtCore.Qt.Key_Left)
        prev_slot = partial(self.next_last_time, direction=-1)
        prev_time.triggered.connect(prev_slot)
        self.ui.animBack.clicked.connect(prev_slot)

        next_band = QtGui.QAction("Next Band", self)
        next_band.setShortcut(QtCore.Qt.Key_Up)
        next_band.triggered.connect(partial(self.next_last_band, direction=1))

        prev_band = QtGui.QAction("Previous Band", self)
        prev_band.setShortcut(QtCore.Qt.Key_Down)
        prev_band.triggered.connect(partial(self.next_last_band, direction=-1))

        toggle_vis = QtGui.QAction("Toggle &Visibility", self)
        toggle_vis.setShortcut('V')
        toggle_vis.triggered.connect(self.toggle_visibility_on_selected_layers)


        animate = QtGui.QAction("Animate", self)
        animate.setShortcut('A')
        animate.triggered.connect(partial(self.toggle_animation, action=animate))

        change_order = QtGui.QAction("Set Animation &Order", self)
        change_order.setShortcut('O')
        change_order.triggered.connect(self.change_animation_to_current_selection_siblings)

        flip_colormap = QtGui.QAction("Flip Color Limits (Top Layer)", self)
        flip_colormap.setShortcut("/")
        flip_colormap.triggered.connect(lambda: self.document.flip_climits_for_layers([self.document.current_visible_layer]))

        remove = QtGui.QAction("Remove Layer", self)
        remove.setShortcut(QtCore.Qt.Key_Delete)
        remove.triggered.connect(self.remove_layer)

        cycle_borders = QtGui.QAction("Cycle &Borders", self)
        cycle_borders.setShortcut('B')
        cycle_borders.triggered.connect(self.scene_manager.cycle_borders_color)

        cycle_grid = QtGui.QAction("Cycle &Lat/Lon Grid", self)
        cycle_grid.setShortcut('L')
        cycle_grid.triggered.connect(self.scene_manager.cycle_grid_color)

        edit_menu = menubar.addMenu('&Edit')
        edit_menu.addAction(remove)

        view_menu = menubar.addMenu('&View')
        view_menu.addAction(animate)
        view_menu.addAction(prev_time)
        view_menu.addAction(focus_current)
        view_menu.addAction(next_time)
        view_menu.addAction(prev_band)
        view_menu.addAction(next_band)
        view_menu.addAction(change_order)
        view_menu.addAction(toggle_vis)
        view_menu.addAction(flip_colormap)
        view_menu.addAction(cycle_borders)
        view_menu.addAction(cycle_grid)

        menubar.setEnabled(True)

    def setup_key_releases(self):
        def cb_factory(required_key, cb):
            def tmp_cb(key, cb=cb):
                if key.text == required_key:
                    return cb()
            return tmp_cb

        self.scene_manager.main_canvas.events.key_release.connect(cb_factory("c", self.scene_manager.next_camera))

        class ColormapSlot(object):
            def __init__(self, sgm, key='e'):
                self.index = 0
                self.key = key
                self.sgm = sgm
                self.colormaps = ALL_COLORMAPS.keys()

            def __call__(self, key):
                if key.text == self.key:
                    self.sgm.set_colormap(self.colormaps[self.index])
                    self.index = (self.index + 1) % len(self.colormaps)

        self.scene_manager.main_canvas.events.key_release.connect(ColormapSlot(self.scene_manager))

    def updateLayerList(self):
        # self.ui.layers.add
        pass


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run CSPOV")
    parser.add_argument("-w", "--workspace", default='.',
                        help="Specify workspace base directory")
    parser.add_argument("-s", "--space", default=256, type=int,
                        help="Specify max amount of data to hold in workspace in Gigabytes")
    parser.add_argument("--border-shapefile", default=None,
                        help="Specify alternative coastline/border shapefile")
    parser.add_argument("--glob-pattern", default=os.environ.get("TIFF_GLOB", None),
                        help="Specify glob pattern for input images")
    parser.add_argument("-c", "--center", nargs=2, type=float,
                        help="Specify center longitude and latitude for camera")
    parser.add_argument('-v', '--verbose', dest='verbosity', action="count", default=int(os.environ.get("VERBOSITY", 2)),
                        help='each occurrence increases verbosity 1 level through ERROR-WARNING-INFO-DEBUG (default INFO)')
    args = parser.parse_args()

    levels = [logging.ERROR, logging.WARN, logging.INFO, logging.DEBUG]
    level=levels[min(3, args.verbosity)]
    logging.basicConfig(level=level)
    # logging.getLogger('vispy').setLevel(level)

    app.create()
    # app = QApplication(sys.argv)
    window = Main(
        workspace_dir=args.workspace,
        workspace_size=args.space,
        glob_pattern=args.glob_pattern,
        border_shapefile=args.border_shapefile,
        center=args.center,
    )
    window.show()
    print("running")
    # bring window to front
    window.raise_()
    app.run()

if __name__ == '__main__':
    import sys
    sys.exit(main())
