#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

PURPOSE
Manage the layer sets.

REFERENCES

REQUIRES

:author: Eva Schiffer <evas@ssec.wisc.edu>
:copyright: 2015 by University of Wisconsin Regents, see AUTHORS for more details
:license: GPLv3, see LICENSE for more details
"""
__author__ = 'evas'
__docformat__ = 'reStructuredText'

import logging
from PyQt4.QtCore import SIGNAL, QObject, Qt
from PyQt4.QtGui import QWidget, QListView, QGridLayout, QLabel, QScrollArea, QLayout, QTextDocument
from PyQt4.QtWebKit import QWebView
from cspov.model.guidebook import GUIDE
from cspov.common import INFO, KIND
from cspov.control.layer_list import LayerStackListViewModel
import numpy as np
from cspov.view.Colormap import ALL_COLORMAPS

LOG = logging.getLogger(__name__)

class LayerSetsManager (QObject) :
    """This is the controller object that manages the extended functionality of the layer sets.
    """

    document = None
    tab_widget = None
    layer_sets = None
    max_tab_number = None
    set_behaviors = None
    layer_info_object = None

    def __init__ (self, tab_view_widget, layer_info_widget, document) :

        super(LayerSetsManager, self).__init__(tab_view_widget)

        self.document = document

        # hang on to the various widgets for later
        self.tab_widget = tab_view_widget
        self.layer_info_object = SingleLayerInfoDisplay(layer_info_widget, document)

        if tab_view_widget.count() > 1 :
            LOG.info("Unexpected number of tabs present at start up in the layer list set pane.")

        # set up our layer sets and make the first one
        self.layer_sets = [ ]

        self.max_tab_number = 1
        self.set_up_tab(0, do_increment_tab_number=False)
        self.set_behaviors.uuidSelectionChanged.connect(self.layer_info_object.update_display)

        # hook things up so we know when the selected tab changes
        self.tab_widget.connect(self.tab_widget,
                                SIGNAL('currentChanged(int)'),
                                self.handle_tab_change)

    def handle_tab_change (self, ) :
        """deal with the fact that the tab changed in the tab widget
        """

        newTabIndex = self.tab_widget.currentIndex()

        # if this is the last tab, make a new tab and switch to that
        if newTabIndex == (self.tab_widget.count() - 1) :
            LOG.info ("Creating new layer set tab.")

            self.set_up_tab(newTabIndex)

        # tell the document which layer set we're using now
        self.document.select_layer_set(newTabIndex)

    def set_up_tab(self, new_tab_index, do_increment_tab_number=True) :
        """Build a new layer set tab
        """

        # increment our tab label number if desired
        if do_increment_tab_number :
            self.max_tab_number = self.max_tab_number + 1

        # create our tab
        temp_widget = QWidget()
        self.tab_widget.insertTab(new_tab_index, temp_widget, str(self.max_tab_number))

        # create the associated graph display object
        new_layer_set = SingleLayerSetManager(temp_widget, str(self.max_tab_number))
        self.layer_sets.append(new_layer_set)
        layer_list_obj = new_layer_set.getLayerList()
        if self.set_behaviors is None :
            self.set_behaviors = LayerStackListViewModel([layer_list_obj], self.document)
        else :
            self.set_behaviors.add_widget(layer_list_obj)

        # go to the tab we just created
        self.tab_widget.setCurrentIndex(new_tab_index)

    def getLayerStackListViewModel (self, ) :
        return self.set_behaviors

class SingleLayerSetManager (QWidget) :
    """handles controls and data for a single layer list
    """

    my_name = None
    my_layer_list = None

    def __init__(self, parent, set_name) :
        """set up the controls and signals for this layer set
        """

        super(SingleLayerSetManager, self).__init__(parent)

        self.my_name = set_name

        # create our controls

        # the list of layers
        self.my_layer_list = QListView (parent)

        # set the layout
        # Note: add in a grid is (widget, row#, col#) or (widget, row#, col#, row_span, col_span)
        layout = QGridLayout()
        layout.addWidget(self.my_layer_list,   1, 1)
        parent.setLayout(layout)

    def getLayerList (self,) :

        return self.my_layer_list

class SingleLayerInfoDisplay (QWidget) :
    """shows details about one layer that is selected in the list
    """

    document = None
    name_text = None
    time_text = None
    instrument_text = None
    band_text = None
    colormap_text = None
    clims_text = None

    def __init__(self, parent, document) :
        """build our info display
        """

        super(SingleLayerInfoDisplay, self).__init__(parent)

        self.document = document

        # build our layer detail info display controls
        self.name_text = QLabel("")
        self.time_text = QLabel("")
        self.instrument_text = QLabel("")
        self.band_text = QLabel("")
        self.colormap_text = QLabel("")
        self.clims_text = QLabel("")
        self.cmap_vis = QWebView()
        self.cmap_vis.setFixedSize(3 * 100, 30)
        self.cmap_vis.page().mainFrame().setScrollBarPolicy(Qt.Vertical, Qt.ScrollBarAlwaysOff)

        # set the layout
        # Note: add in a grid is (widget, row#, col#) or (widget, row#, col#, row_span, col_span)
        layout = QGridLayout()
        layout.addWidget(self.name_text,       1, 1)
        layout.addWidget(self.time_text,       2, 1)
        layout.addWidget(self.instrument_text, 3, 1)
        layout.addWidget(self.band_text,       4, 1)
        layout.addWidget(self.colormap_text,   5, 1)
        layout.addWidget(self.clims_text,      6, 1)
        layout.addWidget(self.cmap_vis, 7, 1)
        parent.setLayout(layout)

        # TODO put the informational text into an area with scroll bars so it won't force the sidebar size minimum

        # clear out the display
        self.update_display()

    def update_display (self, selected_UUID_list=None) :
        """update the information being displayed to match the given UUID(s)

        If the uuid list parameter is None, clear out the information instead
        """

        # clear the list if we got None
        if selected_UUID_list is None or len(selected_UUID_list) <= 0 :
            # set the various text displays
            self.name_text.setText("Name: ")
            self.time_text.setText("Time: ")
            self.instrument_text.setText("Instrument: ")
            self.band_text.setText("Band: ")
            self.colormap_text.setText("Colormap: ")
            self.clims_text.setText("C-Limits: ")
            self.cmap_vis.setHtml("")

        # otherwise display information on the selected layer(s)
        else :

            # figure out the info shared between all the layers currently selected
            shared_info = { }
            presentation_info = self.document.current_layer_set
            for layer_uuid in selected_UUID_list :

                layer_info = self.document.get_info(uuid=layer_uuid)

                this_prez = None
                for prez_tuple in presentation_info :
                    if prez_tuple.uuid == layer_uuid :
                        this_prez = prez_tuple

                # compare our various values

                # name
                new_name = layer_info[INFO.NAME] if INFO.NAME in layer_info else ""
                if INFO.NAME not in shared_info :
                    shared_info[INFO.NAME] = new_name
                else :
                    shared_info[INFO.NAME] = "" if shared_info[INFO.NAME] != new_name else new_name

                # time
                new_time = layer_info[GUIDE.DISPLAY_TIME] if GUIDE.DISPLAY_TIME in layer_info else ""
                if GUIDE.DISPLAY_TIME not in shared_info :
                    shared_info[GUIDE.DISPLAY_TIME] = new_time
                else :
                    shared_info[GUIDE.DISPLAY_TIME] = "" if shared_info[GUIDE.DISPLAY_TIME] != new_time else new_time

                # instrument
                new_inst = str(layer_info[GUIDE.INSTRUMENT].value) if GUIDE.INSTRUMENT in layer_info else ""
                if GUIDE.INSTRUMENT not in shared_info :
                    shared_info[GUIDE.INSTRUMENT] = new_inst
                else :
                    shared_info[GUIDE.INSTRUMENT] = "" if shared_info[GUIDE.INSTRUMENT] != new_inst else new_inst

                # band
                new_band = str(layer_info[GUIDE.BAND]) if GUIDE.BAND in layer_info else ""
                if GUIDE.BAND not in shared_info :
                    shared_info[GUIDE.BAND] = new_band
                else :
                    shared_info[GUIDE.BAND] = "" if shared_info[GUIDE.BAND] != new_band else new_band

                # colormap
                new_cmap = this_prez.colormap if this_prez is not None else ""
                if "colormap" not in shared_info :
                    shared_info["colormap"] = new_cmap
                else :
                    shared_info["colormap"] = "" if shared_info["colormap"] != new_cmap else new_cmap

                # c-limits
                new_clims = ""
                if this_prez is not None:
                    new_clims = np.array(this_prez.climits)
                    fmt, unit, new_clims = self.document.convert_units(this_prez.uuid, new_clims, inverse=False)
                    new_clims = (fmt + ' ~ ' + fmt + '{}').format(new_clims[0], new_clims[1], unit)
                if "climits" not in shared_info :
                    shared_info["climits"] = new_clims
                else :
                    shared_info["climits"] = "" if shared_info["climits"] != new_clims else new_clims

                # color map
                cmap = this_prez.colormap if this_prez is not None else None
                if "colormap" not in shared_info:
                    shared_info["colormap"] = cmap
                else:
                    shared_info["colormap"] = None if shared_info["colormap"] != cmap else cmap

            #print("*** layer info: " + str(layer_info))
            # *** layer info: {
            # <GUIDE.INSTRUMENT: 'instrument'>: <INSTRUMENT.AHI: 'AHI'>,
            # <INFO.PROJ: 'proj4_string'>: '+proj=merc +lon_0=0 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs ',
            # <GUIDE.SCHED_TIME: 'timeline'>: datetime.datetime(2015, 8, 24, 19, 0),
            # <INFO.ORIGIN_X: 'origin_x'>: 6614709.252,
            # <GUIDE.DISPLAY_TIME: 'display_time'>: '2015-08-24 19:00',
            # <INFO.CLIM: 'clim'>: (-0.012, 1.192),
            # <INFO.CELL_HEIGHT: 'cell_height'>: -1000.0,
            # <GUIDE.UUID: 'uuid'>: UUID('5f547fae-7c26-11e5-bece-28cfe915d94b'),
            # <INFO.NAME: 'name'>: 'AHI B01 Refl 2015-08-24 19:00',
            # <INFO.UUID: 'uuid'>: UUID('5f547fae-7c26-11e5-bece-28cfe915d94b'),
            # <INFO.CELL_WIDTH: 'cell_width'>: 1000.0,
            # <INFO.KIND: 'kind'>: <KIND.IMAGE: 1>,
            # <INFO.SHAPE: 'shape'>: (30993, 18096),
            # <GUIDE.SPACECRAFT: 'spacecraft'>: 'Himawari-8',
            # <GUIDE.BAND: 'band'>: 1,
            # <INFO.PATHNAME: 'pathname'>: './test_data/ahi 2015_08_24_236 1900/HS_H08_20150824_1900_B01_FLDK_R20.merc.tif',
            # <GUIDE.SCENE: 'scene'>: 'FLDK',
            # <INFO.ORIGIN_Y: 'origin_y'>: 15496570.74,
            # <GUIDE.DISPLAY_NAME: 'display_name'>: 'AHI B01 Refl 2015-08-24 19:00'}

            #print ("*** layer presentation info: " + str(this_prez))
            # *** layer presentation info: [
            # prez(uuid=UUID('728da4de-7c26-11e5-bac8-28cfe915d94b'),
            #   kind=<KIND.IMAGE: 1>,
            #   visible=True,
            #   a_order=None,
            #   colormap='Square Root (Vis Default)',
            #   climits=(-0.012, 1.192), mixing=<mixing.NORMAL: 1>),
            # prez(uuid=UUID('5f547fae-7c26-11e5-bece-28cfe915d94b'),
            #   kind=<KIND.IMAGE: 1>,
            #   visible=True,
            #   a_order=None,
            #   colormap='Square Root (Vis Default)',
            #   climits=(-0.012, 1.192),
            #   mixing=<mixing.NORMAL: 1>)]

            # set the various text displays
            temp_name = shared_info[INFO.NAME] if INFO.NAME in shared_info else ""
            self.name_text.setText("Name: " + temp_name)
            temp_time = shared_info[GUIDE.DISPLAY_TIME] if GUIDE.DISPLAY_TIME in shared_info else ""
            self.time_text.setText("Time: " + temp_time)
            temp_inst = shared_info[GUIDE.INSTRUMENT] if GUIDE.INSTRUMENT in shared_info else ""
            self.instrument_text.setText("Instrument: " + temp_inst)
            temp_band = shared_info[GUIDE.BAND] if GUIDE.BAND in shared_info else ""
            self.band_text.setText("Band: " + temp_band)
            temp_cmap = shared_info["colormap"] if shared_info.get("colormap", None) is not None else ""
            self.colormap_text.setText("Colormap: " + temp_cmap)
            temp_clims = shared_info["climits"] if "climits" in shared_info else ""
            self.clims_text.setText("C-Limits: " + temp_clims)

            # format colormap
            if shared_info.get("colormap", None) is None:
                self.cmap_vis.setHtml("")
            else:
                cmap_html = ALL_COLORMAPS[shared_info["colormap"]]._repr_html_()
                cmap_html = cmap_html.replace("height", "border-collapse: collapse;\nheight")
                self.cmap_vis.setHtml("""<html><head></head><body style="margin: 0px"><div>%s</div></body></html>""" % (cmap_html,))

