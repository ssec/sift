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
from PyQt4.QtCore import QObject, Qt
from PyQt4.QtGui import (QWidget, QTreeView,
                         QGridLayout, QLabel, QTextEdit, QFont, QSizePolicy)
from sift.common import INFO, KIND
from sift.control.layer_tree import LayerStackTreeViewModel
from sift.model.layer import DocRGBLayer
from sift.ui.custom_widgets import QNoScrollWebView
import numpy as np
from sift.view.Colormap import COLORMAP_MANAGER

LOG = logging.getLogger(__name__)


class LayerSetsManager(QObject):
    """This is the controller object that manages the extended functionality of the layer sets.
    """

    document = None
    layer_list_widget = None
    layer_list_model = None
    layer_info_pane = None

    def __init__(self, layer_list_widget:QWidget, layer_info_widget:QWidget, document):

        super(LayerSetsManager, self).__init__(layer_list_widget)
        self.document = document

        # hang on to the various widgets for later
        self.layer_list_widget = layer_list_widget
        self.layer_info_pane = SingleLayerInfoPane(layer_info_widget, document)

        # set up our layer sets and make the first one
        new_layer_set = QTreeView(self.layer_list_widget)
        self.layer_list_widget.layout().addWidget(new_layer_set)
        self.layer_list_model = LayerStackTreeViewModel([new_layer_set], self.document, parent=self.layer_list_widget)
        self.layer_list_model.uuidSelectionChanged.connect(self.layer_info_pane.update_display)


class SingleLayerInfoPane(QObject):
    """shows details about one layer that is selected in the list
    """

    document = None
    name_text = None
    time_text = None
    instrument_text = None
    band_text = None
    colormap_text = None
    clims_text = None

    def __init__(self, parent, document):
        """build our info display
        """
        super(SingleLayerInfoPane, self).__init__(parent)

        self.document = document  # FUTURE: make this a weakref?

        # build our layer detail info display controls
        self.name_text = QLabel("")
        self.name_text.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.time_text = QLabel("")
        self.time_text.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.instrument_text = QLabel("")
        self.instrument_text.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.band_text = QLabel("")
        self.band_text.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.wavelength_text = QLabel("")
        self.wavelength_text.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.colormap_text = QLabel("")
        self.colormap_text.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.clims_text = QLabel("")
        self.clims_text.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.cmap_vis = QNoScrollWebView()
        self.cmap_vis.setFixedSize(3 * 100, 30)
        self.cmap_vis.page().mainFrame().setScrollBarPolicy(Qt.Vertical, Qt.ScrollBarAlwaysOff)
        self.composite_details = QLabel("Composite Details")
        self.composite_details.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        f = QFont()
        f.setUnderline(True)
        self.composite_details.setFont(f)
        self.composite_codeblock = QTextEdit()
        self.composite_codeblock.setReadOnly(True)
        self.composite_codeblock.setMinimumSize(3 * 100, 100)
        self.composite_codeblock.setDisabled(True)
        self.composite_codeblock.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)

        # set the layout
        # Note: add in a grid is (widget, row#, col#) or (widget, row#, col#, row_span, col_span)
        layout = QGridLayout()
        layout.addWidget(self.name_text,       1, 1)
        layout.addWidget(self.time_text,       2, 1)
        layout.addWidget(self.instrument_text, 3, 1)
        layout.addWidget(self.band_text,       4, 1)
        layout.addWidget(self.wavelength_text, 5, 1)
        layout.addWidget(self.colormap_text,   6, 1)
        layout.addWidget(self.clims_text,      7, 1)
        layout.addWidget(self.cmap_vis, 8, 1)
        layout.addWidget(self.composite_details, 9, 1)
        layout.addWidget(self.composite_codeblock, 10, 1)
        parent.setLayout(layout)

        # clear out the display
        self.update_display()

    def update_display(self, selected_uuid_list=None):
        """update the information being displayed to match the given UUID(s)

        If the uuid list parameter is None, clear out the information instead
        """
        if selected_uuid_list is not None and len(selected_uuid_list)==1:
            layer_uuid, = list(selected_uuid_list)
            layer_info = self.document[layer_uuid]
            is_rgb = isinstance(layer_info, DocRGBLayer)

        # clear the list if we got None
        if selected_uuid_list is None or len(selected_uuid_list) <= 0:
            # set the various text displays
            self.name_text.setText("Name: ")
            self.time_text.setText("Time: ")
            self.instrument_text.setText("Instrument: ")
            self.band_text.setText("Band: ")
            self.wavelength_text.setText("Wavelength: ")
            self.colormap_text.setText("Colormap: ")
            self.clims_text.setText("Color Limits: ")
            self.cmap_vis.setHtml("")
            self.composite_codeblock.setText("")
        else:
            # otherwise display information on the selected layer(s)
            # figure out the info shared between all the layers currently selected
            shared_info = {}
            presentation_info = self.document.current_layer_set
            for layer_uuid in selected_uuid_list:

                layer_info = self.document.get_info(uuid=layer_uuid)

                this_prez = None
                for prez_tuple in presentation_info :
                    if prez_tuple.uuid == layer_uuid :
                        this_prez = prez_tuple

                # compare our various values

                # name
                new_name = layer_info[INFO.DISPLAY_NAME] if INFO.DISPLAY_NAME in layer_info else ""
                if INFO.DISPLAY_NAME not in shared_info:
                    shared_info[INFO.DISPLAY_NAME] = new_name
                else:
                    shared_info[INFO.DISPLAY_NAME] = "" if shared_info[INFO.DISPLAY_NAME] != new_name else new_name

                # time
                new_time = layer_info[INFO.DISPLAY_TIME] if INFO.DISPLAY_TIME in layer_info else ""
                if INFO.DISPLAY_TIME not in shared_info :
                    shared_info[INFO.DISPLAY_TIME] = new_time
                else :
                    shared_info[INFO.DISPLAY_TIME] = "" if shared_info[INFO.DISPLAY_TIME] != new_time else new_time

                # instrument
                new_inst = str(layer_info[INFO.INSTRUMENT].value) if layer_info.get(INFO.INSTRUMENT) else ""
                if INFO.INSTRUMENT not in shared_info :
                    shared_info[INFO.INSTRUMENT] = new_inst
                else :
                    shared_info[INFO.INSTRUMENT] = "" if shared_info[INFO.INSTRUMENT] != new_inst else new_inst

                # band
                new_band = layer_info.get(INFO.BAND)
                if isinstance(new_band, (tuple, list)):
                    new_band = "(" + ", ".join([str(x) if x is not None else '---' for x in new_band]) + ")"
                else:
                    new_band = str(new_band) if new_band is not None else '---'
                if INFO.BAND not in shared_info:
                    shared_info[INFO.BAND] = new_band
                else:
                    shared_info[INFO.BAND] = "---" if shared_info[INFO.BAND] != new_band else new_band

                # wavelength
                wl = layer_info.get(INFO.CENTRAL_WAVELENGTH)
                fmt = "{:0.2f} µm"
                if isinstance(wl, (tuple, list)):
                    wl = [fmt.format(x) if x is not None else '---' for x in wl]
                    wl = "(" + ", ".join(wl) + ")"
                else:
                    wl = fmt.format(wl) if wl is not None else '---'
                if INFO.CENTRAL_WAVELENGTH not in shared_info:
                    shared_info[INFO.CENTRAL_WAVELENGTH] = wl
                else:
                    shared_info[INFO.CENTRAL_WAVELENGTH] = "---" if shared_info[INFO.CENTRAL_WAVELENGTH] != wl else wl

                # colormap
                new_cmap = this_prez.colormap if this_prez is not None else ""
                if "colormap" not in shared_info:
                    shared_info["colormap"] = new_cmap
                else :
                    shared_info["colormap"] = "" if shared_info["colormap"] != new_cmap else new_cmap

                # c-limits
                new_clims = ""
                if this_prez is not None:
                    new_clims = np.array(this_prez.climits)
                    unit_info = self.document[this_prez.uuid][INFO.UNIT_CONVERSION]
                    new_clims = unit_info[1](new_clims, inverse=False)
                    try:
                        if layer_info[INFO.KIND] in [KIND.IMAGE, KIND.COMPOSITE, KIND.CONTOUR]:
                            min_str = layer_info[INFO.UNIT_CONVERSION][2](new_clims[0], include_units=False)
                            max_str = layer_info[INFO.UNIT_CONVERSION][2](new_clims[1])
                            new_clims = '{} ~ {}'.format(min_str, max_str)
                        elif layer_info[INFO.KIND] in [KIND.RGB]:
                            # FUTURE: Other layer types
                            deps = (layer_info.r, layer_info.g, layer_info.b)

                            tmp_clims = []
                            for i, dep in enumerate(deps):
                                if dep is None:
                                    tmp_clims.append('N/A')
                                    continue

                                min_str = dep[INFO.UNIT_CONVERSION][2](new_clims[i][0], include_units=False)
                                max_str = dep[INFO.UNIT_CONVERSION][2](new_clims[i][1])
                                tmp_clims.append('{} ~ {}'.format(min_str, max_str))
                            new_clims = ", ".join(tmp_clims)
                        else:
                            new_clims = "N/A"
                    except TypeError as err:
                        LOG.warning("unable to format color limit: %r" % (new_clims,), exc_info=True)
                        new_clims = "N/A"
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

                ns, codeblock = self.document.get_algebraic_namespace(layer_uuid)
                if codeblock:
                    short_names = []
                    for name, uuid in ns.items():
                        try:
                            dep_info = self.document[uuid]
                            short_name = dep_info.get(INFO.SHORT_NAME, '<N/A>')
                        except KeyError:
                            LOG.debug("Layer '{}' not found in document".format(uuid))
                            short_name = '<Unknown>'
                        short_names.append("# {} = {}".format(name, short_name))
                    ns_str = "\n".join(short_names)
                    codeblock_str = ns_str + '\n\n' + codeblock
                else:
                    codeblock_str = ''
                if 'codeblock' not in shared_info:
                    shared_info['codeblock'] = codeblock_str
                else:
                    shared_info['codeblock'] = '' if shared_info['codeblock'] != codeblock_str else codeblock_str

            # set the various text displays
            temp_name = shared_info[INFO.DISPLAY_NAME] if INFO.DISPLAY_NAME in shared_info else ""
            self.name_text.setText("Name: " + temp_name)
            temp_time = shared_info[INFO.DISPLAY_TIME] if INFO.DISPLAY_TIME in shared_info else ""
            self.time_text.setText("Time: " + (temp_time or ""))
            temp_inst = shared_info[INFO.INSTRUMENT] if INFO.INSTRUMENT in shared_info else ""
            self.instrument_text.setText("Instrument: " + temp_inst)
            temp_band = shared_info[INFO.BAND] if INFO.BAND in shared_info else ""
            self.band_text.setText("Band: " + temp_band)
            temp_wl = shared_info[INFO.CENTRAL_WAVELENGTH] if INFO.CENTRAL_WAVELENGTH in shared_info else ""
            self.wavelength_text.setText("Wavelength: " + temp_wl)
            temp_cmap = shared_info["colormap"] if shared_info.get("colormap", None) is not None else ""
            self.colormap_text.setText("Colormap: " + temp_cmap)
            temp_clims = shared_info["climits"] if "climits" in shared_info else ""
            self.clims_text.setText("Color Limits: " + temp_clims)
            self.composite_codeblock.setText(shared_info['codeblock'])

            # format colormap
            if shared_info.get("colormap", None) is None:
                self.cmap_vis.setHtml("")
            else:
                cmap_html = COLORMAP_MANAGER[shared_info["colormap"]]._repr_html_()
                cmap_html = cmap_html.replace("height", "border-collapse: collapse;\nheight")
                self.cmap_vis.setHtml("""<html><head></head><body style="margin: 0px"><div>%s</div></body></html>""" % (cmap_html,))
