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
import enum
import typing as typ
from collections import defaultdict
from uuid import UUID

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QGridLayout, QLabel, QTextEdit, QSizePolicy, QWidget, QVBoxLayout

from uwsift.model.layer import DocBasicLayer
from uwsift.common import Info, Kind, Presentation
from uwsift.ui.custom_widgets import QNoScrollWebView
from uwsift.view.colormap import COLORMAP_MANAGER

LOG = logging.getLogger(__name__)


class SingleLayerInfoPane(QWidget):
    """Shows details about one layer that is current selected."""

    document = None
    name_text = None
    time_text = None
    instrument_text = None
    colormap_text = None
    clims_text = None

    def __init__(self, document, parent=None):
        """Create subwidgets and layout."""
        super(SingleLayerInfoPane, self).__init__(parent)
        self.document = document  # FUTURE: make this a weakref?
        min_width = 3 * 100

        # build our layer detail info display controls
        self.name_text = QLabel("")
        self.name_text.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.time_text = QLabel("")
        self.time_text.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.instrument_text = QLabel("")
        self.instrument_text.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.wavelength_text = QLabel("")
        self.wavelength_text.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.colormap_text = QLabel("")
        self.colormap_text.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.clims_text = QLabel("")
        self.clims_text.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.cmap_vis = QNoScrollWebView()
        self.cmap_vis.setFixedSize(min_width, 30)
        self.cmap_vis.page().runJavaScript('document.body.style.overflow = "hidden";')
        self.composite_details = QLabel("Composite Details")
        self.composite_details.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        f = QFont()
        f.setUnderline(True)
        self.composite_details.setFont(f)
        self.composite_codeblock = QTextEdit()
        self.composite_codeblock.setReadOnly(True)
        self.composite_codeblock.setMinimumSize(min_width, 100)
        self.composite_codeblock.setDisabled(True)
        self.composite_codeblock.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)

        # set the layout for this widget
        # Note: add in a grid is (widget, row#, col#) or (widget, row#, col#, row_span, col_span)
        layout = QGridLayout()
        layout.setAlignment(Qt.AlignLeft)
        layout.addWidget(self.name_text, 0, 0)
        layout.addWidget(self.time_text, 1, 0)
        layout.addWidget(self.instrument_text, 2, 0)
        layout.addWidget(self.wavelength_text, 3, 0)
        layout.addWidget(self.colormap_text, 4, 0)
        layout.addWidget(self.clims_text, 5, 0)
        layout.addWidget(self.cmap_vis, 6, 0)
        layout.addWidget(self.composite_details, 7, 0)
        layout.addWidget(self.composite_codeblock, 8, 0)
        self.setLayout(layout)

        # add this widget to the scrollable parent widget
        layout = QVBoxLayout()
        layout.addWidget(self)
        self.parent().setLayout(layout)

        # clear out the display
        self.update_display()

    def _display_shared_info(self, shared_info: defaultdict):
        self.name_text.setText("Name: " + shared_info[Info.DISPLAY_NAME])
        self.time_text.setText("Time: " + shared_info[Info.DISPLAY_TIME])
        self.instrument_text.setText("Instrument: " + shared_info[Info.INSTRUMENT])
        self.wavelength_text.setText("Wavelength: " + shared_info[Info.CENTRAL_WAVELENGTH])
        self.colormap_text.setText("Colormap: " + (shared_info["colormap"] or ""))
        self.clims_text.setText("Color Limits: " + shared_info["climits"])
        self.composite_codeblock.setText(shared_info['codeblock'])

        # format colormap
        if not shared_info["colormap"]:
            self.cmap_vis.setHtml("")
        else:
            cmap_html = COLORMAP_MANAGER[shared_info["colormap"]]._repr_html_()
            cmap_html = cmap_html.replace("height", "border-collapse: collapse;\nheight")
            self.cmap_vis.setHtml(
                """<html><head></head><body style="margin: 0px"><div>%s</div></body></html>""" % (cmap_html,))

    def _update_if_different(self, shared_info: defaultdict, layer_info: typ.Union[DocBasicLayer, Presentation],
                             key: typ.Hashable = None, attr: str = None, default=""):
        """Get information from the provided layer or presentation.

        Compares current layer information with existing info in ``shared_info``. If they are the same then keep the
        value. Otherwise reset the value to ``default`` (empty string by default).
        """
        if key is not None:
            new_name = layer_info[key] if key in layer_info else default
        elif attr is not None:
            new_name = getattr(layer_info, attr, default)
            key = attr
        else:
            raise ValueError("Either 'key' or 'attr' must be provided.")

        if isinstance(new_name, enum.Enum):
            new_name = new_name.value
        if key not in shared_info:
            shared_info[key] = new_name
        else:
            shared_info[key] = default if shared_info[key] != new_name else new_name

    def _get_shared_color_limits(self, shared_info: defaultdict, layer_info: DocBasicLayer, this_prez: Presentation):
        new_clims = ""
        if this_prez is not None:
            new_clims = np.array(this_prez.climits)
            unit_info = self.document[this_prez.uuid][Info.UNIT_CONVERSION]
            new_clims = unit_info[1](new_clims, inverse=False)
            try:
                if layer_info[Info.KIND] in [Kind.IMAGE, Kind.COMPOSITE, Kind.CONTOUR]:
                    min_str = layer_info[Info.UNIT_CONVERSION][2](new_clims[0], include_units=False)
                    max_str = layer_info[Info.UNIT_CONVERSION][2](new_clims[1])
                    new_clims = '{} ~ {}'.format(min_str, max_str)
                elif layer_info[Info.KIND] in [Kind.RGB]:
                    # FUTURE: Other layer types
                    deps = (layer_info.r, layer_info.g, layer_info.b)

                    tmp_clims = []
                    for i, dep in enumerate(deps):
                        if dep is None:
                            tmp_clims.append('N/A')
                            continue

                        min_str = dep[Info.UNIT_CONVERSION][2](new_clims[i][0], include_units=False)
                        max_str = dep[Info.UNIT_CONVERSION][2](new_clims[i][1])
                        tmp_clims.append('{} ~ {}'.format(min_str, max_str))
                    new_clims = ", ".join(tmp_clims)
                else:
                    new_clims = "N/A"
            except TypeError:
                LOG.warning("unable to format color limit: %r" % (new_clims,), exc_info=True)
                new_clims = "N/A"
        if "climits" not in shared_info:
            shared_info["climits"] = new_clims
        else:
            shared_info["climits"] = "" if shared_info["climits"] != new_clims else new_clims

    def _get_code_block(self, shared_info: defaultdict, layer_uuid: UUID):
        ns, codeblock = self.document.get_algebraic_namespace(layer_uuid)
        if codeblock:
            short_names = []
            for name, uuid in ns.items():
                try:
                    dep_info = self.document[uuid]
                    short_name = dep_info.get(Info.SHORT_NAME, '<N/A>')
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

    def update_display(self, selected_uuid_list: list = None):
        """update the information being displayed to match the given UUID(s)

        If the uuid list parameter is None, clear out the information instead
        """
        shared_info = defaultdict(str)

        # clear the list if we got None
        if selected_uuid_list is None or len(selected_uuid_list) <= 0:
            self._display_shared_info(shared_info)
            return

        # otherwise display information on the selected layer(s)
        # figure out the info shared between all the layers currently selected
        presentation_info = self.document.current_layer_set
        for layer_uuid in selected_uuid_list:
            layer_info = self.document.get_info(uuid=layer_uuid)

            this_prez = None
            for prez_tuple in presentation_info:
                if prez_tuple.uuid == layer_uuid:
                    this_prez = prez_tuple
                    break

            # compare our various values
            self._update_if_different(shared_info, layer_info, Info.DISPLAY_NAME)
            self._update_if_different(shared_info, layer_info, Info.DISPLAY_TIME)
            self._update_if_different(shared_info, layer_info, Info.INSTRUMENT)
            self._update_if_different(shared_info, this_prez, attr='colormap')
            self._get_shared_color_limits(shared_info, layer_info, this_prez)
            self._get_code_block(shared_info, layer_uuid)

            # wavelength
            wl = layer_info.get(Info.CENTRAL_WAVELENGTH)
            fmt = "{:0.2f} µm"
            if isinstance(wl, (tuple, list)):
                wl = [fmt.format(x) if x is not None else '---' for x in wl]
                wl = "(" + ", ".join(wl) + ")"
            else:
                wl = fmt.format(wl) if wl is not None else '---'
            if Info.CENTRAL_WAVELENGTH not in shared_info:
                shared_info[Info.CENTRAL_WAVELENGTH] = wl
            else:
                shared_info[Info.CENTRAL_WAVELENGTH] = "---" if shared_info[Info.CENTRAL_WAVELENGTH] != wl else wl

        # set the various text displays
        self._display_shared_info(shared_info)
