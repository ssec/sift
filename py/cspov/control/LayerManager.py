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
from PyQt4.QtCore import SIGNAL, QObject
from PyQt4.QtGui import QWidget, QListView, QGridLayout
from cspov.model.document import Document
from cspov.common import INFO, KIND
from cspov.control.layer_list import LayerStackListViewModel

LOG = logging.getLogger(__name__)

class LayerSetsManager (QObject) :
    """This is the controller object that manages the extended functionality of the layer sets.
    """

    document = None
    tab_widget = None
    layer_sets = None
    max_tab_number = None
    set_behaviors = None

    def __init__ (self, tab_view_widget, document) :

        super(LayerSetsManager, self).__init__(tab_view_widget)

        self.document = document

        # hang on to the list model and the tab view widget for later
        self.tab_widget = tab_view_widget

        if tab_view_widget.count() > 1 :
            LOG.info("Unexpected number of tabs present at start up in the layer list set pane.")

        # set up our layer sets and make the first one
        self.layer_sets = [ ]

        self.max_tab_number = 1
        self.set_up_tab(0, do_increment_tab_number=False)

        # hook things up so we know when the selected tab changes
        self.tab_widget.connect(self.tab_widget,
                                SIGNAL('currentChanged(int)'),
                                self.handle_tab_change)

        # todo hook up the various document signals that would mean we need to reload things

    def handle_tab_change (self, ) :
        """deal with the fact that the tab changed in the tab widget
        """

        newTabIndex = self.tab_widget.currentIndex()

        # if this is the last tab, make a new tab and switch to that
        if newTabIndex == (self.tab_widget.count() - 1) :
            LOG.info ("Creating new layer set tab.")

            self.set_up_tab(newTabIndex)

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


















