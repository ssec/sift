Design Overview
===============

SIFT's software design revolves around a few key components:

- Main Window (GUI)
- Workspace
- Document
- Scene Graph

Each of these components is described in the sections below. Other
components involved in accomplishing SIFT's feature.

Main Window
-----------

Currently the main window for the SIFT GUI connects all other components
and helper objects. This may change in future versions of SIFT. By defining
things this way the main window has access to UI events and can connect them
to the other SIFT components that need to use them like those listed below.

Workspace
---------

The Workspace acts as the manager of on-disk or remote data. It will
handle importing requested datasets, caching binary data, and storing
dataset metadata in a database for easier querying. Since the Workspace
manages all of the cached data it is also the best place that SIFT components
will go for variations on the data (different resolutions, data within a
polygon, etc).

Document
--------

The Document acts as the "model" of the Model-View-Controller design of SIFT.
Through the Document a developer can get access to individual layer objects
containing metadata, layer order, and animation order. In the future as other
features are added to SIFT the Document may provide user profile or
configuration information.

Scene Graph
-----------

The Scene Graph wraps all map canvas elements visual elements.
It handles connecting all mouse events from the map canvas like pan and zoom
events. The majority of this components responsibility is to map SIFT
functions to the python ``vispy`` library.
