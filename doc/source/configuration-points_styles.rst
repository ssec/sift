Points Styles
-------------

Definition of Points Styles
===========================

Named point styles are defined below the setting ``point_styles``. They follow
the conventions of SVG/HTML styling (see https://www.w3.org/TR/SVG/styling.html
and https://www.w3.org/TR/SVG/painting.html) but only a small subset is
supported (yet)::

    point_styles:
      STYLE_NAME:
        symbol: SYMBOL_NAME
        size: LENGTH_DEFINITON
        fill: COLOR
        stroke: COLOR
        stroke-width: LENGTH_DEFINITON

where

- ``STYLE_NAME`` is an arbitrary name for the marker style to be defined
- ``SYMBOL_NAME`` is one of ``disc``, ``arrow``, ``ring``, ``clobber``,
  ``square``, ``diamond``, ``vbar``, ``hbar``, ``cross``, ``tailed_arrow``,
  ``x``, ``triangle_up``, ``triangle_down``, ``star``
- ``LENGTH_DEFINITON`` is a number directly followed by one of the units ``px``
  or ``%`` which controls, whether the value is interpreted in screen pixels or
  relative to the scene, i.e. zoom level dependant
- ``COLOR`` is a string recognized by the underlying Vispy library (see
  http://vispy.org/color.html#vispy.color.Color), i.e., something like ``black``
  or ``yellow`` or a hexadecimal representation of an RGB or RGBA value
  introduced by '``#``' (e.g., ``'#00FF0080'`` is semitransparent green). Note,
  that in the latter case the string *must be quoted* to prevent the
  configuration parser from interpreting the  '``#``' as begin of a comment.

All styling settings are optional, for missing ones a predefined default is
chosen.

**Example**

The following would be an explicit definition of the default marker style::

    point_styles:
      red_empty_cross:
        symbol: cross
        size: 9px
        fill: '#00000000'
        stroke: white
        stroke-width: 1px

Configuring Point Styles Based on the Dataset Standard Name
-----------------------------------------------------------

Analogous to the default colormap configuration the style of the markers used to
render a dataset can (and - since there are no default definitions in the
SIFT's internal *Guidebook* for this - should) be configured by associating a
named points style to the standard name of each points dataset::

    default_point_styles:
      STANDARD_NAME_1: STYLE_NAME_1
      STANDARD_NAME_2: STYLE_NAME_2
      ...

**Example** ::

    default_colormaps:
      observed_lmk_locations_ir_105: red_empty_cross
