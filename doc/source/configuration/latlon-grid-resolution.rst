.. role:: yaml(code)

Changing spacing of lat/lon grid
--------------------------------

The spacing of the lat/lon grid can be controlled with the setting::

    latlon_grid:
        resolution: [value]

where ``[value]`` is the grid spacing in degrees as float value between 0.1 and
10.0.  For values smaller and larger than these bounds the value is clamped
before applied and a warning is logged.

**Example** ::

      latlon_grid:
          resolution: 3.5
