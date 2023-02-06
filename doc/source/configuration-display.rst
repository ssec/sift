.. role:: yaml(code)

Configuring Display Options
---------------------------

Display settings can be configured below the item ``display``.

Geostationary satellite imagery can be displayed in its original projection or
with a projection transformation applied to fit the currently selected map
projection. The former is referred to here as *pixel matrix display*, while the
latter is referred to as *geolocated display*. SIFT implements two kinds of
geolocated display: the so-called *tiled geolocated image display* and the
*simple geolocated image display*.

The kind of display used for image data (data with an *AreaDefinition* in Satpy
terms) is controlled by setting the `display.image_mode` configuration to one of
three options:

- `simple_geolocated`: In this mode the rasterisation of image data to the
  screen pixels is done entirely on the GPU. Technically, the image is placed as
  a texture on a coarser grid (a regular subdivision of a quad). This is then
  re-projected, resulting in the appropriate distortion of the texture to
  resemble the map projection. The accuracy of this kind of projection
  can be controlled by setting the `display.grid_cell_width` and
  `display.grid_cell_height` options, which specify the size in metres of the
  subdivision grid cell containing the sub-satellite point. The smaller the
  value, the finer the grid and the more accurate the projection, but the more
  time and memory consuming the projection will be.

  As long as the projection of the data is exactly the same as the currently
  selected display projection the result is the same as when using the pixel
  matrix display.

- `tiled_geolocated`: The image is broken into smaller pieces (called tiles),
  which are resampled to currently 512x512 pixels each before being passed to
  the GPU, which does the final rasterisation to screen pixels. The advantage
  of this is that images that exceed the texture size limitations of the GPU
  can still be rendered.

- `pixel_matrix`: The image is only scaled and moved when zooming in and out
  or panning, but otherwise no map projection is applied. This ensures that
  the original data can be examined without any resampling artefacts. Changing
  the current map projection has no effect on the visualisation of the data,
  so the coastline and latitude/longitude grid overlay would no longer fit the
  data. Also, loading data with different projections is usually not useful
  with this setting.

Currently the modes are mutually exclusive and only one mode can be used during
the same SIFT session. By default, *simple geolocated image display* with a
subdivision grid resolution of 24000 metres is active, as in this
configuration::

    display:
      image_mode: simple_geolocated
      grid_cell_width: 24000
      grid_cell_height: 24000
