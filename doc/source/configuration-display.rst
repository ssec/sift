.. role:: yaml(code)

Configuring Display Options
---------------------------

Display settings can be configured below the item ``display``.

Currently the default display for geostationary satellite images is using a
tiled geolocated image node in the scene graph which supports map
reprojection. To avoid any resampling and reprojection this can be switched
off. The result is called Pixel Matrix display::

    display:
      use_tiled_geolocated_images: [boolean]
