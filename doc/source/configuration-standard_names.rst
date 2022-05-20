.. role:: yaml(code)

Configuring Display Names For Layer Manager
-------------------------------------------

The ``STANDARD_NAME`` of many products coming from Satpy is too long to display
well in the Layer Manager column *Name*. Using the configration setting
``standard_names`` it is possible to define shorter names for display::

    standard_names:
      '[STANDARD NAME 1]': [DISPLAY NAME 1]
      '[STANDARD NAME 2]': [DISPLAY NAME 2]

For example, with the following configuration for ``reflectance_mean_all`` the
displayed name will be *REFL (mean, all)*, for
``toa_outgoing_radiance_per_unit_wavelength`` it will be *RAD*::

    standard_names:
      'toa_outgoing_radiance_per_unit_wavelength': RAD
      'reflectance_mean_all': REFL (mean, all)
