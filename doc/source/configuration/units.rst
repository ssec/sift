.. role:: yaml(code)

Configuring Unit Display
------------------------

The unit of measurement for the display of physical values can be configured
under the item ``units``.

Currently, only the units for ``temperature`` calibrations can be configured to
either ``kelvin`` (aliases ``K``, ``Kelvin``) or ``degrees_Celsius`` (aliases
``°C``, ``C``).

The default for all temperatures can be configured using the keyword
``all``. The display setting for specific calibrations can be configured
diffently. Currently known are ``toa_brightnes_temperature`` and
``brightnes_temperature``.

If no configuration is provided every temperature display defaults to ``K``
(``kelvin``).

An exemplary configuration may look as follows::

    units:
      temperature:
        all: kelvin
        toa_brightness_temperature: degrees_Celsius
        brightness_temperature: kelvin
