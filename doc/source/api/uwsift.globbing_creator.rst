GlobbingCreator Documentation
=============================

The *GlobbingCreator* is responsible for creating globbing patterns suitable for collecting files from a directory
with `glob.glob() <https://docs.python.org/3.7/library/glob.html#glob.glob>`_.

Do do this the *GlobbingCreator* takes:

* a MTG-SIFT/satpy/trollsift *filter_pattern* like ``"{platform_name:4s}-{channel:_<6s}-{service:3s}-{start_time:%Y%m%d%H%M%}"``
* a *constraints* dictionary, which is part of a dictionary/query entry of a catalogue configuration associated to a
  ``reader``.

Thus from the following catalogue configuration ::

    catalogue:
      - reader: seviri_l1b_hrit
        search_path: /path/to/seviri/data/
        constraints:
          platform_name: MSG4
          channel:
            - ______
            - IR_108
          start_time:
            type: datetime
            Y: 2019
            m: 12
            d: 31
            H: [0, 6, 12, 18] # equivalent to range(0, 24, 6)

it gets the *constraints* dictionary::

    {
        'platform_name' : "MSG4",
        'channel' : ["______", "IR_108"],
        'start_time' : {
            'type' : "datetime",
            'Y' : 2019,
            'm' : 12,
            'd' : 31,
            'H' : [0, 6, 12, 18]
        }
    }

Expanding *filter_pattern*
--------------------------

First the filter pattern is expanded to become ::

    expanded_filter_pattern = "{platform_name:4s}-{channel:_<6s}-{service:3s}-{start_time_:%Y%}{start_time_m:m%}{start_time_d:d%}{start_time_H:H%}{start_time_M:M%}"

Expanding the *constraints*
---------------------------

The constraints dictionary is expanded to become a list of dictionaries, where each single dictionary contains only
key-value pairs with scalar values (no sequences or mappings). The list of dictionaries contains all combinations which
can be created from the given constraints.

Expanding an entry of ``type: datetime``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The given ``start_time`` configuration represents several actual datetime values because of the sequence given for
``H``. These are (the ``dt_XX`` variables are only for abbreviation to be used later)::

    dt_00 = datetime(2019, 12, 31, hour=0,  tz=timezone.utc)
    dt_06 = datetime(2019, 12, 31, hour=6,  tz=timezone.utc)
    dt_12 = datetime(2019, 12, 31, hour=12, tz=timezone.utc)
    dt_18 = datetime(2019, 12, 31, hour=18, tz=timezone.utc)

Having this a list of *expanded_datetime* dictionaries is generated::

    [{'start_time_Y': dt_00, 'start_time_m': dt_00, 'start_time_d': dt_00, 'start_time_H': dt_00},
     {'start_time_Y': dt_06, 'start_time_m': dt_06, 'start_time_d': dt_06, 'start_time_H': dt_06},
     {'start_time_Y': dt_12, 'start_time_m': dt_12, 'start_time_d': dt_12, 'start_time_H': dt_12},
     {'start_time_Y': dt_18, 'start_time_m': dt_18, 'start_time_d': dt_18, 'start_time_H': dt_18}]

Note, that there are new keys generated, one for each of the datetime format code directives (``%Y``,  ``%m``, ..., see
`datetime / strftime() and strptime() Behavior <https://docs.python.org/3.7/library/datetime.html#strftime-strptime-behavior>`_)
which are given as keys (without the percent sign prefix) in the original *constraints*.

**CAUTION:** Expansion of sequences for ``type: datetime`` constraints is not implemented yet, entries for the datetime
format directives must be single integers for now!

Result of expansion
~~~~~~~~~~~~~~~~~~~

For the given example this *expanded_constraints* list is::

    [{'platform_name': 'MSG4', 'channel': '______', 'start_time_Y': dt_00, 'start_time_m': dt_00, 'start_time_d': dt_00, 'start_time_H': dt_00},
     {'platform_name': 'MSG4', 'channel': '______', 'start_time_Y': dt_06, 'start_time_m': dt_06, 'start_time_d': dt_06, 'start_time_H': dt_06},
     {'platform_name': 'MSG4', 'channel': '______', 'start_time_Y': dt_12, 'start_time_m': dt_12, 'start_time_d': dt_12, 'start_time_H': dt_12},
     {'platform_name': 'MSG4', 'channel': '______', 'start_time_Y': dt_18, 'start_time_m': dt_18, 'start_time_d': dt_18, 'start_time_H': dt_18},
     {'platform_name': 'MSG4', 'channel': 'IR_108', 'start_time_Y': dt_00, 'start_time_m': dt_00, 'start_time_d': dt_00, 'start_time_H': dt_00},
     {'platform_name': 'MSG4', 'channel': 'IR_108', 'start_time_Y': dt_06, 'start_time_m': dt_06, 'start_time_d': dt_06, 'start_time_H': dt_06},
     {'platform_name': 'MSG4', 'channel': 'IR_108', 'start_time_Y': dt_12, 'start_time_m': dt_12, 'start_time_d': dt_12, 'start_time_H': dt_12},
     {'platform_name': 'MSG4', 'channel': 'IR_108', 'start_time_Y': dt_18, 'start_time_m': dt_18, 'start_time_d': dt_18, 'start_time_H': dt_18}]

Expanding an entry of ``type: relative_datetime``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To match the replacement field ``{{start_time:%Y%m%d%H%M%}`` of the *file_pattern* relative to the current time a
different configuration must be given for the constraint ``start_time``::

    start_time:
        type: relative_datetime
        d: [0, -1] # equivalent to range(-2)

From that configuration the following list of *expanded_datetime* dictionaries is generated.::

     [{'start_time_Y': dt_r0, 'start_time_m': dt_r0, 'start_time_d': dt_r0},
      {'start_time_Y': dt_r1, 'start_time_m': dt_r1, 'start_time_d': dt_r1}]

where the ``dt_XX`` variables (used for abbreviation here again) are ::

    now_utc = datetime.now(timezone.utc)
    dt_r0 = now_utc + relativedelta(days=0)
    dt_r1 = now_utc + relativedelta(days=-1)

which means when assuming it is 2020-10-01 12:45:06 UTC now::

    dt_r0 == datetime.fromisoformat("2020-10-01T12:45:06+00:00")
    dt_r1 == datetime.fromisoformat("2020-09-30T12:45:06+00:00")

Note that new keys are generated analogously to the ``type: datetime`` case. For now which of these keys are generated is
computed from the one given key by taking all from the list ``['Y', 'm', 'd', 'H', 'M']`` until before the given one.

This approach is not suitable for all possible datetime-like replacement fields, notably not for the datetime filename
parts of GOES-R data which use day of the year as a zero-padded decimal number (directive ``%j``) or if the year is
represented only with two digits (directive ``%y``) for example. These cases are left for future improvements.

Putting everything together
---------------------------

The wanted globbing patterns are generated by using `trollsift.parser.globify() <https://trollsift.readthedocs.io/en/latest/api.html#trollsift.parser.globify>`_
for the *file_pattern* with each of the dictionaries in *expanded_constraints*.

For the ``type: datetime`` example case this yields::

    MSG4-______-???-2019123100??
    MSG4-______-???-2019123106??
    MSG4-______-???-2019123112??
    MSG4-______-???-2019123118??
    MSG4-IR_108-???-2019123100??
    MSG4-IR_108-???-2019123106??
    MSG4-IR_108-???-2019123112??
    MSG4-IR_108-???-2019123118??

and for the ``type: relative_datetime`` case::

    MSG4-______-???-20200930????
    MSG4-______-???-20201001????
    MSG4-IR_108-???-20200930????
    MSG4-IR_108-???-20201001????

General Note
------------

The current implementation is not robust against bad Catalogue configuration as it doesn't profoundly check for errors
in it.
It should work for correct ones but fail stupidly even without giving any helpful feedback for broken ones, thus the
writer of the configuration is asked to be gracious. Resist from using sequence entries for too many replacement fields
since this would lead to combinatorial explosion (which is *not* retained).

Actually the Catalogue defines kind of a query language which to implement a complete validation for would require
considerable effort.


