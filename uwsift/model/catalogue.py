import collections
import logging
import os
from datetime import datetime, timezone
from glob import glob
from typing import Dict, List, Optional, Set, Tuple, Union

import trollsift
from dateutil.relativedelta import relativedelta
from satpy import DataID
from satpy.readers import group_files

from uwsift import config
from uwsift.util.common import create_scenes, is_datetime_format

LOG = logging.getLogger(__name__)


class Catalogue:
    @staticmethod
    def extract_query_parameters(query: dict):
        """
        Extract the values of parameters relevant for a catalogue query
        from the given dictionary *query* and return them as tuple.
        """
        reader = query.get("reader")
        _reader_info = config.get(f"data_reading.{reader}", None)
        filter_patterns = _reader_info.get("filter_patterns")
        group_keys = _reader_info.get("group_keys")

        search_path = query.get("search_path")
        constraints = query.get("constraints")
        products = query.get("products")

        return (reader, search_path, filter_patterns, group_keys, constraints, products)

    @staticmethod
    def query_for_satpy_importer_kwargs_and_readers(
        reader: str, search_path: str, filter_patterns: List[str], group_keys: List[str], constraints: dict, products
    ):
        """
        Create a data catalogue with the given parameters and generate
        importer keywords arguments.
        If an error occurred, it will be caught and the message will be
        logged. If no files were found with the given parameters, then the
        importer keyword arguments won't be created.
        """
        LOG.debug(
            f"Processing query: {reader}, {search_path},"
            f" {filter_patterns}, {group_keys},"
            f" {constraints}, {products}"
        )
        try:
            files = Catalogue.collect_files_for_data_catalogue(search_path, filter_patterns, constraints)
        except Exception as e:
            LOG.error(f"Create data catalogue failed. Error occurred: {e}")
            return None, None

        if not files:
            LOG.info("No files were found for the given query.")
            return None, None

        LOG.info(f"Found files: {files}")

        file_group_map: Optional[dict] = Catalogue.group_files_by_group_keys(files, group_keys, reader)

        return Catalogue._compose_satpy_importer_kwargs(file_group_map, products, reader)

    @staticmethod
    def _compose_satpy_importer_kwargs(
        file_group_map, products: List[dict], reader: str
    ) -> Tuple[Dict[str, Union[str, dict, list]], list]:
        """
        Set up a dictionary which can be used as ***kwargs* in according
        function calls which pass them through to ``SatpyImporter`` for actually
        loading the given *products* for the *file_group_map* using the given
        *reader*.
        """
        scn_mng: SceneManager = SceneManager()
        all_available_products = create_scenes(scn_mng.scenes, file_group_map)
        dataset_ids: List[DataID] = scn_mng.get_data_ids_for_products(all_available_products, products)
        importer_kwargs = {"reader": reader, "scenes": scn_mng.scenes, "dataset_ids": dataset_ids}
        files_to_load: List[str] = [fn for fgroup in file_group_map.values() for fn in fgroup]
        return importer_kwargs, files_to_load

    @staticmethod
    def glob_find_files(patterns: List[str], search_path: str) -> Set[str]:
        """
        Use given globbing *patterns* to find matching files in the directory
        given by *search_path*.
        """
        found_files: List[str] = []
        for p in patterns:
            globbing_pattern_with_path = os.path.join(search_path, p)
            found_files.extend(glob(globbing_pattern_with_path))

        # Make sure there are no duplicates in the result list
        unique_found_files = set(found_files)
        return unique_found_files

    # FIXME refactor.rename/split into call sequence:
    #   pattern = compute_globbing_pattern(...)
    #   glob_find_files(pattern, search_path)
    @staticmethod
    def collect_files_for_data_catalogue(
        search_path: str, filter_patterns: List[str], filter: dict
    ) -> Optional[Set[str]]:
        """
        This method summarize all methods which are needed to create the
        data catalogue. So it regulates the creation.
        """

        # For datetime constraints calculated relative to the current time
        # we need to (re)initialize the GlobbingCreator's understanding of what
        # is ...
        GlobbingCreator.init_now()

        globbing_patterns: Optional[list] = GlobbingCreator.construct_globbing_patterns(filter_patterns, filter)

        if not globbing_patterns:
            return None
        for i in globbing_patterns:
            LOG.debug(f"Globbing pattern: {i}")
        return Catalogue.glob_find_files(globbing_patterns, search_path)

    @staticmethod
    def group_files_by_group_keys(files: Set[str], group_keys: List[str], reader: str) -> Optional[dict]:
        """
        Group given *files* according to the *group_keys* configured for the
        given *reader*.

        A file group contains the name of the reader and the list of those file
        paths in *files* which share the same file name parts identified by the
        group keys.

        The returned dictionary associates each file group to it's group ID.
        The group ID itself is a sorted tuple of all file paths contained in the
        group (the reader is not part of that group ID tuple though).
        """
        if files is None or len(files) == 0:
            return None
        if group_keys is None:
            LOG.debug("No group keys available. Files can't be grouped.")
            return None

        file_groups = group_files(files, reader=reader, group_keys=group_keys)
        # TODO(ar): The following code is borrowed from OpenFileWizard(2)._group_files()
        file_group_map = {}
        for file_group in file_groups:
            # file_group includes what reader to use
            # NOTE: We only allow a single reader at a time
            # TODO(ar) refactor this into a function 'group_id_from_file_group' or so
            group_id = tuple(sorted(fn for group_list in file_group.values() for fn in group_list))
            file_group_map[group_id] = file_group
        return file_group_map


class GlobbingCreator:
    """Create glob patterns from series of constraints.

    The *GlobbingCreator* is responsible for creating globbing patterns suitable for collecting files from a directory
    with `glob.glob() <https://docs.python.org/3.7/library/glob.html#glob.glob>`_.

    To do this the *GlobbingCreator* takes:

    * a MTG-SIFT/satpy/trollsift *filter_pattern* like
      ``"{platform_name:4s}-{channel:_<6s}-{service:3s}-{start_time:%Y%m%d%H%M%}"``
    * a *constraints* dictionary, which is part of a dictionary/query entry of
      a catalogue configuration associated to a ``reader``.

    Thus from the following catalogue configuration::

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
    ^^^^^^^^^^^^^^^^^^^^^^^^^^

    First the filter pattern is expanded to become::

        expanded_filter_pattern = "{platform_name:4s}-{channel:_<6s}-{service:3s}-{start_time_:%Y%}{start_time_m:m%}{start_time_d:d%}{start_time_H:H%}{start_time_M:M%}"

    Expanding the *constraints*
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^

    The constraints dictionary is expanded to become a list of dictionaries,
    where each single dictionary contains only key-value pairs with scalar
    values (no sequences or mappings). The list of dictionaries contains all
    combinations which can be created from the given constraints.

    Expanding an entry of ``type: datetime``
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    The given ``start_time`` configuration represents several actual datetime values because of the sequence given for
    ``H``. These ``dt_XX`` variables are only for abbreviation to be used later::

        dt_00 = datetime(2019, 12, 31, hour=0,  tz=timezone.utc)
        dt_06 = datetime(2019, 12, 31, hour=6,  tz=timezone.utc)
        dt_12 = datetime(2019, 12, 31, hour=12, tz=timezone.utc)
        dt_18 = datetime(2019, 12, 31, hour=18, tz=timezone.utc)

    Having this, a list of *expanded_datetime* dictionaries is generated::

        [{'start_time_Y': dt_00, 'start_time_m': dt_00, 'start_time_d': dt_00, 'start_time_H': dt_00},
         {'start_time_Y': dt_06, 'start_time_m': dt_06, 'start_time_d': dt_06, 'start_time_H': dt_06},
         {'start_time_Y': dt_12, 'start_time_m': dt_12, 'start_time_d': dt_12, 'start_time_H': dt_12},
         {'start_time_Y': dt_18, 'start_time_m': dt_18, 'start_time_d': dt_18, 'start_time_H': dt_18}]

    Note, that there are new keys generated, one for each of the datetime format code directives
    (``%Y``, ``%m``, ..., see `datetime / strftime() and strptime() Behavior
    <https://docs.python.org/3.7/library/datetime.html#strftime-strptime-behavior>`_)
    which are given as keys (without the percent sign prefix) in the original *constraints*.

    **CAUTION:** Expansion of sequences for ``type: datetime`` constraints is
    not implemented yet, entries for the datetime format directives must be single
    integers for now!

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
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^

    The wanted globbing patterns are generated by using
    `trollsift.parser.globify() <https://trollsift.readthedocs.io/en/latest/api.html#trollsift.parser.globify>`_
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
    ^^^^^^^^^^^^

    The current implementation is not robust against bad Catalogue
    configuration as it doesn't profoundly check for errors in it.
    It should work for correct ones but fail stupidly even without giving any
    helpful feedback for broken ones, thus the
    writer of the configuration is asked to be gracious. Resist from using
    sequence entries for too many replacement fields
    since this would lead to combinatorial explosion (which is *not* retained).

    Actually the Catalogue defines kind of a query language which to implement
    a complete validation for would require considerable effort.

    """  # noqa: E501

    # TODO: There are a lot more datetime format codes (like %D and else) than
    #  currently handled => upgrade this class for needed ones case by case

    # TODO make this "private" and add method to re-initialize it
    now_utc: datetime = datetime.now(timezone.utc)

    @staticmethod
    def init_now():
        """
        Initialize the GlobbingCreator so that "now" at the time of the call is
        used as reference for "recent_datetime", i.e., datetime constraints
        relative to current time.
        """
        GlobbingCreator.now_utc = datetime.now(timezone.utc)

    @staticmethod
    def _convert_to_relativedelta(value: int, code: str) -> relativedelta:
        """
        Interpret *value* as relative time delta in the unit given by *code*
        according to the `datetime strftime() and strptime() Format Codes
        <https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes>`_.
        """

        if "S" == code:
            return relativedelta(seconds=value)
        if "M" == code:
            return relativedelta(minutes=value)
        if "H" == code:
            return relativedelta(hours=value)
        if "d" == code:
            return relativedelta(days=value)
        if "m" == code:
            return relativedelta(months=value)
        if "Y" == code:
            return relativedelta(years=value)

        raise ValueError(f"Unknown time format code '{code}'.")

    @staticmethod
    def _expand_datetime_pattern(field_name: str, format_spec: str) -> List[Tuple[str, str]]:
        """
        Get a list of several single-directive datetime patterns made
        from the *field_name* and *format_spec* of one datetime pattern.

        The elements of the returned list are pairs of field name and format
        specification representing datetime replacement fields ("datetime
        patterns"). Each of the pairs references only one of the according
        datetime format codes (also called "directives", see
        https://docs.python.org/3.7/library/datetime.html#strftime-strptime-behavior)
        of the given *format_spec*. The according field names are derived from
        the original *field_name* by appending an underscore ("``_``") and the
        directive letter.

        **Example:**

        Given the field name ``foo`` and format specification ``%Y%m%S``
        (representing a datetime pattern ``{foo:%Y%m%S}``) this function returns
        the list ::

            [ ("foo_Y", "%Y"),
              ("foo_m", "%m"),
              ("foo_S", "%S") ]

        which represents ``{foo_Y:%Y}{foo_m:%m}{foo_s:%s}``.

        **NOTE:** Currently only *pure* format specifications are supported,
        i.e, *format_spec* must consist of directives solely - at least one -
        and must not contain other characters. E.g. for arguments representing
        ``{foo:%Y-%m-%d}`` this function would return a list representing the
        invalid datetime patterns string `{foo_Y-:%Y-}{foo_m-:%m-}{foo_d:%d}``.

        TODO: Support format specifications that contain extra characters.

        **See also:** https://docs.python.org/3/library/string.html#formatstrings
        """
        if not is_datetime_format(format_spec):  # TODO passing this test does not guarantee *pureness*!
            raise ValueError(f"Given format '{format_spec}' is not a recognized datetime format.")

        single_directive_patterns: List[Tuple[str, str]] = []

        # For any pure datetime format specification the call to split("%")
        # will return a list with first entry "" which can be skipped by list
        # splicing:
        codes = format_spec.split("%")[1:]

        for code in codes:
            single_directive_patterns.append((f"{field_name}_{code}", f"%{code}"))

        return single_directive_patterns

    @staticmethod
    def _make_replacement_field(field_name: str, conversion: str, format_spec: str) -> str:
        """
        Build a replacement field from its components *field_name*, *conversion*
        and *format_spec* according to the replacement field grammar, see
        https://docs.python.org/3/library/string.html#formatstrings
        """
        return (
            "{"
            + ("" if not field_name else field_name)
            + ("" if not conversion else f"!{conversion}")
            + ("" if not format_spec else f":{format_spec}")
            + "}"
        )

    @staticmethod
    def _expand_filter_pattern(filter_pattern: str) -> str:
        """
        Get a pattern with all datetime patterns expanded by
        :func:`GlobbingCreator._expand_datetime_pattern`

        **NOTE:** This function is not idempotent, so don't apply it to an
        already expanded filter pattern.

        **SEE:**

        - https://docs.python.org/3/library/string.html#formatstrings

        TODO: Add option to define which of the datetime patterns should be
          expanded since not all may need expansion.
        """

        expanded_filter_pattern_parts: List[str] = []

        for parse_result in trollsift.formatter.parse(filter_pattern):
            literal_text, field_name, format_spec, conversion = parse_result

            replacement_fields_list: List[str] = []

            if not field_name:
                # Nothing to be done here, only literal_text is used below
                pass

            elif not is_datetime_format(format_spec):
                replacement_fields_list.append(
                    GlobbingCreator._make_replacement_field(field_name, conversion, format_spec)
                )
            else:
                # Iterate with 'sdp' (short for [s]ingle [d]irective [p]attern)
                # over the list generated when expanding the current datetime
                # pattern as represented by (field_name, format_spec)
                for sdp in GlobbingCreator._expand_datetime_pattern(field_name, format_spec):
                    sdp_field_name, sdp_format_spec = sdp
                    replacement_fields_list.append(
                        GlobbingCreator._make_replacement_field(sdp_field_name, conversion, sdp_format_spec)
                    )

            replacement_fields = "".join(replacement_fields_list)
            expanded_filter_pattern_parts.append(f"{literal_text}{replacement_fields}")

        return "".join(expanded_filter_pattern_parts)

    @staticmethod
    def _expand_datetime_constraint(field_name, dt_constraints) -> List[dict]:
        """
        Expand the constraint details given by *dt_constraints* as defined for
        a ``datetime`` or ``recent_datetime`` type replacement field
        *field_name* to correspond to the according replacement fields as
        derived by :func:`GlobbingCreator._expand_filter_pattern`

        For details see the documentation for the catalogue constraints
        specification.

        :return: List of "scalar" constraints dictionaries representing all
        combinations which result from the interpretation of the given
        *dt_constraints*
        """
        supported_codes = ["Y", "m", "d", "H", "M", "S"]  # Devel hint: Don't change this to a string.

        dt_constraints_type = dt_constraints.get("type")
        if dt_constraints_type == "datetime":
            return GlobbingCreator._expand_dt_constraints_datetime(dt_constraints, field_name, supported_codes)

        if dt_constraints_type == "recent_datetime":
            return GlobbingCreator._expand_dt_constraints_recent_datetime(dt_constraints, field_name, supported_codes)

        raise ValueError("Invalid datetime constraint type '{dt_constraints_type}'.")

    @staticmethod
    def _expand_dt_constraints_datetime(dt_constraints, field_name, supported_codes) -> List[dict]:
        year = dt_constraints.get("Y", 2000)
        month = dt_constraints.get("m", 1)
        day = dt_constraints.get("d", 1)
        hours = dt_constraints.get("H", 0)
        minutes = dt_constraints.get("M", 0)
        seconds = dt_constraints.get("S", 0)

        # FIXME: catch here for wrong parameters and generate helpful exception to re-raise
        try:
            dt = datetime(year, month, day, hours, minutes, seconds, tzinfo=timezone.utc)
        except TypeError as exc:
            msg = (
                f"Got data incompatible to datetime initialisation"
                f" for constraint '{field_name}'."
                f" Original message: {exc}"
            )
            raise TypeError(msg) from exc

        expanded_dt_constraint = {}
        for code in supported_codes:
            if dt_constraints.get(code):
                expanded_dt_constraint[f"{field_name}_{code}"] = dt
        return [expanded_dt_constraint]

    @staticmethod
    def _expand_dt_constraints_recent_datetime(dt_constraints, field_name, supported_codes) -> List[dict]:
        expanded_dt_constraints: List[dict] = []
        codes_to_set = []
        given_code = None
        sequence_of_given_code = []
        for code in supported_codes:
            codes_to_set.append(code)
            sequence_of_given_code = dt_constraints.get(code)
            if sequence_of_given_code:
                given_code = code
                break

        if not given_code:
            msg = f"No valid time code specification for constraint" f" '{field_name}' given."
            raise ValueError(msg)

        if not isinstance(sequence_of_given_code, collections.abc.Sequence):
            sequence_of_given_code = [sequence_of_given_code]

        for value in sequence_of_given_code:
            try:
                delta_dt = GlobbingCreator._convert_to_relativedelta(value, given_code)
            except TypeError as exc:
                msg = (
                    f"Got incompatible data for constraint"
                    f" '{field_name} / {given_code}'."
                    f" Original message: {exc}"
                )
                raise TypeError(msg) from exc
            except ValueError as exc:
                msg = (
                    f"Got incompatible time format code in constraint"
                    f" '{field_name} / {given_code}'."
                    f" Original message: {exc}"
                )
                raise ValueError(msg)

            dt = GlobbingCreator.now_utc + delta_dt
            expanded_dt_constraint = {}
            for code in codes_to_set:
                expanded_dt_constraint[f"{field_name}_{code}"] = dt

            expanded_dt_constraints.append(expanded_dt_constraint)

        return expanded_dt_constraints

    @staticmethod
    def _expand_to_dict_of_scalars(initial_list_of_dict_of_scalars: List[dict], dict_of_sequences: dict) -> List[dict]:
        """
        Convert the dictionary *dict_of_sequences* with list items to list of
        dictionaries with only scalar items by combination including the given
        *initial_list_of_dict_of_scalars*

        Correctly speaking the items of *dict_of_sequences* don't need to have
        values of type list, but any iterable type (which corresponds to a
        sequence node in YAML). Already scalar valued items are detected and
        treated as single-element lists.

        **NOTE:** String values are treated as scalars (special care is taken
        to not interpret them as iterables).

        **NOTE:** Dictionaries are also iterable with respect to the list of
        their keys. Thus, also items representing YAML mapping nodes, i.e., those
        with dictionary values, are handled, but maybe not as wanted: their
        mapping information will be lost.

        **IMPLEMENTATION DETAIL:** In YAML a string is a *scalar* node but the
        Python YAML reader must map it to a Python ``str``, which is an instance
        of ``collection.abc.Sequence`` and therefore iterable. A naive
        implementation carelessly iterating over it would split the string
        into pieces, which must be avoided.
        """

        # FIXME beware of combinatorial explosion
        # FIXME make this a generator to avoid to actually create the list of
        #  dictionaries?

        list_of_dicts_of_scalars = initial_list_of_dict_of_scalars

        # Guarantee that we have an initial element to "iterate"
        if not list_of_dicts_of_scalars:
            list_of_dicts_of_scalars = [{}]

        for key, value in dict_of_sequences.items():
            new_list_of_dicts_of_scalars = []
            for dict_of_scalars in list_of_dicts_of_scalars:
                if not isinstance(value, collections.abc.Iterable) or isinstance(value, str):
                    new_dict_of_scalars = dict_of_scalars.copy()
                    new_dict_of_scalars[key] = value
                    new_list_of_dicts_of_scalars.append(new_dict_of_scalars)
                else:
                    for scalar in value:
                        new_dict_of_scalars = dict_of_scalars.copy()
                        new_dict_of_scalars[key] = scalar
                        new_list_of_dicts_of_scalars.append(new_dict_of_scalars)
            list_of_dicts_of_scalars = new_list_of_dicts_of_scalars

        return list_of_dicts_of_scalars

    @staticmethod
    def _expand_constraints(constraints: dict) -> List[dict]:
        """
        Expand the given *constraints* dictionary to a list of dictionaries,
        where each item contains only "scalar" values.

        Special treatment is applied to items, which are marked as of
        "type: datetime" or "type: recent_datetime". Their sub-structure is
        interpreted to contribute to the list as groups of scalar items (for
        details see documentation for the constraints section of the catalogue
        configuration).

        **IMPLEMENTATION DETAILS:**

        Nomenclature:

        - a "normalized" prefix denotes a constraint dictionaries, where all
          entry items have only *scalar* or *sequence* values, *no mappings*.
        - an "expanded" prefix denotes a *list* of constraint dictionaries,
          where all entry items have only items with *scalar*, any sequences are
          resolved to scalars by creating a list of all combinations.

        The implementation covers subtle details, handle with care when changing
        things.
        """
        normalized_constraints: dict = {}
        expanded_datetime_constraints: List[dict] = [{}]
        have_datetime_constraint_already = False

        for key, value in constraints.items():
            # Classify all constraint configs ...

            if not isinstance(value, collections.abc.Mapping):
                # Now accept all left over: This may be coming from a YAML
                # scalar or a sequence
                normalized_constraints[key] = value
                continue

            # ... from those which deal with a datetime.
            _type = value.get("type", None)
            if _type == "datetime" or _type == "recent_datetime":
                # TODO: Since (for now) we cannot deal with more than one of
                #  them, ignore all but the first ...
                if have_datetime_constraint_already:
                    LOG.warning(f"Skipping datetime constraint '{key}'" f" because it is not the first one.")
                    continue

                try:
                    expanded_datetime_constraints = GlobbingCreator._expand_datetime_constraint(key, value)
                    have_datetime_constraint_already = True
                except (ValueError, TypeError) as exc:
                    LOG.warning(exc)

            else:
                # No other 'type' is handled (yet)
                LOG.warning(f"Ignoring constraint '{key}'" f" of unknown type '{_type}'")

        # Now create a list of constraints (dictionaries) which
        # are the combination of the variants for each constraint.
        return GlobbingCreator._expand_to_dict_of_scalars(expanded_datetime_constraints, normalized_constraints)

    @staticmethod
    def construct_globbing_patterns(filter_patterns: List[str], constraints: dict) -> List[str]:
        """Construct a list of globbing patterns from the given *filter_patterns* with the given *constraints* applied.

        Returns: a list of strings, each usable as parameter for glob.glob()
        """
        globbing_patterns: List[str] = []

        expanded_constraints: List[dict] = GlobbingCreator._expand_constraints(constraints)

        for filter_pattern in filter_patterns:
            expanded_filter_pattern = GlobbingCreator._expand_filter_pattern(filter_pattern)

            for expanded_constraint in expanded_constraints:
                globbing_patterns.append(trollsift.globify(expanded_filter_pattern, expanded_constraint))

        return globbing_patterns


class SceneManager:
    """The (future) purpose of this class is to keep information about already seen Satpy Scenes.

    Satpy Scenes are in a way collections of files as well as the information
    which products can be "made" from them.

    TODO: This purpose may overlap with similar task elsewhere implemented in
      SIFT already, check this

    TODO Adopt the function create_scenes()...
    """

    def __init__(self):
        self.scenes = {}

    def get_data_ids_for_products(self, all_available_data_ids, products) -> List[DataID]:
        """
        Look up DataIDs of *products* in *all_available_data_ids*

        TODO: Notify about products for which no DataID was found
        """
        products_data_ids = []
        for data_id in all_available_data_ids:
            for channel, calibrations in products.items():
                if data_id.get("name") == channel and data_id.get("calibration").name in calibrations:
                    products_data_ids.append(data_id)

        return products_data_ids


if __name__ == "__main__":
    catalogue_config = config.get("catalogue", None)
    first_query = catalogue_config[0]

    (reader, search_path, filter_patterns, group_keys, constraints, products) = Catalogue.extract_query_parameters(
        first_query
    )

    (importer_kwargs, files_to_load) = Catalogue.query_for_satpy_importer_kwargs_and_readers(
        reader, search_path, filter_patterns, group_keys, constraints, products
    )

    print(importer_kwargs)
