import datetime

import pytest


def test_single_directive_patterns(globbing_creator, field_name, format_spec):
    sdp = globbing_creator._expand_datetime_pattern(field_name, format_spec)
    assert sdp == [
        ("start_time_Y", "%Y"),
        ("start_time_m", "%m"),
        ("start_time_d", "%d"),
        ("start_time_H", "%H"),
        ("start_time_M", "%M"),
        ("start_time_S", "%S"),
    ]


def test_expand_filter_pattern(globbing_creator, filter_patterns):
    efp = globbing_creator._expand_filter_pattern(filter_patterns[0])
    assert efp == (
        "A-{undef}-{platform_name:4s}-{channel:_<6s}-{undef}-{service:3s}"
        "-{start_time_Y:%Y}{start_time_m:%m}{start_time_d:%d}{start_time_H:%H}{start_time_M:%M}-B"
        "-{end_time_Y:%Y}{end_time_m:%m}{end_time_d:%d}{end_time_H:%H}{end_time_M:%M}-C"
    )


@pytest.mark.parametrize("constraints", ["constraints_absolute", "constraints_relative"])
def test_expand_datetime_constraint(globbing_creator, constraints, request):
    if constraints == "constraints_absolute":
        replacement_field = "end_time"
        expected_result = [
            {
                "end_time_H": datetime.datetime(2019, 12, 31, 12, 0, tzinfo=datetime.timezone.utc),
                "end_time_Y": datetime.datetime(2019, 12, 31, 12, 0, tzinfo=datetime.timezone.utc),
                "end_time_d": datetime.datetime(2019, 12, 31, 12, 0, tzinfo=datetime.timezone.utc),
                "end_time_m": datetime.datetime(2019, 12, 31, 12, 0, tzinfo=datetime.timezone.utc),
            }
        ]
    elif constraints == "constraints_relative":
        replacement_field = "start_time"
        expected_result = [
            {
                "start_time_H": datetime.datetime(2000, 1, 1, 0, 0, 0),
                "start_time_Y": datetime.datetime(2000, 1, 1, 0, 0, 0),
                "start_time_d": datetime.datetime(2000, 1, 1, 0, 0, 0),
                "start_time_m": datetime.datetime(2000, 1, 1, 0, 0, 0),
            },
            {
                "start_time_H": datetime.datetime(1999, 12, 31, 23, 0, 0),
                "start_time_Y": datetime.datetime(1999, 12, 31, 23, 0, 0),
                "start_time_d": datetime.datetime(1999, 12, 31, 23, 0, 0),
                "start_time_m": datetime.datetime(1999, 12, 31, 23, 0, 0),
            },
        ]
    constraints = request.getfixturevalue(constraints)

    edc = globbing_creator._expand_datetime_constraint(replacement_field, constraints.get(replacement_field))
    assert edc == expected_result


@pytest.mark.parametrize("constraints", ["constraints_absolute", "constraints_relative"])
def test_expand_constraints(globbing_creator, constraints, request):
    if constraints == "constraints_absolute":
        expected_result = [
            {
                "channel": "______",
                "platform_name": "MSG4",
                "end_time_H": datetime.datetime(2019, 12, 31, 12, 0, tzinfo=datetime.timezone.utc),
                "end_time_Y": datetime.datetime(2019, 12, 31, 12, 0, tzinfo=datetime.timezone.utc),
                "end_time_d": datetime.datetime(2019, 12, 31, 12, 0, tzinfo=datetime.timezone.utc),
                "end_time_m": datetime.datetime(2019, 12, 31, 12, 0, tzinfo=datetime.timezone.utc),
            },
            {
                "channel": "IR_108",
                "platform_name": "MSG4",
                "end_time_H": datetime.datetime(2019, 12, 31, 12, 0, tzinfo=datetime.timezone.utc),
                "end_time_Y": datetime.datetime(2019, 12, 31, 12, 0, tzinfo=datetime.timezone.utc),
                "end_time_d": datetime.datetime(2019, 12, 31, 12, 0, tzinfo=datetime.timezone.utc),
                "end_time_m": datetime.datetime(2019, 12, 31, 12, 0, tzinfo=datetime.timezone.utc),
            },
        ]
    elif constraints == "constraints_relative":
        expected_result = [
            {
                "channel": "______",
                "platform_name": "MSG4",
                "start_time_H": datetime.datetime(2000, 1, 1, 0, 0, 0),
                "start_time_Y": datetime.datetime(2000, 1, 1, 0, 0, 0),
                "start_time_d": datetime.datetime(2000, 1, 1, 0, 0, 0),
                "start_time_m": datetime.datetime(2000, 1, 1, 0, 0, 0),
            },
            {
                "channel": "IR_108",
                "platform_name": "MSG4",
                "start_time_H": datetime.datetime(2000, 1, 1, 0, 0, 0),
                "start_time_Y": datetime.datetime(2000, 1, 1, 0, 0, 0),
                "start_time_d": datetime.datetime(2000, 1, 1, 0, 0, 0),
                "start_time_m": datetime.datetime(2000, 1, 1, 0, 0, 0),
            },
            {
                "channel": "______",
                "platform_name": "MSG4",
                "start_time_H": datetime.datetime(1999, 12, 31, 23, 0, 0),
                "start_time_Y": datetime.datetime(1999, 12, 31, 23, 0, 0),
                "start_time_d": datetime.datetime(1999, 12, 31, 23, 0, 0),
                "start_time_m": datetime.datetime(1999, 12, 31, 23, 0, 0),
            },
            {
                "channel": "IR_108",
                "platform_name": "MSG4",
                "start_time_H": datetime.datetime(1999, 12, 31, 23, 0, 0),
                "start_time_Y": datetime.datetime(1999, 12, 31, 23, 0, 0),
                "start_time_d": datetime.datetime(1999, 12, 31, 23, 0, 0),
                "start_time_m": datetime.datetime(1999, 12, 31, 23, 0, 0),
            },
        ]
    constraints = request.getfixturevalue(constraints)
    ec_abs = globbing_creator._expand_constraints(constraints)
    assert ec_abs == expected_result


@pytest.mark.parametrize("constraints", ["constraints_absolute", "constraints_relative"])
def test_construct_globbing_patterns(globbing_creator, filter_patterns, constraints, request):
    if constraints == "constraints_absolute":
        expected_result = [
            "A-*-MSG4-______-*-???-????????????-B-2019123112??-C",
            "A-*-MSG4-IR_108-*-???-????????????-B-2019123112??-C",
        ]
    elif constraints == "constraints_relative":
        expected_result = [
            "A-*-MSG4-______-*-???-2000010100??-B-????????????-C",
            "A-*-MSG4-IR_108-*-???-2000010100??-B-????????????-C",
            "A-*-MSG4-______-*-???-1999123123??-B-????????????-C",
            "A-*-MSG4-IR_108-*-???-1999123123??-B-????????????-C",
        ]
    constraints = request.getfixturevalue(constraints)
    gp_abs = globbing_creator.construct_globbing_patterns(filter_patterns, constraints)
    assert gp_abs == expected_result
