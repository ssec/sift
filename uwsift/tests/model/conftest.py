import datetime

import pytest

from uwsift.model.catalogue import GlobbingCreator


@pytest.fixture(scope="session")
def globbing_creator():
    GlobbingCreator.now_utc = datetime.datetime(2000, 1, 1, 0, 0, 0)
    return GlobbingCreator


@pytest.fixture
def filter_patterns():
    return [
        "A-{undef}-{platform_name:4s}-{channel:_<6s}-{undef}-{service:3s}"
        "-{start_time:%Y%m%d%H%M}-B-{end_time:%Y%m%d%H%M}-C"
    ]


@pytest.fixture
def constraints():
    """
    constraints = constraints_absolute
    """
    return {
        "platform_name": "MSG4",
        "channel": ["______", "IR_108"],
        "end_time": {
            "type": "datetime",
            "Y": 2019,
            "m": 12,
            "d": 31,
            "H": 12,  # Future: [3,9,15,21]
        },
    }


constraints_absolute = constraints


@pytest.fixture
def constraints_relative():
    return {
        "platform_name": "MSG4",
        "channel": ["______", "IR_108"],
        "start_time": {
            "type": "recent_datetime",
            "H": [0, -1],
        },
    }


@pytest.fixture
def field_name():
    return "start_time"


@pytest.fixture
def format_spec():
    return "%Y%m%d%H%M%S"
