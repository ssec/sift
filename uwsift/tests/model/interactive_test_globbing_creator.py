#!/usr/bin/env python3

# This script is suitable to do some quick interactive testing especially for
# the Catalogue's GlobbingCreator

import logging
from pprint import pprint

from uwsift.model.catalogue import GlobbingCreator

LOG = logging.getLogger(__name__)

filter_patterns = [
    "A-{platform_name:4s}-{channel:_<6s}-B_-{undefined}-"
    "{service:3s}-{start_time:%Y%m%d%H%M}-CCC-{end_time:%Y%m%d%H%M}-D_"
]

constraints = constraints_absolute = {
    "platform_name": "MSG4",
    "channel": ["______", "IR_108"],
    "start_time": {
        "type": "datetime",
        "Y": 2019,
        "m": 12,
        "d": 31,
        "H": 12,  # [3, 9, 15, 21] # 12 # Future:
    },
}

constraints_relative = {
    "platform_name": "MSG4",
    "channel": ["______", "IR_108"],
    "start_time": {
        "type": "recent_datetime",
        "H": [-1],
    },
}

print("\n------------------------------------------------------------------")
GlobbingCreator.init_now()

print("\n------------------------------------------------------------------")

field_name = "start_time"
format_spec = "%Y%m%d%H%M%S"

print("Single Directive Patterns:")
sdp = GlobbingCreator._expand_datetime_pattern(field_name, format_spec)
pprint(sdp)

print("\n------------------------------------------------------------------")

print("Expand Filter Pattern:")
print(filter_patterns[0])
efp = GlobbingCreator._expand_filter_pattern(filter_patterns[0])
pprint(efp)

print("\n=====================================================================")
print("Expand Datetime 'start_time Constraint, type: datetime:")
print(constraints_absolute.get("start_time"))
edc_abs = GlobbingCreator._expand_datetime_constraint("start_time", constraints_absolute.get("start_time"))
pprint(edc_abs)

print("\n----------------------------------------------------------------")
print("Expand Constraints, type: datetime:")
try:
    ec_abs = GlobbingCreator._expand_constraints(constraints_absolute)
    pprint(ec_abs)
except (ValueError, TypeError) as exc:
    LOG.warning(exc)

print("\n----------------------------------------------------------------")
print("Globbing Patterns, type: datetime:")
gp_abs = GlobbingCreator.construct_globbing_patterns(filter_patterns, constraints_absolute)
pprint(gp_abs)

print("\n=====================================================================")
print(f"Now: {GlobbingCreator.now_utc}")
print("----------------------------------------------------------------")

print("Expand Datetime 'start_time Constraint, type: recent_datetime:")
print(constraints_relative.get("start_time"))
try:
    edc_rel = GlobbingCreator._expand_datetime_constraint("start_time", constraints_relative.get("start_time"))
    pprint(edc_rel)
except (ValueError, TypeError) as exc:
    LOG.warning(exc)

print("\n----------------------------------------------------------------")
print("Expand Constraints, type: recent_datetime:")
ec_rel = GlobbingCreator._expand_constraints(constraints_relative)
pprint(ec_rel)

print("\n----------------------------------------------------------------")
print("Globbing Patterns, type: recent_datetime:")
gp_rel = GlobbingCreator.construct_globbing_patterns(filter_patterns, constraints_relative)
pprint(gp_rel)
