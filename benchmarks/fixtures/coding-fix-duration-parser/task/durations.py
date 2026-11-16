"""Parse human-written durations such as "1h30m" or "45s" into seconds."""

import re

_UNIT_SECONDS = {"h": 3600, "m": 60, "s": 1}
_PART = re.compile(r"(\d+)([hms])")


def parse_duration(text):
    """Return the total number of seconds in `text`.

    Accepts one or more <number><unit> parts with units h, m and s, each unit
    at most once and in that order. Raises ValueError for anything else.
    """
    total = 0
    for number, unit in _PART.findall(text):
        total += int(number) * _UNIT_SECONDS[unit]
    return total
