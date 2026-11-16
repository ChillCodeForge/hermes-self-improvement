"""Parse human-written durations such as "1h30m" or "45s" into seconds."""

import re

_DURATION = re.compile(r"(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?")


def parse_duration(text):
    """Return the total number of seconds in `text`.

    Accepts one or more <number><unit> parts with units h, m and s, each unit
    at most once and in that order. Raises ValueError for anything else.
    """
    match = _DURATION.fullmatch(text)
    if not text or match is None:
        raise ValueError(f"invalid duration: {text!r}")
    hours, minutes, seconds = (int(value) if value else 0 for value in match.groups())
    return hours * 3600 + minutes * 60 + seconds
