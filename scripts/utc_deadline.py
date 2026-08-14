#!/usr/bin/env python3
"""Convert a positive duration such as 7h into a UTC RFC3339 deadline."""

import argparse
import re
from datetime import datetime, timedelta, timezone


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("duration")
    arguments = parser.parse_args()
    match = re.fullmatch(r"([1-9][0-9]*)([mhd])", arguments.duration)
    if match is None:
        parser.error("duration must use a value such as 30m, 7h, or 2d")
    value = int(match.group(1))
    unit = match.group(2)
    seconds = value * {"m": 60, "h": 3600, "d": 86400}[unit]
    deadline = datetime.now(timezone.utc) + timedelta(seconds=seconds)
    print(deadline.replace(microsecond=0).isoformat().replace("+00:00", "Z"))


if __name__ == "__main__":
    main()
