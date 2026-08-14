"""Leakage-aware chronological splits for window records."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

import pandas as pd


def chronological_split(
    records: Sequence[dict[str, Any]],
    *,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    purge_bars: int = 20,
    embargo_bars: int = 5,
) -> list[dict[str, Any]]:
    """Assign train/validation/test using global cutoff dates.

    Purge bars are removed before each boundary and embargo bars after each
    boundary on the observed trading calendar, so their meaning is independent
    of sample stride. Global dates prevent the same date from crossing splits.
    """

    if not records:
        raise ValueError("Cannot split an empty record collection.")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1.")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1.")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("train_fraction + validation_fraction must be less than 1.")
    if purge_bars < 0 or embargo_bars < 0:
        raise ValueError("purge_bars and embargo_bars must be non-negative.")

    cutoff_dates = sorted({pd.Timestamp(record["cutoff_at"]) for record in records})
    observed_dates = sorted(
        {
            pd.Timestamp(timestamp)
            for record in records
            for timestamp in record["context"]["timestamp"]
        }
    )
    observed_date_index = {date: index for index, date in enumerate(observed_dates)}
    count = len(cutoff_dates)
    train_boundary = int(count * train_fraction)
    validation_boundary = int(count * (train_fraction + validation_fraction))
    train_boundary_index = observed_date_index[cutoff_dates[train_boundary]]
    validation_boundary_index = observed_date_index[cutoff_dates[validation_boundary]]
    train_stop = train_boundary_index - purge_bars
    validation_start = train_boundary_index + embargo_bars
    validation_stop = validation_boundary_index - purge_bars
    test_start = validation_boundary_index + embargo_bars
    if min(train_stop, validation_stop - validation_start, len(observed_dates) - test_start) <= 0:
        raise ValueError(
            "Not enough unique cutoff dates for non-empty purged splits. "
            "Use more data or smaller purge/embargo values."
        )

    date_to_split: dict[pd.Timestamp, str] = {}
    for date in cutoff_dates:
        date_index = observed_date_index[date]
        if date_index < train_stop:
            date_to_split[date] = "train"
        elif validation_start <= date_index < validation_stop:
            date_to_split[date] = "validation"
        elif date_index >= test_start:
            date_to_split[date] = "test"

    assigned: list[dict[str, Any]] = []
    for record in records:
        split = date_to_split.get(pd.Timestamp(record["cutoff_at"]))
        if split is not None:
            assigned.append({**record, "split": split})

    counts = Counter(record["split"] for record in assigned)
    if any(counts[name] == 0 for name in ("train", "validation", "test")):
        raise RuntimeError("Chronological split unexpectedly produced an empty split.")
    return assigned
