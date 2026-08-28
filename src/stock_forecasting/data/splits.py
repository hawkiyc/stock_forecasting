"""Leakage-aware chronological splits for window records."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, MutableMapping, Sequence
from typing import Any

import pandas as pd

SPLIT_POLICY = "global_chronological_cutoff_with_purge_embargo_and_label_end_guard_v1"


def _label_end_at(record: Mapping[str, Any]) -> pd.Timestamp:
    """Return the latest ground-truth timestamp and reject malformed labels."""

    cutoff = pd.Timestamp(record.get("cutoff_at"))
    label = record.get("label")
    if not isinstance(label, Mapping):
        raise ValueError("Every split candidate must contain a label object")
    end_at = label.get("end_at")
    if not isinstance(end_at, Mapping) or not end_at:
        raise ValueError("Every split candidate label must contain non-empty end_at values")
    try:
        end_timestamps = [pd.Timestamp(value) for value in end_at.values()]
    except (TypeError, ValueError) as error:
        raise ValueError("Split candidate label end_at contains an invalid timestamp") from error
    if pd.isna(cutoff) or any(pd.isna(timestamp) for timestamp in end_timestamps):
        raise ValueError("Split candidate timestamps must not be missing")
    if any(timestamp <= cutoff for timestamp in end_timestamps):
        raise ValueError("Every ground-truth label timestamp must be after cutoff_at")
    return max(end_timestamps)


def _split_summary(records: Sequence[dict[str, Any]], split: str) -> dict[str, Any]:
    selected = [record for record in records if record["split"] == split]
    cutoffs = [pd.Timestamp(record["cutoff_at"]) for record in selected]
    label_ends = [_label_end_at(record) for record in selected]
    return {
        "records": len(selected),
        "cutoff_start_at": min(cutoffs).isoformat(),
        "cutoff_end_at": max(cutoffs).isoformat(),
        "label_end_max_at": max(label_ends).isoformat(),
    }


def chronological_split(
    records: Sequence[dict[str, Any]],
    *,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    purge_bars: int = 20,
    embargo_bars: int = 5,
    audit: MutableMapping[str, Any] | None = None,
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

    assigned: list[dict[str, Any]] = []
    dropped: Counter[str] = Counter()
    train_boundary_at = cutoff_dates[train_boundary]
    validation_boundary_at = cutoff_dates[validation_boundary]
    for record in records:
        cutoff_at = pd.Timestamp(record["cutoff_at"])
        date_index = observed_date_index[cutoff_at]
        label_end_at = _label_end_at(record)
        split: str | None = None
        if date_index < train_stop:
            if label_end_at < train_boundary_at:
                split = "train"
            else:
                dropped["label_crosses_train_boundary"] += 1
        elif validation_start <= date_index < validation_stop:
            if label_end_at < validation_boundary_at:
                split = "validation"
            else:
                dropped["label_crosses_validation_boundary"] += 1
        elif date_index >= test_start:
            split = "test"
        else:
            dropped["purge_or_embargo"] += 1
        if split is not None:
            assigned.append({**record, "split": split})

    counts = Counter(record["split"] for record in assigned)
    if any(counts[name] == 0 for name in ("train", "validation", "test")):
        raise RuntimeError("Chronological split unexpectedly produced an empty split.")
    if max(
        _label_end_at(record) for record in assigned if record["split"] == "train"
    ) >= train_boundary_at:
        raise RuntimeError("Train labels cross the global train boundary")
    if max(
        _label_end_at(record) for record in assigned if record["split"] == "validation"
    ) >= validation_boundary_at:
        raise RuntimeError("Validation labels cross the global validation boundary")
    if audit is not None:
        audit.clear()
        split_summaries = {
            split: _split_summary(assigned, split)
            for split in ("train", "validation", "test")
        }
        audit.update(
            {
                "schema_version": "causal-split-audit-v1",
                "policy": SPLIT_POLICY,
                "comparison": "label.end_at < next_split_start",
                "violations": 0,
                "train_boundary_exclusive": train_boundary_at.isoformat(),
                "validation_boundary_exclusive": validation_boundary_at.isoformat(),
                "validation_start": split_summaries["validation"]["cutoff_start_at"],
                "test_start": split_summaries["test"]["cutoff_start_at"],
                "label_end_counts": {
                    split: split_summaries[split]["records"]
                    for split in ("train", "validation", "test")
                },
                "maximum_label_end": {
                    split: split_summaries[split]["label_end_max_at"]
                    for split in ("train", "validation", "test")
                },
                "dropped_counts_by_reason": dict(sorted(dropped.items())),
                "splits": split_summaries,
            }
        )
    return assigned
