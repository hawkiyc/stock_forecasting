"""Dependency-free contracts for durable dataset storage identity.

The dataset namespace must change only when acquisition inputs or persisted
bars/cutoff ranges can change. Training targets, runtime resources, retry
policy, logging, lifecycle state, and descriptive provenance stay outside this
contract.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime, timezone
from typing import Any, Final

DATASET_REQUEST_SCHEMA_VERSION: Final = 3
PREPARATION_PROVENANCE_SCHEMA_VERSION: Final = 5
BAR_STORE_SCHEMA_VERSION: Final = "1.0"
BAR_STORE_KIND: Final = "symbol-oriented-ohlcv-bar-store"
BAR_STORE_BUILD_CHECKPOINT_SCHEMA_VERSION: Final = "1.0"
FIXED_SPLIT_POLICY: Final = "fixed_dates_with_exact_label_end_guard_v1"
MIN_FIXED_EVALUATION_DATES: Final = 80
FIXED_EVALUATION_SPLIT: Final[dict[str, str]] = {
    "train_end": "2025-06-01",
    "validation_end": "2025-12-01",
    "test_end": "2026-06-01",
}


def validated_fixed_split(payload: Any) -> dict[str, str] | None:
    """Validate exclusive UTC date boundaries without importing runtime dependencies."""

    if payload is None:
        return None
    if not isinstance(payload, Mapping) or set(payload) != set(FIXED_EVALUATION_SPLIT):
        raise ValueError("Fixed split requires train_end, validation_end, and test_end")
    values = [payload[name] for name in FIXED_EVALUATION_SPLIT]
    if any(not isinstance(value, str) for value in values):
        raise ValueError("Fixed split dates must use YYYY-MM-DD strings")
    dates = [date.fromisoformat(value) for value in values]
    if any(value.isoformat() != values[index] for index, value in enumerate(dates)):
        raise ValueError("Fixed split dates must use YYYY-MM-DD strings")
    if not dates[0] < dates[1] < dates[2]:
        raise ValueError("Fixed split boundaries must be strictly increasing")
    return {name: values[index] for index, name in enumerate(FIXED_EVALUATION_SPLIT)}


def validate_fixed_split_audit(
    audit: Any, fixed_split: Any, *, minimum_evaluation_dates: int = 0,
) -> None:
    """Fail before GPU admission if exclusive boundaries or date coverage disagree."""

    fixed = validated_fixed_split(fixed_split)
    if fixed is None:
        return
    if not isinstance(audit, Mapping) or audit.get("fixed_split") != fixed:
        raise ValueError("Fixed-split audit does not match the requested dates")
    if audit.get("policy") != FIXED_SPLIT_POLICY:
        raise ValueError("Fixed-split audit uses a different boundary policy")
    boundaries = {
        split: datetime.fromisoformat(fixed[field]).replace(tzinfo=timezone.utc)
        for split, field in (
            ("train", "train_end"), ("validation", "validation_end"), ("test", "test_end")
        )
    }
    lower = None
    for split, upper in boundaries.items():
        boundary_field = f"{split}_boundary_exclusive"
        if audit.get(boundary_field) != upper.isoformat():
            raise ValueError(f"Fixed-split audit {boundary_field} is inconsistent")
        summaries = audit.get("splits")
        if not isinstance(summaries, Mapping) or not isinstance(summaries.get(split), Mapping):
            raise ValueError("Fixed-split audit has invalid split summaries")
        summary = summaries[split]
        first = datetime.fromisoformat(str(summary.get("cutoff_start_at", "")))
        last = datetime.fromisoformat(str(summary.get("cutoff_end_at", "")))
        label_end = datetime.fromisoformat(str(summary.get("label_end_max_at", "")))
        if any(value.tzinfo is None for value in (first, last, label_end)):
            raise ValueError("Fixed-split audit timestamps must include timezones")
        if not first <= last < label_end < upper or (lower is not None and first < lower):
            raise ValueError(f"Fixed-split {split} samples or labels cross their boundaries")
        lower = upper
    by_market = audit.get("dates_by_market")
    if not isinstance(by_market, Mapping) or not by_market:
        raise ValueError("Fixed-split audit has no per-market effective-date coverage")
    for market, splits in by_market.items():
        if not isinstance(splits, Mapping):
            raise ValueError("Fixed-split market coverage must contain split mappings")
        for split in ("validation", "test"):
            entry = splits.get(split, {})
            if not isinstance(entry, Mapping):
                raise ValueError("Fixed-split date coverage must be a mapping")
            dates = entry.get("cutoff_dates", [])
            if (
                not isinstance(dates, list) or any(not isinstance(day, str) for day in dates)
                or dates != sorted(set(dates))
            ):
                raise ValueError("Effective evaluation dates must be sorted and unique")
            start = fixed["train_end" if split == "validation" else "validation_end"]
            end = fixed["validation_end" if split == "validation" else "test_end"]
            if any(
                not start <= day < end or date.fromisoformat(day).isoformat() != day
                for day in dates
            ):
                raise ValueError("Effective evaluation dates cross the fixed boundaries")
            if entry.get("unique_cutoff_count") != len(dates):
                raise ValueError("Effective evaluation date count is inconsistent")
            if len(dates) < minimum_evaluation_dates:
                raise ValueError(
                    f"{market} {split} has only {len(dates)} effective dates; "
                    f"at least {minimum_evaluation_dates} are required before GPU training"
                )


DATASET_STORAGE_PREPARATION_FIELDS: Final[tuple[str, ...]] = (
    "bar_store_schema_version",
    "storage_kind",
    "window_size",
    "max_horizon",
    "window_materialized",
    "labels_materialized",
    "benchmark_mapping_sha256",
    "max_abs_log_return",
    "train_fraction",
    "validation_fraction",
    "purge_bars",
    "effective_embargo_bars",
)

DEFAULT_DATASET_STORAGE_PREPARATION: Final[dict[str, Any]] = {
    "bar_store_schema_version": BAR_STORE_SCHEMA_VERSION,
    "storage_kind": BAR_STORE_KIND,
    "window_size": 128,
    "max_horizon": 14,
    "window_materialized": False,
    "labels_materialized": False,
    "benchmark_mapping_sha256": (
        "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"
    ),
    "max_abs_log_return": 0.5,
    "train_fraction": 0.70,
    "validation_fraction": 0.15,
    "purge_bars": 20,
    "effective_embargo_bars": 14,
}


def storage_preparation_spec(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Select values that alter persisted bars or lazy cutoff ranges."""

    missing = [
        field
        for field in DATASET_STORAGE_PREPARATION_FIELDS
        if field not in payload
    ]
    if missing:
        raise ValueError(
            "Dataset preparation provenance is missing storage fields: "
            + ", ".join(missing)
        )
    selected = {
        field: payload[field]
        for field in DATASET_STORAGE_PREPARATION_FIELDS
    }
    fixed_split = validated_fixed_split(payload.get("fixed_split"))
    if fixed_split is not None:
        selected["fixed_split"] = fixed_split
    return selected


def dataset_request_identity_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return the immutable acquisition and storage request identity."""

    preparation = payload.get("preparation")
    storage_preparation: Any = preparation
    if isinstance(preparation, Mapping):
        storage_preparation = storage_preparation_spec(preparation)
    selected_datasets = payload.get("selected_datasets")
    if isinstance(selected_datasets, (list, tuple)) and all(
        isinstance(dataset, str) for dataset in selected_datasets
    ):
        selected_datasets = sorted(set(selected_datasets))
    return {
        "profile": payload.get("profile"),
        "revision": payload.get("revision"),
        "selected_datasets": selected_datasets,
        "date_range": payload.get("date_range"),
        "universe": payload.get("universe"),
        "storage_preparation": storage_preparation,
    }
