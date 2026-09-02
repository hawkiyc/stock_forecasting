"""Dependency-free contracts for durable dataset storage identity.

The dataset namespace must change only when acquisition inputs or persisted
bars/cutoff ranges can change. Training targets, runtime resources, retry
policy, logging, lifecycle state, and descriptive provenance stay outside this
contract.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

DATASET_REQUEST_SCHEMA_VERSION: Final = 3
PREPARATION_PROVENANCE_SCHEMA_VERSION: Final = 5
BAR_STORE_SCHEMA_VERSION: Final = "1.0"
BAR_STORE_KIND: Final = "symbol-oriented-ohlcv-bar-store"
BAR_STORE_BUILD_CHECKPOINT_SCHEMA_VERSION: Final = "1.0"

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
    return {
        field: payload[field]
        for field in DATASET_STORAGE_PREPARATION_FIELDS
    }


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
