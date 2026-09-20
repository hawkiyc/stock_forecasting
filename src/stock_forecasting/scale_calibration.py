"""Bounded, multiprocessing calibration shared by training and checkpoint reloads."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from torch.utils.data import DataLoader

from stock_forecasting.data.dataset import BlockwisePermutationSampler, LazyFinancialWindowDataset
from stock_forecasting.data.manifest import atomic_write_json, canonical_json_sha256, sha256_file
from stock_forecasting.models.scale_features import (
    EXTENDED_FEATURE_VERSION,
    SCALE_FEATURE_VERSION,
    fit_scale_feature_statistics,
    historical_scale_features,
    validate_scale_feature_statistics,
)
from stock_forecasting.representation_scale_probe import HistoricalScaleDataset


def resolve_scale_feature_statistics(
    dataset: LazyFinancialWindowDataset,
    *,
    sample_count: int,
    seed: int,
    loader_options: dict[str, Any],
    extended: bool = False,
) -> dict[str, Any]:
    """Read only historical inputs; persist aggregate statistics, never window rows."""

    if dataset.split != "train":
        raise ValueError("Numerical feature calibration must use the train split")
    count = min(sample_count, len(dataset))
    if count < 4:
        raise ValueError("Numerical feature calibration requires at least four train rows")
    identity = {
        "version": EXTENDED_FEATURE_VERSION if extended else SCALE_FEATURE_VERSION,
        "bar_store_manifest_sha256": sha256_file(dataset.root / "bar-store.json"),
        "window_size": dataset.window_size,
        "split": "train",
        "sample_count": count,
        "seed": seed,
        "sampler": "blockwise-16-v1",
    }
    namespace = (
        dataset.root.parent.parent
        if dataset.root.parent.name == "prepared"
        else dataset.root.parent
    )
    filename = f"{canonical_json_sha256(identity)}.json"
    path = namespace / "training-cache" / "scale-features" / filename
    if path.is_file():
        cached = validate_scale_feature_statistics(json.loads(path.read_text(encoding="utf-8")))
        if cached["identity"] != identity:
            raise ValueError("Scale-feature cache belongs to a different training dataset")
        print(
            json.dumps(
                {
                    "scale_feature_calibration": "cache_hit",
                    "sample_count": count,
                    "statistics_sha256": cached["sha256"],
                }
            ),
            flush=True,
        )
        return cached
    print(
        json.dumps(
            {
                "scale_feature_calibration": "running",
                "sample_count": count,
                "num_workers": loader_options.get("num_workers", 0),
            }
        ),
        flush=True,
    )
    sampler = BlockwisePermutationSampler(len(dataset), max_samples=count, seed=seed, block_size=16)
    loader = DataLoader(
        HistoricalScaleDataset(dataset),
        batch_size=min(256, count),
        sampler=sorted(sampler),
        pin_memory=False,
        **loader_options,
    )
    # At most 50k x 20 float64 values in production, independent of full dataset size.
    values = np.empty((count, 20 if extended else 8), dtype=np.float64)
    offset = 0
    for batch in loader:
        current = (
            historical_scale_features(
                batch["asset_series"], batch["benchmark_series"], extended=True
            ).numpy()
            if extended
            else batch["scale_targets"].numpy()
        )
        values[offset : offset + len(current)] = current
        offset += len(current)
    if offset != count:
        raise RuntimeError("Scale-feature calibration emitted incomplete sample membership")
    statistics = fit_scale_feature_statistics(values, identity)
    atomic_write_json(path, statistics)
    print(
        json.dumps(
            {
                "scale_feature_calibration": "complete",
                "sample_count": count,
                "statistics_sha256": statistics["sha256"],
            }
        ),
        flush=True,
    )
    return statistics
