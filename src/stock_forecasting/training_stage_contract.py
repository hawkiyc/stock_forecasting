"""Dependency-light sample contracts for production training stages."""

from __future__ import annotations

STAGE1_TRAIN_FRACTION = 0.05
STAGE1_MAX_SAMPLES = 500_000
STAGE2_TRAIN_FRACTION = 1.0
STAGE2_MAX_SAMPLES = None

PRODUCTION_STAGE_SAMPLE_CONTRACTS: dict[str, dict[str, float | int | None]] = {
    "stage1": {
        "train_fraction": STAGE1_TRAIN_FRACTION,
        "max_samples": STAGE1_MAX_SAMPLES,
    },
    "stage2": {
        "train_fraction": STAGE2_TRAIN_FRACTION,
        "max_samples": STAGE2_MAX_SAMPLES,
    },
}

