"""Shared evaluation membership and date-paired forecast comparisons."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

from stock_forecasting.config import ExperimentConfig
from stock_forecasting.data.dataset import BlockwisePermutationSampler

EVALUATION_PROTOCOL_VERSION = "paired-fixed-holdout-v1"
EVALUATION_SELECTION_BLOCK_SIZE = 128
EVALUATION_SELECTION_SEED = 42


def evaluation_sampler(
    count: int, config: ExperimentConfig, split: str,
) -> BlockwisePermutationSampler:
    if split not in ("validation", "test"):
        raise ValueError("Evaluation membership is only defined for validation and test")
    seed = EVALUATION_SELECTION_SEED if config.data.fixed_split else config.training.seed
    return BlockwisePermutationSampler(
        count, max_samples=config.training.evaluation_max_samples,
        seed=seed + (1 if split == "validation" else 2),
        block_size=EVALUATION_SELECTION_BLOCK_SIZE,
    )


def sample_membership(symbols: list[str], dates: list[str]) -> dict[str, Any]:
    if len(symbols) != len(dates) or not dates:
        raise ValueError("Evaluation symbols and dates must be nonempty and aligned")
    rows = list(zip(symbols, dates, strict=True))
    if len(set(rows)) != len(rows):
        raise ValueError("Evaluation must not contain duplicated symbol-date samples")
    encoded = json.dumps(rows, separators=(",", ":"), ensure_ascii=False).encode()
    return {
        "ordered_symbol_dates_sha256": hashlib.sha256(encoded).hexdigest(),
        "samples": len(rows), "unique_dates": len(set(dates)),
        "cutoff_start": min(dates), "cutoff_end": max(dates),
    }


def daily_normalized_pinball(
    targets: np.ndarray, predictions: np.ndarray, scales: list[float], dates: list[str],
) -> dict[str, float]:
    errors = (targets[..., None] - predictions) / np.asarray(scales)[None, :, None]
    levels = np.asarray([0.1, 0.5, 0.9])
    losses = np.maximum(levels * errors, (levels - 1) * errors).mean(axis=(1, 2))
    unique, inverse = np.unique(dates, return_inverse=True)
    means = np.bincount(inverse, weights=losses) / np.bincount(inverse)
    return {str(day): float(value) for day, value in zip(unique, means, strict=True)}


def paired_block_comparison(
    candidate: dict[str, float], reference: dict[str, float], *,
    block_length: int = 14, replicates: int = 1000, seed: int = 42,
) -> dict[str, Any]:
    """Bootstrap contiguous forecast-date blocks, never individual stock rows.

    The small date-level matrix is vectorized in bounded batches rather than
    spawning processes that would duplicate the larger prediction arrays.
    """

    if set(candidate) != set(reference) or not candidate:
        raise ValueError("Paired comparisons require identical evaluation dates")
    if block_length < 1 or replicates < 100:
        raise ValueError("Invalid block bootstrap configuration")
    dates = sorted(candidate)
    delta = np.asarray([candidate[day] - reference[day] for day in dates])
    if not np.isfinite(delta).all():
        raise ValueError("Paired daily losses must be finite")
    result: dict[str, Any] = {
        "mean_daily_loss_difference": float(delta.mean()),
        "negative_favors_candidate": True,
        "dates": len(dates), "block_length": block_length,
        "replicates": replicates, "seed": seed,
        "method": "circular_moving_block_bootstrap_by_forecast_date",
        "confidence_interval_95": None,
    }
    if len(delta) < 2 * block_length:
        result["status"] = "insufficient_dates_for_block_interval"
        return result
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates)
    blocks = (len(delta) + block_length - 1) // block_length
    for offset in range(0, replicates, 64):
        size = min(64, replicates - offset)
        starts = rng.integers(0, len(delta), size=(size, blocks))
        indices = (starts[..., None] + np.arange(block_length)) % len(delta)
        draws[offset:offset + size] = delta[indices.reshape(size, -1)[:, :len(delta)]].mean(axis=1)
    result["confidence_interval_95"] = np.quantile(draws, [0.025, 0.975]).tolist()
    result["status"] = "estimated_not_multiple_comparison_adjusted"
    return result
