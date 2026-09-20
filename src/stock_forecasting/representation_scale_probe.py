"""Frozen-checkpoint probes for past-only scale information, independent of training."""

from __future__ import annotations

import logging
import math
from collections import Counter
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset

from stock_forecasting.data.dataset import (
    LazyFinancialWindowDataset,
    _series_from_asof_adjusted_frame,
    _timestamp_features,
)
from stock_forecasting.models.outputs import QuantForecastOutput
from stock_forecasting.models.ranking import market_ids

LOGGER = logging.getLogger(__name__)
TARGET_NAMES = (
    *(
        f"{stream}_log_return_std_{window}d"
        for window in (20, 60)
        for stream in ("asset", "benchmark", "relative")
    ),
    "asset_close_cv_context",
    "benchmark_close_cv_context",
)
READOUT_DESCRIPTIONS = {
    "kronos_pair_mean": "Concatenated masked time means of asset and benchmark hidden states.",
    "kronos_pair_last": "Concatenated last valid asset and benchmark hidden states.",
    "resampler_pair_mean": "Concatenated latent-token means of asset and benchmark streams.",
    "conditioned_mean": "Mean conditioned latents: the exact pooled input to the alpha head.",
}


@dataclass(frozen=True)
class ScaleProbeSettings:
    """Bound extraction cost and fix the probe before viewing validation scores."""

    train_samples: int = 16_384
    validation_samples: int = 4_096
    batch_size: int = 16
    num_workers: int = 0
    ridge_alpha: float = 10.0
    seed: int = 42

    def __post_init__(self) -> None:
        for name in ("train_samples", "validation_samples", "batch_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.train_samples < 2 or self.validation_samples < 2:
            raise ValueError("Each probe split needs at least two samples")
        if (
            isinstance(self.num_workers, bool)
            or not isinstance(self.num_workers, int)
            or not 0 <= self.num_workers <= 16
        ):
            raise ValueError("num_workers must be between zero and 16")
        if not math.isfinite(self.ridge_alpha) or self.ridge_alpha <= 0:
            raise ValueError("ridge_alpha must be finite and positive")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or not 0 <= self.seed < 2**32 - 2
        ):
            raise ValueError("seed must be in [0, 2**32 - 2)")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def historical_scale_targets(
    asset_close: np.ndarray,
    benchmark_close: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Measure population scales on observed adjusted closes, without future returns."""

    asset = np.asarray(asset_close, dtype=np.float64)
    benchmark = np.asarray(benchmark_close, dtype=np.float64)
    if asset.ndim != 1 or asset.shape != benchmark.shape:
        raise ValueError("Historical close streams must be aligned one-dimensional arrays")
    if valid_mask is not None:
        mask = np.asarray(valid_mask)
        if mask.dtype != np.bool_ or mask.shape != asset.shape:
            raise ValueError("valid_mask must be a matching boolean array")
        asset, benchmark = asset[mask], benchmark[mask]
    if len(asset) < 61:
        raise ValueError("Historical scales require at least 61 valid observed bars")
    for values in (asset, benchmark):
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError("Historical adjusted closes must be finite and positive")
    asset_returns = np.diff(np.log(asset))
    benchmark_returns = np.diff(np.log(benchmark))
    relative_returns = asset_returns - benchmark_returns
    values = [
        float(returns[-window:].std(ddof=0))
        for window in (20, 60)
        for returns in (asset_returns, benchmark_returns, relative_returns)
    ]
    values.extend(float(close.std(ddof=0) / close.mean()) for close in (asset, benchmark))
    return np.asarray(values, dtype=np.float64)


class HistoricalScaleDataset(Dataset[dict[str, Any]]):
    """Adapt the existing bar store without invoking its forward-label construction.

    Composition intentionally avoids inheriting the lazy dataset's __getitems__,
    which calculates future alpha labels. The private index/alignment helpers keep
    this diagnostic on the exact same calendar and as-of adjustment path.
    """

    def __init__(self, source: LazyFinancialWindowDataset) -> None:
        if source.split not in ("train", "validation"):
            raise ValueError("Scale probes intentionally exclude the test split")
        if source.window_size < 61 or source.series_mode != "raw":
            raise ValueError("Scale probes require raw OHLCV inputs with at least 61 bars")
        self.source = source

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, ordinal: int) -> dict[str, Any]:
        source = self.source
        row, cutoff_index = source._resolve_cutoff(int(ordinal))
        symbol = str(row["symbol"])
        index_row = source._index[symbol]
        benchmark_symbol = str(index_row["benchmark_symbol"])
        frame = source._load_symbol(symbol)
        observed = frame.iloc[cutoff_index - source.window_size + 1 : cutoff_index + 1]
        timestamps = pd.DatetimeIndex(observed["timestamp"])
        benchmark = source._load_symbol(benchmark_symbol)
        benchmark_observed = source._aligned_rows(benchmark_symbol, benchmark, timestamps)
        asset_series = _series_from_asof_adjusted_frame(observed, "raw")
        benchmark_series = _series_from_asof_adjusted_frame(benchmark_observed, "raw")
        targets = historical_scale_targets(
            asset_series[:, 3].numpy(), benchmark_series[:, 3].numpy()
        )
        cutoff_at = timestamps[-1]
        return {
            "asset_series": asset_series,
            "benchmark_series": benchmark_series,
            "timestamps": _timestamp_features(timestamps),
            "scale_targets": torch.from_numpy(targets),
            "ordinal": int(ordinal),
            "sample_id": f"{symbol}-{cutoff_at.strftime('%Y%m%dT%H%M%SZ')}",
            "symbol": symbol,
            "benchmark_symbol": benchmark_symbol,
            "market": str(index_row["market"]),
            "cutoff_at": cutoff_at.isoformat(),
        }


def sampled_ordinals(candidate_count: int, limit: int, seed: int) -> list[int]:
    """Uniformly sample cutoffs without replacement; sort only for IO locality."""

    if candidate_count < 2 or limit < 2:
        raise ValueError("Each probe split needs at least two candidate samples")
    indices = np.random.default_rng(seed).choice(
        candidate_count, size=min(candidate_count, limit), replace=False
    )
    return np.sort(indices).tolist()


def _hidden_pool(hidden: Tensor, mask: Tensor, *, last: bool) -> Tensor:
    if hidden.ndim != 3 or mask.shape != hidden.shape[:2]:
        raise ValueError("Hidden states and attention mask have incompatible shapes")
    mask = mask.bool()
    if not bool(mask.any(dim=1).all()):
        raise ValueError("Each hidden sequence needs a valid token")
    hidden = hidden.float()
    if last:
        positions = torch.arange(hidden.shape[1], device=hidden.device).expand_as(mask)
        indices = positions.masked_fill(~mask, -1).max(dim=1).values
        return hidden[torch.arange(hidden.shape[0], device=hidden.device), indices]
    return hidden.masked_fill(~mask.unsqueeze(-1), 0).sum(dim=1) / mask.sum(dim=1, keepdim=True)


def representation_readouts(output: QuantForecastOutput) -> dict[str, Tensor]:
    """Expose fixed, interpretable readouts rather than train another deep network."""

    readouts = {
        f"kronos_pair_{name}": torch.cat(
            [
                _hidden_pool(hidden, mask, last=last)
                for hidden, mask in (
                    (output.asset_last_hidden_state, output.asset_attention_mask),
                    (output.benchmark_last_hidden_state, output.benchmark_attention_mask),
                )
            ],
            dim=-1,
        )
        for name, last in (("mean", False), ("last", True))
    }
    readouts["resampler_pair_mean"] = torch.cat(
        [
            output.asset_latent_tokens.float().mean(dim=1),
            output.benchmark_latent_tokens.float().mean(dim=1),
        ],
        dim=-1,
    )
    readouts["conditioned_mean"] = output.conditioned_latent_tokens.mean(dim=1).float()
    if any(not bool(torch.isfinite(values).all()) for values in readouts.values()):
        raise ValueError("Non-finite representation values cannot be probed")
    return readouts


@dataclass
class ProbeSplit:
    features: dict[str, np.ndarray]
    targets: np.ndarray
    samples: list[dict[str, Any]]
    candidate_count: int

    def summary(self) -> dict[str, Any]:
        return {
            "candidate_count": self.candidate_count,
            "samples": len(self.samples),
            "cutoff_min": min(row["cutoff_at"] for row in self.samples),
            "cutoff_max": max(row["cutoff_at"] for row in self.samples),
            "market_counts": dict(Counter(row["market"] for row in self.samples)),
            "symbol_count": len({row["symbol"] for row in self.samples}),
            "unique_cutoff_count": len({row["cutoff_at"] for row in self.samples}),
        }


def extract_probe_split(
    model: torch.nn.Module,
    dataset: HistoricalScaleDataset,
    *,
    device: torch.device,
    settings: ScaleProbeSettings,
    mixed_precision: str = "no",
) -> ProbeSplit:
    """Extract all readouts together, preserving identical row membership/order."""

    is_train = dataset.source.split == "train"
    indices = sampled_ordinals(
        len(dataset),
        settings.train_samples if is_train else settings.validation_samples,
        settings.seed if is_train else settings.seed + 1,
    )
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=settings.batch_size,
        shuffle=False,
        num_workers=settings.num_workers,
        pin_memory=device.type == "cuda",
    )
    model.eval()
    features: dict[str, np.ndarray] = {}
    targets = np.empty((len(indices), len(TARGET_NAMES)), dtype=np.float64)
    samples: list[dict[str, Any]] = []
    offset = 0
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            autocast = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if device.type == "cuda" and mixed_precision != "no"
                else nullcontext()
            )
            timestamps = batch["timestamps"].to(device, non_blocking=True)
            with autocast:
                output = model(
                    batch["asset_series"].to(device, non_blocking=True),
                    batch["benchmark_series"].to(device, non_blocking=True),
                    asset_timestamps=timestamps,
                    benchmark_timestamps=timestamps,
                    target_alpha=None,
                    market_ids=market_ids(batch["market"], device),
                )
            readouts = representation_readouts(output)
            count = len(batch["sample_id"])
            for name, values in readouts.items():
                if name not in features:
                    features[name] = np.empty((len(indices), values.shape[1]), dtype=np.float32)
                features[name][offset : offset + count] = values.cpu().numpy()
            targets[offset : offset + count] = batch["scale_targets"].numpy()
            for index in range(count):
                samples.append(
                    {
                        **{
                            key: batch[key][index]
                            for key in (
                                "sample_id",
                                "symbol",
                                "benchmark_symbol",
                                "market",
                                "cutoff_at",
                            )
                        },
                        "ordinal": int(batch["ordinal"][index]),
                    }
                )
            offset += count
            if batch_index % 64 == 0 or offset == len(indices):
                LOGGER.info("Extracted %s: %d/%d", dataset.source.split, offset, len(indices))
    if offset != len(indices):
        raise RuntimeError("Representation extraction returned incomplete sample membership")
    return ProbeSplit(features, targets, samples, len(dataset))


def regression_metrics(
    truth: np.ndarray, prediction: np.ndarray, train_mean: np.ndarray
) -> dict[str, dict[str, float | int | None]]:
    """Keep constant-target R2 and constant-vector correlations undefined, not zero."""

    if truth.shape != prediction.shape or truth.ndim != 2 or not len(truth):
        raise ValueError("Probe metrics require nonempty matching target matrices")
    if truth.shape[1] != len(TARGET_NAMES) or train_mean.shape != (len(TARGET_NAMES),):
        raise ValueError("Probe target dimensions differ from the historical scale contract")
    if not all(np.isfinite(values).all() for values in (truth, prediction, train_mean)):
        raise ValueError("Probe metrics require finite values")
    results = {}
    for index, name in enumerate(TARGET_NAMES):
        actual, predicted = truth[:, index], prediction[:, index]
        residual = actual - predicted
        mse = float(np.mean(residual**2))
        centered = actual - actual.mean()
        pred_centered = predicted - predicted.mean()
        variance = float(np.mean(centered**2))
        baseline_mse = float(np.mean((actual - train_mean[index]) ** 2))
        correlation = None
        if len(actual) >= 2 and np.ptp(actual) > 0 and np.ptp(predicted) > 0:
            correlation = float(
                np.clip(
                    np.dot(centered, pred_centered)
                    / (np.linalg.norm(centered) * np.linalg.norm(pred_centered)),
                    -1.0,
                    1.0,
                )
            )
        results[name] = {
            "samples": len(actual),
            "mae": float(np.mean(np.abs(residual))),
            "rmse": math.sqrt(mse),
            "r2": 1.0 - mse / variance if len(actual) >= 2 and np.ptp(actual) > 0 else None,
            "mse_skill_vs_train_mean": 1.0 - mse / baseline_mse if baseline_mse > 0 else None,
            "pearson_r": correlation,
            "target_mean": float(actual.mean()),
            "target_std": math.sqrt(variance),
        }
    return results


@dataclass
class ProbeResult:
    metrics: dict[str, Any]
    fitted_arrays: dict[str, np.ndarray]
    validation_predictions: dict[str, np.ndarray]


def fit_scale_probes(
    train: ProbeSplit, validation: ProbeSplit, settings: ScaleProbeSettings
) -> ProbeResult:
    """Fit scalers, ridge weights and a shuffled-label control exclusively on train."""

    if set(train.features) != set(validation.features) or not train.features:
        raise ValueError("Train and validation must expose the same nonempty readout set")
    for split in (train, validation):
        if split.targets.shape != (len(split.samples), len(TARGET_NAMES)) or len(split.samples) < 2:
            raise ValueError("Probe splits need aligned targets and at least two samples")
        if not np.isfinite(split.targets).all():
            raise ValueError("Scale targets must be finite")
    train_ids = {row["sample_id"] for row in train.samples}
    validation_ids = {row["sample_id"] for row in validation.samples}
    if (
        len(train_ids) != len(train.samples)
        or len(validation_ids) != len(validation.samples)
        or train_ids & validation_ids
    ):
        raise ValueError("Probe splits must have unique, disjoint sample IDs")
    if max(row["cutoff_at"] for row in train.samples) >= min(
        row["cutoff_at"] for row in validation.samples
    ):
        raise ValueError("Probe validation cutoffs must follow all probe training cutoffs")

    target_scaler = StandardScaler().fit(train.targets)
    train_y = target_scaler.transform(train.targets)
    permutation = np.random.default_rng(settings.seed + 2).permutation(len(train_y))
    # Both controls share one factorization and the same train-only preprocessing.
    joint_y = np.concatenate([train_y, train_y[permutation]], axis=1)
    train_mean = target_scaler.mean_
    mean_prediction = np.broadcast_to(train_mean, validation.targets.shape).copy()
    metrics: dict[str, Any] = {
        "train_mean_baseline": regression_metrics(validation.targets, mean_prediction, train_mean),
        "readouts": {},
    }
    arrays = {
        "target_mean": train_mean,
        "target_scale": target_scaler.scale_,
        "shuffled_train_row_permutation": permutation,
    }
    predictions = {"targets": validation.targets, "train_mean_baseline": mean_prediction}
    markets = np.asarray([row["market"] for row in validation.samples])
    metrics["train_mean_baseline_by_market"] = {
        market: regression_metrics(
            validation.targets[markets == market], mean_prediction[markets == market], train_mean
        )
        for market in sorted(set(markets))
    }
    for name, raw_train_x in train.features.items():
        LOGGER.info("Fitting ridge and shuffled-label control: %s", name)
        raw_validation_x = validation.features[name]
        if name not in READOUT_DESCRIPTIONS:
            raise ValueError(f"Unknown representation readout: {name}")
        for split, values in ((train, raw_train_x), (validation, raw_validation_x)):
            if values.ndim != 2 or len(values) != len(split.samples) or values.shape[1] < 1:
                raise ValueError("Readout rows must align with probe samples")
            if not np.isfinite(values).all():
                raise ValueError("Probe readouts must be finite")
        if raw_train_x.shape[1] != raw_validation_x.shape[1]:
            raise ValueError("Readout feature dimensions must match across splits")
        scaler = StandardScaler().fit(raw_train_x.astype(np.float64))
        train_x = scaler.transform(raw_train_x.astype(np.float64))
        validation_x = scaler.transform(raw_validation_x.astype(np.float64))
        ridge = Ridge(alpha=settings.ridge_alpha, solver="cholesky").fit(train_x, joint_y)
        train_prediction = (
            ridge.predict(train_x)[:, : len(TARGET_NAMES)] * target_scaler.scale_ + train_mean
        )
        joint_prediction = ridge.predict(validation_x)
        prediction = joint_prediction[:, : len(TARGET_NAMES)] * target_scaler.scale_ + train_mean
        shuffled_prediction = (
            joint_prediction[:, len(TARGET_NAMES) :] * target_scaler.scale_ + train_mean
        )
        metrics["readouts"][name] = {
            "description": READOUT_DESCRIPTIONS[name],
            "feature_dim": raw_train_x.shape[1],
            "constant_train_features": int(np.sum(scaler.var_ == 0)),
            "train_samples_per_feature": len(train_x) / raw_train_x.shape[1],
            "train": regression_metrics(train.targets, train_prediction, train_mean),
            "validation": regression_metrics(validation.targets, prediction, train_mean),
            "shuffled_label_validation": regression_metrics(
                validation.targets, shuffled_prediction, train_mean
            ),
            "validation_by_market": {
                market: {
                    "probe": regression_metrics(
                        validation.targets[markets == market],
                        prediction[markets == market],
                        train_mean,
                    ),
                    "shuffled_labels": regression_metrics(
                        validation.targets[markets == market],
                        shuffled_prediction[markets == market],
                        train_mean,
                    ),
                }
                for market in sorted(set(markets))
            },
        }
        predictions[name] = prediction
        predictions[f"{name}__shuffled_labels"] = shuffled_prediction
        arrays.update(
            {
                f"{name}__feature_mean": scaler.mean_,
                f"{name}__feature_scale": scaler.scale_,
                f"{name}__coef": ridge.coef_,
                f"{name}__intercept": ridge.intercept_,
            }
        )
    return ProbeResult(metrics, arrays, predictions)
