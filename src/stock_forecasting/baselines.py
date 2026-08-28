"""Continuous-alpha rule, tree, and causal neural baselines."""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.ensemble import HistGradientBoostingRegressor
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from stock_forecasting.data.horizons import (
    DEFAULT_ALPHA_HORIZONS,
    validate_alpha_horizons,
)
from stock_forecasting.metrics import cross_sectional_metrics, multi_horizon_alpha_metrics

ALPHA_QUANTILES = (0.1, 0.5, 0.9)
RULE_BASELINE_NAMES = (
    "always_buy",
    "zero_return",
    "momentum_5d",
    "reversal_5d",
    "ma_crossover",
    "rsi",
    "macd",
    "volatility_scaled",
)
OHLCV_SUMMARY_FEATURES = (
    "return_1d",
    "return_5d",
    "return_20d",
    "return_window",
    "annualized_volatility_20d",
    "average_true_range_14d",
    "volume_zscore_20d",
    "close_vs_sma20",
    "close_vs_sma50",
    "rsi14",
)


@dataclass(frozen=True)
class BaselineArrays:
    """Paired past-only features with multi-horizon continuous alpha targets."""

    features: NDArray[np.float64]
    sequences: NDArray[np.float32]
    instrument_mask: NDArray[np.bool_]
    auxiliary: NDArray[np.float32]
    targets: NDArray[np.float64]
    dates: list[str]
    symbols: list[str]
    asset_types: list[str]
    horizons: tuple[int, ...]


def _safe_log_return(values: NDArray[np.float64], periods: int) -> float:
    previous_index = max(0, values.size - periods - 1)
    return float(
        math.log(max(float(values[-1]), 1e-12))
        - math.log(max(float(values[previous_index]), 1e-12))
    )


def _numeric_summary(context: dict[str, Any]) -> list[float]:
    """Derive a fixed-width feature vector exclusively from historical OHLCV."""

    close = np.asarray(context["close"], dtype=np.float64)
    high = np.asarray(context["high"], dtype=np.float64)
    low = np.asarray(context["low"], dtype=np.float64)
    volume = np.asarray(context["volume"], dtype=np.float64)
    if close.size < 2:
        raise ValueError("Baseline context requires at least two OHLCV rows")
    log_returns = np.diff(np.log(np.maximum(close, 1e-12)))
    recent_returns = log_returns[-20:]
    volatility = (
        float(recent_returns.std(ddof=0) * math.sqrt(252.0))
        if recent_returns.size
        else 0.0
    )
    prior_close = np.concatenate(([close[0]], close[:-1]))
    true_range = np.maximum(
        high - low,
        np.maximum(np.abs(high - prior_close), np.abs(low - prior_close)),
    )
    average_true_range = float(true_range[-14:].mean() / max(float(close[-1]), 1e-12))
    recent_volume = np.log1p(np.maximum(volume[-20:], 0.0))
    volume_scale = max(float(recent_volume.std(ddof=0)), 1e-6)
    volume_zscore = float((recent_volume[-1] - recent_volume.mean()) / volume_scale)
    sma20 = max(float(close[-20:].mean()), 1e-12)
    sma50 = max(float(close[-50:].mean()), 1e-12)
    deltas = np.diff(close)[-14:]
    gains = np.clip(deltas, 0.0, None)
    losses = np.clip(-deltas, 0.0, None)
    average_gain = float(gains.mean()) if gains.size else 0.0
    average_loss = float(losses.mean()) if losses.size else 0.0
    rsi = 50.0
    if average_loss == 0.0 and average_gain > 0.0:
        rsi = 100.0
    elif average_loss > 0.0:
        rsi = 100.0 - 100.0 / (1.0 + average_gain / average_loss)
    return [
        _safe_log_return(close, 1),
        _safe_log_return(close, 5),
        _safe_log_return(close, 20),
        _safe_log_return(close, close.size - 1),
        volatility,
        average_true_range,
        volume_zscore,
        float(close[-1] / sma20 - 1.0),
        float(close[-1] / sma50 - 1.0),
        rsi,
    ]


def _relative_sequence(context: dict[str, Any]) -> NDArray[np.float32]:
    values = np.column_stack(
        [
            np.asarray(context[field], dtype=np.float32)
            for field in ("open", "high", "low", "close", "volume")
        ]
    )
    reference = max(float(values[-1, 3]), 1e-12)
    values[:, :4] = np.log(np.maximum(values[:, :4], 1e-12) / reference)
    volume = np.log1p(np.maximum(values[:, 4], 0.0))
    scale = max(float(volume.std()), 1e-6)
    values[:, 4] = (volume - float(volume.mean())) / scale
    return values


def baseline_arrays(records: list[dict[str, Any]]) -> BaselineArrays:
    """Convert conditional-alpha records without reading any future feature."""

    if not records:
        raise ValueError("Baseline records cannot be empty")
    features: list[list[float]] = []
    sequences: list[NDArray[np.float32]] = []
    targets: list[list[float]] = []
    dates: list[str] = []
    symbols: list[str] = []
    asset_types: list[str] = []
    first_label = records[0].get("label", {})
    horizons = validate_alpha_horizons(first_label.get("horizons", ()))
    for record in records:
        asset_summary = _numeric_summary(record["context"])
        benchmark_summary = _numeric_summary(record["benchmark_context"])
        feature_row = [
            *asset_summary,
            *benchmark_summary,
            *(
                asset - benchmark
                for asset, benchmark in zip(asset_summary, benchmark_summary, strict=True)
            ),
        ]
        asset_sequence = _relative_sequence(record["context"])
        benchmark_sequence = _relative_sequence(record["benchmark_context"])
        if asset_sequence.shape != benchmark_sequence.shape:
            raise ValueError("Asset and benchmark windows must have the same shape")
        label = record.get("label", {})
        if tuple(label.get("horizons", ())) != horizons:
            raise ValueError("Baseline records do not share one alpha horizon contract")
        alpha_values = label.get("alpha_log_returns")
        if not isinstance(alpha_values, dict):
            raise ValueError("Baseline record has no alpha_log_returns")
        features.append(feature_row)
        sequences.append(np.stack([asset_sequence, benchmark_sequence]))
        targets.append([float(alpha_values[f"{horizon}d"]) for horizon in horizons])
        dates.append(str(record["cutoff_at"]))
        symbols.append(str(record["symbol"]))
        asset_types.append(str(record["asset_type"]))
    sequence_shapes = {value.shape for value in sequences}
    if len(sequence_shapes) != 1:
        raise ValueError("Neural baselines require fixed-length paired windows")
    target_array = np.asarray(targets, dtype=np.float64)
    if not np.isfinite(target_array).all():
        raise ValueError("Baseline targets must be finite")
    feature_array = np.asarray(features, dtype=np.float64)
    return BaselineArrays(
        features=feature_array,
        sequences=np.stack(sequences).astype(np.float32),
        instrument_mask=np.ones((len(records), 2), dtype=np.bool_),
        auxiliary=feature_array.astype(np.float32),
        targets=target_array,
        dates=dates,
        symbols=symbols,
        asset_types=asset_types,
        horizons=horizons,
    )


def _robust_scales(targets: NDArray[np.float64]) -> list[float]:
    scales: list[float] = []
    for index in range(targets.shape[1]):
        values = targets[:, index]
        q25, q75 = np.quantile(values, [0.25, 0.75])
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)) * 1.4826)
        scales.append(max(float(q75 - q25), mad, 1e-4))
    return scales


def _evaluate_predictions(
    arrays: BaselineArrays,
    predictions: NDArray[np.float64],
    robust_scales: list[float],
) -> dict[str, Any]:
    predictions = np.asarray(predictions, dtype=np.float64)
    if predictions.shape != (
        len(arrays.targets),
        len(arrays.horizons),
        len(ALPHA_QUANTILES),
    ):
        raise ValueError("Baseline predictions have an invalid multi-horizon shape")
    predictions.sort(axis=-1)
    metrics = multi_horizon_alpha_metrics(
        targets=arrays.targets,
        quantile_predictions=predictions,
        horizons=list(arrays.horizons),
        quantiles=list(ALPHA_QUANTILES),
        robust_scales=robust_scales,
    )
    median_index = ALPHA_QUANTILES.index(0.5)
    cross_sectional = {
        f"{horizon}d": cross_sectional_metrics(
            targets=arrays.targets[:, horizon_index],
            signals=predictions[:, horizon_index, median_index],
            dates=arrays.dates,
            symbols=arrays.symbols,
            annualization_horizon=horizon,
        )
        for horizon_index, horizon in enumerate(arrays.horizons)
    }
    subgroups: dict[str, Any] = {}
    for asset_type in sorted(set(arrays.asset_types)):
        indices = np.asarray(
            [index for index, value in enumerate(arrays.asset_types) if value == asset_type],
            dtype=np.int64,
        )
        subgroups[f"asset_type/{asset_type}"] = {
            "samples": int(indices.size),
            **multi_horizon_alpha_metrics(
                targets=arrays.targets[indices],
                quantile_predictions=predictions[indices],
                horizons=list(arrays.horizons),
                quantiles=list(ALPHA_QUANTILES),
                robust_scales=robust_scales,
            ),
        }
    return {
        **metrics,
        "cross_sectional_by_horizon": cross_sectional,
        "cross_sectional_5d": cross_sectional["5d"],
        "subgroups": subgroups,
    }


def _ema(values: NDArray[np.float64], span: int) -> NDArray[np.float64]:
    alpha = 2.0 / (span + 1.0)
    output = np.empty_like(values)
    output[:, 0] = values[:, 0]
    for index in range(1, values.shape[1]):
        output[:, index] = alpha * values[:, index] + (1.0 - alpha) * output[:, index - 1]
    return output


def _relative_rule_signals(arrays: BaselineArrays) -> dict[str, NDArray[np.float64]]:
    asset_close = arrays.sequences[:, 0, :, 3].astype(np.float64)
    benchmark_close = arrays.sequences[:, 1, :, 3].astype(np.float64)
    asset_5d = asset_close[:, -1] - asset_close[:, -6]
    benchmark_5d = benchmark_close[:, -1] - benchmark_close[:, -6]
    momentum = asset_5d - benchmark_5d

    asset_short = asset_close[:, -20:].mean(axis=1)
    asset_long = asset_close[:, -50:].mean(axis=1)
    benchmark_short = benchmark_close[:, -20:].mean(axis=1)
    benchmark_long = benchmark_close[:, -50:].mean(axis=1)
    relative_scale = np.maximum(
        (asset_close[:, -50:] - benchmark_close[:, -50:]).std(axis=1),
        1e-6,
    )
    ma_signal = ((asset_short - asset_long) - (benchmark_short - benchmark_long)) / relative_scale

    asset_rsi = arrays.features[:, OHLCV_SUMMARY_FEATURES.index("rsi14")]
    benchmark_rsi = arrays.features[
        :, len(OHLCV_SUMMARY_FEATURES) + OHLCV_SUMMARY_FEATURES.index("rsi14")
    ]
    rsi_signal = np.clip((benchmark_rsi - asset_rsi) / 20.0, -2.5, 2.5)

    asset_macd = _ema(asset_close, 12) - _ema(asset_close, 26)
    benchmark_macd = _ema(benchmark_close, 12) - _ema(benchmark_close, 26)
    relative_macd = asset_macd - benchmark_macd
    macd_signal_line = _ema(relative_macd, 9)
    macd_signal = np.clip(
        (relative_macd[:, -1] - macd_signal_line[:, -1]) / relative_scale,
        -4.0,
        4.0,
    )
    relative_daily = np.diff(asset_close - benchmark_close, axis=1)
    relative_volatility = np.maximum(relative_daily[:, -20:].std(axis=1), 1e-6)
    volatility_scaled = np.clip(momentum / relative_volatility, -4.0, 4.0)
    return {
        "momentum_5d": momentum,
        "reversal_5d": -momentum,
        "ma_crossover": ma_signal * relative_volatility,
        "rsi": rsi_signal * relative_volatility,
        "macd": macd_signal * relative_volatility,
        "volatility_scaled": volatility_scaled * relative_volatility,
    }


def _rule_medians(
    train: BaselineArrays,
    arrays: BaselineArrays,
) -> dict[str, NDArray[np.float64]]:
    train_scales = np.asarray(_robust_scales(train.targets), dtype=np.float64)
    positive_location = np.maximum(np.median(train.targets, axis=0), 0.25 * train_scales)
    horizon_multiplier = np.asarray(arrays.horizons, dtype=np.float64) / 5.0
    signals = _relative_rule_signals(arrays)
    output = {
        "always_buy": np.broadcast_to(positive_location, arrays.targets.shape).copy(),
        "zero_return": np.zeros_like(arrays.targets),
    }
    output.update(
        {
            name: signal[:, None] * horizon_multiplier[None, :]
            for name, signal in signals.items()
        }
    )
    return output


def rule_baseline_suite(
    train: BaselineArrays,
    validation: BaselineArrays,
) -> dict[str, dict[str, Any]]:
    """Calibrate rule distributions on train and evaluate on validation only."""

    train_medians = _rule_medians(train, train)
    validation_medians = _rule_medians(train, validation)
    scales = _robust_scales(train.targets)
    results: dict[str, dict[str, Any]] = {}
    for name in RULE_BASELINE_NAMES:
        residual_quantiles = np.quantile(
            train.targets - train_medians[name],
            ALPHA_QUANTILES,
            axis=0,
        ).T
        predictions = validation_medians[name][..., None] + residual_quantiles[None, ...]
        results[name] = _evaluate_predictions(validation, predictions, scales)
    return results


def rule_baseline(arrays: BaselineArrays) -> dict[str, Any]:
    """Compatibility helper for a self-calibrated past-only rule smoke test."""

    return rule_baseline_suite(arrays, arrays)["momentum_5d"]


@dataclass
class GradientBoostingBaseline:
    quantile_models: list[list[HistGradientBoostingRegressor]]
    robust_scales: list[float]

    @classmethod
    def fit(cls, train: BaselineArrays, seed: int = 42) -> GradientBoostingBaseline:
        models: list[list[HistGradientBoostingRegressor]] = []
        for horizon_index in range(len(train.horizons)):
            models.append(
                [
                    HistGradientBoostingRegressor(
                        loss="quantile",
                        quantile=quantile,
                        learning_rate=0.05,
                        max_iter=200,
                        max_leaf_nodes=31,
                        l2_regularization=1.0,
                        random_state=seed,
                    ).fit(train.features, train.targets[:, horizon_index])
                    for quantile in ALPHA_QUANTILES
                ]
            )
        return cls(models, _robust_scales(train.targets))

    def evaluate(self, arrays: BaselineArrays) -> dict[str, Any]:
        predictions = np.stack(
            [
                np.column_stack([model.predict(arrays.features) for model in horizon_models])
                for horizon_models in self.quantile_models
            ],
            axis=1,
        )
        return _evaluate_predictions(arrays, predictions, self.robust_scales)


def _ordered_quantiles(raw: Tensor) -> Tensor:
    median = raw[..., 1]
    lower = median - torch.nn.functional.softplus(raw[..., 0])
    upper = median + torch.nn.functional.softplus(raw[..., 2])
    return torch.stack([lower, median, upper], dim=-1)


class CausalGRUBaseline(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 64,
        *,
        horizons: tuple[int, ...] = DEFAULT_ALPHA_HORIZONS,
    ) -> None:
        super().__init__()
        self.horizons = validate_alpha_horizons(horizons)
        self.robust_scales: tuple[float, ...] | None = None
        self.encoder = nn.GRU(input_size=5, hidden_size=hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, len(self.horizons) * 3),
        )

    def forward(self, sequences: Tensor) -> Tensor:
        batch, instruments, steps, fields = sequences.shape
        if instruments != 2 or fields != 5:
            raise ValueError("GRU requires [batch, asset_and_benchmark, time, five_fields]")
        _output, hidden = self.encoder(sequences.reshape(batch * instruments, steps, fields))
        encoded = hidden[-1].reshape(batch, instruments, -1)
        fused = torch.cat([encoded[:, 0], encoded[:, 1], encoded[:, 0] - encoded[:, 1]], dim=-1)
        raw = self.head(fused).reshape(batch, len(self.horizons), 3)
        return _ordered_quantiles(raw)


class DLinearBaseline(nn.Module):
    def __init__(
        self,
        context_length: int,
        *,
        horizons: tuple[int, ...] = DEFAULT_ALPHA_HORIZONS,
    ) -> None:
        super().__init__()
        self.context_length = context_length
        self.horizons = validate_alpha_horizons(horizons)
        self.trend = nn.AvgPool1d(kernel_size=25, stride=1, padding=12)
        self.linear = nn.Linear(context_length * 4, len(self.horizons) * 3)

    def forward(self, sequences: Tensor) -> Tensor:
        relative_close = sequences[:, 0, :, 3] - sequences[:, 1, :, 3]
        relative_volume = sequences[:, 0, :, 4] - sequences[:, 1, :, 4]
        trend_close = self.trend(relative_close[:, None, :]).squeeze(1)
        trend_volume = self.trend(relative_volume[:, None, :]).squeeze(1)
        seasonal_close = relative_close - trend_close
        seasonal_volume = relative_volume - trend_volume
        features = torch.cat(
            [trend_close, seasonal_close, trend_volume, seasonal_volume],
            dim=-1,
        )
        raw = self.linear(features).reshape(-1, len(self.horizons), 3)
        return _ordered_quantiles(raw)


class PatchTSTBaseline(nn.Module):
    def __init__(
        self,
        *,
        patch_length: int = 16,
        stride: int = 8,
        hidden_dim: int = 64,
        horizons: tuple[int, ...] = DEFAULT_ALPHA_HORIZONS,
    ) -> None:
        super().__init__()
        self.horizons = validate_alpha_horizons(horizons)
        self.patch = nn.Conv1d(10, hidden_dim, kernel_size=patch_length, stride=stride)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Linear(hidden_dim, len(self.horizons) * 3)

    def forward(self, sequences: Tensor) -> Tensor:
        batch, instruments, steps, fields = sequences.shape
        values = sequences.permute(0, 1, 3, 2).reshape(batch, instruments * fields, steps)
        patches = self.patch(values).transpose(1, 2)
        encoded = self.encoder(patches).mean(dim=1)
        raw = self.head(encoded).reshape(batch, len(self.horizons), 3)
        return _ordered_quantiles(raw)


def _seed_baseline(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _normalized_pinball_torch(
    predictions: Tensor,
    targets: Tensor,
    scales: Tensor,
) -> Tensor:
    errors = (targets[..., None] - predictions) / scales[None, :, None]
    levels = predictions.new_tensor(ALPHA_QUANTILES)
    return torch.maximum(levels * errors, (levels - 1.0) * errors).mean()


@torch.no_grad()
def _evaluate_neural(
    model: nn.Module,
    arrays: BaselineArrays,
    robust_scales: list[float],
    *,
    batch_size: int = 256,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    model.eval()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(arrays.sequences)),
        batch_size=batch_size,
        shuffle=False,
    )
    predictions = [model(batch[0].to(device)).float().cpu().numpy() for batch in loader]
    return _evaluate_predictions(
        arrays,
        np.concatenate(predictions).astype(np.float64),
        robust_scales,
    )


def _fit_neural(
    model: nn.Module,
    train: BaselineArrays,
    validation: BaselineArrays,
    *,
    seed: int,
    epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
) -> tuple[nn.Module, dict[str, Any]]:
    _seed_baseline(seed)
    if min(epochs, patience, batch_size) < 1 or learning_rate <= 0.0:
        raise ValueError("Neural baseline training parameters must be positive")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    scales = _robust_scales(train.targets)
    if hasattr(model, "robust_scales"):
        model.robust_scales = tuple(scales)
    scale_tensor = torch.tensor(scales, dtype=torch.float32, device=device)
    dataset = TensorDataset(
        torch.from_numpy(train.sequences),
        torch.from_numpy(train.targets.astype(np.float32)),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    best_state: dict[str, Tensor] | None = None
    best_score = math.inf
    stale_epochs = 0
    for _epoch in range(epochs):
        model.train()
        for sequences, targets in loader:
            predictions = model(sequences.to(device))
            loss = _normalized_pinball_torch(predictions, targets.to(device), scale_tensor)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        metrics = _evaluate_neural(model, validation, scales, batch_size=batch_size)
        score = float(metrics["primary_5d"]["selection_score"])
        if score < best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    if best_state is None:
        raise RuntimeError("Neural baseline did not produce a validation checkpoint")
    model.load_state_dict(best_state)
    return model, _evaluate_neural(model, validation, scales, batch_size=batch_size)


def fit_causal_gru(
    train: BaselineArrays,
    validation: BaselineArrays,
    *,
    seed: int = 42,
    epochs: int = 30,
    patience: int = 5,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
) -> tuple[CausalGRUBaseline, dict[str, Any]]:
    _seed_baseline(seed)
    model = CausalGRUBaseline(horizons=train.horizons)
    trained, metrics = _fit_neural(
        model,
        train,
        validation,
        seed=seed,
        epochs=epochs,
        patience=patience,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )
    return cast(CausalGRUBaseline, trained), metrics


def evaluate_causal_gru(
    model: CausalGRUBaseline,
    arrays: BaselineArrays,
    *,
    robust_scales: list[float] | None = None,
) -> dict[str, Any]:
    stored_scales = model.robust_scales
    return _evaluate_neural(
        model,
        arrays,
        robust_scales or list(stored_scales or _robust_scales(arrays.targets)),
    )


def fit_dlinear(
    train: BaselineArrays,
    validation: BaselineArrays,
    *,
    seed: int = 42,
    epochs: int = 30,
    patience: int = 5,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
) -> tuple[DLinearBaseline, dict[str, Any]]:
    _seed_baseline(seed)
    model = DLinearBaseline(
        context_length=train.sequences.shape[2],
        horizons=train.horizons,
    )
    trained, metrics = _fit_neural(
        model,
        train,
        validation,
        seed=seed,
        epochs=epochs,
        patience=patience,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )
    return cast(DLinearBaseline, trained), metrics


def fit_patchtst(
    train: BaselineArrays,
    validation: BaselineArrays,
    *,
    seed: int = 42,
    epochs: int = 30,
    patience: int = 5,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
) -> tuple[PatchTSTBaseline, dict[str, Any]]:
    _seed_baseline(seed)
    model = PatchTSTBaseline(horizons=train.horizons)
    trained, metrics = _fit_neural(
        model,
        train,
        validation,
        seed=seed,
        epochs=epochs,
        patience=patience,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )
    return cast(PatchTSTBaseline, trained), metrics
