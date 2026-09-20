"""Continuous multi-horizon alpha metrics and post-processing diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating[Any]]

POSTPROCESS_SIGNAL_NAMES = (
    "strong_bearish",
    "bearish",
    "neutral",
    "bullish",
    "strong_bullish",
)


def pinball_loss(
    targets: FloatArray,
    predictions: FloatArray,
    quantiles: list[float],
) -> float:
    """Return mean pinball loss for one horizon without changing units."""

    target_array = np.asarray(targets, dtype=np.float64)
    prediction_array = np.asarray(predictions, dtype=np.float64)
    if prediction_array.shape != (target_array.size, len(quantiles)):
        raise ValueError("Predictions must have shape [samples, quantiles]")
    levels = np.asarray(quantiles, dtype=np.float64).reshape(1, -1)
    errors = target_array.reshape(-1, 1) - prediction_array
    return float(np.maximum(levels * errors, (levels - 1.0) * errors).mean())


def postprocess_alpha_signal(
    alpha_quantiles: FloatArray,
    *,
    threshold: float = 0.0,
) -> NDArray[np.int64]:
    """Derive five ordered signals from q10/q50/q90 without a classifier."""

    values = np.asarray(alpha_quantiles, dtype=np.float64)
    if values.ndim < 2 or values.shape[-1] != 3:
        raise ValueError("alpha_quantiles must end with q10/q50/q90")
    if not np.isfinite(values).all() or threshold < 0.0:
        raise ValueError("Signal inputs must be finite and threshold must be non-negative")
    q10 = values[..., 0]
    q50 = values[..., 1]
    q90 = values[..., 2]
    result = np.full(q50.shape, 2, dtype=np.int64)
    result[q50 < -threshold] = 1
    result[q90 < -threshold] = 0
    result[q50 > threshold] = 3
    result[q10 > threshold] = 4
    return result


def alpha_quantile_metrics(
    *,
    targets: FloatArray,
    quantile_predictions: FloatArray,
    quantiles: list[float],
    robust_scale: float,
) -> dict[str, float]:
    """Evaluate one horizon of the conditional alpha distribution."""

    target_array = np.asarray(targets, dtype=np.float64)
    prediction_array = np.asarray(quantile_predictions, dtype=np.float64)
    if target_array.ndim != 1:
        raise ValueError("One-horizon targets must be one-dimensional")
    if prediction_array.shape != (target_array.size, len(quantiles)):
        raise ValueError("One-horizon predictions have an invalid shape")
    if not np.isfinite(target_array).all() or not np.isfinite(prediction_array).all():
        raise ValueError("Alpha metrics require finite targets and predictions")
    if not np.isfinite(robust_scale) or robust_scale <= 0.0:
        raise ValueError("robust_scale must be finite and positive")
    median_index = quantiles.index(0.5)
    pinball = pinball_loss(target_array, prediction_array, quantiles)
    coverage = float(
        (
            (target_array >= prediction_array[:, 0]) & (target_array <= prediction_array[:, -1])
        ).mean()
    )
    median = prediction_array[:, median_index]
    target_std = float(target_array.std(ddof=1)) if target_array.size > 1 else 0.0
    median_correlation = (
        float(np.corrcoef(target_array, median)[0, 1])
        if target_std > 0.0 and float(median.std(ddof=1)) > 0.0
        else 0.0
    )
    if not np.isfinite(median_correlation):
        median_correlation = 0.0
    return {
        "pinball": pinball,
        "normalized_pinball": pinball / robust_scale,
        "median_mae": float(np.abs(target_array - median).mean()),
        "normalized_median_mae": float(np.abs(target_array - median).mean()) / robust_scale,
        "median_correlation": median_correlation,
        "median_direction_agreement": float((np.sign(target_array) == np.sign(median)).mean()),
        "interval_coverage": coverage,
        "interval_width": float((prediction_array[:, -1] - prediction_array[:, 0]).mean()),
        "coverage_error": abs(coverage - float(quantiles[-1] - quantiles[0])),
    }


def multi_horizon_alpha_metrics(
    *,
    targets: FloatArray,
    quantile_predictions: FloatArray,
    horizons: list[int],
    quantiles: list[float],
    robust_scales: list[float],
) -> dict[str, Any]:
    """Return per-horizon metrics and an all-horizon pinball selection score."""

    target_array = np.asarray(targets, dtype=np.float64)
    prediction_array = np.asarray(quantile_predictions, dtype=np.float64)
    expected_target_shape = (target_array.shape[0], len(horizons))
    expected_prediction_shape = (*expected_target_shape, len(quantiles))
    if target_array.shape != expected_target_shape:
        raise ValueError("Targets must have shape [samples, horizons]")
    if prediction_array.shape != expected_prediction_shape:
        raise ValueError("Predictions must have shape [samples, horizons, quantiles]")
    if len(robust_scales) != len(horizons):
        raise ValueError("robust_scales must match horizons")
    per_horizon = {
        f"{horizon}d": alpha_quantile_metrics(
            targets=target_array[:, index],
            quantile_predictions=prediction_array[:, index, :],
            quantiles=quantiles,
            robust_scale=float(robust_scales[index]),
        )
        for index, horizon in enumerate(horizons)
    }
    selection_score = float(
        np.mean([metrics["normalized_pinball"] for metrics in per_horizon.values()])
    )
    aggregate = {
        "selection_score": selection_score,
        "normalized_pinball": selection_score,
        "pinball": float(np.mean([metrics["pinball"] for metrics in per_horizon.values()])),
        "median_mae": float(np.mean([metrics["median_mae"] for metrics in per_horizon.values()])),
        "interval_coverage": float(
            np.mean([metrics["interval_coverage"] for metrics in per_horizon.values()])
        ),
        "coverage_error": float(
            np.mean([metrics["coverage_error"] for metrics in per_horizon.values()])
        ),
    }
    if "5d" not in per_horizon:
        raise ValueError("The deployment compatibility alias requires horizon 5")
    primary_5d = {**per_horizon["5d"], "selection_score": selection_score}
    return {
        "aggregate": aggregate,
        "per_horizon": per_horizon,
        # Stable RunPod monitor alias; selection_score still covers every horizon.
        "primary_5d": primary_5d,
    }


def _rank(values: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.asarray(values).argsort(kind="stable").argsort(kind="stable").astype(np.float64)


def cross_sectional_metrics(
    *,
    targets: FloatArray,
    signals: FloatArray,
    dates: list[str],
    symbols: list[str],
    transaction_cost_bps: float = 10.0,
    annualization_horizon: int = 5,
    date_groups: tuple[NDArray[np.int64], NDArray[np.int64]] | None = None,
) -> dict[str, float]:
    """Compute date-level RankIC and an equal-weight long/short diagnostic."""

    target_array = np.asarray(targets, dtype=np.float64)
    signal_array = np.asarray(signals, dtype=np.float64)
    if not (len(target_array) == len(signal_array) == len(dates) == len(symbols)):
        raise ValueError("Cross-sectional metric inputs must have equal lengths")
    if annualization_horizon < 1:
        raise ValueError("annualization_horizon must be positive")
    information_coefficients: list[float] = []
    period_returns: list[float] = []
    turnovers: list[float] = []
    previous_weights: dict[str, float] = {}
    if date_groups is None:
        date_array = np.asarray(dates)
        date_order = np.argsort(date_array, kind="stable")
        boundaries = np.flatnonzero(date_array[date_order][1:] != date_array[date_order][:-1]) + 1
    else:
        date_order, boundaries = date_groups
    # Sort once instead of scanning millions of rows for every forecast date.
    for indices in np.split(date_order, boundaries):
        if indices.size < 3:
            continue
        period_targets = target_array[indices]
        period_signals = signal_array[indices]
        valid = np.isfinite(period_targets) & np.isfinite(period_signals)
        if valid.sum() < 3 or np.ptp(period_signals[valid]) == 0:
            continue
        valid_indices = indices[valid]
        correlation = np.corrcoef(
            _rank(target_array[valid_indices]),
            _rank(signal_array[valid_indices]),
        )[0, 1]
        if np.isfinite(correlation):
            information_coefficients.append(float(correlation))
        order = valid_indices[np.argsort(signal_array[valid_indices], kind="stable")]
        bucket = max(1, len(order) // 5)
        short_indices = order[:bucket]
        long_indices = order[-bucket:]
        weights = {
            **{symbols[index]: -0.5 / bucket for index in short_indices},
            **{symbols[index]: 0.5 / bucket for index in long_indices},
        }
        turnover = sum(
            abs(weights.get(symbol, 0.0) - previous_weights.get(symbol, 0.0))
            for symbol in set(weights) | set(previous_weights)
        )
        gross = float(target_array[long_indices].mean() - target_array[short_indices].mean())
        period_returns.append(gross - turnover * transaction_cost_bps / 10000.0)
        turnovers.append(turnover)
        previous_weights = weights

    ic_array = np.asarray(information_coefficients, dtype=np.float64)
    return_array = np.asarray(period_returns, dtype=np.float64)
    if return_array.size:
        wealth = np.exp(np.cumsum(return_array))
        peaks = np.maximum.accumulate(wealth)
        max_drawdown = float((wealth / peaks - 1.0).min())
        standard_deviation = float(return_array.std(ddof=1)) if return_array.size > 1 else 0.0
        sharpe = (
            float(return_array.mean() / standard_deviation * np.sqrt(252.0 / annualization_horizon))
            if standard_deviation > 0
            else 0.0
        )
    else:
        max_drawdown = sharpe = 0.0
    ic_std = float(ic_array.std(ddof=1)) if ic_array.size > 1 else 0.0
    return {
        "rank_ic_mean": float(ic_array.mean()) if ic_array.size else 0.0,
        "rank_ic_ir": float(ic_array.mean() / ic_std) if ic_std > 0 else 0.0,
        "net_long_short_mean_log_return": (
            float(return_array.mean()) if return_array.size else 0.0
        ),
        "net_long_short_sharpe": sharpe,
        "net_long_short_max_drawdown": max_drawdown,
        "turnover_mean": float(np.mean(turnovers)) if turnovers else 0.0,
        "evaluated_dates": float(len(period_returns)),
    }
