"""Point-in-time OHLCV adjustment and execution-return utilities."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

PRICE_FIELDS = ("open", "high", "low", "close")
ADJUSTED_CLOSE_FIELD = "adjusted_close"
ADJUSTED_VOLUME_FIELD = "split_adjusted_volume"


def ensure_adjustment_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Return raw bars with explicit total-return and split-volume anchors.

    ``adjusted_close`` may use a globally back-adjusted vendor series. Callers
    must divide its factor by the factor observed at the sample cutoff before
    constructing model inputs. That ratio cancels actions after the cutoff.
    """

    enriched = frame.copy()
    if ADJUSTED_CLOSE_FIELD not in enriched:
        enriched[ADJUSTED_CLOSE_FIELD] = enriched["close"]
    if ADJUSTED_VOLUME_FIELD not in enriched:
        enriched[ADJUSTED_VOLUME_FIELD] = enriched["volume"]
    return enriched


def normalize_action_frame(events: pd.DataFrame) -> pd.DataFrame:
    """Validate the compact corporate-action factors used by the pipeline."""

    required = {"timestamp", "symbol", "price_factor", "share_multiplier", "source"}
    missing = sorted(required.difference(events.columns))
    if missing:
        raise ValueError("Missing corporate-action columns: " + ", ".join(missing))
    normalized = events.loc[:, sorted(required)].copy()
    normalized["timestamp"] = pd.to_datetime(normalized["timestamp"], utc=True, errors="coerce")
    normalized["symbol"] = normalized["symbol"].astype("string").str.strip().str.upper()
    normalized["source"] = normalized["source"].astype("string").str.strip()
    normalized["price_factor"] = pd.to_numeric(normalized["price_factor"], errors="coerce")
    normalized["share_multiplier"] = pd.to_numeric(
        normalized["share_multiplier"], errors="coerce"
    )
    if (
        normalized[["timestamp", "symbol", "source"]].isna().any(axis=None)
        or (normalized["symbol"].str.len() == 0).any()
        or (normalized["source"].str.len() == 0).any()
    ):
        raise ValueError("Corporate-action identifiers must be non-empty")
    numeric = normalized[["price_factor", "share_multiplier"]].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all() or (numeric <= 0.0).any():
        raise ValueError("Corporate-action factors must be finite and positive")
    if normalized.duplicated(["symbol", "timestamp"]).any():
        grouped = (
            normalized.groupby(["symbol", "timestamp"], as_index=False, sort=True)
            .agg(
                price_factor=("price_factor", "prod"),
                share_multiplier=("share_multiplier", "prod"),
                source=("source", lambda values: "+".join(sorted(set(map(str, values))))),
            )
            .reset_index(drop=True)
        )
        normalized = grouped
    return normalized.sort_values(["symbol", "timestamp"], kind="stable").reset_index(drop=True)


def apply_cumulative_adjustments(
    frame: pd.DataFrame,
    events: pd.DataFrame | None,
    *,
    preserve_adjusted_close: bool = False,
) -> pd.DataFrame:
    """Back-adjust bars while preserving every raw OHLCV field unchanged."""

    enriched = ensure_adjustment_columns(frame)
    if events is None or events.empty:
        return enriched
    actions = normalize_action_frame(events)
    output: list[pd.DataFrame] = []
    for symbol, rows in enriched.groupby("symbol", sort=True):
        ordered = rows.sort_values("timestamp", kind="stable").copy()
        symbol_actions = actions[actions["symbol"] == str(symbol).upper()]
        if symbol_actions.empty:
            output.append(ordered)
            continue
        timestamps = pd.DatetimeIndex(ordered["timestamp"])
        price_factor = np.ones(len(ordered), dtype=np.float64)
        share_factor = np.ones(len(ordered), dtype=np.float64)
        for action in symbol_actions.itertuples(index=False):
            before = timestamps < pd.Timestamp(action.timestamp)
            price_factor[before] *= float(action.price_factor)
            share_factor[before] *= float(action.share_multiplier)
        if not preserve_adjusted_close:
            ordered[ADJUSTED_CLOSE_FIELD] = (
                ordered["close"].to_numpy(dtype=np.float64) * price_factor
            )
        ordered[ADJUSTED_VOLUME_FIELD] = (
            ordered["volume"].to_numpy(dtype=np.float64) * share_factor
        )
        output.append(ordered)
    return pd.concat(output, ignore_index=True).sort_values(
        ["symbol", "timestamp"], kind="stable"
    ).reset_index(drop=True)


def asof_adjusted_window(window: pd.DataFrame) -> pd.DataFrame:
    """Create cutoff-causal adjusted OHLCV from one historical-only window."""

    if window.empty:
        raise ValueError("Cannot adjust an empty OHLCV window")
    enriched = ensure_adjustment_columns(window)
    close = enriched["close"].to_numpy(dtype=np.float64)
    adjusted_close = enriched[ADJUSTED_CLOSE_FIELD].to_numpy(dtype=np.float64)
    price_factor = adjusted_close / np.maximum(close, 1e-12)
    cutoff_factor = float(price_factor[-1])
    if not np.isfinite(cutoff_factor) or cutoff_factor <= 0.0:
        raise ValueError("Cutoff total-return adjustment factor is invalid")
    causal_price_factor = price_factor / cutoff_factor

    raw_volume = enriched["volume"].to_numpy(dtype=np.float64)
    adjusted_volume = enriched[ADJUSTED_VOLUME_FIELD].to_numpy(dtype=np.float64)
    volume_factor = np.divide(
        adjusted_volume,
        raw_volume,
        out=np.ones_like(adjusted_volume),
        where=raw_volume > 0.0,
    )
    cutoff_volume_factor = float(volume_factor[-1])
    if not np.isfinite(cutoff_volume_factor) or cutoff_volume_factor <= 0.0:
        raise ValueError("Cutoff share adjustment factor is invalid")

    adjusted = enriched.copy()
    for field in PRICE_FIELDS:
        adjusted[field] = enriched[field].to_numpy(dtype=np.float64) * causal_price_factor
    adjusted["volume"] = raw_volume * (volume_factor / cutoff_volume_factor)
    return adjusted


def execution_total_return(
    rows_by_date: pd.DataFrame,
    *,
    entry_at: pd.Timestamp,
    exit_at: pd.Timestamp,
) -> float:
    """Return adjusted open-to-close total return for an exact holding interval."""

    enriched = ensure_adjustment_columns(rows_by_date).set_index("timestamp", drop=False)
    try:
        entry = enriched.loc[entry_at]
        exit_row = enriched.loc[exit_at]
    except KeyError as error:
        raise ValueError("Execution dates are absent from the aligned OHLCV series") from error
    if isinstance(entry, pd.DataFrame) or isinstance(exit_row, pd.DataFrame):
        raise ValueError("Execution dates must identify exactly one OHLCV row")
    entry_close = float(entry["close"])
    entry_adjusted_close = float(entry[ADJUSTED_CLOSE_FIELD])
    entry_factor = entry_adjusted_close / max(entry_close, 1e-12)
    adjusted_entry_open = float(entry["open"]) * entry_factor
    adjusted_exit_close = float(exit_row[ADJUSTED_CLOSE_FIELD])
    gross = adjusted_exit_close / max(adjusted_entry_open, 1e-12)
    if not np.isfinite(gross) or gross <= 0.0:
        raise ValueError("Execution total-return gross factor must be finite and positive")
    return float(gross - 1.0)


def robust_horizon_scales(
    records: Iterable[dict[str, Any]],
    horizons: Iterable[int],
    *,
    minimum_scale: float = 1e-4,
) -> list[float]:
    """Estimate train-only IQR scales without changing prediction units."""

    ordered_horizons = tuple(int(horizon) for horizon in horizons)
    columns: dict[int, list[float]] = {horizon: [] for horizon in ordered_horizons}
    for record in records:
        if record.get("split") != "train":
            continue
        label = record.get("label", {})
        values = label.get("alpha_log_returns", {})
        for horizon in ordered_horizons:
            value = values.get(f"{horizon}d") if isinstance(values, dict) else None
            if isinstance(value, int | float) and np.isfinite(float(value)):
                columns[horizon].append(float(value))
    scales: list[float] = []
    for horizon in ordered_horizons:
        values = np.asarray(columns[horizon], dtype=np.float64)
        if values.size < 4:
            raise ValueError(f"At least four train labels are required for horizon {horizon}")
        q25, q75 = np.quantile(values, [0.25, 0.75])
        iqr = float(q75 - q25)
        median = float(np.median(values))
        mad_scale = float(np.median(np.abs(values - median)) * 1.4826)
        scales.append(max(iqr, mad_scale, minimum_scale))
    return scales
