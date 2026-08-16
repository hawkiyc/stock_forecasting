"""Leakage-safe asset and benchmark window construction."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping, MutableMapping
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import pandas as pd

from stock_forecasting.data.adjustments import (
    asof_adjusted_window,
    ensure_adjustment_columns,
    execution_total_return,
)
from stock_forecasting.data.benchmarks import resolve_benchmark
from stock_forecasting.data.schema import normalize_ohlcv_frame

CONTEXT_FIELDS = ("open", "high", "low", "close", "volume")
DEFAULT_ALPHA_HORIZONS = tuple(range(3, 15))
PROCESSED_SCHEMA_VERSION = "3.0"
_PROVENANCE_FIELDS = (
    "provider",
    "market",
    "currency",
    "source_symbol",
    "is_active",
    "dataset_profile",
    "adjustment_source",
)


def _iso_timestamp(value: Any) -> str:
    return pd.Timestamp(value).isoformat()


def _context_payload(window: pd.DataFrame) -> dict[str, list[Any]]:
    payload: dict[str, list[Any]] = {
        "timestamp": [_iso_timestamp(value) for value in window["timestamp"]],
    }
    for field in CONTEXT_FIELDS:
        payload[field] = window[field].astype(float).tolist()
    return payload


def _record_metadata(symbol_frame: pd.DataFrame) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for field in _PROVENANCE_FIELDS:
        if field not in symbol_frame.columns:
            continue
        values = symbol_frame[field].dropna().unique().tolist()
        if len(values) == 1:
            value = values[0]
            metadata[field] = bool(value) if field == "is_active" else str(value)
    return metadata


def _validate_horizons(horizons: Iterable[int]) -> tuple[int, ...]:
    ordered = tuple(int(horizon) for horizon in horizons)
    if ordered != DEFAULT_ALPHA_HORIZONS:
        raise ValueError("alpha_horizons are fixed at trading days 3 through 14")
    return ordered


def _aligned_rows(frame: pd.DataFrame, timestamps: pd.Series) -> pd.DataFrame | None:
    indexed = frame.set_index("timestamp", drop=False)
    requested = pd.DatetimeIndex(timestamps)
    if not requested.isin(indexed.index).all():
        return None
    selected = indexed.loc[requested].reset_index(drop=True)
    if len(selected) != len(requested):
        raise ValueError("Benchmark timestamps are not unique")
    return selected


def _adjusted_transition_mask(frame: pd.DataFrame, threshold: float) -> np.ndarray:
    close = frame["adjusted_close"].to_numpy(dtype=np.float64)
    returns = np.diff(np.log(np.maximum(close, 1e-12)))
    return np.abs(returns) > threshold


def _increment(audit: Counter[str], reason: str) -> None:
    audit[reason] += 1


def build_causal_windows(
    frame: pd.DataFrame,
    *,
    window_size: int = 128,
    stride: int = 1,
    alpha_horizons: Iterable[int] = DEFAULT_ALPHA_HORIZONS,
    benchmark_mapping: Mapping[str, str] | None = None,
    max_abs_log_return: float = 0.5,
    audit: MutableMapping[str, Any] | None = None,
    # Legacy readiness sentinels remain accepted but do not define labels.
    target_horizon: int = 5,
    diagnostic_horizons: Iterable[int] = (1, 20),
    flat_volatility_multiplier: float = 0.25,
    workers: int = 1,
) -> list[dict[str, Any]]:
    """Build next-open multi-horizon alpha labels and dual OHLCV contexts.

    Model inputs contain only cutoff-inclusive adjusted OHLCV for the asset and
    its historical benchmark. Benchmark values after the cutoff are used only
    inside offline label construction and never serialized into either context.
    """

    if window_size < 2:
        raise ValueError("window_size must be at least 2")
    if stride < 1:
        raise ValueError("stride must be at least 1")
    if max_abs_log_return <= 0:
        raise ValueError("max_abs_log_return must be positive")
    if target_horizon != 5:
        raise ValueError("Legacy RunPod readiness sentinel target_horizon must remain 5")
    if sorted(set(int(value) for value in diagnostic_horizons)) != [1, 20]:
        raise ValueError("Legacy RunPod readiness diagnostic horizons must remain 1 and 20")
    if flat_volatility_multiplier < 0.0:
        raise ValueError("flat_volatility_multiplier must be non-negative")
    if workers < 1:
        raise ValueError("workers must be positive")
    horizons = _validate_horizons(alpha_horizons)
    maximum_horizon = max(horizons)

    normalized = ensure_adjustment_columns(normalize_ohlcv_frame(frame))
    symbol_frames = {
        str(symbol): rows.sort_values("timestamp", kind="stable").reset_index(drop=True)
        for symbol, rows in normalized.groupby("symbol", sort=True)
    }
    exclusions: Counter[str] = Counter()
    eligible_symbols: set[str] = set()
    benchmark_symbols: set[str] = set()
    records: list[dict[str, Any]] = []
    explicit_mapping = dict(benchmark_mapping or {})

    def build_symbol(item: tuple[str, pd.DataFrame]) -> tuple[
        list[dict[str, Any]], Counter[str], str | None, str | None
    ]:
        symbol, symbol_frame = item
        symbol_exclusions: Counter[str] = Counter()
        symbol_output: list[dict[str, Any]] = []
        metadata = _record_metadata(symbol_frame)
        asset_type = str(symbol_frame.loc[0, "asset_type"])
        decision = resolve_benchmark(
            symbol=symbol,
            asset_type=asset_type,
            market=str(metadata.get("market", "")),
            explicit_mapping=explicit_mapping,
        )
        if not decision.eligible or decision.benchmark_symbol is None:
            _increment(symbol_exclusions, decision.reason)
            return symbol_output, symbol_exclusions, None, None
        benchmark_symbol = decision.benchmark_symbol
        benchmark_frame = symbol_frames.get(benchmark_symbol)
        if benchmark_frame is None:
            _increment(symbol_exclusions, f"missing_benchmark:{benchmark_symbol}")
            return symbol_output, symbol_exclusions, None, None
        if len(symbol_frame) < window_size + maximum_horizon:
            _increment(symbol_exclusions, "insufficient_asset_history")
            return symbol_output, symbol_exclusions, None, None

        bad_transitions = _adjusted_transition_mask(symbol_frame, max_abs_log_return)
        final_cutoff_index = len(symbol_frame) - maximum_horizon - 1
        for cutoff_index in range(window_size - 1, final_cutoff_index + 1, stride):
            start_index = cutoff_index - window_size + 1
            final_label_index = cutoff_index + maximum_horizon
            if bad_transitions[start_index:final_label_index].any():
                _increment(symbol_exclusions, "extreme_adjusted_transition")
                continue

            observed_raw = symbol_frame.iloc[start_index : cutoff_index + 1].copy()
            benchmark_observed_raw = _aligned_rows(benchmark_frame, observed_raw["timestamp"])
            if benchmark_observed_raw is None:
                _increment(symbol_exclusions, "benchmark_context_calendar_gap")
                continue
            holding_dates = symbol_frame.loc[
                cutoff_index + 1 : cutoff_index + maximum_horizon,
                "timestamp",
            ].reset_index(drop=True)
            entry_at = pd.Timestamp(holding_dates.iloc[0])
            exit_timestamps = {
                horizon: pd.Timestamp(holding_dates.iloc[horizon - 1])
                for horizon in horizons
            }
            if _aligned_rows(benchmark_frame, holding_dates) is None:
                _increment(symbol_exclusions, "benchmark_label_calendar_gap")
                continue

            asset_returns: dict[str, float] = {}
            benchmark_returns: dict[str, float] = {}
            alpha_log_returns: dict[str, float] = {}
            end_at: dict[str, str] = {}
            valid_label = True
            for horizon in horizons:
                exit_at = exit_timestamps[horizon]
                try:
                    asset_return = execution_total_return(
                        symbol_frame,
                        entry_at=entry_at,
                        exit_at=exit_at,
                    )
                    benchmark_return = execution_total_return(
                        benchmark_frame,
                        entry_at=entry_at,
                        exit_at=exit_at,
                    )
                except ValueError:
                    valid_label = False
                    break
                key = f"{horizon}d"
                asset_returns[key] = asset_return
                benchmark_returns[key] = benchmark_return
                alpha_log_returns[key] = float(
                    math.log1p(asset_return) - math.log1p(benchmark_return)
                )
                end_at[key] = _iso_timestamp(exit_at)
            if not valid_label:
                _increment(symbol_exclusions, "invalid_execution_return")
                continue

            observed = asof_adjusted_window(observed_raw)
            benchmark_observed = asof_adjusted_window(benchmark_observed_raw)
            cutoff_timestamp = pd.Timestamp(symbol_frame.loc[cutoff_index, "timestamp"])
            benchmark_metadata = _record_metadata(benchmark_frame)
            label = {
                "kind": "benchmark_relative_adjusted_log_return",
                "horizons": list(horizons),
                "entry_at": _iso_timestamp(entry_at),
                "entry_price_field": "raw_regular_session_open",
                "entry_day_counts_as_holding_day_one": True,
                "end_at": end_at,
                "alpha_log_returns": alpha_log_returns,
                "asset_total_returns": asset_returns,
                "benchmark_total_returns": benchmark_returns,
            }
            symbol_output.append(
                {
                    "schema_version": PROCESSED_SCHEMA_VERSION,
                    "sample_id": f"{symbol}-{cutoff_timestamp.strftime('%Y%m%dT%H%M%SZ')}",
                    "symbol": symbol,
                    "asset_type": asset_type,
                    "benchmark_symbol": benchmark_symbol,
                    "window_start_at": _iso_timestamp(symbol_frame.loc[start_index, "timestamp"]),
                    "cutoff_at": _iso_timestamp(cutoff_timestamp),
                    "context": _context_payload(observed),
                    "benchmark_context": _context_payload(benchmark_observed),
                    "label": label,
                    "diagnostics": {
                        "capm_abnormal_return": None,
                        "capm_status": "reserved_for_diagnostic_ablation",
                    },
                    "metadata": {
                        **metadata,
                        "benchmark_policy": decision.policy,
                        "benchmark_assignment_reason": decision.reason,
                        "benchmark_provider": benchmark_metadata.get("provider", "unknown"),
                        "benchmark_market": benchmark_metadata.get("market", "unknown"),
                        "input_adjustment_cutoff": "cutoff_at",
                    },
                }
            )
        if not symbol_output:
            _increment(symbol_exclusions, "no_valid_windows")
        return symbol_output, symbol_exclusions, symbol, benchmark_symbol

    items = list(symbol_frames.items())
    if workers == 1:
        symbol_results = list(map(build_symbol, items))
    else:
        with ThreadPoolExecutor(
            max_workers=min(workers, max(len(items), 1)),
            thread_name_prefix="causal-window",
        ) as executor:
            symbol_results = list(executor.map(build_symbol, items))
    for symbol_output, symbol_exclusions, eligible_symbol, benchmark_symbol in symbol_results:
        records.extend(symbol_output)
        exclusions.update(symbol_exclusions)
        if eligible_symbol is not None:
            eligible_symbols.add(eligible_symbol)
        if benchmark_symbol is not None:
            benchmark_symbols.add(benchmark_symbol)

    if audit is not None:
        audit.clear()
        audit.update(
            {
                "eligible_training_symbols": sorted(eligible_symbols),
                "benchmark_symbols": sorted(benchmark_symbols),
                "excluded_counts_by_reason": dict(sorted(exclusions.items())),
                "written_windows": len(records),
                "alpha_horizons": list(horizons),
                "execution_workers": workers,
            }
        )
    return sorted(records, key=lambda record: (record["cutoff_at"], record["symbol"]))
