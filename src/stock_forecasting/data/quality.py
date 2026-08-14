"""Deterministic OHLCV quality diagnostics used before window generation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from stock_forecasting.data.schema import normalize_ohlcv_frame


def assess_ohlcv_quality(
    frame: pd.DataFrame,
    *,
    max_abs_log_return: float,
) -> dict[str, Any]:
    """Report source issues without silently fabricating or forward-filling bars."""

    if max_abs_log_return <= 0.0:
        raise ValueError("max_abs_log_return must be positive")
    normalized = normalize_ohlcv_frame(frame)
    extreme_transitions = 0
    long_calendar_gaps = 0
    symbol_date_ranges: dict[str, dict[str, str | int]] = {}
    return_basis = "adjusted_close" if "adjusted_close" in normalized else "close"
    for symbol, rows in normalized.groupby("symbol", sort=True):
        ordered = rows.sort_values("timestamp", kind="stable")
        close = ordered[return_basis].to_numpy(dtype=np.float64)
        log_returns = np.diff(np.log(np.maximum(close, 1e-12)))
        extreme_transitions += int((np.abs(log_returns) > max_abs_log_return).sum())
        gaps = ordered["timestamp"].diff().dt.days.fillna(0).to_numpy()
        long_calendar_gaps += int((gaps > 10).sum())
        symbol_date_ranges[str(symbol)] = {
            "rows": len(ordered),
            "start": pd.Timestamp(ordered["timestamp"].iloc[0]).isoformat(),
            "end": pd.Timestamp(ordered["timestamp"].iloc[-1]).isoformat(),
        }
    provider_counts = (
        normalized["provider"].value_counts().sort_index().to_dict()
        if "provider" in normalized
        else {}
    )
    market_counts = (
        normalized["market"].value_counts().sort_index().to_dict() if "market" in normalized else {}
    )
    return {
        "rows": len(normalized),
        "symbols": int(normalized["symbol"].nunique()),
        "zero_volume_rows": int((normalized["volume"] == 0).sum()),
        "extreme_return_transitions": extreme_transitions,
        "return_transition_basis": return_basis,
        "excluded_transition_threshold_abs_log_return": max_abs_log_return,
        "calendar_gaps_over_10_days": long_calendar_gaps,
        "provider_row_counts": {str(key): int(value) for key, value in provider_counts.items()},
        "market_row_counts": {str(key): int(value) for key, value in market_counts.items()},
        "symbol_date_ranges": symbol_date_ranges,
    }
