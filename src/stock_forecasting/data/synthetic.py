"""Reproducible synthetic daily OHLCV data for offline smoke tests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from fin_ts_multimodal.data.schema import normalize_ohlcv_frame

DEFAULT_SYMBOLS: Mapping[str, str] = {
    "AAPL.US": "stock",
    "MSFT.US": "stock",
    "NVDA.US": "stock",
    "SPY.US": "etf",
    "QQQ.US": "etf",
    "VTI.US": "etf",
}


def generate_synthetic_ohlcv(
    *,
    symbols: Mapping[str, str] | Sequence[str] = DEFAULT_SYMBOLS,
    periods: int = 600,
    start: str = "2018-01-02",
    seed: int = 42,
) -> pd.DataFrame:
    """Generate plausible, non-investable US stock/ETF daily bars.

    Each symbol uses independent random draws and deterministic volatility
    regimes. The series are intended for pipeline tests, not model evaluation.
    """

    if periods < 2:
        raise ValueError("periods must be at least 2.")
    if isinstance(symbols, Mapping):
        symbol_types = {
            (
                str(symbol).upper()
                if "." in str(symbol)
                else f"{str(symbol).upper()}.US"
            ): str(asset_type).lower()
            for symbol, asset_type in symbols.items()
        }
    else:
        symbol_types = {
            (
                str(symbol).upper()
                if "." in str(symbol)
                else f"{str(symbol).upper()}.US"
            ): "stock"
            for symbol in symbols
        }
    if not symbol_types:
        raise ValueError("At least one symbol is required.")

    dates = pd.bdate_range(start=start, periods=periods, tz="UTC")
    root_seed = np.random.SeedSequence(seed)
    symbol_seeds = root_seed.spawn(len(symbol_types))
    frames: list[pd.DataFrame] = []
    for index, (symbol, asset_type) in enumerate(symbol_types.items()):
        rng = np.random.default_rng(symbol_seeds[index])
        phase = np.arange(periods)
        regime = (phase // 80) % 4
        drift = np.choose(regime, [0.00035, -0.00015, 0.00005, 0.00055])
        volatility = np.choose(regime, [0.010, 0.019, 0.007, 0.014])
        if asset_type == "etf":
            volatility *= 0.72
        close_to_close = drift + volatility * rng.normal(size=periods)
        start_price = 55.0 + 35.0 * index
        close = start_price * np.exp(np.cumsum(close_to_close))

        overnight = rng.normal(0.0, volatility * 0.28, size=periods)
        open_price = np.r_[start_price, close[:-1]] * np.exp(overnight)
        intraday_width = np.abs(rng.normal(volatility * 0.75, volatility * 0.25, size=periods))
        high = np.maximum(open_price, close) * np.exp(intraday_width)
        low = np.minimum(open_price, close) * np.exp(-intraday_width)

        base_volume = (8_000_000 if asset_type == "etf" else 3_000_000) * (1.0 + index * 0.12)
        volume_multiplier = np.exp(rng.normal(0.0, 0.32, size=periods))
        median_volatility = max(float(np.median(volatility)), 1e-8)
        volatility_multiplier = 1.0 + np.abs(close_to_close) / median_volatility
        volume = np.rint(base_volume * volume_multiplier * volatility_multiplier).astype(np.int64)

        frames.append(
            pd.DataFrame(
                {
                    "timestamp": dates,
                    "symbol": symbol,
                    "asset_type": asset_type,
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                    "adjusted_close": close,
                    "split_adjusted_volume": volume,
                    "adjustment_source": "synthetic_no_actions",
                    "provider": "synthetic",
                    "market": "US",
                    "currency": "USD",
                    "source_symbol": symbol.removesuffix(".US"),
                    "is_active": True,
                    "dataset_profile": "us_only_eodhd",
                }
            )
        )
    return normalize_ohlcv_frame(pd.concat(frames, ignore_index=True))
