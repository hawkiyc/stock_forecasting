"""Keep unit tests independent from the active RunPod Pod environment."""

import os

import numpy as np
import pandas as pd
import pytest

from stock_forecasting.data.splits import chronological_split
from stock_forecasting.data.windows import build_causal_windows

_PERSISTENT_ENVIRONMENT_KEYS = frozenset(
    {
        "NETWORK_VOLUME_ROOT",
        "PROJECT_ROOT",
        "DATA_ROOT",
        "CACHE_ROOT",
        "MODEL_ROOT",
        "LOG_ROOT",
        "LIFECYCLE_ROOT",
        "SAVED_MODEL_ROOT",
        "CHECKPOINT_ROOT",
        "WANDB_DIR",
        "XDG_CACHE_HOME",
        "HF_HOME",
        "TRANSFORMERS_CACHE",
        "TORCH_HOME",
        "POETRY_CACHE_DIR",
        "PIP_CACHE_DIR",
        "KRONOS_ROOT",
        "MODEL_CACHE_MANIFEST",
    }
)
_ISOLATED_ENVIRONMENT_PREFIXES = ("RUNPOD_", "WANDB_", "HF_", "AWS_")


@pytest.fixture(autouse=True)
def isolate_runpod_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent Pod paths and credentials from leaking into subprocess fixtures."""

    for name in tuple(os.environ):
        if name in _PERSISTENT_ENVIRONMENT_KEYS or name.startswith(_ISOLATED_ENVIRONMENT_PREFIXES):
            monkeypatch.delenv(name, raising=False)


@pytest.fixture
def market_frame() -> pd.DataFrame:
    """Return deterministic assets plus exact US and Taiwan benchmarks."""

    dates = pd.bdate_range("2022-01-03", periods=360, tz="UTC")
    frames: list[pd.DataFrame] = []
    instruments = (
        ("AAPL.US", "stock", "US", "eodhd", 150.0, 0.00045),
        ("VTI.US", "etf", "US", "eodhd", 220.0, 0.00025),
        ("0050.TW", "etf", "TWSE", "twse_official", 120.0, 0.00040),
        ("TAIEX.TW", "index", "TWSE", "twse_official", 15000.0, 0.00022),
    )
    for symbol, asset_type, market, provider, base_price, drift in instruments:
        index = np.arange(len(dates), dtype=np.float64)
        close = base_price * np.exp(drift * index + 0.01 * np.sin(index / 13.0))
        total_return_factor = (
            np.exp(0.00005 * index) if asset_type == "index" else np.ones_like(index)
        )
        frames.append(
            pd.DataFrame(
                {
                    "timestamp": dates,
                    "symbol": symbol,
                    "asset_type": asset_type,
                    "open": close * 0.998,
                    "high": close * 1.006,
                    "low": close * 0.994,
                    "close": close,
                    "volume": 1_000_000.0 + index * 100.0,
                    "adjusted_close": close * total_return_factor,
                    "split_adjusted_volume": 1_000_000.0 + index * 100.0,
                    "adjustment_source": (
                        "official_total_return_index" if asset_type == "index" else "fixture"
                    ),
                    "provider": provider,
                    "market": market,
                    "currency": "USD" if market == "US" else "TWD",
                    "source_symbol": symbol.split(".", maxsplit=1)[0],
                    "is_active": True,
                    "dataset_profile": "us_tw_eodhd",
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def window_records(market_frame: pd.DataFrame) -> list[dict[str, object]]:
    """Return processed schema 3.0 records with non-empty purged splits."""

    windows = build_causal_windows(
        market_frame,
        window_size=32,
        stride=2,
    )
    return chronological_split(
        windows,
        train_fraction=0.70,
        validation_fraction=0.15,
        purge_bars=20,
        embargo_bars=14,
    )
