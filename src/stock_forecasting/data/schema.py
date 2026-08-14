"""Canonical OHLCV schema and validation utilities."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = (
    "timestamp",
    "symbol",
    "asset_type",
    "open",
    "high",
    "low",
    "close",
    "volume",
)
OPTIONAL_COLUMNS = (
    "adjusted_close",
    "split_adjusted_volume",
    "adjustment_source",
    "provider",
    "market",
    "currency",
    "source_symbol",
    "is_active",
    "dataset_profile",
)
PRICE_COLUMNS = ("open", "high", "low", "close")
SUPPORTED_ASSET_TYPES = frozenset({"stock", "etf", "future", "option", "index", "other"})

_ALIASES = {
    "date": "timestamp",
    "datetime": "timestamp",
    "time": "timestamp",
    "ticker": "symbol",
    "asset": "symbol",
    "type": "asset_type",
    "adj close": "adjusted_close",
    "adj_close": "adjusted_close",
    "adjclose": "adjusted_close",
    "code": "source_symbol",
}


class MarketDataValidationError(ValueError):
    """Raised when a market-data frame violates the canonical schema."""


def _canonical_column_name(column: object) -> str:
    name = str(column).strip().lower().replace("-", "_")
    return _ALIASES.get(name, name.replace(" ", "_"))


def normalize_ohlcv_frame(
    frame: pd.DataFrame,
    *,
    default_symbol: str | None = None,
    default_asset_type: str = "stock",
) -> pd.DataFrame:
    """Normalize common vendor column names and validate canonical OHLCV data.

    The function never forward-fills prices because that would create synthetic
    observations that were not present in the source data.
    """

    if frame.empty:
        raise MarketDataValidationError("Market data is empty.")

    normalized = frame.rename(
        columns={column: _canonical_column_name(column) for column in frame.columns}
    ).copy()
    if "symbol" not in normalized.columns and default_symbol:
        normalized["symbol"] = default_symbol
    if "asset_type" not in normalized.columns:
        normalized["asset_type"] = default_asset_type

    missing = sorted(set(REQUIRED_COLUMNS).difference(normalized.columns))
    if missing:
        raise MarketDataValidationError(f"Missing required OHLCV columns: {', '.join(missing)}")

    normalized["timestamp"] = pd.to_datetime(normalized["timestamp"], errors="coerce", utc=True)
    normalized["symbol"] = normalized["symbol"].astype("string").str.strip().str.upper()
    normalized["asset_type"] = normalized["asset_type"].astype("string").str.strip().str.lower()
    for column in (
        "adjustment_source",
        "provider",
        "market",
        "currency",
        "source_symbol",
        "dataset_profile",
    ):
        if column in normalized.columns:
            normalized[column] = normalized[column].astype("string").str.strip()
    if "is_active" in normalized.columns:
        normalized["is_active"] = normalized["is_active"].astype("boolean")

    numeric_columns = [*PRICE_COLUMNS, "volume"]
    if "adjusted_close" in normalized.columns:
        numeric_columns.append("adjusted_close")
    if "split_adjusted_volume" in normalized.columns:
        numeric_columns.append("split_adjusted_volume")
    for column in numeric_columns:
        normalized[column] = pd.to_numeric(
            normalized[column], errors="coerce"
        ).astype("float64")

    _validate_ohlcv_frame(normalized)

    selected = [*REQUIRED_COLUMNS]
    selected.extend(column for column in OPTIONAL_COLUMNS if column in normalized.columns)
    return (
        normalized.loc[:, selected]
        .sort_values(["symbol", "timestamp"], kind="stable")
        .reset_index(drop=True)
    )


def _validate_ohlcv_frame(frame: pd.DataFrame) -> None:
    if frame["timestamp"].isna().any():
        raise MarketDataValidationError("timestamp contains invalid or missing values.")
    if frame["symbol"].isna().any() or (frame["symbol"].str.len() == 0).any():
        raise MarketDataValidationError("symbol contains empty values.")
    unsupported = sorted(set(frame["asset_type"].dropna()).difference(SUPPORTED_ASSET_TYPES))
    if unsupported:
        raise MarketDataValidationError(f"Unsupported asset_type values: {', '.join(unsupported)}")
    if frame["asset_type"].isna().any():
        raise MarketDataValidationError("asset_type contains missing values.")
    for column in (
        "adjustment_source",
        "provider",
        "market",
        "currency",
        "source_symbol",
        "dataset_profile",
    ):
        if column in frame.columns and (
            frame[column].isna().any() or (frame[column].str.len() == 0).any()
        ):
            raise MarketDataValidationError(f"{column} contains empty values.")
    numeric_columns = [*PRICE_COLUMNS, "volume"]
    if "adjusted_close" in frame.columns:
        numeric_columns.append("adjusted_close")
    if "split_adjusted_volume" in frame.columns:
        numeric_columns.append("split_adjusted_volume")
    numeric = frame[numeric_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise MarketDataValidationError("OHLCV values must be finite and non-missing.")
    if (frame[list(PRICE_COLUMNS)] <= 0).any(axis=None):
        raise MarketDataValidationError("OHLC prices must be strictly positive.")
    if "adjusted_close" in frame.columns and (frame["adjusted_close"] <= 0).any():
        raise MarketDataValidationError("adjusted_close must be strictly positive.")
    if "split_adjusted_volume" in frame.columns and (frame["split_adjusted_volume"] < 0).any():
        raise MarketDataValidationError("split_adjusted_volume must be non-negative.")
    if (frame["volume"] < 0).any():
        raise MarketDataValidationError("volume must be non-negative.")

    expected_high = frame[["open", "low", "close"]].max(axis=1)
    expected_low = frame[["open", "high", "close"]].min(axis=1)
    tolerance = 1e-8
    if (frame["high"] + tolerance < expected_high).any():
        raise MarketDataValidationError(
            "high must be greater than or equal to open, low, and close."
        )
    if (frame["low"] - tolerance > expected_low).any():
        raise MarketDataValidationError("low must be less than or equal to open, high, and close.")
    if frame.duplicated(["symbol", "timestamp"]).any():
        raise MarketDataValidationError("Duplicate (symbol, timestamp) rows are not allowed.")


def read_market_data(
    path: str | Path,
    *,
    default_symbol: str | None = None,
    default_asset_type: str = "stock",
) -> pd.DataFrame:
    """Read a CSV or Parquet file into the canonical OHLCV schema."""

    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(source)
    elif suffix in {".parquet", ".pq"}:
        try:
            frame = pd.read_parquet(source)
        except ImportError as error:
            raise RuntimeError(
                "Reading Parquet requires pyarrow. Install the parquet dependency group."
            ) from error
    else:
        raise ValueError(f"Unsupported market-data format: {source.suffix}. Use CSV or Parquet.")
    inferred_symbol = default_symbol or source.stem.split("_")[0]
    return normalize_ohlcv_frame(
        frame,
        default_symbol=inferred_symbol,
        default_asset_type=default_asset_type,
    )


def write_market_data(frame: pd.DataFrame, path: str | Path) -> Path:
    """Validate and write canonical market data as CSV or Parquet."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    normalized = normalize_ohlcv_frame(frame)
    suffix = destination.suffix.lower()
    if suffix == ".csv":
        normalized.to_csv(destination, index=False)
    elif suffix in {".parquet", ".pq"}:
        try:
            normalized.to_parquet(destination, index=False)
        except ImportError as error:
            raise RuntimeError(
                "Writing Parquet requires pyarrow. Install the parquet dependency group."
            ) from error
    else:
        raise ValueError(
            f"Unsupported market-data format: {destination.suffix}. Use CSV or Parquet."
        )
    return destination


def asset_type_map(symbols: list[str], etf_symbols: set[str] | None = None) -> Mapping[str, str]:
    """Build a normalized symbol-to-asset-type mapping for download tools."""

    etfs = {symbol.upper() for symbol in (etf_symbols or set())}
    return {symbol.upper(): "etf" if symbol.upper() in etfs else "stock" for symbol in symbols}
