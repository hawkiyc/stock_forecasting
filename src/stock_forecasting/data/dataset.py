"""PyTorch dataset for conditional asset-versus-benchmark alpha windows."""

from __future__ import annotations

import json
import math
from collections import OrderedDict
from collections.abc import Iterator, Sequence
from math import gcd
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, Sampler

from stock_forecasting.data.adjustments import (
    ADJUSTED_CLOSE_FIELD,
    ADJUSTED_VOLUME_FIELD,
    asof_adjusted_window,
)
from stock_forecasting.data.benchmarks import is_training_target_security
from stock_forecasting.data.horizons import (
    DEFAULT_ALPHA_HORIZONS,
    MAX_ALPHA_HORIZON,
    validate_alpha_horizons,
)
from stock_forecasting.data.io import read_processed_records
from stock_forecasting.data.manifest import SUPPORTED_H_START
from stock_forecasting.data.schema import TRAINING_TARGET_ASSET_TYPES
from stock_forecasting.data.windows import (
    CONTEXT_FIELDS,
    PROCESSED_SCHEMA_VERSION,
)

TIMESTAMP_FEATURE_FIELDS = ("minute", "hour", "weekday", "day", "month")


def _context_payload(window: pd.DataFrame) -> dict[str, list[Any]]:
    return {
        "timestamp": [pd.Timestamp(value).isoformat() for value in window["timestamp"]],
        **{field: window[field].astype(float).tolist() for field in CONTEXT_FIELDS},
    }


def _series_from_values(values: np.ndarray, series_mode: str) -> torch.Tensor:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != len(CONTEXT_FIELDS):
        raise ValueError("OHLCV context has an invalid shape")
    if not np.isfinite(values).all():
        raise ValueError("OHLCV context must contain only finite values")
    if series_mode == "relative":
        reference_price = max(float(values[-1, 3]), 1e-12)
        values[:, :4] = np.log(np.maximum(values[:, :4], 1e-12) / reference_price)
        log_volume = np.log1p(np.maximum(values[:, 4], 0.0))
        scale = float(log_volume.std())
        values[:, 4] = (log_volume - float(log_volume.mean())) / (scale if scale > 1e-6 else 1.0)
    return torch.from_numpy(values)


def _series_from_context(context: dict[str, Any], series_mode: str) -> torch.Tensor:
    values = np.column_stack(
        [np.asarray(context[field], dtype=np.float32) for field in CONTEXT_FIELDS]
    )
    return _series_from_values(values, series_mode)


def _series_from_asof_adjusted_frame(
    frame: pd.DataFrame,
    series_mode: str,
) -> torch.Tensor:
    """Create cutoff-causal OHLCV tensors without temporary DataFrame copies."""

    raw_prices = frame.loc[:, list(CONTEXT_FIELDS[:4])].to_numpy(
        dtype=np.float64,
        copy=False,
    )
    raw_close = raw_prices[:, 3]
    adjusted_close = (
        frame[ADJUSTED_CLOSE_FIELD].to_numpy(dtype=np.float64, copy=False)
        if ADJUSTED_CLOSE_FIELD in frame
        else raw_close
    )
    price_factor = adjusted_close / np.maximum(raw_close, 1e-12)
    cutoff_price_factor = float(price_factor[-1])
    if not np.isfinite(cutoff_price_factor) or cutoff_price_factor <= 0.0:
        raise ValueError("Cutoff total-return adjustment factor is invalid")

    raw_volume = frame[CONTEXT_FIELDS[4]].to_numpy(dtype=np.float64, copy=False)
    adjusted_volume = (
        frame[ADJUSTED_VOLUME_FIELD].to_numpy(dtype=np.float64, copy=False)
        if ADJUSTED_VOLUME_FIELD in frame
        else raw_volume
    )
    volume_factor = np.divide(
        adjusted_volume,
        raw_volume,
        out=np.ones_like(adjusted_volume),
        where=raw_volume > 0.0,
    )
    cutoff_volume_factor = float(volume_factor[-1])
    if not np.isfinite(cutoff_volume_factor) or cutoff_volume_factor <= 0.0:
        raise ValueError("Cutoff share adjustment factor is invalid")

    values = np.empty((len(frame), len(CONTEXT_FIELDS)), dtype=np.float32)
    values[:, :4] = raw_prices * (price_factor / cutoff_price_factor)[:, None]
    values[:, 4] = raw_volume * (volume_factor / cutoff_volume_factor)
    return _series_from_values(values, series_mode)


def _timestamp_features(timestamps: pd.DatetimeIndex) -> torch.Tensor:
    values = np.column_stack(
        [
            timestamps.minute,
            timestamps.hour,
            timestamps.dayofweek,
            timestamps.day,
            timestamps.month,
        ]
    ).astype(np.int64)
    return torch.from_numpy(values)


def _timestamp_features_from_context(context: dict[str, Any]) -> torch.Tensor:
    timestamps = pd.DatetimeIndex(pd.to_datetime(context["timestamp"], utc=True))
    return _timestamp_features(timestamps)


def _item_from_record(
    record: dict[str, Any],
    *,
    horizons: tuple[int, ...],
    series_mode: str,
) -> dict[str, Any]:
    context = record["context"]
    benchmark_context = record["benchmark_context"]
    label = record["label"]
    if tuple(label.get("horizons", ())) != horizons:
        raise ValueError("Record alpha horizons do not match the model contract")
    alpha_values = label.get("alpha_log_returns")
    if not isinstance(alpha_values, dict):
        raise ValueError("Record has no alpha_log_returns mapping")
    target = np.asarray(
        [alpha_values[f"{horizon}d"] for horizon in horizons],
        dtype=np.float32,
    )
    if not np.isfinite(target).all():
        raise ValueError("Alpha targets must be finite")
    metadata = record.get("metadata", {})
    return {
        "asset_series": _series_from_context(context, series_mode),
        "asset_timestamp_features": _timestamp_features_from_context(context),
        "benchmark_series": _series_from_context(benchmark_context, series_mode),
        "benchmark_timestamp_features": _timestamp_features_from_context(benchmark_context),
        "target_alpha": torch.from_numpy(target),
        "sample_id": str(record["sample_id"]),
        "symbol": str(record["symbol"]),
        "benchmark_symbol": str(record["benchmark_symbol"]),
        "asset_type": str(record["asset_type"]),
        "market": str(metadata.get("market", "unknown")),
        "provider": str(metadata.get("provider", "unknown")),
        "dataset_profile": str(metadata.get("dataset_profile", "unknown")),
        "cutoff_at": str(record["cutoff_at"]),
        "diagnostics": record["diagnostics"],
    }


class FinancialWindowDataset(Dataset[dict[str, Any]]):
    """Expose paired historical OHLCV streams and continuous alpha targets."""

    def __init__(
        self,
        records: Sequence[dict[str, Any]] | str | Path,
        *,
        split: Literal["train", "validation", "test"] | None = None,
        series_mode: Literal["raw", "relative"] = "raw",
        alpha_horizons: Sequence[int] = DEFAULT_ALPHA_HORIZONS,
    ) -> None:
        if series_mode not in {"raw", "relative"}:
            raise ValueError("series_mode must be 'raw' or 'relative'")
        loaded = read_processed_records(records) if isinstance(records, (str, Path)) else records
        self.records = [
            record for record in loaded if split is None or record.get("split") == split
        ]
        if not self.records:
            raise ValueError("No records remain after applying the requested split")
        invalid_versions = sorted(
            {
                str(record.get("schema_version"))
                for record in self.records
                if record.get("schema_version") != PROCESSED_SCHEMA_VERSION
            }
        )
        if invalid_versions:
            raise ValueError(
                f"Conditional alpha training requires processed schema "
                f"{PROCESSED_SCHEMA_VERSION}; found " + ", ".join(invalid_versions)
            )
        invalid_asset_types = sorted(
            {
                str(record.get("asset_type", "")).strip().lower()
                for record in self.records
                if str(record.get("asset_type", "")).strip().lower()
                not in TRAINING_TARGET_ASSET_TYPES
            }
        )
        if invalid_asset_types:
            raise ValueError(
                "Conditional alpha training supports only common stock/ADR/TDR and "
                "allowlisted unleveraged equity ETF targets; found "
                + ", ".join(value or "<missing>" for value in invalid_asset_types)
            )
        invalid_security_symbols: set[str] = set()
        for record in self.records:
            metadata = record.get("metadata")
            market = metadata.get("market", "") if isinstance(metadata, dict) else ""
            symbol = str(record.get("symbol", "")).strip().upper()
            if not is_training_target_security(
                symbol=symbol,
                asset_type=str(record.get("asset_type", "")),
                market=str(market),
            ):
                invalid_security_symbols.add(symbol or "<missing>")
        if invalid_security_symbols:
            raise ValueError(
                "Conditional alpha training rejected targets outside the common-stock/ADR/TDR "
                "and audited unleveraged-equity-ETF scope: "
                + ", ".join(sorted(invalid_security_symbols))
            )
        self.split = split
        self.series_mode = series_mode
        self.horizons = validate_alpha_horizons(alpha_horizons)

    def __len__(self) -> int:
        return len(self.records)

    def _series_from_context(self, context: dict[str, Any]) -> torch.Tensor:
        return _series_from_context(context, self.series_mode)

    @staticmethod
    def _timestamp_features_from_context(context: dict[str, Any]) -> torch.Tensor:
        return _timestamp_features_from_context(context)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return _item_from_record(
            self.records[index],
            horizons=self.horizons,
            series_mode=self.series_mode,
        )


class LazyFinancialWindowDataset(Dataset[dict[str, Any]]):
    """Resolve a valid cutoff and calculate its context and labels on demand."""

    def __init__(
        self,
        bar_store_root: str | Path,
        *,
        split: Literal["train", "validation", "test"],
        window_size: int = 128,
        h_start: int = 3,
        series_mode: Literal["raw", "relative"] = "raw",
        symbol_cache_size: int = 32,
    ) -> None:
        if series_mode not in {"raw", "relative"}:
            raise ValueError("series_mode must be 'raw' or 'relative'")
        if symbol_cache_size < 2:
            raise ValueError("symbol_cache_size must be at least two")
        if isinstance(h_start, bool) or h_start not in SUPPORTED_H_START:
            raise ValueError("h_start must be 1, 2, or 3")
        self.root = Path(bar_store_root).resolve(strict=True)
        manifest_path = self.root / "bar-store.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as error:
            raise ValueError(f"Bar-store manifest is invalid: {manifest_path}") from error
        execution = manifest.get("execution")
        identity = manifest.get("identity")
        requested_window_size = int(window_size)
        if (
            manifest.get("schema_version") != "1.0"
            or manifest.get("kind") != "symbol-oriented-ohlcv-bar-store"
            or manifest.get("state") != "ready"
            or not isinstance(execution, dict)
            or execution.get("window_materialized") is not False
            or execution.get("labels_materialized") is not False
            or not isinstance(identity, dict)
            or identity.get("window_size") != requested_window_size
            or identity.get("max_horizon") != MAX_ALPHA_HORIZON
        ):
            raise ValueError("Lazy training requires a ready non-materialized bar store")
        self.window_size = requested_window_size
        self.horizons = validate_alpha_horizons(range(int(h_start), 15))
        self.series_mode = series_mode
        self.split = split
        self.symbol_cache_size = symbol_cache_size
        ranges = pd.read_parquet(self.root / "cutoff-ranges.parquet")
        self.ranges = ranges.loc[ranges["split"] == split].reset_index(drop=True)
        if self.ranges.empty:
            raise ValueError(f"No lazy cutoff ranges remain for split={split}")
        # Validate compact ranges, not an expanded list of millions of windows.
        # Reject reordered/overlapping indexes rather than silently changing old identities.
        canonical = self.ranges.sort_values(["symbol", "start_index"], kind="stable")
        if not canonical.index.equals(self.ranges.index):
            raise ValueError("Lazy cutoff ranges are not in canonical symbol/cutoff order")
        previous_stops = self.ranges.groupby("symbol", sort=False)["stop_index"].shift()
        if (self.ranges["start_index"] < previous_stops).any():
            raise ValueError("Lazy cutoff ranges overlap and would duplicate windows")
        counts = self.ranges["count"].to_numpy(dtype=np.int64)
        starts = self.ranges["start_index"].to_numpy(dtype=np.int64)
        stops = self.ranges["stop_index"].to_numpy(dtype=np.int64)
        if (
            (counts <= 0).any()
            or (counts != stops - starts).any()
            or (starts < self.window_size - 1).any()
        ):
            raise ValueError("Lazy cutoff ranges have invalid index bounds")
        split_counts = manifest.get("split_counts")
        if not isinstance(split_counts, dict) or split_counts.get(split) != int(counts.sum()):
            raise ValueError("Lazy cutoff ranges disagree with the bar-store split count")
        self._range_ends = np.cumsum(counts, dtype=np.int64)
        index = pd.read_parquet(self.root / "symbol-index.parquet")
        if index["symbol"].duplicated().any():
            raise ValueError("Bar-store symbol index contains duplicate symbols")
        self._index = {str(row["symbol"]): row for row in index.to_dict(orient="records")}
        range_symbols = self.ranges["symbol"].astype(str)
        missing_symbols = sorted(set(range_symbols) - set(self._index))
        if missing_symbols:
            raise ValueError("Lazy cutoff ranges reference symbols absent from the index")
        row_counts = range_symbols.map(
            {symbol: int(row["row_count"]) for symbol, row in self._index.items()}
        ).to_numpy(dtype=np.int64)
        if (stops + MAX_ALPHA_HORIZON > row_counts).any():
            raise ValueError("Lazy cutoff ranges exceed the indexed symbol histories")
        invalid_targets = sorted(
            {
                symbol
                for symbol in set(range_symbols)
                if not bool(self._index[symbol].get("eligible"))
                or str(self._index[symbol].get("benchmark_symbol", "")) not in self._index
            }
        )
        if invalid_targets:
            raise ValueError("Lazy cutoff ranges contain ineligible or unaligned targets")
        self._cache: OrderedDict[str, pd.DataFrame] = OrderedDict()
        self._timestamp_indexes: OrderedDict[str, pd.DatetimeIndex] = OrderedDict()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_cache"] = OrderedDict()
        state["_timestamp_indexes"] = OrderedDict()
        return state

    def __len__(self) -> int:
        return int(self._range_ends[-1])

    def _resolve_cutoff(self, index: int) -> tuple[dict[str, Any], int]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        range_index = int(np.searchsorted(self._range_ends, index, side="right"))
        previous_end = 0 if range_index == 0 else int(self._range_ends[range_index - 1])
        row = self.ranges.iloc[range_index].to_dict()
        cutoff_index = int(row["start_index"]) + index - previous_end
        return row, cutoff_index

    def _load_symbol(self, symbol: str) -> pd.DataFrame:
        cached = self._cache.pop(symbol, None)
        if cached is not None:
            self._cache[symbol] = cached
            timestamp_index = self._timestamp_indexes.pop(symbol)
            self._timestamp_indexes[symbol] = timestamp_index
            return cached
        row = self._index.get(symbol)
        if row is None:
            raise ValueError(f"Bar-store index is missing symbol: {symbol}")
        try:
            import pyarrow.parquet as pq
        except ImportError as error:
            raise RuntimeError("Lazy bar-store loading requires pyarrow") from error
        shard = self.root / str(row["shard_relative_path"])
        frame = pq.ParquetFile(shard).read_row_group(int(row["row_group"])).to_pandas()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
        self._cache[symbol] = frame
        self._timestamp_indexes[symbol] = pd.DatetimeIndex(frame["timestamp"])
        while len(self._cache) > self.symbol_cache_size:
            evicted_symbol, _evicted_frame = self._cache.popitem(last=False)
            self._timestamp_indexes.pop(evicted_symbol)
        return frame

    def _aligned_rows(
        self,
        symbol: str,
        frame: pd.DataFrame,
        timestamps: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        positions = self._timestamp_indexes[symbol].get_indexer(timestamps)
        if (positions < 0).any():
            raise RuntimeError("Prepared cutoff range lost exact benchmark calendar coverage")
        selected = frame.iloc[positions].reset_index(drop=True)
        if len(selected) != len(timestamps):
            raise RuntimeError("Benchmark timestamps are not unique")
        return selected

    def _sample_frames(
        self,
        ordinal: int,
    ) -> tuple[
        str,
        dict[str, Any],
        str,
        pd.DataFrame,
        pd.DataFrame,
        int,
        pd.DatetimeIndex,
        pd.DataFrame,
    ]:
        range_row, cutoff_index = self._resolve_cutoff(ordinal)
        symbol = str(range_row["symbol"])
        index_row = self._index[symbol]
        benchmark_symbol = str(index_row["benchmark_symbol"])
        frame = self._load_symbol(symbol)
        benchmark = self._load_symbol(benchmark_symbol)
        start_index = cutoff_index - self.window_size + 1
        final_index = cutoff_index + 14
        if start_index < 0 or final_index >= len(frame):
            raise RuntimeError("Prepared cutoff index is outside the symbol history")
        holding_dates = pd.DatetimeIndex(
            frame.iloc[cutoff_index + 1 : final_index + 1]["timestamp"]
        )
        benchmark_holding = self._aligned_rows(
            benchmark_symbol,
            benchmark,
            holding_dates,
        )
        return (
            symbol,
            index_row,
            benchmark_symbol,
            frame,
            benchmark,
            cutoff_index,
            holding_dates,
            benchmark_holding,
        )

    def _target_components(
        self,
        *,
        frame: pd.DataFrame,
        cutoff_index: int,
        holding_dates: pd.DatetimeIndex,
        benchmark_holding: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]:
        asset_holding = frame.iloc[cutoff_index + 1 : cutoff_index + MAX_ALPHA_HORIZON + 1]
        asset_close = asset_holding["close"].to_numpy(dtype=np.float64)
        asset_adjusted_close = asset_holding["adjusted_close"].to_numpy(dtype=np.float64)
        benchmark_close = benchmark_holding["close"].to_numpy(dtype=np.float64)
        benchmark_adjusted_close = benchmark_holding["adjusted_close"].to_numpy(dtype=np.float64)
        asset_entry = (
            float(asset_holding["open"].iloc[0])
            * asset_adjusted_close[0]
            / max(asset_close[0], 1e-12)
        )
        benchmark_entry = (
            float(benchmark_holding["open"].iloc[0])
            * benchmark_adjusted_close[0]
            / max(benchmark_close[0], 1e-12)
        )
        asset_gross = asset_adjusted_close / max(asset_entry, 1e-12)
        benchmark_gross = benchmark_adjusted_close / max(benchmark_entry, 1e-12)
        if (
            not np.isfinite(asset_gross).all()
            or not np.isfinite(benchmark_gross).all()
            or (asset_gross <= 0.0).any()
            or (benchmark_gross <= 0.0).any()
        ):
            raise ValueError("Execution total-return gross factors must be finite and positive")
        positions = np.asarray(self.horizons, dtype=np.int64) - 1
        asset_returns = asset_gross[positions] - 1.0
        benchmark_returns = benchmark_gross[positions] - 1.0
        alpha_log_returns = np.log(asset_gross[positions]) - np.log(benchmark_gross[positions])
        return (
            alpha_log_returns,
            asset_returns,
            benchmark_returns,
            holding_dates[positions],
        )

    def _record(self, ordinal: int) -> dict[str, Any]:
        (
            symbol,
            index_row,
            benchmark_symbol,
            frame,
            benchmark,
            cutoff_index,
            holding_dates,
            benchmark_holding,
        ) = self._sample_frames(ordinal)
        start_index = cutoff_index - self.window_size + 1
        observed_raw = frame.iloc[start_index : cutoff_index + 1].copy()
        observed_timestamps = pd.DatetimeIndex(observed_raw["timestamp"])
        benchmark_observed_raw = self._aligned_rows(
            benchmark_symbol,
            benchmark,
            observed_timestamps,
        )
        (
            alpha_values,
            asset_values,
            benchmark_values,
            exit_dates,
        ) = self._target_components(
            frame=frame,
            cutoff_index=cutoff_index,
            holding_dates=holding_dates,
            benchmark_holding=benchmark_holding,
        )
        entry_at = holding_dates[0]
        alpha_log_returns: dict[str, float] = {}
        asset_returns: dict[str, float] = {}
        benchmark_returns: dict[str, float] = {}
        end_at: dict[str, str] = {}
        for position, horizon in enumerate(self.horizons):
            key = f"{horizon}d"
            asset_returns[key] = float(asset_values[position])
            benchmark_returns[key] = float(benchmark_values[position])
            alpha_log_returns[key] = float(alpha_values[position])
            end_at[key] = exit_dates[position].isoformat()
        observed = asof_adjusted_window(observed_raw)
        benchmark_observed = asof_adjusted_window(benchmark_observed_raw)
        cutoff_at = pd.Timestamp(frame.loc[cutoff_index, "timestamp"])
        return {
            "schema_version": PROCESSED_SCHEMA_VERSION,
            "sample_id": f"{symbol}-{cutoff_at.strftime('%Y%m%dT%H%M%SZ')}",
            "symbol": symbol,
            "asset_type": str(index_row["asset_type"]),
            "benchmark_symbol": benchmark_symbol,
            "window_start_at": pd.Timestamp(frame.loc[start_index, "timestamp"]).isoformat(),
            "cutoff_at": cutoff_at.isoformat(),
            "context": _context_payload(observed),
            "benchmark_context": _context_payload(benchmark_observed),
            "label": {
                "kind": "benchmark_relative_adjusted_log_return",
                "horizons": list(self.horizons),
                "entry_at": entry_at.isoformat(),
                "entry_price_field": "raw_regular_session_open",
                "entry_day_counts_as_holding_day_one": True,
                "end_at": end_at,
                "alpha_log_returns": alpha_log_returns,
                "asset_total_returns": asset_returns,
                "benchmark_total_returns": benchmark_returns,
            },
            "diagnostics": {
                "capm_abnormal_return": None,
                "capm_status": "reserved_for_diagnostic_ablation",
            },
            "metadata": {
                "market": str(index_row["market"]),
                "provider": str(index_row["provider"]),
                "dataset_profile": str(index_row["dataset_profile"]),
                "benchmark_policy": str(index_row["benchmark_policy"]),
                "input_adjustment_cutoff": "cutoff_at",
                "storage": "lazy_symbol_bar_store",
            },
        }

    def _item(self, ordinal: int) -> dict[str, Any]:
        """Build tensors directly from cached frames without a Python-list round trip."""

        (
            symbol,
            index_row,
            benchmark_symbol,
            frame,
            benchmark,
            cutoff_index,
            holding_dates,
            benchmark_holding,
        ) = self._sample_frames(ordinal)
        start_index = cutoff_index - self.window_size + 1
        observed_raw = frame.iloc[start_index : cutoff_index + 1]
        observed_timestamps = pd.DatetimeIndex(observed_raw["timestamp"])
        benchmark_observed_raw = self._aligned_rows(
            benchmark_symbol,
            benchmark,
            observed_timestamps,
        )
        alpha_values, _asset_returns, _benchmark_returns, _exit_dates = self._target_components(
            frame=frame,
            cutoff_index=cutoff_index,
            holding_dates=holding_dates,
            benchmark_holding=benchmark_holding,
        )
        if not np.isfinite(alpha_values).all():
            raise ValueError("Alpha targets must be finite")
        cutoff_at = pd.Timestamp(frame.loc[cutoff_index, "timestamp"])
        metadata = {
            "market": str(index_row["market"]),
            "provider": str(index_row["provider"]),
            "dataset_profile": str(index_row["dataset_profile"]),
        }
        return {
            "asset_series": _series_from_asof_adjusted_frame(
                observed_raw,
                self.series_mode,
            ),
            "asset_timestamp_features": _timestamp_features(observed_timestamps),
            "benchmark_series": _series_from_asof_adjusted_frame(
                benchmark_observed_raw,
                self.series_mode,
            ),
            "benchmark_timestamp_features": _timestamp_features(observed_timestamps),
            "target_alpha": torch.from_numpy(alpha_values.astype(np.float32, copy=False)),
            "sample_id": f"{symbol}-{cutoff_at.strftime('%Y%m%dT%H%M%SZ')}",
            "symbol": symbol,
            "benchmark_symbol": benchmark_symbol,
            "asset_type": str(index_row["asset_type"]),
            "market": metadata["market"],
            "provider": metadata["provider"],
            "dataset_profile": metadata["dataset_profile"],
            "cutoff_at": cutoff_at.isoformat(),
            "diagnostics": {
                "capm_abnormal_return": None,
                "capm_status": "reserved_for_diagnostic_ablation",
            },
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._item(index)

    def __getitems__(self, indices: list[int]) -> list[dict[str, Any]]:
        """Serve one DataLoader batch in index order that maximizes cache locality."""

        ordered = sorted(enumerate(indices), key=lambda item: item[1])
        resolved = [(position, self._item(index)) for position, index in ordered]
        items: list[dict[str, Any] | None] = [None] * len(indices)
        for position, item in resolved:
            items[position] = item
        if any(item is None for item in items):
            raise RuntimeError("Batched lazy dataset lookup did not resolve every index")
        return [item for item in items if item is not None]

    def record_at(self, index: int) -> dict[str, Any]:
        """Return one ephemeral record for numerical baseline compatibility."""

        return self._record(index)

    def target_at(self, index: int) -> torch.Tensor:
        """Calculate only runtime labels without constructing the model context."""

        (
            _symbol,
            _index_row,
            _benchmark_symbol,
            frame,
            _benchmark,
            cutoff_index,
            holding_dates,
            benchmark_holding,
        ) = self._sample_frames(index)
        alpha_values, _asset, _benchmark_returns, _exit_dates = self._target_components(
            frame=frame,
            cutoff_index=cutoff_index,
            holding_dates=holding_dates,
            benchmark_holding=benchmark_holding,
        )
        return torch.from_numpy(alpha_values.astype(np.float32, copy=False))


class BlockwisePermutationSampler(Sampler[int]):
    """Deterministically shuffle bounded blocks without allocating one index per sample."""

    def __init__(
        self,
        sample_count: int,
        *,
        fraction: float = 1.0,
        max_samples: int | None = None,
        seed: int = 42,
        block_size: int = 16,
    ) -> None:
        if sample_count < 1:
            raise ValueError("sample_count must be positive")
        if not 0.0 < fraction <= 1.0:
            raise ValueError("fraction must be in (0, 1]")
        if block_size < 1:
            raise ValueError("block_size must be positive")
        target = sample_count if fraction >= 1.0 else max(1, int(sample_count * fraction))
        if max_samples is not None:
            target = min(target, max_samples)
        self.sample_count = sample_count
        self.target_count = target
        self.seed = int(seed)
        self.block_size = block_size
        self.epoch = 0

    def __len__(self) -> int:
        return self.target_count

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    @staticmethod
    def _affine_parameters(modulus: int, seed: int) -> tuple[int, int]:
        if modulus == 1:
            return 1, 0
        multiplier = 2 * (abs(seed) % modulus) + 1
        multiplier %= modulus
        if multiplier == 0:
            multiplier = 1
        while gcd(multiplier, modulus) != 1:
            multiplier = (multiplier + 2) % modulus or 1
        offset = (seed * 1_103_515_245 + 12_345) % modulus
        return multiplier, offset

    def __iter__(self) -> Iterator[int]:
        block_count = math.ceil(self.sample_count / self.block_size)
        # Keep membership independent of epoch so multiple epochs revisit one exact subset.
        selection_multiplier, selection_offset = self._affine_parameters(
            block_count,
            self.seed,
        )
        last_block_size = self.sample_count - (block_count - 1) * self.block_size
        last_block_deficit = self.block_size - last_block_size
        if block_count == 1:
            last_block_rank = 0
        else:
            inverse = pow(selection_multiplier, -1, block_count)
            last_block_rank = (inverse * ((block_count - 1) - selection_offset)) % block_count

        selected_block_count = math.ceil(self.target_count / self.block_size)
        selected_capacity = selected_block_count * self.block_size
        if last_block_rank < selected_block_count:
            selected_capacity -= last_block_deficit
        if selected_capacity < self.target_count:
            selected_block_count += 1

        # Permute only the selected block ranks to vary traversal without storing indices.
        epoch_seed = self.seed + self.epoch * 1_000_003
        order_multiplier, order_offset = self._affine_parameters(
            selected_block_count,
            epoch_seed,
        )
        emitted = 0
        reverse = bool(epoch_seed & 1)
        for position in range(selected_block_count):
            selection_rank = (order_multiplier * position + order_offset) % selected_block_count
            block = (selection_multiplier * selection_rank + selection_offset) % block_count
            start = block * self.block_size
            stop = min(start + self.block_size, self.sample_count)
            preceding_capacity = selection_rank * self.block_size
            if last_block_rank < selection_rank:
                preceding_capacity -= last_block_deficit
            take = min(stop - start, self.target_count - preceding_capacity)
            if take < 1:
                raise RuntimeError("Sampler selected an empty deterministic block")
            selected_stop = start + take
            values = (
                range(selected_stop - 1, start - 1, -1) if reverse else range(start, selected_stop)
            )
            for index in values:
                yield index
                emitted += 1
        if emitted != self.target_count:
            raise RuntimeError("Sampler did not emit its exact deterministic target")


class FixedSizeBatchSampler(Sampler[list[int]]):
    """Emit fixed-size batches while retaining the sampler's exact target set."""

    def __init__(
        self,
        sampler: BlockwisePermutationSampler,
        *,
        batch_size: int,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if len(sampler) < batch_size:
            raise ValueError("The sampler must contain at least one complete batch")
        self.sampler = sampler
        self.batch_size = batch_size

    def __len__(self) -> int:
        return math.ceil(len(self.sampler) / self.batch_size)

    @property
    def padded_sample_count(self) -> int:
        return len(self) * self.batch_size - len(self.sampler)

    def set_epoch(self, epoch: int) -> None:
        set_epoch = getattr(self.sampler, "set_epoch", None)
        if callable(set_epoch):
            set_epoch(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        batch: list[int] = []
        fillers: list[int] = []
        for index in self.sampler:
            if len(fillers) < self.batch_size:
                fillers.append(index)
            batch.append(index)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch:
            for position in range(self.batch_size - len(batch)):
                batch.append(fillers[position % len(fillers)])
            yield batch


def _padded_stream(
    items: Sequence[dict[str, Any]],
    *,
    series_key: str,
    timestamp_key: str,
    batch_first: bool,
    padding_value: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    series_values = [item[series_key] for item in items]
    timestamp_values = [item[timestamp_key] for item in items]
    lengths = torch.tensor([value.shape[0] for value in series_values], dtype=torch.long)
    if bool(torch.all(lengths == lengths[0])):
        stack_dimension = 0 if batch_first else 1
        series = torch.stack(series_values, dim=stack_dimension)
        timestamps = torch.stack(timestamp_values, dim=stack_dimension)
        mask_shape = (len(items), int(lengths[0])) if batch_first else (int(lengths[0]), len(items))
        mask = torch.ones(mask_shape, dtype=torch.bool)
        return series, timestamps, mask, lengths
    series = pad_sequence(
        series_values,
        batch_first=batch_first,
        padding_value=padding_value,
    )
    timestamps = pad_sequence(
        timestamp_values,
        batch_first=batch_first,
        padding_value=0.0,
    )
    max_length = int(lengths.max())
    indices = torch.arange(max_length).unsqueeze(0)
    mask = indices < lengths.unsqueeze(1)
    if not batch_first:
        mask = mask.transpose(0, 1)
    return series, timestamps, mask, lengths


class FinancialBatchCollator:
    """Pad both OHLCV streams and preserve provenance for diagnostics."""

    def __init__(self, *, batch_first: bool = True, padding_value: float = 0.0) -> None:
        self.batch_first = batch_first
        self.padding_value = padding_value

    def __call__(self, items: Sequence[dict[str, Any]]) -> dict[str, Any]:
        if not items:
            raise ValueError("Cannot collate an empty batch")
        asset_series, asset_timestamps, asset_mask, asset_lengths = _padded_stream(
            items,
            series_key="asset_series",
            timestamp_key="asset_timestamp_features",
            batch_first=self.batch_first,
            padding_value=self.padding_value,
        )
        benchmark_series, benchmark_timestamps, benchmark_mask, benchmark_lengths = _padded_stream(
            items,
            series_key="benchmark_series",
            timestamp_key="benchmark_timestamp_features",
            batch_first=self.batch_first,
            padding_value=self.padding_value,
        )
        return {
            "asset_series": asset_series,
            "asset_timestamps": asset_timestamps,
            "asset_attention_mask": asset_mask,
            "asset_lengths": asset_lengths,
            "benchmark_series": benchmark_series,
            "benchmark_timestamps": benchmark_timestamps,
            "benchmark_attention_mask": benchmark_mask,
            "benchmark_lengths": benchmark_lengths,
            "target_alpha": torch.stack([item["target_alpha"] for item in items]),
            "sample_ids": [item["sample_id"] for item in items],
            "symbols": [item["symbol"] for item in items],
            "benchmark_symbols": [item["benchmark_symbol"] for item in items],
            "asset_types": [item["asset_type"] for item in items],
            "markets": [item["market"] for item in items],
            "providers": [item["provider"] for item in items],
            "dataset_profiles": [item["dataset_profile"] for item in items],
            "cutoff_at": [item["cutoff_at"] for item in items],
            "diagnostics": [item["diagnostics"] for item in items],
        }
