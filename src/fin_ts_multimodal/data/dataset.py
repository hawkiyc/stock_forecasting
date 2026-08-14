"""PyTorch dataset for conditional asset-versus-benchmark alpha windows."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from fin_ts_multimodal.data.io import read_processed_records
from fin_ts_multimodal.data.windows import (
    CONTEXT_FIELDS,
    DEFAULT_ALPHA_HORIZONS,
    PROCESSED_SCHEMA_VERSION,
)

TIMESTAMP_FEATURE_FIELDS = ("minute", "hour", "weekday", "day", "month")


class FinancialWindowDataset(Dataset[dict[str, Any]]):
    """Expose paired historical OHLCV streams and continuous alpha targets."""

    def __init__(
        self,
        records: Sequence[dict[str, Any]] | str | Path,
        *,
        split: Literal["train", "validation", "test"] | None = None,
        series_mode: Literal["raw", "relative"] = "raw",
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
        self.split = split
        self.series_mode = series_mode
        self.horizons = DEFAULT_ALPHA_HORIZONS

    def __len__(self) -> int:
        return len(self.records)

    def _series_from_context(self, context: dict[str, Any]) -> torch.Tensor:
        values = np.column_stack(
            [np.asarray(context[field], dtype=np.float32) for field in CONTEXT_FIELDS]
        )
        if self.series_mode == "relative":
            reference_price = max(float(values[-1, 3]), 1e-12)
            values[:, :4] = np.log(np.maximum(values[:, :4], 1e-12) / reference_price)
            log_volume = np.log1p(np.maximum(values[:, 4], 0.0))
            scale = float(log_volume.std())
            values[:, 4] = (log_volume - float(log_volume.mean())) / (
                scale if scale > 1e-6 else 1.0
            )
        return torch.from_numpy(values)

    @staticmethod
    def _timestamp_features_from_context(context: dict[str, Any]) -> torch.Tensor:
        timestamps = pd.DatetimeIndex(pd.to_datetime(context["timestamp"], utc=True))
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

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        context = record["context"]
        benchmark_context = record["benchmark_context"]
        label = record["label"]
        if tuple(label.get("horizons", ())) != self.horizons:
            raise ValueError("Record alpha horizons do not match the model contract")
        alpha_values = label.get("alpha_log_returns")
        if not isinstance(alpha_values, dict):
            raise ValueError("Record has no alpha_log_returns mapping")
        target = np.asarray(
            [alpha_values[f"{horizon}d"] for horizon in self.horizons],
            dtype=np.float32,
        )
        if not np.isfinite(target).all():
            raise ValueError("Alpha targets must be finite")
        metadata = record.get("metadata", {})
        return {
            "asset_series": self._series_from_context(context),
            "asset_timestamp_features": self._timestamp_features_from_context(context),
            "benchmark_series": self._series_from_context(benchmark_context),
            "benchmark_timestamp_features": self._timestamp_features_from_context(
                benchmark_context
            ),
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


def _padded_stream(
    items: Sequence[dict[str, Any]],
    *,
    series_key: str,
    timestamp_key: str,
    batch_first: bool,
    padding_value: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([item[series_key].shape[0] for item in items], dtype=torch.long)
    series = pad_sequence(
        [item[series_key] for item in items],
        batch_first=batch_first,
        padding_value=padding_value,
    )
    timestamps = pad_sequence(
        [item[timestamp_key] for item in items],
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
