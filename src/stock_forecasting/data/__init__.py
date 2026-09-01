"""Cycle-safe lazy exports for the public data-pipeline API."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from stock_forecasting.data.dataset import (
        BlockwisePermutationSampler,
        FinancialBatchCollator,
        FinancialWindowDataset,
        FixedSizeBatchSampler,
        LazyFinancialWindowDataset,
    )
    from stock_forecasting.data.io import read_processed_records, write_processed_records
    from stock_forecasting.data.schema import (
        MarketDataValidationError,
        normalize_ohlcv_frame,
        read_market_data,
        write_market_data,
    )
    from stock_forecasting.data.splits import chronological_split
    from stock_forecasting.data.synthetic import generate_synthetic_ohlcv
    from stock_forecasting.data.windows import build_causal_windows

_EXPORTS = {
    "BlockwisePermutationSampler": (
        "stock_forecasting.data.dataset",
        "BlockwisePermutationSampler",
    ),
    "FinancialBatchCollator": (
        "stock_forecasting.data.dataset",
        "FinancialBatchCollator",
    ),
    "FinancialWindowDataset": (
        "stock_forecasting.data.dataset",
        "FinancialWindowDataset",
    ),
    "FixedSizeBatchSampler": (
        "stock_forecasting.data.dataset",
        "FixedSizeBatchSampler",
    ),
    "LazyFinancialWindowDataset": (
        "stock_forecasting.data.dataset",
        "LazyFinancialWindowDataset",
    ),
    "MarketDataValidationError": (
        "stock_forecasting.data.schema",
        "MarketDataValidationError",
    ),
    "build_causal_windows": (
        "stock_forecasting.data.windows",
        "build_causal_windows",
    ),
    "chronological_split": (
        "stock_forecasting.data.splits",
        "chronological_split",
    ),
    "generate_synthetic_ohlcv": (
        "stock_forecasting.data.synthetic",
        "generate_synthetic_ohlcv",
    ),
    "normalize_ohlcv_frame": (
        "stock_forecasting.data.schema",
        "normalize_ohlcv_frame",
    ),
    "read_market_data": ("stock_forecasting.data.schema", "read_market_data"),
    "read_processed_records": (
        "stock_forecasting.data.io",
        "read_processed_records",
    ),
    "write_market_data": ("stock_forecasting.data.schema", "write_market_data"),
    "write_processed_records": (
        "stock_forecasting.data.io",
        "write_processed_records",
    ),
}

__all__ = [
    "BlockwisePermutationSampler",
    "FinancialBatchCollator",
    "FinancialWindowDataset",
    "FixedSizeBatchSampler",
    "LazyFinancialWindowDataset",
    "MarketDataValidationError",
    "build_causal_windows",
    "chronological_split",
    "generate_synthetic_ohlcv",
    "normalize_ohlcv_frame",
    "read_market_data",
    "read_processed_records",
    "write_market_data",
    "write_processed_records",
]


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
