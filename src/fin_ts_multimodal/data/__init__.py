"""Public data-pipeline API."""

from fin_ts_multimodal.data.dataset import FinancialBatchCollator, FinancialWindowDataset
from fin_ts_multimodal.data.io import read_processed_records, write_processed_records
from fin_ts_multimodal.data.schema import (
    MarketDataValidationError,
    normalize_ohlcv_frame,
    read_market_data,
    write_market_data,
)
from fin_ts_multimodal.data.splits import chronological_split
from fin_ts_multimodal.data.synthetic import generate_synthetic_ohlcv
from fin_ts_multimodal.data.windows import build_causal_windows

__all__ = [
    "FinancialBatchCollator",
    "FinancialWindowDataset",
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
