"""Public data-pipeline API."""

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
