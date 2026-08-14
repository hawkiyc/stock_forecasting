"""Shared provider contracts for offline daily OHLCV ingestion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import pandas as pd


@dataclass(frozen=True)
class Instrument:
    provider_symbol: str
    canonical_symbol: str
    asset_type: str
    market: str
    currency: str
    is_active: bool


@dataclass(frozen=True)
class RequestRecord:
    provider: str
    request_sha256: str
    cache_relative_path: str
    response_sha256: str
    response_size_bytes: int
    cache_hit: bool
    requested_at: str


@dataclass(frozen=True)
class ProviderFetch:
    frame: pd.DataFrame
    requests: tuple[RequestRecord, ...]
    metadata: dict[str, Any]


class DailyOHLCVProvider(Protocol):
    """Protocol implemented by every optional data channel."""

    name: str

    def discover(
        self,
        *,
        include_delisted: bool,
    ) -> tuple[list[Instrument], tuple[RequestRecord, ...]]: ...

    def fetch_instrument(
        self,
        instrument: Instrument,
        *,
        start: str,
        end: str,
        dataset_profile: str,
    ) -> ProviderFetch: ...
