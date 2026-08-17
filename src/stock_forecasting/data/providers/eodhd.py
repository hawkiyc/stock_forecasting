"""EODHD US stock and ETF daily-history provider."""

from __future__ import annotations

from typing import Any

import pandas as pd

from stock_forecasting.data.adjustments import apply_cumulative_adjustments
from stock_forecasting.data.schema import normalize_ohlcv_frame

from .base import Instrument, ProviderFetch, RequestRecord
from .http import CachedJsonClient

EODHD_DEFAULT_DAILY_API_CALL_LIMIT = 100_000
EODHD_DEFAULT_REQUESTS_PER_MINUTE = 1_000
EODHD_DEFAULT_REQUESTS_PER_SECOND = float(EODHD_DEFAULT_REQUESTS_PER_MINUTE // 60)
EODHD_DELISTED_AUXILIARY_DATA_START = "2018-01-01"


class EODHDProvider:
    """Fetch one full daily history per symbol using one API call."""

    name = "eodhd"
    base_url = "https://eodhd.com/api"

    def __init__(self, client: CachedJsonClient, *, api_token: str) -> None:
        if not api_token.strip():
            raise ValueError("EODHD API token is required")
        self.client = client
        self._api_token = api_token

    @staticmethod
    def _asset_type(value: Any) -> str | None:
        normalized = str(value).strip().lower().replace(" ", "_")
        if normalized in {"common_stock", "stock"}:
            return "stock"
        if normalized == "etf":
            return "etf"
        return None

    def _discover_one(
        self,
        *,
        asset_filter: str,
        delisted: bool,
    ) -> tuple[list[Instrument], RequestRecord]:
        payload, request = self.client.get_json(
            f"{self.base_url}/exchange-symbol-list/US",
            params={
                "api_token": self._api_token,
                "fmt": "json",
                "type": asset_filter,
                "delisted": 1 if delisted else 0,
            },
        )
        if not isinstance(payload, list):
            raise ValueError("EODHD exchange-symbol-list did not return a JSON array")
        instruments: list[Instrument] = []
        for row in payload:
            if not isinstance(row, dict):
                continue
            code = str(row.get("Code", "")).strip().upper()
            asset_type = self._asset_type(row.get("Type", asset_filter))
            if not code or asset_type is None:
                continue
            instruments.append(
                Instrument(
                    provider_symbol=f"{code}.US",
                    canonical_symbol=f"{code}.US",
                    asset_type=asset_type,
                    market="US",
                    currency=str(row.get("Currency") or "USD").strip().upper(),
                    is_active=not delisted,
                )
            )
        return instruments, request

    def discover(
        self,
        *,
        include_delisted: bool,
    ) -> tuple[list[Instrument], tuple[RequestRecord, ...]]:
        instruments: list[Instrument] = []
        requests: list[RequestRecord] = []
        statuses = (False, True) if include_delisted else (False,)
        for delisted in statuses:
            for asset_filter in ("common_stock", "etf"):
                rows, request = self._discover_one(
                    asset_filter=asset_filter,
                    delisted=delisted,
                )
                instruments.extend(rows)
                requests.append(request)
        by_symbol: dict[str, Instrument] = {}
        for instrument in sorted(
            instruments,
            key=lambda item: (item.canonical_symbol, not item.is_active, item.asset_type),
        ):
            by_symbol.setdefault(instrument.canonical_symbol, instrument)
        return list(by_symbol.values()), tuple(requests)

    def fetch_instrument(
        self,
        instrument: Instrument,
        *,
        start: str,
        end: str,
        dataset_profile: str,
        split_events: pd.DataFrame | None = None,
        split_adjustment_source: str = "historical_splits",
    ) -> ProviderFetch:
        payload, request = self.client.get_json(
            f"{self.base_url}/eod/{instrument.provider_symbol}",
            params={
                "api_token": self._api_token,
                "fmt": "json",
                "period": "d",
                "order": "a",
                "from": start,
                "to": end,
            },
        )
        if not isinstance(payload, list):
            raise ValueError("EODHD EOD endpoint did not return a JSON array")
        if not payload:
            return ProviderFetch(
                frame=pd.DataFrame(),
                requests=(request,),
                metadata={"empty": True},
            )
        frame = pd.DataFrame(payload)
        frame["symbol"] = instrument.canonical_symbol
        frame["asset_type"] = instrument.asset_type
        frame["provider"] = self.name
        frame["market"] = instrument.market
        frame["currency"] = instrument.currency
        frame["source_symbol"] = instrument.provider_symbol
        frame["is_active"] = instrument.is_active
        frame["dataset_profile"] = dataset_profile
        frame["adjustment_source"] = (
            f"eodhd_adjusted_close+{split_adjustment_source}+reconstructed_raw_volume"
        )
        frame = normalize_ohlcv_frame(frame)
        frame = apply_cumulative_adjustments(
            frame,
            split_events,
            preserve_adjusted_close=True,
            source_volume_is_split_adjusted=True,
        )
        return ProviderFetch(
            frame=normalize_ohlcv_frame(frame),
            requests=(request,),
            metadata={"empty": False},
        )

    def fetch_historical_split_events(
        self,
        instrument: Instrument,
        *,
        start: str,
        end: str,
    ) -> ProviderFetch:
        """Fetch all split factors needed to undo EODHD's global volume adjustment."""

        requested_range = {"start": start, "end": end}

        payload, request = self.client.get_json(
            f"{self.base_url}/splits/{instrument.provider_symbol}",
            params={
                "api_token": self._api_token,
                "fmt": "json",
            },
        )
        if not isinstance(payload, list):
            raise ValueError("EODHD Historical Splits API did not return a JSON array")
        rows: list[dict[str, Any]] = []
        dropped = 0
        for raw in payload:
            if not isinstance(raw, dict):
                dropped += 1
                continue
            ratio = str(raw.get("split", "")).strip().replace(":", "/")
            parts = ratio.split("/")
            if len(parts) != 2:
                dropped += 1
                continue
            try:
                new_shares, old_shares = (float(value) for value in parts)
            except (TypeError, ValueError):
                dropped += 1
                continue
            if old_shares <= 0.0 or new_shares <= 0.0:
                dropped += 1
                continue
            rows.append(
                {
                    "timestamp": raw.get("date"),
                    "symbol": instrument.canonical_symbol,
                    "price_factor": old_shares / new_shares,
                    "share_multiplier": new_shares / old_shares,
                    "source": "eodhd_historical_splits",
                }
            )
        return ProviderFetch(
            frame=pd.DataFrame(rows),
            requests=(request,),
            metadata={
                "split_events": len(rows),
                "dropped_rows": dropped,
                "coverage": "provider_full_historical_splits_response",
                "requested_eod_range": requested_range,
            },
        )
