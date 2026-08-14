"""Offline multi-provider daily OHLCV ingestion into canonical Parquet."""

from __future__ import annotations

import json
import os
import uuid
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from fin_ts_multimodal.config import DatasetProfile
from fin_ts_multimodal.data.adjustments import apply_cumulative_adjustments
from fin_ts_multimodal.data.benchmarks import US_BENCHMARK
from fin_ts_multimodal.data.manifest import (
    DATASET_MANIFEST_SCHEMA_VERSION,
    artifact_metadata,
    atomic_write_json,
)
from fin_ts_multimodal.data.providers import (
    CachedJsonClient,
    EODHDProvider,
    Instrument,
    MassiveProvider,
    NetworkRequestBudget,
    RequestRecord,
    TPExProvider,
    TWSEProvider,
)
from fin_ts_multimodal.data.schema import normalize_ohlcv_frame


@dataclass(frozen=True)
class IngestionOptions:
    profile: DatasetProfile
    start: str
    end: str
    output: Path
    manifest_root: Path
    raw_cache_root: Path
    include_delisted: bool = True
    explicit_us_symbols: tuple[str, ...] = ()
    explicit_us_etfs: tuple[str, ...] = ()
    symbol_limit: int | None = None
    max_api_calls: int = 90_000
    eodhd_requests_per_second: float = 5.0
    taiwan_requests_per_second: float = 0.5
    max_attempts: int = 3


class _ParquetSink:
    def __init__(self, destination: Path) -> None:
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite immutable Parquet: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.destination = destination
        self.staging = destination.with_name(f".{destination.name}.staging-{uuid.uuid4().hex}")
        self.writer: Any = None
        self.schema: Any = None
        self.row_count = 0
        self.symbols: set[str] = set()
        self.providers: set[str] = set()
        self.markets: set[str] = set()
        self.asset_counts: Counter[str] = Counter()

    def write(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as error:
            raise RuntimeError("Writing Parquet requires pyarrow") from error
        normalized = normalize_ohlcv_frame(frame)
        table = pa.Table.from_pandas(normalized, preserve_index=False)
        if self.writer is None:
            self.schema = table.schema
            self.writer = pq.ParquetWriter(
                self.staging,
                self.schema,
                compression="zstd",
            )
        elif table.schema != self.schema:
            table = table.cast(self.schema)
        self.writer.write_table(table)
        self.row_count += len(normalized)
        self.symbols.update(str(value) for value in normalized["symbol"].unique())
        if "provider" in normalized:
            self.providers.update(str(value) for value in normalized["provider"].unique())
        if "market" in normalized:
            self.markets.update(str(value) for value in normalized["market"].unique())
        self.asset_counts.update(str(value) for value in normalized["asset_type"])

    def close(self) -> None:
        if self.writer is None or self.row_count == 0:
            raise ValueError("No OHLCV rows were available to write")
        self.writer.close()
        self.writer = None
        self.staging.replace(self.destination)

    def abort(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None
        self.staging.unlink(missing_ok=True)


class _RequestLog:
    def __init__(self, destination: Path) -> None:
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite request log: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.destination = destination
        self.staging = destination.with_name(f".{destination.name}.staging-{uuid.uuid4().hex}")
        self.stream = self.staging.open("x", encoding="utf-8")
        self.count = 0
        self.cache_hits = 0

    def append(self, records: Iterable[RequestRecord]) -> None:
        for record in records:
            self.stream.write(
                json.dumps(
                    asdict(record),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            self.stream.write("\n")
            self.count += 1
            self.cache_hits += int(record.cache_hit)

    def close(self) -> None:
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.stream.close()
        if self.count == 0:
            self.staging.unlink(missing_ok=True)
            raise ValueError("No provider requests were recorded")
        self.staging.replace(self.destination)

    def abort(self) -> None:
        if not self.stream.closed:
            self.stream.close()
        self.staging.unlink(missing_ok=True)


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def _inclusive_end(exclusive_end: str) -> str:
    end = _parse_date(exclusive_end)
    return (end - timedelta(days=1)).isoformat()


def _weekdays(start: str, exclusive_end: str) -> list[str]:
    current = _parse_date(start)
    end = _parse_date(exclusive_end)
    if current >= end:
        raise ValueError("Ingestion start must be earlier than exclusive end")
    values: list[str] = []
    while current < end:
        if current.weekday() < 5:
            values.append(current.isoformat())
        current += timedelta(days=1)
    return values


def _months(start: str, exclusive_end: str) -> list[str]:
    first = pd.Timestamp(start).to_period("M")
    final = (pd.Timestamp(exclusive_end) - pd.Timedelta(days=1)).to_period("M")
    return [str(value) for value in pd.period_range(first, final, freq="M")]


def _selected_datasets(profile: DatasetProfile) -> list[str]:
    values = {
        "tw_only": ["tpex_official", "twse_official"],
        "us_only_eodhd": ["eodhd_us"],
        "us_tw_eodhd": ["eodhd_us", "tpex_official", "twse_official"],
        "us_tw_massive": ["massive_us", "tpex_official", "twse_official"],
    }
    return sorted(values[profile])


def _explicit_instruments(
    symbols: tuple[str, ...],
    etf_symbols: tuple[str, ...],
) -> list[Instrument]:
    etfs = {symbol.upper().removesuffix(".US") for symbol in etf_symbols}
    instruments: list[Instrument] = []
    requested_symbols = {symbol.upper().removesuffix(".US") for symbol in (*symbols, *etf_symbols)}
    for code in sorted(requested_symbols):
        if not code:
            raise ValueError("Explicit US symbols must be non-empty")
        instruments.append(
            Instrument(
                provider_symbol=f"{code}.US",
                canonical_symbol=f"{code}.US",
                asset_type="etf" if code in etfs else "stock",
                market="US",
                currency="USD",
                is_active=True,
            )
        )
    return instruments


def _limit_instruments(
    instruments: list[Instrument],
    limit: int | None,
) -> list[Instrument]:
    ordered = sorted(
        instruments,
        key=lambda item: (
            item.asset_type,
            not item.is_active,
            item.canonical_symbol,
        ),
    )
    if limit is None:
        return ordered
    if limit < 1:
        raise ValueError("symbol_limit must be positive")
    return ordered[:limit]


def _ensure_us_benchmark(instruments: list[Instrument]) -> list[Instrument]:
    """Guarantee that every US training download contains the VTI benchmark."""

    if any(item.canonical_symbol == US_BENCHMARK for item in instruments):
        return instruments
    return [
        *instruments,
        Instrument(
            provider_symbol=US_BENCHMARK,
            canonical_symbol=US_BENCHMARK,
            asset_type="etf",
            market="US",
            currency="USD",
            is_active=True,
        ),
    ]


def ingest_daily_ohlcv(
    options: IngestionOptions,
    *,
    eodhd_api_token: str | None = None,
) -> dict[str, Any]:
    """Fetch providers only here; training and evaluation never call this function."""

    if options.profile == "us_tw_massive":
        MassiveProvider().discover(include_delisted=options.include_delisted)
    if options.max_api_calls < 1:
        raise ValueError("max_api_calls must be positive")
    start = _parse_date(options.start)
    end = _parse_date(options.end)
    if start >= end:
        raise ValueError("start must be earlier than the exclusive end date")
    options.manifest_root.mkdir(parents=True, exist_ok=True)
    request_log_path = options.manifest_root / "manifests" / "api-request-log.jsonl"
    download_manifest_path = options.manifest_root / "download-manifest.json"
    if download_manifest_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite immutable download manifest: {download_manifest_path}"
        )

    sink = _ParquetSink(options.output)
    request_log = _RequestLog(request_log_path)
    empty_series = 0
    dropped_source_rows = 0
    corporate_action_rows = 0
    benchmark_rows = 0
    estimated_calls = 0
    request_budget = NetworkRequestBudget(options.max_api_calls)
    try:
        if options.profile in {"us_only_eodhd", "us_tw_eodhd"}:
            if not eodhd_api_token:
                raise ValueError("EODHD_API_TOKEN is required for the selected dataset profile")
            eod_client = CachedJsonClient(
                provider="eodhd",
                raw_cache_root=options.raw_cache_root,
                max_requests_per_second=options.eodhd_requests_per_second,
                max_attempts=options.max_attempts,
                request_budget=request_budget,
            )
            provider = EODHDProvider(eod_client, api_token=eodhd_api_token)
            if options.explicit_us_symbols or options.explicit_us_etfs:
                instruments = _explicit_instruments(
                    options.explicit_us_symbols,
                    options.explicit_us_etfs,
                )
                discovery_requests: tuple[RequestRecord, ...] = ()
            else:
                discovered, discovery_requests = provider.discover(
                    include_delisted=options.include_delisted
                )
                instruments = discovered
                request_log.append(discovery_requests)
            instruments = _ensure_us_benchmark(
                _limit_instruments(instruments, options.symbol_limit)
            )
            provider_end = _inclusive_end(options.end)
            uses_historical_split_api = start < provider.calendar_split_history_start
            split_call_count = len(instruments) if uses_historical_split_api else 1
            estimated_calls += (
                len(discovery_requests) + len(instruments) + split_call_count
            )
            if estimated_calls > options.max_api_calls:
                raise ValueError(
                    f"Estimated API calls ({estimated_calls}) exceed max_api_calls="
                    f"{options.max_api_calls}"
                )
            shared_split_fetch = None
            if not uses_historical_split_api:
                shared_split_fetch = provider.fetch_split_events(
                    start=options.start,
                    end=provider_end,
                )
                request_log.append(shared_split_fetch.requests)
                corporate_action_rows += len(shared_split_fetch.frame)
                dropped_source_rows += int(
                    shared_split_fetch.metadata.get("dropped_rows", 0)
                )
            for instrument in instruments:
                if uses_historical_split_api:
                    instrument_split_fetch = provider.fetch_historical_split_events(
                        instrument,
                        start=options.start,
                        end=provider_end,
                    )
                    request_log.append(instrument_split_fetch.requests)
                    corporate_action_rows += len(instrument_split_fetch.frame)
                    dropped_source_rows += int(
                        instrument_split_fetch.metadata.get("dropped_rows", 0)
                    )
                    split_events = instrument_split_fetch.frame
                    split_adjustment_source = "historical_splits"
                else:
                    if shared_split_fetch is None:
                        raise RuntimeError("Shared EODHD split response was not initialized")
                    split_events = shared_split_fetch.frame
                    split_adjustment_source = "calendar_splits"
                fetched = provider.fetch_instrument(
                    instrument,
                    start=options.start,
                    end=provider_end,
                    dataset_profile=options.profile,
                    split_events=split_events,
                    split_adjustment_source=split_adjustment_source,
                )
                request_log.append(fetched.requests)
                if fetched.frame.empty:
                    empty_series += 1
                    continue
                sink.write(fetched.frame)

        if options.profile in {"tw_only", "us_tw_eodhd", "us_tw_massive"}:
            dates = _weekdays(options.start, options.end)
            months = _months(options.start, options.end)
            estimated_calls += len(dates) * 2 + len(months) * 4 + 2
            if estimated_calls > options.max_api_calls:
                raise ValueError(
                    f"Estimated API calls ({estimated_calls}) exceed max_api_calls="
                    f"{options.max_api_calls}"
                )
            for provider_class in (TWSEProvider, TPExProvider):
                provider_name = provider_class.name
                client = CachedJsonClient(
                    provider=provider_name,
                    raw_cache_root=options.raw_cache_root,
                    max_requests_per_second=options.taiwan_requests_per_second,
                    max_attempts=options.max_attempts,
                    request_budget=request_budget,
                )
                provider = provider_class(client)
                action_fetch = provider.fetch_actions(
                    start=options.start,
                    end=_inclusive_end(options.end),
                )
                request_log.append(action_fetch.requests)
                estimated_calls += max(0, len(action_fetch.requests) - 1)
                if estimated_calls > options.max_api_calls:
                    raise ValueError(
                        f"Actual corporate-action request count raised estimated calls "
                        f"({estimated_calls}) above max_api_calls={options.max_api_calls}"
                    )
                corporate_action_rows += len(action_fetch.frame)
                dropped_source_rows += int(action_fetch.metadata.get("dropped_rows", 0))
                for trading_date in dates:
                    fetched = provider.fetch_date(
                        date=trading_date,
                        dataset_profile=options.profile,
                    )
                    request_log.append(fetched.requests)
                    dropped_source_rows += int(fetched.metadata.get("dropped_rows", 0))
                    if not fetched.frame.empty:
                        adjusted = apply_cumulative_adjustments(
                            fetched.frame,
                            action_fetch.frame,
                        )
                        adjusted["adjustment_source"] = (
                            "twse_twt49u" if provider_name == "twse_official" else "tpex_exdailyq"
                        )
                        sink.write(adjusted)
                for month in months:
                    benchmark_fetch = provider.fetch_benchmark_month(
                        month=month,
                        dataset_profile=options.profile,
                    )
                    request_log.append(benchmark_fetch.requests)
                    benchmark_rows += len(benchmark_fetch.frame)
                    sink.write(benchmark_fetch.frame)

        sink.close()
        request_log.close()
    except BaseException:
        sink.abort()
        request_log.abort()
        raise

    raw_artifact = artifact_metadata(
        options.output,
        root=options.manifest_root,
        row_count=sink.row_count,
    )
    request_artifact = artifact_metadata(
        request_log_path,
        root=options.manifest_root,
        row_count=request_log.count,
    )
    payload = {
        "schema_version": DATASET_MANIFEST_SCHEMA_VERSION,
        "kind": "ohlcv-dataset",
        "state": "downloaded",
        "created_at": datetime.now(UTC).isoformat(),
        "dataset_profile": options.profile,
        "selected_datasets": _selected_datasets(options.profile),
        "providers": sorted(sink.providers),
        "markets": sorted(sink.markets),
        "date_range": {
            "start_inclusive": options.start,
            "end_exclusive": options.end,
        },
        "symbols": {
            "count": len(sink.symbols),
            "values": sorted(sink.symbols),
            "asset_type_counts": dict(sorted(sink.asset_counts.items())),
        },
        "api_policy": {
            "estimated_calls": estimated_calls,
            "max_api_calls": options.max_api_calls,
            "recorded_requests": request_log.count,
            "network_requests": request_budget.network_requests,
            "network_requests_by_provider": request_budget.provider_counts,
            "cache_hits": request_log.cache_hits,
            "eodhd_requests_per_second": options.eodhd_requests_per_second,
            "taiwan_requests_per_second": options.taiwan_requests_per_second,
            "include_delisted": options.include_delisted,
            "symbol_limit": options.symbol_limit,
            "eodhd_split_strategy": (
                "per_symbol_historical_splits"
                if options.profile in {"us_only_eodhd", "us_tw_eodhd"}
                and start < EODHDProvider.calendar_split_history_start
                else "exchange_wide_calendar_splits"
                if options.profile in {"us_only_eodhd", "us_tw_eodhd"}
                else None
            ),
        },
        "quality": {
            "schema_validated_rows": sink.row_count,
            "empty_symbol_histories": empty_series,
            "dropped_source_rows": dropped_source_rows,
            "corporate_action_rows": corporate_action_rows,
            "benchmark_rows": benchmark_rows,
            "adjustment_contract": {
                "raw_ohlcv_immutable": True,
                "price": "total_return_adjusted_as_of_cutoff",
                "volume": "share_change_adjusted_as_of_cutoff",
                "benchmark": "official_total_return_index_or_vti",
            },
        },
        "artifacts": {
            "raw": raw_artifact,
            "request_log": request_artifact,
        },
    }
    atomic_write_json(download_manifest_path, payload)
    return {
        **payload,
        "download_manifest_path": str(download_manifest_path),
        "raw_parquet_path": str(options.output),
        "request_log_path": str(request_log_path),
    }
