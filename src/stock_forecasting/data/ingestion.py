"""Offline multi-provider daily OHLCV ingestion into canonical Parquet."""

from __future__ import annotations

import json
import math
import os
import re
import uuid
from collections import Counter, deque
from collections.abc import Callable, Generator, Iterable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import closing, suppress
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from stock_forecasting.data.benchmarks import (
    US_BENCHMARK,
    is_allowlisted_us_equity_etf,
)
from stock_forecasting.data.content_identity import (
    CONTENT_IDENTITY_SCHEMA_VERSION,
    provider_materialization_digest,
    raw_dataset_materialization_digest,
)
from stock_forecasting.data.download_progress import DownloadProgress
from stock_forecasting.data.manifest import (
    DOWNLOAD_MANIFEST_SCHEMA_VERSION,
    artifact_metadata,
    atomic_write_json,
)
from stock_forecasting.data.provider_checkpoint import (
    ProviderCheckpoint,
    load_provider_checkpoint,
    publish_provider_checkpoint,
    quarantine_provider_checkpoint,
)
from stock_forecasting.data.providers import (
    EODHD_DEFAULT_DAILY_API_CALL_LIMIT,
    EODHD_DEFAULT_REQUESTS_PER_MINUTE,
    EODHD_DEFAULT_REQUESTS_PER_SECOND,
    EODHD_DELISTED_AUXILIARY_DATA_START,
    AcquisitionDeadlineExceeded,
    CachedJsonClient,
    EODHDProvider,
    Instrument,
    MassiveProvider,
    NetworkRequestBudget,
    NetworkRequestBudgetExceeded,
    ProviderAcquisitionError,
    ProviderRequestError,
    RequestRecord,
    TPExProvider,
    TpexRelayTransport,
    TWSEProvider,
)
from stock_forecasting.data.schema import (
    TRAINING_SECURITY_SCOPE,
    TRAINING_TARGET_ASSET_TYPES,
    normalize_ohlcv_frame,
)
from stock_forecasting.dataset_profiles import (
    DatasetProfile,
    runtime_providers,
    selected_datasets,
)

RAW_DATASET_ASSEMBLY_POLICY = "complete_provider_checkpoint_concat_v1"


def _canonical_raw_batch(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the logical rows written by the infrastructure-only Parquet sink."""

    return normalize_ohlcv_frame(frame)


def _ordered_thread_map[InputT, ResultT](
    function: Callable[[InputT], ResultT],
    items: Iterable[InputT],
    *,
    max_workers: int,
    thread_name_prefix: str,
) -> Generator[ResultT, None, None]:
    """Yield deterministic results with a bounded two-task prefetch per worker."""

    if max_workers < 1:
        raise ValueError("max_workers must be positive")
    iterator = iter(items)
    executor = ThreadPoolExecutor(
        max_workers=max_workers,
        thread_name_prefix=thread_name_prefix,
    )
    pending: deque[Future[ResultT]] = deque()

    def submit_next() -> bool:
        try:
            item = next(iterator)
        except StopIteration:
            return False
        pending.append(executor.submit(function, item))
        return True

    try:
        for _ in range(max_workers * 2):
            if not submit_next():
                break
        while pending:
            future = pending.popleft()
            result = future.result()
            submit_next()
            yield result
    finally:
        for future in pending:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)


@dataclass(frozen=True)
class IngestionOptions:
    profile: DatasetProfile
    start: str
    end: str
    output: Path
    manifest_root: Path
    raw_cache_root: Path
    cache_revision: str = "v1"
    include_delisted: bool = True
    explicit_us_symbols: tuple[str, ...] = ()
    explicit_us_etfs: tuple[str, ...] = ()
    symbol_limit: int | None = None
    max_api_calls: int = EODHD_DEFAULT_DAILY_API_CALL_LIMIT
    eodhd_requests_per_second: float = EODHD_DEFAULT_REQUESTS_PER_SECOND
    taiwan_requests_per_second: float = 0.5
    max_backoff_seconds: float = 60.0
    progress_path: Path | None = None
    dataset_request_sha256: str | None = None
    selection_id: str | None = None
    selection_sha256: str | None = None
    launch_id: str | None = None
    workers: int = 1
    acquisition_deadline_epoch_seconds: float | None = None
    preparation_reserve_seconds: int | None = None
    tpex_proxy_url: str | None = None
    provider_checkpoint_root: Path | None = None


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
        normalized = _canonical_raw_batch(frame)
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

    def append_parquet(
        self,
        source: Path,
        *,
        check_time: Callable[[], None] | None = None,
    ) -> None:
        """Merge bounded row groups without loading a complete provider part."""

        try:
            import pyarrow.parquet as pq
        except ImportError as error:
            raise RuntimeError("Reading Parquet requires pyarrow") from error
        parquet = pq.ParquetFile(source)
        for row_group in range(parquet.num_row_groups):
            if check_time is not None:
                check_time()
            self.write(parquet.read_row_group(row_group).to_pandas())

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

    def append_file(self, source: Path) -> None:
        """Merge a provider-local log while verifying its generated JSONL rows."""

        with source.open(encoding="utf-8") as input_stream:
            for line in input_stream:
                payload = json.loads(line)
                if not isinstance(payload, dict) or not isinstance(payload.get("cache_hit"), bool):
                    raise ValueError(f"Malformed provider request log: {source}")
                self.stream.write(line)
                self.count += 1
                self.cache_hits += int(payload["cache_hit"])

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


@dataclass
class _ProviderStats:
    estimated_calls: int = 0
    empty_series: int = 0
    dropped_source_rows: int = 0
    corporate_action_rows: int = 0
    missing_share_multiplier_details: int = 0
    skipped_unsupported_action_rows: int = 0
    skipped_unsupported_action_types: Counter[str] = field(default_factory=Counter)
    benchmark_rows: int = 0
    taiwan_trading_dates: int | None = None
    delisted_eod_only_symbols: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class _ProviderArtifacts:
    provider: str
    parquet_path: Path
    request_log_path: Path
    row_count: int
    request_count: int
    cache_hits: int
    checkpoint_identity_sha256: str | None = None
    checkpoint_reused: bool = False

    def cleanup(self) -> None:
        """Remove only launch-local provider fragments after publication."""

        if self.checkpoint_identity_sha256 is not None:
            return
        self.parquet_path.unlink(missing_ok=True)
        self.request_log_path.unlink(missing_ok=True)


@dataclass(frozen=True)
class _ProviderLoopOutcome:
    provider: str
    result: Any | None = None
    error: Exception | None = None


def _ordered_completed_provider_artifacts(
    outcomes: Mapping[str, _ProviderLoopOutcome],
) -> list[_ProviderArtifacts]:
    """Return complete provider parts in the canonical raw concatenation order."""

    artifacts: list[_ProviderArtifacts] = []
    for provider_name in sorted(outcomes):
        provider_artifacts = outcomes[provider_name].result
        if not isinstance(provider_artifacts, _ProviderArtifacts):
            raise TypeError(
                f"Provider loop returned invalid artifacts: {provider_name}"
            )
        artifacts.append(provider_artifacts)
    return artifacts


def _nonnegative_stat(payload: Mapping[str, Any], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"Provider checkpoint has an invalid {key} statistic")
    return value


def _provider_stats_payload(stats: _ProviderStats) -> dict[str, Any]:
    return {
        "benchmark_rows": stats.benchmark_rows,
        "corporate_action_rows": stats.corporate_action_rows,
        "delisted_eod_only_symbols": sorted(stats.delisted_eod_only_symbols),
        "dropped_source_rows": stats.dropped_source_rows,
        "empty_series": stats.empty_series,
        "estimated_calls": stats.estimated_calls,
        "missing_share_multiplier_details": stats.missing_share_multiplier_details,
        "skipped_unsupported_action_rows": stats.skipped_unsupported_action_rows,
        "skipped_unsupported_action_types": dict(
            sorted(stats.skipped_unsupported_action_types.items())
        ),
        "taiwan_trading_dates": stats.taiwan_trading_dates,
    }


def _provider_stats_from_payload(payload: Any) -> _ProviderStats:
    required_keys = {
        "benchmark_rows",
        "corporate_action_rows",
        "delisted_eod_only_symbols",
        "dropped_source_rows",
        "empty_series",
        "estimated_calls",
        "missing_share_multiplier_details",
        "skipped_unsupported_action_rows",
        "skipped_unsupported_action_types",
        "taiwan_trading_dates",
    }
    if not isinstance(payload, dict) or set(payload) != required_keys:
        raise ValueError("Provider checkpoint statistics are invalid")
    delisted_symbols = payload["delisted_eod_only_symbols"]
    if (
        not isinstance(delisted_symbols, list)
        or any(not isinstance(value, str) or not value for value in delisted_symbols)
        or delisted_symbols != sorted(set(delisted_symbols))
    ):
        raise ValueError("Provider checkpoint delisted-symbol statistics are invalid")
    skipped_types = payload["skipped_unsupported_action_types"]
    if not isinstance(skipped_types, dict) or any(
        not isinstance(key, str)
        or not key
        or not isinstance(value, int)
        or isinstance(value, bool)
        or value < 0
        for key, value in skipped_types.items()
    ):
        raise ValueError("Provider checkpoint skipped-action statistics are invalid")
    taiwan_trading_dates = payload["taiwan_trading_dates"]
    if taiwan_trading_dates is not None and (
        not isinstance(taiwan_trading_dates, int)
        or isinstance(taiwan_trading_dates, bool)
        or taiwan_trading_dates < 0
    ):
        raise ValueError("Provider checkpoint Taiwan session statistic is invalid")
    return _ProviderStats(
        estimated_calls=_nonnegative_stat(payload, "estimated_calls"),
        empty_series=_nonnegative_stat(payload, "empty_series"),
        dropped_source_rows=_nonnegative_stat(payload, "dropped_source_rows"),
        corporate_action_rows=_nonnegative_stat(payload, "corporate_action_rows"),
        missing_share_multiplier_details=_nonnegative_stat(
            payload,
            "missing_share_multiplier_details",
        ),
        skipped_unsupported_action_rows=_nonnegative_stat(
            payload,
            "skipped_unsupported_action_rows",
        ),
        skipped_unsupported_action_types=Counter(skipped_types),
        benchmark_rows=_nonnegative_stat(payload, "benchmark_rows"),
        taiwan_trading_dates=taiwan_trading_dates,
        delisted_eod_only_symbols=set(delisted_symbols),
    )


def _artifacts_from_provider_checkpoint(
    checkpoint: ProviderCheckpoint,
    *,
    reused: bool,
) -> tuple[_ProviderArtifacts, _ProviderStats]:
    metadata = checkpoint.metadata
    if set(metadata) != {"cache_hits", "provider_stats"}:
        raise ValueError("Provider checkpoint metadata is invalid")
    cache_hits = metadata["cache_hits"]
    if (
        not isinstance(cache_hits, int)
        or isinstance(cache_hits, bool)
        or not 0 <= cache_hits <= checkpoint.request_count
    ):
        raise ValueError("Provider checkpoint cache-hit count is invalid")
    stats = _provider_stats_from_payload(metadata["provider_stats"])
    return (
        _ProviderArtifacts(
            provider=checkpoint.provider,
            parquet_path=checkpoint.parquet_path,
            request_log_path=checkpoint.request_log_path,
            row_count=checkpoint.row_count,
            request_count=checkpoint.request_count,
            cache_hits=cache_hits,
            checkpoint_identity_sha256=checkpoint.identity_sha256,
            checkpoint_reused=reused,
        ),
        stats,
    )


def _checkpointed_provider_runner(
    *,
    provider: str,
    runner: Callable[[], _ProviderArtifacts],
    checkpoint_root: Path,
    checkpoint_identity: dict[str, Any],
    stats: dict[str, _ProviderStats],
) -> Callable[[], _ProviderArtifacts]:
    try:
        checkpoint = load_provider_checkpoint(
            checkpoint_root,
            provider=provider,
            identity=checkpoint_identity,
        )
        if checkpoint is not None:
            artifacts, restored_stats = _artifacts_from_provider_checkpoint(
                checkpoint,
                reused=True,
            )
            stats[provider] = restored_stats
            return lambda: artifacts
    except ValueError:
        quarantine_provider_checkpoint(
            checkpoint_root,
            provider=provider,
            identity=checkpoint_identity,
        )

    def run_and_publish() -> _ProviderArtifacts:
        local_artifacts = runner()
        try:
            published = publish_provider_checkpoint(
                checkpoint_root,
                provider=provider,
                identity=checkpoint_identity,
                parquet_path=local_artifacts.parquet_path,
                request_log_path=local_artifacts.request_log_path,
                row_count=local_artifacts.row_count,
                request_count=local_artifacts.request_count,
                metadata={
                    "cache_hits": local_artifacts.cache_hits,
                    "provider_stats": _provider_stats_payload(stats[provider]),
                },
            )
            artifacts, restored_stats = _artifacts_from_provider_checkpoint(
                published,
                reused=False,
            )
            stats[provider] = restored_stats
            return artifacts
        finally:
            local_artifacts.cleanup()

    return run_and_publish


class _ProviderDataContractError(RuntimeError):
    """Attach safe provider/item context to a non-HTTP data-contract failure."""

    def __init__(
        self,
        *,
        provider: str,
        operation: str,
        item: str,
        error: Exception,
    ) -> None:
        self.provider = provider
        self.operation = operation
        self.item = item
        self.error_type = type(error).__name__
        detail = str(error).strip()
        self.detail = detail[:500] if detail else None
        super().__init__(f"{provider} {operation} failed for {item} ({self.error_type})")

    def metadata(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "category": "provider_data_contract_error",
            "provider": self.provider,
            "operation": self.operation,
            "item": self.item,
            "error_type": self.error_type,
            "retryable": False,
        }
        if self.detail is not None:
            payload["detail"] = self.detail
        return payload


def _provider_data_call[ResultT](
    *,
    provider: str,
    operation: str,
    item: str,
    function: Callable[[], ResultT],
) -> ResultT:
    """Preserve resumable transport errors and safely contextualize data errors."""

    try:
        return function()
    except (
        AcquisitionDeadlineExceeded,
        NetworkRequestBudgetExceeded,
        ProviderRequestError,
    ):
        raise
    except Exception as error:
        raise _ProviderDataContractError(
            provider=provider,
            operation=operation,
            item=item,
            error=error,
        ) from error


def _run_parallel_provider_loops(
    runners: Mapping[str, Callable[[], Any]],
) -> dict[str, _ProviderLoopOutcome]:
    """Run independent provider loops and join every loop before returning."""

    if not runners:
        raise ValueError("At least one provider loop is required")

    def invoke(provider: str, runner: Callable[[], Any]) -> _ProviderLoopOutcome:
        try:
            return _ProviderLoopOutcome(provider=provider, result=runner())
        except Exception as error:
            if isinstance(
                error,
                (
                    AcquisitionDeadlineExceeded,
                    NetworkRequestBudgetExceeded,
                    ProviderRequestError,
                    _ProviderDataContractError,
                ),
            ):
                captured = error
            else:
                captured = _ProviderDataContractError(
                    provider=provider,
                    operation="provider_loop",
                    item=provider,
                    error=error,
                )
            return _ProviderLoopOutcome(provider=provider, error=captured)

    with ThreadPoolExecutor(
        max_workers=len(runners),
        thread_name_prefix="provider-loop",
    ) as executor:
        futures = {
            provider: executor.submit(invoke, provider, runner)
            for provider, runner in sorted(runners.items())
        }
        return {provider: futures[provider].result() for provider in sorted(futures)}


def _execute_provider_loop(
    *,
    provider: str,
    parts_root: Path,
    operation: Callable[[_ParquetSink, _RequestLog], None],
) -> _ProviderArtifacts:
    parquet_path = parts_root / f"{provider}.parquet"
    request_log_path = parts_root / f"{provider}-requests.jsonl"
    sink = _ParquetSink(parquet_path)
    request_log = _RequestLog(request_log_path)
    try:
        operation(sink, request_log)
        sink.close()
        request_log.close()
    except BaseException:
        sink.abort()
        request_log.abort()
        parquet_path.unlink(missing_ok=True)
        request_log_path.unlink(missing_ok=True)
        raise
    return _ProviderArtifacts(
        provider=provider,
        parquet_path=parquet_path,
        request_log_path=request_log_path,
        row_count=sink.row_count,
        request_count=request_log.count,
        cache_hits=request_log.cache_hits,
    )


def _safe_provider_outcome(error: Exception) -> dict[str, Any]:
    if isinstance(error, _ProviderDataContractError):
        return {"state": "failed", "last_error": error.metadata()}
    if isinstance(error, ProviderRequestError):
        return {
            "state": "waiting_for_provider" if error.retryable else "failed",
            "last_error": error.metadata(),
        }
    if isinstance(error, NetworkRequestBudgetExceeded):
        budget: dict[str, Any] = {
            "category": "network_request_safety_budget_exhausted",
            "retryable": True,
            "consumed": error.consumed,
            "maximum": error.maximum,
        }
        if error.provider is not None:
            budget["provider"] = error.provider
        return {"state": "waiting_for_budget", "last_error": budget}
    if isinstance(error, AcquisitionDeadlineExceeded):
        time_budget: dict[str, Any] = {
            "category": "acquisition_time_budget_exhausted",
            "retryable": True,
            "deadline_epoch_seconds": error.deadline_epoch_seconds,
            "observed_epoch_seconds": error.observed_epoch_seconds,
        }
        if error.required_wait_seconds is not None:
            time_budget["required_wait_seconds"] = error.required_wait_seconds
        return {"state": "waiting_for_resume", "last_error": time_budget}
    return {
        "state": "failed",
        "last_error": {
            "category": "non_provider_failure",
            "retryable": False,
            "error_type": type(error).__name__,
        },
    }


def _skipped_action_outcome(stats: _ProviderStats) -> dict[str, Any]:
    if stats.skipped_unsupported_action_rows == 0:
        return {}
    return {
        "skipped_unsupported_action_rows": stats.skipped_unsupported_action_rows,
        "skipped_unsupported_action_types": dict(
            sorted(stats.skipped_unsupported_action_types.items())
        ),
    }


def _skipped_action_quality(
    stats: Mapping[str, _ProviderStats],
) -> dict[str, Any]:
    combined_types: Counter[str] = Counter()
    by_provider: dict[str, dict[str, Any]] = {}
    total_rows = 0
    for provider, provider_stats in sorted(stats.items()):
        audit = _skipped_action_outcome(provider_stats)
        if not audit:
            continue
        total_rows += provider_stats.skipped_unsupported_action_rows
        combined_types.update(provider_stats.skipped_unsupported_action_types)
        by_provider[provider] = {
            "rows": provider_stats.skipped_unsupported_action_rows,
            "types": dict(sorted(provider_stats.skipped_unsupported_action_types.items())),
        }
    return {
        "skipped_unsupported_action_rows": total_rows,
        "skipped_unsupported_action_types": dict(sorted(combined_types.items())),
        "skipped_unsupported_actions_by_provider": by_provider,
    }


def _aggregate_provider_error(
    outcomes: Mapping[str, _ProviderLoopOutcome],
    stats: Mapping[str, _ProviderStats],
) -> ProviderAcquisitionError | None:
    provider_outcomes: dict[str, dict[str, Any]] = {}
    states: set[str] = set()
    for provider, outcome in sorted(outcomes.items()):
        if outcome.error is None:
            artifacts = outcome.result
            if not isinstance(artifacts, _ProviderArtifacts):
                raise TypeError(f"Provider loop returned invalid artifacts: {provider}")
            provider_outcomes[provider] = {
                "state": "complete",
                "recorded_requests": artifacts.request_count,
                "cache_hits": artifacts.cache_hits,
                "rows": artifacts.row_count,
                "materialization_checkpoint": {
                    "identity_sha256": artifacts.checkpoint_identity_sha256,
                    "reused": artifacts.checkpoint_reused,
                },
                **_skipped_action_outcome(stats[provider]),
            }
            continue
        payload = _safe_provider_outcome(outcome.error)
        payload["estimated_calls"] = stats[provider].estimated_calls or None
        payload.update(_skipped_action_outcome(stats[provider]))
        provider_outcomes[provider] = payload
        states.add(str(payload["state"]))
    if not states:
        return None
    if "failed" in states:
        state = "failed"
    elif "waiting_for_resume" in states:
        state = "waiting_for_resume"
    elif "waiting_for_provider" in states:
        state = "waiting_for_provider"
    else:
        state = "waiting_for_budget"
    return ProviderAcquisitionError(
        state=state,
        provider_outcomes=provider_outcomes,
        retryable=state != "failed",
    )


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


def _benchmark_trading_dates(
    frame: pd.DataFrame,
    *,
    start: str,
    exclusive_end: str,
) -> set[str]:
    """Derive official sessions from the already-required benchmark history."""

    if frame.empty:
        return set()
    return {
        trading_date
        for timestamp in frame["timestamp"]
        if start <= (trading_date := pd.Timestamp(timestamp).date().isoformat()) < exclusive_end
    }


def _discover_eodhd_materialization_universe(
    provider: EODHDProvider,
    *,
    include_delisted: bool,
    explicit_us_symbols: tuple[str, ...],
    explicit_us_etfs: tuple[str, ...],
    symbol_limit: int | None,
) -> tuple[list[Instrument], tuple[RequestRecord, ...]]:
    """Resolve exactly which EODHD instruments enter durable materialization."""

    discovered, requests = provider.discover(include_delisted=include_delisted)
    return (
        _select_eodhd_materialization_universe(
            discovered,
            explicit_us_symbols=explicit_us_symbols,
            explicit_us_etfs=explicit_us_etfs,
            symbol_limit=symbol_limit,
        ),
        requests,
    )


def _fetch_eodhd_materialized_instrument(
    provider: EODHDProvider,
    instrument: Instrument,
    *,
    start: str,
    exclusive_end: str,
    dataset_profile: str,
) -> tuple[Any, Any]:
    """Apply the exclusive dataset boundary to one EODHD materialization."""

    return provider.fetch_materialized_instrument(
        instrument,
        start=start,
        end=_inclusive_end(exclusive_end),
        dataset_profile=dataset_profile,
    )


def _taiwan_materialization_months(start: str, exclusive_end: str) -> list[str]:
    """Return every benchmark month needed by a Taiwan provider."""

    return _months(start, exclusive_end)


def _fetch_taiwan_actions(
    provider: TWSEProvider | TPExProvider,
    *,
    start: str,
    exclusive_end: str,
) -> Any:
    """Fetch Taiwan actions using the dataset's exclusive upper boundary."""

    return provider.fetch_actions(
        start=start,
        end=_inclusive_end(exclusive_end),
    )


def _fetch_taiwan_benchmark_month(
    provider: TWSEProvider | TPExProvider,
    *,
    month: str,
    dataset_profile: str,
) -> Any:
    """Fetch the benchmark history that defines official trading sessions."""

    return provider.fetch_benchmark_month(
        month=month,
        dataset_profile=dataset_profile,
    )


def _taiwan_materialization_dates(
    benchmark_frame: pd.DataFrame,
    *,
    start: str,
    exclusive_end: str,
) -> set[str]:
    """Keep only official benchmark sessions inside the dataset interval."""

    return _benchmark_trading_dates(
        benchmark_frame,
        start=start,
        exclusive_end=exclusive_end,
    )


def _fetch_taiwan_adjusted_date(
    provider: TWSEProvider | TPExProvider,
    *,
    trading_date: str,
    dataset_profile: str,
    action_frame: pd.DataFrame,
) -> tuple[Any, pd.DataFrame]:
    """Fetch and adjust one official Taiwan market session."""

    return provider.fetch_adjusted_date(
        date=trading_date,
        dataset_profile=dataset_profile,
        action_frame=action_frame,
    )


def _dataset_cache_fallback_roots(
    raw_cache_root: Path,
    *,
    cache_revision: str = "v1",
) -> tuple[Path, ...]:
    """Find read-only cache roots with the same provider-cache revision."""

    primary = raw_cache_root.resolve(strict=False)
    dataset_root = primary.parent
    datasets_root = dataset_root.parent
    if datasets_root.name != "datasets" or not datasets_root.is_dir():
        return ()
    candidates: list[Path] = []
    for namespace in sorted(datasets_root.iterdir()):
        candidate = namespace / "api-cache"
        declared_revision = "v1"
        for metadata_path, revision_path in (
            (namespace / "download-progress.json", ("identity", "cache_revision")),
            (namespace / "download-manifest.json", ("api_policy", "cache_revision")),
        ):
            if not metadata_path.is_file() or metadata_path.is_symlink():
                continue
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(metadata, dict):
                continue
            container = metadata.get(revision_path[0])
            if not isinstance(container, dict) or revision_path[1] not in container:
                # Manifests written before cache revisions belong to legacy v1.
                continue
            value = container.get(revision_path[1])
            declared_revision = value if isinstance(value, str) else ""
            break
        if (
            namespace != dataset_root
            and namespace.is_dir()
            and not namespace.is_symlink()
            and candidate.is_dir()
            and not candidate.is_symlink()
            and declared_revision == cache_revision
        ):
            candidates.append(candidate)
    return tuple(candidates)


def _explicit_instruments(
    symbols: tuple[str, ...],
    etf_symbols: tuple[str, ...],
) -> list[Instrument]:
    etfs = {symbol.upper().removesuffix(".US") for symbol in etf_symbols}
    unsupported_etfs = sorted(
        code
        for code in etfs
        if not is_allowlisted_us_equity_etf(symbol=f"{code}.US")
    )
    if unsupported_etfs:
        raise ValueError(
            "Explicit US ETFs must belong to the audited unleveraged equity allowlist: "
            + ", ".join(unsupported_etfs)
        )
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


def _validate_explicit_instruments(
    requested: list[Instrument],
    discovered: list[Instrument],
) -> list[Instrument]:
    """Bind explicit symbols to the provider's own stock-versus-ETF typing."""

    by_symbol = {item.canonical_symbol: item for item in discovered}
    validated: list[Instrument] = []
    for item in requested:
        provider_item = by_symbol.get(item.canonical_symbol)
        if provider_item is None:
            raise ValueError(
                f"Explicit US symbol is absent from EODHD discovery: {item.canonical_symbol}"
            )
        if provider_item.asset_type != item.asset_type:
            raise ValueError(
                "Explicit US symbol type disagrees with EODHD discovery: "
                f"{item.canonical_symbol} requested={item.asset_type} "
                f"provider={provider_item.asset_type}"
            )
        validated.append(provider_item)
    return validated


def _limit_instruments(
    instruments: list[Instrument],
    limit: int | None,
) -> list[Instrument]:
    if limit is not None and limit < 1:
        raise ValueError("symbol_limit must be positive")
    unsupported_types = sorted(
        {item.asset_type for item in instruments} - TRAINING_TARGET_ASSET_TYPES
    )
    if unsupported_types:
        raise ValueError(
            f"US symbol limiting supports only ETF and stock instruments: {unsupported_types}"
        )
    instruments = [
        item
        for item in instruments
        if item.asset_type == "stock"
        or is_allowlisted_us_equity_etf(symbol=item.canonical_symbol)
    ]
    selected: list[Instrument] = []
    for asset_type in ("etf", "stock"):
        ordered = sorted(
            (item for item in instruments if item.asset_type == asset_type),
            key=lambda item: (
                not item.is_active,
                item.canonical_symbol,
            ),
        )
        selected.extend(ordered if limit is None else ordered[:limit])
    return selected


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


def _select_eodhd_materialization_universe(
    discovered: list[Instrument],
    *,
    explicit_us_symbols: tuple[str, ...],
    explicit_us_etfs: tuple[str, ...],
    symbol_limit: int | None,
) -> list[Instrument]:
    """Resolve the complete EODHD universe that will be persisted."""

    if explicit_us_symbols or explicit_us_etfs:
        instruments = _validate_explicit_instruments(
            _explicit_instruments(explicit_us_symbols, explicit_us_etfs),
            discovered,
        )
    else:
        instruments = discovered
    return _ensure_us_benchmark(_limit_instruments(instruments, symbol_limit))


def _progress_identity(options: IngestionOptions) -> dict[str, Any]:
    identity: dict[str, Any] = {
        "cache_revision": options.cache_revision,
        "dataset_profile": options.profile,
        "selected_datasets": selected_datasets(options.profile),
        "date_range": {
            "start_inclusive": options.start,
            "end_exclusive": options.end,
        },
        "universe": {
            "mode": (
                "explicit" if options.explicit_us_symbols or options.explicit_us_etfs else "all"
            ),
            "us_stocks": sorted(
                {f"{value.upper().removesuffix('.US')}.US" for value in options.explicit_us_symbols}
            ),
            "us_etfs": sorted(
                {f"{value.upper().removesuffix('.US')}.US" for value in options.explicit_us_etfs}
            ),
            "symbol_limit": options.symbol_limit,
            "include_delisted_us": options.include_delisted,
        },
    }
    return identity


def _provider_checkpoint_identity(
    options: IngestionOptions,
    *,
    provider: str,
) -> dict[str, Any]:
    """Bind reusable provider materialization to content-affecting inputs only."""

    materialization_request: dict[str, Any] = {
        "cache_revision": options.cache_revision,
        "dataset_profile": options.profile,
        "date_range": {
            "start_inclusive": options.start,
            "end_exclusive": options.end,
        },
    }
    if provider == "eodhd":
        materialization_request["universe"] = {
            "mode": (
                "explicit"
                if options.explicit_us_symbols or options.explicit_us_etfs
                else "all"
            ),
            "us_stocks": sorted(
                {
                    f"{value.upper().removesuffix('.US')}.US"
                    for value in options.explicit_us_symbols
                }
            ),
            "us_etfs": sorted(
                {
                    f"{value.upper().removesuffix('.US')}.US"
                    for value in options.explicit_us_etfs
                }
            ),
            "symbol_limit": options.symbol_limit,
            "include_delisted_us": options.include_delisted,
        }
    return {
        "materialization_digest": provider_materialization_digest(provider),
        "provider": provider,
        "materialization_request": materialization_request,
    }


def _progress_context(options: IngestionOptions) -> dict[str, Any]:
    context: dict[str, Any] = {
        "acquisition_policy": {
            "max_api_calls": options.max_api_calls,
            "max_api_calls_limited_providers": ["eodhd"],
            "eodhd_requests_per_second": options.eodhd_requests_per_second,
            "taiwan_requests_per_second_per_provider": options.taiwan_requests_per_second,
            "provider_max_backoff_seconds": options.max_backoff_seconds,
            "provider_max_backoff_scope": ["eodhd", "tpex_official", "twse_official"],
            "tpex_transport": (
                "cloud_run_relay_v1" if options.tpex_proxy_url else "direct"
            ),
        }
    }
    optional = {
        "selection_id": options.selection_id,
        "selection_sha256": options.selection_sha256,
        "dataset_request_sha256": options.dataset_request_sha256,
        "launch_id": options.launch_id,
        "acquisition_deadline_epoch_seconds": (options.acquisition_deadline_epoch_seconds),
        "preparation_reserve_seconds": options.preparation_reserve_seconds,
    }
    context.update({key: value for key, value in optional.items() if value is not None})
    return context


def ingest_daily_ohlcv(
    options: IngestionOptions,
    *,
    eodhd_api_token: str | None = None,
    tpex_proxy_token: str | None = None,
) -> dict[str, Any]:
    """Fetch providers only here; training and evaluation never call this function."""

    try:
        active_providers = runtime_providers(options.profile)
    except KeyError as error:
        raise ValueError(f"Unsupported dataset profile: {options.profile}") from error
    if "massive" in active_providers:
        MassiveProvider().discover(include_delisted=options.include_delisted)
    taiwan_provider_classes = tuple(
        provider_class
        for provider_class in (TWSEProvider, TPExProvider)
        if provider_class.name in active_providers
    )
    if options.max_api_calls < 1:
        raise ValueError("max_api_calls must be positive")
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", options.cache_revision) is None:
        raise ValueError("cache_revision must be a safe 1-64 character label")
    if not math.isfinite(options.max_backoff_seconds) or options.max_backoff_seconds <= 0.0:
        raise ValueError("max_backoff_seconds must be finite and positive")
    if options.workers < 1:
        raise ValueError("workers must be positive")
    if options.preparation_reserve_seconds is not None and options.preparation_reserve_seconds < 1:
        raise ValueError("preparation_reserve_seconds must be positive")
    if bool(options.tpex_proxy_url) != bool(tpex_proxy_token):
        raise ValueError("TPEX_PROXY_URL and TPEX_PROXY_TOKEN must be configured together")
    has_explicit_us_universe = bool(options.explicit_us_symbols or options.explicit_us_etfs)
    if options.profile == "tw_only" and (
        has_explicit_us_universe or options.symbol_limit is not None
    ):
        raise ValueError("tw_only cannot accept US symbols or symbol_limit")
    if has_explicit_us_universe and options.symbol_limit is not None:
        raise ValueError("symbol_limit cannot be combined with explicit US symbols")
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
    if request_log_path.exists():
        raise FileExistsError(f"Refusing to overwrite request log: {request_log_path}")

    request_budget = NetworkRequestBudget(
        options.max_api_calls,
        deadline_epoch_seconds=options.acquisition_deadline_epoch_seconds,
        limited_providers={"eodhd"},
    )
    cache_fallback_roots = _dataset_cache_fallback_roots(
        options.raw_cache_root,
        cache_revision=options.cache_revision,
    )
    provider_checkpoint_root = (
        options.provider_checkpoint_root
        if options.provider_checkpoint_root is not None
        else options.raw_cache_root.parent / "provider-checkpoints"
    )
    if provider_checkpoint_root.exists() and provider_checkpoint_root.is_symlink():
        raise ValueError("Provider checkpoint root must not be a symlink")
    progress = DownloadProgress(
        path=options.progress_path or options.manifest_root / "download-progress.json",
        raw_cache_root=options.raw_cache_root,
        read_cache_roots=cache_fallback_roots,
        identity=_progress_identity(options),
        context=_progress_context(options),
    )
    progress.start(request_budget)
    provider_workers = max(1, options.workers // len(active_providers))
    stats = {provider: _ProviderStats() for provider in active_providers}
    parts_root = options.manifest_root / f".provider-parts-{uuid.uuid4().hex}"
    parts_root.mkdir(parents=True, exist_ok=False)
    runners: dict[str, Callable[[], _ProviderArtifacts]] = {}
    taiwan_pre_calendar_upper_bound_calls = 0
    taiwan_months: list[str] = []
    outcomes: dict[str, _ProviderLoopOutcome] = {}
    sink: _ParquetSink | None = None
    request_log: _RequestLog | None = None
    try:
        if taiwan_provider_classes:
            taiwan_weekdays = _weekdays(options.start, options.end)
            taiwan_months = _taiwan_materialization_months(
                options.start,
                options.end,
            )
            taiwan_provider_count = len(taiwan_provider_classes)
            taiwan_fixed_calls = (
                len(taiwan_months) * 2 + 1
            ) * taiwan_provider_count
            taiwan_pre_calendar_upper_bound_calls = (
                len(taiwan_weekdays) * taiwan_provider_count
                + taiwan_fixed_calls
            )
            for provider_class in taiwan_provider_classes:
                provider_name = provider_class.name
                stats[provider_name].estimated_calls = len(taiwan_months) * 2 + 1
        if "eodhd" in active_providers:
            if not eodhd_api_token:
                raise ValueError("EODHD_API_TOKEN is required for the selected dataset profile")

            def run_eodhd(sink_part: _ParquetSink, log_part: _RequestLog) -> None:
                provider_stats = stats["eodhd"]
                eod_client = CachedJsonClient(
                    provider="eodhd",
                    raw_cache_root=options.raw_cache_root,
                    read_cache_roots=cache_fallback_roots,
                    cache_revision=options.cache_revision,
                    max_requests_per_second=options.eodhd_requests_per_second,
                    max_backoff_seconds=options.max_backoff_seconds,
                    request_budget=request_budget,
                )
                provider = EODHDProvider(eod_client, api_token=eodhd_api_token)
                instruments, discovery_requests = _provider_data_call(
                    provider="eodhd",
                    operation="discover",
                    item="US",
                    function=lambda: _discover_eodhd_materialization_universe(
                        provider,
                        include_delisted=options.include_delisted,
                        explicit_us_symbols=options.explicit_us_symbols,
                        explicit_us_etfs=options.explicit_us_etfs,
                        symbol_limit=options.symbol_limit,
                    ),
                )
                log_part.append(discovery_requests)
                provider_stats.estimated_calls += len(discovery_requests) + len(instruments) * 2

                def fetch_eodhd_instrument(
                    instrument: Instrument,
                ) -> tuple[Any, Any, Instrument]:
                    instrument_split_fetch, fetched = _provider_data_call(
                        provider="eodhd",
                        operation="materialize_instrument",
                        item=instrument.canonical_symbol,
                        function=lambda: _fetch_eodhd_materialized_instrument(
                            provider,
                            instrument,
                            start=options.start,
                            exclusive_end=options.end,
                            dataset_profile=options.profile,
                        ),
                    )
                    return instrument_split_fetch, fetched, instrument

                with closing(
                    _ordered_thread_map(
                        fetch_eodhd_instrument,
                        instruments,
                        max_workers=min(provider_workers, max(len(instruments), 1)),
                        thread_name_prefix="eodhd",
                    )
                ) as fetched_instruments:
                    for instrument_split_fetch, fetched, instrument in fetched_instruments:
                        request_budget.check_time()
                        log_part.append(instrument_split_fetch.requests)
                        provider_stats.corporate_action_rows += len(instrument_split_fetch.frame)
                        provider_stats.dropped_source_rows += int(
                            instrument_split_fetch.metadata.get("dropped_rows", 0)
                        )
                        log_part.append(fetched.requests)
                        provider_stats.dropped_source_rows += int(
                            fetched.metadata.get("dropped_rows", 0)
                        )
                        if fetched.frame.empty:
                            provider_stats.empty_series += 1
                            continue
                        if not instrument.is_active and fetched.frame[
                            "timestamp"
                        ].max() < pd.Timestamp(EODHD_DELISTED_AUXILIARY_DATA_START, tz="UTC"):
                            provider_stats.delisted_eod_only_symbols.add(
                                instrument.canonical_symbol
                            )
                        sink_part.write(fetched.frame)

            runners["eodhd"] = lambda: _execute_provider_loop(
                provider="eodhd",
                parts_root=parts_root,
                operation=run_eodhd,
            )

        if taiwan_provider_classes:

            def make_taiwan_runner(
                provider_class: type[TWSEProvider] | type[TPExProvider],
            ) -> Callable[[], _ProviderArtifacts]:
                provider_name = provider_class.name

                def run_taiwan(sink_part: _ParquetSink, log_part: _RequestLog) -> None:
                    provider_stats = stats[provider_name]
                    tpex_transport = (
                        TpexRelayTransport(
                            origin=options.tpex_proxy_url,
                            token=tpex_proxy_token,
                        )
                        if provider_name == "tpex_official"
                        and options.tpex_proxy_url is not None
                        and tpex_proxy_token is not None
                        else None
                    )
                    client = CachedJsonClient(
                        provider=provider_name,
                        raw_cache_root=options.raw_cache_root,
                        read_cache_roots=cache_fallback_roots,
                        cache_revision=options.cache_revision,
                        max_requests_per_second=options.taiwan_requests_per_second,
                        max_backoff_seconds=options.max_backoff_seconds,
                        headers={
                            "User-Agent": (
                                "Mozilla/5.0 (X11; Linux x86_64) "
                                "AppleWebKit/537.36 (KHTML, like Gecko) "
                                "Chrome/140.0 Safari/537.36"
                            ),
                            "Accept": "application/json,text/plain,*/*",
                            "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
                            "Referer": (
                                "https://www.twse.com.tw/"
                                if provider_name == "twse_official"
                                else "https://www.tpex.org.tw/"
                            ),
                        },
                        retryable_status_codes={403, 408, 425, 429},
                        transport=tpex_transport,
                        request_budget=request_budget,
                    )
                    provider = provider_class(client)
                    action_fetch = _provider_data_call(
                        provider=provider_name,
                        operation="corporate_actions",
                        item=f"{options.start}:{_inclusive_end(options.end)}",
                        function=lambda: _fetch_taiwan_actions(
                            provider,
                            start=options.start,
                            exclusive_end=options.end,
                        ),
                    )
                    log_part.append(action_fetch.requests)
                    provider_stats.estimated_calls += max(
                        0,
                        len(action_fetch.requests) - 1,
                    )
                    provider_stats.corporate_action_rows += len(action_fetch.frame)
                    provider_stats.dropped_source_rows += int(
                        action_fetch.metadata.get("dropped_rows", 0)
                    )
                    provider_stats.missing_share_multiplier_details += int(
                        action_fetch.metadata.get("missing_share_multiplier_details", 0)
                    )
                    provider_stats.skipped_unsupported_action_rows += int(
                        action_fetch.metadata.get("skipped_unsupported_action_rows", 0)
                    )
                    provider_stats.skipped_unsupported_action_types.update(
                        action_fetch.metadata.get("skipped_unsupported_action_types", {})
                    )

                    def fetch_taiwan_date(
                        trading_date: str,
                    ) -> tuple[Any, pd.DataFrame]:
                        return _provider_data_call(
                            provider=provider_name,
                            operation="daily_quotes",
                            item=trading_date,
                            function=lambda: _fetch_taiwan_adjusted_date(
                                provider,
                                trading_date=trading_date,
                                dataset_profile=options.profile,
                                action_frame=action_fetch.frame,
                            ),
                        )

                    def fetch_taiwan_month(month: str) -> Any:
                        return _provider_data_call(
                            provider=provider_name,
                            operation="benchmark_month",
                            item=month,
                            function=lambda: _fetch_taiwan_benchmark_month(
                                provider,
                                month=month,
                                dataset_profile=options.profile,
                            ),
                        )

                    provider_trading_dates: set[str] = set()
                    with closing(
                        _ordered_thread_map(
                            fetch_taiwan_month,
                            taiwan_months,
                            max_workers=min(
                                provider_workers,
                                max(len(taiwan_months), 1),
                            ),
                            thread_name_prefix=f"{provider_name}-month",
                        )
                    ) as fetched_months:
                        for benchmark_fetch in fetched_months:
                            request_budget.check_time()
                            log_part.append(benchmark_fetch.requests)
                            provider_stats.benchmark_rows += len(benchmark_fetch.frame)
                            sink_part.write(benchmark_fetch.frame)
                            provider_trading_dates.update(
                                _taiwan_materialization_dates(
                                    benchmark_fetch.frame,
                                    start=options.start,
                                    exclusive_end=options.end,
                                )
                            )
                    if not provider_trading_dates:
                        raise ValueError(
                            f"{provider_name} benchmark history exposed no trading dates "
                            "inside the requested interval"
                        )
                    scheduled_dates = sorted(provider_trading_dates)
                    provider_stats.taiwan_trading_dates = len(scheduled_dates)
                    provider_stats.estimated_calls += len(scheduled_dates)
                    with closing(
                        _ordered_thread_map(
                            fetch_taiwan_date,
                            scheduled_dates,
                            max_workers=min(provider_workers, len(scheduled_dates)),
                            thread_name_prefix=f"{provider_name}-date",
                        )
                    ) as fetched_dates:
                        for fetched, adjusted in fetched_dates:
                            request_budget.check_time()
                            log_part.append(fetched.requests)
                            provider_stats.dropped_source_rows += int(
                                fetched.metadata.get("dropped_rows", 0)
                            )
                            sink_part.write(adjusted)

                return lambda: _execute_provider_loop(
                    provider=provider_name,
                    parts_root=parts_root,
                    operation=run_taiwan,
                )

            for provider_class in taiwan_provider_classes:
                runners[provider_class.name] = make_taiwan_runner(provider_class)

        runners = {
            provider: _checkpointed_provider_runner(
                provider=provider,
                runner=runner,
                checkpoint_root=provider_checkpoint_root,
                checkpoint_identity=_provider_checkpoint_identity(
                    options,
                    provider=provider,
                ),
                stats=stats,
            )
            for provider, runner in sorted(runners.items())
        }
        outcomes = _run_parallel_provider_loops(runners)
        aggregate_error = _aggregate_provider_error(outcomes, stats)
        if aggregate_error is not None:
            raise aggregate_error

        completed_artifacts = _ordered_completed_provider_artifacts(outcomes)
        sink = _ParquetSink(options.output)
        request_log = _RequestLog(request_log_path)
        for artifacts in completed_artifacts:
            sink.append_parquet(
                artifacts.parquet_path,
                check_time=request_budget.check_time,
            )
            request_log.append_file(artifacts.request_log_path)
        request_budget.check_time()
        sink.close()
        request_log.close()
    except BaseException as error:
        if sink is not None:
            sink.abort()
            options.output.unlink(missing_ok=True)
        if request_log is not None:
            request_log.abort()
            request_log_path.unlink(missing_ok=True)
        progress.fail(
            error,
            request_budget,
            estimated_http_requests=(
                sum(provider_stats.estimated_calls for provider_stats in stats.values()) or None
            ),
        )
        raise
    finally:
        for outcome in outcomes.values():
            if isinstance(outcome.result, _ProviderArtifacts):
                outcome.result.cleanup()
        with suppress(OSError):
            parts_root.rmdir()

    if sink is None or request_log is None:
        raise RuntimeError("Provider acquisition completed without merged artifacts")
    estimated_calls = sum(provider_stats.estimated_calls for provider_stats in stats.values())
    empty_series = sum(provider_stats.empty_series for provider_stats in stats.values())
    dropped_source_rows = sum(
        provider_stats.dropped_source_rows for provider_stats in stats.values()
    )
    corporate_action_rows = sum(
        provider_stats.corporate_action_rows for provider_stats in stats.values()
    )
    missing_share_multiplier_details = {
        provider: provider_stats.missing_share_multiplier_details
        for provider, provider_stats in sorted(stats.items())
        if provider_stats.missing_share_multiplier_details > 0
    }
    skipped_action_quality = _skipped_action_quality(stats)
    benchmark_rows = sum(provider_stats.benchmark_rows for provider_stats in stats.values())
    delisted_eod_only_symbols = set().union(
        *(provider_stats.delisted_eod_only_symbols for provider_stats in stats.values())
    )
    taiwan_trading_date_counts = {
        provider: provider_stats.taiwan_trading_dates
        for provider, provider_stats in sorted(stats.items())
        if provider_stats.taiwan_trading_dates is not None
    }
    provider_materialization_checkpoints: dict[str, dict[str, Any]] = {}
    for artifacts in _ordered_completed_provider_artifacts(outcomes):
        provider = artifacts.provider
        if artifacts.checkpoint_identity_sha256 is None:
            raise RuntimeError(
                f"Provider completed without a durable materialization checkpoint: {provider}"
            )
        provider_materialization_checkpoints[provider] = {
            "identity_sha256": artifacts.checkpoint_identity_sha256,
            "reused": artifacts.checkpoint_reused,
        }

    selected_dataset_ids = selected_datasets(options.profile)
    provider_materialization_digests = {
        selected: provider_materialization_digest(selected)
        for selected in selected_dataset_ids
    }

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
        "schema_version": DOWNLOAD_MANIFEST_SCHEMA_VERSION,
        "kind": "ohlcv-dataset",
        "state": "downloaded",
        "created_at": datetime.now(UTC).isoformat(),
        "training_security_scope": TRAINING_SECURITY_SCOPE,
        "dataset_profile": options.profile,
        "selected_datasets": selected_dataset_ids,
        "data_content_identity": {
            "schema_version": CONTENT_IDENTITY_SCHEMA_VERSION,
            "selected_datasets": selected_dataset_ids,
            "provider_materialization_digests": provider_materialization_digests,
            "raw_materialization_digest": raw_dataset_materialization_digest(),
        },
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
            "cache_revision": options.cache_revision,
            "estimated_calls": estimated_calls,
            "estimated_calls_semantics": (
                "informational_cache_agnostic_plan_with_official_trading_sessions_"
                "not_an_admission_gate"
            ),
            "taiwan_pre_calendar_weekday_upper_bound_calls": (
                taiwan_pre_calendar_upper_bound_calls or None
            ),
            "max_api_calls": options.max_api_calls,
            "max_api_calls_semantics": (
                "per_acquisition_attempt_eodhd_cache_miss_and_retry_network_safety_ceiling"
            ),
            "max_api_calls_limited_providers": ["eodhd"],
            "provider_billing_quota_accounting": "external_to_max_api_calls",
            "recorded_requests": request_log.count,
            "network_requests": request_budget.network_requests,
            "network_requests_semantics": (
                "current_acquisition_attempt_only_excludes_reused_provider_checkpoints"
            ),
            "eodhd_limited_network_requests": request_budget.limited_network_requests,
            "network_requests_by_provider": request_budget.provider_counts,
            "cache_hits": request_log.cache_hits,
            "eodhd_requests_per_second": options.eodhd_requests_per_second,
            "eodhd_requests_per_minute_equivalent": (options.eodhd_requests_per_second * 60.0),
            "eodhd_official_default_daily_api_call_limit": (EODHD_DEFAULT_DAILY_API_CALL_LIMIT),
            "eodhd_official_default_requests_per_minute": (EODHD_DEFAULT_REQUESTS_PER_MINUTE),
            "taiwan_requests_per_second": options.taiwan_requests_per_second,
            "taiwan_requests_per_second_scope": "per_provider",
            "tpex_transport": (
                "cloud_run_relay_v1" if options.tpex_proxy_url else "direct"
            ),
            "provider_max_backoff_seconds": options.max_backoff_seconds,
            "provider_max_backoff_scope": ["eodhd", "tpex_official", "twse_official"],
            "taiwan_request_count_ceiling": None,
            "provider_execution": (
                "parallel_independent_loops_joined_before_process_exit_with_"
                "durable_provider_checkpoints_and_deterministic_merge"
            ),
            "provider_execution_workers_each": provider_workers,
            "provider_materialization_identity": "content_only_semantic_ast_v1",
            "raw_dataset_assembly_policy": RAW_DATASET_ASSEMBLY_POLICY,
            "provider_materialization_checkpoint_scope": (
                "per_dataset_request_and_provider_fail_closed_integrity_checked"
            ),
            "provider_materialization_checkpoints": (
                provider_materialization_checkpoints
            ),
            "read_only_cache_fallback_namespaces": len(cache_fallback_roots),
            "taiwan_trading_dates_by_provider": taiwan_trading_date_counts,
            "execution_workers": options.workers,
            "include_delisted": options.include_delisted,
            "symbol_limit": options.symbol_limit,
            "symbol_limit_semantics": (
                "up_to_n_allowlisted_unleveraged_equity_etfs_and_n_stocks_"
                "then_required_vti_benchmark"
            ),
            "eodhd_split_strategy": (
                "per_symbol_historical_splits_reconstruct_unadjusted_volume"
                if "eodhd" in active_providers
                else None
            ),
            "preparation_reserve_seconds": options.preparation_reserve_seconds,
        },
        "quality": {
            "schema_validated_rows": sink.row_count,
            "empty_symbol_histories": empty_series,
            "dropped_source_rows": dropped_source_rows,
            "corporate_action_rows": corporate_action_rows,
            "missing_share_multiplier_details_by_provider": (missing_share_multiplier_details),
            **skipped_action_quality,
            "benchmark_rows": benchmark_rows,
            "delisted_pre_2018_auxiliary_coverage_warning": {
                "count": len(delisted_eod_only_symbols),
                "symbols": sorted(delisted_eod_only_symbols),
                "meaning": ("eodhd_documents_eod_only_for_symbols_delisted_before_2018"),
            },
            "adjustment_contract": {
                "raw_price_fields_immutable": True,
                "canonical_volume": (
                    "provider_raw_except_eodhd_reconstructed_from_vendor_split_adjusted_volume"
                ),
                "price": "total_return_adjusted_as_of_cutoff",
                "volume": "share_change_adjusted_as_of_cutoff",
                "eodhd_volume_anchor": (
                    "vendor_split_adjusted_volume_with_unadjusted_volume_reconstructed"
                ),
                "benchmark": "official_total_return_index_or_vti",
                "taiwan_volume_adjustment_coverage": (
                    "official_share_multiplier_when_available; identity_multiplier_"
                    "for_missing_twse_historical_detail_with_gap_count"
                ),
                "delisted_before_2018": (
                    "eod_available_but_auxiliary_split_coverage_not_guaranteed"
                ),
            },
            "training_security_scope": {
                "contract": TRAINING_SECURITY_SCOPE,
                "stock_semantics": "common_stock_including_adr_and_tdr",
                "etf_semantics": "audited_allowlist_excluding_leveraged_and_inverse_products",
            },
        },
        "artifacts": {
            "raw": raw_artifact,
            "request_log": request_artifact,
        },
    }
    atomic_write_json(download_manifest_path, payload)
    progress.complete(
        request_budget,
        estimated_http_requests=estimated_calls or None,
    )
    return {
        **payload,
        "download_manifest_path": str(download_manifest_path),
        "download_progress_path": str(progress.path),
        "raw_parquet_path": str(options.output),
        "request_log_path": str(request_log_path),
    }
