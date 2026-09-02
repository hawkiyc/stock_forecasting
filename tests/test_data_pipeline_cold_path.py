"""Deterministic cold-path coverage for acquisition, resume, and lazy preparation."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

import stock_forecasting.data.ingestion as ingestion
from stock_forecasting.cli.prepare_data import main as prepare_data_main
from stock_forecasting.data.ingestion import IngestionOptions, ingest_daily_ohlcv
from stock_forecasting.data.manifest import (
    load_dataset_manifest,
    validate_download_manifest,
    validate_training_dataset_manifest,
)
from stock_forecasting.data.providers.base import (
    Instrument,
    ProviderFetch,
    RequestRecord,
)
from stock_forecasting.data.providers.eodhd import EODHDProvider
from stock_forecasting.data.providers.taiwan import TPExProvider, TWSEProvider
from stock_forecasting.data.schema import normalize_ohlcv_frame

_DATES = pd.bdate_range("2023-01-02", "2024-07-05", tz="UTC")


def _request(provider: str, operation: str) -> RequestRecord:
    digest = hashlib.sha256(f"{provider}:{operation}".encode()).hexdigest()
    return RequestRecord(
        provider=provider,
        request_sha256=digest,
        cache_relative_path=f"{provider}/{digest[:2]}/{digest}.json",
        response_sha256=hashlib.sha256(b"{}").hexdigest(),
        response_size_bytes=2,
        cache_hit=True,
        requested_at="2026-09-03T00:00:00+00:00",
    )


def _frame(
    *,
    symbol: str,
    asset_type: str,
    provider: str,
    market: str,
    dates: pd.DatetimeIndex,
    dataset_profile: str,
    base_price: float,
) -> pd.DataFrame:
    index = np.arange(len(dates), dtype=np.float64)
    close = base_price * np.exp(index * 0.0005)
    frame = pd.DataFrame(
        {
            "timestamp": dates,
            "symbol": symbol,
            "asset_type": asset_type,
            "open": close * 0.999,
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
            "volume": np.full(len(dates), 1_000_000.0),
            "adjusted_close": close,
            "split_adjusted_volume": np.full(len(dates), 1_000_000.0),
            "adjustment_source": (
                "official_total_return_index" if asset_type == "index" else "fixture"
            ),
            "provider": provider,
            "market": market,
            "currency": "USD" if market == "US" else "TWD",
            "source_symbol": symbol.split(".", maxsplit=1)[0],
            "is_active": True,
            "dataset_profile": dataset_profile,
        }
    )
    return normalize_ohlcv_frame(frame)


class _FixtureEODHDProvider(EODHDProvider):
    calls: ClassVar[list[str]] = []

    def discover(
        self,
        *,
        include_delisted: bool,
    ) -> tuple[list[Instrument], tuple[RequestRecord, ...]]:
        assert include_delisted is True
        self.calls.append("discover")
        instruments = [
            Instrument("AAPL.US", "AAPL.US", "stock", "US", "USD", True),
            Instrument("VTI.US", "VTI.US", "etf", "US", "USD", True),
        ]
        return instruments, (_request(self.name, "discover"),)

    def fetch_historical_split_events(
        self,
        instrument: Instrument,
        *,
        start: str,
        end: str,
    ) -> ProviderFetch:
        del start, end
        self.calls.append(f"splits:{instrument.canonical_symbol}")
        return ProviderFetch(
            frame=pd.DataFrame(),
            requests=(_request(self.name, f"splits:{instrument.canonical_symbol}"),),
            metadata={"dropped_rows": 0},
        )

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
        assert split_events is not None and split_events.empty
        assert split_adjustment_source == "historical_splits"
        self.calls.append(f"daily:{instrument.canonical_symbol}")
        dates = _DATES[pd.Timestamp(start, tz="UTC") <= _DATES]
        dates = dates[dates <= pd.Timestamp(end, tz="UTC")]
        return ProviderFetch(
            frame=_frame(
                symbol=instrument.canonical_symbol,
                asset_type=instrument.asset_type,
                provider=self.name,
                market="US",
                dates=dates,
                dataset_profile=dataset_profile,
                base_price=220.0 if instrument.asset_type == "etf" else 180.0,
            ),
            requests=(_request(self.name, f"daily:{instrument.canonical_symbol}"),),
            metadata={"dropped_rows": 0},
        )


class _FixtureTaiwanMixin:
    calls: ClassVar[list[str]]
    target_symbol: str
    benchmark_symbol: str
    base_price: float

    def fetch_actions(self, *, start: str, end: str) -> ProviderFetch:
        del start, end
        self.calls.append("actions")
        return ProviderFetch(
            frame=pd.DataFrame(),
            requests=(_request(self.name, "actions"),),
            metadata={"dropped_rows": 0},
        )

    def fetch_benchmark_month(
        self,
        *,
        month: str,
        dataset_profile: str,
    ) -> ProviderFetch:
        self.calls.append(f"benchmark:{month}")
        dates = pd.DatetimeIndex(
            [timestamp for timestamp in _DATES if timestamp.strftime("%Y-%m") == month]
        )
        return ProviderFetch(
            frame=_frame(
                symbol=self.benchmark_symbol,
                asset_type="index",
                provider=self.name,
                market=self.market,
                dates=dates,
                dataset_profile=dataset_profile,
                base_price=self.base_price * 100.0,
            ),
            requests=(_request(self.name, f"benchmark:{month}"),),
            metadata={"dropped_rows": 0},
        )

    def fetch_date(self, *, date: str, dataset_profile: str) -> ProviderFetch:
        self.calls.append(f"daily:{date}")
        dates = pd.DatetimeIndex([pd.Timestamp(date, tz="UTC")])
        return ProviderFetch(
            frame=_frame(
                symbol=self.target_symbol,
                asset_type="stock",
                provider=self.name,
                market=self.market,
                dates=dates,
                dataset_profile=dataset_profile,
                base_price=self.base_price,
            ),
            requests=(_request(self.name, f"daily:{date}"),),
            metadata={"dropped_rows": 0},
        )


class _FixtureTWSEProvider(_FixtureTaiwanMixin, TWSEProvider):
    calls: ClassVar[list[str]] = []
    target_symbol = "2330.TW"
    benchmark_symbol = "TAIEX.TW"
    base_price = 600.0


class _FixtureTPExProvider(_FixtureTaiwanMixin, TPExProvider):
    calls: ClassVar[list[str]] = []
    target_symbol = "6488.TWO"
    benchmark_symbol = "TPEX.TWO"
    base_price = 400.0


def _options(tmp_path: Path, attempt: str) -> IngestionOptions:
    root = tmp_path / attempt
    return IngestionOptions(
        profile="us_tw_eodhd",
        start="2023-01-02",
        end="2024-06-29",
        output=root / "raw" / "market.parquet",
        manifest_root=root,
        raw_cache_root=root / "api-cache",
        provider_checkpoint_root=tmp_path / "provider-checkpoints",
        dataset_request_sha256="a" * 64,
        workers=6,
    )


def test_three_provider_cold_path_reuses_checkpoints_and_builds_lazy_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ingestion, "EODHDProvider", _FixtureEODHDProvider)
    monkeypatch.setattr(ingestion, "TWSEProvider", _FixtureTWSEProvider)
    monkeypatch.setattr(ingestion, "TPExProvider", _FixtureTPExProvider)
    _FixtureEODHDProvider.calls.clear()
    _FixtureTWSEProvider.calls.clear()
    _FixtureTPExProvider.calls.clear()

    first_options = _options(tmp_path, "first")
    first = ingest_daily_ohlcv(first_options, eodhd_api_token="fixture")
    validate_download_manifest(
        first_options.manifest_root / "download-manifest.json",
        input_path=first_options.output,
    )
    calls_after_cold_path = {
        "eodhd": list(_FixtureEODHDProvider.calls),
        "twse": list(_FixtureTWSEProvider.calls),
        "tpex": list(_FixtureTPExProvider.calls),
    }
    assert all(calls_after_cold_path.values())
    assert set(first["providers"]) == {"eodhd", "tpex_official", "twse_official"}

    second_options = _options(tmp_path, "second")
    second = ingest_daily_ohlcv(second_options, eodhd_api_token="fixture")
    assert calls_after_cold_path == {
        "eodhd": _FixtureEODHDProvider.calls,
        "twse": _FixtureTWSEProvider.calls,
        "tpex": _FixtureTPExProvider.calls,
    }
    assert all(
        checkpoint["reused"] is True
        for checkpoint in second["api_policy"][
            "provider_materialization_checkpoints"
        ].values()
    )
    pd.testing.assert_frame_equal(
        pd.read_parquet(first_options.output),
        pd.read_parquet(second_options.output),
    )

    store = second_options.manifest_root / "prepared" / "bar-store"
    dataset_manifest_path = second_options.manifest_root / "dataset-manifest.json"
    assert prepare_data_main(
        [
            "--input",
            str(second_options.output),
            "--output",
            str(store),
            "--download-manifest",
            str(second_options.manifest_root / "download-manifest.json"),
            "--dataset-manifest",
            str(dataset_manifest_path),
            "--window-size",
            "32",
            "--bucket-count",
            "4",
            "--batch-rows",
            "100",
            "--workers",
            "1",
        ]
    ) == 0
    prepared = validate_training_dataset_manifest(
        dataset_manifest_path,
        profile="us_tw_eodhd",
        raw_path=second_options.output,
        bar_store_path=store,
    )
    assert (store / "_SUCCESS.json").is_file()
    assert all(count > 0 for count in prepared["split_counts"].values())
    assert prepared["execution"]["window_materialized"] is False
    assert prepared["execution"]["labels_materialized"] is False
    assert not list(store.rglob("*window*.parquet"))
    assert not list(store.rglob("*label*.parquet"))


def test_changed_end_date_runs_a_complete_new_provider_materialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ingestion, "EODHDProvider", _FixtureEODHDProvider)
    monkeypatch.setattr(ingestion, "TWSEProvider", _FixtureTWSEProvider)
    monkeypatch.setattr(ingestion, "TPExProvider", _FixtureTPExProvider)
    _FixtureEODHDProvider.calls.clear()
    _FixtureTWSEProvider.calls.clear()
    _FixtureTPExProvider.calls.clear()

    baseline_options = _options(tmp_path, "baseline")
    baseline = ingest_daily_ohlcv(baseline_options, eodhd_api_token="fixture")
    baseline_call_counts = {
        "eodhd": len(_FixtureEODHDProvider.calls),
        "twse": len(_FixtureTWSEProvider.calls),
        "tpex": len(_FixtureTPExProvider.calls),
    }

    changed_options = replace(
        _options(tmp_path, "changed-end"),
        end="2024-07-06",
        dataset_request_sha256="b" * 64,
    )
    result = ingest_daily_ohlcv(changed_options, eodhd_api_token="fixture")
    validate_download_manifest(
        changed_options.manifest_root / "download-manifest.json",
        input_path=changed_options.output,
    )
    store = changed_options.manifest_root / "prepared" / "bar-store"
    dataset_manifest_path = changed_options.manifest_root / "dataset-manifest.json"
    assert prepare_data_main(
        [
            "--input",
            str(changed_options.output),
            "--output",
            str(store),
            "--download-manifest",
            str(changed_options.manifest_root / "download-manifest.json"),
            "--dataset-manifest",
            str(dataset_manifest_path),
            "--window-size",
            "32",
            "--bucket-count",
            "4",
            "--batch-rows",
            "100",
            "--workers",
            "1",
        ]
    ) == 0
    prepared = load_dataset_manifest(
        dataset_manifest_path,
        required_state="ready",
    )

    assert result["date_range"]["end_exclusive"] == "2024-07-06"
    assert result["artifacts"]["raw"]["row_count"] > baseline["artifacts"]["raw"][
        "row_count"
    ]
    assert (store / "_SUCCESS.json").is_file()
    assert all(count > 0 for count in prepared["split_counts"].values())
    assert len(_FixtureEODHDProvider.calls) > baseline_call_counts["eodhd"]
    assert len(_FixtureTWSEProvider.calls) > baseline_call_counts["twse"]
    assert len(_FixtureTPExProvider.calls) > baseline_call_counts["tpex"]
    assert all(
        checkpoint["reused"] is False
        for checkpoint in result["api_policy"][
            "provider_materialization_checkpoints"
        ].values()
    )
