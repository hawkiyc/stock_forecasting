"""Provider parsing, cache, rate-boundary, and extension-point tests."""

from __future__ import annotations

import json
import threading
import traceback
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import requests

from stock_forecasting.cli import download_market_data
from stock_forecasting.data.adjustments import asof_adjusted_window
from stock_forecasting.data.download_progress import DownloadProgress
from stock_forecasting.data.ingestion import (
    _benchmark_trading_dates,
    _ensure_us_benchmark,
    _explicit_instruments,
    _inclusive_end,
    _limit_instruments,
    _run_parallel_provider_loops,
)
from stock_forecasting.data.providers.base import Instrument, RequestRecord
from stock_forecasting.data.providers.eodhd import (
    EODHD_DEFAULT_DAILY_API_CALL_LIMIT,
    EODHD_DEFAULT_REQUESTS_PER_MINUTE,
    EODHD_DEFAULT_REQUESTS_PER_SECOND,
    EODHDProvider,
)
from stock_forecasting.data.providers.http import (
    AcquisitionDeadlineExceeded,
    CachedJsonClient,
    NetworkRequestBudget,
    NetworkRequestBudgetExceeded,
    ProviderAcquisitionError,
    ProviderRequestError,
)
from stock_forecasting.data.providers.massive import MassiveProvider
from stock_forecasting.data.providers.taiwan import TWSEProvider


class _FakeSession:
    def __init__(self, payload: Any) -> None:
        self.payload = payload
        self.calls: list[dict[str, Any]] = []

    def get(self, endpoint: str, **kwargs: Any) -> requests.Response:
        self.calls.append({"endpoint": endpoint, **kwargs})
        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps(self.payload).encode("utf-8")
        response.headers = {}
        return response


def _request(provider: str = "fixture") -> RequestRecord:
    return RequestRecord(
        provider=provider,
        request_sha256="a" * 64,
        cache_relative_path="aa/fixture.json",
        response_sha256="b" * 64,
        response_size_bytes=2,
        cache_hit=False,
        requested_at="2026-01-01T00:00:00+00:00",
    )


class _StubClient:
    def __init__(self, payload: Any) -> None:
        self.payload = payload
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def get_json(
        self,
        endpoint: str,
        *,
        params: dict[str, Any],
    ) -> tuple[Any, RequestRecord]:
        self.calls.append((endpoint, params))
        return self.payload, _request()


def test_raw_cache_identity_omits_api_token_and_reuses_response(tmp_path: Path) -> None:
    session = _FakeSession([{"date": "2026-01-02", "close": 100.0}])
    client = CachedJsonClient(
        provider="eodhd",
        raw_cache_root=tmp_path,
        max_requests_per_second=10.0,
        session=session,
        clock=lambda: 1.0,
        sleeper=lambda _seconds: None,
    )

    first_payload, first = client.get_json(
        "https://example.invalid/eod/AAPL.US",
        params={"api_token": "first-secret", "from": "2026-01-01"},
    )
    second_payload, second = client.get_json(
        "https://example.invalid/eod/AAPL.US",
        params={"api_token": "rotated-secret", "from": "2026-01-01"},
    )

    assert first_payload == second_payload
    assert len(session.calls) == 1
    assert first.request_sha256 == second.request_sha256
    assert not first.cache_hit
    assert second.cache_hit
    assert "secret" not in first.cache_relative_path
    assert "first-secret" not in next(tmp_path.rglob("*.json")).read_text(encoding="utf-8")


def test_shared_network_budget_counts_retries_and_fails_closed(tmp_path: Path) -> None:
    class _RetrySession:
        def get(self, _endpoint: str, **_kwargs: Any) -> requests.Response:
            response = requests.Response()
            response.status_code = 500
            response._content = b"{}"
            response.headers = {}
            return response

    budget = NetworkRequestBudget(
        max_network_requests=2,
        limited_providers={"eodhd"},
    )
    client = CachedJsonClient(
        provider="eodhd",
        raw_cache_root=tmp_path,
        max_requests_per_second=10.0,
        max_attempts=3,
        session=_RetrySession(),
        request_budget=budget,
        clock=lambda: 1.0,
        sleeper=lambda _seconds: None,
    )

    with pytest.raises(NetworkRequestBudgetExceeded, match="budget exhausted"):
        client.get_json("https://example.invalid/eod/AAPL.US", params={})
    assert budget.network_requests == 2
    assert budget.limited_network_requests == 2
    assert budget.provider_counts == {"eodhd": 2}


def test_eodhd_ceiling_never_limits_twse_or_tpex_requests() -> None:
    budget = NetworkRequestBudget(
        max_network_requests=1,
        limited_providers={"eodhd"},
    )

    for _ in range(3):
        budget.consume("twse_official")
        budget.consume("tpex_official")
    budget.consume("eodhd")

    with pytest.raises(NetworkRequestBudgetExceeded) as captured:
        budget.consume("eodhd")
    assert captured.value.provider == "eodhd"
    assert budget.network_requests == 7
    assert budget.limited_network_requests == 1
    assert budget.provider_counts == {
        "eodhd": 1,
        "tpex_official": 3,
        "twse_official": 3,
    }


def test_parallel_provider_loops_wait_for_every_provider_after_eodhd_exits() -> None:
    started = {
        "twse_official": threading.Event(),
        "tpex_official": threading.Event(),
    }
    completed: list[str] = []

    def eodhd() -> None:
        assert started["twse_official"].wait(timeout=5.0)
        assert started["tpex_official"].wait(timeout=5.0)
        raise NetworkRequestBudgetExceeded(consumed=1, maximum=1, provider="eodhd")

    def taiwan(provider: str) -> str:
        started[provider].set()
        completed.append(provider)
        return provider

    outcomes = _run_parallel_provider_loops(
        {
            "eodhd": eodhd,
            "twse_official": lambda: taiwan("twse_official"),
            "tpex_official": lambda: taiwan("tpex_official"),
        }
    )

    assert set(completed) == {"twse_official", "tpex_official"}
    assert isinstance(outcomes["eodhd"].error, NetworkRequestBudgetExceeded)
    assert outcomes["twse_official"].result == "twse_official"
    assert outcomes["tpex_official"].result == "tpex_official"


def test_acquisition_deadline_stops_work_before_the_preparation_reserve() -> None:
    budget = NetworkRequestBudget(
        max_network_requests=10,
        deadline_epoch_seconds=100.0,
        wall_clock=lambda: 100.0,
    )

    with pytest.raises(AcquisitionDeadlineExceeded, match="preparation"):
        budget.check_time()
    assert budget.network_requests == 0


def test_non_retryable_http_error_consumes_only_one_attempt(tmp_path: Path) -> None:
    class _UnauthorizedSession:
        def __init__(self) -> None:
            self.calls = 0

        def get(self, _endpoint: str, **_kwargs: Any) -> requests.Response:
            self.calls += 1
            response = requests.Response()
            response.status_code = 401
            response._content = b'{"error":"unauthorized"}'
            response.headers = {}
            return response

    session = _UnauthorizedSession()
    budget = NetworkRequestBudget(max_network_requests=10)
    client = CachedJsonClient(
        provider="eodhd",
        raw_cache_root=tmp_path,
        max_requests_per_second=10.0,
        max_attempts=3,
        session=session,
        request_budget=budget,
        clock=lambda: 1.0,
        sleeper=lambda _seconds: None,
    )

    with pytest.raises(ProviderRequestError, match="after 1 attempt") as captured:
        client.get_json("https://example.invalid/eod/AAPL.US", params={})
    assert captured.value.category == "provider_rejected"
    assert captured.value.status_code == 401
    assert not captured.value.retryable
    assert session.calls == 1
    assert budget.network_requests == 1


def test_rate_limit_failure_exposes_safe_resume_metadata(tmp_path: Path) -> None:
    class _RateLimitedSession:
        def get(self, _endpoint: str, **_kwargs: Any) -> requests.Response:
            response = requests.Response()
            response.status_code = 429
            response._content = b'{"error":"rate limited"}'
            response.headers = {
                "Retry-After": "86400",
                "X-RateLimit-Limit": "1000",
                "X-RateLimit-Remaining": "0",
            }
            return response

    client = CachedJsonClient(
        provider="eodhd",
        raw_cache_root=tmp_path,
        max_requests_per_second=10.0,
        max_attempts=1,
        session=_RateLimitedSession(),
        clock=lambda: 1.0,
        sleeper=lambda _seconds: None,
    )

    with pytest.raises(ProviderRequestError) as captured:
        client.get_json("https://example.invalid/eod/AAPL.US", params={})

    assert captured.value.metadata() == {
        "category": "rate_limited",
        "provider": "eodhd",
        "attempts": 1,
        "retryable": True,
        "status_code": 429,
        "retry_after_seconds": 86400.0,
        "rate_limit_limit": 1000,
        "rate_limit_remaining": 0,
        "backoff": {
            "wait_count": 0,
            "total_wait_seconds": 0.0,
            "proposed_wait_seconds": 86400.0,
            "exit_reason": "maximum_attempts_reached",
        },
    }


def test_taiwan_retry_loop_exits_when_next_backoff_exceeds_maximum(
    tmp_path: Path,
) -> None:
    class _UnavailableSession:
        def __init__(self) -> None:
            self.calls = 0

        def get(self, _endpoint: str, **_kwargs: Any) -> requests.Response:
            self.calls += 1
            response = requests.Response()
            response.status_code = 503
            response._content = b"{}"
            response.headers = {}
            return response

    session = _UnavailableSession()
    waits: list[float] = []
    budget = NetworkRequestBudget(1, limited_providers={"eodhd"})
    client = CachedJsonClient(
        provider="twse_official",
        raw_cache_root=tmp_path,
        max_requests_per_second=10.0,
        max_backoff_seconds=2.0,
        session=session,
        request_budget=budget,
        clock=lambda: float(session.calls * 10),
        sleeper=waits.append,
    )

    with pytest.raises(ProviderRequestError) as captured:
        client.get_json("https://example.invalid/twse", params={})

    assert session.calls == 3
    assert waits == [1.0, 2.0]
    assert budget.network_requests == 3
    assert budget.limited_network_requests == 0
    assert captured.value.metadata()["backoff"] == {
        "wait_count": 2,
        "total_wait_seconds": 3.0,
        "last_wait_seconds": 2.0,
        "proposed_wait_seconds": 4.0,
        "maximum_seconds": 2.0,
        "exit_reason": "proposed_backoff_exceeds_maximum",
    }


def test_taiwan_403_uses_bounded_backoff_and_browser_headers(tmp_path: Path) -> None:
    class _ForbiddenSession:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def get(self, _endpoint: str, **kwargs: Any) -> requests.Response:
            self.calls.append(kwargs)
            response = requests.Response()
            response.status_code = 403
            response._content = b"{}"
            response.headers = {}
            return response

    session = _ForbiddenSession()
    waits: list[float] = []
    client = CachedJsonClient(
        provider="tpex_official",
        raw_cache_root=tmp_path,
        max_requests_per_second=10.0,
        max_backoff_seconds=2.0,
        headers={
            "User-Agent": "Mozilla/5.0 fixture",
            "Referer": "https://www.tpex.org.tw/",
        },
        retryable_status_codes={403, 429},
        session=session,
        clock=lambda: float(len(session.calls) * 10),
        sleeper=waits.append,
    )

    with pytest.raises(ProviderRequestError) as captured:
        client.get_json("https://example.invalid/tpex", params={})

    assert len(session.calls) == 3
    assert waits == [1.0, 2.0]
    assert session.calls[0]["headers"]["User-Agent"] == "Mozilla/5.0 fixture"
    assert session.calls[0]["headers"]["Referer"] == "https://www.tpex.org.tw/"
    assert captured.value.category == "access_temporarily_denied"
    assert captured.value.retryable
    assert captured.value.status_code == 403


def test_new_client_reuses_successful_cache_after_rate_limit_interruption(
    tmp_path: Path,
) -> None:
    first_session = _FakeSession([{"date": "2026-01-02", "close": 100.0}])
    first_client = CachedJsonClient(
        provider="eodhd",
        raw_cache_root=tmp_path,
        max_requests_per_second=10.0,
        session=first_session,
        clock=lambda: 1.0,
        sleeper=lambda _seconds: None,
    )
    first_client.get_json("https://example.invalid/eod/AAPL.US", params={})

    class _RateLimitedSession:
        def get(self, _endpoint: str, **_kwargs: Any) -> requests.Response:
            response = requests.Response()
            response.status_code = 429
            response._content = b"{}"
            response.headers = {}
            return response

    interrupted_client = CachedJsonClient(
        provider="eodhd",
        raw_cache_root=tmp_path,
        max_requests_per_second=10.0,
        max_attempts=1,
        session=_RateLimitedSession(),
        clock=lambda: 1.0,
        sleeper=lambda _seconds: None,
    )
    with pytest.raises(ProviderRequestError, match="rate_limited"):
        interrupted_client.get_json("https://example.invalid/eod/MSFT.US", params={})

    resumed_session = _FakeSession([{"date": "2026-01-02", "close": 200.0}])
    resumed_client = CachedJsonClient(
        provider="eodhd",
        raw_cache_root=tmp_path,
        max_requests_per_second=10.0,
        session=resumed_session,
        clock=lambda: 1.0,
        sleeper=lambda _seconds: None,
    )
    _, cached = resumed_client.get_json(
        "https://example.invalid/eod/AAPL.US",
        params={},
    )
    _, downloaded = resumed_client.get_json(
        "https://example.invalid/eod/MSFT.US",
        params={},
    )

    assert cached.cache_hit
    assert not downloaded.cache_hit
    assert len(resumed_session.calls) == 1


def test_http_failure_traceback_never_exposes_api_token(tmp_path: Path) -> None:
    secret = "must-not-appear"

    class _UnauthorizedSession:
        def get(self, endpoint: str, **kwargs: Any) -> requests.Response:
            request = requests.Request("GET", endpoint, params=kwargs["params"]).prepare()
            response = requests.Response()
            response.status_code = 401
            response._content = b'{"error":"unauthorized"}'
            response.headers = {}
            response.request = request
            response.url = request.url
            return response

    client = CachedJsonClient(
        provider="eodhd",
        raw_cache_root=tmp_path,
        max_requests_per_second=10.0,
        session=_UnauthorizedSession(),
        sleeper=lambda _seconds: None,
    )

    with pytest.raises(RuntimeError) as captured:
        client.get_json(
            "https://example.invalid/eod/AAPL.US",
            params={"api_token": secret},
        )

    rendered = "".join(
        traceback.format_exception(
            type(captured.value),
            captured.value,
            captured.value.__traceback__,
        )
    )
    assert secret not in rendered


def test_eodhd_preserves_total_return_anchor_and_split_adjusted_volume() -> None:
    client = _StubClient(
        [
            {
                "date": "2026-01-02",
                "open": 100.0,
                "high": 103.0,
                "low": 99.0,
                "close": 102.0,
                "adjusted_close": 51.0,
                "volume": 1000,
            }
        ]
    )
    provider = EODHDProvider(client, api_token="fixture-token")
    instrument = Instrument("AAPL.US", "AAPL.US", "stock", "US", "USD", True)

    fetched = provider.fetch_instrument(
        instrument,
        start="2026-01-01",
        end="2026-01-31",
        dataset_profile="us_only_eodhd",
    )

    assert list(fetched.frame["symbol"]) == ["AAPL.US"]
    assert list(fetched.frame["provider"]) == ["eodhd"]
    assert list(fetched.frame["dataset_profile"]) == ["us_only_eodhd"]
    assert fetched.frame.loc[0, "adjusted_close"] == pytest.approx(51.0)
    assert fetched.frame.loc[0, "split_adjusted_volume"] == pytest.approx(1000.0)
    assert fetched.frame.loc[0, "adjustment_source"] == (
        "eodhd_adjusted_close+historical_splits+reconstructed_raw_volume"
    )


def test_eodhd_drops_zero_vendor_rows_without_synthesizing_prices() -> None:
    client = _StubClient(
        [
            {
                "date": "2023-11-06",
                "open": 0,
                "high": 0,
                "low": 0,
                "close": 0,
                "adjusted_close": 0,
                "volume": 0,
            },
            {
                "date": "2023-11-20",
                "open": 9.95,
                "high": 10.1,
                "low": 9.9,
                "close": 10.0,
                "adjusted_close": 10.0,
                "volume": 100,
            },
        ]
    )
    instrument = Instrument("PLTL.US", "PLTL.US", "etf", "US", "USD", True)

    fetched = EODHDProvider(client, api_token="fixture-token").fetch_instrument(
        instrument,
        start="2005-01-01",
        end="2026-04-30",
        dataset_profile="us_only_eodhd",
    )

    assert list(fetched.frame["timestamp"].dt.strftime("%Y-%m-%d")) == ["2023-11-20"]
    assert fetched.metadata["source_rows"] == 2
    assert fetched.metadata["dropped_rows"] == 1
    assert fetched.metadata["invalid_row_counts"] == {"non_positive_price": 1}


def test_eodhd_does_not_apply_split_adjustment_to_volume_twice() -> None:
    client = _StubClient(
        [
            {
                "date": "2020-08-28",
                "open": 500.0,
                "high": 505.0,
                "low": 495.0,
                "close": 500.0,
                "adjusted_close": 125.0,
                "volume": 400.0,
            },
            {
                "date": "2020-08-31",
                "open": 125.0,
                "high": 127.0,
                "low": 124.0,
                "close": 125.0,
                "adjusted_close": 125.0,
                "volume": 200.0,
            },
        ]
    )
    split_events = pd.DataFrame(
        [
            {
                "timestamp": "2020-08-31",
                "symbol": "AAPL.US",
                "price_factor": 0.25,
                "share_multiplier": 4.0,
                "source": "eodhd_historical_splits",
            }
        ]
    )
    instrument = Instrument("AAPL.US", "AAPL.US", "stock", "US", "USD", True)

    fetched = EODHDProvider(client, api_token="fixture-token").fetch_instrument(
        instrument,
        start="2020-08-28",
        end="2020-08-31",
        dataset_profile="us_only_eodhd",
        split_events=split_events,
    )

    assert list(fetched.frame["volume"]) == pytest.approx([100.0, 200.0])
    assert list(fetched.frame["split_adjusted_volume"]) == pytest.approx([400.0, 200.0])
    causal = asof_adjusted_window(fetched.frame)
    assert list(causal["volume"]) == pytest.approx([400.0, 200.0])


def test_eodhd_future_split_reconstruction_cancels_at_the_sample_cutoff() -> None:
    client = _StubClient(
        [
            {
                "date": "2020-01-02",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "adjusted_close": 50.0,
                "volume": 200.0,
            }
        ]
    )
    future_split = pd.DataFrame(
        [
            {
                "timestamp": "2021-01-04",
                "symbol": "AAPL.US",
                "price_factor": 0.5,
                "share_multiplier": 2.0,
                "source": "eodhd_historical_splits",
            }
        ]
    )
    instrument = Instrument("AAPL.US", "AAPL.US", "stock", "US", "USD", True)

    fetched = EODHDProvider(client, api_token="fixture-token").fetch_instrument(
        instrument,
        start="2020-01-01",
        end="2020-12-31",
        dataset_profile="us_only_eodhd",
        split_events=future_split,
    )

    assert fetched.frame.loc[0, "volume"] == pytest.approx(100.0)
    assert fetched.frame.loc[0, "split_adjusted_volume"] == pytest.approx(200.0)
    causal = asof_adjusted_window(fetched.frame)
    assert causal.loc[0, "volume"] == pytest.approx(100.0)


def test_eodhd_end_boundary_is_converted_from_exclusive_to_inclusive() -> None:
    assert _inclusive_end("2026-04-30") == "2026-04-29"


def test_delisted_instrument_rows_keep_discovery_time_status() -> None:
    client = _StubClient(
        [
            {
                "date": "2010-01-04",
                "open": 10.0,
                "high": 11.0,
                "low": 9.0,
                "close": 10.5,
                "adjusted_close": 10.5,
                "volume": 1000,
            },
            {
                "date": "2020-06-30",
                "open": 20.0,
                "high": 21.0,
                "low": 19.0,
                "close": 20.5,
                "adjusted_close": 20.5,
                "volume": 2000,
            },
        ]
    )
    instrument = Instrument(
        "FORMER.US",
        "FORMER.US",
        "stock",
        "US",
        "USD",
        False,
    )

    fetched = EODHDProvider(client, api_token="fixture-token").fetch_instrument(
        instrument,
        start="2005-01-01",
        end="2026-04-29",
        dataset_profile="us_only_eodhd",
    )

    assert list(fetched.frame["timestamp"].dt.strftime("%Y-%m-%d")) == [
        "2010-01-04",
        "2020-06-30",
    ]
    assert list(fetched.frame["is_active"]) == [False, False]
    assert client.calls[0][1]["from"] == "2005-01-01"
    assert client.calls[0][1]["to"] == "2026-04-29"


def test_eodhd_historical_splits_parse_new_over_old_ratio() -> None:
    client = _StubClient(
        [
            {"date": "2014-06-09", "split": "7.000000/1.000000"},
            {"date": "2020-08-31", "split": "4:1"},
        ]
    )
    instrument = Instrument("AAPL.US", "AAPL.US", "stock", "US", "USD", True)

    fetched = EODHDProvider(
        client,
        api_token="fixture-token",
    ).fetch_historical_split_events(
        instrument,
        start="2010-01-01",
        end="2026-01-31",
    )

    assert client.calls[0][0].endswith("/splits/AAPL.US")
    assert fetched.metadata["coverage"] == "provider_full_historical_splits_response"
    assert list(fetched.frame["share_multiplier"]) == pytest.approx([7.0, 4.0])
    assert list(fetched.frame["price_factor"]) == pytest.approx([1.0 / 7.0, 0.25])
    assert "from" not in client.calls[0][1]
    assert "to" not in client.calls[0][1]


def test_twse_parser_includes_four_digit_etf_and_stock() -> None:
    payload = {
        "tables": [
            {
                "fields": [
                    "證券代號",
                    "開盤價",
                    "最高價",
                    "最低價",
                    "收盤價",
                    "成交股數",
                ],
                "data": [
                    ["0050", "100", "102", "99", "101", "1,000"],
                    ["2330", "900", "920", "895", "910", "2,000"],
                ],
            }
        ]
    }
    fetched = TWSEProvider(_StubClient(payload)).fetch_date(
        date="2026-01-02",
        dataset_profile="tw_only",
    )

    by_symbol = fetched.frame.set_index("symbol")
    assert by_symbol.loc["0050.TW", "asset_type"] == "etf"
    assert by_symbol.loc["2330.TW", "asset_type"] == "stock"
    assert set(by_symbol["provider"]) == {"twse_official"}
    assert by_symbol["is_active"].isna().all()


def test_twse_historical_action_without_detail_keeps_price_factor_and_records_gap() -> None:
    class _ActionClient:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def get_json(
            self,
            endpoint: str,
            *,
            params: dict[str, Any],
        ) -> tuple[Any, RequestRecord]:
            del params
            self.calls.append(endpoint)
            if endpoint.endswith("TWT49UDetail"):
                return {"stat": "無相關資料"}, _request("twse_official")
            return (
                {
                    "fields": [
                        "資料日期",
                        "股票代號",
                        "除權息前收盤價",
                        "除權息參考價",
                        "權/息",
                    ],
                    "data": [["94年01月11日", "6280", "33.00", "27.48", "權"]],
                },
                _request("twse_official"),
            )

    client = _ActionClient()
    fetched = TWSEProvider(client).fetch_actions(start="2005-01-01", end="2005-12-31")

    assert len(client.calls) == 2
    assert fetched.frame.loc[0, "price_factor"] == pytest.approx(27.48 / 33.0)
    assert fetched.frame.loc[0, "share_multiplier"] == pytest.approx(1.0)
    assert fetched.frame.loc[0, "source"] == "twse_twt49u_price_only_missing_detail"
    assert fetched.metadata["missing_share_multiplier_details"] == 1


def test_download_cli_renders_fatal_provider_outcomes_without_a_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    error = ProviderAcquisitionError(
        state="failed",
        retryable=False,
        provider_outcomes={
            "eodhd": {
                "state": "failed",
                "last_error": {
                    "category": "provider_data_contract_error",
                    "provider": "eodhd",
                    "operation": "daily_eod",
                    "item": "PLTL.US",
                    "error_type": "MarketDataValidationError",
                    "retryable": False,
                },
            }
        },
    )

    def fail_ingestion(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise error

    monkeypatch.setattr(download_market_data, "ingest_daily_ohlcv", fail_ingestion)
    exit_code = download_market_data.main(
        [
            "--profile",
            "tw_only",
            "--start",
            "2026-01-01",
            "--end",
            "2026-02-01",
            "--output",
            str(tmp_path / "raw" / "market.parquet"),
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.err)
    assert exit_code == 1
    assert payload["provider_outcomes"]["eodhd"]["last_error"]["item"] == "PLTL.US"
    assert "Traceback" not in captured.err


def test_explicit_etf_only_universe_does_not_require_stock_symbols() -> None:
    instruments = _explicit_instruments((), ("SPY", "QQQ.US"))
    assert [item.canonical_symbol for item in instruments] == ["QQQ.US", "SPY.US"]
    assert {item.asset_type for item in instruments} == {"etf"}


def test_taiwan_sessions_come_from_official_benchmark_rows() -> None:
    benchmark = pd.DataFrame(
        {"timestamp": pd.to_datetime(["2026-01-02", "2026-01-05", "2026-02-02"], utc=True)}
    )

    assert _benchmark_trading_dates(
        benchmark,
        start="2026-01-01",
        exclusive_end="2026-02-01",
    ) == {"2026-01-02", "2026-01-05"}


def test_symbol_limit_keeps_n_per_asset_type_before_benchmark_insertion() -> None:
    instruments = [
        Instrument("ZZZ.US", "ZZZ.US", "etf", "US", "USD", True),
        Instrument("AAA.US", "AAA.US", "etf", "US", "USD", False),
        Instrument("BBB.US", "BBB.US", "etf", "US", "USD", True),
        Instrument("MSFT.US", "MSFT.US", "stock", "US", "USD", True),
        Instrument("AAPL.US", "AAPL.US", "stock", "US", "USD", True),
        Instrument("AAA-S.US", "AAA-S.US", "stock", "US", "USD", False),
    ]

    limited = _limit_instruments(instruments, 2)
    with_benchmark = _ensure_us_benchmark(limited)

    assert [item.canonical_symbol for item in limited] == [
        "BBB.US",
        "ZZZ.US",
        "AAPL.US",
        "MSFT.US",
    ]
    assert [item.canonical_symbol for item in with_benchmark] == [
        "BBB.US",
        "ZZZ.US",
        "AAPL.US",
        "MSFT.US",
        "VTI.US",
    ]


def test_existing_vti_benchmark_is_not_added_twice() -> None:
    instruments = [
        Instrument("VTI.US", "VTI.US", "etf", "US", "USD", True),
        Instrument("AAPL.US", "AAPL.US", "stock", "US", "USD", True),
    ]

    assert _ensure_us_benchmark(instruments) == instruments


def test_eodhd_paid_plan_limits_define_pipeline_defaults() -> None:
    assert EODHD_DEFAULT_DAILY_API_CALL_LIMIT == 100000
    assert EODHD_DEFAULT_REQUESTS_PER_MINUTE == 1000
    assert pytest.approx(16.0) == EODHD_DEFAULT_REQUESTS_PER_SECOND
    assert pytest.approx(960.0) == EODHD_DEFAULT_REQUESTS_PER_SECOND * 60.0
    assert EODHD_DEFAULT_REQUESTS_PER_SECOND * 60.0 < EODHD_DEFAULT_REQUESTS_PER_MINUTE


def test_download_progress_persists_quota_state_and_attempt_number(tmp_path: Path) -> None:
    cache_root = tmp_path / "api-cache"
    cached_response = cache_root / "eodhd" / "aa" / f"{'a' * 64}.json"
    cached_response.parent.mkdir(parents=True)
    cached_response.write_text("[]", encoding="utf-8")
    identity = {
        "dataset_profile": "us_only_eodhd",
        "dataset_request_sha256": "b" * 64,
    }
    progress_path = tmp_path / "download-progress.json"
    budget = NetworkRequestBudget(100)
    progress = DownloadProgress(
        path=progress_path,
        raw_cache_root=cache_root,
        identity=identity,
    )
    progress.start(budget)
    state = progress.fail(
        ProviderRequestError(
            provider="eodhd",
            category="rate_limited",
            attempts=3,
            retryable=True,
            status_code=429,
        ),
        budget,
    )

    waiting_payload = json.loads(progress_path.read_text(encoding="utf-8"))
    assert state == "waiting_for_provider"
    assert waiting_payload["state"] == "waiting_for_provider"
    assert waiting_payload["cache"]["responses"] == 1
    assert waiting_payload["attempt"]["number"] == 1

    resumed = DownloadProgress(
        path=progress_path,
        raw_cache_root=cache_root,
        identity=identity,
    )
    resumed.start(NetworkRequestBudget(100))
    resumed_payload = json.loads(progress_path.read_text(encoding="utf-8"))
    assert resumed_payload["state"] == "acquiring"
    assert resumed_payload["attempt"]["number"] == 2


def test_download_progress_treats_request_ceiling_as_resumable(tmp_path: Path) -> None:
    progress_path = tmp_path / "download-progress.json"
    budget = NetworkRequestBudget(2)
    budget.consume("eodhd")
    budget.consume("eodhd")
    progress = DownloadProgress(
        path=progress_path,
        raw_cache_root=tmp_path / "api-cache",
        identity={"dataset_request_sha256": "c" * 64},
    )

    state = progress.fail(
        NetworkRequestBudgetExceeded(consumed=2, maximum=2),
        budget,
        estimated_http_requests=130742,
    )

    payload = json.loads(progress_path.read_text(encoding="utf-8"))
    assert state == "waiting_for_budget"
    assert payload["state"] == "waiting_for_budget"
    assert payload["last_error"]["retryable"] is True
    assert payload["plan"]["estimated_http_requests"] == 130742
    assert payload["attempt"]["limited_network_requests"] == 2


def test_download_progress_treats_acquisition_deadline_as_resumable(tmp_path: Path) -> None:
    progress_path = tmp_path / "download-progress.json"
    budget = NetworkRequestBudget(
        10,
        deadline_epoch_seconds=100.0,
        wall_clock=lambda: 99.0,
    )
    progress = DownloadProgress(
        path=progress_path,
        raw_cache_root=tmp_path / "api-cache",
        identity={"dataset_request_sha256": "d" * 64},
    )

    state = progress.fail(
        AcquisitionDeadlineExceeded(
            deadline_epoch_seconds=100.0,
            observed_epoch_seconds=100.5,
        ),
        budget,
    )

    payload = json.loads(progress_path.read_text(encoding="utf-8"))
    assert state == "waiting_for_resume"
    assert payload["state"] == "waiting_for_resume"
    assert payload["last_error"]["retryable"] is True
    assert payload["attempt"]["acquisition_deadline_epoch_seconds"] == 100.0


def test_download_progress_rejects_a_different_dataset_identity(tmp_path: Path) -> None:
    progress_path = tmp_path / "download-progress.json"
    first = DownloadProgress(
        path=progress_path,
        raw_cache_root=tmp_path / "api-cache",
        identity={"dataset_request_sha256": "a" * 64},
    )
    first.start(NetworkRequestBudget(10))

    with pytest.raises(ValueError, match="different dataset request"):
        DownloadProgress(
            path=progress_path,
            raw_cache_root=tmp_path / "api-cache",
            identity={"dataset_request_sha256": "b" * 64},
        )


def test_massive_channel_is_reserved_but_typed() -> None:
    with pytest.raises(NotImplementedError, match="license"):
        MassiveProvider().discover(include_delisted=True)
