"""Provider parsing, cache, rate-boundary, and extension-point tests."""

from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any

import pytest
import requests

from fin_ts_multimodal.data.ingestion import _explicit_instruments
from fin_ts_multimodal.data.providers.base import Instrument, RequestRecord
from fin_ts_multimodal.data.providers.eodhd import EODHDProvider
from fin_ts_multimodal.data.providers.http import CachedJsonClient, NetworkRequestBudget
from fin_ts_multimodal.data.providers.massive import MassiveProvider
from fin_ts_multimodal.data.providers.taiwan import TWSEProvider


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

    budget = NetworkRequestBudget(max_network_requests=2)
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

    with pytest.raises(RuntimeError, match="budget exhausted"):
        client.get_json("https://example.invalid/eod/AAPL.US", params={})
    assert budget.network_requests == 2
    assert budget.provider_counts == {"eodhd": 2}


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

    with pytest.raises(RuntimeError, match="after 1 attempt"):
        client.get_json("https://example.invalid/eod/AAPL.US", params={})
    assert session.calls == 1
    assert budget.network_requests == 1


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
        "eodhd_adjusted_close+calendar_splits"
    )


def test_eodhd_split_calendar_is_one_exchange_wide_typed_request() -> None:
    client = _StubClient(
        {
            "splits": [
                {
                    "code": "AAPL.US",
                    "split_date": "2026-01-15",
                    "old_shares": 1,
                    "new_shares": 4,
                },
                {
                    "code": "NOT-US.LSE",
                    "split_date": "2026-01-20",
                    "old_shares": 1,
                    "new_shares": 2,
                },
            ]
        }
    )
    fetched = EODHDProvider(client, api_token="fixture-token").fetch_split_events(
        start="2026-01-01",
        end="2026-01-31",
    )

    assert len(client.calls) == 1
    assert client.calls[0][0].endswith("/calendar/splits")
    assert fetched.metadata["split_events"] == 1
    row = fetched.frame.iloc[0]
    assert row["symbol"] == "AAPL.US"
    assert row["price_factor"] == pytest.approx(0.25)
    assert row["share_multiplier"] == pytest.approx(4.0)


def test_eodhd_split_calendar_rejects_pre_2015_silent_coverage_gap() -> None:
    provider = EODHDProvider(_StubClient({"splits": []}), api_token="fixture-token")

    with pytest.raises(ValueError, match="begins at 2015-01-01"):
        provider.fetch_split_events(start="2010-01-01", end="2014-12-31")


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
    assert fetched.metadata["coverage"] == "per_symbol_full_history"
    assert list(fetched.frame["share_multiplier"]) == pytest.approx([7.0, 4.0])
    assert list(fetched.frame["price_factor"]) == pytest.approx([1.0 / 7.0, 0.25])


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


def test_explicit_etf_only_universe_does_not_require_stock_symbols() -> None:
    instruments = _explicit_instruments((), ("SPY", "QQQ.US"))
    assert [item.canonical_symbol for item in instruments] == ["QQQ.US", "SPY.US"]
    assert {item.asset_type for item in instruments} == {"etf"}


def test_massive_channel_is_reserved_but_typed() -> None:
    with pytest.raises(NotImplementedError, match="license"):
        MassiveProvider().discover(include_delisted=True)
