"""Official TWSE and TPEx market-wide daily quote providers."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from stock_forecasting.data.adjustments import apply_cumulative_adjustments
from stock_forecasting.data.benchmarks import is_allowlisted_taiwan_equity_etf
from stock_forecasting.data.schema import normalize_ohlcv_frame

from .base import Instrument, ProviderFetch, RequestRecord
from .http import CachedJsonClient

_ETF_CODE = re.compile(r"^00[0-9A-Z]{2,6}$")
_STOCK_CODE = re.compile(r"^[1-9][0-9]{3}$")
_DEPOSITARY_RECEIPT_CODE = re.compile(r"^91(?:[0-9]{2}|[0-9]{4})$")
_PREFERRED_STOCK_CODE = re.compile(r"^[1-9][0-9]{3}[A-E]$")
_RIGHTS_CERTIFICATE_CODE = re.compile(r"^[1-9][0-9]{3}[L-Z]$")
_SIX_DIGIT_NUMERIC_SECURITY_CODE = re.compile(r"^[0-9]{6}$")


def _clean_field(value: Any) -> str:
    return re.sub(r"\s+", "", str(value)).replace("\u3000", "")


def _number(value: Any) -> float | None:
    text = str(value).strip().replace(",", "")
    if text in {"", "--", "---", "N/A", "nan", "None"}:
        return None
    text = text.replace("+", "")
    try:
        return float(text)
    except ValueError:
        return None


def _gregorian_date(value: Any) -> str:
    """Normalize Gregorian or ROC dates returned by Taiwan exchanges."""

    text = str(value).strip().replace("年", "/").replace("月", "/").replace("日", "")
    compact = re.sub(r"[^0-9]", "", text)
    parts = [part for part in re.split(r"[^0-9]+", text) if part]
    if len(parts) == 3:
        year = int(parts[0]) + (1911 if len(parts[0]) <= 3 else 0)
        return f"{year:04d}-{int(parts[1]):02d}-{int(parts[2]):02d}"
    if len(compact) == 6:
        roc_year = int(compact[:2])
        if roc_year < 1:
            raise ValueError(f"Unsupported Taiwan date value: {value!r}")
        return f"{roc_year + 1911:04d}-{compact[2:4]}-{compact[4:6]}"
    if len(compact) == 7:
        return f"{int(compact[:3]) + 1911:04d}-{compact[3:5]}-{compact[5:7]}"
    if len(compact) == 8:
        return f"{compact[:4]}-{compact[4:6]}-{compact[6:8]}"
    raise ValueError(f"Unsupported Taiwan date value: {value!r}")


def _field_index(fields: list[Any], aliases: set[str]) -> int:
    normalized = [_clean_field(field) for field in fields]
    try:
        return next(index for index, field in enumerate(normalized) if field in aliases)
    except StopIteration as error:
        raise ValueError("Official Taiwan response is missing a required field") from error


def _first_table(payload: Any) -> tuple[list[Any], list[list[Any]]]:
    for table in _tables(payload):
        fields = table.get("fields")
        data = table.get("data")
        if isinstance(fields, list) and isinstance(data, list):
            return fields, [row for row in data if isinstance(row, list)]
    raise ValueError("Official Taiwan response has no data table")


def _is_no_data_payload(payload: Any) -> bool:
    """Recognize official empty-result responses without hiding schema drift."""

    if not isinstance(payload, dict):
        return False
    status = str(payload.get("stat", payload.get("status", "")))
    tables = payload.get("tables")
    data = payload.get("data")
    return (
        "沒有符合條件" in status
        or "無相關資料" in status
        or "很抱歉" in status
        or "no data" in status.lower()
        or tables == []
        or data == []
    )


def _index_frame(
    price_payload: Any,
    return_payload: Any,
    *,
    price_aliases: dict[str, set[str]],
    return_date_aliases: set[str],
    return_value_aliases: set[str],
    symbol: str,
    provider: str,
    market: str,
    source_symbol: str,
    dataset_profile: str,
) -> pd.DataFrame:
    price_fields, price_rows = _first_table(price_payload)
    price_indexes = {
        field: _field_index(price_fields, aliases) for field, aliases in price_aliases.items()
    }
    return_fields, return_rows = _first_table(return_payload)
    return_date_index = _field_index(return_fields, return_date_aliases)
    return_value_index = _field_index(return_fields, return_value_aliases)
    total_return_by_date = {
        _gregorian_date(row[return_date_index]): _number(row[return_value_index])
        for row in return_rows
        if len(row) > max(return_date_index, return_value_index)
    }
    rows: list[dict[str, Any]] = []
    for raw in price_rows:
        if len(raw) <= max(price_indexes.values()):
            continue
        trading_date = _gregorian_date(raw[price_indexes["timestamp"]])
        prices = {
            field: _number(raw[price_indexes[field]]) for field in ("open", "high", "low", "close")
        }
        total_return = total_return_by_date.get(trading_date)
        if total_return is None or any(value is None for value in prices.values()):
            continue
        rows.append(
            {
                "timestamp": trading_date,
                "symbol": symbol,
                "asset_type": "index",
                **prices,
                "volume": 0.0,
                "adjusted_close": total_return,
                "split_adjusted_volume": 0.0,
                "adjustment_source": "official_total_return_index",
                "provider": provider,
                "market": market,
                "currency": "TWD",
                "source_symbol": source_symbol,
                "is_active": True,
                "dataset_profile": dataset_profile,
            }
        )
    if not rows:
        return pd.DataFrame()
    return normalize_ohlcv_frame(pd.DataFrame(rows))


def _tables(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        raise ValueError("Taiwan official endpoint did not return a JSON object")
    tables = payload.get("tables")
    output = (
        [table for table in tables if isinstance(table, dict)] if isinstance(tables, list) else []
    )
    for key, fields in payload.items():
        if not key.startswith("fields") or not isinstance(fields, list):
            continue
        suffix = key.removeprefix("fields")
        data = payload.get(f"data{suffix}")
        if isinstance(data, list):
            output.append({"fields": fields, "data": data})
    return output


def _find_quote_table(
    payload: Any,
    aliases: dict[str, set[str]],
) -> tuple[list[str], list[list[Any]]]:
    required = set(aliases)
    for table in _tables(payload):
        fields = table.get("fields")
        data = table.get("data")
        if not isinstance(fields, list) or not isinstance(data, list):
            continue
        normalized = [_clean_field(field) for field in fields]
        mapped = {
            canonical: next(
                (index for index, field in enumerate(normalized) if field in names),
                None,
            )
            for canonical, names in aliases.items()
        }
        if set(key for key, index in mapped.items() if index is not None) != required:
            continue
        return normalized, [row for row in data if isinstance(row, list)]
    raise ValueError("Official Taiwan response has no compatible OHLCV quote table")


def _asset_type(code: str, *, market: str) -> str | None:
    if _DEPOSITARY_RECEIPT_CODE.fullmatch(code):
        return "stock"
    if _ETF_CODE.fullmatch(code):
        suffix = "TW" if market == "TWSE" else "TWO"
        return (
            "etf"
            if is_allowlisted_taiwan_equity_etf(
                symbol=f"{code}.{suffix}",
                market=market,
            )
            else None
        )
    if _STOCK_CODE.fullmatch(code):
        return "stock"
    return None


def _unsupported_action_type(code: str) -> str:
    if _ETF_CODE.fullmatch(code):
        return "non_allowlisted_etf_code"
    if _PREFERRED_STOCK_CODE.fullmatch(code):
        return "preferred_stock_code"
    if _RIGHTS_CERTIFICATE_CODE.fullmatch(code):
        return "rights_certificate_code"
    if _SIX_DIGIT_NUMERIC_SECURITY_CODE.fullmatch(code):
        return "six_digit_numeric_security_code"
    return "unsupported_symbol_code"


def _parse_quotes(
    payload: Any,
    *,
    aliases: dict[str, set[str]],
    date: str,
    provider: str,
    market: str,
    suffix: str,
    dataset_profile: str,
) -> tuple[pd.DataFrame, int]:
    fields, data = _find_quote_table(payload, aliases)
    indexes = {
        canonical: next(index for index, field in enumerate(fields) if field in names)
        for canonical, names in aliases.items()
    }
    rows: list[dict[str, Any]] = []
    dropped = 0
    for row in data:
        if len(row) <= max(indexes.values()):
            dropped += 1
            continue
        code = str(row[indexes["symbol"]]).strip().upper()
        asset_type = _asset_type(code, market=market)
        prices = {
            field: _number(row[indexes[field]])
            for field in ("open", "high", "low", "close")
        }
        volume = _number(row[indexes["volume"]])
        if (
            asset_type is None
            or volume is None
            or volume < 0.0
            or any(value is None or value <= 0.0 for value in prices.values())
        ):
            dropped += 1
            continue
        rows.append(
            {
                "timestamp": date,
                "symbol": f"{code}.{suffix}",
                "asset_type": asset_type,
                **prices,
                "volume": volume,
                "provider": provider,
                "market": market,
                "currency": "TWD",
                "source_symbol": code,
                # The daily quote endpoint does not prove current listing status.
                "is_active": pd.NA,
                "dataset_profile": dataset_profile,
            }
        )
    if not rows:
        return pd.DataFrame(), dropped
    return normalize_ohlcv_frame(pd.DataFrame(rows)), dropped


class _TaiwanMarketProvider:
    name: str
    market: str
    suffix: str
    endpoint: str
    action_adjustment_source: str
    aliases: dict[str, set[str]]

    def __init__(self, client: CachedJsonClient) -> None:
        self.client = client

    def discover(
        self,
        *,
        include_delisted: bool,
    ) -> tuple[list[Instrument], tuple[RequestRecord, ...]]:
        del include_delisted
        return [], ()

    def _params(self, date: str) -> dict[str, Any]:
        raise NotImplementedError

    def fetch_date(self, *, date: str, dataset_profile: str) -> ProviderFetch:
        payload, request = self.client.get_json(
            self.endpoint,
            params=self._params(date),
        )
        try:
            frame, dropped = _parse_quotes(
                payload,
                aliases=self.aliases,
                date=date,
                provider=self.name,
                market=self.market,
                suffix=self.suffix,
                dataset_profile=dataset_profile,
            )
        except ValueError:
            if _is_no_data_payload(payload):
                frame, dropped = pd.DataFrame(), 0
            else:
                raise
        return ProviderFetch(
            frame=frame,
            requests=(request,),
            metadata={"dropped_rows": dropped, "date": date},
        )

    def fetch_adjusted_date(
        self,
        *,
        date: str,
        dataset_profile: str,
        action_frame: pd.DataFrame,
    ) -> tuple[ProviderFetch, pd.DataFrame]:
        """Fetch one session and apply the provider's durable action factors."""

        fetched = self.fetch_date(
            date=date,
            dataset_profile=dataset_profile,
        )
        if fetched.frame.empty:
            return fetched, fetched.frame
        adjusted = apply_cumulative_adjustments(fetched.frame, action_frame)
        adjusted["adjustment_source"] = self.action_adjustment_source
        return fetched, adjusted

    def fetch_instrument(
        self,
        instrument: Instrument,
        *,
        start: str,
        end: str,
        dataset_profile: str,
    ) -> ProviderFetch:
        del instrument, start, end, dataset_profile
        raise NotImplementedError("Taiwan providers fetch market-wide data by trading date")


class TWSEProvider(_TaiwanMarketProvider):
    name = "twse_official"
    market = "TWSE"
    suffix = "TW"
    endpoint = "https://www.twse.com.tw/exchangeReport/MI_INDEX"
    action_adjustment_source = "twse_twt49u"
    aliases: ClassVar[dict[str, set[str]]] = {
        "symbol": {"證券代號"},
        "open": {"開盤價"},
        "high": {"最高價"},
        "low": {"最低價"},
        "close": {"收盤價"},
        "volume": {"成交股數"},
    }

    def _params(self, date: str) -> dict[str, Any]:
        return {
            "response": "json",
            "date": date.replace("-", ""),
            "type": "ALLBUT0999",
        }

    def fetch_actions(self, *, start: str, end: str) -> ProviderFetch:
        payload, request = self.client.get_json(
            "https://www.twse.com.tw/rwd/zh/exRight/TWT49U",
            params={
                "response": "json",
                "startDate": start.replace("-", ""),
                "endDate": end.replace("-", ""),
            },
        )
        fields, data = _first_table(payload)
        indexes = {
            "timestamp": _field_index(fields, {"資料日期"}),
            "symbol": _field_index(fields, {"股票代號"}),
            "prior_close": _field_index(fields, {"除權息前收盤價"}),
            "reference_price": _field_index(fields, {"除權息參考價"}),
            "kind": _field_index(fields, {"權/息"}),
        }
        rows: list[dict[str, Any]] = []
        requests: list[RequestRecord] = [request]
        dropped = 0
        missing_share_multiplier_details = 0
        skipped_unsupported_action_rows = 0
        skipped_unsupported_action_types: Counter[str] = Counter()
        for raw in data:
            if len(raw) <= max(indexes.values()):
                dropped += 1
                continue
            symbol_code = str(raw[indexes["symbol"]]).strip().upper()
            if _asset_type(symbol_code, market=self.market) is None:
                skipped_unsupported_action_rows += 1
                skipped_unsupported_action_types[_unsupported_action_type(symbol_code)] += 1
                continue
            prior_close = _number(raw[indexes["prior_close"]])
            reference_price = _number(raw[indexes["reference_price"]])
            if prior_close is None or reference_price is None or prior_close <= 0.0:
                dropped += 1
                continue
            effective_date = _gregorian_date(raw[indexes["timestamp"]])
            share_multiplier = 1.0
            action_source = "twse_twt49u"
            if "權" in str(raw[indexes["kind"]]):
                detail, detail_request = self.client.get_json(
                    "https://www.twse.com.tw/rwd/zh/exRight/TWT49UDetail",
                    params={
                        "response": "json",
                        "STK_NO": symbol_code,
                        "T1": effective_date.replace("-", ""),
                    },
                )
                requests.append(detail_request)
                if _is_no_data_payload(detail):
                    missing_share_multiplier_details += 1
                    action_source = "twse_twt49u_price_only_missing_detail"
                else:
                    detail_fields, detail_rows = _first_table(detail)
                    try:
                        free_share_index = _field_index(
                            detail_fields,
                            {"A.按普通股股東持股比例每千股無償配股"},
                        )
                    except ValueError:
                        free_share_index = None
                    if (
                        free_share_index is not None
                        and detail_rows
                        and len(detail_rows[0]) > free_share_index
                    ):
                        free_shares = _number(
                            re.sub(
                                r"[^0-9.+-]",
                                "",
                                str(detail_rows[0][free_share_index]),
                            )
                        )
                        if free_shares is not None and free_shares >= 0.0:
                            share_multiplier += free_shares / 1000.0
                        else:
                            missing_share_multiplier_details += 1
                            action_source = "twse_twt49u_price_only_missing_detail"
                    else:
                        missing_share_multiplier_details += 1
                        action_source = "twse_twt49u_price_only_missing_detail"
            price_factor = reference_price / prior_close
            if not np.isfinite(price_factor) or price_factor <= 0.0:
                dropped += 1
                continue
            rows.append(
                {
                    "timestamp": effective_date,
                    "symbol": f"{symbol_code}.TW",
                    "price_factor": price_factor,
                    "share_multiplier": share_multiplier,
                    "source": action_source,
                }
            )
        return ProviderFetch(
            frame=pd.DataFrame(rows),
            requests=tuple(requests),
            metadata={
                "corporate_actions": len(rows),
                "dropped_rows": dropped,
                "missing_share_multiplier_details": missing_share_multiplier_details,
                "skipped_unsupported_action_rows": skipped_unsupported_action_rows,
                "skipped_unsupported_action_types": dict(
                    sorted(skipped_unsupported_action_types.items())
                ),
            },
        )

    def fetch_benchmark_month(self, *, month: str, dataset_profile: str) -> ProviderFetch:
        date_value = month.replace("-", "")[:6] + "01"
        price_payload, price_request = self.client.get_json(
            "https://www.twse.com.tw/indicesReport/MI_5MINS_HIST",
            params={"response": "json", "date": date_value},
        )
        return_payload, return_request = self.client.get_json(
            "https://www.twse.com.tw/indicesReport/MFI94U",
            params={"response": "json", "date": date_value},
        )
        frame = _index_frame(
            price_payload,
            return_payload,
            price_aliases={
                "timestamp": {"日期", "日日期"},
                "open": {"開盤指數"},
                "high": {"最高指數"},
                "low": {"最低指數"},
                "close": {"收盤指數"},
            },
            return_date_aliases={"日期", "日日期"},
            return_value_aliases={"發行量加權股價報酬指數"},
            symbol="TAIEX.TW",
            provider=self.name,
            market=self.market,
            source_symbol="TAIEX",
            dataset_profile=dataset_profile,
        )
        return ProviderFetch(
            frame=frame,
            requests=(price_request, return_request),
            metadata={"month": month, "benchmark": "TAIEX.TW"},
        )


class TPExProvider(_TaiwanMarketProvider):
    name = "tpex_official"
    market = "TPEX"
    suffix = "TWO"
    endpoint = "https://www.tpex.org.tw/www/zh-tw/afterTrading/dailyQuotes"
    action_adjustment_source = "tpex_exdailyq"
    aliases: ClassVar[dict[str, set[str]]] = {
        "symbol": {"代號", "證券代號"},
        "open": {"開盤", "開盤價"},
        "high": {"最高", "最高價"},
        "low": {"最低", "最低價"},
        "close": {"收盤", "收盤價"},
        "volume": {"成交股數", "成交量"},
    }

    def _params(self, date: str) -> dict[str, Any]:
        return {
            "date": date.replace("-", "/"),
            "id": "",
            "response": "json",
        }

    def fetch_actions(self, *, start: str, end: str) -> ProviderFetch:
        payload, request = self.client.get_json(
            "https://www.tpex.org.tw/www/zh-tw/bulletin/exDailyQ",
            params={
                "startDate": start.replace("-", "/"),
                "endDate": end.replace("-", "/"),
                "response": "json",
            },
        )
        fields, data = _first_table(payload)
        indexes = {
            "timestamp": _field_index(fields, {"除權息日期"}),
            "symbol": _field_index(fields, {"代號"}),
            "prior_close": _field_index(fields, {"除權息前收盤價"}),
            "reference_price": _field_index(fields, {"除權息參考價"}),
            "free_shares": _field_index(fields, {"每仟股無償配股"}),
        }
        rows: list[dict[str, Any]] = []
        dropped = 0
        skipped_unsupported_action_rows = 0
        skipped_unsupported_action_types: Counter[str] = Counter()
        for raw in data:
            if len(raw) <= max(indexes.values()):
                dropped += 1
                continue
            symbol_code = str(raw[indexes["symbol"]]).strip().upper()
            if _asset_type(symbol_code, market=self.market) is None:
                skipped_unsupported_action_rows += 1
                skipped_unsupported_action_types[_unsupported_action_type(symbol_code)] += 1
                continue
            prior_close = _number(raw[indexes["prior_close"]])
            reference_price = _number(raw[indexes["reference_price"]])
            free_shares = _number(raw[indexes["free_shares"]])
            if (
                prior_close is None
                or reference_price is None
                or free_shares is None
                or prior_close <= 0.0
                or free_shares < 0.0
            ):
                dropped += 1
                continue
            price_factor = reference_price / prior_close
            if not np.isfinite(price_factor) or price_factor <= 0.0:
                dropped += 1
                continue
            rows.append(
                {
                    "timestamp": _gregorian_date(raw[indexes["timestamp"]]),
                    "symbol": f"{symbol_code}.TWO",
                    "price_factor": price_factor,
                    "share_multiplier": 1.0 + free_shares / 1000.0,
                    "source": "tpex_exdailyq",
                }
            )
        return ProviderFetch(
            frame=pd.DataFrame(rows),
            requests=(request,),
            metadata={
                "corporate_actions": len(rows),
                "dropped_rows": dropped,
                "skipped_unsupported_action_rows": skipped_unsupported_action_rows,
                "skipped_unsupported_action_types": dict(
                    sorted(skipped_unsupported_action_types.items())
                ),
            },
        )

    def fetch_benchmark_month(self, *, month: str, dataset_profile: str) -> ProviderFetch:
        query_date = month[:7].replace("-", "/") + "/01"
        price_payload, price_request = self.client.get_json(
            "https://www.tpex.org.tw/www/zh-tw/indexInfo/inx",
            params={"date": query_date, "response": "json"},
        )
        return_payload, return_request = self.client.get_json(
            "https://www.tpex.org.tw/www/zh-tw/indexInfo/ROE",
            params={"date": query_date, "response": "json"},
        )
        frame = _index_frame(
            price_payload,
            return_payload,
            price_aliases={
                "timestamp": {"日期"},
                "open": {"開市"},
                "high": {"最高"},
                "low": {"最低"},
                "close": {"收市"},
            },
            return_date_aliases={"日期"},
            return_value_aliases={"櫃買報酬指數(基期:94/12/30)"},
            symbol="TPEX.TWO",
            provider=self.name,
            market=self.market,
            source_symbol="TPEX",
            dataset_profile=dataset_profile,
        )
        return ProviderFetch(
            frame=frame,
            requests=(price_request, return_request),
            metadata={"month": month, "benchmark": "TPEX.TWO"},
        )
