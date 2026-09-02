"""Point-in-time benchmark assignment policy for the single-instrument model."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from stock_forecasting.data.schema import TRAINING_TARGET_ASSET_TYPES

US_BENCHMARK = "VTI.US"
TWSE_BENCHMARK = "TAIEX.TW"
TPEX_BENCHMARK = "TPEX.TWO"

# These are deliberately narrow audited allowlists. An explicit benchmark map
# may select a different benchmark for an allowlisted ETF, but it must never
# turn an unknown bond, commodity, inverse, leveraged, or volatility product
# into a training target.
US_DOMESTIC_EQUITY_ETFS = frozenset(
    {
        "DIA.US",
        "IWM.US",
        "IVV.US",
        "QQQ.US",
        "SCHB.US",
        "SCHX.US",
        "SPY.US",
        "VTI.US",
        "VOO.US",
        "VTV.US",
        "VUG.US",
        "XLB.US",
        "XLC.US",
        "XLE.US",
        "XLF.US",
        "XLI.US",
        "XLK.US",
        "XLP.US",
        "XLRE.US",
        "XLU.US",
        "XLV.US",
        "XLY.US",
    }
)
TWSE_DOMESTIC_EQUITY_ETFS = frozenset(
    {
        "0050.TW",
        "0051.TW",
        "0052.TW",
        "0053.TW",
        "0055.TW",
        "0057.TW",
        "006203.TW",
        "006204.TW",
        "006208.TW",
        "00692.TW",
        "00850.TW",
    }
)
TPEX_EXPOSURE_ETFS = frozenset({"006201.TW"})
_TWSE_COMMON_OR_TDR_SYMBOL = re.compile(r"^(?:[1-9][0-9]{3}|91[0-9]{4})[.]TW$")
_TPEX_COMMON_STOCK_SYMBOL = re.compile(r"^[1-9][0-9]{3}[.]TWO$")


def is_allowlisted_us_equity_etf(*, symbol: str) -> bool:
    """Return whether a US symbol belongs to the audited equity ETF set."""

    return symbol.strip().upper() in US_DOMESTIC_EQUITY_ETFS


def is_allowlisted_taiwan_equity_etf(*, symbol: str, market: str) -> bool:
    """Return whether a Taiwan symbol belongs to the market's audited ETF set."""

    canonical_symbol = symbol.strip().upper()
    canonical_market = str(market or "").strip().upper()
    if canonical_market == "TWSE":
        return canonical_symbol in (TWSE_DOMESTIC_EQUITY_ETFS | TPEX_EXPOSURE_ETFS)
    # The current single-benchmark policy has no audited TPEx-listed ETF target.
    return False


def is_allowlisted_unleveraged_equity_etf(*, symbol: str, market: str) -> bool:
    """Return whether an ETF belongs to the audited model-target allowlist."""

    canonical_symbol = symbol.strip().upper()
    canonical_market = _market(market)
    if canonical_market == "US":
        return is_allowlisted_us_equity_etf(symbol=canonical_symbol)
    return is_allowlisted_taiwan_equity_etf(
        symbol=canonical_symbol,
        market=canonical_market,
    )


def is_training_target_security(*, symbol: str, asset_type: str, market: str) -> bool:
    """Enforce the common-stock/DR/audited-equity-ETF target boundary."""

    canonical_symbol = symbol.strip().upper()
    canonical_type = asset_type.strip().lower()
    canonical_market = _market(market)
    if canonical_type == "etf":
        return is_allowlisted_unleveraged_equity_etf(
            symbol=canonical_symbol,
            market=canonical_market,
        )
    if canonical_type != "stock":
        return False
    if canonical_market == "US":
        # EODHD discovery supplies the common-stock/ADR type boundary.
        return canonical_symbol.endswith(".US") and len(canonical_symbol) > 3
    if canonical_market == "TWSE":
        return _TWSE_COMMON_OR_TDR_SYMBOL.fullmatch(canonical_symbol) is not None
    if canonical_market == "TPEX":
        return _TPEX_COMMON_STOCK_SYMBOL.fullmatch(canonical_symbol) is not None
    return False


@dataclass(frozen=True)
class BenchmarkDecision:
    benchmark_symbol: str | None
    eligible: bool
    reason: str
    policy: str


def _market(value: Any) -> str:
    normalized = str(value or "").strip().upper()
    aliases = {
        "NASDAQ": "US",
        "NYSE": "US",
        "NYSE ARCA": "US",
        "NYSEARCA": "US",
        "AMEX": "US",
    }
    return aliases.get(normalized, normalized)


def resolve_benchmark(
    *,
    symbol: str,
    asset_type: str,
    market: str,
    explicit_mapping: dict[str, str] | None = None,
) -> BenchmarkDecision:
    """Return one benchmark or a fail-closed exclusion reason."""

    canonical_symbol = symbol.strip().upper()
    canonical_type = asset_type.strip().lower()
    canonical_market = _market(market)
    mapping = {key.upper(): value.upper() for key, value in (explicit_mapping or {}).items()}
    if canonical_type == "index":
        return BenchmarkDecision(None, False, "benchmark_series_not_training_target", "index")
    if canonical_type not in TRAINING_TARGET_ASSET_TYPES:
        return BenchmarkDecision(None, False, "unsupported_asset_type", "mvp")
    if not is_training_target_security(
        symbol=canonical_symbol,
        asset_type=canonical_type,
        market=canonical_market,
    ):
        return BenchmarkDecision(
            None,
            False,
            (
                "etf_not_in_audited_unleveraged_equity_allowlist"
                if canonical_type == "etf"
                else "security_not_in_common_stock_or_depositary_receipt_scope"
            ),
            "audited_training_security_scope",
        )
    if canonical_symbol in mapping:
        benchmark = mapping[canonical_symbol]
        if benchmark == canonical_symbol:
            return BenchmarkDecision(None, False, "self_benchmark", "explicit")
        return BenchmarkDecision(benchmark, True, "explicit_mapping", "explicit")

    if canonical_market == "US":
        benchmark = US_BENCHMARK
        if canonical_symbol == benchmark:
            return BenchmarkDecision(None, False, "self_benchmark", "us_vti")
        return BenchmarkDecision(benchmark, True, "us_domestic_equity", "us_vti")

    if canonical_market == "TWSE":
        benchmark = TPEX_BENCHMARK if canonical_symbol in TPEX_EXPOSURE_ETFS else TWSE_BENCHMARK
        return BenchmarkDecision(benchmark, True, "taiwan_domestic_equity", "taiwan_market")

    if canonical_market == "TPEX":
        return BenchmarkDecision(TPEX_BENCHMARK, True, "tpex_common_stock", "taiwan_market")

    return BenchmarkDecision(None, False, "unmapped_market", "mvp")
