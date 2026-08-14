"""Point-in-time benchmark assignment policy for the single-instrument model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

US_BENCHMARK = "VTI.US"
TWSE_BENCHMARK = "TAIEX.TW"
TPEX_BENCHMARK = "TPEX.TWO"

# These are deliberately narrow PoC allowlists. Unknown ETFs must be mapped
# explicitly rather than silently treating foreign, bond, commodity, inverse,
# leveraged, or volatility products as domestic long-only equity exposure.
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
    if canonical_symbol in mapping:
        benchmark = mapping[canonical_symbol]
        if benchmark == canonical_symbol:
            return BenchmarkDecision(None, False, "self_benchmark", "explicit")
        return BenchmarkDecision(benchmark, True, "explicit_mapping", "explicit")
    if canonical_type == "index":
        return BenchmarkDecision(None, False, "benchmark_series_not_training_target", "index")
    if canonical_type not in {"stock", "etf"}:
        return BenchmarkDecision(None, False, "unsupported_asset_type", "mvp")

    if canonical_market == "US":
        benchmark = US_BENCHMARK
        if canonical_symbol == benchmark:
            return BenchmarkDecision(None, False, "self_benchmark", "us_vti")
        if canonical_type == "etf" and canonical_symbol not in US_DOMESTIC_EQUITY_ETFS:
            return BenchmarkDecision(
                None,
                False,
                "etf_requires_explicit_economic_exposure_mapping",
                "us_vti",
            )
        return BenchmarkDecision(benchmark, True, "us_domestic_equity", "us_vti")

    if canonical_market == "TWSE":
        benchmark = TPEX_BENCHMARK if canonical_symbol in TPEX_EXPOSURE_ETFS else TWSE_BENCHMARK
        if canonical_type == "etf" and canonical_symbol not in (
            TWSE_DOMESTIC_EQUITY_ETFS | TPEX_EXPOSURE_ETFS
        ):
            return BenchmarkDecision(
                None,
                False,
                "etf_requires_explicit_economic_exposure_mapping",
                "taiwan_economic_exposure",
            )
        return BenchmarkDecision(benchmark, True, "taiwan_domestic_equity", "taiwan_market")

    if canonical_market == "TPEX":
        if canonical_type == "etf":
            return BenchmarkDecision(
                None,
                False,
                "tpex_etf_requires_explicit_economic_exposure_mapping",
                "taiwan_economic_exposure",
            )
        return BenchmarkDecision(TPEX_BENCHMARK, True, "tpex_common_stock", "taiwan_market")

    return BenchmarkDecision(None, False, "unmapped_market", "mvp")
