"""Generate deterministic synthetic US stock and ETF OHLCV data."""

from __future__ import annotations

import argparse
from pathlib import Path

from fin_ts_multimodal.data.schema import write_market_data
from fin_ts_multimodal.data.synthetic import DEFAULT_SYMBOLS, generate_synthetic_ohlcv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True, help="Destination .csv or .parquet file."
    )
    parser.add_argument(
        "--symbols", nargs="+", default=list(DEFAULT_SYMBOLS), help="Ticker symbols to generate."
    )
    parser.add_argument(
        "--etf-symbols",
        nargs="*",
        default=["SPY", "QQQ"],
        help="Symbols classified as ETFs.",
    )
    parser.add_argument(
        "--periods", type=int, default=600, help="Number of business-day bars per symbol."
    )
    parser.add_argument("--start", default="2018-01-02", help="First business date (YYYY-MM-DD).")
    parser.add_argument("--seed", type=int, default=42, help="Root random seed.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    etf_symbols = {symbol.upper() for symbol in args.etf_symbols}
    symbol_types = {
        symbol.upper(): "etf" if symbol.upper() in etf_symbols else "stock"
        for symbol in args.symbols
    }
    frame = generate_synthetic_ohlcv(
        symbols=symbol_types,
        periods=args.periods,
        start=args.start,
        seed=args.seed,
    )
    destination = write_market_data(frame, args.output)
    print(f"Wrote {len(frame)} rows for {len(symbol_types)} symbols to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
