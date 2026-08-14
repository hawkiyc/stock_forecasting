"""Fetch daily OHLCV offline, cache raw responses, and write canonical Parquet."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from fin_ts_multimodal.data.ingestion import IngestionOptions, ingest_daily_ohlcv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=("tw_only", "us_only_eodhd", "us_tw_eodhd", "us_tw_massive"),
        default=os.environ.get("FIN_TS_DATASET_PROFILE", "us_tw_eodhd"),
    )
    parser.add_argument(
        "--symbols",
        nargs="*",
        default=[],
        help="Optional US verification subset; omit it to discover active and delisted symbols.",
    )
    parser.add_argument("--etf-symbols", nargs="*", default=[])
    parser.add_argument("--start", required=True, help="Inclusive date (YYYY-MM-DD).")
    parser.add_argument(
        "--end",
        required=True,
        help="Exclusive date (YYYY-MM-DD), retained for RunPod script compatibility.",
    )
    parser.add_argument(
        "--interval",
        default="1d",
        choices=("1d",),
        help="Only causal daily bars are supported.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path)
    parser.add_argument("--raw-cache-root", type=Path)
    parser.add_argument("--symbol-limit", type=int)
    parser.add_argument("--max-api-calls", type=int, default=90_000)
    parser.add_argument("--eodhd-qps", type=float, default=5.0)
    parser.add_argument("--taiwan-qps", type=float, default=0.5)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument(
        "--exclude-delisted",
        action="store_true",
        help="Exclude EODHD delisted symbols; active and delisted are included by default.",
    )
    return parser


def _default_manifest_root(output: Path) -> Path:
    return output.parent.parent if output.parent.name == "raw" else output.parent


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest_root = args.manifest_root or _default_manifest_root(args.output)
    raw_cache_root = args.raw_cache_root or manifest_root / "api-cache"
    options = IngestionOptions(
        profile=args.profile,
        start=args.start,
        end=args.end,
        output=args.output,
        manifest_root=manifest_root,
        raw_cache_root=raw_cache_root,
        include_delisted=not args.exclude_delisted,
        explicit_us_symbols=tuple(args.symbols),
        explicit_us_etfs=tuple(args.etf_symbols),
        symbol_limit=args.symbol_limit,
        max_api_calls=args.max_api_calls,
        eodhd_requests_per_second=args.eodhd_qps,
        taiwan_requests_per_second=args.taiwan_qps,
        max_attempts=args.max_attempts,
    )
    payload = ingest_daily_ohlcv(
        options,
        eodhd_api_token=os.environ.get("EODHD_API_TOKEN"),
    )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
