"""Fetch daily OHLCV offline, cache raw responses, and write canonical Parquet."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from stock_forecasting.data.ingestion import IngestionOptions, ingest_daily_ohlcv
from stock_forecasting.data.providers import (
    EODHD_DEFAULT_DAILY_API_CALL_LIMIT,
    EODHD_DEFAULT_REQUESTS_PER_SECOND,
    ProviderRequestError,
)

TEMPORARY_PROVIDER_EXIT_CODE = 75


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
    parser.add_argument(
        "--progress-path",
        type=Path,
        help="Persistent progress JSON used to resume the same dataset request.",
    )
    parser.add_argument(
        "--dataset-request-sha256",
        default=os.environ.get("RUNPOD_DATASET_REQUEST_SHA256"),
    )
    parser.add_argument("--selection-id", default=os.environ.get("RUNPOD_SELECTION_ID"))
    parser.add_argument(
        "--selection-sha256",
        default=os.environ.get("RUNPOD_SELECTION_SHA256"),
    )
    parser.add_argument("--launch-id", default=os.environ.get("RUNPOD_LAUNCH_ID"))
    parser.add_argument(
        "--symbol-limit",
        type=int,
        help=(
            "Deterministic bounded US-discovery check: keep up to N ETFs and N "
            "stocks, then add the required VTI benchmark."
        ),
    )
    parser.add_argument(
        "--max-api-calls",
        type=int,
        default=EODHD_DEFAULT_DAILY_API_CALL_LIMIT,
        help=(
            "Project-side HTTP request safety ceiling; defaults to the official "
            "paid-plan daily API-call limit but remains distinct from provider usage."
        ),
    )
    parser.add_argument(
        "--eodhd-qps",
        type=float,
        default=EODHD_DEFAULT_REQUESTS_PER_SECOND,
        help=(
            "Even EODHD pacing in requests/second; the default floors the official "
            "1000 requests/minute limit to 16 requests/second (960/minute)."
        ),
    )
    parser.add_argument(
        "--taiwan-qps",
        type=float,
        default=0.5,
        help="Short-term TWSE/TPEx request pacing.",
    )
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("FIN_TS_CPU_WORKERS", "1")),
        help="Thread workers for independent symbols and trading dates.",
    )
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
        progress_path=args.progress_path,
        dataset_request_sha256=args.dataset_request_sha256,
        selection_id=args.selection_id,
        selection_sha256=args.selection_sha256,
        launch_id=args.launch_id,
        workers=args.workers,
    )
    try:
        payload = ingest_daily_ohlcv(
            options,
            eodhd_api_token=os.environ.get("EODHD_API_TOKEN"),
        )
    except ProviderRequestError as error:
        if not error.retryable:
            raise
        waiting = {
            "state": "waiting_for_provider",
            "exit_code": TEMPORARY_PROVIDER_EXIT_CODE,
            "progress_path": str(
                options.progress_path
                or options.manifest_root / "download-progress.json"
            ),
            "provider_error": error.metadata(),
        }
        print(json.dumps(waiting, ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return TEMPORARY_PROVIDER_EXIT_CODE
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
