"""Build purged quant-only windows from an offline canonical OHLCV Parquet."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from stock_forecasting.data.adjustments import robust_horizon_scales
from stock_forecasting.data.io import write_processed_records
from stock_forecasting.data.manifest import (
    DATASET_MANIFEST_SCHEMA_VERSION,
    artifact_metadata,
    atomic_write_json,
    canonical_json_sha256,
    sha256_file,
    validate_download_manifest,
)
from stock_forecasting.data.quality import assess_ohlcv_quality
from stock_forecasting.data.schema import TRAINING_SECURITY_SCOPE, read_market_data
from stock_forecasting.data.splits import SPLIT_POLICY, chronological_split
from stock_forecasting.data.windows import (
    DEFAULT_ALPHA_HORIZONS,
    PROCESSED_SCHEMA_VERSION,
    build_causal_windows,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Canonical raw Parquet.")
    parser.add_argument("--output", type=Path, required=True, help="Processed windows.parquet.")
    parser.add_argument("--download-manifest", type=Path)
    parser.add_argument("--dataset-manifest", type=Path)
    parser.add_argument("--benchmark-mapping", type=Path)
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--stride", type=int, default=5, help="RunPod compatibility sentinel.")
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument(
        "--alpha-horizons",
        type=int,
        nargs="+",
        default=list(DEFAULT_ALPHA_HORIZONS),
    )
    parser.add_argument("--target-horizon", type=int, default=5)
    parser.add_argument("--diagnostic-horizons", type=int, nargs="+", default=[1, 20])
    parser.add_argument("--flat-volatility-multiplier", type=float, default=0.25)
    parser.add_argument("--max-abs-log-return", type=float, default=0.5)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--purge-bars", type=int, default=20)
    parser.add_argument(
        "--embargo-bars",
        type=int,
        default=5,
        help="RunPod compatibility sentinel.",
    )
    parser.add_argument("--effective-embargo-bars", type=int, default=14)
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("FIN_TS_CPU_WORKERS", "1")),
        help="Thread workers used to build independent symbol windows.",
    )
    return parser


def _manifest_root(input_path: Path) -> Path:
    return input_path.parent.parent if input_path.parent.name == "raw" else input_path.parent


def _pipeline_digest() -> str:
    package_root = Path(__file__).resolve().parents[1]
    files = [
        package_root / "cli" / "prepare_data.py",
        package_root / "data" / "schema.py",
        package_root / "data" / "adjustments.py",
        package_root / "data" / "benchmarks.py",
        package_root / "data" / "quality.py",
        package_root / "data" / "windows.py",
        package_root / "data" / "splits.py",
        package_root / "data" / "io.py",
    ]
    return canonical_json_sha256(
        {path.relative_to(package_root).as_posix(): sha256_file(path) for path in files}
    )


def _load_benchmark_mapping(path: Path | None) -> tuple[dict[str, str], str]:
    if path is None:
        mapping: dict[str, str] = {}
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Benchmark mapping must be a JSON object")
        mapping = {
            str(symbol).strip().upper(): str(benchmark).strip().upper()
            for symbol, benchmark in payload.items()
        }
        if any(not symbol or not benchmark for symbol, benchmark in mapping.items()):
            raise ValueError("Benchmark mapping identifiers must be non-empty")
    return mapping, canonical_json_sha256(mapping)


def prepare_dataset(
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    frame = read_market_data(args.input)
    quality = assess_ohlcv_quality(
        frame,
        max_abs_log_return=args.max_abs_log_return,
    )
    benchmark_mapping, benchmark_mapping_sha256 = _load_benchmark_mapping(args.benchmark_mapping)
    window_audit: dict[str, Any] = {}
    windows = build_causal_windows(
        frame,
        window_size=args.window_size,
        stride=args.sample_stride,
        alpha_horizons=args.alpha_horizons,
        benchmark_mapping=benchmark_mapping,
        target_horizon=args.target_horizon,
        diagnostic_horizons=args.diagnostic_horizons,
        flat_volatility_multiplier=args.flat_volatility_multiplier,
        max_abs_log_return=args.max_abs_log_return,
        audit=window_audit,
        workers=args.workers,
    )
    if not windows:
        raise ValueError(
            "No quality-approved windows were created; check history length and source quality"
        )
    split_audit: dict[str, Any] = {}
    assigned = chronological_split(
        windows,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        purge_bars=args.purge_bars,
        embargo_bars=args.effective_embargo_bars,
        audit=split_audit,
    )
    counts = Counter(record["split"] for record in assigned)
    if set(counts) != {"train", "validation", "test"}:
        raise ValueError("Purged split did not produce non-empty train/validation/test partitions")
    scales = robust_horizon_scales(assigned, args.alpha_horizons)
    label_statistics = {
        "source_split": "train",
        "horizons": list(args.alpha_horizons),
        "robust_scale_method": "max(iqr,mad_x_1.4826,1e-4)",
        "robust_scales": scales,
        "prediction_units": "benchmark_relative_log_return",
    }
    preparation_spec = {
        "processed_schema_version": PROCESSED_SCHEMA_VERSION,
        "window_size": args.window_size,
        "stride": args.stride,
        "effective_sample_stride": args.sample_stride,
        "alpha_horizons": list(args.alpha_horizons),
        "label_kind": "benchmark_relative_adjusted_log_return",
        "signal_timing": "after_close_t",
        "entry_timing": "regular_session_open_t_plus_1",
        "entry_day_counts_as_holding_day_one": True,
        "exit_timing": "regular_session_close_t_plus_h",
        "input_adjustment": "point_in_time_total_return_ohlc_split_adjusted_volume",
        "training_security_scope": TRAINING_SECURITY_SCOPE,
        "split_policy": SPLIT_POLICY,
        "benchmark_mapping_sha256": benchmark_mapping_sha256,
        "target_horizon": args.target_horizon,
        "diagnostic_horizons": args.diagnostic_horizons,
        "flat_volatility_multiplier": args.flat_volatility_multiplier,
        "max_abs_log_return": args.max_abs_log_return,
        "train_fraction": args.train_fraction,
        "validation_fraction": args.validation_fraction,
        "purge_bars": args.purge_bars,
        "embargo_bars": args.embargo_bars,
        "effective_embargo_bars": args.effective_embargo_bars,
    }
    summary = {
        "input_rows": len(frame),
        "candidate_windows_after_quality": len(windows),
        "written_windows": len(assigned),
        "dropped_before_split_assignment": len(windows) - len(assigned),
        "split_counts": dict(sorted(counts.items())),
        "symbols": sorted(str(value) for value in frame["symbol"].unique()),
        "preparation_spec": preparation_spec,
        "quality": quality,
        "window_audit": window_audit,
        "split_audit": split_audit,
        "label_statistics": label_statistics,
        "execution": {
            "parallelism": "thread_pool_by_symbol",
            "workers": args.workers,
        },
    }
    return assigned, summary, preparation_spec


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.input.suffix.lower() not in {".parquet", ".pq"}:
        raise ValueError("Raw training data must be materialized as Parquet")
    if args.output.suffix.lower() not in {".parquet", ".pq"}:
        raise ValueError("Processed training windows must be materialized as Parquet")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite processed dataset: {args.output}")

    manifest_root = _manifest_root(args.input)
    download_manifest_path = args.download_manifest or manifest_root / "download-manifest.json"
    dataset_manifest_path = args.dataset_manifest or manifest_root / "dataset-manifest.json"
    if dataset_manifest_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite ready dataset manifest: {dataset_manifest_path}"
        )
    download = validate_download_manifest(
        download_manifest_path,
        input_path=args.input,
    )
    records, summary, preparation_spec = prepare_dataset(args)
    write_processed_records(records, args.output)
    processed_artifact = artifact_metadata(
        args.output,
        root=manifest_root,
        row_count=len(records),
    )
    split_counts = summary["split_counts"]
    final_manifest = {
        "schema_version": DATASET_MANIFEST_SCHEMA_VERSION,
        "kind": "ohlcv-dataset",
        "state": "ready",
        "created_at": datetime.now(UTC).isoformat(),
        "training_security_scope": TRAINING_SECURITY_SCOPE,
        "dataset_profile": download["dataset_profile"],
        "selected_datasets": download["selected_datasets"],
        "providers": download["providers"],
        "markets": download["markets"],
        "date_range": download["date_range"],
        "symbols": download["symbols"],
        "split_counts": split_counts,
        "label_statistics": summary["label_statistics"],
        "window_audit": summary["window_audit"],
        "split_audit": summary["split_audit"],
        "preparation_spec": preparation_spec,
        "preparation_spec_sha256": canonical_json_sha256(preparation_spec),
        "data_pipeline_digest": _pipeline_digest(),
        "universe_sha256": canonical_json_sha256(download["symbols"]),
        "download_manifest": {
            "relative_path": download_manifest_path.relative_to(manifest_root).as_posix(),
            "sha256": sha256_file(download_manifest_path),
        },
        "request_log": download["artifacts"]["request_log"],
        "api_policy": download["api_policy"],
        "quality": {
            "download": download["quality"],
            "processed": summary["quality"],
            "quality_approved_windows": summary["candidate_windows_after_quality"],
            "split_boundary_dropped_windows": summary[
                "dropped_before_split_assignment"
            ],
            "split_dropped_counts_by_reason": summary["split_audit"][
                "dropped_counts_by_reason"
            ],
        },
        "execution": summary["execution"],
        "artifacts": {
            "raw": download["artifacts"]["raw"],
            "processed": processed_artifact,
        },
    }
    atomic_write_json(dataset_manifest_path, final_manifest)
    payload = {
        **summary,
        "dataset_profile": final_manifest["dataset_profile"],
        "selected_datasets": final_manifest["selected_datasets"],
        "dataset_manifest_path": str(dataset_manifest_path),
        "dataset_manifest_sha256": sha256_file(dataset_manifest_path),
        "output": str(args.output),
    }
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
