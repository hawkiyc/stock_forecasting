"""Build a resumable symbol bar store for lazy, leakage-safe training samples."""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from stock_forecasting.data.bar_store import (
    DEFAULT_BATCH_ROWS,
    DEFAULT_BUCKET_COUNT,
    DEFAULT_MAX_HORIZON,
    PreparationPaused,
    bar_store_preparation_provenance,
    build_symbol_bar_store,
)
from stock_forecasting.data.content_identity import (
    bar_store_materialization_digest,
    content_identity_digest,
    dataset_content_identity,
)
from stock_forecasting.data.horizons import DEFAULT_H_START
from stock_forecasting.data.manifest import (
    DATASET_MANIFEST_KIND,
    DATASET_MANIFEST_SCHEMA_VERSION,
    artifact_metadata,
    atomic_write_json,
    canonical_json_sha256,
    sha256_file,
    storage_preparation_spec,
    validate_download_manifest,
)
from stock_forecasting.data.schema import TRAINING_SECURITY_SCOPE
from stock_forecasting.dataset_identity import (
    DEFAULT_DATASET_STORAGE_PREPARATION,
    FIXED_EVALUATION_SPLIT,
    MIN_FIXED_EVALUATION_DATES,
    validate_fixed_split_audit,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Canonical raw Parquet.")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Persistent prepared/bar-store directory; partial builds resume in place.",
    )
    parser.add_argument("--download-manifest", type=Path)
    parser.add_argument("--dataset-manifest", type=Path)
    parser.add_argument("--benchmark-mapping", type=Path)
    parser.add_argument(
        "--fixed-evaluation", action="store_true",
        help="Use the production 2025-06 / 2025-12 / 2026-06 exclusive boundaries.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=DEFAULT_DATASET_STORAGE_PREPARATION["window_size"],
    )
    parser.add_argument("--stride", type=int, default=5, help="RunPod compatibility sentinel.")
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument(
        "--h-start",
        type=int,
        choices=(1, 2, 3),
        default=DEFAULT_H_START,
        help="Runtime label subset; it does not change the bar-store identity.",
    )
    parser.add_argument("--target-horizon", type=int, default=5)
    parser.add_argument("--diagnostic-horizons", type=int, nargs="+", default=[1, 20])
    parser.add_argument("--flat-volatility-multiplier", type=float, default=0.25)
    parser.add_argument(
        "--max-abs-log-return",
        type=float,
        default=DEFAULT_DATASET_STORAGE_PREPARATION["max_abs_log_return"],
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=DEFAULT_DATASET_STORAGE_PREPARATION["train_fraction"],
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=DEFAULT_DATASET_STORAGE_PREPARATION["validation_fraction"],
    )
    parser.add_argument(
        "--purge-bars",
        type=int,
        default=DEFAULT_DATASET_STORAGE_PREPARATION["purge_bars"],
    )
    parser.add_argument(
        "--embargo-bars",
        type=int,
        default=5,
        help="RunPod compatibility sentinel.",
    )
    parser.add_argument(
        "--effective-embargo-bars",
        type=int,
        default=DEFAULT_DATASET_STORAGE_PREPARATION["effective_embargo_bars"],
    )
    parser.add_argument("--bucket-count", type=int, default=DEFAULT_BUCKET_COUNT)
    parser.add_argument("--batch-rows", type=int, default=DEFAULT_BATCH_ROWS)
    parser.add_argument(
        "--deadline-epoch-seconds",
        type=float,
        help="Pause safely before this absolute Unix timestamp.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("FIN_TS_CPU_WORKERS", "1")),
        help=(
            "Maximum process workers; each preparation phase lowers this value when "
            "CPU visibility or its conservative memory budget requires it."
        ),
    )
    return parser


def _manifest_root(input_path: Path) -> Path:
    return input_path.parent.parent if input_path.parent.name == "raw" else input_path.parent


def _validate_dataset_manifest_destination(path: Path, manifest_root: Path) -> None:
    """Require relative artifact paths to remain valid before and after publication."""

    expected_parent = manifest_root.resolve(strict=True)
    actual_parent = path.parent.resolve(strict=False)
    if actual_parent != expected_parent:
        raise ValueError("--dataset-manifest must be written directly under the dataset root")


def _pipeline_identity(download: dict[str, Any]) -> dict[str, Any]:
    acquisition_identity = download["data_content_identity"]
    return dataset_content_identity(
        download["selected_datasets"],
        provider_digests=acquisition_identity["provider_materialization_digests"],
        raw_digest=acquisition_identity["raw_materialization_digest"],
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


def _relative_download_manifest(path: Path, manifest_root: Path) -> dict[str, Any]:
    return {
        "relative_path": path.resolve(strict=True)
        .relative_to(manifest_root.resolve(strict=True))
        .as_posix(),
        "sha256": sha256_file(path),
    }


def _ready_manifest(
    *,
    arguments: argparse.Namespace,
    manifest_root: Path,
    download_manifest_path: Path,
    download: dict[str, Any],
    result: Any,
    benchmark_mapping_sha256: str,
) -> dict[str, Any]:
    validate_fixed_split_audit(
        result.split_audit,
        FIXED_EVALUATION_SPLIT if arguments.fixed_evaluation else None,
        minimum_evaluation_dates=MIN_FIXED_EVALUATION_DATES,
    )
    preparation_provenance = bar_store_preparation_provenance(
        window_size=arguments.window_size,
        max_horizon=DEFAULT_MAX_HORIZON,
        benchmark_mapping_sha256=benchmark_mapping_sha256,
        max_abs_log_return=arguments.max_abs_log_return,
        train_fraction=arguments.train_fraction,
        validation_fraction=arguments.validation_fraction,
        purge_bars=arguments.purge_bars,
        compatibility_stride=arguments.stride,
        effective_sample_stride=arguments.sample_stride,
        compatibility_embargo_bars=arguments.embargo_bars,
        effective_embargo_bars=arguments.effective_embargo_bars,
        target_horizon=arguments.target_horizon,
        diagnostic_horizons=list(arguments.diagnostic_horizons),
        flat_volatility_multiplier=arguments.flat_volatility_multiplier,
        fixed_split=FIXED_EVALUATION_SPLIT if arguments.fixed_evaluation else None,
    )
    storage_spec = storage_preparation_spec(preparation_provenance)
    bar_store_payload = json.loads(result.bar_store_manifest_path.read_text(encoding="utf-8"))
    pipeline_identity = _pipeline_identity(download)
    return {
        "schema_version": DATASET_MANIFEST_SCHEMA_VERSION,
        "kind": DATASET_MANIFEST_KIND,
        "state": "ready",
        "created_at": datetime.now(UTC).isoformat(),
        "training_security_scope": TRAINING_SECURITY_SCOPE,
        "dataset_profile": download["dataset_profile"],
        "selected_datasets": download["selected_datasets"],
        "providers": download["providers"],
        "markets": download["markets"],
        "date_range": download["date_range"],
        "symbols": download["symbols"],
        "split_counts": result.split_counts,
        "label_statistics": {
            "state": "runtime_calibration_required",
            "source_split": "train",
            "horizons_available": list(range(1, DEFAULT_MAX_HORIZON + 1)),
            "robust_scale_method": "max(iqr,mad_x_1.4826,1e-4)",
            "prediction_units": "benchmark_relative_log_return",
        },
        "window_audit": {
            "storage": "lazy_cutoff_ranges",
            "windows_materialized": False,
            "labels_materialized": False,
            "valid_cutoff_count": sum(result.split_counts.values()),
            "cutoff_range_rows": int(bar_store_payload["cutoff_ranges"]["row_count"]),
        },
        "split_audit": result.split_audit,
        "preparation_provenance": preparation_provenance,
        "storage_preparation_spec": storage_spec,
        "storage_preparation_spec_sha256": canonical_json_sha256(storage_spec),
        "data_content_identity": pipeline_identity,
        "data_pipeline_digest": content_identity_digest(pipeline_identity),
        "universe_sha256": canonical_json_sha256(download["symbols"]),
        "download_manifest": _relative_download_manifest(
            download_manifest_path,
            manifest_root,
        ),
        "request_log": download["artifacts"]["request_log"],
        "api_policy": download["api_policy"],
        "quality": {
            "download": download["quality"],
            "bar_store": result.quality,
            "split_dropped_counts_by_reason": result.split_audit["dropped_counts_by_reason"],
        },
        "execution": {
            **result.execution,
            "requested_workers": arguments.workers,
        },
        "artifacts": {
            "raw": download["artifacts"]["raw"],
            "bar_store_manifest": artifact_metadata(
                result.bar_store_manifest_path,
                root=manifest_root,
                row_count=1,
            ),
            "symbol_index": artifact_metadata(
                result.symbol_index_path,
                root=manifest_root,
                row_count=int(bar_store_payload["symbol_index"]["row_count"]),
            ),
            "cutoff_ranges": artifact_metadata(
                result.cutoff_ranges_path,
                root=manifest_root,
                row_count=int(bar_store_payload["cutoff_ranges"]["row_count"]),
            ),
        },
    }


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.input.suffix.lower() not in {".parquet", ".pq"}:
        raise ValueError("Raw training data must be materialized as Parquet")
    if arguments.output.suffix:
        raise ValueError("--output must be a persistent bar-store directory")
    if arguments.target_horizon != 5 or sorted(arguments.diagnostic_horizons) != [1, 20]:
        raise ValueError("Stable RunPod target and diagnostic sentinels must remain 5 and [1,20]")
    if arguments.sample_stride != 1 or arguments.stride != 5:
        raise ValueError("Stable lazy sampling requires sample_stride=1 and stride sentinel=5")
    if (
        arguments.embargo_bars != 5
        or arguments.effective_embargo_bars
        != DEFAULT_DATASET_STORAGE_PREPARATION["effective_embargo_bars"]
    ):
        raise ValueError("Stable embargo sentinels must remain 5 and effective 14")
    if arguments.workers < 1:
        raise ValueError("workers must be positive")

    manifest_root = _manifest_root(arguments.input)
    download_manifest_path = arguments.download_manifest or manifest_root / "download-manifest.json"
    dataset_manifest_path = arguments.dataset_manifest or manifest_root / "dataset-manifest.json"
    _validate_dataset_manifest_destination(dataset_manifest_path, manifest_root)
    if dataset_manifest_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite ready dataset manifest: {dataset_manifest_path}"
        )
    download = validate_download_manifest(
        download_manifest_path,
        input_path=arguments.input,
    )
    benchmark_mapping, benchmark_mapping_sha256 = _load_benchmark_mapping(
        arguments.benchmark_mapping
    )
    try:
        result = build_symbol_bar_store(
            raw_path=arguments.input,
            output_root=arguments.output,
            download_manifest=download,
            benchmark_mapping=benchmark_mapping,
            window_size=arguments.window_size,
            max_horizon=DEFAULT_MAX_HORIZON,
            max_abs_log_return=arguments.max_abs_log_return,
            train_fraction=arguments.train_fraction,
            validation_fraction=arguments.validation_fraction,
            purge_bars=arguments.purge_bars,
            embargo_bars=arguments.effective_embargo_bars,
            fixed_split=FIXED_EVALUATION_SPLIT if arguments.fixed_evaluation else None,
            bucket_count=arguments.bucket_count,
            batch_rows=arguments.batch_rows,
            deadline_epoch_seconds=arguments.deadline_epoch_seconds,
            workers=arguments.workers,
            materialization_digest=bar_store_materialization_digest(),
        )
    except PreparationPaused as error:
        print(
            json.dumps(
                {
                    "state": "waiting_for_preparation",
                    "resumable": True,
                    "bar_store_root": str(arguments.output),
                    "detail": str(error),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 75

    final_manifest = _ready_manifest(
        arguments=arguments,
        manifest_root=manifest_root,
        download_manifest_path=download_manifest_path,
        download=download,
        result=result,
        benchmark_mapping_sha256=benchmark_mapping_sha256,
    )
    atomic_write_json(dataset_manifest_path, final_manifest)
    print(
        json.dumps(
            {
                "state": "ready",
                "dataset_profile": final_manifest["dataset_profile"],
                "selected_datasets": final_manifest["selected_datasets"],
                "dataset_manifest_path": str(dataset_manifest_path),
                "dataset_manifest_sha256": sha256_file(dataset_manifest_path),
                "bar_store_root": str(arguments.output),
                "split_counts": result.split_counts,
                "effective_dates_by_market": result.split_audit.get("dates_by_market", {}),
                "window_materialized": False,
                "labels_materialized": False,
                "requested_h_start": arguments.h_start,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
