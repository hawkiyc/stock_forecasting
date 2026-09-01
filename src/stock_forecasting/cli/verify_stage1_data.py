"""Validate numerical Stage 1 artifacts and publish the RunPod readiness marker."""

from __future__ import annotations

import argparse
import json
import os
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from stock_forecasting.config import ExperimentConfig
from stock_forecasting.data.manifest import (
    canonical_json_sha256,
    load_dataset_manifest,
    sha256_file,
    validate_dataset_preparation_contract,
    validate_training_dataset_manifest,
)
from stock_forecasting.training_paths import resolve_bar_store_path

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_PIPELINE_PATHS = (
    "src/stock_forecasting/cli/prepare_data.py",
    "src/stock_forecasting/data/bar_store.py",
    "src/stock_forecasting/data/adjustments.py",
    "src/stock_forecasting/data/benchmarks.py",
    "src/stock_forecasting/data/horizons.py",
    "src/stock_forecasting/data/manifest.py",
    "src/stock_forecasting/data/schema.py",
    "src/stock_forecasting/data/splits.py",
)


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"{label} is missing: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload


def _relative_to_root(path: Path, root: Path, label: str) -> str:
    resolved = path.resolve(strict=True)
    resolved_root = root.resolve(strict=True)
    if resolved == Path("/workspace") or Path("/workspace") in resolved.parents:
        raise ValueError(f"{label} must never use /workspace")
    try:
        return resolved.relative_to(resolved_root).as_posix()
    except ValueError as error:
        raise ValueError(f"{label} must be stored on NETWORK_VOLUME_ROOT") from error


def _artifact(
    path: Path,
    *,
    volume_root: Path,
    row_count: int,
    label: str,
) -> dict[str, Any]:
    if row_count < 1:
        raise ValueError(f"{label} row count must be positive")
    return {
        "relative_path": _relative_to_root(path, volume_root, label),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "row_count": row_count,
    }


def _validate_code_manifest(path: Path) -> dict[str, Any]:
    payload = _load_json_object(path, "Code readiness manifest")
    if payload.get("kind") != "code" or payload.get("state") != "ready":
        raise ValueError("Code readiness manifest is not ready")
    records = payload.get("files")
    if not isinstance(records, list) or not records:
        raise ValueError("Code readiness manifest contains no file records")
    by_path: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Code readiness manifest contains an invalid file record")
        relative = record.get("path")
        digest = record.get("sha256")
        if (
            not isinstance(relative, str)
            or relative in by_path
            or PurePosixPath(relative).is_absolute()
            or ".." in PurePosixPath(relative).parts
            or not isinstance(digest, str)
            or _SHA256_PATTERN.fullmatch(digest) is None
        ):
            raise ValueError("Code readiness manifest contains an unsafe file record")
        by_path[relative] = record
    missing = sorted(set(_PIPELINE_PATHS).difference(by_path))
    if missing:
        raise ValueError("Code manifest omits numerical pipeline files: " + ", ".join(missing))
    pipeline_payload = {
        relative.removeprefix("src/stock_forecasting/"): by_path[relative]["sha256"]
        for relative in _PIPELINE_PATHS
    }
    payload["quant_pipeline_digest"] = canonical_json_sha256(pipeline_payload)
    return payload


def _validate_model_manifest(
    path: Path,
    *,
    volume_root: Path,
    config: ExperimentConfig,
) -> dict[str, Any]:
    payload = _load_json_object(path, "Kronos cache manifest")
    if payload.get("local_files_only_verified") is not True:
        raise ValueError("Kronos cache has not passed local-files-only verification")
    if payload.get("kronos_source_revision") != config.model.kronos_source_revision:
        raise ValueError("Kronos cache source revision differs from the selected model config")
    repositories = payload.get("repositories")
    if not isinstance(repositories, dict) or not repositories:
        raise ValueError("Kronos cache manifest contains no repositories")
    expected = {
        config.model.time_series_model_id,
        config.model.time_series_tokenizer_id,
    }
    if set(repositories) != expected:
        raise ValueError("Kronos cache repositories differ from the selected model config")
    expected_revisions = {
        config.model.time_series_model_id: config.model.time_series_model_revision,
        config.model.time_series_tokenizer_id: config.model.time_series_tokenizer_revision,
    }
    if payload.get("repository_revisions") != expected_revisions:
        raise ValueError("Kronos cache revisions differ from the selected model config")
    for repository, snapshot in repositories.items():
        snapshot_path = Path(str(snapshot))
        _relative_to_root(snapshot_path, volume_root, f"Model snapshot {repository}")
        if not snapshot_path.is_dir():
            raise ValueError(f"Cached model snapshot is missing: {repository}")
    smoke = payload.get("time_series_smoke_test")
    if (
        not isinstance(smoke, dict)
        or smoke.get("passed") is not True
        or smoke.get("local_files_only") is not True
        or smoke.get("backend") != "kronos"
        or smoke.get("kronos_source_revision") != config.model.kronos_source_revision
        or smoke.get("time_series_model_revision") != config.model.time_series_model_revision
        or smoke.get("time_series_tokenizer_revision")
        != config.model.time_series_tokenizer_revision
    ):
        raise ValueError("Kronos hidden-state smoke test has not passed offline")
    return payload


def _manifest_artifact_path(
    manifest_path: Path,
    artifact: Any,
    *,
    label: str,
) -> Path:
    if not isinstance(artifact, dict):
        raise ValueError(f"Dataset manifest has no valid {label} artifact")
    relative = artifact.get("relative_path")
    if (
        not isinstance(relative, str)
        or not relative
        or PurePosixPath(relative).is_absolute()
        or ".." in PurePosixPath(relative).parts
    ):
        raise ValueError(f"Dataset manifest has an unsafe {label} artifact path")
    path = (manifest_path.parent / relative).resolve(strict=True)
    if sha256_file(path) != artifact.get("sha256") or path.stat().st_size != artifact.get(
        "size_bytes"
    ):
        raise ValueError(f"Dataset manifest {label} artifact integrity mismatch")
    return path


def build_readiness_manifest(
    *,
    dataset_manifest_path: Path,
    code_manifest_path: Path,
    model_manifest_path: Path,
    config_path: Path,
    volume_root: Path,
    launch_id: str,
) -> dict[str, Any]:
    """Bind exact offline data, code, and Kronos cache artifacts."""

    if not re.fullmatch(r"[A-Za-z0-9._-]+", launch_id):
        raise ValueError("RunPod launch ID is not canonical")
    config = ExperimentConfig.from_yaml(config_path)
    dataset = load_dataset_manifest(dataset_manifest_path, required_state="ready")
    artifacts = dataset.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("Dataset manifest artifacts are invalid")
    raw_path = _manifest_artifact_path(
        dataset_manifest_path,
        artifacts.get("raw"),
        label="raw",
    )
    bar_store_manifest_path = _manifest_artifact_path(
        dataset_manifest_path,
        artifacts.get("bar_store_manifest"),
        label="bar-store manifest",
    )
    symbol_index_path = _manifest_artifact_path(
        dataset_manifest_path,
        artifacts.get("symbol_index"),
        label="symbol index",
    )
    cutoff_ranges_path = _manifest_artifact_path(
        dataset_manifest_path,
        artifacts.get("cutoff_ranges"),
        label="cutoff ranges",
    )
    resolved_bar_store = resolve_bar_store_path(bar_store_manifest_path.parent)
    if resolved_bar_store != bar_store_manifest_path.parent.resolve(strict=True):
        raise ValueError("Resolved bar store differs from the dataset manifest")
    validate_training_dataset_manifest(
        dataset_manifest_path,
        profile=config.data.dataset_profile,
        raw_path=raw_path,
        bar_store_path=bar_store_manifest_path.parent,
    )
    storage_preparation_spec = validate_dataset_preparation_contract(
        dataset,
        input_length=config.data.input_length,
        h_start=config.data.h_start,
        max_horizon=config.data.max_horizon,
        alpha_horizons=config.data.alpha_horizons,
        benchmark_mapping_path=config.data.benchmark_mapping_path,
        sample_stride=config.data.sample_stride,
        effective_embargo_trading_days=config.data.effective_embargo_trading_days,
        forecast_horizon=config.data.forecast_horizon,
        diagnostic_horizons=config.data.diagnostic_horizons,
        stride=config.data.stride,
        flat_volatility_multiplier=config.data.flat_volatility_multiplier,
        max_abs_log_return=config.data.max_abs_log_return,
        embargo_trading_days=config.data.embargo_trading_days,
    )
    if dataset.get("selected_datasets") != sorted(config.data.selected_datasets):
        raise ValueError("Dataset providers differ from the selected data profile")
    resolved_preparation_spec = {
        **storage_preparation_spec,
        "h_start": config.data.h_start,
        "alpha_horizons": config.data.alpha_horizons,
    }

    download_manifest = dataset.get("download_manifest")
    download_path = _manifest_artifact_path(
        dataset_manifest_path,
        {
            **download_manifest,
            "size_bytes": (
                dataset_manifest_path.parent / str(download_manifest.get("relative_path", ""))
            )
            .stat()
            .st_size,
        }
        if isinstance(download_manifest, dict)
        else download_manifest,
        label="download manifest",
    )
    if not isinstance(download_manifest, dict) or sha256_file(
        download_path
    ) != download_manifest.get("sha256"):
        raise ValueError("Download manifest integrity mismatch")
    request_log = dataset.get("request_log")
    request_log_path = _manifest_artifact_path(
        dataset_manifest_path,
        request_log,
        label="API request log",
    )

    code = _validate_code_manifest(code_manifest_path)
    if dataset.get("data_pipeline_digest") != code["quant_pipeline_digest"]:
        raise ValueError("Dataset was prepared with a different numerical pipeline revision")
    model = _validate_model_manifest(
        model_manifest_path,
        volume_root=volume_root,
        config=config,
    )

    symbol_payload = dataset.get("symbols")
    split_counts = dataset.get("split_counts")
    if (
        not isinstance(symbol_payload, dict)
        or not isinstance(symbol_payload.get("count"), int)
        or symbol_payload["count"] < 1
        or not isinstance(symbol_payload.get("values"), list)
        or symbol_payload["count"] != len(symbol_payload["values"])
        or symbol_payload["values"] != sorted(set(symbol_payload["values"]))
    ):
        raise ValueError("Dataset manifest contains an invalid symbol universe")
    if (
        not isinstance(split_counts, dict)
        or set(split_counts) != {"train", "validation", "test"}
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 1
            for value in split_counts.values()
        )
    ):
        raise ValueError("Dataset manifest contains invalid split counts")

    return {
        "schema_version": 2,
        "kind": "stage1-dataset",
        "state": "ready",
        "generated_at": datetime.now(UTC).isoformat(),
        "pod_id": os.environ.get("RUNPOD_POD_ID", ""),
        "launch_id": launch_id,
        "training_security_scope": dataset["training_security_scope"],
        "dataset_profile": dataset["dataset_profile"],
        "selected_datasets": dataset["selected_datasets"],
        "providers": dataset.get("providers", []),
        "markets": dataset.get("markets", []),
        "date_range": dataset.get("date_range", {}),
        "symbols": symbol_payload,
        "universe_sha256": dataset["universe_sha256"],
        "split_counts": split_counts,
        "valid_cutoff_count": sum(split_counts.values()),
        "split_audit": dataset["split_audit"],
        "preparation_spec": resolved_preparation_spec,
        "preparation_spec_sha256": canonical_json_sha256(resolved_preparation_spec),
        "storage_preparation_spec_sha256": dataset["preparation_spec_sha256"],
        "data_pipeline_digest": dataset["data_pipeline_digest"],
        "code_release_digest": code.get("release_digest"),
        "models_local_files_only_verified": True,
        "model_repositories": sorted(model["repositories"]),
        "model_manifest_sha256": sha256_file(model_manifest_path),
        "time_series_smoke_test": model["time_series_smoke_test"],
        "dataset_manifest": _artifact(
            dataset_manifest_path,
            volume_root=volume_root,
            row_count=1,
            label="Dataset manifest",
        ),
        "download_manifest": _artifact(
            download_path,
            volume_root=volume_root,
            row_count=1,
            label="Download manifest",
        ),
        "request_log": _artifact(
            request_log_path,
            volume_root=volume_root,
            row_count=int(request_log["row_count"]),
            label="API request log",
        ),
        "model_manifest": _artifact(
            model_manifest_path,
            volume_root=volume_root,
            row_count=len(model["repositories"]),
            label="Model cache manifest",
        ),
        "raw": _artifact(
            raw_path,
            volume_root=volume_root,
            row_count=int(artifacts["raw"]["row_count"]),
            label="Raw dataset",
        ),
        "bar_store_manifest": _artifact(
            bar_store_manifest_path,
            volume_root=volume_root,
            row_count=1,
            label="Bar-store manifest",
        ),
        "symbol_index": _artifact(
            symbol_index_path,
            volume_root=volume_root,
            row_count=int(artifacts["symbol_index"]["row_count"]),
            label="Symbol index",
        ),
        "cutoff_ranges": _artifact(
            cutoff_ranges_path,
            volume_root=volume_root,
            row_count=int(artifacts["cutoff_ranges"]["row_count"]),
            label="Cutoff ranges",
        ),
    }


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--code-manifest", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--volume-root", type=Path, default=Path("/runpod-volume"))
    parser.add_argument("--launch-id", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    if not arguments.verify_only and arguments.output is None:
        parser.error("--output is required unless --verify-only is used")
    payload = build_readiness_manifest(
        dataset_manifest_path=arguments.dataset_manifest,
        code_manifest_path=arguments.code_manifest,
        model_manifest_path=arguments.model_manifest,
        config_path=arguments.config,
        volume_root=arguments.volume_root,
        launch_id=arguments.launch_id,
    )
    if arguments.output is not None:
        expected = (
            arguments.volume_root.resolve(strict=True) / "lifecycle" / "stage1" / "dataset.json"
        )
        if arguments.output.resolve(strict=False) != expected:
            raise ValueError("Dataset readiness output must use the canonical lifecycle path")
        _atomic_write(arguments.output, payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
