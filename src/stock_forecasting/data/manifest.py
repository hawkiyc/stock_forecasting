"""Immutable artifact checks and dataset provenance manifests."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import uuid
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

from stock_forecasting.config import DatasetProfile
from stock_forecasting.data.horizons import MAX_ALPHA_HORIZON, alpha_horizons_from_start
from stock_forecasting.data.schema import TRAINING_SECURITY_SCOPE
from stock_forecasting.data.splits import SPLIT_POLICY

DATASET_MANIFEST_SCHEMA_VERSION = "2.0"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_SECRET_KEY_PATTERN = re.compile(r"(?i)(api[_-]?key|api[_-]?token|password|secret)")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _assert_no_secrets(value: Any, path: str = "manifest") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if _SECRET_KEY_PATTERN.search(str(key)):
                raise ValueError(f"Secret-like key is forbidden in {path}: {key}")
            _assert_no_secrets(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_no_secrets(item, f"{path}[{index}]")
    elif isinstance(value, str) and "api_token=" in value.lower():
        raise ValueError(f"API token query parameters are forbidden in {path}")


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write a manifest atomically after checking that it contains no credentials."""

    _assert_no_secrets(payload)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def artifact_metadata(
    path: str | Path,
    *,
    root: str | Path,
    row_count: int,
) -> dict[str, Any]:
    source = Path(path).resolve(strict=True)
    base = Path(root).resolve(strict=True)
    try:
        relative = source.relative_to(base)
    except ValueError as error:
        raise ValueError(f"Artifact must be below manifest root: {source}") from error
    if row_count < 1:
        raise ValueError("Artifact row_count must be positive")
    return {
        "relative_path": relative.as_posix(),
        "sha256": sha256_file(source),
        "size_bytes": source.stat().st_size,
        "row_count": row_count,
    }


def _validate_artifact(
    payload: Any,
    *,
    manifest_root: Path,
    label: str,
) -> tuple[Path, dict[str, Any]]:
    if not isinstance(payload, dict):
        raise ValueError(f"Dataset manifest has no valid {label} artifact")
    relative_path = payload.get("relative_path")
    digest = payload.get("sha256")
    size_bytes = payload.get("size_bytes")
    row_count = payload.get("row_count")
    if (
        not isinstance(relative_path, str)
        or not relative_path
        or "\\" in relative_path
        or PurePosixPath(relative_path).is_absolute()
        or ".." in PurePosixPath(relative_path).parts
    ):
        raise ValueError(f"Dataset manifest has an unsafe {label} path")
    if not isinstance(digest, str) or _SHA256_PATTERN.fullmatch(digest) is None:
        raise ValueError(f"Dataset manifest has an invalid {label} digest")
    if (
        not isinstance(size_bytes, int)
        or isinstance(size_bytes, bool)
        or size_bytes <= 0
        or not isinstance(row_count, int)
        or isinstance(row_count, bool)
        or row_count <= 0
    ):
        raise ValueError(f"Dataset manifest has invalid {label} bounds")
    source = (manifest_root / relative_path).resolve(strict=False)
    try:
        source.relative_to(manifest_root.resolve(strict=True))
    except ValueError as error:
        raise ValueError(f"Dataset manifest {label} escapes its root") from error
    if not source.is_file():
        raise FileNotFoundError(f"Dataset manifest {label} is missing: {source}")
    actual = {
        "relative_path": relative_path,
        "sha256": sha256_file(source),
        "size_bytes": source.stat().st_size,
        "row_count": row_count,
    }
    if actual != payload:
        raise ValueError(f"Dataset manifest {label} integrity mismatch")
    return source, actual


def load_dataset_manifest(
    path: str | Path,
    *,
    required_state: str | None = None,
) -> dict[str, Any]:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Dataset manifest is missing: {source}") from error
    if not isinstance(payload, dict):
        raise ValueError("Dataset manifest must contain a JSON object")
    _assert_no_secrets(payload)
    if (
        payload.get("schema_version") != DATASET_MANIFEST_SCHEMA_VERSION
        or payload.get("kind") != "ohlcv-dataset"
    ):
        raise ValueError("Dataset manifest schema or kind is unsupported")
    if required_state is not None and payload.get("state") != required_state:
        raise ValueError(f"Dataset manifest state must be {required_state}")
    return payload


def validate_download_manifest(
    path: str | Path,
    *,
    input_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate the complete immutable acquisition checkpoint."""

    manifest_path = Path(path).resolve(strict=True)
    payload = load_dataset_manifest(manifest_path, required_state="downloaded")
    if payload.get("training_security_scope") != TRAINING_SECURITY_SCOPE:
        raise ValueError(
            "Download manifest training security scope is missing or obsolete"
        )
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != {"raw", "request_log"}:
        raise ValueError("Download manifest must bind raw Parquet and API request log")
    raw_path, _ = _validate_artifact(
        artifacts["raw"],
        manifest_root=manifest_path.parent,
        label="raw",
    )
    _validate_artifact(
        artifacts["request_log"],
        manifest_root=manifest_path.parent,
        label="API request log",
    )
    if input_path is not None and raw_path != Path(input_path).resolve(strict=False):
        raise ValueError("Input Parquet does not match the download manifest")
    return payload


def validate_training_dataset_manifest(
    path: str | Path,
    *,
    profile: DatasetProfile,
    raw_path: str | Path,
    processed_path: str | Path,
) -> dict[str, Any]:
    """Bind a training run to exact offline raw and processed Parquet artifacts."""

    manifest_path = Path(path).resolve(strict=True)
    payload = load_dataset_manifest(manifest_path, required_state="ready")
    if payload.get("dataset_profile") != profile:
        raise ValueError("Dataset manifest profile does not match the experiment config")
    if payload.get("training_security_scope") != TRAINING_SECURITY_SCOPE:
        raise ValueError(
            "Dataset manifest training security scope is missing or obsolete"
        )
    selected = payload.get("selected_datasets")
    expected_selected = {
        "tw_only": ["tpex_official", "twse_official"],
        "us_only_eodhd": ["eodhd_us"],
        "us_tw_eodhd": ["eodhd_us", "tpex_official", "twse_official"],
        "us_tw_massive": ["massive_us", "tpex_official", "twse_official"],
    }[profile]
    if selected != expected_selected:
        raise ValueError("Dataset manifest selected_datasets do not match its profile")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != {"raw", "processed"}:
        raise ValueError("Dataset manifest must bind raw and processed artifacts")
    raw_artifact, _ = _validate_artifact(
        artifacts["raw"],
        manifest_root=manifest_path.parent,
        label="raw",
    )
    processed_artifact, _ = _validate_artifact(
        artifacts["processed"],
        manifest_root=manifest_path.parent,
        label="processed",
    )
    configured_raw = Path(raw_path).resolve(strict=False)
    configured_processed = Path(processed_path).resolve(strict=False)
    if configured_processed.is_dir():
        configured_processed = configured_processed / "windows.parquet"
    if raw_artifact != configured_raw:
        raise ValueError("Configured raw_path does not match the dataset manifest")
    if processed_artifact != configured_processed:
        raise ValueError("Configured processed_path does not match the dataset manifest")
    split_counts = payload.get("split_counts")
    if (
        not isinstance(split_counts, dict)
        or set(split_counts) != {"test", "train", "validation"}
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in split_counts.values()
        )
    ):
        raise ValueError("Dataset manifest has invalid split counts")
    return payload


def validate_dataset_preparation_contract(
    payload: dict[str, Any],
    *,
    input_length: int,
    h_start: int,
    max_horizon: int,
    alpha_horizons: list[int],
    benchmark_mapping_path: str | Path | None,
    sample_stride: int,
    effective_embargo_trading_days: int,
    forecast_horizon: int,
    diagnostic_horizons: list[int],
    stride: int,
    flat_volatility_multiplier: float,
    max_abs_log_return: float,
    embargo_trading_days: int,
) -> dict[str, Any]:
    """Bind model-facing window semantics to the ready dataset manifest."""

    expected_horizons = list(alpha_horizons_from_start(h_start))
    if max_horizon != MAX_ALPHA_HORIZON or alpha_horizons != expected_horizons:
        raise ValueError(
            "Configured alpha horizons must be contiguous from h_start through day 14"
        )

    preparation_spec = payload.get("preparation_spec")
    if not isinstance(preparation_spec, dict):
        raise ValueError("Dataset manifest has no preparation_spec")
    digest = payload.get("preparation_spec_sha256")
    if digest != canonical_json_sha256(preparation_spec):
        raise ValueError("Dataset manifest preparation_spec digest is invalid")

    expected_integers = {
        "window_size": input_length,
        "h_start": h_start,
        "max_horizon": max_horizon,
        "target_horizon": forecast_horizon,
        "stride": stride,
        "purge_bars": 20,
        "embargo_bars": embargo_trading_days,
    }
    for key, expected in expected_integers.items():
        if preparation_spec.get(key) != expected:
            raise ValueError(
                f"Dataset preparation {key}={preparation_spec.get(key)!r} "
                f"does not match config value {expected!r}"
            )
    expected_exact = {
        "processed_schema_version": "4.0",
        "alpha_horizons": alpha_horizons,
        "label_kind": "benchmark_relative_adjusted_log_return",
        "signal_timing": "after_close_t",
        "entry_timing": "regular_session_open_t_plus_1",
        "entry_day_counts_as_holding_day_one": True,
        "exit_timing": "regular_session_close_t_plus_h",
        "input_adjustment": "point_in_time_total_return_ohlc_split_adjusted_volume",
        "training_security_scope": TRAINING_SECURITY_SCOPE,
        "split_policy": SPLIT_POLICY,
        "effective_sample_stride": sample_stride,
        "effective_embargo_bars": effective_embargo_trading_days,
    }
    for key, expected in expected_exact.items():
        if preparation_spec.get(key) != expected:
            raise ValueError(
                f"Dataset preparation {key}={preparation_spec.get(key)!r} "
                f"does not match config contract {expected!r}"
            )
    if benchmark_mapping_path is None:
        benchmark_mapping: dict[str, str] = {}
    else:
        mapping_payload = json.loads(Path(benchmark_mapping_path).read_text(encoding="utf-8"))
        if not isinstance(mapping_payload, dict):
            raise ValueError("Configured benchmark mapping must be a JSON object")
        benchmark_mapping = {
            str(symbol).strip().upper(): str(benchmark).strip().upper()
            for symbol, benchmark in mapping_payload.items()
        }
    if preparation_spec.get("benchmark_mapping_sha256") != canonical_json_sha256(benchmark_mapping):
        raise ValueError("Dataset benchmark mapping does not match the experiment config")
    actual_diagnostics = preparation_spec.get("diagnostic_horizons")
    if not isinstance(actual_diagnostics, list) or sorted(actual_diagnostics) != sorted(
        diagnostic_horizons
    ):
        raise ValueError("Dataset diagnostic_horizons do not match the experiment config")
    expected_floats = {
        "flat_volatility_multiplier": flat_volatility_multiplier,
        "max_abs_log_return": max_abs_log_return,
        "train_fraction": 0.70,
        "validation_fraction": 0.15,
    }
    for key, expected in expected_floats.items():
        actual = preparation_spec.get(key)
        if (
            not isinstance(actual, int | float)
            or isinstance(actual, bool)
            or not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12)
        ):
            raise ValueError(
                f"Dataset preparation {key}={actual!r} does not match config value {expected!r}"
            )
    split_audit = payload.get("split_audit")
    if (
        not isinstance(split_audit, dict)
        or split_audit.get("schema_version") != "causal-split-audit-v1"
        or split_audit.get("policy") != SPLIT_POLICY
        or split_audit.get("comparison") != "label.end_at < next_split_start"
        or split_audit.get("violations") != 0
    ):
        raise ValueError("Dataset manifest split_audit policy is invalid")
    split_summaries = split_audit.get("splits")
    if not isinstance(split_summaries, dict) or set(split_summaries) != {
        "train",
        "validation",
        "test",
    }:
        raise ValueError("Dataset manifest split_audit summaries are invalid")

    def audit_timestamp(value: Any, label: str) -> datetime:
        if not isinstance(value, str):
            raise ValueError(f"Dataset manifest {label} timestamp is invalid")
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as error:
            raise ValueError(f"Dataset manifest {label} timestamp is invalid") from error
        if parsed.tzinfo is None:
            raise ValueError(f"Dataset manifest {label} timestamp must include a timezone")
        return parsed

    train_boundary = audit_timestamp(
        split_audit.get("train_boundary_exclusive"),
        "train boundary",
    )
    validation_boundary = audit_timestamp(
        split_audit.get("validation_boundary_exclusive"),
        "validation boundary",
    )
    split_counts = payload.get("split_counts")
    label_end_counts = split_audit.get("label_end_counts")
    maximum_label_end = split_audit.get("maximum_label_end")
    for split, summary in split_summaries.items():
        if (
            not isinstance(summary, dict)
            or not isinstance(split_counts, dict)
            or summary.get("records") != split_counts.get(split)
            or not isinstance(label_end_counts, dict)
            or label_end_counts.get(split) != split_counts.get(split)
            or not isinstance(maximum_label_end, dict)
            or maximum_label_end.get(split) != summary.get("label_end_max_at")
        ):
            raise ValueError(f"Dataset manifest {split} split audit count is invalid")
        audit_timestamp(summary.get("cutoff_start_at"), f"{split} cutoff start")
        audit_timestamp(summary.get("cutoff_end_at"), f"{split} cutoff end")
        audit_timestamp(summary.get("label_end_max_at"), f"{split} label end")
    train = split_summaries["train"]
    validation = split_summaries["validation"]
    test = split_summaries["test"]
    if (
        split_audit.get("validation_start") != validation.get("cutoff_start_at")
        or split_audit.get("test_start") != test.get("cutoff_start_at")
    ):
        raise ValueError("Dataset manifest split start audit is inconsistent")
    if audit_timestamp(train["label_end_max_at"], "train label end") >= train_boundary:
        raise ValueError("Dataset manifest train labels cross the split boundary")
    if (
        audit_timestamp(validation["label_end_max_at"], "validation label end")
        >= validation_boundary
    ):
        raise ValueError("Dataset manifest validation labels cross the split boundary")
    if not (
        audit_timestamp(train["cutoff_end_at"], "train cutoff end")
        < audit_timestamp(validation["cutoff_start_at"], "validation cutoff start")
        and audit_timestamp(validation["cutoff_end_at"], "validation cutoff end")
        < audit_timestamp(test["cutoff_start_at"], "test cutoff start")
    ):
        raise ValueError("Dataset manifest split cutoff ranges are not chronological")
    statistics = payload.get("label_statistics")
    if not isinstance(statistics, dict):
        raise ValueError("Dataset manifest has no label_statistics")
    scales = statistics.get("robust_scales")
    if (
        statistics.get("source_split") != "train"
        or statistics.get("horizons") != alpha_horizons
        or statistics.get("prediction_units") != "benchmark_relative_log_return"
        or not isinstance(scales, list)
        or len(scales) != len(alpha_horizons)
        or any(
            not isinstance(value, int | float)
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) <= 0.0
            for value in scales
        )
    ):
        raise ValueError("Dataset manifest label_statistics are invalid")
    return preparation_spec


def provenance_summary(path: str | Path) -> dict[str, Any]:
    """Return the non-secret subset attached to evaluation and inference output."""

    manifest_path = Path(path)
    payload = load_dataset_manifest(manifest_path, required_state="ready")
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "dataset_profile": payload["dataset_profile"],
        "selected_datasets": payload["selected_datasets"],
        "training_security_scope": payload.get("training_security_scope"),
        "providers": payload.get("providers", []),
        "markets": payload.get("markets", []),
        "symbols": payload.get("symbols", {}),
        "date_range": payload.get("date_range", {}),
        "split_counts": payload.get("split_counts", {}),
        "preparation_spec": payload.get("preparation_spec", {}),
        "label_statistics": payload.get("label_statistics", {}),
        "window_audit": payload.get("window_audit", {}),
        "split_audit": payload.get("split_audit", {}),
        "quality": payload.get("quality", {}),
        "artifacts": payload["artifacts"],
    }
