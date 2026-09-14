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

from stock_forecasting.data.content_identity import (
    CONTENT_IDENTITY_SCHEMA_VERSION,
    content_identity_digest,
    dataset_content_identity,
    provider_materialization_digest,
    raw_dataset_materialization_digest,
)
from stock_forecasting.data.horizons import MAX_ALPHA_HORIZON
from stock_forecasting.dataset_identity import (
    BAR_STORE_KIND,
    BAR_STORE_SCHEMA_VERSION,
    DEFAULT_DATASET_STORAGE_PREPARATION,
)
from stock_forecasting.dataset_identity import (
    storage_preparation_spec as _storage_preparation_spec,
)
from stock_forecasting.dataset_profiles import (
    DatasetProfile,
    runtime_providers,
    selected_datasets,
)

DOWNLOAD_MANIFEST_SCHEMA_VERSION = "2.0"
DATASET_MANIFEST_SCHEMA_VERSION = "4.0"
DATASET_MANIFEST_KIND = "ohlcv-bar-store-dataset"
SUPPORTED_H_START = (1, 2, 3)
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


def storage_preparation_spec(payload: Any) -> dict[str, Any]:
    """Select only values that change persisted bars or cutoff ranges."""

    if not isinstance(payload, dict):
        raise ValueError("Dataset preparation provenance must be a JSON object")
    return _storage_preparation_spec(payload)


def _validated_storage_preparation_spec(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate the hashed subset without treating audit prose as data identity."""

    provenance = payload.get("preparation_provenance")
    if not isinstance(provenance, dict):
        raise ValueError("Dataset manifest has no preparation provenance")
    stored = payload.get("storage_preparation_spec")
    if not isinstance(stored, dict) or set(stored) != set(storage_preparation_spec(provenance)):
        raise ValueError("Dataset manifest storage_preparation_spec is invalid")
    if stored != storage_preparation_spec(provenance):
        raise ValueError(
            "Dataset storage preparation contract disagrees with its provenance"
        )
    digest = payload.get("storage_preparation_spec_sha256")
    if digest != canonical_json_sha256(stored):
        raise ValueError("Dataset storage preparation digest is invalid")
    return stored


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
    state = payload.get("state")
    expected_contract = {
        "downloaded": (DOWNLOAD_MANIFEST_SCHEMA_VERSION, "ohlcv-dataset"),
        "ready": (DATASET_MANIFEST_SCHEMA_VERSION, DATASET_MANIFEST_KIND),
    }.get(state)
    if expected_contract is None or (
        payload.get("schema_version"), payload.get("kind")
    ) != expected_contract:
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
    recorded_security_scope = payload.get("training_security_scope")
    if not isinstance(recorded_security_scope, str) or not recorded_security_scope:
        raise ValueError("Download manifest training security scope is invalid")
    profile = payload.get("dataset_profile")
    try:
        expected_datasets = selected_datasets(profile)
        expected_providers = runtime_providers(profile)
    except (KeyError, TypeError) as error:
        raise ValueError("Download manifest dataset profile is unsupported") from error
    selected_dataset_ids = payload.get("selected_datasets")
    recorded_providers = payload.get("providers")
    api_policy = payload.get("api_policy")
    provider_checkpoints = (
        api_policy.get("provider_materialization_checkpoints")
        if isinstance(api_policy, dict)
        else None
    )
    content_identity = payload.get("data_content_identity")
    if (
        selected_dataset_ids != expected_datasets
        or recorded_providers != expected_providers
        or not isinstance(provider_checkpoints, dict)
        or sorted(provider_checkpoints) != expected_providers
        or not isinstance(content_identity, dict)
        or set(content_identity)
        != {
            "schema_version",
            "selected_datasets",
            "provider_materialization_digests",
            "raw_materialization_digest",
        }
        or content_identity.get("schema_version")
        != CONTENT_IDENTITY_SCHEMA_VERSION
        or content_identity.get("selected_datasets") != selected_dataset_ids
    ):
        raise ValueError("Download manifest data content identity is invalid")
    provider_digests = content_identity.get("provider_materialization_digests")
    if (
        not isinstance(provider_digests, dict)
        or set(provider_digests) != set(selected_dataset_ids)
        or any(
            not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None
            for value in provider_digests.values()
        )
    ):
        raise ValueError("Download manifest provider content identities are invalid")
    expected_provider_digests = {
        provider: provider_materialization_digest(provider)
        for provider in selected_dataset_ids
    }
    if provider_digests != expected_provider_digests:
        raise ValueError(
            "Download manifest was materialized with different provider data semantics"
        )
    if content_identity.get(
        "raw_materialization_digest"
    ) != raw_dataset_materialization_digest():
        raise ValueError(
            "Download manifest was composed with different raw data semantics"
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
    bar_store_path: str | Path,
    require_current_pipeline: bool = True,
) -> dict[str, Any]:
    """Bind training to immutable raw bars and the lazy bar-store indexes."""

    manifest_path = Path(path).resolve(strict=True)
    payload = load_dataset_manifest(manifest_path, required_state="ready")
    if payload.get("dataset_profile") != profile:
        raise ValueError("Dataset manifest profile does not match the experiment config")
    recorded_security_scope = payload.get("training_security_scope")
    if not isinstance(recorded_security_scope, str) or not recorded_security_scope:
        raise ValueError("Dataset manifest training security scope is invalid")
    selected = payload.get("selected_datasets")
    expected_selected = selected_datasets(profile)
    if selected != expected_selected:
        raise ValueError("Dataset manifest selected_datasets do not match its profile")
    content_identity = payload.get("data_content_identity")
    if (
        not isinstance(content_identity, dict)
        or set(content_identity)
        != {
            "schema_version",
            "selected_datasets",
            "provider_materialization_digests",
            "raw_materialization_digest",
            "bar_store_materialization_digest",
        }
        or content_identity.get("schema_version")
        != CONTENT_IDENTITY_SCHEMA_VERSION
        or content_identity.get("selected_datasets") != selected
        or payload.get("data_pipeline_digest")
        != content_identity_digest(content_identity)
    ):
        raise ValueError("Dataset manifest data content identity is invalid")
    if require_current_pipeline and content_identity != dataset_content_identity(selected):
        raise ValueError("Dataset was prepared with different data semantics")
    artifacts = payload.get("artifacts")
    expected_artifacts = {
        "raw",
        "bar_store_manifest",
        "symbol_index",
        "cutoff_ranges",
    }
    if not isinstance(artifacts, dict) or set(artifacts) != expected_artifacts:
        raise ValueError("Dataset manifest must bind raw bars and lazy bar-store indexes")
    raw_artifact, _ = _validate_artifact(
        artifacts["raw"],
        manifest_root=manifest_path.parent,
        label="raw",
    )
    bar_store_manifest, _ = _validate_artifact(
        artifacts["bar_store_manifest"],
        manifest_root=manifest_path.parent,
        label="bar-store manifest",
    )
    symbol_index, _ = _validate_artifact(
        artifacts["symbol_index"],
        manifest_root=manifest_path.parent,
        label="symbol index",
    )
    cutoff_ranges, _ = _validate_artifact(
        artifacts["cutoff_ranges"],
        manifest_root=manifest_path.parent,
        label="cutoff ranges",
    )
    configured_raw = Path(raw_path).resolve(strict=False)
    configured_bar_store = Path(bar_store_path).resolve(strict=False)
    if configured_bar_store.name == "bar-store.json":
        configured_bar_store_root = configured_bar_store.parent
    else:
        configured_bar_store_root = configured_bar_store
        configured_bar_store = configured_bar_store / "bar-store.json"
    if raw_artifact != configured_raw:
        raise ValueError("Configured raw_path does not match the dataset manifest")
    if bar_store_manifest != configured_bar_store:
        raise ValueError("Configured bar_store_path does not match the dataset manifest")
    if symbol_index != configured_bar_store_root / "symbol-index.parquet":
        raise ValueError("Dataset symbol index is outside the configured bar store")
    if cutoff_ranges != configured_bar_store_root / "cutoff-ranges.parquet":
        raise ValueError("Dataset cutoff ranges are outside the configured bar store")
    try:
        store_payload = json.loads(bar_store_manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("Bar-store manifest is not valid JSON") from error
    store_identity = store_payload.get("identity") if isinstance(store_payload, dict) else None
    if (
        not isinstance(store_payload, dict)
        or store_payload.get("schema_version") != BAR_STORE_SCHEMA_VERSION
        or store_payload.get("kind") != BAR_STORE_KIND
        or store_payload.get("state") != "ready"
        or store_payload.get("split_counts") != payload.get("split_counts")
        or not isinstance(store_identity, dict)
        or store_identity.get("materialization_digest")
        != content_identity["bar_store_materialization_digest"]
    ):
        raise ValueError("Bar-store manifest contract is invalid")
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
    _validated_storage_preparation_spec(payload)
    return payload


def validate_dataset_storage_contract(
    payload: dict[str, Any],
    *,
    input_length: int,
    max_horizon: int,
    benchmark_mapping_path: str | Path | None,
    effective_embargo_trading_days: int,
    max_abs_log_return: float,
    fixed_split: dict[str, str] | None = None,
    minimum_evaluation_dates: int = 0,
) -> dict[str, Any]:
    """Bind only persisted bar and cutoff semantics to the current config."""

    if max_horizon != MAX_ALPHA_HORIZON:
        raise ValueError("Configured maximum alpha horizon must remain day 14")
    storage_spec = _validated_storage_preparation_spec(payload)
    if storage_spec.get("fixed_split") != fixed_split:
        raise ValueError("Dataset fixed dates differ from config; prepare a new dataset namespace")
    from stock_forecasting.dataset_identity import validate_fixed_split_audit

    validate_fixed_split_audit(
        payload.get("split_audit"), fixed_split,
        minimum_evaluation_dates=minimum_evaluation_dates,
    )
    expected_integers = {
        "window_size": input_length,
        "max_horizon": max_horizon,
        "purge_bars": DEFAULT_DATASET_STORAGE_PREPARATION["purge_bars"],
        "effective_embargo_bars": effective_embargo_trading_days,
    }
    for key, expected in expected_integers.items():
        if storage_spec.get(key) != expected:
            raise ValueError(
                f"Dataset storage preparation {key}={storage_spec.get(key)!r} "
                f"does not match config value {expected!r}"
            )
    expected_exact = {
        "bar_store_schema_version": BAR_STORE_SCHEMA_VERSION,
        "storage_kind": BAR_STORE_KIND,
        "window_materialized": False,
        "labels_materialized": False,
    }
    for key, expected in expected_exact.items():
        if storage_spec.get(key) != expected:
            raise ValueError(
                f"Dataset storage preparation {key}={storage_spec.get(key)!r} "
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
    if storage_spec.get("benchmark_mapping_sha256") != canonical_json_sha256(benchmark_mapping):
        raise ValueError("Dataset benchmark mapping does not match the experiment config")
    expected_floats = {
        "max_abs_log_return": max_abs_log_return,
        "train_fraction": DEFAULT_DATASET_STORAGE_PREPARATION["train_fraction"],
        "validation_fraction": DEFAULT_DATASET_STORAGE_PREPARATION[
            "validation_fraction"
        ],
    }
    for key, expected in expected_floats.items():
        actual = storage_spec.get(key)
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
        or not isinstance(split_audit.get("policy"), str)
        or not split_audit.get("policy")
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
    if (
        statistics.get("source_split") != "train"
        or statistics.get("state") != "runtime_calibration_required"
        or statistics.get("horizons_available")
        != list(range(1, MAX_ALPHA_HORIZON + 1))
        or statistics.get("prediction_units") != "benchmark_relative_log_return"
        or "robust_scales" in statistics
    ):
        raise ValueError("Dataset manifest label_statistics are invalid")
    return storage_spec


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
        "preparation_provenance": payload.get("preparation_provenance", {}),
        "storage_preparation_spec": payload.get("storage_preparation_spec", {}),
        "storage_preparation_spec_sha256": payload.get(
            "storage_preparation_spec_sha256"
        ),
        "label_statistics": payload.get("label_statistics", {}),
        "window_audit": payload.get("window_audit", {}),
        "split_audit": payload.get("split_audit", {}),
        "quality": payload.get("quality", {}),
        "artifacts": payload["artifacts"],
    }
