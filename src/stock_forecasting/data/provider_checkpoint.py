"""Durable, integrity-checked provider materialization checkpoints."""

from __future__ import annotations

import json
import os
import re
import uuid
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from stock_forecasting.data.manifest import (
    artifact_metadata,
    atomic_write_json,
    canonical_json_sha256,
)
from stock_forecasting.data.schema import TRAINING_SECURITY_SCOPE

PROVIDER_CHECKPOINT_SCHEMA_VERSION = "1.0"
_PROVIDER_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_PARQUET_NAME = "market.parquet"
_REQUEST_LOG_NAME = "request-log.jsonl"
_MANIFEST_NAME = "manifest.json"
_CHECKPOINT_FILES = (_PARQUET_NAME, _REQUEST_LOG_NAME, _MANIFEST_NAME)


class ProviderCheckpointValidationError(ValueError):
    """Reject a provider checkpoint that cannot prove identity and integrity."""


@dataclass(frozen=True)
class ProviderCheckpoint:
    """Validated paths and metadata for one provider materialization."""

    provider: str
    root: Path
    manifest_path: Path
    parquet_path: Path
    request_log_path: Path
    identity_sha256: str
    row_count: int
    request_count: int
    metadata: dict[str, Any]


def _validated_provider(provider: str) -> str:
    if _PROVIDER_PATTERN.fullmatch(provider) is None:
        raise ValueError("Provider checkpoint name is unsafe")
    return provider


def provider_checkpoint_path(
    root: str | Path,
    *,
    provider: str,
    identity: dict[str, Any],
) -> Path:
    """Return the immutable directory for one provider identity."""

    provider = _validated_provider(provider)
    digest = canonical_json_sha256(identity)
    return Path(root) / provider / digest


def _validate_artifact(
    checkpoint_root: Path,
    payload: Any,
    *,
    expected_name: str,
    label: str,
) -> tuple[Path, dict[str, Any]]:
    if not isinstance(payload, dict) or payload.get("relative_path") != expected_name:
        raise ProviderCheckpointValidationError(
            f"Provider checkpoint has an invalid {label} path"
        )
    row_count = payload.get("row_count")
    if not isinstance(row_count, int) or isinstance(row_count, bool) or row_count < 1:
        raise ProviderCheckpointValidationError(
            f"Provider checkpoint has an invalid {label} row count"
        )
    artifact_path = checkpoint_root / expected_name
    try:
        actual = artifact_metadata(
            artifact_path,
            root=checkpoint_root,
            row_count=row_count,
        )
    except (FileNotFoundError, OSError, ValueError) as error:
        raise ProviderCheckpointValidationError(
            f"Provider checkpoint {label} is unavailable"
        ) from error
    if actual != payload:
        raise ProviderCheckpointValidationError(
            f"Provider checkpoint {label} integrity mismatch"
        )
    return artifact_path, actual


def load_provider_checkpoint(
    root: str | Path,
    *,
    provider: str,
    identity: dict[str, Any],
) -> ProviderCheckpoint | None:
    """Load a complete checkpoint or return ``None`` when it does not exist."""

    provider = _validated_provider(provider)
    identity_sha256 = canonical_json_sha256(identity)
    checkpoint_root = provider_checkpoint_path(
        root,
        provider=provider,
        identity=identity,
    )
    if checkpoint_root.is_symlink():
        raise ProviderCheckpointValidationError(
            "Provider checkpoint path is not a regular directory"
        )
    if not checkpoint_root.exists():
        return None
    if not checkpoint_root.is_dir():
        raise ProviderCheckpointValidationError(
            "Provider checkpoint path is not a regular directory"
        )
    manifest_path = checkpoint_root / _MANIFEST_NAME
    if manifest_path.is_symlink():
        raise ProviderCheckpointValidationError("Provider checkpoint manifest is a symlink")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError) as error:
        raise ProviderCheckpointValidationError(
            "Provider checkpoint manifest is unreadable"
        ) from error
    required_keys = {
        "artifacts",
        "created_at",
        "identity",
        "identity_sha256",
        "kind",
        "metadata",
        "provider",
        "schema_version",
        "state",
        "training_security_scope",
    }
    if not isinstance(payload, dict) or set(payload) != required_keys:
        raise ProviderCheckpointValidationError(
            "Provider checkpoint manifest fields are invalid"
        )
    if (
        payload.get("schema_version") != PROVIDER_CHECKPOINT_SCHEMA_VERSION
        or payload.get("kind") != "ohlcv-provider-materialization"
        or payload.get("state") != "complete"
        or payload.get("provider") != provider
        or payload.get("training_security_scope") != TRAINING_SECURITY_SCOPE
        or payload.get("identity") != identity
        or payload.get("identity_sha256") != identity_sha256
        or not isinstance(payload.get("created_at"), str)
        or not isinstance(payload.get("metadata"), dict)
    ):
        raise ProviderCheckpointValidationError(
            "Provider checkpoint manifest contract is invalid"
        )
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != {"parquet", "request_log"}:
        raise ProviderCheckpointValidationError(
            "Provider checkpoint artifact contract is invalid"
        )
    parquet_path, parquet = _validate_artifact(
        checkpoint_root,
        artifacts["parquet"],
        expected_name=_PARQUET_NAME,
        label="Parquet",
    )
    request_log_path, request_log = _validate_artifact(
        checkpoint_root,
        artifacts["request_log"],
        expected_name=_REQUEST_LOG_NAME,
        label="request log",
    )
    return ProviderCheckpoint(
        provider=provider,
        root=checkpoint_root,
        manifest_path=manifest_path,
        parquet_path=parquet_path,
        request_log_path=request_log_path,
        identity_sha256=identity_sha256,
        row_count=parquet["row_count"],
        request_count=request_log["row_count"],
        metadata=dict(payload["metadata"]),
    )


def _cleanup_staging(staging: Path) -> None:
    """Remove only known files from an unpublished checkpoint directory."""

    for name in _CHECKPOINT_FILES:
        (staging / name).unlink(missing_ok=True)
    with suppress(FileNotFoundError):
        staging.rmdir()


def publish_provider_checkpoint(
    root: str | Path,
    *,
    provider: str,
    identity: dict[str, Any],
    parquet_path: str | Path,
    request_log_path: str | Path,
    row_count: int,
    request_count: int,
    metadata: dict[str, Any],
) -> ProviderCheckpoint:
    """Atomically publish hard-linked provider artifacts on the persistent volume."""

    provider = _validated_provider(provider)
    checkpoint_root = provider_checkpoint_path(
        root,
        provider=provider,
        identity=identity,
    )
    existing = load_provider_checkpoint(root, provider=provider, identity=identity)
    if existing is not None:
        return existing
    source_parquet = Path(parquet_path)
    source_request_log = Path(request_log_path)
    for source, label in (
        (source_parquet, "Parquet"),
        (source_request_log, "request log"),
    ):
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"Provider checkpoint source {label} is unavailable")
    if (
        not isinstance(row_count, int)
        or isinstance(row_count, bool)
        or row_count < 1
        or not isinstance(request_count, int)
        or isinstance(request_count, bool)
        or request_count < 1
    ):
        raise ValueError("Provider checkpoint artifact row counts must be positive")

    provider_root = checkpoint_root.parent
    provider_root.mkdir(parents=True, exist_ok=True)
    if provider_root.is_symlink():
        raise ValueError("Provider checkpoint root must not be a symlink")
    staging = provider_root / f".{checkpoint_root.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        os.link(source_parquet, staging / _PARQUET_NAME, follow_symlinks=False)
        os.link(source_request_log, staging / _REQUEST_LOG_NAME, follow_symlinks=False)
        manifest = {
            "schema_version": PROVIDER_CHECKPOINT_SCHEMA_VERSION,
            "kind": "ohlcv-provider-materialization",
            "state": "complete",
            "created_at": datetime.now(UTC).isoformat(),
            "training_security_scope": TRAINING_SECURITY_SCOPE,
            "provider": provider,
            "identity": identity,
            "identity_sha256": canonical_json_sha256(identity),
            "metadata": metadata,
            "artifacts": {
                "parquet": artifact_metadata(
                    staging / _PARQUET_NAME,
                    root=staging,
                    row_count=row_count,
                ),
                "request_log": artifact_metadata(
                    staging / _REQUEST_LOG_NAME,
                    root=staging,
                    row_count=request_count,
                ),
            },
        }
        atomic_write_json(staging / _MANIFEST_NAME, manifest)
        try:
            staging.rename(checkpoint_root)
        except OSError:
            if not checkpoint_root.is_dir():
                raise
            _cleanup_staging(staging)
        loaded = load_provider_checkpoint(root, provider=provider, identity=identity)
        if loaded is None:
            raise ProviderCheckpointValidationError(
                "Published provider checkpoint is missing"
            )
        return loaded
    except BaseException:
        _cleanup_staging(staging)
        raise


def quarantine_provider_checkpoint(
    root: str | Path,
    *,
    provider: str,
    identity: dict[str, Any],
) -> Path | None:
    """Move one invalid checkpoint aside without deleting its artifacts."""

    provider = _validated_provider(provider)
    checkpoint_root = provider_checkpoint_path(
        root,
        provider=provider,
        identity=identity,
    )
    if not checkpoint_root.exists() and not checkpoint_root.is_symlink():
        return None
    quarantine_root = Path(root) / "quarantine" / provider
    quarantine_root.mkdir(parents=True, exist_ok=True)
    destination = quarantine_root / f"{checkpoint_root.name}-{uuid.uuid4().hex}"
    checkpoint_root.rename(destination)
    return destination
