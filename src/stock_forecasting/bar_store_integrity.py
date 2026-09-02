"""Dependency-light integrity checks for immutable lazy bar-store artifacts.

This module validates already materialized files. It does not define numerical
cleaning, split assignment, window selection, or labels, so changes here must
never become part of the durable dataset content identity.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from stock_forecasting.dataset_identity import (
    BAR_STORE_KIND,
    BAR_STORE_SCHEMA_VERSION,
)

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_BUCKET_DIRECTORY_PATTERN = re.compile(r"bucket-[0-9]{4}")


@dataclass(frozen=True)
class BarStoreIntegrity:
    """Validated immutable metadata used by preparation and lazy training."""

    root: Path
    manifest: dict[str, Any]
    manifest_sha256: str
    symbol_index_sha256: str
    cutoff_ranges_sha256: str


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest without loading an artifact in memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json_object(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"{label} is missing or is a symlink: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is not valid JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return payload


def _positive_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _artifact_digest(
    root: Path,
    payload: Any,
    *,
    expected_relative_path: str,
    label: str,
) -> str:
    if not isinstance(payload, Mapping):
        raise ValueError(f"Lazy bar-store {label} metadata is invalid")
    if (
        payload.get("relative_path") != expected_relative_path
        or not isinstance(payload.get("sha256"), str)
        or _SHA256_PATTERN.fullmatch(payload["sha256"]) is None
        or not _positive_integer(payload.get("size_bytes"))
        or not _positive_integer(payload.get("row_count"))
    ):
        raise ValueError(f"Lazy bar-store {label} metadata is invalid")
    path = root / expected_relative_path
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(
            f"Lazy bar-store {label} is missing or is a symlink: {path}"
        )
    if path.stat().st_size != payload["size_bytes"]:
        raise ValueError(f"Lazy bar-store {label} size mismatch: {path}")
    digest = sha256_file(path)
    if digest != payload["sha256"]:
        raise ValueError(f"Lazy bar-store {label} integrity mismatch: {path}")
    return digest


def _validated_shards(root: Path, manifest: Mapping[str, Any]) -> None:
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError(f"Lazy bar-store shard manifest is empty below {root}")

    shards_root = root / "shards"
    if shards_root.is_symlink() or not shards_root.is_dir():
        raise FileNotFoundError(
            f"Lazy bar-store shard root is missing or is a symlink: {shards_root}"
        )

    seen: set[str] = set()
    shard_rows = 0
    shard_row_groups = 0
    for shard in shards:
        if not isinstance(shard, Mapping):
            raise ValueError("Lazy bar-store shard metadata is invalid")
        relative = shard.get("relative_path")
        digest = shard.get("sha256")
        size_bytes = shard.get("size_bytes")
        row_count = shard.get("row_count")
        row_groups = shard.get("row_groups")
        relative_path = PurePosixPath(relative) if isinstance(relative, str) else None
        if (
            relative_path is None
            or relative in seen
            or relative_path.is_absolute()
            or ".." in relative_path.parts
            or len(relative_path.parts) != 3
            or relative_path.parts[0] != "shards"
            or _BUCKET_DIRECTORY_PATTERN.fullmatch(relative_path.parts[1]) is None
            or relative_path.parts[2] != "shard.parquet"
            or not isinstance(digest, str)
            or _SHA256_PATTERN.fullmatch(digest) is None
            or not _positive_integer(size_bytes)
            or not _positive_integer(row_count)
            or not _positive_integer(row_groups)
        ):
            raise ValueError("Lazy bar-store shard metadata is invalid")
        seen.add(relative)

        bucket_root = shards_root / relative_path.parts[1]
        shard_path = bucket_root / "shard.parquet"
        if bucket_root.is_symlink() or shard_path.is_symlink() or not shard_path.is_file():
            raise FileNotFoundError(
                f"Lazy bar-store shard is missing or is a symlink: {shard_path}"
            )
        try:
            shard_path.resolve(strict=True).relative_to(root)
        except ValueError as error:
            raise ValueError("Lazy bar-store shard escapes its root") from error
        if shard_path.stat().st_size != size_bytes:
            raise ValueError(f"Lazy bar-store shard size mismatch: {shard_path}")
        if sha256_file(shard_path) != digest:
            raise ValueError(f"Lazy bar-store shard integrity mismatch: {shard_path}")
        shard_rows += row_count
        shard_row_groups += row_groups

    actual: set[str] = set()
    for bucket_root in shards_root.iterdir():
        if bucket_root.is_symlink() or not bucket_root.is_dir():
            raise ValueError(f"Lazy bar-store has an unsafe shard entry: {bucket_root}")
        if _BUCKET_DIRECTORY_PATTERN.fullmatch(bucket_root.name) is None:
            raise ValueError(f"Lazy bar-store has an unexpected shard directory: {bucket_root}")
        shard_path = bucket_root / "shard.parquet"
        if shard_path.is_symlink() or not shard_path.is_file():
            raise FileNotFoundError(f"Lazy bar-store shard is incomplete: {shard_path}")
        actual.add((Path("shards") / bucket_root.name / "shard.parquet").as_posix())
    if actual != seen:
        raise ValueError(f"Lazy bar-store shard inventory is inconsistent below {root}")
    if (
        shard_rows != manifest.get("row_count")
        or shard_row_groups != manifest.get("symbol_count")
    ):
        raise ValueError(f"Lazy bar-store shard totals are inconsistent below {root}")


def validate_bar_store_artifacts(
    path: str | Path,
    *,
    expected_identity_sha256: str | None = None,
    require_success: bool = True,
) -> BarStoreIntegrity:
    """Validate metadata, every shard, and optionally the publication sentinel."""

    source = Path(path)
    root_candidate = source.parent if source.name == "bar-store.json" else source
    if root_candidate.is_symlink():
        raise ValueError(f"Lazy bar-store root must not be a symlink: {root_candidate}")
    root = root_candidate.resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(f"Lazy bar-store root is not a directory: {root}")

    manifest_path = root / "bar-store.json"
    success_path = root / "_SUCCESS.json"
    manifest = _json_object(manifest_path, "Lazy bar-store manifest")
    identity_sha256 = manifest.get("identity_sha256")
    split_counts = manifest.get("split_counts")
    if (
        manifest.get("schema_version") != BAR_STORE_SCHEMA_VERSION
        or manifest.get("kind") != BAR_STORE_KIND
        or manifest.get("state") != "ready"
        or not isinstance(identity_sha256, str)
        or _SHA256_PATTERN.fullmatch(identity_sha256) is None
        or (
            expected_identity_sha256 is not None
            and identity_sha256 != expected_identity_sha256
        )
        or not _positive_integer(manifest.get("row_count"))
        or not _positive_integer(manifest.get("symbol_count"))
        or not isinstance(split_counts, dict)
        or set(split_counts) != {"train", "validation", "test"}
        or any(not _positive_integer(value) for value in split_counts.values())
    ):
        raise ValueError(f"Lazy bar-store manifest contract is invalid below {root}")

    manifest_sha256 = sha256_file(manifest_path)
    index_sha256 = _artifact_digest(
        root,
        manifest.get("symbol_index"),
        expected_relative_path="symbol-index.parquet",
        label="symbol index",
    )
    ranges_sha256 = _artifact_digest(
        root,
        manifest.get("cutoff_ranges"),
        expected_relative_path="cutoff-ranges.parquet",
        label="cutoff ranges",
    )
    if manifest["symbol_index"]["row_count"] != manifest["symbol_count"]:
        raise ValueError(f"Lazy bar-store symbol count is inconsistent below {root}")
    _validated_shards(root, manifest)

    if require_success:
        success = _json_object(success_path, "Lazy bar-store success marker")
        if (
            success.get("schema_version") != BAR_STORE_SCHEMA_VERSION
            or success.get("kind") != "bar-store-success"
            or success.get("state") != "ready"
            or success.get("identity_sha256") != identity_sha256
            or success.get("bar_store_manifest_sha256") != manifest_sha256
            or success.get("symbol_index_sha256") != index_sha256
            or success.get("cutoff_ranges_sha256") != ranges_sha256
            or success.get("split_counts") != split_counts
        ):
            raise ValueError(f"Lazy bar-store success contract is invalid below {root}")

    return BarStoreIntegrity(
        root=root,
        manifest=manifest,
        manifest_sha256=manifest_sha256,
        symbol_index_sha256=index_sha256,
        cutoff_ranges_sha256=ranges_sha256,
    )
