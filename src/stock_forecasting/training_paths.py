"""Dependency-light lazy bar-store path resolution."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_bar_store_path(path: str | Path) -> Path:
    """Require the small indexes and success marker used by lazy loading."""

    source = Path(path)
    root = (source.parent if source.name == "bar-store.json" else source).resolve(
        strict=True
    )
    required = (
        root / "bar-store.json",
        root / "symbol-index.parquet",
        root / "cutoff-ranges.parquet",
        root / "_SUCCESS.json",
    )
    missing = [candidate.name for candidate in required if not candidate.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Lazy bar store is incomplete below {root}; missing: {', '.join(missing)}"
        )

    try:
        manifest = json.loads(required[0].read_text(encoding="utf-8"))
        success = json.loads(required[3].read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise ValueError(f"Lazy bar-store metadata is invalid below {root}") from error
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != "1.0"
        or manifest.get("state") != "ready"
        or manifest.get("kind") != "symbol-oriented-ohlcv-bar-store"
        or not isinstance(success, dict)
        or success.get("schema_version") != "1.0"
        or success.get("kind") != "bar-store-success"
        or success.get("state") != "ready"
        or success.get("identity_sha256") != manifest.get("identity_sha256")
        or success.get("bar_store_manifest_sha256") != _sha256(required[0])
        or success.get("symbol_index_sha256") != _sha256(required[1])
        or success.get("cutoff_ranges_sha256") != _sha256(required[2])
    ):
        raise ValueError(f"Lazy bar-store success contract is invalid below {root}")

    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError(f"Lazy bar-store shard manifest is empty below {root}")
    seen: set[str] = set()
    shard_rows = 0
    shard_row_groups = 0
    for shard in shards:
        if not isinstance(shard, dict):
            raise ValueError("Lazy bar-store shard metadata is invalid")
        relative = shard.get("relative_path")
        digest = shard.get("sha256")
        size_bytes = shard.get("size_bytes")
        row_count = shard.get("row_count")
        row_groups = shard.get("row_groups")
        if (
            not isinstance(relative, str)
            or relative in seen
            or PurePosixPath(relative).is_absolute()
            or ".." in PurePosixPath(relative).parts
            or len(PurePosixPath(relative).parts) != 3
            or PurePosixPath(relative).parts[0] != "shards"
            or PurePosixPath(relative).parts[2] != "shard.parquet"
            or not isinstance(digest, str)
            or _SHA256_PATTERN.fullmatch(digest) is None
            or not isinstance(size_bytes, int)
            or isinstance(size_bytes, bool)
            or size_bytes <= 0
            or not isinstance(row_count, int)
            or isinstance(row_count, bool)
            or row_count <= 0
            or not isinstance(row_groups, int)
            or isinstance(row_groups, bool)
            or row_groups <= 0
        ):
            raise ValueError("Lazy bar-store shard metadata is invalid")
        seen.add(relative)
        shard_path = (root / relative).resolve(strict=False)
        try:
            shard_path.relative_to(root)
        except ValueError as error:
            raise ValueError("Lazy bar-store shard escapes its root") from error
        if (
            not shard_path.is_file()
            or shard_path.stat().st_size != size_bytes
        ):
            raise FileNotFoundError(f"Lazy bar-store shard is incomplete: {shard_path}")
        if _sha256(shard_path) != digest:
            raise ValueError(f"Lazy bar-store shard integrity mismatch: {shard_path}")
        shard_rows += row_count
        shard_row_groups += row_groups
    if (
        shard_rows != manifest.get("row_count")
        or shard_row_groups != manifest.get("symbol_count")
    ):
        raise ValueError(f"Lazy bar-store shard totals are inconsistent below {root}")
    return root
