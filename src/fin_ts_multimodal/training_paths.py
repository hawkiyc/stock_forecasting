"""Dependency-light training artifact path resolution."""

from __future__ import annotations

from pathlib import Path


def resolve_processed_dataset_path(path: str | Path) -> Path:
    source = Path(path)
    if source.is_file():
        if source.suffix.lower() not in {".parquet", ".pq"}:
            raise ValueError("Processed quant dataset must be Parquet")
        return source
    candidate = source / "windows.parquet"
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(
        f"No processed quant dataset found below {source}; expected windows.parquet"
    )
