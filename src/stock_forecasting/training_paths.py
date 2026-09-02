"""Dependency-light lazy bar-store path resolution."""

from __future__ import annotations

from pathlib import Path

from stock_forecasting.bar_store_integrity import validate_bar_store_artifacts


def resolve_bar_store_path(path: str | Path) -> Path:
    """Require a fully published and integrity-checked lazy bar store."""

    return validate_bar_store_artifacts(path).root
