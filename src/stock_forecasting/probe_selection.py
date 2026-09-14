"""Read-only run selection for historical-scale probes, without model imports."""

from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from itertools import islice
from pathlib import Path
from typing import Any

from stock_forecasting.run_paths import validate_run_id
from stock_forecasting.runtime_resources import (
    detect_available_memory,
    detect_visible_cpu_count,
)

LOGGER = logging.getLogger(__name__)
COMPLETION_METADATA = Path("completion-result/training-result.json")
MAX_COMPLETION_METADATA_BYTES = 4 * 1024**2
MAX_SELECTION_WORKERS = 8
# Allow for decoded JSON objects and thread overhead, not just encoded file size.
SELECTION_WORKER_MEMORY_BYTES = 128 * 1024**2
SELECTION_BATCH_TIMEOUT_SECONDS = 120


def _completion_time(run: Path, schema_version: str) -> datetime | None:
    """Read only an atomically published completion marker; never inspect weights."""

    marker = run / COMPLETION_METADATA
    if marker.parent.is_symlink() or marker.is_symlink():
        raise ValueError(f"Completion metadata must not traverse symlinks: {marker}")
    try:
        with marker.open("rb") as stream:
            content = stream.read(MAX_COMPLETION_METADATA_BYTES + 1)
    except FileNotFoundError:
        return None
    if len(content) > MAX_COMPLETION_METADATA_BYTES:
        raise ValueError(f"Completion metadata exceeds the bounded read limit: {marker}")
    try:
        payload: Any = json.loads(content)
    except (ValueError, UnicodeError) as error:
        raise ValueError(f"Invalid completion metadata: {marker}") from error
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != schema_version
        or payload.get("kind") != "training-completion-result"
        or payload.get("run_id") != run.name
        or payload.get("run_key") != run.name
        or payload.get("stop_reason") not in ("epochs_completed", "early_stopping")
    ):
        raise ValueError(f"Invalid completion identity, schema, or stop reason: {marker}")
    created_at = payload.get("created_at")
    try:
        completed = datetime.fromisoformat(created_at) if isinstance(created_at, str) else None
    except ValueError as error:
        raise ValueError(f"Invalid completion timestamp: {marker}") from error
    if completed is None or completed.utcoffset() is None:
        raise ValueError(f"Completion timestamp must include a timezone: {marker}")
    return completed.astimezone(UTC)


def _selection_worker_count(requested: int | None) -> int:
    """Reserve at least 75% of available memory and cap network-volume I/O."""

    if requested is not None and (
        isinstance(requested, bool)
        or not isinstance(requested, int)
        or not 1 <= requested <= MAX_SELECTION_WORKERS
    ):
        raise ValueError(f"selection_workers must be between 1 and {MAX_SELECTION_WORKERS}")
    cpu_count = detect_visible_cpu_count()
    memory = detect_available_memory()
    memory_limit = memory.available_bytes // 4 // SELECTION_WORKER_MEMORY_BYTES
    if memory_limit < 1:
        raise MemoryError("Insufficient memory headroom for bounded completion-metadata scanning")
    workers = min(
        requested or MAX_SELECTION_WORKERS, MAX_SELECTION_WORKERS, cpu_count, memory_limit
    )
    LOGGER.info(
        "Completion scan: workers=%d, visible_cpus=%d, available_memory=%d (%s), "
        "estimated_memory_per_worker=%d, max_pending=%d",
        workers,
        cpu_count,
        memory.available_bytes,
        memory.source,
        SELECTION_WORKER_MEMORY_BYTES,
        workers,
    )
    if workers == 1:
        LOGGER.warning(
            "Completion scan uses one worker: requested=%s, CPU limit=%d, memory limit=%d",
            requested,
            cpu_count,
            memory_limit,
        )
    return workers


def latest_completed_probe_run(
    saved_model_root: Path,
    *,
    schema_version: str,
    selection_workers: int | None = None,
) -> Path:
    """Select by completion time, independently of the active lifecycle or directory mtime."""

    root = saved_model_root.expanduser().resolve(strict=False)
    workers = _selection_worker_count(selection_workers)
    newest: tuple[datetime, str] | None = None
    executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="probe-selection")
    try:
        with os.scandir(root) as entries:
            # Scandir and each submitted batch are bounded; no dataset or model is loaded.
            while batch := list(islice(entries, workers)):
                runs = []
                for entry in batch:
                    try:
                        validate_run_id(entry.name)
                    except ValueError:
                        continue
                    if entry.is_symlink():
                        raise ValueError(f"Saved-model runs must not be symlinks: {entry.path}")
                    if entry.is_dir(follow_symlinks=False):
                        runs.append(root / entry.name)
                futures = [executor.submit(_completion_time, run, schema_version) for run in runs]
                for run, future in zip(runs, futures, strict=True):
                    completed = future.result(timeout=SELECTION_BATCH_TIMEOUT_SECONDS)
                    if completed is not None:
                        candidate = (completed, run.name)
                        if newest is None or candidate > newest:
                            newest = candidate
    finally:
        # Propagate worker failures/timeouts and cancel unstarted reads before returning.
        executor.shutdown(wait=True, cancel_futures=True)
    if newest is None:
        raise FileNotFoundError(
            f"No completed training run found under {root}; a published "
            f"{COMPLETION_METADATA} is required. Use --checkpoint RUN_ID for an explicit run."
        )
    LOGGER.info(
        "Latest completed training run: %s (completed at %s)", newest[1], newest[0].isoformat()
    )
    return root / newest[1]


def probe_source_path(
    selector: str | Path | None,
    *,
    saved_model_root: Path,
    schema_version: str,
    selection_workers: int | None = None,
) -> Path:
    """Accept a run ID, an omitted selector, or a legacy canonical absolute path."""

    root = saved_model_root.expanduser().resolve(strict=False)
    if selector is None:
        source = latest_completed_probe_run(
            root, schema_version=schema_version, selection_workers=selection_workers
        )
    else:
        value = str(selector)
        candidate = Path(value)
        source = candidate if candidate.is_absolute() else root / validate_run_id(value)
    if source.resolve(strict=False) != source:
        raise ValueError("Probe selection must not traverse symlinks or noncanonical paths")
    if source.parent != root and source.parent.parent != root:
        raise ValueError("Probe selection must belong to the canonical saved-model root")
    return source
