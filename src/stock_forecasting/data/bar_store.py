"""Resumable symbol-oriented OHLCV storage for lazy training samples."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import uuid
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from multiprocessing import get_context
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import pandas as pd

from stock_forecasting.bar_store_integrity import validate_bar_store_artifacts
from stock_forecasting.data.adjustments import ensure_adjustment_columns
from stock_forecasting.data.benchmarks import resolve_benchmark
from stock_forecasting.data.content_identity import (
    bar_store_materialization_digest,
)
from stock_forecasting.data.manifest import (
    SUPPORTED_H_START,
    artifact_metadata,
    atomic_write_json,
    canonical_json_sha256,
    sha256_file,
)
from stock_forecasting.data.schema import (
    TRAINING_SECURITY_SCOPE,
    normalize_ohlcv_frame,
)
from stock_forecasting.data.splits import SPLIT_POLICY
from stock_forecasting.dataset_identity import (
    BAR_STORE_BUILD_CHECKPOINT_SCHEMA_VERSION,
    BAR_STORE_KIND,
    BAR_STORE_SCHEMA_VERSION,
    DEFAULT_DATASET_STORAGE_PREPARATION,
    PREPARATION_PROVENANCE_SCHEMA_VERSION,
)
from stock_forecasting.runtime_resources import (
    detect_available_memory,
    detect_visible_cpu_count,
)

DEFAULT_BUCKET_COUNT = 128
DEFAULT_BATCH_ROWS = 1_000_000
DEFAULT_WINDOW_SIZE = int(DEFAULT_DATASET_STORAGE_PREPARATION["window_size"])
DEFAULT_MAX_HORIZON = int(DEFAULT_DATASET_STORAGE_PREPARATION["max_horizon"])
DEFAULT_MAX_ABS_LOG_RETURN = float(
    DEFAULT_DATASET_STORAGE_PREPARATION["max_abs_log_return"]
)
DEFAULT_TRAIN_FRACTION = float(DEFAULT_DATASET_STORAGE_PREPARATION["train_fraction"])
DEFAULT_VALIDATION_FRACTION = float(
    DEFAULT_DATASET_STORAGE_PREPARATION["validation_fraction"]
)
DEFAULT_PURGE_BARS = int(DEFAULT_DATASET_STORAGE_PREPARATION["purge_bars"])
DEFAULT_EFFECTIVE_EMBARGO_BARS = int(
    DEFAULT_DATASET_STORAGE_PREPARATION["effective_embargo_bars"]
)
SYMBOL_BUCKET_ALGORITHM = "blake2b-64-modulo-v1"
RAW_SCAN_ALGORITHM = "partitioned-bucket-row-groups-v3"
SPLIT_ASSIGNMENT_ALGORITHM = "vectorized-bucket-checkpoints-v1"
MULTIPROCESS_BACKEND = "process_pool_spawn"
NATIVE_THREADS_PER_WORKER = 1
SOURCE_PARTITION_BATCH_MULTIPLIER = 4
RAW_BATCH_BYTES_PER_ROW_ESTIMATE = 1024
MIB = 1024**2
GIB = 1024**3
WORKER_BASE_MEMORY_BYTES = 512 * MIB
SAFE_MEMORY_FRACTION = 0.60
MINIMUM_PARENT_HEADROOM_BYTES = GIB
_NATIVE_THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "PYARROW_NUM_THREADS",
)


class PreparationPaused(RuntimeError):
    """Signal that all durable checkpoints are safe and another Pod may resume."""


class PreparationMemoryLimitExceeded(PreparationPaused):
    """Signal that the current Pod cannot safely run even one pending task."""


@dataclass(frozen=True)
class BarStoreBuildResult:
    """Paths and statistics produced by a complete bar-store build."""

    bar_store_manifest_path: Path
    symbol_index_path: Path
    cutoff_ranges_path: Path
    success_path: Path
    split_counts: dict[str, int]
    split_audit: dict[str, Any]
    quality: dict[str, Any]
    execution: dict[str, Any]


@dataclass(frozen=True)
class _PhasePlan:
    """Memory-bounded process count for one durable preparation phase."""

    phase: str
    requested_workers: int
    detected_cpu_count: int
    task_count: int
    pending_tasks: int
    reused_tasks: int
    worker_cap: int | None
    available_memory_bytes: int
    worker_memory_budget_bytes: int
    maximum_task_memory_bytes: int
    effective_workers: int
    estimated_peak_worker_memory_bytes: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend": MULTIPROCESS_BACKEND,
            "requested_workers": self.requested_workers,
            "detected_cpu_count": self.detected_cpu_count,
            "task_count": self.task_count,
            "pending_tasks": self.pending_tasks,
            "reused_tasks": self.reused_tasks,
            "worker_cap": self.worker_cap,
            "available_memory_bytes": self.available_memory_bytes,
            "worker_memory_budget_bytes": self.worker_memory_budget_bytes,
            "maximum_task_memory_bytes": self.maximum_task_memory_bytes,
            "effective_workers": self.effective_workers,
            "estimated_peak_worker_memory_bytes": (self.estimated_peak_worker_memory_bytes),
            "native_threads_per_worker": NATIVE_THREADS_PER_WORKER,
        }


@dataclass(frozen=True)
class _ScanPartitionTask:
    raw_path: str
    work_root: str
    partition: int
    source_row_groups: tuple[int, ...]
    row_count: int
    bucket_count: int
    batch_rows: int
    deadline_epoch_seconds: float | None


@dataclass(frozen=True)
class _BucketSource:
    partition: int
    path: str
    row_group: int
    rows: int
    compressed_bytes: int


@dataclass(frozen=True)
class _CompactBucketTask:
    output_root: str
    bucket: int
    bucket_count: int
    sources: tuple[_BucketSource, ...]
    benchmark_mapping: dict[str, str]
    max_abs_log_return: float
    deadline_epoch_seconds: float | None


@dataclass(frozen=True)
class _CandidateBucketTask:
    output_root: str
    work_root: str
    bucket: int
    planning_index_path: str
    window_size: int
    max_horizon: int
    deadline_epoch_seconds: float | None


@dataclass(frozen=True)
class _SplitBucketTask:
    output_root: str
    work_root: str
    candidate_part: str
    planning_index_path: str
    boundaries: dict[str, Any]
    max_horizon: int
    deadline_epoch_seconds: float | None


class _Deadline:
    def __init__(self, epoch_seconds: float | None) -> None:
        self.epoch_seconds = epoch_seconds

    def check(self, operation: str) -> None:
        if self.epoch_seconds is not None and time.time() >= self.epoch_seconds:
            raise PreparationPaused(
                f"Bar-store preparation paused before {operation}; durable checkpoints remain"
            )


def _safe_worker_memory_budget(
    available_memory_bytes: int,
    memory_budget_bytes: int | None,
) -> int:
    if available_memory_bytes < 1:
        raise ValueError("available_memory_bytes must be positive")
    if memory_budget_bytes is not None and memory_budget_bytes < 1:
        raise ValueError("memory_budget_bytes must be positive")

    proportional = int(available_memory_bytes * SAFE_MEMORY_FRACTION)
    reserved = min(
        MINIMUM_PARENT_HEADROOM_BYTES,
        max(available_memory_bytes // 3, 256 * MIB),
    )
    after_headroom = max(available_memory_bytes - reserved, 0)
    automatic = min(proportional, after_headroom)
    if memory_budget_bytes is not None:
        automatic = min(automatic, memory_budget_bytes)
    if automatic < 256 * MIB:
        raise PreparationMemoryLimitExceeded(
            "Available memory leaves less than 256 MiB for bar-store workers after "
            "reserving parent-process headroom; completed checkpoints remain reusable"
        )
    return automatic


def _plan_phase_workers(
    *,
    phase: str,
    requested_workers: int,
    detected_cpu_count: int,
    task_memory_bytes: Sequence[int],
    task_count: int,
    reused_tasks: int,
    available_memory_bytes: int,
    worker_memory_budget_bytes: int,
    worker_cap: int | None = None,
) -> _PhasePlan:
    """Choose a conservative worker count before allocating any child process."""

    if requested_workers < 1:
        raise ValueError("requested_workers must be positive")
    if detected_cpu_count < 1:
        raise ValueError("detected_cpu_count must be positive")
    if task_count < 0 or reused_tasks < 0 or reused_tasks > task_count:
        raise ValueError("phase task counters are invalid")
    if worker_cap is not None and worker_cap < 1:
        raise ValueError("worker_cap must be positive")
    if any(value < 1 for value in task_memory_bytes):
        raise ValueError("task memory estimates must be positive")

    pending_tasks = len(task_memory_bytes)
    if pending_tasks != task_count - reused_tasks:
        raise ValueError("pending task estimates do not match phase task counters")
    maximum_task_memory = max(task_memory_bytes, default=0)
    if maximum_task_memory > worker_memory_budget_bytes:
        raise PreparationMemoryLimitExceeded(
            f"{phase} requires an estimated {maximum_task_memory} bytes for one task, "
            f"but only {worker_memory_budget_bytes} safe worker bytes are available; "
            "completed checkpoints remain reusable"
        )

    effective_workers = 0
    if pending_tasks:
        memory_workers = worker_memory_budget_bytes // maximum_task_memory
        limits = [
            requested_workers,
            detected_cpu_count,
            pending_tasks,
            max(int(memory_workers), 1),
        ]
        if worker_cap is not None:
            limits.append(worker_cap)
        effective_workers = min(limits)

    return _PhasePlan(
        phase=phase,
        requested_workers=requested_workers,
        detected_cpu_count=detected_cpu_count,
        task_count=task_count,
        pending_tasks=pending_tasks,
        reused_tasks=reused_tasks,
        worker_cap=worker_cap,
        available_memory_bytes=available_memory_bytes,
        worker_memory_budget_bytes=worker_memory_budget_bytes,
        maximum_task_memory_bytes=maximum_task_memory,
        effective_workers=effective_workers,
        estimated_peak_worker_memory_bytes=(effective_workers * maximum_task_memory),
    )


@contextmanager
def _single_threaded_child_environment() -> Any:
    """Prevent native BLAS/Arrow pools from multiplying every process worker."""

    previous = {name: os.environ.get(name) for name in _NATIVE_THREAD_ENVIRONMENT}
    try:
        for name in _NATIVE_THREAD_ENVIRONMENT:
            os.environ[name] = str(NATIVE_THREADS_PER_WORKER)
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _initialize_process_worker() -> None:
    """Apply per-process native thread limits after the spawned interpreter starts."""

    for name in _NATIVE_THREAD_ENVIRONMENT:
        os.environ[name] = str(NATIVE_THREADS_PER_WORKER)
    try:
        import pyarrow as pa
    except ImportError:
        return
    pa.set_cpu_count(NATIVE_THREADS_PER_WORKER)
    if hasattr(pa, "set_io_thread_count"):
        pa.set_io_thread_count(NATIVE_THREADS_PER_WORKER)


@contextmanager
def _single_threaded_arrow_runtime() -> Any:
    """Temporarily cap Arrow threads when a phase runs in the parent process."""

    try:
        import pyarrow as pa
    except ImportError:
        yield
        return
    original_cpu_count = pa.cpu_count()
    original_io_thread_count = pa.io_thread_count() if hasattr(pa, "io_thread_count") else None
    try:
        pa.set_cpu_count(NATIVE_THREADS_PER_WORKER)
        if hasattr(pa, "set_io_thread_count"):
            pa.set_io_thread_count(NATIVE_THREADS_PER_WORKER)
        yield
    finally:
        pa.set_cpu_count(original_cpu_count)
        if original_io_thread_count is not None and hasattr(pa, "set_io_thread_count"):
            pa.set_io_thread_count(original_io_thread_count)


def _run_phase_tasks(
    *,
    plan: _PhasePlan,
    tasks: Sequence[Any],
    runner: Callable[[Any], Any],
    deadline: _Deadline,
    on_result: Callable[[Any], None] | None = None,
) -> None:
    """Run bounded spawn workers while allowing only one in-flight task per worker."""

    if len(tasks) != plan.pending_tasks:
        raise ValueError("phase task list does not match its execution plan")
    if not tasks:
        return
    if plan.effective_workers == 1:
        with (
            _single_threaded_child_environment(),
            _single_threaded_arrow_runtime(),
        ):
            for task_index, task in enumerate(tasks):
                deadline.check(f"{plan.phase} task {task_index}")
                result = runner(task)
                if on_result is not None:
                    on_result(result)
        return

    task_iterator = iter(tasks)
    futures: dict[Any, int] = {}
    try:
        with (
            _single_threaded_child_environment(),
            ProcessPoolExecutor(
                max_workers=plan.effective_workers,
                mp_context=get_context("spawn"),
                initializer=_initialize_process_worker,
            ) as executor,
        ):
            for task_index in range(plan.effective_workers):
                deadline.check(f"{plan.phase} task {task_index} submission")
                futures[executor.submit(runner, next(task_iterator))] = task_index

            next_task_index = plan.effective_workers
            while futures:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    futures.pop(future)
                    result = future.result()
                    if on_result is not None:
                        on_result(result)
                    try:
                        next_task = next(task_iterator)
                    except StopIteration:
                        continue
                    deadline.check(f"{plan.phase} task {next_task_index} submission")
                    futures[executor.submit(runner, next_task)] = next_task_index
                    next_task_index += 1
    except BrokenProcessPool as error:
        raise RuntimeError(
            f"{plan.phase} process worker exited unexpectedly, possibly because of "
            "an external OOM kill; completed checkpoints remain reusable"
        ) from error


def _initialize_execution_plan(
    *,
    work_root: Path,
    requested_workers: int,
    detected_cpu_count: int,
    available_memory_bytes: int,
    worker_memory_budget_bytes: int,
    memory_budget_override_bytes: int | None,
) -> None:
    atomic_write_json(
        work_root / "execution-plan.json",
        {
            "schema_version": 1,
            "kind": "bar-store-execution-plan",
            "backend": MULTIPROCESS_BACKEND,
            "requested_workers": requested_workers,
            "detected_cpu_count": detected_cpu_count,
            "available_memory_bytes_at_planning": available_memory_bytes,
            "worker_memory_budget_bytes": worker_memory_budget_bytes,
            "memory_budget_override_bytes": memory_budget_override_bytes,
            "safe_memory_fraction": SAFE_MEMORY_FRACTION,
            "minimum_parent_headroom_bytes": MINIMUM_PARENT_HEADROOM_BYTES,
            "native_threads_per_worker": NATIVE_THREADS_PER_WORKER,
            "phases": {},
        },
    )


def _record_phase_plan(work_root: Path, plan: _PhasePlan) -> None:
    path = work_root / "execution-plan.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    phases = payload.setdefault("phases", {})
    if not isinstance(phases, dict):
        raise ValueError("Bar-store execution plan has invalid phase metadata")
    phases[plan.phase] = plan.as_dict()
    atomic_write_json(path, payload)


def _emit_phase_progress(
    *,
    phase: str,
    state: str,
    completed_tasks: int,
    total_tasks: int,
    effective_workers: int,
) -> None:
    print(
        json.dumps(
            {
                "bar_store_phase": phase,
                "completed_tasks": completed_tasks,
                "effective_workers": effective_workers,
                "state": state,
                "total_tasks": total_tasks,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _require_pyarrow() -> tuple[Any, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("Building the lazy bar store requires pyarrow") from error
    return pa, pq


def _symbol_bucket(symbol: str, bucket_count: int) -> int:
    digest = hashlib.blake2b(symbol.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False) % bucket_count


def _atomic_parquet(frame: pd.DataFrame, destination: Path) -> None:
    pa, pq = _require_pyarrow()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
    try:
        table = pa.Table.from_pandas(frame, preserve_index=False)
        pq.write_table(
            table,
            temporary,
            compression="zstd",
            use_dictionary=True,
            write_statistics=True,
        )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_directory(staging: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to overwrite immutable build checkpoint: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging.replace(destination)


def _staging_directory(parent: Path, name: str) -> Path:
    staging = parent / f".{name}.staging-{uuid.uuid4().hex}"
    staging.mkdir(parents=True, exist_ok=False)
    return staging


def _discard_stale_staging(parent: Path, name: str) -> None:
    for stale in parent.glob(f".{name}.staging-*"):
        _remove_generated_tree(stale)


def _remove_generated_tree(path: Path) -> None:
    """Remove only a build-owned tree without following symbolic links."""

    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
        return
    if not path.is_dir():
        return
    for child in path.iterdir():
        _remove_generated_tree(child)
    path.rmdir()


def _cleanup_completed_work(work_root: Path) -> None:
    """Reclaim resumability artifacts only after immutable outputs are complete."""

    _load_planning_symbol_index.cache_clear()
    generated_names = (
        "scan-partitions",
        "segments",
        "compaction-staging",
        "candidate-staging",
        "candidates",
        "planning",
        "split-staging",
        "split-ranges",
    )
    for name in generated_names:
        _remove_generated_tree(work_root / name)
    (work_root / "build-state.json").unlink(missing_ok=True)
    (work_root / "execution-plan.json").unlink(missing_ok=True)
    (work_root / "scan-index.json").unlink(missing_ok=True)


def _quarantine_obsolete_bar_store(
    *,
    root: Path,
    previous_state: Mapping[str, Any],
) -> Path:
    """Move an incompatible generated store aside without deleting raw data."""

    if root.is_symlink() or not root.is_dir():
        raise ValueError("Bar-store checkpoint root must be a regular directory")
    previous_sha256 = str(
        previous_state.get(
            "checkpoint_identity_sha256",
            previous_state.get("identity_sha256", "unknown"),
        )
    )[:12]
    quarantine = root.parent / (
        f".{root.name}-obsolete-{previous_sha256}-{uuid.uuid4().hex[:8]}"
    )
    root.replace(quarantine)
    return quarantine


def _reset_generated_bar_store(
    *,
    root: Path,
    previous_state: Mapping[str, Any],
) -> tuple[Path, Path]:
    """Quarantine only derived outputs and recreate an empty build workspace."""

    quarantine = _quarantine_obsolete_bar_store(
        root=root,
        previous_state=previous_state,
    )
    root.mkdir(parents=False, exist_ok=False)
    work_root = root / ".work"
    work_root.mkdir(parents=False, exist_ok=False)
    return work_root, quarantine


def _completed_bar_store_result(
    *,
    work_root: Path,
    success_path: Path,
    manifest_path: Path,
    index_path: Path,
    ranges_path: Path,
    identity_sha256: str,
) -> BarStoreBuildResult:
    """Validate finalized metadata before ignoring execution-only checkpoints."""

    integrity = validate_bar_store_artifacts(
        manifest_path.parent,
        expected_identity_sha256=identity_sha256,
    )
    manifest = integrity.manifest
    _cleanup_completed_work(work_root)
    return BarStoreBuildResult(
        bar_store_manifest_path=manifest_path,
        symbol_index_path=index_path,
        cutoff_ranges_path=ranges_path,
        success_path=success_path,
        split_counts={str(k): int(v) for k, v in manifest["split_counts"].items()},
        split_audit=dict(manifest["split_audit"]),
        quality=dict(manifest["quality"]),
        execution=dict(manifest["execution"]),
    )


def _recover_bar_store_success_marker(
    *,
    success_path: Path,
    manifest_path: Path,
    index_path: Path,
    ranges_path: Path,
    identity_sha256: str,
) -> None:
    """Publish only the final sentinel after validating interrupted outputs."""

    integrity = validate_bar_store_artifacts(
        manifest_path.parent,
        expected_identity_sha256=identity_sha256,
        require_success=False,
    )
    manifest = integrity.manifest
    atomic_write_json(
        success_path,
        {
            "schema_version": BAR_STORE_SCHEMA_VERSION,
            "kind": "bar-store-success",
            "state": "ready",
            "identity_sha256": identity_sha256,
            "bar_store_manifest_sha256": integrity.manifest_sha256,
            "symbol_index_sha256": integrity.symbol_index_sha256,
            "cutoff_ranges_sha256": integrity.cutoff_ranges_sha256,
            "split_counts": manifest["split_counts"],
            "recovered_after_interrupted_publication": True,
        },
    )


def _metadata_value(frame: pd.DataFrame, column: str) -> Any:
    if column not in frame:
        return None
    values = frame[column].dropna().unique().tolist()
    if len(values) != 1:
        return None
    value = values[0]
    return bool(value) if column == "is_active" else str(value)


def _read_symbol(index_row: Mapping[str, Any], root: Path) -> pd.DataFrame:
    _, pq = _require_pyarrow()
    shard = root / str(index_row["shard_relative_path"])
    row_group = int(index_row["row_group"])
    frame = pq.ParquetFile(shard).read_row_group(row_group).to_pandas()
    return _ordered_symbol_rows(frame)


def _ordered_symbol_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize the row order used by every lazy window and label lookup."""

    ordered = frame.copy()
    ordered["timestamp"] = pd.to_datetime(ordered["timestamp"], utc=True)
    return ordered.sort_values("timestamp", kind="stable").reset_index(drop=True)


def _compacted_symbol_rows(
    frame: pd.DataFrame,
) -> Iterable[tuple[str, pd.DataFrame]]:
    """Yield the logical per-symbol row groups written to the immutable store."""

    for symbol, rows in frame.groupby("symbol", sort=True):
        yield str(symbol), _ordered_symbol_rows(rows)


def _true_ranges(mask: np.ndarray, *, offset: int = 0) -> list[tuple[int, int]]:
    positions = np.flatnonzero(mask)
    if positions.size == 0:
        return []
    boundaries = np.flatnonzero(np.diff(positions) != 1) + 1
    groups = np.split(positions, boundaries)
    return [(int(group[0]) + offset, int(group[-1]) + offset + 1) for group in groups]


def _source_partition_layouts(
    parquet: Any,
    *,
    target_rows: int,
) -> list[tuple[int, tuple[int, ...], int]]:
    """Group tiny source row groups into a small number of stable scan partitions."""

    layouts: list[tuple[int, tuple[int, ...], int]] = []
    pending_row_groups: list[int] = []
    pending_rows = 0
    for row_group in range(parquet.num_row_groups):
        row_group_rows = int(parquet.metadata.row_group(row_group).num_rows)
        if pending_row_groups and pending_rows + row_group_rows > target_rows:
            layouts.append((len(layouts), tuple(pending_row_groups), pending_rows))
            pending_row_groups = []
            pending_rows = 0
        pending_row_groups.append(row_group)
        pending_rows += row_group_rows
    if pending_row_groups:
        layouts.append((len(layouts), tuple(pending_row_groups), pending_rows))
    if not layouts:
        raise ValueError("Raw Parquet contains no source row groups")
    return layouts


def _iter_partition_tables(
    parquet: Any,
    *,
    row_groups: tuple[int, ...],
    batch_rows: int,
) -> Any:
    """Yield bounded Arrow tables even when the source contains tiny row groups."""

    pa, _ = _require_pyarrow()
    pending: list[Any] = []
    pending_rows = 0
    for batch in parquet.iter_batches(batch_size=batch_rows, row_groups=list(row_groups)):
        offset = 0
        while offset < batch.num_rows:
            take = min(batch_rows - pending_rows, batch.num_rows - offset)
            pending.append(batch.slice(offset, take))
            pending_rows += take
            offset += take
            if pending_rows == batch_rows:
                yield pa.Table.from_batches(pending)
                pending = []
                pending_rows = 0
    if pending:
        yield pa.Table.from_batches(pending)


def _scan_partition_name(partition: int) -> str:
    return f"partition-{partition:04d}"


def _validated_scan_partition(
    task: _ScanPartitionTask,
) -> dict[str, Any] | None:
    """Validate one atomic partition with one footer read and no directory glob."""

    _, pq = _require_pyarrow()
    partition_name = _scan_partition_name(task.partition)
    partition_root = Path(task.work_root) / "scan-partitions" / partition_name
    checkpoint_path = partition_root / "checkpoint.json"
    artifact_path = partition_root / "partition.parquet"
    if not checkpoint_path.is_file() or not artifact_path.is_file():
        return None
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    artifact = checkpoint.get("artifact")
    buckets = checkpoint.get("buckets")
    if (
        checkpoint.get("schema_version") != BAR_STORE_SCHEMA_VERSION
        or checkpoint.get("kind") != "bar-store-scan-partition"
        or checkpoint.get("algorithm") != RAW_SCAN_ALGORITHM
        or checkpoint.get("partition") != task.partition
        or checkpoint.get("source_row_groups") != list(task.source_row_groups)
        or checkpoint.get("rows") != task.row_count
        or checkpoint.get("batch_rows") != task.batch_rows
        or checkpoint.get("bucket_count") != task.bucket_count
        or not isinstance(artifact, dict)
        or artifact.get("relative_path")
        != (Path("scan-partitions") / partition_name / "partition.parquet").as_posix()
        or not isinstance(artifact.get("sha256"), str)
        or len(artifact["sha256"]) != 64
        or any(character not in "0123456789abcdef" for character in artifact["sha256"])
        or not isinstance(artifact.get("size_bytes"), int)
        or isinstance(artifact.get("size_bytes"), bool)
        or artifact["size_bytes"] < 1
        or artifact_path.stat().st_size != artifact["size_bytes"]
        or not isinstance(buckets, dict)
        or not buckets
    ):
        raise ValueError(f"Raw-scan partition checkpoint is invalid: {checkpoint_path}")

    row_groups: list[int] = []
    bucket_rows = 0
    parsed_buckets: dict[int, dict[str, int]] = {}
    for bucket_text, metadata in buckets.items():
        try:
            bucket = int(bucket_text)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Raw-scan partition bucket is invalid: {checkpoint_path}") from error
        if (
            str(bucket) != bucket_text
            or bucket < 0
            or bucket >= task.bucket_count
            or not isinstance(metadata, dict)
            or not isinstance(metadata.get("row_group"), int)
            or isinstance(metadata.get("row_group"), bool)
            or metadata["row_group"] < 0
            or not isinstance(metadata.get("rows"), int)
            or isinstance(metadata.get("rows"), bool)
            or metadata["rows"] < 1
            or not isinstance(metadata.get("compressed_bytes"), int)
            or isinstance(metadata.get("compressed_bytes"), bool)
            or metadata["compressed_bytes"] < 1
        ):
            raise ValueError(f"Raw-scan partition bucket is invalid: {checkpoint_path}")
        row_groups.append(metadata["row_group"])
        bucket_rows += metadata["rows"]
        parsed_buckets[bucket] = metadata
    if sorted(row_groups) != list(range(len(row_groups))) or bucket_rows != task.row_count:
        raise ValueError(f"Raw-scan partition totals are invalid: {checkpoint_path}")

    parquet = pq.ParquetFile(artifact_path)
    if parquet.num_row_groups != len(parsed_buckets) or parquet.metadata.num_rows != task.row_count:
        raise ValueError(f"Raw-scan partition Parquet is invalid: {artifact_path}")
    for metadata in parsed_buckets.values():
        if parquet.metadata.row_group(metadata["row_group"]).num_rows != metadata["rows"]:
            raise ValueError(f"Raw-scan partition row group is invalid: {artifact_path}")
    return checkpoint


def _scan_partition_task(task: _ScanPartitionTask) -> dict[str, Any]:
    """Stream one coarse source partition and publish one bucket-indexed Parquet."""

    pa, pq = _require_pyarrow()
    completed = _validated_scan_partition(task)
    if completed is not None:
        return completed

    deadline = _Deadline(task.deadline_epoch_seconds)
    work_root = Path(task.work_root)
    partitions_root = work_root / "scan-partitions"
    partitions_root.mkdir(parents=True, exist_ok=True)
    partition_name = _scan_partition_name(task.partition)
    destination = partitions_root / partition_name
    if destination.exists() or destination.is_symlink():
        _remove_generated_tree(destination)
    _discard_stale_staging(partitions_root, partition_name)

    with TemporaryDirectory(prefix=f"fin-ts-{partition_name}-", dir="/tmp") as local_text:
        local_root = Path(local_text)
        parquet = pq.ParquetFile(task.raw_path)
        bucket_parts: dict[int, list[Path]] = defaultdict(list)
        observed_rows = 0
        for batch_index, table in enumerate(
            _iter_partition_tables(
                parquet,
                row_groups=task.source_row_groups,
                batch_rows=task.batch_rows,
            )
        ):
            deadline.check(f"raw scan partition {task.partition} batch {batch_index}")
            frame = ensure_adjustment_columns(normalize_ohlcv_frame(table.to_pandas()))
            observed_rows += len(frame)
            symbol_buckets = {
                str(symbol): _symbol_bucket(str(symbol), task.bucket_count)
                for symbol in frame["symbol"].unique()
            }
            bucket_ids = frame["symbol"].map(symbol_buckets).to_numpy(dtype=np.int64)
            for bucket in sorted(set(int(value) for value in bucket_ids)):
                selected = frame.loc[bucket_ids == bucket].reset_index(drop=True)
                part_path = local_root / f"batch-{batch_index:04d}-bucket-{bucket:04d}.parquet"
                _atomic_parquet(selected, part_path)
                bucket_parts[bucket].append(part_path)
        if observed_rows != task.row_count:
            raise RuntimeError(
                f"Raw scan partition {task.partition} yielded {observed_rows} rows; "
                f"expected {task.row_count}"
            )

        local_artifact = local_root / "partition.parquet"
        writer: Any = None
        bucket_metadata: dict[str, dict[str, int]] = {}
        try:
            for output_row_group, bucket in enumerate(sorted(bucket_parts)):
                tables = [pq.read_table(path) for path in bucket_parts[bucket]]
                bucket_table = pa.concat_tables(tables, promote_options="default")
                if writer is None:
                    writer = pq.ParquetWriter(
                        local_artifact,
                        bucket_table.schema,
                        compression="zstd",
                        use_dictionary=True,
                        write_statistics=True,
                    )
                elif bucket_table.schema != writer.schema:
                    bucket_table = bucket_table.cast(writer.schema)
                writer.write_table(bucket_table, row_group_size=bucket_table.num_rows)
                bucket_metadata[str(bucket)] = {
                    "row_group": output_row_group,
                    "rows": bucket_table.num_rows,
                    "compressed_bytes": 0,
                }
        finally:
            if writer is not None:
                writer.close()
        if writer is None:
            raise RuntimeError(f"Raw scan partition {task.partition} produced no buckets")

        local_parquet = pq.ParquetFile(local_artifact)
        for metadata in bucket_metadata.values():
            row_group_metadata = local_parquet.metadata.row_group(metadata["row_group"])
            metadata["compressed_bytes"] = max(
                sum(
                    int(row_group_metadata.column(column).total_compressed_size)
                    for column in range(row_group_metadata.num_columns)
                ),
                1,
            )
        artifact_sha256 = sha256_file(local_artifact)
        artifact_size = local_artifact.stat().st_size
        deadline.check(f"raw scan partition {task.partition} publication")
        staging = _staging_directory(partitions_root, partition_name)
        try:
            published_artifact = staging / "partition.parquet"
            shutil.copyfile(local_artifact, published_artifact)
            if published_artifact.stat().st_size != artifact_size:
                raise RuntimeError("Raw-scan partition copy changed artifact size")
            checkpoint = {
                "schema_version": BAR_STORE_SCHEMA_VERSION,
                "kind": "bar-store-scan-partition",
                "algorithm": RAW_SCAN_ALGORITHM,
                "partition": task.partition,
                "source_row_groups": list(task.source_row_groups),
                "rows": task.row_count,
                "batch_rows": task.batch_rows,
                "bucket_count": task.bucket_count,
                "artifact": {
                    "relative_path": (
                        Path("scan-partitions") / partition_name / "partition.parquet"
                    ).as_posix(),
                    "sha256": artifact_sha256,
                    "size_bytes": artifact_size,
                },
                "buckets": bucket_metadata,
            }
            atomic_write_json(staging / "checkpoint.json", checkpoint)
            _atomic_directory(staging, destination)
        except Exception:
            _remove_generated_tree(staging)
            raise
    return checkpoint


def _scan_index_payload(
    *,
    state: str,
    raw_rows: int,
    source_row_groups: int,
    target_partition_rows: int,
    bucket_count: int,
    batch_rows: int,
    partition_count: int,
    completed: Mapping[int, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": BAR_STORE_SCHEMA_VERSION,
        "kind": "bar-store-scan-index",
        "state": state,
        "algorithm": RAW_SCAN_ALGORITHM,
        "raw_rows": raw_rows,
        "source_row_groups": source_row_groups,
        "target_partition_rows": target_partition_rows,
        "bucket_count": bucket_count,
        "batch_rows": batch_rows,
        "partition_count": partition_count,
        "completed_partitions": [completed[index] for index in sorted(completed)],
    }


def _write_scan_partitions(
    *,
    raw_path: Path,
    work_root: Path,
    bucket_count: int,
    batch_rows: int,
    deadline: _Deadline,
    requested_workers: int,
    detected_cpu_count: int,
    available_memory_bytes: int,
    worker_memory_budget_bytes: int,
) -> tuple[dict[str, Any], _PhasePlan]:
    """Build a few resumable bucket-indexed partitions instead of tiny segments."""

    _, pq = _require_pyarrow()
    partitions_root = work_root / "scan-partitions"
    partitions_root.mkdir(parents=True, exist_ok=True)
    parquet = pq.ParquetFile(raw_path)
    target_partition_rows = batch_rows * SOURCE_PARTITION_BATCH_MULTIPLIER
    layouts = _source_partition_layouts(parquet, target_rows=target_partition_rows)
    tasks = [
        _ScanPartitionTask(
            raw_path=str(raw_path),
            work_root=str(work_root),
            partition=partition,
            source_row_groups=row_groups,
            row_count=rows,
            bucket_count=bucket_count,
            batch_rows=batch_rows,
            deadline_epoch_seconds=deadline.epoch_seconds,
        )
        for partition, row_groups, rows in layouts
    ]
    completed: dict[int, dict[str, Any]] = {}
    pending: list[_ScanPartitionTask] = []
    for task in tasks:
        checkpoint = _validated_scan_partition(task)
        if checkpoint is None:
            pending.append(task)
        else:
            completed[task.partition] = checkpoint

    index_path = work_root / "scan-index.json"

    def publish_index(state: str) -> dict[str, Any]:
        payload = _scan_index_payload(
            state=state,
            raw_rows=int(parquet.metadata.num_rows),
            source_row_groups=parquet.num_row_groups,
            target_partition_rows=target_partition_rows,
            bucket_count=bucket_count,
            batch_rows=batch_rows,
            partition_count=len(tasks),
            completed=completed,
        )
        atomic_write_json(index_path, payload)
        return payload

    publish_index("building")
    maximum_batch_memory = WORKER_BASE_MEMORY_BYTES + max(
        batch_rows * RAW_BATCH_BYTES_PER_ROW_ESTIMATE,
        max(
            int(parquet.metadata.row_group(row_group).total_byte_size) * 6
            for task in pending
            for row_group in task.source_row_groups
        )
        if pending
        else 1,
    )
    plan = _plan_phase_workers(
        phase="raw_scan",
        requested_workers=requested_workers,
        detected_cpu_count=detected_cpu_count,
        task_memory_bytes=[maximum_batch_memory] * len(pending),
        task_count=len(tasks),
        reused_tasks=len(completed),
        available_memory_bytes=available_memory_bytes,
        worker_memory_budget_bytes=worker_memory_budget_bytes,
    )
    _record_phase_plan(work_root, plan)
    _emit_phase_progress(
        phase=plan.phase,
        state="running" if pending else "complete",
        completed_tasks=len(completed),
        total_tasks=len(tasks),
        effective_workers=plan.effective_workers,
    )

    def record_completed(checkpoint: dict[str, Any]) -> None:
        partition = int(checkpoint["partition"])
        completed[partition] = checkpoint
        publish_index("building")
        _emit_phase_progress(
            phase=plan.phase,
            state="running" if len(completed) < len(tasks) else "complete",
            completed_tasks=len(completed),
            total_tasks=len(tasks),
            effective_workers=plan.effective_workers,
        )

    _run_phase_tasks(
        plan=plan,
        tasks=pending,
        runner=_scan_partition_task,
        deadline=deadline,
        on_result=record_completed,
    )
    if len(completed) != len(tasks):
        raise RuntimeError("Raw scan did not complete every source partition")
    validated_rows = sum(int(checkpoint["rows"]) for checkpoint in completed.values())
    if validated_rows != parquet.metadata.num_rows:
        raise RuntimeError(
            f"Raw scan observed {validated_rows} rows; expected {parquet.metadata.num_rows}"
        )
    deadline.check("raw scan index publication")
    return publish_index("complete"), plan


def _bucket_sources_from_scan_index(
    *,
    work_root: Path,
    scan_index: Mapping[str, Any],
    bucket_count: int,
) -> dict[int, tuple[_BucketSource, ...]]:
    """Resolve exact Parquet row groups without listing partition directories."""

    completed = scan_index.get("completed_partitions")
    if (
        scan_index.get("schema_version") != BAR_STORE_SCHEMA_VERSION
        or scan_index.get("kind") != "bar-store-scan-index"
        or scan_index.get("state") != "complete"
        or scan_index.get("algorithm") != RAW_SCAN_ALGORITHM
        or scan_index.get("bucket_count") != bucket_count
        or not isinstance(scan_index.get("partition_count"), int)
        or isinstance(scan_index.get("partition_count"), bool)
        or not isinstance(completed, list)
        or len(completed) != scan_index["partition_count"]
    ):
        raise ValueError("Raw-scan index is incomplete or incompatible")

    sources: dict[int, list[_BucketSource]] = defaultdict(list)
    observed_partitions: set[int] = set()
    for checkpoint in completed:
        if not isinstance(checkpoint, dict):
            raise ValueError("Raw-scan index contains an invalid partition checkpoint")
        partition = checkpoint.get("partition")
        buckets = checkpoint.get("buckets")
        if (
            not isinstance(partition, int)
            or isinstance(partition, bool)
            or partition < 0
            or partition in observed_partitions
            or not isinstance(buckets, dict)
        ):
            raise ValueError("Raw-scan index contains an invalid partition checkpoint")
        observed_partitions.add(partition)
        artifact_path = (
            work_root / "scan-partitions" / _scan_partition_name(partition) / "partition.parquet"
        )
        if not artifact_path.is_file():
            raise ValueError(f"Raw-scan partition artifact is missing: {artifact_path}")
        for bucket_text, metadata in buckets.items():
            if not isinstance(metadata, dict):
                raise ValueError("Raw-scan index contains invalid bucket metadata")
            try:
                bucket = int(bucket_text)
            except (TypeError, ValueError) as error:
                raise ValueError("Raw-scan index contains invalid bucket metadata") from error
            row_group = metadata.get("row_group")
            rows = metadata.get("rows")
            compressed_bytes = metadata.get("compressed_bytes")
            if (
                str(bucket) != bucket_text
                or bucket < 0
                or bucket >= bucket_count
                or not isinstance(row_group, int)
                or isinstance(row_group, bool)
                or row_group < 0
                or not isinstance(rows, int)
                or isinstance(rows, bool)
                or rows < 1
                or not isinstance(compressed_bytes, int)
                or isinstance(compressed_bytes, bool)
                or compressed_bytes < 1
            ):
                raise ValueError("Raw-scan index contains invalid bucket metadata")
            sources[bucket].append(
                _BucketSource(
                    partition=partition,
                    path=str(artifact_path),
                    row_group=row_group,
                    rows=rows,
                    compressed_bytes=compressed_bytes,
                )
            )
    if observed_partitions != set(range(scan_index["partition_count"])):
        raise ValueError("Raw-scan index partition sequence is incomplete")
    return {
        bucket: tuple(sorted(bucket_sources, key=lambda source: source.partition))
        for bucket, bucket_sources in sources.items()
    }


def _validated_compacted_bucket(
    destination: Path,
    *,
    bucket: int,
    bucket_count: int,
    sources: tuple[_BucketSource, ...],
) -> bool:
    checkpoint_path = destination / "checkpoint.json"
    if not destination.exists() and not destination.is_symlink():
        return False
    if not checkpoint_path.is_file():
        _remove_generated_tree(destination)
        return False
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    source_partitions = [source.partition for source in sources]
    expected_rows = sum(source.rows for source in sources)
    shard_path = destination / "shard.parquet"
    if (
        checkpoint.get("schema_version") != BAR_STORE_SCHEMA_VERSION
        or checkpoint.get("kind") != "bar-store-compacted-bucket"
        or checkpoint.get("source_scan_algorithm") != RAW_SCAN_ALGORITHM
        or checkpoint.get("source_partitions") != source_partitions
        or checkpoint.get("bucket") != bucket
        or checkpoint.get("bucket_count") != bucket_count
        or checkpoint.get("rows") != expected_rows
        or not isinstance(checkpoint.get("symbols"), int)
        or isinstance(checkpoint.get("symbols"), bool)
        or checkpoint["symbols"] < 1
        or not isinstance(checkpoint.get("shard_size_bytes"), int)
        or isinstance(checkpoint.get("shard_size_bytes"), bool)
        or checkpoint["shard_size_bytes"] < 1
        or not shard_path.is_file()
        or shard_path.stat().st_size != checkpoint["shard_size_bytes"]
        or not (destination / "symbol-index.parquet").is_file()
        or not (destination / "observed-dates.parquet").is_file()
    ):
        raise ValueError(f"Compaction checkpoint is invalid: {checkpoint_path}")
    return True


def _enrich_compacted_frame(
    frame: pd.DataFrame,
    *,
    max_abs_log_return: float,
) -> pd.DataFrame:
    """Apply content-affecting cleaning fields independently of checkpoint I/O."""

    enriched = ensure_adjustment_columns(normalize_ohlcv_frame(frame))
    adjusted_close = enriched["adjusted_close"].to_numpy(dtype=np.float64)
    symbols = enriched["symbol"].astype(str).to_numpy()
    timestamps = pd.DatetimeIndex(enriched["timestamp"])
    transitions = np.zeros(len(enriched), dtype=bool)
    calendar_gap_days = np.zeros(len(enriched), dtype=np.int32)
    same_symbol = symbols[1:] == symbols[:-1]
    log_returns = np.zeros(max(len(enriched) - 1, 0), dtype=np.float64)
    if len(enriched) > 1:
        log_returns[same_symbol] = np.diff(
            np.log(np.maximum(adjusted_close, 1e-12))
        )[same_symbol]
        transitions[1:] = same_symbol & (np.abs(log_returns) > max_abs_log_return)
        timestamp_days = timestamps.asi8 // (24 * 60 * 60 * 1_000_000_000)
        calendar_gap_days[1:] = np.where(
            same_symbol,
            np.maximum(np.diff(timestamp_days), 0),
            0,
        ).astype(np.int32)
    enriched["adjusted_transition_extreme"] = transitions
    enriched["calendar_gap_days"] = calendar_gap_days
    return enriched


def _symbol_index_metadata(
    ordered: pd.DataFrame,
    *,
    symbol: str,
    benchmark_mapping: Mapping[str, str],
) -> dict[str, Any]:
    """Return content-facing eligibility and audit metadata for one symbol."""

    asset_type = _metadata_value(ordered, "asset_type")
    market = _metadata_value(ordered, "market")
    metadata_consistent = asset_type is not None and market is not None
    decision = resolve_benchmark(
        symbol=symbol,
        asset_type=str(asset_type or ""),
        market=str(market or ""),
        explicit_mapping=dict(benchmark_mapping),
    )
    return {
        "start_at": pd.Timestamp(ordered["timestamp"].iloc[0]).isoformat(),
        "end_at": pd.Timestamp(ordered["timestamp"].iloc[-1]).isoformat(),
        "asset_type": str(asset_type or "unknown"),
        "market": str(market or "unknown"),
        "provider": str(_metadata_value(ordered, "provider") or "unknown"),
        "currency": str(_metadata_value(ordered, "currency") or "unknown"),
        "source_symbol": str(_metadata_value(ordered, "source_symbol") or symbol),
        "is_active": _metadata_value(ordered, "is_active"),
        "dataset_profile": str(_metadata_value(ordered, "dataset_profile") or "unknown"),
        "eligible": bool(metadata_consistent and decision.eligible),
        "eligibility_reason": (
            decision.reason if metadata_consistent else "inconsistent_symbol_metadata"
        ),
        "benchmark_symbol": decision.benchmark_symbol or "",
        "benchmark_policy": decision.policy,
        "extreme_transition_count": int(ordered["adjusted_transition_extreme"].sum()),
        "long_calendar_gap_count": int((ordered["calendar_gap_days"] > 10).sum()),
    }


def _exclude_symbols_with_missing_benchmarks(index: pd.DataFrame) -> pd.DataFrame:
    """Disable targets whose benchmark series is absent from the same bar store."""

    resolved = index.copy()
    available_symbols = set(resolved["symbol"].astype(str))
    missing = resolved["eligible"].astype(bool) & ~resolved["benchmark_symbol"].isin(
        available_symbols
    )
    resolved.loc[missing, "eligible"] = False
    resolved.loc[missing, "eligibility_reason"] = "missing_benchmark"
    return resolved


def _compact_bucket(
    *,
    output_root: Path,
    bucket: int,
    bucket_count: int,
    sources: tuple[_BucketSource, ...],
    benchmark_mapping: Mapping[str, str],
    max_abs_log_return: float,
    deadline: _Deadline,
) -> None:
    """Sort one bounded bucket and write one row group per symbol."""

    deadline.check(f"bucket {bucket} compaction")
    bucket_name = f"bucket-{bucket:04d}"
    destination = output_root / "shards" / bucket_name
    if _validated_compacted_bucket(
        destination,
        bucket=bucket,
        bucket_count=bucket_count,
        sources=sources,
    ):
        return
    if not sources:
        return
    pa, pq = _require_pyarrow()
    tables = []
    for source in sources:
        deadline.check(f"bucket {bucket} partition {source.partition} read")
        parquet = pq.ParquetFile(source.path)
        table = parquet.read_row_group(source.row_group)
        if table.num_rows != source.rows:
            raise ValueError(
                f"Raw-scan row group changed for bucket {bucket}, partition {source.partition}"
            )
        tables.append(table)
    table = pa.concat_tables(tables, promote_options="default")
    frame = _enrich_compacted_frame(
        table.to_pandas(),
        max_abs_log_return=max_abs_log_return,
    )

    work_root = output_root / ".work"
    staging_parent = work_root / "compaction-staging"
    staging_parent.mkdir(parents=True, exist_ok=True)
    _discard_stale_staging(staging_parent, bucket_name)
    staging = _staging_directory(staging_parent, bucket_name)
    shard_path = staging / "shard.parquet"
    writer: Any = None
    index_rows: list[dict[str, Any]] = []
    observed_dates: set[pd.Timestamp] = set()
    row_group = 0
    try:
        for symbol_text, ordered in _compacted_symbol_rows(frame):
            deadline.check(f"bucket {bucket} symbol {symbol_text} compaction")
            symbol_table = pa.Table.from_pandas(ordered, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(
                    shard_path,
                    symbol_table.schema,
                    compression="zstd",
                    use_dictionary=True,
                    write_statistics=True,
                )
            elif symbol_table.schema != writer.schema:
                symbol_table = symbol_table.cast(writer.schema)
            writer.write_table(symbol_table, row_group_size=len(ordered))
            observed_dates.update(pd.DatetimeIndex(ordered["timestamp"]))
            index_rows.append(
                {
                    "symbol": symbol_text,
                    "bucket": bucket,
                    "shard_relative_path": (
                        Path("shards") / bucket_name / "shard.parquet"
                    ).as_posix(),
                    "row_group": row_group,
                    "row_count": len(ordered),
                    **_symbol_index_metadata(
                        ordered,
                        symbol=symbol_text,
                        benchmark_mapping=benchmark_mapping,
                    ),
                }
            )
            row_group += 1
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise RuntimeError(f"Bucket {bucket} contained no symbol rows")
    _atomic_parquet(pd.DataFrame(index_rows), staging / "symbol-index.parquet")
    _atomic_parquet(
        pd.DataFrame(
            {"timestamp": sorted(observed_dates)},
        ),
        staging / "observed-dates.parquet",
    )
    atomic_write_json(
        staging / "checkpoint.json",
        {
            "schema_version": BAR_STORE_SCHEMA_VERSION,
            "kind": "bar-store-compacted-bucket",
            "source_scan_algorithm": RAW_SCAN_ALGORITHM,
            "source_partitions": [source.partition for source in sources],
            "bucket": bucket,
            "bucket_count": bucket_count,
            "rows": len(frame),
            "symbols": len(index_rows),
            "shard_sha256": sha256_file(shard_path),
            "shard_size_bytes": shard_path.stat().st_size,
        },
    )
    _atomic_directory(staging, destination)


def _compact_bucket_task(task: _CompactBucketTask) -> int:
    _compact_bucket(
        output_root=Path(task.output_root),
        bucket=task.bucket,
        bucket_count=task.bucket_count,
        sources=task.sources,
        benchmark_mapping=task.benchmark_mapping,
        max_abs_log_return=task.max_abs_log_return,
        deadline=_Deadline(task.deadline_epoch_seconds),
    )
    return task.bucket


def _run_bucket_compaction(
    *,
    output_root: Path,
    work_root: Path,
    scan_index: Mapping[str, Any],
    bucket_count: int,
    benchmark_mapping: dict[str, str],
    max_abs_log_return: float,
    deadline: _Deadline,
    requested_workers: int,
    detected_cpu_count: int,
    available_memory_bytes: int,
    worker_memory_budget_bytes: int,
) -> _PhasePlan:
    """Compact independent hash buckets with a conservative expansion estimate."""

    sources_by_bucket = _bucket_sources_from_scan_index(
        work_root=work_root,
        scan_index=scan_index,
        bucket_count=bucket_count,
    )
    pending: list[tuple[int, _CompactBucketTask]] = []
    reused_tasks = 0
    task_count = len(sources_by_bucket)
    for bucket, sources in sorted(sources_by_bucket.items()):
        destination = output_root / "shards" / f"bucket-{bucket:04d}"
        if _validated_compacted_bucket(
            destination,
            bucket=bucket,
            bucket_count=bucket_count,
            sources=sources,
        ):
            reused_tasks += 1
            continue
        compressed_bytes = sum(source.compressed_bytes for source in sources)
        estimated_bytes = WORKER_BASE_MEMORY_BYTES + max(
            compressed_bytes * 48,
            compressed_bytes + 128 * MIB,
        )
        pending.append(
            (
                estimated_bytes,
                _CompactBucketTask(
                    output_root=str(output_root),
                    bucket=bucket,
                    bucket_count=bucket_count,
                    sources=sources,
                    benchmark_mapping=benchmark_mapping,
                    max_abs_log_return=max_abs_log_return,
                    deadline_epoch_seconds=deadline.epoch_seconds,
                ),
            )
        )
    pending.sort(key=lambda item: (-item[0], item[1].bucket))
    plan = _plan_phase_workers(
        phase="bucket_compaction",
        requested_workers=requested_workers,
        detected_cpu_count=detected_cpu_count,
        task_memory_bytes=[item[0] for item in pending],
        task_count=task_count,
        reused_tasks=reused_tasks,
        available_memory_bytes=available_memory_bytes,
        worker_memory_budget_bytes=worker_memory_budget_bytes,
    )
    _record_phase_plan(work_root, plan)
    completed_tasks = reused_tasks
    _emit_phase_progress(
        phase=plan.phase,
        state="running" if pending else "complete",
        completed_tasks=completed_tasks,
        total_tasks=task_count,
        effective_workers=plan.effective_workers,
    )

    def record_completed(_: int) -> None:
        nonlocal completed_tasks
        completed_tasks += 1
        _emit_phase_progress(
            phase=plan.phase,
            state="running" if completed_tasks < task_count else "complete",
            completed_tasks=completed_tasks,
            total_tasks=task_count,
            effective_workers=plan.effective_workers,
        )

    _run_phase_tasks(
        plan=plan,
        tasks=[item[1] for item in pending],
        runner=_compact_bucket_task,
        deadline=deadline,
        on_result=record_completed,
    )
    return plan


def _load_symbol_index(output_root: Path) -> pd.DataFrame:
    parts = sorted((output_root / "shards").glob("bucket-*/symbol-index.parquet"))
    if not parts:
        raise RuntimeError("Bar-store compaction produced no symbol index parts")
    return _ordered_symbol_index(
        pd.concat([pd.read_parquet(path) for path in parts], ignore_index=True)
    )


def _ordered_symbol_index(index: pd.DataFrame) -> pd.DataFrame:
    """Return the canonical symbol-index order independent of bucket layout."""

    return index.sort_values("symbol", kind="stable").reset_index(drop=True)


@lru_cache(maxsize=4)
def _load_planning_symbol_index(
    path_text: str,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    index = pd.read_parquet(path_text)
    index_by_symbol = {str(row["symbol"]): row for row in index.to_dict(orient="records")}
    return index, index_by_symbol


def _candidate_mask(
    frame: pd.DataFrame,
    benchmark: pd.DataFrame,
    *,
    window_size: int,
    max_horizon: int,
) -> np.ndarray:
    count = len(frame)
    result = np.zeros(count, dtype=bool)
    first_cutoff = window_size - 1
    final_cutoff = count - max_horizon - 1
    if final_cutoff < first_cutoff:
        return result
    cutoff_indices = np.arange(first_cutoff, final_cutoff + 1, dtype=np.int64)
    starts = cutoff_indices - window_size + 1
    ends = cutoff_indices + max_horizon

    bad = frame["adjusted_transition_extreme"].to_numpy(dtype=bool)
    bad_prefix = np.concatenate(([0], np.cumsum(bad, dtype=np.int64)))
    bad_counts = bad_prefix[ends + 1] - bad_prefix[starts + 1]

    asset_timestamps = pd.DatetimeIndex(frame["timestamp"]).asi8
    benchmark_timestamps = pd.DatetimeIndex(benchmark["timestamp"]).asi8
    missing = ~np.isin(asset_timestamps, benchmark_timestamps, assume_unique=False)
    missing_prefix = np.concatenate(([0], np.cumsum(missing, dtype=np.int64)))
    missing_counts = missing_prefix[ends + 1] - missing_prefix[starts]
    result[cutoff_indices] = (bad_counts == 0) & (missing_counts == 0)
    return result


def _candidate_range_records(
    *,
    symbol: str,
    bucket: int,
    mask: np.ndarray,
    timestamps: pd.DatetimeIndex,
) -> tuple[list[dict[str, Any]], set[pd.Timestamp]]:
    """Compress one symbol's valid cutoff mask and collect its cutoff dates."""

    ranges: list[dict[str, Any]] = []
    candidate_dates: set[pd.Timestamp] = set()
    for start_index, stop_index in _true_ranges(mask):
        ranges.append(
            {
                "symbol": symbol,
                "bucket": bucket,
                "start_index": start_index,
                "stop_index": stop_index,
                "count": stop_index - start_index,
            }
        )
        candidate_dates.update(timestamps[start_index:stop_index])
    return ranges, candidate_dates


def _eligible_bucket_rows(index: pd.DataFrame, *, bucket: int) -> pd.DataFrame:
    """Select target symbols for one physical bucket without changing semantics."""

    return index[(index["bucket"] == bucket) & index["eligible"].astype(bool)]


def _candidate_bucket_records(
    *,
    index: pd.DataFrame,
    bucket: int,
    index_by_symbol: Mapping[str, Mapping[str, Any]],
    read_symbol: Callable[[Mapping[str, Any]], pd.DataFrame],
    window_size: int,
    max_horizon: int,
    check_symbol: Callable[[str], None] | None = None,
) -> tuple[list[dict[str, Any]], set[pd.Timestamp], Counter[str]]:
    """Compute one bucket's logical lazy-cutoff candidates without storage I/O."""

    ranges: list[dict[str, Any]] = []
    candidate_dates: set[pd.Timestamp] = set()
    exclusions: Counter[str] = Counter()
    selected = _eligible_bucket_rows(index, bucket=bucket)
    benchmark_cache: dict[str, pd.DataFrame] = {}
    for row in selected.to_dict(orient="records"):
        symbol = str(row["symbol"])
        if check_symbol is not None:
            check_symbol(symbol)
        benchmark_symbol = str(row["benchmark_symbol"])
        benchmark_row = index_by_symbol.get(benchmark_symbol)
        if benchmark_row is None:
            exclusions["missing_benchmark"] += 1
            continue
        frame = read_symbol(row)
        benchmark = benchmark_cache.get(benchmark_symbol)
        if benchmark is None:
            benchmark = read_symbol(benchmark_row)
            benchmark_cache[benchmark_symbol] = benchmark
        mask = _candidate_mask(
            frame,
            benchmark,
            window_size=window_size,
            max_horizon=max_horizon,
        )
        symbol_ranges, symbol_dates = _candidate_range_records(
            symbol=symbol,
            bucket=bucket,
            mask=mask,
            timestamps=pd.DatetimeIndex(frame["timestamp"]),
        )
        if not symbol_ranges:
            exclusions["no_valid_cutoffs"] += 1
            continue
        ranges.extend(symbol_ranges)
        candidate_dates.update(symbol_dates)
    return ranges, candidate_dates, exclusions


def _build_candidate_bucket(
    *,
    output_root: Path,
    work_root: Path,
    bucket: int,
    index: pd.DataFrame,
    index_by_symbol: dict[str, dict[str, Any]],
    window_size: int,
    max_horizon: int,
    deadline: _Deadline,
) -> None:
    deadline.check(f"bucket {bucket} candidate ranges")
    bucket_name = f"bucket-{bucket:04d}"
    destination = work_root / "candidates" / bucket_name
    if destination.is_dir() and (destination / "checkpoint.json").is_file():
        return
    staging_parent = work_root / "candidate-staging"
    staging_parent.mkdir(parents=True, exist_ok=True)
    _discard_stale_staging(staging_parent, bucket_name)
    staging = _staging_directory(staging_parent, bucket_name)
    ranges, candidate_dates, exclusions = _candidate_bucket_records(
        index=index,
        bucket=bucket,
        index_by_symbol=index_by_symbol,
        read_symbol=lambda row: _read_symbol(row, output_root),
        window_size=window_size,
        max_horizon=max_horizon,
        check_symbol=lambda symbol: deadline.check(f"candidate symbol {symbol}"),
    )
    if ranges:
        _atomic_parquet(pd.DataFrame(ranges), staging / "candidate-ranges.parquet")
    if candidate_dates:
        _atomic_parquet(
            pd.DataFrame({"timestamp": sorted(candidate_dates)}),
            staging / "candidate-dates.parquet",
        )
    atomic_write_json(
        staging / "checkpoint.json",
        {
            "schema_version": BAR_STORE_SCHEMA_VERSION,
            "kind": "bar-store-candidate-bucket",
            "bucket": bucket,
            "range_rows": len(ranges),
            "candidate_dates": len(candidate_dates),
            "excluded_symbols_by_reason": dict(sorted(exclusions.items())),
        },
    )
    _atomic_directory(staging, destination)


def _candidate_bucket_task(task: _CandidateBucketTask) -> None:
    index, index_by_symbol = _load_planning_symbol_index(task.planning_index_path)
    _build_candidate_bucket(
        output_root=Path(task.output_root),
        work_root=Path(task.work_root),
        bucket=task.bucket,
        index=index,
        index_by_symbol=index_by_symbol,
        window_size=task.window_size,
        max_horizon=task.max_horizon,
        deadline=_Deadline(task.deadline_epoch_seconds),
    )


def _run_candidate_buckets(
    *,
    output_root: Path,
    work_root: Path,
    bucket_count: int,
    planning_index_path: Path,
    window_size: int,
    max_horizon: int,
    deadline: _Deadline,
    requested_workers: int,
    detected_cpu_count: int,
    available_memory_bytes: int,
    worker_memory_budget_bytes: int,
) -> _PhasePlan:
    pending: list[tuple[int, _CandidateBucketTask]] = []
    reused_tasks = 0
    for bucket in range(bucket_count):
        destination = work_root / "candidates" / f"bucket-{bucket:04d}"
        if destination.is_dir() and (destination / "checkpoint.json").is_file():
            reused_tasks += 1
            continue
        shard = output_root / "shards" / f"bucket-{bucket:04d}" / "shard.parquet"
        compressed_bytes = shard.stat().st_size if shard.is_file() else 0
        estimated_bytes = WORKER_BASE_MEMORY_BYTES + max(
            compressed_bytes * 24,
            128 * MIB,
        )
        pending.append(
            (
                estimated_bytes,
                _CandidateBucketTask(
                    output_root=str(output_root),
                    work_root=str(work_root),
                    bucket=bucket,
                    planning_index_path=str(planning_index_path),
                    window_size=window_size,
                    max_horizon=max_horizon,
                    deadline_epoch_seconds=deadline.epoch_seconds,
                ),
            )
        )
    pending.sort(key=lambda item: (-item[0], item[1].bucket))
    plan = _plan_phase_workers(
        phase="candidate_ranges",
        requested_workers=requested_workers,
        detected_cpu_count=detected_cpu_count,
        task_memory_bytes=[item[0] for item in pending],
        task_count=bucket_count,
        reused_tasks=reused_tasks,
        available_memory_bytes=available_memory_bytes,
        worker_memory_budget_bytes=worker_memory_budget_bytes,
    )
    _record_phase_plan(work_root, plan)
    _run_phase_tasks(
        plan=plan,
        tasks=[item[1] for item in pending],
        runner=_candidate_bucket_task,
        deadline=deadline,
    )
    return plan


def _global_dates(
    output_root: Path,
    work_root: Path,
) -> tuple[list[pd.Timestamp], list[pd.Timestamp]]:
    observed_parts = sorted((output_root / "shards").glob("bucket-*/observed-dates.parquet"))
    candidate_parts = sorted((work_root / "candidates").glob("bucket-*/candidate-dates.parquet"))
    return _global_calendars(
        observed_values=(
            value
            for path in observed_parts
            for value in pd.read_parquet(path, columns=["timestamp"])["timestamp"]
        ),
        candidate_values=(
            value
            for path in candidate_parts
            for value in pd.read_parquet(path, columns=["timestamp"])["timestamp"]
        ),
    )


def _global_calendars(
    *,
    observed_values: Iterable[Any],
    candidate_values: Iterable[Any],
) -> tuple[list[pd.Timestamp], list[pd.Timestamp]]:
    """Build the logical observed and candidate calendars from persisted values."""

    observed = _ordered_unique_timestamps(observed_values)
    candidates = _ordered_unique_timestamps(candidate_values)
    if not observed or not candidates:
        raise ValueError("No quality-approved cutoff dates were produced")
    return observed, candidates


def _ordered_unique_timestamps(values: Iterable[Any]) -> list[pd.Timestamp]:
    """Return the canonical global calendar used by chronological splitting."""

    return sorted({pd.Timestamp(value) for value in values})


def _split_boundaries(
    observed_dates: list[pd.Timestamp],
    candidate_dates: list[pd.Timestamp],
    *,
    train_fraction: float,
    validation_fraction: float,
    purge_bars: int,
    embargo_bars: int,
) -> dict[str, Any]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("train_fraction + validation_fraction must be less than 1")
    train_position = int(len(candidate_dates) * train_fraction)
    validation_position = int(len(candidate_dates) * (train_fraction + validation_fraction))
    if not 0 < train_position < validation_position < len(candidate_dates):
        raise ValueError("Not enough candidate cutoff dates for three chronological splits")
    observed_index = {timestamp.value: index for index, timestamp in enumerate(observed_dates)}
    train_boundary = candidate_dates[train_position]
    validation_boundary = candidate_dates[validation_position]
    train_boundary_index = observed_index[train_boundary.value]
    validation_boundary_index = observed_index[validation_boundary.value]
    bounds = {
        "train_boundary": train_boundary,
        "validation_boundary": validation_boundary,
        "train_stop": train_boundary_index - purge_bars,
        "validation_start": train_boundary_index + embargo_bars,
        "validation_stop": validation_boundary_index - purge_bars,
        "test_start": validation_boundary_index + embargo_bars,
        "observed_index": observed_index,
    }
    if (
        min(
            int(bounds["train_stop"]),
            int(bounds["validation_stop"]) - int(bounds["validation_start"]),
            len(observed_dates) - int(bounds["test_start"]),
        )
        <= 0
    ):
        raise ValueError("Purge and embargo leave an empty chronological split")
    return bounds


def _observed_timestamp_array(observed_index: Mapping[int, int]) -> np.ndarray:
    observed = np.empty(len(observed_index), dtype=np.int64)
    for timestamp_ns, position in observed_index.items():
        observed[int(position)] = int(timestamp_ns)
    if len(observed) > 1 and np.any(np.diff(observed) <= 0):
        raise RuntimeError("Global observed timestamps are not strictly increasing")
    return observed


def _split_codes(
    *,
    timestamp_ns: np.ndarray,
    indices: np.ndarray,
    observed_ns: np.ndarray,
    boundaries: Mapping[str, Any],
    max_horizon: int,
) -> tuple[np.ndarray, Counter[str]]:
    """Assign numerical split codes without filesystem or process concerns."""

    cutoff_ns = timestamp_ns[indices]
    observed_positions = np.searchsorted(observed_ns, cutoff_ns)
    if np.any(observed_positions >= len(observed_ns)) or np.any(
        observed_ns[observed_positions] != cutoff_ns
    ):
        raise RuntimeError("Candidate cutoff timestamps are absent from the trading calendar")
    label_end_ns = timestamp_ns[indices + max_horizon]
    codes = np.zeros(len(indices), dtype=np.int8)
    dropped: Counter[str] = Counter()

    train_region = observed_positions < int(boundaries["train_stop"])
    train_ok = train_region & (label_end_ns < pd.Timestamp(boundaries["train_boundary"]).value)
    train_crossed = train_region & ~train_ok
    codes[train_ok] = 1
    dropped["label_crosses_train_boundary"] += int(train_crossed.sum())

    validation_region = (observed_positions >= int(boundaries["validation_start"])) & (
        observed_positions < int(boundaries["validation_stop"])
    )
    validation_ok = validation_region & (
        label_end_ns < pd.Timestamp(boundaries["validation_boundary"]).value
    )
    validation_crossed = validation_region & ~validation_ok
    codes[validation_ok] = 2
    dropped["label_crosses_validation_boundary"] += int(validation_crossed.sum())

    test_region = observed_positions >= int(boundaries["test_start"])
    codes[test_region] = 3
    unassigned = codes == 0
    dropped["purge_or_embargo"] += int(
        unassigned.sum() - train_crossed.sum() - validation_crossed.sum()
    )
    return codes, dropped


def _split_range_records(
    *,
    symbol: str,
    indices: np.ndarray,
    codes: np.ndarray,
    timestamps: pd.DatetimeIndex,
    max_horizon: int,
) -> list[dict[str, Any]]:
    """Compress per-cutoff split codes into lazy contiguous range records."""

    records: list[dict[str, Any]] = []
    for code, split in ((1, "train"), (2, "validation"), (3, "test")):
        for local_start, local_stop in _true_ranges(codes == code):
            start_index = int(indices[local_start])
            stop_index = int(indices[local_stop - 1]) + 1
            records.append(
                {
                    "symbol": symbol,
                    "split": split,
                    "start_index": start_index,
                    "stop_index": stop_index,
                    "count": stop_index - start_index,
                    "cutoff_start_at": timestamps[start_index].isoformat(),
                    "cutoff_end_at": timestamps[stop_index - 1].isoformat(),
                    "label_end_max_at": timestamps[
                        stop_index - 1 + max_horizon
                    ].isoformat(),
                }
            )
    return records


def _ordered_split_ranges(ranges: pd.DataFrame) -> pd.DataFrame:
    """Return the canonical lazy-range order used by training datasets."""

    return ranges.sort_values(
        ["split", "symbol", "start_index"],
        kind="stable",
    ).reset_index(drop=True)


def _split_bucket_records(
    *,
    candidates: pd.DataFrame,
    index_by_symbol: Mapping[str, Mapping[str, Any]],
    read_symbol: Callable[[Mapping[str, Any]], pd.DataFrame],
    boundaries: Mapping[str, Any],
    max_horizon: int,
    check_symbol: Callable[[str], None] | None = None,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """Assign and compress one bucket's logical chronological split records."""

    output: list[dict[str, Any]] = []
    dropped: Counter[str] = Counter()
    observed_ns = _observed_timestamp_array(boundaries["observed_index"])
    for symbol, rows in candidates.groupby("symbol", sort=True):
        symbol_text = str(symbol)
        if check_symbol is not None:
            check_symbol(symbol_text)
        frame = read_symbol(index_by_symbol[symbol_text])
        timestamps = pd.DatetimeIndex(frame["timestamp"])
        timestamp_ns = timestamps.asi8
        for candidate in rows.itertuples(index=False):
            indices = np.arange(
                int(candidate.start_index),
                int(candidate.stop_index),
                dtype=np.int64,
            )
            codes, candidate_dropped = _split_codes(
                timestamp_ns=timestamp_ns,
                indices=indices,
                observed_ns=observed_ns,
                boundaries=boundaries,
                max_horizon=max_horizon,
            )
            dropped.update(candidate_dropped)
            output.extend(
                _split_range_records(
                    symbol=symbol_text,
                    indices=indices,
                    codes=codes,
                    timestamps=timestamps,
                    max_horizon=max_horizon,
                )
            )
    return output, dropped


def _build_split_bucket(
    *,
    output_root: Path,
    work_root: Path,
    candidate_part: Path,
    index_by_symbol: dict[str, dict[str, Any]],
    boundaries: dict[str, Any],
    max_horizon: int,
    deadline: _Deadline,
) -> None:
    bucket_name = candidate_part.parent.name
    destination = work_root / "split-ranges" / bucket_name
    if destination.is_dir() and (destination / "checkpoint.json").is_file():
        return

    deadline.check(f"split assignment for {bucket_name}")
    staging_parent = work_root / "split-staging"
    staging_parent.mkdir(parents=True, exist_ok=True)
    _discard_stale_staging(staging_parent, bucket_name)
    staging = _staging_directory(staging_parent, bucket_name)
    candidates = pd.read_parquet(candidate_part)

    try:
        output, dropped = _split_bucket_records(
            candidates=candidates,
            index_by_symbol=index_by_symbol,
            read_symbol=lambda row: _read_symbol(row, output_root),
            boundaries=boundaries,
            max_horizon=max_horizon,
            check_symbol=lambda symbol: deadline.check(
                f"split assignment for {symbol}"
            ),
        )
        if output:
            ranges = _ordered_split_ranges(pd.DataFrame(output))
            _atomic_parquet(ranges, staging / "cutoff-ranges.parquet")
        atomic_write_json(
            staging / "checkpoint.json",
            {
                "schema_version": BAR_STORE_SCHEMA_VERSION,
                "kind": "bar-store-split-bucket",
                "algorithm": SPLIT_ASSIGNMENT_ALGORITHM,
                "bucket": int(bucket_name.removeprefix("bucket-")),
                "range_rows": len(output),
                "dropped_counts_by_reason": dict(sorted(dropped.items())),
            },
        )
        _atomic_directory(staging, destination)
    except Exception:
        _remove_generated_tree(staging)
        raise


def _split_bucket_task(task: _SplitBucketTask) -> None:
    _, index_by_symbol = _load_planning_symbol_index(task.planning_index_path)
    _build_split_bucket(
        output_root=Path(task.output_root),
        work_root=Path(task.work_root),
        candidate_part=Path(task.candidate_part),
        index_by_symbol=index_by_symbol,
        boundaries=task.boundaries,
        max_horizon=task.max_horizon,
        deadline=_Deadline(task.deadline_epoch_seconds),
    )


def _assign_split_ranges(
    *,
    output_root: Path,
    work_root: Path,
    planning_index_path: Path,
    boundaries: dict[str, Any],
    max_horizon: int,
    deadline: _Deadline,
    requested_workers: int,
    detected_cpu_count: int,
    available_memory_bytes: int,
    worker_memory_budget_bytes: int,
) -> tuple[pd.DataFrame, dict[str, Any], _PhasePlan]:
    candidate_parts = sorted((work_root / "candidates").glob("bucket-*/candidate-ranges.parquet"))
    pending: list[tuple[int, _SplitBucketTask]] = []
    reused_tasks = 0
    for part in candidate_parts:
        destination = work_root / "split-ranges" / part.parent.name
        if destination.is_dir() and (destination / "checkpoint.json").is_file():
            reused_tasks += 1
            continue
        shard = output_root / "shards" / part.parent.name / "shard.parquet"
        shard_bytes = shard.stat().st_size if shard.is_file() else 0
        candidate_bytes = part.stat().st_size
        estimated_bytes = WORKER_BASE_MEMORY_BYTES + max(
            shard_bytes * 16 + candidate_bytes * 32,
            128 * MIB,
        )
        pending.append(
            (
                estimated_bytes,
                _SplitBucketTask(
                    output_root=str(output_root),
                    work_root=str(work_root),
                    candidate_part=str(part),
                    planning_index_path=str(planning_index_path),
                    boundaries=boundaries,
                    max_horizon=max_horizon,
                    deadline_epoch_seconds=deadline.epoch_seconds,
                ),
            )
        )
    pending.sort(key=lambda item: (-item[0], item[1].candidate_part))
    plan = _plan_phase_workers(
        phase="split_ranges",
        requested_workers=requested_workers,
        detected_cpu_count=detected_cpu_count,
        task_memory_bytes=[item[0] for item in pending],
        task_count=len(candidate_parts),
        reused_tasks=reused_tasks,
        available_memory_bytes=available_memory_bytes,
        worker_memory_budget_bytes=worker_memory_budget_bytes,
    )
    _record_phase_plan(work_root, plan)
    _run_phase_tasks(
        plan=plan,
        tasks=[item[1] for item in pending],
        runner=_split_bucket_task,
        deadline=deadline,
    )

    split_parts = sorted((work_root / "split-ranges").glob("bucket-*/cutoff-ranges.parquet"))
    if not split_parts:
        raise ValueError("Chronological split produced no lazy cutoff ranges")
    ranges = _ordered_split_ranges(
        pd.concat(
            [pd.read_parquet(path) for path in split_parts],
            ignore_index=True,
        )
    )
    dropped: Counter[str] = Counter()
    for checkpoint_path in sorted((work_root / "split-ranges").glob("bucket-*/checkpoint.json")):
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("algorithm") != SPLIT_ASSIGNMENT_ALGORITHM:
            raise ValueError(f"Unsupported split checkpoint: {checkpoint_path}")
        dropped.update(
            {
                str(key): int(value)
                for key, value in checkpoint.get("dropped_counts_by_reason", {}).items()
            }
        )

    train_boundary = pd.Timestamp(boundaries["train_boundary"])
    validation_boundary = pd.Timestamp(boundaries["validation_boundary"])
    counts = {
        split: int(ranges.loc[ranges["split"] == split, "count"].sum())
        for split in ("train", "validation", "test")
    }
    if any(value < 1 for value in counts.values()):
        raise ValueError("Chronological split produced an empty lazy split")

    summaries: dict[str, dict[str, Any]] = {}
    for split in ("train", "validation", "test"):
        selected = ranges[ranges["split"] == split]
        summaries[split] = {
            "records": counts[split],
            "range_rows": len(selected),
            "cutoff_start_at": min(selected["cutoff_start_at"]),
            "cutoff_end_at": max(selected["cutoff_end_at"]),
            "label_end_max_at": max(selected["label_end_max_at"]),
        }
    if pd.Timestamp(summaries["train"]["label_end_max_at"]) >= train_boundary:
        raise RuntimeError("Train labels cross the global train boundary")
    if pd.Timestamp(summaries["validation"]["label_end_max_at"]) >= validation_boundary:
        raise RuntimeError("Validation labels cross the global validation boundary")
    split_audit = {
        "schema_version": "causal-split-audit-v1",
        "policy": SPLIT_POLICY,
        "comparison": "label.end_at < next_split_start",
        "violations": 0,
        "train_boundary_exclusive": train_boundary.isoformat(),
        "validation_boundary_exclusive": validation_boundary.isoformat(),
        "validation_start": summaries["validation"]["cutoff_start_at"],
        "test_start": summaries["test"]["cutoff_start_at"],
        "label_end_counts": counts,
        "maximum_label_end": {
            split: summaries[split]["label_end_max_at"] for split in ("train", "validation", "test")
        },
        "dropped_counts_by_reason": dict(sorted(dropped.items())),
        "splits": summaries,
    }
    return ranges, split_audit, plan


def _combine_quality(
    index: pd.DataFrame,
    ranges: pd.DataFrame,
    work_root: Path,
) -> dict[str, Any]:
    candidate_exclusions: Counter[str] = Counter()
    for checkpoint_path in sorted((work_root / "candidates").glob("bucket-*/checkpoint.json")):
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        candidate_exclusions.update(
            {
                str(key): int(value)
                for key, value in checkpoint.get("excluded_symbols_by_reason", {}).items()
            }
        )
    return {
        "rows": int(index["row_count"].sum()),
        "symbols": len(index),
        "eligible_security_symbols": int(index["eligible"].astype(bool).sum()),
        "symbols_with_valid_cutoffs": int(ranges["symbol"].nunique()),
        "valid_cutoff_ranges": len(ranges),
        "valid_cutoffs": int(ranges["count"].sum()),
        "excluded_symbols_by_reason": {
            str(key): int(value)
            for key, value in index.loc[~index["eligible"].astype(bool), "eligibility_reason"]
            .value_counts()
            .sort_index()
            .items()
        },
        "eligible_symbols_without_valid_cutoffs": dict(sorted(candidate_exclusions.items())),
        "extreme_return_transitions": int(index["extreme_transition_count"].sum()),
        "calendar_gaps_over_10_days": int(index["long_calendar_gap_count"].sum()),
        "quality_policy": (
            "invalid cutoffs are excluded lazily; source bars remain immutable and auditable"
        ),
    }


def build_symbol_bar_store(
    *,
    raw_path: str | Path,
    output_root: str | Path,
    download_manifest: Mapping[str, Any],
    benchmark_mapping: Mapping[str, str] | None = None,
    window_size: int = DEFAULT_WINDOW_SIZE,
    max_horizon: int = DEFAULT_MAX_HORIZON,
    max_abs_log_return: float = DEFAULT_MAX_ABS_LOG_RETURN,
    train_fraction: float = DEFAULT_TRAIN_FRACTION,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    purge_bars: int = DEFAULT_PURGE_BARS,
    embargo_bars: int = DEFAULT_EFFECTIVE_EMBARGO_BARS,
    bucket_count: int = DEFAULT_BUCKET_COUNT,
    batch_rows: int = DEFAULT_BATCH_ROWS,
    deadline_epoch_seconds: float | None = None,
    workers: int = 1,
    memory_budget_bytes: int | None = None,
    materialization_digest: str | None = None,
) -> BarStoreBuildResult:
    """Build or resume an immutable compressed bar store without materializing windows."""

    source = Path(raw_path).resolve(strict=True)
    root = Path(output_root).resolve(strict=False)
    if bucket_count < 1 or bucket_count > 4096:
        raise ValueError("bucket_count must be between 1 and 4096")
    if batch_rows < 1:
        raise ValueError("batch_rows must be positive")
    if workers < 1:
        raise ValueError("workers must be positive")
    if memory_budget_bytes is not None and memory_budget_bytes < 1:
        raise ValueError("memory_budget_bytes must be positive")
    content_digest = materialization_digest or bar_store_materialization_digest()
    if (
        len(content_digest) != 64
        or any(character not in "0123456789abcdef" for character in content_digest)
    ):
        raise ValueError("materialization_digest must be a lowercase SHA-256 digest")
    if window_size < 2 or max_horizon != DEFAULT_MAX_HORIZON:
        raise ValueError("The approved bar store requires window_size>=2 and max_horizon=14")
    if max_abs_log_return <= 0.0:
        raise ValueError("max_abs_log_return must be positive")
    if root.is_symlink():
        raise ValueError("Bar-store root must not be a symlink")
    root.mkdir(parents=True, exist_ok=True)
    work_root = root / ".work"
    if work_root.is_symlink():
        raise ValueError("Bar-store work root must not be a symlink")
    work_root.mkdir(parents=True, exist_ok=True)
    success_path = root / "_SUCCESS.json"
    manifest_path = root / "bar-store.json"
    index_path = root / "symbol-index.parquet"
    ranges_path = root / "cutoff-ranges.parquet"
    deadline = _Deadline(deadline_epoch_seconds)
    mapping = {
        str(key).upper(): str(value).upper() for key, value in (benchmark_mapping or {}).items()
    }
    raw_artifact = download_manifest.get("artifacts", {}).get("raw")
    if not isinstance(raw_artifact, Mapping):
        raise ValueError("Download manifest has no raw artifact")
    raw_sha256 = raw_artifact.get("sha256")
    raw_rows = raw_artifact.get("row_count")
    if (
        not isinstance(raw_sha256, str)
        or len(raw_sha256) != 64
        or any(character not in "0123456789abcdef" for character in raw_sha256)
        or not isinstance(raw_rows, int)
        or isinstance(raw_rows, bool)
        or raw_rows < 1
    ):
        raise ValueError("Download manifest raw artifact identity is invalid")
    identity = {
        "schema_version": BAR_STORE_SCHEMA_VERSION,
        "kind": BAR_STORE_KIND,
        "materialization_digest": content_digest,
        "raw_sha256": raw_sha256,
        "raw_rows": raw_rows,
        "window_size": window_size,
        "max_horizon": max_horizon,
        "max_abs_log_return": max_abs_log_return,
        "train_fraction": train_fraction,
        "validation_fraction": validation_fraction,
        "purge_bars": purge_bars,
        "effective_embargo_bars": embargo_bars,
        "benchmark_mapping_sha256": canonical_json_sha256(mapping),
    }
    identity_sha256 = canonical_json_sha256(identity)
    checkpoint_identity = {
        "schema_version": BAR_STORE_BUILD_CHECKPOINT_SCHEMA_VERSION,
        "kind": "bar-store-build-checkpoint",
        "content_identity_sha256": identity_sha256,
        "bucket_count": bucket_count,
        "batch_rows": batch_rows,
        "symbol_bucket_algorithm": SYMBOL_BUCKET_ALGORITHM,
        "raw_scan_algorithm": RAW_SCAN_ALGORITHM,
        "raw_scan_target_partition_rows": (
            batch_rows * SOURCE_PARTITION_BATCH_MULTIPLIER
        ),
        "split_assignment_algorithm": SPLIT_ASSIGNMENT_ALGORITHM,
    }
    checkpoint_identity_sha256 = canonical_json_sha256(checkpoint_identity)

    final_data_paths = (manifest_path, index_path, ranges_path)
    final_paths = (success_path, *final_data_paths)
    final_path_has_symlink = any(path.is_symlink() for path in final_paths)
    success_present = success_path.exists() or success_path.is_symlink()
    all_final_files = all(path.is_file() and not path.is_symlink() for path in final_paths)
    any_final_data_file = any(
        path.exists() or path.is_symlink() for path in final_data_paths
    )
    all_final_data_files = all(
        path.is_file() and not path.is_symlink() for path in final_data_paths
    )

    if (
        final_path_has_symlink
        or (success_present and not all_final_files)
        or (
            not success_present
            and any_final_data_file
            and not all_final_data_files
        )
    ):
        work_root, quarantine = _reset_generated_bar_store(
            root=root,
            previous_state={"identity_sha256": "incomplete-finalized-store"},
        )
        atomic_write_json(
            work_root / "build-state.json",
            {
                "identity": identity,
                "identity_sha256": identity_sha256,
                "checkpoint_identity": checkpoint_identity,
                "checkpoint_identity_sha256": checkpoint_identity_sha256,
                "incomplete_finalized_store_quarantine": str(quarantine),
            },
        )
    elif all_final_files:
        try:
            return _completed_bar_store_result(
                work_root=work_root,
                success_path=success_path,
                manifest_path=manifest_path,
                index_path=index_path,
                ranges_path=ranges_path,
                identity_sha256=identity_sha256,
            )
        except (KeyError, OSError, TypeError, ValueError):
            work_root, quarantine = _reset_generated_bar_store(
                root=root,
                previous_state={"identity_sha256": "invalid-finalized-store"},
            )
            atomic_write_json(
                work_root / "build-state.json",
                {
                    "identity": identity,
                    "identity_sha256": identity_sha256,
                    "checkpoint_identity": checkpoint_identity,
                    "checkpoint_identity_sha256": checkpoint_identity_sha256,
                    "invalid_finalized_store_quarantine": str(quarantine),
                },
            )
    elif not success_present and all_final_data_files:
        try:
            _recover_bar_store_success_marker(
                success_path=success_path,
                manifest_path=manifest_path,
                index_path=index_path,
                ranges_path=ranges_path,
                identity_sha256=identity_sha256,
            )
            return _completed_bar_store_result(
                work_root=work_root,
                success_path=success_path,
                manifest_path=manifest_path,
                index_path=index_path,
                ranges_path=ranges_path,
                identity_sha256=identity_sha256,
            )
        except (KeyError, OSError, TypeError, ValueError):
            work_root, quarantine = _reset_generated_bar_store(
                root=root,
                previous_state={"identity_sha256": "invalid-recoverable-store"},
            )
            atomic_write_json(
                work_root / "build-state.json",
                {
                    "identity": identity,
                    "identity_sha256": identity_sha256,
                    "checkpoint_identity": checkpoint_identity,
                    "checkpoint_identity_sha256": checkpoint_identity_sha256,
                    "invalid_recoverable_store_quarantine": str(quarantine),
                },
            )

    if manifest_path.exists() or manifest_path.is_symlink():
        try:
            existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            existing_manifest = {}
        if (
            not isinstance(existing_manifest, Mapping)
            or existing_manifest.get("identity_sha256") != identity_sha256
        ):
            work_root, quarantine = _reset_generated_bar_store(
                root=root,
                previous_state=(
                    existing_manifest if isinstance(existing_manifest, Mapping) else {}
                ),
            )
            atomic_write_json(
                work_root / "build-state.json",
                {
                    "identity": identity,
                    "identity_sha256": identity_sha256,
                    "checkpoint_identity": checkpoint_identity,
                    "checkpoint_identity_sha256": checkpoint_identity_sha256,
                    "incompatible_completed_store_quarantine": str(quarantine),
                },
            )
    state_path = work_root / "build-state.json"
    if state_path.is_file():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            state = {}
        if (
            not isinstance(state, Mapping)
            or state.get("identity_sha256") != identity_sha256
            or state.get("checkpoint_identity_sha256")
            != checkpoint_identity_sha256
        ):
            work_root, quarantine = _reset_generated_bar_store(
                root=root,
                previous_state=state if isinstance(state, Mapping) else {},
            )
            state_path = work_root / "build-state.json"
            atomic_write_json(
                state_path,
                {
                    "identity": identity,
                    "identity_sha256": identity_sha256,
                    "checkpoint_identity": checkpoint_identity,
                    "checkpoint_identity_sha256": checkpoint_identity_sha256,
                    "incompatible_checkpoint_quarantine": str(quarantine),
                },
            )
    else:
        atomic_write_json(
            state_path,
            {
                "identity": identity,
                "identity_sha256": identity_sha256,
                "checkpoint_identity": checkpoint_identity,
                "checkpoint_identity_sha256": checkpoint_identity_sha256,
            },
        )

    detected_cpu_count = detect_visible_cpu_count()
    memory_estimate = detect_available_memory()
    available_memory_bytes = memory_estimate.available_bytes
    worker_memory_budget_bytes = _safe_worker_memory_budget(
        available_memory_bytes,
        memory_budget_bytes,
    )
    _initialize_execution_plan(
        work_root=work_root,
        requested_workers=workers,
        detected_cpu_count=detected_cpu_count,
        available_memory_bytes=available_memory_bytes,
        worker_memory_budget_bytes=worker_memory_budget_bytes,
        memory_budget_override_bytes=memory_budget_bytes,
    )
    scan, scan_plan = _write_scan_partitions(
        raw_path=source,
        work_root=work_root,
        bucket_count=bucket_count,
        batch_rows=batch_rows,
        deadline=deadline,
        requested_workers=workers,
        detected_cpu_count=detected_cpu_count,
        available_memory_bytes=available_memory_bytes,
        worker_memory_budget_bytes=worker_memory_budget_bytes,
    )
    if scan["raw_rows"] != raw_rows:
        raise ValueError("Raw Parquet row count differs from the download manifest")
    compaction_plan = _run_bucket_compaction(
        output_root=root,
        work_root=work_root,
        scan_index=scan,
        bucket_count=bucket_count,
        benchmark_mapping=mapping,
        max_abs_log_return=max_abs_log_return,
        deadline=deadline,
        requested_workers=workers,
        detected_cpu_count=detected_cpu_count,
        available_memory_bytes=available_memory_bytes,
        worker_memory_budget_bytes=worker_memory_budget_bytes,
    )
    index = _load_symbol_index(root)
    if index["symbol"].duplicated().any():
        raise ValueError("Bar-store symbol index contains duplicate symbols")
    if int(index["row_count"].sum()) != raw_rows:
        raise ValueError("Bar-store symbol rows differ from the immutable raw artifact")
    index = _exclude_symbols_with_missing_benchmarks(index)
    planning_root = work_root / "planning"
    planning_root.mkdir(parents=True, exist_ok=True)
    planning_index_path = planning_root / "symbol-index.parquet"
    _load_planning_symbol_index.cache_clear()
    _atomic_parquet(index, planning_index_path)
    candidate_plan = _run_candidate_buckets(
        output_root=root,
        work_root=work_root,
        bucket_count=bucket_count,
        planning_index_path=planning_index_path,
        window_size=window_size,
        max_horizon=max_horizon,
        deadline=deadline,
        requested_workers=workers,
        detected_cpu_count=detected_cpu_count,
        available_memory_bytes=available_memory_bytes,
        worker_memory_budget_bytes=worker_memory_budget_bytes,
    )
    observed_dates, candidate_dates = _global_dates(root, work_root)
    boundaries = _split_boundaries(
        observed_dates,
        candidate_dates,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        purge_bars=purge_bars,
        embargo_bars=embargo_bars,
    )
    ranges, split_audit, split_plan = _assign_split_ranges(
        output_root=root,
        work_root=work_root,
        planning_index_path=planning_index_path,
        boundaries=boundaries,
        max_horizon=max_horizon,
        deadline=deadline,
        requested_workers=workers,
        detected_cpu_count=detected_cpu_count,
        available_memory_bytes=available_memory_bytes,
        worker_memory_budget_bytes=worker_memory_budget_bytes,
    )
    split_counts = {
        split: int(ranges.loc[ranges["split"] == split, "count"].sum())
        for split in ("train", "validation", "test")
    }
    _atomic_parquet(index, index_path)
    _atomic_parquet(ranges, ranges_path)
    quality = _combine_quality(index, ranges, work_root)
    shard_records: list[dict[str, Any]] = []
    for shard in sorted((root / "shards").glob("bucket-*/shard.parquet")):
        deadline.check(f"final metadata for {shard.parent.name}")
        relative = shard.relative_to(root).as_posix()
        bucket_index = int(shard.parent.name.removeprefix("bucket-"))
        selected = index[index["bucket"] == bucket_index]
        checkpoint_path = shard.parent / "checkpoint.json"
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if (
            checkpoint.get("kind") != "bar-store-compacted-bucket"
            or checkpoint.get("bucket") != bucket_index
            or checkpoint.get("rows") != int(selected["row_count"].sum())
            or checkpoint.get("symbols") != len(selected)
            or not isinstance(checkpoint.get("shard_sha256"), str)
            or len(checkpoint["shard_sha256"]) != 64
            or any(character not in "0123456789abcdef" for character in checkpoint["shard_sha256"])
        ):
            raise ValueError(f"Compaction checkpoint is invalid: {checkpoint_path}")
        shard_records.append(
            {
                "relative_path": relative,
                "sha256": checkpoint["shard_sha256"],
                "size_bytes": shard.stat().st_size,
                "row_count": int(selected["row_count"].sum()),
                "row_groups": len(selected),
            }
        )
    execution = {
        "layout": "hash_bucketed_symbol_row_groups",
        "compression": "zstd",
        "bucket_count": bucket_count,
        "symbol_bucket_algorithm": SYMBOL_BUCKET_ALGORITHM,
        "raw_scan_algorithm": RAW_SCAN_ALGORITHM,
        "raw_scan_batch_rows": batch_rows,
        "raw_scan_source_row_groups": scan["source_row_groups"],
        "raw_scan_partitions": scan["partition_count"],
        "raw_scan_target_partition_rows": scan["target_partition_rows"],
        "parallelism": {
            "backend": MULTIPROCESS_BACKEND,
            "requested_workers": workers,
            "detected_cpu_count": detected_cpu_count,
            "available_memory_bytes_at_planning": available_memory_bytes,
            "available_memory_source": memory_estimate.source,
            "available_memory_observations_bytes": dict(
                memory_estimate.observations
            ),
            "worker_memory_budget_bytes": worker_memory_budget_bytes,
            "memory_budget_override_bytes": memory_budget_bytes,
            "safe_memory_fraction": SAFE_MEMORY_FRACTION,
            "minimum_parent_headroom_bytes": MINIMUM_PARENT_HEADROOM_BYTES,
            "native_threads_per_worker": NATIVE_THREADS_PER_WORKER,
            "phases": {
                plan.phase: plan.as_dict()
                for plan in (
                    scan_plan,
                    compaction_plan,
                    candidate_plan,
                    split_plan,
                )
            },
        },
        "resumable_checkpoints": [
            "raw_scan_partitions",
            "compacted_buckets",
            "candidate_ranges",
            "split_ranges",
        ],
        "window_materialized": False,
        "labels_materialized": False,
        "temporary_work_reclaimed_after_success": True,
    }
    manifest = {
        "schema_version": BAR_STORE_SCHEMA_VERSION,
        "kind": BAR_STORE_KIND,
        "state": "ready",
        "identity": identity,
        "identity_sha256": identity_sha256,
        "row_count": int(index["row_count"].sum()),
        "symbol_count": len(index),
        "split_counts": split_counts,
        "split_audit": split_audit,
        "quality": quality,
        "execution": execution,
        "symbol_index": artifact_metadata(index_path, root=root, row_count=len(index)),
        "cutoff_ranges": artifact_metadata(
            ranges_path,
            root=root,
            row_count=len(ranges),
        ),
        "shards": shard_records,
    }
    deadline.check("bar-store manifest publication")
    atomic_write_json(manifest_path, manifest)
    integrity = validate_bar_store_artifacts(
        root,
        expected_identity_sha256=identity_sha256,
        require_success=False,
    )
    atomic_write_json(
        success_path,
        {
            "schema_version": BAR_STORE_SCHEMA_VERSION,
            "kind": "bar-store-success",
            "state": "ready",
            "identity_sha256": identity_sha256,
            "bar_store_manifest_sha256": integrity.manifest_sha256,
            "symbol_index_sha256": integrity.symbol_index_sha256,
            "cutoff_ranges_sha256": integrity.cutoff_ranges_sha256,
            "split_counts": split_counts,
        },
    )
    _cleanup_completed_work(work_root)
    return BarStoreBuildResult(
        bar_store_manifest_path=manifest_path,
        symbol_index_path=index_path,
        cutoff_ranges_path=ranges_path,
        success_path=success_path,
        split_counts=split_counts,
        split_audit=split_audit,
        quality=quality,
        execution=execution,
    )


def bar_store_preparation_provenance(
    *,
    window_size: int,
    max_horizon: int,
    benchmark_mapping_sha256: str,
    max_abs_log_return: float,
    train_fraction: float,
    validation_fraction: float,
    purge_bars: int,
    compatibility_stride: int,
    effective_sample_stride: int,
    compatibility_embargo_bars: int,
    effective_embargo_bars: int,
    target_horizon: int,
    diagnostic_horizons: list[int],
    flat_volatility_multiplier: float,
) -> dict[str, Any]:
    """Describe preparation for audit without defining dataset identity."""

    return {
        "schema_version": PREPARATION_PROVENANCE_SCHEMA_VERSION,
        "processed_schema_version": "4.0",
        "bar_store_schema_version": BAR_STORE_SCHEMA_VERSION,
        "storage_kind": BAR_STORE_KIND,
        "window_size": window_size,
        "stride": compatibility_stride,
        "effective_sample_stride": effective_sample_stride,
        "supported_h_start": list(SUPPORTED_H_START),
        "max_horizon": max_horizon,
        "alpha_horizons_available": list(range(1, max_horizon + 1)),
        "window_materialized": False,
        "labels_materialized": False,
        "label_computation": "lazy_in_dataloader_from_raw_adjusted_execution_bars",
        "label_kind": "benchmark_relative_adjusted_log_return",
        "signal_timing": "after_close_t",
        "entry_timing": "regular_session_open_t_plus_1",
        "entry_day_counts_as_holding_day_one": True,
        "exit_timing": "regular_session_close_t_plus_h",
        "input_adjustment": "point_in_time_total_return_ohlc_split_adjusted_volume",
        "training_security_scope": TRAINING_SECURITY_SCOPE,
        "split_policy": SPLIT_POLICY,
        "benchmark_mapping_sha256": benchmark_mapping_sha256,
        "target_horizon": target_horizon,
        "diagnostic_horizons": diagnostic_horizons,
        "flat_volatility_multiplier": flat_volatility_multiplier,
        "max_abs_log_return": max_abs_log_return,
        "train_fraction": train_fraction,
        "validation_fraction": validation_fraction,
        "purge_bars": purge_bars,
        "embargo_bars": compatibility_embargo_bars,
        "effective_embargo_bars": effective_embargo_bars,
    }
