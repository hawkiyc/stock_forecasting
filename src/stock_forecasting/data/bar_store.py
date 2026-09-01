"""Resumable symbol-oriented OHLCV storage for lazy training samples."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
import uuid
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stock_forecasting.data.adjustments import ensure_adjustment_columns
from stock_forecasting.data.benchmarks import resolve_benchmark
from stock_forecasting.data.manifest import (
    BAR_STORE_KIND,
    BAR_STORE_SCHEMA_VERSION,
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

DEFAULT_BUCKET_COUNT = 128
DEFAULT_BATCH_ROWS = 1_000_000
DEFAULT_MAX_HORIZON = 14
RAW_SCAN_ALGORITHM = "parquet-row-group-checkpoints-v2"
SPLIT_ASSIGNMENT_ALGORITHM = "vectorized-bucket-checkpoints-v1"
MULTIPROCESS_BACKEND = "process_pool_spawn"
NATIVE_THREADS_PER_WORKER = 1
RAW_SCAN_WORKER_CAP = 4
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
class _ScanRowGroupTask:
    raw_path: str
    work_root: str
    row_group: int
    row_group_rows: int
    segment_start: int
    segment_count: int
    bucket_count: int
    batch_rows: int
    deadline_epoch_seconds: float | None


@dataclass(frozen=True)
class _CompactBucketTask:
    output_root: str
    work_root: str
    bucket: int
    bucket_count: int
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


def _visible_cpu_count() -> int:
    """Return the CPU count visible to this process, including affinity limits."""

    affinity_count: int | None = None
    if hasattr(os, "sched_getaffinity"):
        try:
            affinity_count = len(os.sched_getaffinity(0))
        except OSError:
            affinity_count = None
    reported = os.cpu_count() or 1
    return max(1, min(reported, affinity_count or reported))


def _read_positive_integer(path: Path) -> int | None:
    try:
        value = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return None
    if not value or value == "max":
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def _detect_available_memory_bytes() -> int:
    """Return the strictest host/cgroup memory estimate available at runtime."""

    candidates: list[int] = []
    cgroup_pairs = (
        (
            Path("/sys/fs/cgroup/memory.max"),
            Path("/sys/fs/cgroup/memory.current"),
        ),
        (
            Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
            Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"),
        ),
    )
    for limit_path, usage_path in cgroup_pairs:
        limit = _read_positive_integer(limit_path)
        usage = _read_positive_integer(usage_path) or 0
        if limit is not None and limit > usage:
            candidates.append(limit - usage)

    try:
        meminfo = Path("/proc/meminfo").read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        meminfo = ""
    for line in meminfo.splitlines():
        fields = line.split()
        if line.startswith("MemAvailable:") and len(fields) >= 2 and fields[1].isdigit():
            candidates.append(int(fields[1]) * 1024)
            break

    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        available_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
    except (OSError, ValueError, TypeError, AttributeError):
        pass
    else:
        if page_size > 0 and available_pages > 0:
            candidates.append(page_size * available_pages)

    if not candidates:
        raise RuntimeError(
            "Unable to determine available memory for safe bar-store multiprocessing"
        )
    return min(candidates)


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
                runner(task)
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
                    future.result()
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
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame.sort_values("timestamp", kind="stable").reset_index(drop=True)


def _true_ranges(mask: np.ndarray, *, offset: int = 0) -> list[tuple[int, int]]:
    positions = np.flatnonzero(mask)
    if positions.size == 0:
        return []
    boundaries = np.flatnonzero(np.diff(positions) != 1) + 1
    groups = np.split(positions, boundaries)
    return [(int(group[0]) + offset, int(group[-1]) + offset + 1) for group in groups]


def _validated_scan_segment(
    segment_root: Path,
    *,
    segment_index: int,
    row_group: int,
    batch_in_row_group: int,
    expected_rows: int,
    bucket_count: int,
) -> bool:
    checkpoint_path = segment_root / "checkpoint.json"
    if not segment_root.is_dir() or not checkpoint_path.is_file():
        return False
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    populated = checkpoint.get("populated_buckets")
    if (
        checkpoint.get("schema_version") != BAR_STORE_SCHEMA_VERSION
        or checkpoint.get("kind") != "bar-store-scan-segment"
        or checkpoint.get("algorithm") != RAW_SCAN_ALGORITHM
        or checkpoint.get("segment") != segment_index
        or checkpoint.get("row_group") != row_group
        or checkpoint.get("batch_in_row_group") != batch_in_row_group
        or checkpoint.get("rows") != expected_rows
        or not isinstance(populated, list)
        or any(
            not isinstance(bucket, int)
            or isinstance(bucket, bool)
            or bucket < 0
            or bucket >= bucket_count
            for bucket in populated
        )
        or len(populated) != len(set(populated))
    ):
        raise ValueError(f"Raw-scan segment checkpoint is invalid: {checkpoint_path}")
    actual_parts = sorted(
        int(path.stem.removeprefix("bucket-"))
        for path in segment_root.glob("bucket-*.parquet")
        if path.is_file()
    )
    if actual_parts != sorted(populated):
        raise ValueError(f"Raw-scan segment parts are incomplete: {segment_root}")
    return True


def _completed_scan_row_group(
    segments_root: Path,
    *,
    row_group: int,
    row_group_rows: int,
    segment_start: int,
    batch_rows: int,
    bucket_count: int,
) -> int | None:
    checkpoint_path = segments_root / f"row-group-{row_group:06d}.json"
    if not checkpoint_path.is_file():
        return None
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    segment_count = checkpoint.get("segments")
    if (
        checkpoint.get("schema_version") != BAR_STORE_SCHEMA_VERSION
        or checkpoint.get("kind") != "bar-store-scan-row-group"
        or checkpoint.get("algorithm") != RAW_SCAN_ALGORITHM
        or checkpoint.get("row_group") != row_group
        or checkpoint.get("rows") != row_group_rows
        or checkpoint.get("segment_start") != segment_start
        or checkpoint.get("batch_rows") != batch_rows
        or not isinstance(segment_count, int)
        or isinstance(segment_count, bool)
        or segment_count < 0
    ):
        raise ValueError(f"Raw-scan row-group checkpoint is invalid: {checkpoint_path}")
    validated_rows = 0
    for batch_in_row_group in range(segment_count):
        segment_index = segment_start + batch_in_row_group
        segment_root = segments_root / f"segment-{segment_index:06d}"
        segment_checkpoint = json.loads(
            (segment_root / "checkpoint.json").read_text(encoding="utf-8")
        )
        segment_rows = segment_checkpoint.get("rows")
        if (
            not isinstance(segment_rows, int)
            or isinstance(segment_rows, bool)
            or segment_rows < 1
            or segment_rows > batch_rows
            or not _validated_scan_segment(
                segment_root,
                segment_index=segment_index,
                row_group=row_group,
                batch_in_row_group=batch_in_row_group,
                expected_rows=segment_rows,
                bucket_count=bucket_count,
            )
        ):
            raise ValueError(f"Raw-scan row group is incomplete: {checkpoint_path}")
        validated_rows += segment_rows
    if validated_rows != row_group_rows:
        raise ValueError(f"Raw-scan row-group row count is invalid: {checkpoint_path}")
    return segment_count


def _scan_row_group_task(task: _ScanRowGroupTask) -> dict[str, int]:
    """Scan one Parquet row group into uniquely owned durable segment paths."""

    _, pq = _require_pyarrow()
    deadline = _Deadline(task.deadline_epoch_seconds)
    segments_root = Path(task.work_root) / "segments"
    completed = _completed_scan_row_group(
        segments_root,
        row_group=task.row_group,
        row_group_rows=task.row_group_rows,
        segment_start=task.segment_start,
        batch_rows=task.batch_rows,
        bucket_count=task.bucket_count,
    )
    if completed is not None:
        return {"rows": task.row_group_rows, "segments": completed}

    parquet = pq.ParquetFile(task.raw_path)
    observed_rows = 0
    observed_batches = 0
    batches = parquet.iter_batches(
        batch_size=task.batch_rows,
        row_groups=[task.row_group],
    )
    for batch_in_row_group, batch in enumerate(batches):
        segment_index = task.segment_start + batch_in_row_group
        deadline.check(f"raw segment {segment_index}")
        segment_name = f"segment-{segment_index:06d}"
        segment_root = segments_root / segment_name
        observed_rows += batch.num_rows
        observed_batches += 1
        if _validated_scan_segment(
            segment_root,
            segment_index=segment_index,
            row_group=task.row_group,
            batch_in_row_group=batch_in_row_group,
            expected_rows=batch.num_rows,
            bucket_count=task.bucket_count,
        ):
            continue
        if segment_root.exists() or segment_root.is_symlink():
            _remove_generated_tree(segment_root)
        _discard_stale_staging(segments_root, segment_name)
        staging = _staging_directory(segments_root, segment_name)
        try:
            frame = ensure_adjustment_columns(normalize_ohlcv_frame(batch.to_pandas()))
            unique_symbols = [str(value) for value in frame["symbol"].unique()]
            buckets = {
                symbol: _symbol_bucket(symbol, task.bucket_count) for symbol in unique_symbols
            }
            bucket_ids = frame["symbol"].map(buckets).to_numpy(dtype=np.int64)
            populated: list[int] = []
            for bucket in sorted(set(int(value) for value in bucket_ids)):
                selected = frame.loc[bucket_ids == bucket].reset_index(drop=True)
                _atomic_parquet(
                    selected,
                    staging / f"bucket-{bucket:04d}.parquet",
                )
                populated.append(bucket)
            atomic_write_json(
                staging / "checkpoint.json",
                {
                    "schema_version": BAR_STORE_SCHEMA_VERSION,
                    "kind": "bar-store-scan-segment",
                    "algorithm": RAW_SCAN_ALGORITHM,
                    "segment": segment_index,
                    "row_group": task.row_group,
                    "batch_in_row_group": batch_in_row_group,
                    "rows": len(frame),
                    "populated_buckets": populated,
                },
            )
            _atomic_directory(staging, segment_root)
        except Exception:
            _remove_generated_tree(staging)
            raise

    if observed_rows != task.row_group_rows or observed_batches != task.segment_count:
        raise RuntimeError(
            f"Raw row group {task.row_group} yielded {observed_rows} rows and "
            f"{observed_batches} segments; expected {task.row_group_rows} rows and "
            f"{task.segment_count} segments"
        )
    deadline.check(f"raw row group {task.row_group} checkpoint publication")
    atomic_write_json(
        segments_root / f"row-group-{task.row_group:06d}.json",
        {
            "schema_version": BAR_STORE_SCHEMA_VERSION,
            "kind": "bar-store-scan-row-group",
            "algorithm": RAW_SCAN_ALGORITHM,
            "row_group": task.row_group,
            "rows": task.row_group_rows,
            "segment_start": task.segment_start,
            "segments": observed_batches,
            "batch_rows": task.batch_rows,
        },
    )
    return {"rows": observed_rows, "segments": observed_batches}


def _write_scan_segments(
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
) -> tuple[dict[str, int], _PhasePlan]:
    """Partition raw row groups with memory-bounded, resumable processes."""

    _, pq = _require_pyarrow()
    segments_root = work_root / "segments"
    segments_root.mkdir(parents=True, exist_ok=True)
    parquet = pq.ParquetFile(raw_path)
    scan_success = segments_root / "_SUCCESS.json"

    layouts: list[tuple[int, int, int, int]] = []
    segment_count = 0
    for row_group in range(parquet.num_row_groups):
        row_group_rows = parquet.metadata.row_group(row_group).num_rows
        row_group_segments = math.ceil(row_group_rows / batch_rows)
        layouts.append((row_group, row_group_rows, segment_count, row_group_segments))
        segment_count += row_group_segments

    if scan_success.is_file():
        payload = json.loads(scan_success.read_text(encoding="utf-8"))
        completed_segments = sorted(
            path
            for path in segments_root.glob("segment-*")
            if path.is_dir() and (path / "checkpoint.json").is_file()
        )
        completed_row_groups = sorted(segments_root.glob("row-group-*.json"))
        expected_segment_names = [
            f"segment-{index:06d}" for index in range(len(completed_segments))
        ]
        if (
            payload.get("kind") != "bar-store-raw-scan-success"
            or payload.get("algorithm") != RAW_SCAN_ALGORITHM
            or payload.get("row_groups") != parquet.num_row_groups
            or payload.get("batch_rows") != batch_rows
            or payload.get("rows") != parquet.metadata.num_rows
            or payload.get("segments") != segment_count
            or len(completed_segments) != segment_count
            or len(completed_row_groups) != parquet.num_row_groups
            or [path.name for path in completed_segments] != expected_segment_names
            or [path.name for path in completed_row_groups]
            != [f"row-group-{index:06d}.json" for index in range(parquet.num_row_groups)]
        ):
            raise ValueError("Completed raw-scan checkpoint is inconsistent")
        validated_rows = 0
        for row_group, row_group_rows, segment_start, expected_segments in layouts:
            completed_batches = _completed_scan_row_group(
                segments_root,
                row_group=row_group,
                row_group_rows=row_group_rows,
                segment_start=segment_start,
                batch_rows=batch_rows,
                bucket_count=bucket_count,
            )
            if completed_batches != expected_segments:
                raise ValueError("Completed raw scan lost a row-group checkpoint")
            validated_rows += row_group_rows
        if validated_rows != payload.get("rows"):
            raise ValueError("Completed raw-scan totals are inconsistent")
        plan = _plan_phase_workers(
            phase="raw_scan",
            requested_workers=requested_workers,
            detected_cpu_count=detected_cpu_count,
            task_memory_bytes=[],
            task_count=parquet.num_row_groups,
            reused_tasks=parquet.num_row_groups,
            available_memory_bytes=available_memory_bytes,
            worker_memory_budget_bytes=worker_memory_budget_bytes,
            worker_cap=RAW_SCAN_WORKER_CAP,
        )
        _record_phase_plan(work_root, plan)
        return {
            "segments": int(payload["segments"]),
            "rows": int(payload["rows"]),
        }, plan

    pending_tasks: list[_ScanRowGroupTask] = []
    task_memory_bytes: list[int] = []
    reused_row_groups = 0
    for row_group, row_group_rows, segment_start, row_group_segments in layouts:
        completed_batches = _completed_scan_row_group(
            segments_root,
            row_group=row_group,
            row_group_rows=row_group_rows,
            segment_start=segment_start,
            batch_rows=batch_rows,
            bucket_count=bucket_count,
        )
        if completed_batches is not None:
            if completed_batches != row_group_segments:
                raise ValueError(f"Raw row group {row_group} has an inconsistent segment count")
            reused_row_groups += 1
            continue
        pending_tasks.append(
            _ScanRowGroupTask(
                raw_path=str(raw_path),
                work_root=str(work_root),
                row_group=row_group,
                row_group_rows=row_group_rows,
                segment_start=segment_start,
                segment_count=row_group_segments,
                bucket_count=bucket_count,
                batch_rows=batch_rows,
                deadline_epoch_seconds=deadline.epoch_seconds,
            )
        )
        metadata_bytes = max(
            int(parquet.metadata.row_group(row_group).total_byte_size),
            row_group_rows,
        )
        largest_batch_rows = min(batch_rows, row_group_rows)
        batch_fraction_bytes = math.ceil(
            metadata_bytes * largest_batch_rows / max(row_group_rows, 1)
        )
        task_memory_bytes.append(
            WORKER_BASE_MEMORY_BYTES + max(batch_fraction_bytes * 6, largest_batch_rows * 256)
        )

    plan = _plan_phase_workers(
        phase="raw_scan",
        requested_workers=requested_workers,
        detected_cpu_count=detected_cpu_count,
        task_memory_bytes=task_memory_bytes,
        task_count=parquet.num_row_groups,
        reused_tasks=reused_row_groups,
        available_memory_bytes=available_memory_bytes,
        worker_memory_budget_bytes=worker_memory_budget_bytes,
        worker_cap=RAW_SCAN_WORKER_CAP,
    )
    _record_phase_plan(work_root, plan)
    _run_phase_tasks(
        plan=plan,
        tasks=pending_tasks,
        runner=_scan_row_group_task,
        deadline=deadline,
    )

    validated_rows = 0
    for row_group, row_group_rows, segment_start, row_group_segments in layouts:
        completed_batches = _completed_scan_row_group(
            segments_root,
            row_group=row_group,
            row_group_rows=row_group_rows,
            segment_start=segment_start,
            batch_rows=batch_rows,
            bucket_count=bucket_count,
        )
        if completed_batches != row_group_segments:
            raise RuntimeError(f"Raw row group {row_group} was not completed")
        validated_rows += row_group_rows
    if validated_rows != parquet.metadata.num_rows:
        raise RuntimeError(
            f"Raw scan observed {validated_rows} rows; expected {parquet.metadata.num_rows}"
        )
    deadline.check("raw scan checkpoint publication")
    atomic_write_json(
        scan_success,
        {
            "schema_version": BAR_STORE_SCHEMA_VERSION,
            "kind": "bar-store-raw-scan-success",
            "algorithm": RAW_SCAN_ALGORITHM,
            "row_groups": parquet.num_row_groups,
            "batch_rows": batch_rows,
            "segments": segment_count,
            "rows": validated_rows,
        },
    )
    return {"segments": segment_count, "rows": validated_rows}, plan


def _bucket_part_paths(work_root: Path, bucket: int) -> list[Path]:
    return sorted(
        path
        for path in (work_root / "segments").glob(f"segment-*/bucket-{bucket:04d}.parquet")
        if path.is_file()
    )


def _compact_bucket(
    *,
    output_root: Path,
    work_root: Path,
    bucket: int,
    bucket_count: int,
    benchmark_mapping: Mapping[str, str],
    max_abs_log_return: float,
    deadline: _Deadline,
) -> None:
    """Sort one bounded bucket and write one row group per symbol."""

    deadline.check(f"bucket {bucket} compaction")
    bucket_name = f"bucket-{bucket:04d}"
    destination = output_root / "shards" / bucket_name
    if destination.is_dir() and (destination / "checkpoint.json").is_file():
        return
    parts = _bucket_part_paths(work_root, bucket)
    if not parts:
        return
    pa, pq = _require_pyarrow()
    tables = [pq.read_table(path) for path in parts]
    table = pa.concat_tables(tables, promote_options="default")
    frame = ensure_adjustment_columns(normalize_ohlcv_frame(table.to_pandas()))
    adjusted_close = frame["adjusted_close"].to_numpy(dtype=np.float64)
    symbols = frame["symbol"].astype(str).to_numpy()
    timestamps = pd.DatetimeIndex(frame["timestamp"])
    transitions = np.zeros(len(frame), dtype=bool)
    calendar_gap_days = np.zeros(len(frame), dtype=np.int32)
    same_symbol = symbols[1:] == symbols[:-1]
    log_returns = np.zeros(max(len(frame) - 1, 0), dtype=np.float64)
    if len(frame) > 1:
        log_returns[same_symbol] = np.diff(np.log(np.maximum(adjusted_close, 1e-12)))[same_symbol]
        transitions[1:] = same_symbol & (np.abs(log_returns) > max_abs_log_return)
        timestamp_days = timestamps.asi8 // (24 * 60 * 60 * 1_000_000_000)
        calendar_gap_days[1:] = np.where(
            same_symbol,
            np.maximum(np.diff(timestamp_days), 0),
            0,
        ).astype(np.int32)
    frame["adjusted_transition_extreme"] = transitions
    frame["calendar_gap_days"] = calendar_gap_days

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
        for symbol, rows in frame.groupby("symbol", sort=True):
            deadline.check(f"bucket {bucket} symbol {symbol} compaction")
            ordered = rows.sort_values("timestamp", kind="stable").reset_index(drop=True)
            symbol_text = str(symbol)
            asset_type = _metadata_value(ordered, "asset_type")
            market = _metadata_value(ordered, "market")
            metadata_consistent = asset_type is not None and market is not None
            decision = resolve_benchmark(
                symbol=symbol_text,
                asset_type=str(asset_type or ""),
                market=str(market or ""),
                explicit_mapping=dict(benchmark_mapping),
            )
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
                    "start_at": pd.Timestamp(ordered["timestamp"].iloc[0]).isoformat(),
                    "end_at": pd.Timestamp(ordered["timestamp"].iloc[-1]).isoformat(),
                    "asset_type": str(asset_type or "unknown"),
                    "market": str(market or "unknown"),
                    "provider": str(_metadata_value(ordered, "provider") or "unknown"),
                    "currency": str(_metadata_value(ordered, "currency") or "unknown"),
                    "source_symbol": str(_metadata_value(ordered, "source_symbol") or symbol_text),
                    "is_active": _metadata_value(ordered, "is_active"),
                    "dataset_profile": str(
                        _metadata_value(ordered, "dataset_profile") or "unknown"
                    ),
                    "eligible": bool(metadata_consistent and decision.eligible),
                    "eligibility_reason": (
                        decision.reason if metadata_consistent else "inconsistent_symbol_metadata"
                    ),
                    "benchmark_symbol": decision.benchmark_symbol or "",
                    "benchmark_policy": decision.policy,
                    "extreme_transition_count": int(ordered["adjusted_transition_extreme"].sum()),
                    "long_calendar_gap_count": int((ordered["calendar_gap_days"] > 10).sum()),
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
            "bucket": bucket,
            "bucket_count": bucket_count,
            "rows": len(frame),
            "symbols": len(index_rows),
            "shard_sha256": sha256_file(shard_path),
        },
    )
    _atomic_directory(staging, destination)


def _compact_bucket_task(task: _CompactBucketTask) -> None:
    _compact_bucket(
        output_root=Path(task.output_root),
        work_root=Path(task.work_root),
        bucket=task.bucket,
        bucket_count=task.bucket_count,
        benchmark_mapping=task.benchmark_mapping,
        max_abs_log_return=task.max_abs_log_return,
        deadline=_Deadline(task.deadline_epoch_seconds),
    )


def _run_bucket_compaction(
    *,
    output_root: Path,
    work_root: Path,
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

    pending: list[tuple[int, _CompactBucketTask]] = []
    reused_tasks = 0
    task_count = 0
    for bucket in range(bucket_count):
        parts = _bucket_part_paths(work_root, bucket)
        if not parts:
            continue
        task_count += 1
        destination = output_root / "shards" / f"bucket-{bucket:04d}"
        if destination.is_dir() and (destination / "checkpoint.json").is_file():
            reused_tasks += 1
            continue
        compressed_bytes = sum(path.stat().st_size for path in parts)
        estimated_bytes = WORKER_BASE_MEMORY_BYTES + max(
            compressed_bytes * 48,
            compressed_bytes + 128 * MIB,
        )
        pending.append(
            (
                estimated_bytes,
                _CompactBucketTask(
                    output_root=str(output_root),
                    work_root=str(work_root),
                    bucket=bucket,
                    bucket_count=bucket_count,
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
    _run_phase_tasks(
        plan=plan,
        tasks=[item[1] for item in pending],
        runner=_compact_bucket_task,
        deadline=deadline,
    )
    return plan


def _load_symbol_index(output_root: Path) -> pd.DataFrame:
    parts = sorted((output_root / "shards").glob("bucket-*/symbol-index.parquet"))
    if not parts:
        raise RuntimeError("Bar-store compaction produced no symbol index parts")
    return (
        pd.concat([pd.read_parquet(path) for path in parts], ignore_index=True)
        .sort_values("symbol", kind="stable")
        .reset_index(drop=True)
    )


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
    ranges: list[dict[str, Any]] = []
    candidate_dates: set[pd.Timestamp] = set()
    exclusions: Counter[str] = Counter()
    selected = index[(index["bucket"] == bucket) & index["eligible"].astype(bool)]
    benchmark_cache: dict[str, pd.DataFrame] = {}
    for row in selected.to_dict(orient="records"):
        deadline.check(f"candidate symbol {row['symbol']}")
        symbol = str(row["symbol"])
        benchmark_symbol = str(row["benchmark_symbol"])
        benchmark_row = index_by_symbol.get(benchmark_symbol)
        if benchmark_row is None:
            exclusions["missing_benchmark"] += 1
            continue
        frame = _read_symbol(row, output_root)
        benchmark = benchmark_cache.get(benchmark_symbol)
        if benchmark is None:
            benchmark = _read_symbol(benchmark_row, output_root)
            benchmark_cache[benchmark_symbol] = benchmark
        mask = _candidate_mask(
            frame,
            benchmark,
            window_size=window_size,
            max_horizon=max_horizon,
        )
        symbol_ranges = _true_ranges(mask)
        if not symbol_ranges:
            exclusions["no_valid_cutoffs"] += 1
            continue
        timestamps = pd.DatetimeIndex(frame["timestamp"])
        for start_index, stop_index in symbol_ranges:
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
    observed = sorted(
        {
            pd.Timestamp(value)
            for path in observed_parts
            for value in pd.read_parquet(path, columns=["timestamp"])["timestamp"]
        }
    )
    candidates = sorted(
        {
            pd.Timestamp(value)
            for path in candidate_parts
            for value in pd.read_parquet(path, columns=["timestamp"])["timestamp"]
        }
    )
    if not observed or not candidates:
        raise ValueError("No quality-approved cutoff dates were produced")
    return observed, candidates


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
    output: list[dict[str, Any]] = []
    dropped: Counter[str] = Counter()
    train_boundary = pd.Timestamp(boundaries["train_boundary"])
    validation_boundary = pd.Timestamp(boundaries["validation_boundary"])
    observed_ns = _observed_timestamp_array(boundaries["observed_index"])
    train_boundary_ns = train_boundary.value
    validation_boundary_ns = validation_boundary.value
    candidates = pd.read_parquet(candidate_part)

    try:
        for symbol, rows in candidates.groupby("symbol", sort=True):
            deadline.check(f"split assignment for {symbol}")
            symbol_text = str(symbol)
            frame = _read_symbol(index_by_symbol[symbol_text], output_root)
            timestamps = pd.DatetimeIndex(frame["timestamp"])
            timestamp_ns = timestamps.asi8
            for candidate in rows.itertuples(index=False):
                indices = np.arange(
                    int(candidate.start_index),
                    int(candidate.stop_index),
                    dtype=np.int64,
                )
                cutoff_ns = timestamp_ns[indices]
                observed_positions = np.searchsorted(observed_ns, cutoff_ns)
                if np.any(observed_positions >= len(observed_ns)) or np.any(
                    observed_ns[observed_positions] != cutoff_ns
                ):
                    raise RuntimeError(
                        f"Candidate cutoff timestamps for {symbol_text} are absent from "
                        "the global trading calendar"
                    )
                label_end_ns = timestamp_ns[indices + max_horizon]
                codes = np.zeros(len(indices), dtype=np.int8)

                train_region = observed_positions < int(boundaries["train_stop"])
                train_ok = train_region & (label_end_ns < train_boundary_ns)
                train_crossed = train_region & ~train_ok
                codes[train_ok] = 1
                dropped["label_crosses_train_boundary"] += int(train_crossed.sum())

                validation_region = (observed_positions >= int(boundaries["validation_start"])) & (
                    observed_positions < int(boundaries["validation_stop"])
                )
                validation_ok = validation_region & (label_end_ns < validation_boundary_ns)
                validation_crossed = validation_region & ~validation_ok
                codes[validation_ok] = 2
                dropped["label_crosses_validation_boundary"] += int(validation_crossed.sum())

                test_region = observed_positions >= int(boundaries["test_start"])
                codes[test_region] = 3
                unassigned = codes == 0
                dropped["purge_or_embargo"] += int(
                    unassigned.sum() - train_crossed.sum() - validation_crossed.sum()
                )

                for code, split in ((1, "train"), (2, "validation"), (3, "test")):
                    for local_start, local_stop in _true_ranges(codes == code):
                        start_index = int(indices[local_start])
                        stop_index = int(indices[local_stop - 1]) + 1
                        output.append(
                            {
                                "symbol": symbol_text,
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
        if output:
            ranges = pd.DataFrame(output).sort_values(
                ["split", "symbol", "start_index"], kind="stable"
            )
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
    ranges = pd.concat(
        [pd.read_parquet(path) for path in split_parts], ignore_index=True
    ).sort_values(["split", "symbol", "start_index"], kind="stable")
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
    return ranges.reset_index(drop=True), split_audit, plan


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
    window_size: int = 128,
    max_horizon: int = DEFAULT_MAX_HORIZON,
    max_abs_log_return: float = 0.5,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    purge_bars: int = 20,
    embargo_bars: int = 14,
    bucket_count: int = DEFAULT_BUCKET_COUNT,
    batch_rows: int = DEFAULT_BATCH_ROWS,
    deadline_epoch_seconds: float | None = None,
    workers: int = 1,
    memory_budget_bytes: int | None = None,
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
    if window_size < 2 or max_horizon != DEFAULT_MAX_HORIZON:
        raise ValueError("The approved bar store requires window_size>=2 and max_horizon=14")
    if max_abs_log_return <= 0.0:
        raise ValueError("max_abs_log_return must be positive")
    if root.is_symlink():
        raise ValueError("Bar-store root must not be a symlink")
    root.mkdir(parents=True, exist_ok=True)
    work_root = root / ".work"
    work_root.mkdir(parents=True, exist_ok=True)
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
        "raw_sha256": raw_sha256,
        "raw_rows": raw_rows,
        "window_size": window_size,
        "max_horizon": max_horizon,
        "max_abs_log_return": max_abs_log_return,
        "train_fraction": train_fraction,
        "validation_fraction": validation_fraction,
        "purge_bars": purge_bars,
        "embargo_bars": embargo_bars,
        "bucket_count": bucket_count,
        "batch_rows": batch_rows,
        "raw_scan_algorithm": RAW_SCAN_ALGORITHM,
        "split_assignment_algorithm": SPLIT_ASSIGNMENT_ALGORITHM,
        "benchmark_mapping_sha256": canonical_json_sha256(mapping),
        "training_security_scope": TRAINING_SECURITY_SCOPE,
        "split_policy": SPLIT_POLICY,
    }
    identity_sha256 = canonical_json_sha256(identity)
    state_path = work_root / "build-state.json"
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("identity_sha256") != identity_sha256:
            raise ValueError(
                "Existing bar-store checkpoints belong to a different raw or preparation contract"
            )
    else:
        atomic_write_json(
            state_path,
            {"identity": identity, "identity_sha256": identity_sha256},
        )

    success_path = root / "_SUCCESS.json"
    manifest_path = root / "bar-store.json"
    index_path = root / "symbol-index.parquet"
    ranges_path = root / "cutoff-ranges.parquet"
    if not success_path.exists() and all(
        path.is_file() for path in (manifest_path, index_path, ranges_path)
    ):
        recovered_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            recovered_manifest.get("schema_version") != BAR_STORE_SCHEMA_VERSION
            or recovered_manifest.get("kind") != BAR_STORE_KIND
            or recovered_manifest.get("state") != "ready"
            or recovered_manifest.get("identity_sha256") != identity_sha256
        ):
            raise ValueError("Recoverable bar-store outputs have an invalid contract")
        atomic_write_json(
            success_path,
            {
                "schema_version": BAR_STORE_SCHEMA_VERSION,
                "kind": "bar-store-success",
                "state": "ready",
                "identity_sha256": identity_sha256,
                "bar_store_manifest_sha256": sha256_file(manifest_path),
                "symbol_index_sha256": sha256_file(index_path),
                "cutoff_ranges_sha256": sha256_file(ranges_path),
                "split_counts": recovered_manifest["split_counts"],
                "recovered_after_interrupted_publication": True,
            },
        )
    if all(path.is_file() for path in (success_path, manifest_path, index_path, ranges_path)):
        success = json.loads(success_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            success.get("schema_version") != BAR_STORE_SCHEMA_VERSION
            or success.get("kind") != "bar-store-success"
            or success.get("state") != "ready"
            or success.get("identity_sha256") != identity_sha256
            or manifest.get("schema_version") != BAR_STORE_SCHEMA_VERSION
            or manifest.get("kind") != BAR_STORE_KIND
            or manifest.get("state") != "ready"
            or manifest.get("identity_sha256") != identity_sha256
            or success.get("bar_store_manifest_sha256") != sha256_file(manifest_path)
            or success.get("symbol_index_sha256") != sha256_file(index_path)
            or success.get("cutoff_ranges_sha256") != sha256_file(ranges_path)
            or success.get("split_counts") != manifest.get("split_counts")
        ):
            raise ValueError("Completed bar store has an invalid immutable contract")
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

    detected_cpu_count = _visible_cpu_count()
    available_memory_bytes = _detect_available_memory_bytes()
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
    scan, scan_plan = _write_scan_segments(
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
    if scan["rows"] != raw_rows:
        raise ValueError("Raw Parquet row count differs from the download manifest")
    compaction_plan = _run_bucket_compaction(
        output_root=root,
        work_root=work_root,
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
    index_by_symbol = {str(row["symbol"]): row for row in index.to_dict(orient="records")}
    missing_benchmarks = index["eligible"].astype(bool) & ~index["benchmark_symbol"].isin(
        index_by_symbol
    )
    if missing_benchmarks.any():
        index.loc[missing_benchmarks, "eligible"] = False
        index.loc[missing_benchmarks, "eligibility_reason"] = "missing_benchmark"
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
        "raw_scan_algorithm": RAW_SCAN_ALGORITHM,
        "raw_scan_batch_rows": batch_rows,
        "raw_scan_segments": scan["segments"],
        "parallelism": {
            "backend": MULTIPROCESS_BACKEND,
            "requested_workers": workers,
            "detected_cpu_count": detected_cpu_count,
            "available_memory_bytes_at_planning": available_memory_bytes,
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
            "raw_segments",
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
    atomic_write_json(
        success_path,
        {
            "schema_version": BAR_STORE_SCHEMA_VERSION,
            "kind": "bar-store-success",
            "state": "ready",
            "identity_sha256": identity_sha256,
            "bar_store_manifest_sha256": sha256_file(manifest_path),
            "symbol_index_sha256": sha256_file(index_path),
            "cutoff_ranges_sha256": sha256_file(ranges_path),
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


def bar_store_preparation_spec(
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
    """Return the immutable, h_start-independent lazy dataset contract."""

    return {
        "schema_version": 5,
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
