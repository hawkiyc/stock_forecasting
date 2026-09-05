"""Single-GPU quant-only Kronos LoRA training and evaluation."""

from __future__ import annotations

import json
import math
import os
import random
import time
from contextlib import nullcontext, suppress
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterator, cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, Sampler

from stock_forecasting.checkpointing import (
    load_checkpoint,
    reconcile_checkpoint_storage,
    save_ranked_checkpoint,
    save_training_completion_result,
    validate_checkpoint_selection,
    validate_checkpoint_trainer_state,
)
from stock_forecasting.config import ExperimentConfig
from stock_forecasting.data import (
    BlockwisePermutationSampler,
    FinancialBatchCollator,
    FixedSizeBatchSampler,
    LazyFinancialWindowDataset,
)
from stock_forecasting.data.manifest import (
    atomic_write_json,
    canonical_json_sha256,
    sha256_file,
)
from stock_forecasting.factory import ModelBundle, build_model_bundle
from stock_forecasting.metrics import (
    POSTPROCESS_SIGNAL_NAMES,
    cross_sectional_metrics,
    multi_horizon_alpha_metrics,
    postprocess_alpha_signal,
)
from stock_forecasting.preflight import run_preflight
from stock_forecasting.runtime_resources import (
    AvailableMemoryEstimate,
    CGROUP_V1_UNLIMITED_THRESHOLD_BYTES,
    detect_available_memory,
    detect_visible_cpu_count,
)
from stock_forecasting.tracking import (
    TrackingRun,
    start_tracking,
    training_run_lease,
    validate_tracking_run_contract,
)
from stock_forecasting.training_paths import resolve_bar_store_path

DATALOADER_AUTO_MAX_WORKERS = 128
DATALOADER_CPU_RESERVE = 2
DATALOADER_ACTIVE_PERSISTENT_POOLS = 2
DATALOADER_MEMORY_FRACTION = 0.25
DATALOADER_MEMORY_BYTES_PER_WORKER = 512 * 1024**2
DATALOADER_PARENT_MEMORY_RESERVE_BYTES = 4 * 1024**3
DATALOADER_TOTAL_SYMBOL_CACHE_ENTRIES = 128
DATALOADER_INITIAL_PREFETCH_FACTOR = 2
DATALOADER_SELECTION_BLOCK_SIZE = 128
DATALOADER_PREFETCH_MEMORY_FRACTION = 0.10
AUTO_BATCH_THROUGHPUT_TOLERANCE = 0.03
AUTO_BATCH_EXPANSION_THROUGHPUT_TOLERANCE = 0.10
AUTO_BATCH_EXPANSION_MAX_SIZE = 4_096
AUTO_EVALUATION_BATCH_EXPANSION_MAX_SIZE = 8_192
AUTO_BATCH_EXPANSION_MEMORY_QUANTUM_BYTES = 24 * 1024**3
CUDA_FREE_MEMORY_FRACTION = 0.90
ROBUST_SCALE_BATCH_SIZE = 256
ROBUST_SCALE_SELECTION_BLOCK_SIZE = 16
ROBUST_SCALE_CACHE_SCHEMA_VERSION = "1.0"
ROBUST_SCALE_ALGORITHM = "parallel-blockwise-runtime-labels-v1"
TRAINING_PROGRESS_SCHEMA_VERSION = "2.0"
RUNTIME_EXECUTION_PLAN_SCHEMA_VERSION = "1.0"
CANONICAL_TRAINING_SCHEDULE_SCHEMA_VERSION = "1.0"
_DATALOADER_THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "PYARROW_NUM_THREADS",
)


@dataclass(frozen=True)
class DataLoaderWorkerPlan:
    """Bound data-loading parallelism by visible CPU and available host memory."""

    source: str
    requested_workers: int
    visible_cpu_count: int
    available_memory_bytes: int
    available_memory_source: str
    available_memory_observations: tuple[tuple[str, int], ...]
    worker_memory_budget_bytes: int
    effective_workers: int
    active_persistent_pools: int
    symbol_cache_size_per_worker: int
    prefetch_factor: int | None
    prefetched_batches_per_pool: int
    prefetch_memory_budget_bytes: int
    estimated_peak_prefetch_memory_bytes: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "requested_workers": self.requested_workers,
            "visible_cpu_count": self.visible_cpu_count,
            "available_memory_bytes": self.available_memory_bytes,
            "available_memory_source": self.available_memory_source,
            "available_memory_observations_bytes": dict(
                self.available_memory_observations
            ),
            "worker_memory_budget_bytes": self.worker_memory_budget_bytes,
            "effective_workers": self.effective_workers,
            "active_persistent_pools": self.active_persistent_pools,
            "estimated_peak_worker_memory_bytes": (
                self.effective_workers
                * self.active_persistent_pools
                * DATALOADER_MEMORY_BYTES_PER_WORKER
            ),
            "memory_bytes_per_worker": DATALOADER_MEMORY_BYTES_PER_WORKER,
            "symbol_cache_size_per_worker": self.symbol_cache_size_per_worker,
            "prefetch_factor": self.prefetch_factor,
            "prefetched_batches_per_pool": self.prefetched_batches_per_pool,
            "prefetch_memory_budget_bytes": self.prefetch_memory_budget_bytes,
            "estimated_peak_prefetch_memory_bytes": (
                self.estimated_peak_prefetch_memory_bytes
            ),
            "native_threads_per_worker": 1,
        }

    @classmethod
    def from_dict(
        cls,
        payload: Any,
        *,
        source: str | None = None,
    ) -> DataLoaderWorkerPlan:
        if not isinstance(payload, dict):
            raise ValueError("Checkpoint DataLoader worker plan must be a mapping")
        expected = {
            "source",
            "requested_workers",
            "visible_cpu_count",
            "available_memory_bytes",
            "available_memory_source",
            "available_memory_observations_bytes",
            "worker_memory_budget_bytes",
            "effective_workers",
            "active_persistent_pools",
            "estimated_peak_worker_memory_bytes",
            "memory_bytes_per_worker",
            "symbol_cache_size_per_worker",
            "prefetch_factor",
            "prefetched_batches_per_pool",
            "prefetch_memory_budget_bytes",
            "estimated_peak_prefetch_memory_bytes",
            "native_threads_per_worker",
        }
        if set(payload) != expected:
            raise ValueError("Checkpoint DataLoader worker plan is incomplete")
        if not isinstance(payload["source"], str) or not isinstance(
            payload["available_memory_source"], str
        ):
            raise ValueError("Checkpoint DataLoader worker labels are invalid")
        observations = payload["available_memory_observations_bytes"]
        if not isinstance(observations, dict) or not observations or any(
            not isinstance(key, str)
            or not isinstance(value, int)
            or isinstance(value, bool)
            or value < 1
            for key, value in observations.items()
        ):
            raise ValueError("Checkpoint memory observations are invalid")
        integer_fields = (
            "requested_workers",
            "visible_cpu_count",
            "available_memory_bytes",
            "worker_memory_budget_bytes",
            "effective_workers",
            "active_persistent_pools",
            "symbol_cache_size_per_worker",
            "prefetched_batches_per_pool",
            "prefetch_memory_budget_bytes",
            "estimated_peak_prefetch_memory_bytes",
        )
        if any(
            not isinstance(payload[name], int)
            or isinstance(payload[name], bool)
            or payload[name] < 0
            for name in integer_fields
        ):
            raise ValueError("Checkpoint DataLoader worker counters are invalid")
        if payload["visible_cpu_count"] < 1 or payload["available_memory_bytes"] < 1:
            raise ValueError("Checkpoint CPU and memory capacity must be positive")
        if payload["active_persistent_pools"] != DATALOADER_ACTIVE_PERSISTENT_POOLS:
            raise ValueError("Checkpoint DataLoader pool count is unsupported")
        if payload["memory_bytes_per_worker"] != DATALOADER_MEMORY_BYTES_PER_WORKER:
            raise ValueError("Checkpoint DataLoader worker-memory estimate is unsupported")
        if payload["native_threads_per_worker"] != 1:
            raise ValueError("Checkpoint DataLoader native-thread count is unsupported")
        prefetch_factor = payload["prefetch_factor"]
        if prefetch_factor is not None and (
            not isinstance(prefetch_factor, int)
            or isinstance(prefetch_factor, bool)
            or prefetch_factor < 1
        ):
            raise ValueError("Checkpoint DataLoader prefetch factor is invalid")
        plan = cls(
            source=source or str(payload["source"]),
            requested_workers=int(payload["requested_workers"]),
            visible_cpu_count=int(payload["visible_cpu_count"]),
            available_memory_bytes=int(payload["available_memory_bytes"]),
            available_memory_source=str(payload["available_memory_source"]),
            available_memory_observations=tuple(
                sorted((str(key), int(value)) for key, value in observations.items())
            ),
            worker_memory_budget_bytes=int(payload["worker_memory_budget_bytes"]),
            effective_workers=int(payload["effective_workers"]),
            active_persistent_pools=int(payload["active_persistent_pools"]),
            symbol_cache_size_per_worker=int(payload["symbol_cache_size_per_worker"]),
            prefetch_factor=prefetch_factor,
            prefetched_batches_per_pool=int(payload["prefetched_batches_per_pool"]),
            prefetch_memory_budget_bytes=int(payload["prefetch_memory_budget_bytes"]),
            estimated_peak_prefetch_memory_bytes=int(
                payload["estimated_peak_prefetch_memory_bytes"]
            ),
        )
        if payload["estimated_peak_worker_memory_bytes"] != (
            plan.effective_workers
            * plan.active_persistent_pools
            * DATALOADER_MEMORY_BYTES_PER_WORKER
        ):
            raise ValueError("Checkpoint DataLoader worker-memory total is inconsistent")
        expected_prefetched_batches = (
            0
            if plan.prefetch_factor is None
            else plan.effective_workers * plan.prefetch_factor
        )
        if plan.prefetched_batches_per_pool != expected_prefetched_batches:
            raise ValueError("Checkpoint DataLoader prefetch count is inconsistent")
        if plan.effective_workers == 0 and plan.prefetch_factor is not None:
            raise ValueError("Checkpoint serial DataLoader cannot prefetch in workers")
        if plan.effective_workers > 0 and plan.prefetch_factor is None:
            raise ValueError("Checkpoint multiprocessing DataLoader requires prefetching")
        return plan


@dataclass(frozen=True)
class BatchProbeMeasurement:
    """One empirical CUDA batch-size measurement."""

    batch_size: int
    seconds_per_batch: float | None
    samples_per_second: float | None
    peak_allocated_bytes: int | None
    projected_peak_bytes: int | None
    accepted: bool
    outcome: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "seconds_per_batch": self.seconds_per_batch,
            "samples_per_second": self.samples_per_second,
            "peak_allocated_bytes": self.peak_allocated_bytes,
            "projected_peak_bytes": self.projected_peak_bytes,
            "accepted": self.accepted,
            "outcome": self.outcome,
        }


@dataclass(frozen=True)
class RuntimeBatchPlan:
    """Hardware-resolved micro-batches with a stable effective-batch floor."""

    source: str
    training_batch_size: int
    evaluation_batch_size: int
    gradient_accumulation_steps: int
    effective_batch_size: int
    target_effective_batch_size: int
    device_name: str
    device_total_memory_bytes: int
    optimizer_state_reserve_bytes: int
    seconds_per_training_batch: float | None
    training_probe: tuple[BatchProbeMeasurement, ...]
    evaluation_probe: tuple[BatchProbeMeasurement, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "training_batch_size": self.training_batch_size,
            "evaluation_batch_size": self.evaluation_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "effective_batch_size": self.effective_batch_size,
            "target_effective_batch_size": self.target_effective_batch_size,
            "device_name": self.device_name,
            "device_total_memory_bytes": self.device_total_memory_bytes,
            "optimizer_state_reserve_bytes": self.optimizer_state_reserve_bytes,
            "seconds_per_training_batch": self.seconds_per_training_batch,
            "training_probe": [measurement.as_dict() for measurement in self.training_probe],
            "evaluation_probe": [measurement.as_dict() for measurement in self.evaluation_probe],
        }

    @classmethod
    def from_dict(cls, payload: Any, *, source: str | None = None) -> RuntimeBatchPlan:
        if not isinstance(payload, dict):
            raise ValueError("Checkpoint runtime batch plan must be a mapping")
        required = {
            "source",
            "training_batch_size",
            "evaluation_batch_size",
            "gradient_accumulation_steps",
            "effective_batch_size",
            "target_effective_batch_size",
            "device_name",
            "device_total_memory_bytes",
            "optimizer_state_reserve_bytes",
            "seconds_per_training_batch",
            "training_probe",
            "evaluation_probe",
        }
        if set(payload) != required:
            raise ValueError("Checkpoint runtime batch plan is incomplete")

        def measurements(value: Any) -> tuple[BatchProbeMeasurement, ...]:
            if not isinstance(value, list):
                raise ValueError("Checkpoint batch probe must be a list")
            rows: list[BatchProbeMeasurement] = []
            expected = {
                "batch_size",
                "seconds_per_batch",
                "samples_per_second",
                "peak_allocated_bytes",
                "projected_peak_bytes",
                "accepted",
                "outcome",
            }
            for row in value:
                if not isinstance(row, dict) or set(row) != expected:
                    raise ValueError("Checkpoint batch probe row is invalid")
                rows.append(BatchProbeMeasurement(**row))
            return tuple(rows)

        plan = cls(
            source=source or str(payload["source"]),
            training_batch_size=int(payload["training_batch_size"]),
            evaluation_batch_size=int(payload["evaluation_batch_size"]),
            gradient_accumulation_steps=int(payload["gradient_accumulation_steps"]),
            effective_batch_size=int(payload["effective_batch_size"]),
            target_effective_batch_size=int(payload["target_effective_batch_size"]),
            device_name=str(payload["device_name"]),
            device_total_memory_bytes=int(payload["device_total_memory_bytes"]),
            optimizer_state_reserve_bytes=int(payload["optimizer_state_reserve_bytes"]),
            seconds_per_training_batch=(
                None
                if payload["seconds_per_training_batch"] is None
                else float(payload["seconds_per_training_batch"])
            ),
            training_probe=measurements(payload["training_probe"]),
            evaluation_probe=measurements(payload["evaluation_probe"]),
        )
        positive = (
            plan.training_batch_size,
            plan.evaluation_batch_size,
            plan.gradient_accumulation_steps,
            plan.effective_batch_size,
            plan.target_effective_batch_size,
        )
        if any(value < 1 for value in positive):
            raise ValueError("Checkpoint runtime batch sizes must be positive")
        if plan.effective_batch_size != (
            plan.training_batch_size * plan.gradient_accumulation_steps
        ):
            raise ValueError("Checkpoint runtime effective batch size is inconsistent")
        return plan


@dataclass(frozen=True)
class RuntimeHardwareSnapshot:
    """Stable accelerator and host capacities used to decide plan reuse."""

    requested_gpu_id: str
    device_type: str
    device_name: str
    device_total_memory_bytes: int
    compute_capability: str
    host_memory_capacity_bytes: int
    host_memory_capacity_source: str
    visible_cpu_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "requested_gpu_id": self.requested_gpu_id,
            "device_type": self.device_type,
            "device_name": self.device_name,
            "device_total_memory_bytes": self.device_total_memory_bytes,
            "compute_capability": self.compute_capability,
            "host_memory_capacity_bytes": self.host_memory_capacity_bytes,
            "host_memory_capacity_source": self.host_memory_capacity_source,
            "visible_cpu_count": self.visible_cpu_count,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> RuntimeHardwareSnapshot:
        expected = {
            "requested_gpu_id",
            "device_type",
            "device_name",
            "device_total_memory_bytes",
            "compute_capability",
            "host_memory_capacity_bytes",
            "host_memory_capacity_source",
            "visible_cpu_count",
        }
        if not isinstance(payload, dict) or set(payload) != expected:
            raise ValueError("Checkpoint runtime hardware snapshot is incomplete")
        for name in (
            "requested_gpu_id",
            "device_type",
            "device_name",
            "compute_capability",
            "host_memory_capacity_source",
        ):
            if not isinstance(payload[name], str):
                raise ValueError("Checkpoint runtime hardware text field is invalid")
        for name in (
            "device_total_memory_bytes",
            "host_memory_capacity_bytes",
            "visible_cpu_count",
        ):
            if (
                not isinstance(payload[name], int)
                or isinstance(payload[name], bool)
                or payload[name] < 0
            ):
                raise ValueError("Checkpoint runtime hardware capacity is invalid")
        if payload["visible_cpu_count"] < 1 or payload["host_memory_capacity_bytes"] < 1:
            raise ValueError("Checkpoint host capacity must be positive")
        if payload["device_type"] not in {"cpu", "cuda"}:
            raise ValueError("Checkpoint runtime device type is unsupported")
        if payload["device_type"] == "cuda" and payload["device_total_memory_bytes"] < 1:
            raise ValueError("Checkpoint CUDA memory capacity must be positive")
        return cls(**payload)

    def capacity_identity(self) -> tuple[Any, ...]:
        """Exclude transient free-memory observations from hardware identity."""

        return (
            self.requested_gpu_id,
            self.device_type,
            self.device_name,
            self.device_total_memory_bytes,
            self.compute_capability,
            self.host_memory_capacity_bytes,
            self.visible_cpu_count,
        )


@dataclass(frozen=True)
class CanonicalTrainingSchedule:
    """Immutable step coordinates established by the first training Pod."""

    epochs: int
    evaluations_per_epoch: int
    optimizer_steps_per_epoch: int
    configured_optimizer_steps: int

    def __post_init__(self) -> None:
        values = (
            self.epochs,
            self.evaluations_per_epoch,
            self.optimizer_steps_per_epoch,
            self.configured_optimizer_steps,
        )
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 1
            for value in values
        ):
            raise ValueError("Canonical training schedule is invalid")
        if self.configured_optimizer_steps != self.epochs * self.optimizer_steps_per_epoch:
            raise ValueError("Canonical optimizer-step budget is inconsistent")
        if self.optimizer_steps_per_epoch < self.evaluations_per_epoch:
            raise ValueError("Canonical validation schedule is impossible")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CANONICAL_TRAINING_SCHEDULE_SCHEMA_VERSION,
            "epochs": self.epochs,
            "evaluations_per_epoch": self.evaluations_per_epoch,
            "optimizer_steps_per_epoch": self.optimizer_steps_per_epoch,
            "configured_optimizer_steps": self.configured_optimizer_steps,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> CanonicalTrainingSchedule:
        expected = {
            "schema_version",
            "epochs",
            "evaluations_per_epoch",
            "optimizer_steps_per_epoch",
            "configured_optimizer_steps",
        }
        if not isinstance(payload, dict) or set(payload) != expected:
            raise ValueError("Checkpoint canonical training schedule is incomplete")
        if payload["schema_version"] != CANONICAL_TRAINING_SCHEDULE_SCHEMA_VERSION:
            raise ValueError("Checkpoint canonical training schedule version is unsupported")
        schedule = cls(
            epochs=payload["epochs"],
            evaluations_per_epoch=payload["evaluations_per_epoch"],
            optimizer_steps_per_epoch=payload["optimizer_steps_per_epoch"],
            configured_optimizer_steps=payload["configured_optimizer_steps"],
        )
        return schedule


@dataclass(frozen=True)
class RuntimeExecutionPlan:
    """One Pod's hardware-specific data-loading and optimizer-step plan."""

    source: str
    replan_reason: str
    hardware: RuntimeHardwareSnapshot
    batch_plan: RuntimeBatchPlan
    worker_plan: DataLoaderWorkerPlan
    optimizer_steps_per_epoch: int
    configured_optimizer_steps: int
    resume_canonical_global_step: int
    resume_runtime_global_step: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": RUNTIME_EXECUTION_PLAN_SCHEMA_VERSION,
            "source": self.source,
            "replan_reason": self.replan_reason,
            "hardware": self.hardware.as_dict(),
            "batch_plan": self.batch_plan.as_dict(),
            "dataloader_worker_plan": self.worker_plan.as_dict(),
            "optimizer_steps_per_epoch": self.optimizer_steps_per_epoch,
            "configured_optimizer_steps": self.configured_optimizer_steps,
            "resume_canonical_global_step": self.resume_canonical_global_step,
            "resume_runtime_global_step": self.resume_runtime_global_step,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> RuntimeExecutionPlan:
        expected = {
            "schema_version",
            "source",
            "replan_reason",
            "hardware",
            "batch_plan",
            "dataloader_worker_plan",
            "optimizer_steps_per_epoch",
            "configured_optimizer_steps",
            "resume_canonical_global_step",
            "resume_runtime_global_step",
        }
        if not isinstance(payload, dict) or set(payload) != expected:
            raise ValueError("Checkpoint runtime execution plan is incomplete")
        if payload["schema_version"] != RUNTIME_EXECUTION_PLAN_SCHEMA_VERSION:
            raise ValueError("Checkpoint runtime execution plan version is unsupported")
        if not isinstance(payload["source"], str) or not isinstance(
            payload["replan_reason"], str
        ):
            raise ValueError("Checkpoint runtime execution plan labels are invalid")
        counters = tuple(
            payload[name]
            for name in (
                "optimizer_steps_per_epoch",
                "configured_optimizer_steps",
                "resume_canonical_global_step",
                "resume_runtime_global_step",
            )
        )
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in counters
        ):
            raise ValueError("Checkpoint runtime execution counters are invalid")
        if payload["optimizer_steps_per_epoch"] < 1:
            raise ValueError("Checkpoint runtime optimizer-step count must be positive")
        if (
            payload["configured_optimizer_steps"] < payload["optimizer_steps_per_epoch"]
            or payload["configured_optimizer_steps"]
            % payload["optimizer_steps_per_epoch"]
            != 0
            or payload["resume_runtime_global_step"]
            > payload["configured_optimizer_steps"]
        ):
            raise ValueError("Checkpoint runtime optimizer-step budget is inconsistent")
        return cls(
            source=payload["source"],
            replan_reason=payload["replan_reason"],
            hardware=RuntimeHardwareSnapshot.from_dict(payload["hardware"]),
            batch_plan=RuntimeBatchPlan.from_dict(payload["batch_plan"]),
            worker_plan=DataLoaderWorkerPlan.from_dict(
                payload["dataloader_worker_plan"]
            ),
            optimizer_steps_per_epoch=payload["optimizer_steps_per_epoch"],
            configured_optimizer_steps=payload["configured_optimizer_steps"],
            resume_canonical_global_step=payload["resume_canonical_global_step"],
            resume_runtime_global_step=payload["resume_runtime_global_step"],
        )


@dataclass(frozen=True)
class ResumeCoordinates:
    """Equivalent canonical and current-runtime positions for one checkpoint."""

    starting_epoch: int
    resume_batch_index: int
    canonical_global_step: int
    runtime_global_step: int


class ResumableFixedSizeBatchSampler(Sampler[list[int]]):
    """Skip completed batch indices before workers read network-volume data."""

    def __init__(self, source: FixedSizeBatchSampler) -> None:
        self.source = source
        self.start_batch_index = 0

    def __len__(self) -> int:
        return len(self.source)

    @property
    def sampler(self) -> BlockwisePermutationSampler:
        return self.source.sampler

    @property
    def padded_sample_count(self) -> int:
        return self.source.padded_sample_count

    def set_epoch(self, epoch: int, *, start_batch_index: int = 0) -> None:
        if not 0 <= start_batch_index <= len(self.source):
            raise ValueError("Resume batch index is outside the training epoch")
        self.source.set_epoch(epoch)
        self.start_batch_index = start_batch_index

    def __iter__(self) -> Iterator[list[int]]:
        for batch_index, batch in enumerate(self.source):
            if batch_index >= self.start_batch_index:
                yield batch


@dataclass(frozen=True)
class RuntimeRobustScaleResult:
    """Resolved robust scales plus their persistent cache provenance."""

    scales: tuple[float, ...]
    cache_path: Path
    cache_hit: bool
    identity_sha256: str
    sample_count: int


class _RuntimeLabelDataset(Dataset[Tensor]):
    """Expose only on-demand labels to the calibration DataLoader."""

    def __init__(self, source: LazyFinancialWindowDataset) -> None:
        self.source = source

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int) -> Tensor:
        return self.source.target_at(index)


def _available_memory_bytes() -> int:
    """Return scope-compatible available memory for compatibility callers."""

    return detect_available_memory().available_bytes


def _read_positive_integer(path: Path) -> int | None:
    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not value.isascii() or not value.isdigit():
        return None
    parsed = int(value)
    return parsed if parsed > 0 else None


def _linux_total_memory_bytes() -> int | None:
    try:
        lines = Path("/proc/meminfo").read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        if not line.startswith("MemTotal:"):
            continue
        fields = line.split()
        if len(fields) == 3 and fields[1].isdigit() and fields[2] == "kB":
            return int(fields[1]) * 1024
    return None


def _host_memory_capacity() -> tuple[int, str]:
    """Return a stable container memory ceiling instead of transient headroom."""

    candidates: list[tuple[str, int]] = []
    cgroup_v2 = _read_positive_integer(Path("/sys/fs/cgroup/memory.max"))
    if (
        cgroup_v2 is not None
        and cgroup_v2 < CGROUP_V1_UNLIMITED_THRESHOLD_BYTES
    ):
        candidates.append(("cgroup_v2_limit", cgroup_v2))
    cgroup_v1 = _read_positive_integer(
        Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    )
    if (
        cgroup_v1 is not None
        and cgroup_v1 < CGROUP_V1_UNLIMITED_THRESHOLD_BYTES
    ):
        candidates.append(("cgroup_v1_limit", cgroup_v1))
    linux_total = _linux_total_memory_bytes()
    if linux_total is not None:
        candidates.append(("linux_mem_total", linux_total))
    if not candidates:
        estimate = detect_available_memory()
        return estimate.available_bytes, f"available_fallback:{estimate.source}"
    source, capacity = min(candidates, key=lambda row: row[1])
    return capacity, source


def runtime_hardware_snapshot(device: torch.device) -> RuntimeHardwareSnapshot:
    """Capture stable hardware identity plus the user-requested RunPod GPU ID."""

    device_name = "cpu"
    total_device_memory = 0
    compute_capability = ""
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        device_name = properties.name
        total_device_memory = int(properties.total_memory)
        major, minor = torch.cuda.get_device_capability(device)
        compute_capability = f"{major}.{minor}"
    host_memory_capacity, host_memory_source = _host_memory_capacity()
    return RuntimeHardwareSnapshot(
        requested_gpu_id=os.environ.get("RUNPOD_REQUESTED_GPU_ID", "").strip(),
        device_type=device.type,
        device_name=device_name,
        device_total_memory_bytes=total_device_memory,
        compute_capability=compute_capability,
        host_memory_capacity_bytes=host_memory_capacity,
        host_memory_capacity_source=host_memory_source,
        visible_cpu_count=detect_visible_cpu_count(),
    )


def _requested_dataloader_workers(config: ExperimentConfig) -> tuple[int, str]:
    value = os.environ.get("FIN_TS_DATALOADER_WORKERS", "").strip().lower()
    if not value:
        if config.training.num_workers == "auto":
            return DATALOADER_AUTO_MAX_WORKERS, "training_config_auto"
        return config.training.num_workers, "training_config"
    if value == "auto":
        return DATALOADER_AUTO_MAX_WORKERS, "runpod_auto"
    if (
        not value.isascii()
        or not value.isdigit()
        or int(value) > DATALOADER_AUTO_MAX_WORKERS
    ):
        raise ValueError(
            "FIN_TS_DATALOADER_WORKERS must be auto or an integer from 0 to "
            f"{DATALOADER_AUTO_MAX_WORKERS}"
        )
    return int(value), "environment"


def plan_dataloader_workers(
    requested_workers: int,
    *,
    source: str = "explicit",
    visible_cpu_count: int | None = None,
    available_memory_bytes: int | None = None,
) -> DataLoaderWorkerPlan:
    """Choose a conservative process count shared by loading and calibration."""

    if (
        isinstance(requested_workers, bool)
        or requested_workers < 0
        or requested_workers > DATALOADER_AUTO_MAX_WORKERS
    ):
        raise ValueError(
            "requested_workers must be between 0 and "
            f"{DATALOADER_AUTO_MAX_WORKERS}"
        )
    cpu_count = (
        detect_visible_cpu_count()
        if visible_cpu_count is None
        else visible_cpu_count
    )
    memory_estimate = (
        detect_available_memory()
        if available_memory_bytes is None
        else AvailableMemoryEstimate(
            available_bytes=available_memory_bytes,
            source="explicit_argument",
            observations=(("explicit_argument", available_memory_bytes),),
        )
    )
    memory_bytes = memory_estimate.available_bytes
    if (
        isinstance(cpu_count, bool)
        or isinstance(memory_bytes, bool)
        or cpu_count < 1
        or memory_bytes < 1
    ):
        raise ValueError("Visible CPU count and available memory must be positive")

    cpu_reserve = DATALOADER_CPU_RESERVE if cpu_count >= 4 else 1
    cpu_worker_limit = max(cpu_count - cpu_reserve, 0)
    memory_budget = max(
        0,
        min(
            int(memory_bytes * DATALOADER_MEMORY_FRACTION),
            memory_bytes - DATALOADER_PARENT_MEMORY_RESERVE_BYTES,
        ),
    )
    memory_worker_limit = memory_budget // (
        DATALOADER_MEMORY_BYTES_PER_WORKER * DATALOADER_ACTIVE_PERSISTENT_POOLS
    )
    effective_workers = min(
        requested_workers,
        cpu_worker_limit,
        int(memory_worker_limit),
    )
    cache_divisor = max(
        effective_workers * DATALOADER_ACTIVE_PERSISTENT_POOLS,
        1,
    )
    symbol_cache_size = max(
        4,
        min(32, DATALOADER_TOTAL_SYMBOL_CACHE_ENTRIES // cache_divisor),
    )
    return DataLoaderWorkerPlan(
        source=source,
        requested_workers=requested_workers,
        visible_cpu_count=cpu_count,
        available_memory_bytes=memory_bytes,
        available_memory_source=memory_estimate.source,
        available_memory_observations=memory_estimate.observations,
        worker_memory_budget_bytes=memory_budget,
        effective_workers=effective_workers,
        active_persistent_pools=DATALOADER_ACTIVE_PERSISTENT_POOLS,
        symbol_cache_size_per_worker=symbol_cache_size,
        prefetch_factor=(
            DATALOADER_INITIAL_PREFETCH_FACTOR if effective_workers else None
        ),
        prefetched_batches_per_pool=(
            effective_workers * DATALOADER_INITIAL_PREFETCH_FACTOR
        ),
        prefetch_memory_budget_bytes=0,
        estimated_peak_prefetch_memory_bytes=0,
    )


def _batch_tensor_bytes(batch: dict[str, Any]) -> int:
    return sum(
        value.numel() * value.element_size()
        for value in batch.values()
        if isinstance(value, Tensor)
    )


def plan_runtime_prefetch(
    worker_plan: DataLoaderWorkerPlan,
    *,
    config: ExperimentConfig,
    batch_plan: RuntimeBatchPlan,
    largest_host_batch_bytes: int,
) -> DataLoaderWorkerPlan:
    """Size the worker queue from measured GPU demand and host-memory headroom."""

    if (
        isinstance(largest_host_batch_bytes, bool)
        or largest_host_batch_bytes < 1
    ):
        raise ValueError("largest_host_batch_bytes must be a positive integer")
    workers = worker_plan.effective_workers
    if workers == 0:
        return replace(worker_plan, prefetch_factor=None)
    queue_budget = max(
        largest_host_batch_bytes * worker_plan.active_persistent_pools,
        int(
            worker_plan.available_memory_bytes
            * DATALOADER_PREFETCH_MEMORY_FRACTION
        ),
    )
    memory_worker_limit = max(
        1,
        queue_budget
        // (largest_host_batch_bytes * worker_plan.active_persistent_pools),
    )
    workers = min(workers, int(memory_worker_limit))
    seconds_per_batch = batch_plan.seconds_per_training_batch
    if seconds_per_batch is None or seconds_per_batch <= 0.0:
        demand_limit = DATALOADER_INITIAL_PREFETCH_FACTOR
    else:
        buffered_batches = math.ceil(
            config.training.dataloader_prefetch_target_seconds / seconds_per_batch
        )
        demand_limit = max(2, math.ceil(buffered_batches / workers))
    per_factor_bytes = max(
        largest_host_batch_bytes
        * workers
        * worker_plan.active_persistent_pools,
        1,
    )
    memory_limit = max(1, queue_budget // per_factor_bytes)
    factor = max(
        1,
        min(
            demand_limit,
            memory_limit,
            config.training.dataloader_max_prefetch_factor,
        ),
    )
    return replace(
        worker_plan,
        effective_workers=workers,
        prefetch_factor=int(factor),
        prefetched_batches_per_pool=workers * int(factor),
        prefetch_memory_budget_bytes=queue_budget,
        estimated_peak_prefetch_memory_bytes=(
            largest_host_batch_bytes
            * workers
            * int(factor)
            * worker_plan.active_persistent_pools
        ),
    )


def runtime_resource_plan_reuse_reason(
    stored_plan: RuntimeExecutionPlan | None,
    *,
    current_hardware: RuntimeHardwareSnapshot,
    current_worker_plan: DataLoaderWorkerPlan,
) -> str:
    """Explain whether a checkpoint's hardware-specific plan remains safe."""

    if stored_plan is None:
        return "checkpoint_has_no_hardware_plan"
    if stored_plan.hardware.capacity_identity() != current_hardware.capacity_identity():
        return "hardware_capacity_changed"
    if (
        stored_plan.batch_plan.device_name != current_hardware.device_name
        or stored_plan.batch_plan.device_total_memory_bytes
        != current_hardware.device_total_memory_bytes
    ):
        return "accelerator_batch_plan_identity_changed"
    stored_workers = stored_plan.worker_plan
    if stored_workers.requested_workers != current_worker_plan.requested_workers:
        return "requested_worker_count_changed"
    if stored_workers.effective_workers > current_worker_plan.effective_workers:
        return "current_host_memory_headroom_is_lower"
    current_prefetch_limit = int(
        current_worker_plan.available_memory_bytes
        * DATALOADER_PREFETCH_MEMORY_FRACTION
    )
    if stored_workers.estimated_peak_prefetch_memory_bytes > current_prefetch_limit:
        return "current_prefetch_memory_headroom_is_lower"
    return "checkpoint_hardware_match"


def reuse_dataloader_worker_plan(
    stored_plan: DataLoaderWorkerPlan,
    current_plan: DataLoaderWorkerPlan,
) -> DataLoaderWorkerPlan:
    """Reuse queue dimensions while retaining current memory observations."""

    return replace(
        current_plan,
        source="checkpoint_hardware_match",
        effective_workers=stored_plan.effective_workers,
        symbol_cache_size_per_worker=stored_plan.symbol_cache_size_per_worker,
        prefetch_factor=stored_plan.prefetch_factor,
        prefetched_batches_per_pool=stored_plan.prefetched_batches_per_pool,
        prefetch_memory_budget_bytes=max(
            stored_plan.estimated_peak_prefetch_memory_bytes,
            int(
                current_plan.available_memory_bytes
                * DATALOADER_PREFETCH_MEMORY_FRACTION
            ),
        ),
        estimated_peak_prefetch_memory_bytes=(
            stored_plan.estimated_peak_prefetch_memory_bytes
        ),
    )


def _initialize_dataloader_worker(_worker_id: int) -> None:
    """Prevent every loader process from creating its own native thread pools."""

    for name in _DATALOADER_THREAD_ENVIRONMENT:
        os.environ[name] = "1"
    torch.set_num_threads(1)
    try:
        import pyarrow as pa
    except ImportError:
        return
    pa.set_cpu_count(1)
    if hasattr(pa, "set_io_thread_count"):
        pa.set_io_thread_count(1)


def _loader_process_options(
    plan: DataLoaderWorkerPlan,
    *,
    persistent: bool,
) -> dict[str, Any]:
    options: dict[str, Any] = {
        "num_workers": plan.effective_workers,
        "persistent_workers": persistent and plan.effective_workers > 0,
    }
    if plan.effective_workers > 0:
        options.update(
            {
                "prefetch_factor": plan.prefetch_factor,
                "worker_init_fn": _initialize_dataloader_worker,
            }
        )
    return options


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_lazy_datasets(
    config: ExperimentConfig,
    *,
    worker_plan: DataLoaderWorkerPlan,
) -> tuple[
    LazyFinancialWindowDataset,
    LazyFinancialWindowDataset,
    LazyFinancialWindowDataset,
]:
    """Open only the compact lazy indexes used by training-time loaders."""

    bar_store = resolve_bar_store_path(config.data.bar_store_path)
    train_source = LazyFinancialWindowDataset(
        bar_store,
        split="train",
        window_size=config.data.input_length,
        h_start=config.data.h_start,
        symbol_cache_size=worker_plan.symbol_cache_size_per_worker,
    )
    validation_dataset = LazyFinancialWindowDataset(
        bar_store,
        split="validation",
        window_size=config.data.input_length,
        h_start=config.data.h_start,
        symbol_cache_size=worker_plan.symbol_cache_size_per_worker,
    )
    test_dataset = LazyFinancialWindowDataset(
        bar_store,
        split="test",
        window_size=config.data.input_length,
        h_start=config.data.h_start,
        symbol_cache_size=worker_plan.symbol_cache_size_per_worker,
    )
    return train_source, validation_dataset, test_dataset


def _explicit_runtime_batch_plan(config: ExperimentConfig) -> RuntimeBatchPlan:
    if (
        not isinstance(config.training.batch_size, int)
        or not isinstance(config.training.evaluation_batch_size, int)
        or not isinstance(config.training.gradient_accumulation_steps, int)
    ):
        raise ValueError(
            "Automatic batch controls require a resolved RuntimeBatchPlan"
        )
    effective_batch_size = (
        config.training.batch_size * config.training.gradient_accumulation_steps
    )
    return RuntimeBatchPlan(
        source="explicit_config",
        training_batch_size=config.training.batch_size,
        evaluation_batch_size=config.training.evaluation_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        effective_batch_size=effective_batch_size,
        target_effective_batch_size=effective_batch_size,
        device_name="unprobed",
        device_total_memory_bytes=0,
        optimizer_state_reserve_bytes=0,
        seconds_per_training_batch=None,
        training_probe=(),
        evaluation_probe=(),
    )


def build_dataloaders(
    config: ExperimentConfig,
    *,
    worker_plan: DataLoaderWorkerPlan | None = None,
    batch_plan: RuntimeBatchPlan | None = None,
    datasets: tuple[
        LazyFinancialWindowDataset,
        LazyFinancialWindowDataset,
        LazyFinancialWindowDataset,
    ]
    | None = None,
) -> tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any]]:
    if worker_plan is None:
        requested_workers, source = _requested_dataloader_workers(config)
        worker_plan = plan_dataloader_workers(requested_workers, source=source)
    if batch_plan is None:
        batch_plan = _explicit_runtime_batch_plan(config)
    train_source, validation_dataset, test_dataset = (
        datasets
        if datasets is not None
        else build_lazy_datasets(config, worker_plan=worker_plan)
    )
    training_batch_size = batch_plan.training_batch_size
    evaluation_batch_size = batch_plan.evaluation_batch_size
    train_sampler = BlockwisePermutationSampler(
        len(train_source),
        fraction=config.data.train_fraction,
        max_samples=config.data.max_samples,
        seed=config.training.seed,
        block_size=DATALOADER_SELECTION_BLOCK_SIZE,
    )
    validation_sampler = BlockwisePermutationSampler(
        len(validation_dataset),
        max_samples=config.training.evaluation_max_samples,
        seed=config.training.seed + 1,
        block_size=DATALOADER_SELECTION_BLOCK_SIZE,
    )
    test_sampler = BlockwisePermutationSampler(
        len(test_dataset),
        max_samples=config.training.evaluation_max_samples,
        seed=config.training.seed + 2,
        block_size=DATALOADER_SELECTION_BLOCK_SIZE,
    )
    if len(train_sampler) < training_batch_size:
        raise ValueError("Selected training samples do not fill one fixed-size batch")
    train_batch_sampler = ResumableFixedSizeBatchSampler(
        FixedSizeBatchSampler(
            train_sampler,
            batch_size=training_batch_size,
        )
    )
    collator = FinancialBatchCollator()
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_source,
        batch_sampler=train_batch_sampler,
        collate_fn=collator,
        pin_memory=pin_memory,
        **_loader_process_options(worker_plan, persistent=True),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=evaluation_batch_size,
        sampler=validation_sampler,
        shuffle=False,
        collate_fn=collator,
        pin_memory=pin_memory,
        **_loader_process_options(worker_plan, persistent=True),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=evaluation_batch_size,
        sampler=test_sampler,
        shuffle=False,
        collate_fn=collator,
        pin_memory=pin_memory,
        **_loader_process_options(worker_plan, persistent=True),
    )
    return train_loader, validation_loader, test_loader


def estimate_runtime_robust_scales(
    dataset: LazyFinancialWindowDataset,
    *,
    sample_count: int,
    seed: int,
    worker_plan: DataLoaderWorkerPlan | None = None,
) -> tuple[float, ...]:
    """Estimate train-only label scales without persisting any label rows."""

    if sample_count < 4:
        raise ValueError("At least four runtime calibration labels are required")
    if worker_plan is None:
        worker_plan = plan_dataloader_workers(0, source="serial_default")
    selected_sample_count = min(sample_count, len(dataset))
    if selected_sample_count < 4:
        raise ValueError("At least four runtime calibration labels are required")
    sampler = BlockwisePermutationSampler(
        len(dataset),
        fraction=1.0,
        max_samples=selected_sample_count,
        seed=seed,
        block_size=ROBUST_SCALE_SELECTION_BLOCK_SIZE,
    )
    # Sorting only this bounded calibration subset preserves its deterministic
    # membership while grouping nearby row groups for efficient network-volume reads.
    selected_indices = sorted(sampler)
    loader = DataLoader(
        _RuntimeLabelDataset(dataset),
        batch_size=min(ROBUST_SCALE_BATCH_SIZE, selected_sample_count),
        sampler=selected_indices,
        shuffle=False,
        pin_memory=False,
        **_loader_process_options(worker_plan, persistent=False),
    )
    batches = [batch.numpy().astype(np.float64, copy=False) for batch in loader]
    values = np.concatenate(batches, axis=0)
    if values.shape != (selected_sample_count, len(dataset.horizons)):
        raise RuntimeError("Runtime calibration DataLoader emitted an invalid label matrix")
    q25, q75 = np.quantile(values, [0.25, 0.75], axis=0)
    median = np.median(values, axis=0)
    interquartile_range = q75 - q25
    median_absolute_deviation = np.median(np.abs(values - median), axis=0) * 1.4826
    scales = np.maximum.reduce(
        (
            interquartile_range,
            median_absolute_deviation,
            np.full(len(dataset.horizons), 1e-4, dtype=np.float64),
        )
    )
    if not np.isfinite(scales).all() or (scales <= 0.0).any():
        raise ValueError("Runtime calibration produced invalid robust scales")
    return tuple(float(value) for value in scales)


def _robust_scale_identity(
    dataset: LazyFinancialWindowDataset,
    *,
    sample_count: int,
    seed: int,
) -> dict[str, Any]:
    manifest_path = dataset.root / "bar-store.json"
    return {
        "schema_version": ROBUST_SCALE_CACHE_SCHEMA_VERSION,
        "algorithm": ROBUST_SCALE_ALGORITHM,
        "bar_store_manifest_sha256": sha256_file(manifest_path),
        "split": dataset.split,
        "horizons": list(dataset.horizons),
        "sample_count": min(sample_count, len(dataset)),
        "seed": seed,
    }


def _default_robust_scale_cache_root(dataset: LazyFinancialWindowDataset) -> Path:
    if dataset.root.parent.name == "prepared":
        dataset_namespace = dataset.root.parent.parent
    else:
        dataset_namespace = dataset.root.parent
    return dataset_namespace / "training-cache" / "robust-scales"


def _load_cached_robust_scales(
    path: Path,
    *,
    identity: dict[str, Any],
    identity_sha256: str,
) -> tuple[float, ...]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise ValueError(f"Runtime robust-scale cache is unreadable: {path}") from error
    scales = payload.get("scales") if isinstance(payload, dict) else None
    if (
        not isinstance(payload, dict)
        or set(payload)
        != {
            "identity",
            "identity_sha256",
            "kind",
            "scales",
            "schema_version",
        }
        or payload.get("schema_version") != ROBUST_SCALE_CACHE_SCHEMA_VERSION
        or payload.get("kind") != "runtime-robust-scales"
        or payload.get("identity") != identity
        or payload.get("identity_sha256") != identity_sha256
        or not isinstance(scales, list)
        or len(scales) != len(identity["horizons"])
        or any(
            not isinstance(value, int | float)
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) <= 0.0
            for value in scales
        )
    ):
        raise ValueError(f"Runtime robust-scale cache contract is invalid: {path}")
    return tuple(float(value) for value in scales)


def resolve_runtime_robust_scales(
    dataset: LazyFinancialWindowDataset,
    *,
    sample_count: int,
    seed: int,
    worker_plan: DataLoaderWorkerPlan,
    cache_root: str | Path | None = None,
) -> RuntimeRobustScaleResult:
    """Reuse or calculate a tiny aggregate cache without materializing labels."""

    identity = _robust_scale_identity(
        dataset,
        sample_count=sample_count,
        seed=seed,
    )
    identity_sha256 = canonical_json_sha256(identity)
    root = (
        Path(cache_root).resolve(strict=False)
        if cache_root is not None
        else _default_robust_scale_cache_root(dataset)
    )
    cache_path = root / f"{identity_sha256}.json"
    if cache_path.is_file():
        scales = _load_cached_robust_scales(
            cache_path,
            identity=identity,
            identity_sha256=identity_sha256,
        )
        print(
            json.dumps(
                {
                    "runtime_robust_scale_calibration": "cache_hit",
                    "cache_path": str(cache_path),
                    "sample_count": identity["sample_count"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return RuntimeRobustScaleResult(
            scales=scales,
            cache_path=cache_path,
            cache_hit=True,
            identity_sha256=identity_sha256,
            sample_count=int(identity["sample_count"]),
        )

    print(
        json.dumps(
            {
                "runtime_robust_scale_calibration": "running",
                "effective_workers": worker_plan.effective_workers,
                "sample_count": identity["sample_count"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    scales = estimate_runtime_robust_scales(
        dataset,
        sample_count=sample_count,
        seed=seed,
        worker_plan=worker_plan,
    )
    atomic_write_json(
        cache_path,
        {
            "schema_version": ROBUST_SCALE_CACHE_SCHEMA_VERSION,
            "kind": "runtime-robust-scales",
            "identity": identity,
            "identity_sha256": identity_sha256,
            "scales": list(scales),
        },
    )
    print(
        json.dumps(
            {
                "runtime_robust_scale_calibration": "complete",
                "cache_path": str(cache_path),
                "effective_workers": worker_plan.effective_workers,
                "sample_count": identity["sample_count"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return RuntimeRobustScaleResult(
        scales=scales,
        cache_path=cache_path,
        cache_hit=False,
        identity_sha256=identity_sha256,
        sample_count=int(identity["sample_count"]),
    )


def _autocast_context(config: ExperimentConfig, device: torch.device) -> Any:
    if device.type != "cuda" or config.training.mixed_precision == "no":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def _uses_full_length_context(batch: dict[str, Any], prefix: str) -> bool:
    series = cast(Tensor, batch[f"{prefix}_series"])
    lengths = cast(Tensor, batch[f"{prefix}_lengths"])
    return bool(torch.all(lengths == series.shape[1]))


def _move_batch_to_device(
    batch: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Move only model inputs while collapsing fixed-length masks to a fast path."""

    non_blocking = device.type == "cuda"
    moved = dict(batch)
    for key in (
        "asset_series",
        "asset_timestamps",
        "benchmark_series",
        "benchmark_timestamps",
        "target_alpha",
    ):
        moved[key] = cast(Tensor, batch[key]).to(
            device,
            non_blocking=non_blocking,
        )
    for prefix in ("asset", "benchmark"):
        mask_key = f"{prefix}_attention_mask"
        moved[mask_key] = (
            None
            if _uses_full_length_context(batch, prefix)
            else cast(Tensor, batch[mask_key]).to(
                device,
                non_blocking=non_blocking,
            )
        )
    return moved


def _record_batch_stream(batch: dict[str, Any], stream: torch.cuda.Stream) -> None:
    for value in batch.values():
        if isinstance(value, Tensor) and value.device.type == "cuda":
            value.record_stream(stream)


def iter_device_batches(
    loader: DataLoader[Any],
    device: torch.device,
) -> Any:
    """Overlap pinned host-to-device copies with the preceding CUDA batch."""

    if device.type != "cuda":
        for batch in loader:
            yield _move_batch_to_device(batch, device)
        return

    iterator = iter(loader)
    transfer_stream = torch.cuda.Stream(device=device)

    def preload() -> dict[str, Any] | None:
        try:
            host_batch = next(iterator)
        except StopIteration:
            return None
        with torch.cuda.stream(transfer_stream):
            return _move_batch_to_device(host_batch, device)

    next_batch = preload()
    while next_batch is not None:
        compute_stream = torch.cuda.current_stream(device)
        compute_stream.wait_stream(transfer_stream)
        current_batch = next_batch
        next_batch = preload()
        yield current_batch
        _record_batch_stream(current_batch, compute_stream)


def forward_batch(
    bundle: ModelBundle,
    batch: dict[str, Any],
    config: ExperimentConfig,
    device: torch.device,
) -> Any:
    del config
    del device
    return bundle.model(
        batch["asset_series"],
        batch["benchmark_series"],
        asset_attention_mask=batch["asset_attention_mask"],
        benchmark_attention_mask=batch["benchmark_attention_mask"],
        asset_timestamps=batch["asset_timestamps"],
        benchmark_timestamps=batch["benchmark_timestamps"],
        target_alpha=batch["target_alpha"],
    )


def _batch_candidates(minimum: int, maximum: int) -> tuple[int, ...]:
    if minimum < 1 or maximum < minimum:
        raise ValueError("Automatic batch candidate bounds are invalid")
    candidates: list[int] = []
    value = minimum
    while value <= maximum:
        candidates.append(value)
        value *= 2
    if candidates[-1] != maximum:
        candidates.append(maximum)
    return tuple(candidates)


def _is_cuda_out_of_memory(error: BaseException) -> bool:
    return isinstance(error, torch.cuda.OutOfMemoryError) or (
        isinstance(error, RuntimeError)
        and "out of memory" in str(error).lower()
    )


def _measure_cuda_batch(
    *,
    bundle: ModelBundle,
    sample: dict[str, Any],
    batch_size: int,
    config: ExperimentConfig,
    device: torch.device,
    training: bool,
    optimizer_state_reserve_bytes: int,
    device_memory_limit_bytes: int,
) -> BatchProbeMeasurement:
    collator = FinancialBatchCollator()
    output: Any | None = None
    device_batch: dict[str, Any] | None = None
    try:
        host_batch = collator([sample] * batch_size)
        device_batch = _move_batch_to_device(host_batch, device)
        bundle.model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        def run_once() -> None:
            nonlocal output
            output = None
            context = nullcontext() if training else torch.inference_mode()
            with context, _autocast_context(config, device):
                output = forward_batch(bundle, device_batch, config, device)
                if training:
                    if output.loss is None:
                        raise RuntimeError("Batch probe did not produce a training loss")
                    output.loss.backward()
            if training:
                bundle.model.zero_grad(set_to_none=True)

        run_once()
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        for _ in range(config.training.auto_batch_probe_steps):
            run_once()
        torch.cuda.synchronize(device)
        seconds_per_batch = (
            time.perf_counter() - started
        ) / config.training.auto_batch_probe_steps
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        projected_peak = peak_allocated + (
            optimizer_state_reserve_bytes if training else 0
        )
        accepted = projected_peak <= device_memory_limit_bytes
        return BatchProbeMeasurement(
            batch_size=batch_size,
            seconds_per_batch=seconds_per_batch,
            samples_per_second=batch_size / max(seconds_per_batch, 1e-12),
            peak_allocated_bytes=peak_allocated,
            projected_peak_bytes=projected_peak,
            accepted=accepted,
            outcome="accepted" if accepted else "memory_guard",
        )
    except BaseException as error:
        if not _is_cuda_out_of_memory(error):
            raise
        return BatchProbeMeasurement(
            batch_size=batch_size,
            seconds_per_batch=None,
            samples_per_second=None,
            peak_allocated_bytes=None,
            projected_peak_bytes=None,
            accepted=False,
            outcome="cuda_out_of_memory",
        )
    finally:
        del output
        del device_batch
        bundle.model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()


def _select_batch_measurement(
    measurements: list[BatchProbeMeasurement],
) -> BatchProbeMeasurement:
    accepted = [measurement for measurement in measurements if measurement.accepted]
    if not accepted:
        raise RuntimeError(
            "No automatic batch candidate fit the configured CUDA memory guard"
        )
    best_throughput = max(
        cast(float, measurement.samples_per_second) for measurement in accepted
    )
    threshold = best_throughput * (1.0 - AUTO_BATCH_THROUGHPUT_TOLERANCE)
    near_optimal = [
        measurement
        for measurement in accepted
        if cast(float, measurement.samples_per_second) >= threshold
    ]
    return min(near_optimal, key=lambda measurement: measurement.batch_size)


def _hardware_batch_expansion_maximum(
    *,
    configured_maximum: int,
    device_total_memory_bytes: int,
    absolute_maximum: int,
) -> int:
    """Scale the probe ceiling with VRAM while retaining an absolute safety cap."""

    if min(configured_maximum, device_total_memory_bytes, absolute_maximum) < 1:
        raise ValueError("Automatic batch expansion inputs must be positive")
    memory_multiplier = max(
        1,
        (
            device_total_memory_bytes
            + AUTO_BATCH_EXPANSION_MEMORY_QUANTUM_BYTES
            - 1
        )
        // AUTO_BATCH_EXPANSION_MEMORY_QUANTUM_BYTES,
    )
    return min(
        absolute_maximum,
        configured_maximum * memory_multiplier,
    )


def _next_adaptive_batch_candidate(
    measurements: list[BatchProbeMeasurement],
    *,
    expansion_maximum: int,
) -> int | None:
    """Continue past the configured ceiling only while throughput still scales."""

    if not measurements or expansion_maximum < 1:
        return None
    latest = measurements[-1]
    if not latest.accepted or latest.batch_size >= expansion_maximum:
        return None
    accepted = [measurement for measurement in measurements if measurement.accepted]
    best_throughput = max(
        cast(float, measurement.samples_per_second) for measurement in accepted
    )
    threshold = best_throughput * (
        1.0 - AUTO_BATCH_EXPANSION_THROUGHPUT_TOLERANCE
    )
    if cast(float, latest.samples_per_second) < threshold:
        return None
    return min(latest.batch_size * 2, expansion_maximum)


def _automatic_gradient_accumulation_steps(
    *,
    target_effective_batch_size: int,
    training_batch_size: int,
) -> int:
    """Keep the configured effective batch as a floor on larger accelerators."""

    if training_batch_size < target_effective_batch_size:
        if target_effective_batch_size % training_batch_size != 0:
            raise ValueError(
                "The selected automatic batch must divide target_effective_batch_size"
            )
        return target_effective_batch_size // training_batch_size
    return 1


def _probe_cuda_candidates(
    *,
    bundle: ModelBundle,
    sample: dict[str, Any],
    candidates: tuple[int, ...],
    config: ExperimentConfig,
    device: torch.device,
    training: bool,
    optimizer_state_reserve_bytes: int,
    device_memory_limit_bytes: int,
    expansion_maximum: int | None = None,
) -> tuple[BatchProbeMeasurement, tuple[BatchProbeMeasurement, ...]]:
    measurements: list[BatchProbeMeasurement] = []
    pending_candidates = list(candidates)
    previous_mode = bundle.model.training
    cpu_rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state_all()
    bundle.model.train(training)
    try:
        candidate_index = 0
        while candidate_index < len(pending_candidates):
            batch_size = pending_candidates[candidate_index]
            measurement = _measure_cuda_batch(
                bundle=bundle,
                sample=sample,
                batch_size=batch_size,
                config=config,
                device=device,
                training=training,
                optimizer_state_reserve_bytes=optimizer_state_reserve_bytes,
                device_memory_limit_bytes=device_memory_limit_bytes,
            )
            measurements.append(measurement)
            if measurement.outcome in {"cuda_out_of_memory", "memory_guard"}:
                break
            if (
                expansion_maximum is not None
                and candidate_index == len(pending_candidates) - 1
            ):
                next_candidate = _next_adaptive_batch_candidate(
                    measurements,
                    expansion_maximum=expansion_maximum,
                )
                if next_candidate is not None:
                    pending_candidates.append(next_candidate)
            candidate_index += 1
    finally:
        bundle.model.train(previous_mode)
        torch.set_rng_state(cpu_rng_state)
        torch.cuda.set_rng_state_all(cuda_rng_state)
    return _select_batch_measurement(measurements), tuple(measurements)


def resolve_runtime_batch_plan(
    config: ExperimentConfig,
    *,
    bundle: ModelBundle,
    sample: dict[str, Any],
    device: torch.device,
) -> RuntimeBatchPlan:
    """Empirically resolve safe high-throughput batches on the current accelerator."""

    device_name = "cpu"
    device_total_memory_bytes = 0
    device_memory_limit_bytes = 0
    training_expansion_maximum = config.training.auto_batch_max_size
    evaluation_expansion_maximum = (
        config.training.auto_evaluation_batch_max_size
    )
    optimizer_state_reserve_bytes = sum(
        parameter.numel() * parameter.element_size() * 2
        for parameter in bundle.model.parameters()
        if parameter.requires_grad
    )
    training_probe: tuple[BatchProbeMeasurement, ...] = ()
    evaluation_probe: tuple[BatchProbeMeasurement, ...] = ()
    seconds_per_training_batch: float | None = None

    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        device_name = properties.name
        device_total_memory_bytes = int(properties.total_memory)
        free_memory_bytes, _total_memory_bytes = torch.cuda.mem_get_info(device)
        allocated_memory_bytes = int(torch.cuda.memory_allocated(device))
        device_memory_limit_bytes = min(
            int(
                device_total_memory_bytes
                * config.training.auto_batch_memory_fraction
            ),
            allocated_memory_bytes
            + int(free_memory_bytes * CUDA_FREE_MEMORY_FRACTION),
        )
        training_expansion_maximum = _hardware_batch_expansion_maximum(
            configured_maximum=config.training.auto_batch_max_size,
            device_total_memory_bytes=device_total_memory_bytes,
            absolute_maximum=AUTO_BATCH_EXPANSION_MAX_SIZE,
        )
        evaluation_expansion_maximum = _hardware_batch_expansion_maximum(
            configured_maximum=config.training.auto_evaluation_batch_max_size,
            device_total_memory_bytes=device_total_memory_bytes,
            absolute_maximum=AUTO_EVALUATION_BATCH_EXPANSION_MAX_SIZE,
        )
        print(
            json.dumps(
                {
                    "cuda_batch_search": {
                        "adaptive_evaluation_maximum": (
                            evaluation_expansion_maximum
                        ),
                        "adaptive_training_maximum": training_expansion_maximum,
                        "configured_evaluation_maximum": (
                            config.training.auto_evaluation_batch_max_size
                        ),
                        "configured_training_maximum": (
                            config.training.auto_batch_max_size
                        ),
                        "device_allocated_memory_bytes": allocated_memory_bytes,
                        "device_free_memory_bytes": int(free_memory_bytes),
                        "device_memory_limit_bytes": device_memory_limit_bytes,
                        "device_total_memory_bytes": device_total_memory_bytes,
                    }
                },
                sort_keys=True,
            ),
            flush=True,
        )

    if isinstance(config.training.batch_size, int):
        training_batch_size = config.training.batch_size
        if device.type == "cuda":
            selected, training_probe = _probe_cuda_candidates(
                bundle=bundle,
                sample=sample,
                candidates=(training_batch_size,),
                config=config,
                device=device,
                training=True,
                optimizer_state_reserve_bytes=optimizer_state_reserve_bytes,
                device_memory_limit_bytes=device_memory_limit_bytes,
            )
            seconds_per_training_batch = selected.seconds_per_batch
    elif device.type == "cuda":
        # The configured range remains the mandatory baseline. Larger candidates
        # are measured only when throughput at its boundary still justifies them.
        selected, training_probe = _probe_cuda_candidates(
            bundle=bundle,
            sample=sample,
            candidates=_batch_candidates(
                config.training.auto_batch_min_size,
                config.training.auto_batch_max_size,
            ),
            config=config,
            device=device,
            training=True,
            optimizer_state_reserve_bytes=optimizer_state_reserve_bytes,
            device_memory_limit_bytes=device_memory_limit_bytes,
            expansion_maximum=max(
                config.training.auto_batch_max_size,
                training_expansion_maximum,
            ),
        )
        training_batch_size = selected.batch_size
        seconds_per_training_batch = selected.seconds_per_batch
    else:
        training_batch_size = config.training.auto_batch_min_size

    if config.training.gradient_accumulation_steps == "auto":
        accumulation_steps = _automatic_gradient_accumulation_steps(
            target_effective_batch_size=(
                config.training.target_effective_batch_size
            ),
            training_batch_size=training_batch_size,
        )
    else:
        accumulation_steps = config.training.gradient_accumulation_steps
    effective_batch_size = training_batch_size * accumulation_steps

    if isinstance(config.training.evaluation_batch_size, int):
        evaluation_batch_size = config.training.evaluation_batch_size
    elif device.type == "cuda":
        evaluation_initial_maximum = max(
            training_batch_size,
            config.training.auto_evaluation_batch_max_size,
        )
        selected, evaluation_probe = _probe_cuda_candidates(
            bundle=bundle,
            sample=sample,
            candidates=_batch_candidates(
                training_batch_size,
                evaluation_initial_maximum,
            ),
            config=config,
            device=device,
            training=False,
            optimizer_state_reserve_bytes=0,
            device_memory_limit_bytes=device_memory_limit_bytes,
            expansion_maximum=max(
                evaluation_initial_maximum,
                evaluation_expansion_maximum,
            ),
        )
        evaluation_batch_size = selected.batch_size
    else:
        evaluation_batch_size = max(
            training_batch_size,
            config.training.auto_batch_min_size,
        )

    return RuntimeBatchPlan(
        source=("cuda_empirical" if device.type == "cuda" else "cpu_fallback"),
        training_batch_size=training_batch_size,
        evaluation_batch_size=evaluation_batch_size,
        gradient_accumulation_steps=accumulation_steps,
        effective_batch_size=effective_batch_size,
        target_effective_batch_size=config.training.target_effective_batch_size,
        device_name=device_name,
        device_total_memory_bytes=device_total_memory_bytes,
        optimizer_state_reserve_bytes=optimizer_state_reserve_bytes,
        seconds_per_training_batch=seconds_per_training_batch,
        training_probe=training_probe,
        evaluation_probe=evaluation_probe,
    )


def _subgroup_metrics(
    *,
    targets: NDArray[np.float32],
    quantile_predictions: NDArray[np.float32],
    horizons: list[int],
    quantiles: list[float],
    robust_scales: list[float],
    values: list[str],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    value_array = np.asarray(values)
    for name in sorted(set(values)):
        indices = np.flatnonzero(value_array == name)
        output[name] = {
            "samples": int(indices.size),
            **multi_horizon_alpha_metrics(
                targets=targets[indices],
                quantile_predictions=quantile_predictions[indices],
                horizons=horizons,
                quantiles=quantiles,
                robust_scales=robust_scales,
            ),
        }
    return output


@torch.inference_mode()
def evaluate_loader(
    bundle: ModelBundle,
    loader: DataLoader[Any],
    config: ExperimentConfig,
    device: torch.device,
) -> dict[str, Any]:
    was_training = bundle.model.training
    bundle.model.eval()
    targets: list[Tensor] = []
    quantile_predictions: list[Tensor] = []
    losses: list[Tensor] = []
    cutoff_dates: list[str] = []
    years: list[str] = []
    symbols: list[str] = []
    asset_types: list[str] = []
    markets: list[str] = []
    providers: list[str] = []

    for batch in iter_device_batches(loader, device):
        with _autocast_context(config, device):
            output = forward_batch(bundle, batch, config, device)
        if output.loss is not None:
            losses.append(output.loss.detach().float())
        targets.append(batch["target_alpha"].detach().float())
        quantile_predictions.append(output.alpha_quantiles.detach().float())
        batch_cutoffs = [str(value) for value in batch["cutoff_at"]]
        cutoff_dates.extend(batch_cutoffs)
        years.extend(value[:4] for value in batch_cutoffs)
        symbols.extend(str(value) for value in batch["symbols"])
        asset_types.extend(str(value) for value in batch["asset_types"])
        markets.extend(str(value) for value in batch["markets"])
        providers.extend(str(value) for value in batch["providers"])

    target_array = torch.cat(targets).cpu().numpy().astype(np.float32, copy=False)
    quantile_array = (
        torch.cat(quantile_predictions).cpu().numpy().astype(np.float32, copy=False)
    )
    mean_loss = float(torch.stack(losses).mean().cpu()) if losses else None
    horizons = list(config.data.alpha_horizons)
    quantiles = list(config.model.alpha_quantiles)
    robust_scales = [
        float(value)
        for value in bundle.model.alpha_head.robust_scales.detach().float().cpu().tolist()
    ]
    alpha_metrics = multi_horizon_alpha_metrics(
        targets=target_array,
        quantile_predictions=quantile_array,
        horizons=horizons,
        quantiles=quantiles,
        robust_scales=robust_scales,
    )
    median_index = quantiles.index(0.5)
    cross_sectional = {
        f"{horizon}d": cross_sectional_metrics(
            targets=target_array[:, horizon_index],
            signals=quantile_array[:, horizon_index, median_index],
            dates=cutoff_dates,
            symbols=symbols,
            annualization_horizon=horizon,
        )
        for horizon_index, horizon in enumerate(horizons)
    }
    signal_codes = postprocess_alpha_signal(
        quantile_array,
        threshold=config.model.postprocess_alpha_threshold,
    )
    signal_distribution = {
        f"{horizon}d": {
            name: int((signal_codes[:, horizon_index] == code).sum())
            for code, name in enumerate(POSTPROCESS_SIGNAL_NAMES)
        }
        for horizon_index, horizon in enumerate(horizons)
    }
    result = {
        "loss": mean_loss,
        "samples": int(target_array.shape[0]),
        **alpha_metrics,
        "cross_sectional_by_horizon": cross_sectional,
        "cross_sectional_5d": cross_sectional["5d"],
        "postprocess_signal_distribution": signal_distribution,
        "subgroups": {
            "asset_type": _subgroup_metrics(
                targets=target_array,
                quantile_predictions=quantile_array,
                horizons=horizons,
                quantiles=quantiles,
                robust_scales=robust_scales,
                values=asset_types,
            ),
            "market": _subgroup_metrics(
                targets=target_array,
                quantile_predictions=quantile_array,
                horizons=horizons,
                quantiles=quantiles,
                robust_scales=robust_scales,
                values=markets,
            ),
            "provider": _subgroup_metrics(
                targets=target_array,
                quantile_predictions=quantile_array,
                horizons=horizons,
                quantiles=quantiles,
                robust_scales=robust_scales,
                values=providers,
            ),
            "year": _subgroup_metrics(
                targets=target_array,
                quantile_predictions=quantile_array,
                horizons=horizons,
                quantiles=quantiles,
                robust_scales=robust_scales,
                values=years,
            ),
        },
    }
    bundle.model.train(was_training)
    return result


def _flatten_metrics(payload: dict[str, Any], prefix: str = "") -> dict[str, float]:
    flattened: dict[str, float] = {}
    for key, value in payload.items():
        path = f"{prefix}/{key}" if prefix else key
        if isinstance(value, dict):
            flattened.update(_flatten_metrics(value, path))
        elif (
            isinstance(value, int | float)
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        ):
            flattened[path] = float(value)
    return flattened


def _unflatten_metrics(payload: dict[str, float]) -> dict[str, Any]:
    nested: dict[str, Any] = {}
    for path, value in payload.items():
        keys = path.split("/")
        cursor = nested
        for key in keys[:-1]:
            child = cursor.setdefault(key, {})
            if not isinstance(child, dict):
                raise ValueError("Flattened validation metrics contain a path collision")
            cursor = child
        cursor[keys[-1]] = value
    return nested


def _validation_monitor_value(metrics: dict[str, Any], monitor: str) -> float:
    value: Any = metrics
    for key in monitor.split("/"):
        if not isinstance(value, dict) or key not in value:
            raise KeyError(f"Validation metrics do not contain checkpoint monitor: {monitor}")
        value = value[key]
    if (
        not isinstance(value, int | float)
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"Checkpoint monitor is not a finite scalar: {monitor}")
    return float(value)


def _learning_rate_multiplier(step: int, warmup_steps: int, total_steps: int) -> float:
    if warmup_steps and step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    remaining = max(total_steps - warmup_steps, 1)
    progress = min(max(step - warmup_steps, 0) / remaining, 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _scheduler(optimizer: AdamW, warmup_steps: int, total_steps: int) -> LambdaLR:
    return LambdaLR(
        optimizer,
        lambda step: _learning_rate_multiplier(step, warmup_steps, total_steps),
    )


def _realign_scheduler(
    scheduler: LambdaLR,
    *,
    runtime_global_step: int,
    warmup_steps: int,
    total_steps: int,
) -> None:
    """Move a loaded scheduler into this Pod's runtime step coordinate system."""

    if runtime_global_step < 0 or total_steps < 1:
        raise ValueError("Scheduler resume coordinates are invalid")
    multiplier = _learning_rate_multiplier(
        runtime_global_step,
        warmup_steps,
        total_steps,
    )
    learning_rates = [base_lr * multiplier for base_lr in scheduler.base_lrs]
    if len(learning_rates) != len(scheduler.optimizer.param_groups):
        raise ValueError("Scheduler parameter-group count changed during resume")
    for parameter_group, learning_rate in zip(
        scheduler.optimizer.param_groups,
        learning_rates,
        strict=True,
    ):
        parameter_group["lr"] = learning_rate
    scheduler.last_epoch = runtime_global_step
    scheduler._step_count = runtime_global_step + 1
    scheduler._last_lr = learning_rates


def _optimizer_parameter_groups(
    bundle: ModelBundle,
    config: ExperimentConfig,
) -> tuple[list[dict[str, Any]], list[Tensor]]:
    task_parameters: list[Tensor] = []
    lora_parameters: list[Tensor] = []
    trainable: list[Tensor] = []
    forbidden: list[str] = []
    for name, parameter in bundle.model.named_parameters():
        if not parameter.requires_grad:
            continue
        trainable.append(parameter)
        is_lora = ".lora_a." in name or ".lora_b." in name
        if name.startswith("backbone.tokenizer.") or (
            name.startswith("backbone.model.") and not is_lora
        ):
            forbidden.append(name)
        elif is_lora:
            lora_parameters.append(parameter)
        else:
            task_parameters.append(parameter)
    if forbidden:
        raise RuntimeError(
            "Frozen Kronos/tokenizer parameters entered the optimizer: "
            + ", ".join(sorted(forbidden))
        )
    if config.model.lora.enabled and not lora_parameters:
        raise RuntimeError("Kronos LoRA is enabled but no LoRA parameters are trainable")
    if not task_parameters:
        raise RuntimeError("Quant resampler, conditioner, and alpha head have no parameters")
    groups = [
        {
            "params": task_parameters,
            "lr": config.training.learning_rate,
            "group_name": "quant_modules",
        }
    ]
    if lora_parameters:
        groups.append(
            {
                "params": lora_parameters,
                "lr": config.training.lora_learning_rate,
                "group_name": "kronos_lora",
            }
        )
    return groups, trainable


def _checkpoint_runtime_execution_plan(
    state: dict[str, Any],
) -> RuntimeExecutionPlan | None:
    progress = state.get("training_progress")
    if not isinstance(progress, dict):
        raise ValueError("Resume checkpoint has no runtime training progress")
    payload = progress.get("runtime_execution_plan")
    if payload is None:
        if progress.get("schema_version") == TRAINING_PROGRESS_SCHEMA_VERSION:
            raise ValueError("Resume checkpoint has no runtime execution plan")
        return None
    return RuntimeExecutionPlan.from_dict(payload)


def _canonical_training_schedule(
    progress: Any,
    *,
    epochs: int,
    evaluations_per_epoch: int,
    initial_optimizer_steps_per_epoch: int | None = None,
) -> CanonicalTrainingSchedule:
    if progress is None:
        if initial_optimizer_steps_per_epoch is None:
            raise ValueError("Initial optimizer-step count is required")
        return CanonicalTrainingSchedule(
            epochs=epochs,
            evaluations_per_epoch=evaluations_per_epoch,
            optimizer_steps_per_epoch=initial_optimizer_steps_per_epoch,
            configured_optimizer_steps=initial_optimizer_steps_per_epoch * epochs,
        )
    if not isinstance(progress, dict):
        raise ValueError("Checkpoint training progress must be a mapping")
    if progress.get("schema_version") == TRAINING_PROGRESS_SCHEMA_VERSION:
        schedule = CanonicalTrainingSchedule.from_dict(
            progress.get("canonical_schedule")
        )
        if schedule.epochs != epochs or schedule.evaluations_per_epoch != (
            evaluations_per_epoch
        ):
            raise ValueError("Checkpoint canonical schedule differs from the selected config")
        return schedule

    configured_steps = progress.get("configured_optimizer_steps")
    if (
        not isinstance(configured_steps, int)
        or isinstance(configured_steps, bool)
        or configured_steps < 1
        or configured_steps % epochs != 0
    ):
        raise ValueError("Legacy checkpoint optimizer-step budget is invalid")
    return CanonicalTrainingSchedule(
        epochs=epochs,
        evaluations_per_epoch=evaluations_per_epoch,
        optimizer_steps_per_epoch=configured_steps // epochs,
        configured_optimizer_steps=configured_steps,
    )


def aligned_epoch_event_steps(
    epoch_index: int,
    *,
    runtime_optimizer_steps_per_epoch: int,
    canonical_optimizer_steps_per_epoch: int,
    points_per_epoch: int,
) -> tuple[tuple[int, int], ...]:
    """Pair runtime event steps with immutable canonical checkpoint steps."""

    points = min(
        points_per_epoch,
        runtime_optimizer_steps_per_epoch,
        canonical_optimizer_steps_per_epoch,
    )
    runtime_steps = epoch_evaluation_steps(
        epoch_index,
        runtime_optimizer_steps_per_epoch,
        points,
    )
    canonical_steps = epoch_evaluation_steps(
        epoch_index,
        canonical_optimizer_steps_per_epoch,
        points,
    )
    return tuple(zip(runtime_steps, canonical_steps, strict=True))


def resume_coordinates(
    state: dict[str, Any],
    *,
    schedule: CanonicalTrainingSchedule,
    runtime_optimizer_steps_per_epoch: int,
    runtime_batch_count: int,
    gradient_accumulation_steps: int,
) -> ResumeCoordinates:
    """Align a canonical validation checkpoint to the current Pod's batch plan."""

    canonical_global_step = state.get("global_step")
    if (
        not isinstance(canonical_global_step, int)
        or isinstance(canonical_global_step, bool)
        or canonical_global_step < 0
        or canonical_global_step > schedule.configured_optimizer_steps
    ):
        raise ValueError("Resume checkpoint canonical step is invalid")
    if runtime_optimizer_steps_per_epoch < schedule.evaluations_per_epoch:
        raise ValueError("Current hardware plan cannot run every required validation")
    if runtime_batch_count < 1 or gradient_accumulation_steps < 1:
        raise ValueError("Current runtime DataLoader coordinates are invalid")

    starting_epoch, canonical_step_within_epoch = divmod(
        canonical_global_step,
        schedule.optimizer_steps_per_epoch,
    )
    if canonical_step_within_epoch == 0:
        runtime_step_within_epoch = 0
    else:
        if starting_epoch >= schedule.epochs:
            raise ValueError("Resume checkpoint exceeds the canonical epoch budget")
        canonical_boundaries = epoch_evaluation_steps(
            0,
            schedule.optimizer_steps_per_epoch,
            schedule.evaluations_per_epoch,
        )
        try:
            boundary_index = canonical_boundaries.index(canonical_step_within_epoch)
        except ValueError as error:
            raise ValueError(
                "Resume checkpoint is not on a canonical validation boundary"
            ) from error
        runtime_step_within_epoch = epoch_evaluation_steps(
            0,
            runtime_optimizer_steps_per_epoch,
            schedule.evaluations_per_epoch,
        )[boundary_index]

    runtime_global_step = (
        starting_epoch * runtime_optimizer_steps_per_epoch
        + runtime_step_within_epoch
    )
    resume_batch_index = min(
        runtime_batch_count,
        runtime_step_within_epoch * gradient_accumulation_steps,
    )
    return ResumeCoordinates(
        starting_epoch=starting_epoch,
        resume_batch_index=resume_batch_index,
        canonical_global_step=canonical_global_step,
        runtime_global_step=runtime_global_step,
    )


def _gradient_divisor(batch_index: int, batch_count: int, accumulation_steps: int) -> int:
    group_start = (batch_index // accumulation_steps) * accumulation_steps
    return min(accumulation_steps, batch_count - group_start)


def epoch_evaluation_steps(
    epoch_index: int,
    optimizer_steps_per_epoch: int,
    evaluations_per_epoch: int,
) -> tuple[int, ...]:
    """Return exact, evenly spaced global steps for one epoch's validations."""

    if epoch_index < 0:
        raise ValueError("epoch_index must be non-negative")
    if optimizer_steps_per_epoch < 1 or evaluations_per_epoch < 1:
        raise ValueError("Evaluation scheduling requires positive step and evaluation counts")
    if optimizer_steps_per_epoch < evaluations_per_epoch:
        raise ValueError(
            "An epoch must contain at least one optimizer step per requested validation"
        )
    epoch_start = epoch_index * optimizer_steps_per_epoch
    return tuple(
        epoch_start + math.ceil(index * optimizer_steps_per_epoch / evaluations_per_epoch)
        for index in range(1, evaluations_per_epoch + 1)
    )


def epoch_loss_logging_steps(
    epoch_index: int,
    optimizer_steps_per_epoch: int,
    points_per_epoch: int,
) -> tuple[int, ...]:
    """Resolve a bounded number of evenly spaced training-loss observations."""

    if points_per_epoch < 1:
        raise ValueError("Loss logging points per epoch must be positive")
    return epoch_evaluation_steps(
        epoch_index,
        optimizer_steps_per_epoch,
        min(points_per_epoch, optimizer_steps_per_epoch),
    )


@dataclass
class EarlyStoppingState:
    """Track consecutive validation-loss regressions across resumable checkpoints."""

    best_value: float | None = None
    last_value: float | None = None
    stale_evaluations: int = 0
    evaluation_count: int = 0
    triggered: bool = False

    def observe(
        self,
        value: float,
        *,
        mode: str,
        min_delta: float,
        patience: int,
        epoch_number: int,
        start_epoch: int,
        enabled: bool,
    ) -> bool:
        if not math.isfinite(value):
            raise ValueError("Early-stopping monitor must be finite")
        if mode not in {"min", "max"}:
            raise ValueError("Early-stopping mode must be min or max")
        if min_delta < 0.0 or patience < 1 or min(epoch_number, start_epoch) < 1:
            raise ValueError("Early-stopping policy is invalid")
        improved = self.best_value is None or (
            value < self.best_value - min_delta
            if mode == "min"
            else value > self.best_value + min_delta
        )
        self.last_value = value
        self.evaluation_count += 1
        if improved:
            self.best_value = value
            self.stale_evaluations = 0
        else:
            self.stale_evaluations += 1
        self.triggered = bool(
            enabled and epoch_number >= start_epoch and self.stale_evaluations >= patience
        )
        return self.triggered

    def as_dict(self) -> dict[str, Any]:
        return {
            "best_value": self.best_value,
            "last_value": self.last_value,
            "stale_evaluations": self.stale_evaluations,
            "evaluation_count": self.evaluation_count,
            "triggered": self.triggered,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> EarlyStoppingState:
        if not isinstance(payload, dict):
            raise ValueError("Checkpoint early-stopping state must be a mapping")
        expected = {
            "best_value",
            "last_value",
            "stale_evaluations",
            "evaluation_count",
            "triggered",
        }
        if set(payload) != expected:
            raise ValueError("Checkpoint early-stopping state is incomplete")
        best_value = payload["best_value"]
        last_value = payload["last_value"]
        if any(
            value is not None
            and (
                not isinstance(value, int | float)
                or isinstance(value, bool)
                or not math.isfinite(float(value))
            )
            for value in (best_value, last_value)
        ):
            raise ValueError("Checkpoint early-stopping values are invalid")
        stale_evaluations = payload["stale_evaluations"]
        evaluation_count = payload["evaluation_count"]
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in (stale_evaluations, evaluation_count)
        ) or not isinstance(payload["triggered"], bool):
            raise ValueError("Checkpoint early-stopping counters are invalid")
        if stale_evaluations > evaluation_count:
            raise ValueError("Checkpoint early-stopping stale count exceeds evaluations")
        return cls(
            best_value=float(best_value) if best_value is not None else None,
            last_value=float(last_value) if last_value is not None else None,
            stale_evaluations=stale_evaluations,
            evaluation_count=evaluation_count,
            triggered=payload["triggered"],
        )


def _training_progress(
    *,
    processed_train_samples: int,
    completed_epochs: int,
    early_stopping: EarlyStoppingState,
    canonical_schedule: CanonicalTrainingSchedule,
    runtime_execution_plan: RuntimeExecutionPlan,
) -> dict[str, Any]:
    return {
        "schema_version": TRAINING_PROGRESS_SCHEMA_VERSION,
        "processed_train_samples": processed_train_samples,
        "completed_epochs": completed_epochs,
        "early_stopping": early_stopping.as_dict(),
        "canonical_schedule": canonical_schedule.as_dict(),
        "runtime_execution_plan": runtime_execution_plan.as_dict(),
    }


def _restore_training_progress(
    payload: Any,
    *,
    canonical_schedule: CanonicalTrainingSchedule,
) -> tuple[int, int, EarlyStoppingState]:
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint training progress is incomplete")
    if payload.get("schema_version") == TRAINING_PROGRESS_SCHEMA_VERSION:
        expected = {
            "schema_version",
            "processed_train_samples",
            "completed_epochs",
            "early_stopping",
            "canonical_schedule",
            "runtime_execution_plan",
        }
        if set(payload) != expected:
            raise ValueError("Checkpoint training progress is incomplete")
        stored_schedule = CanonicalTrainingSchedule.from_dict(
            payload["canonical_schedule"]
        )
        if stored_schedule != canonical_schedule:
            raise ValueError("Checkpoint canonical schedule changed during resume")
        RuntimeExecutionPlan.from_dict(payload["runtime_execution_plan"])
    else:
        expected = {
            "processed_train_samples",
            "completed_epochs",
            "configured_optimizer_steps",
            "early_stopping",
            "runtime_batch_plan",
        }
        if set(payload) != expected:
            raise ValueError("Legacy checkpoint training progress is incomplete")
        stored_optimizer_steps = payload["configured_optimizer_steps"]
        if (
            not isinstance(stored_optimizer_steps, int)
            or isinstance(stored_optimizer_steps, bool)
            or stored_optimizer_steps != canonical_schedule.configured_optimizer_steps
        ):
            raise ValueError("Checkpoint optimizer-step budget differs from its canonical plan")
        RuntimeBatchPlan.from_dict(payload["runtime_batch_plan"])
    processed_train_samples = payload["processed_train_samples"]
    completed_epochs = payload["completed_epochs"]
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in (
            processed_train_samples,
            completed_epochs,
        )
    ):
        raise ValueError("Checkpoint training progress counters are invalid")
    if completed_epochs > canonical_schedule.epochs:
        raise ValueError("Checkpoint completed-epoch count exceeds the canonical plan")
    return (
        processed_train_samples,
        completed_epochs,
        EarlyStoppingState.from_dict(payload["early_stopping"]),
    )


@dataclass(frozen=True)
class TrainingResult:
    run_directory: Path
    final_checkpoint: Path
    global_step: int
    stage: str
    model_architecture_sha256: str
    dataset_profile: str
    selected_datasets: tuple[str, ...]
    selected_train_samples_per_epoch: int
    processed_train_samples: int
    configured_optimizer_steps: int
    completed_epochs: int
    validation_evaluations: int
    stop_reason: str
    completion_result: Path
    robust_scales: tuple[float, ...]
    validation_metrics: dict[str, Any]


def train(config: ExperimentConfig) -> TrainingResult:
    with training_run_lease(config):
        validate_tracking_run_contract(config)
        return _train_with_lease(config)


def _train_with_lease(config: ExperimentConfig) -> TrainingResult:
    preflight = run_preflight(config, require_data=True)
    preflight.require_success()
    set_global_seed(config.training.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resume_preview: dict[str, Any] | None = None
    stored_execution_plan: RuntimeExecutionPlan | None = None
    if config.training.resume_checkpoint is not None:
        resume_preview = validate_checkpoint_trainer_state(
            config.training.resume_checkpoint
        )
        stored_execution_plan = _checkpoint_runtime_execution_plan(resume_preview)

    hardware = runtime_hardware_snapshot(device)
    requested_workers, worker_source = _requested_dataloader_workers(config)
    current_worker_plan = plan_dataloader_workers(
        requested_workers,
        source=worker_source,
    )
    replan_reason = (
        "fresh_training"
        if resume_preview is None
        else runtime_resource_plan_reuse_reason(
            stored_execution_plan,
            current_hardware=hardware,
            current_worker_plan=current_worker_plan,
        )
    )
    reuse_checkpoint_plan = replan_reason == "checkpoint_hardware_match"
    worker_plan = (
        reuse_dataloader_worker_plan(
            cast(RuntimeExecutionPlan, stored_execution_plan).worker_plan,
            current_worker_plan,
        )
        if reuse_checkpoint_plan
        else current_worker_plan
    )
    datasets = build_lazy_datasets(config, worker_plan=worker_plan)
    train_dataset = datasets[0]
    robust_scale_result = resolve_runtime_robust_scales(
        train_dataset,
        sample_count=config.data.label_scale_calibration_samples,
        seed=config.training.seed + 17,
        worker_plan=worker_plan,
    )
    robust_scales = robust_scale_result.scales
    bundle = build_model_bundle(config, device, robust_scales=robust_scales)
    probe_sample = train_dataset[0]
    batch_plan = (
        RuntimeBatchPlan.from_dict(
            cast(RuntimeExecutionPlan, stored_execution_plan).batch_plan.as_dict(),
            source="checkpoint_hardware_match",
        )
        if reuse_checkpoint_plan
        else resolve_runtime_batch_plan(
            config,
            bundle=bundle,
            sample=probe_sample,
            device=device,
        )
    )
    collator = FinancialBatchCollator()
    largest_host_batch_bytes = max(
        _batch_tensor_bytes(
            collator([probe_sample] * batch_plan.training_batch_size)
        ),
        _batch_tensor_bytes(
            collator([probe_sample] * batch_plan.evaluation_batch_size)
        ),
    )
    if not reuse_checkpoint_plan:
        worker_plan = plan_runtime_prefetch(
            worker_plan,
            config=config,
            batch_plan=batch_plan,
            largest_host_batch_bytes=largest_host_batch_bytes,
        )
    train_loader, validation_loader, _test_loader = build_dataloaders(
        config,
        worker_plan=worker_plan,
        batch_plan=batch_plan,
        datasets=datasets,
    )
    train_batch_sampler = cast(
        ResumableFixedSizeBatchSampler,
        train_loader.batch_sampler,
    )
    selected_train_samples = len(train_batch_sampler.sampler)
    parameter_groups, trainable = _optimizer_parameter_groups(bundle, config)
    optimizer = AdamW(
        parameter_groups,
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        fused=device.type == "cuda",
    )
    optimizer_steps_per_epoch = math.ceil(
        len(train_loader) / batch_plan.gradient_accumulation_steps
    )
    runtime_configured_steps = optimizer_steps_per_epoch * config.training.epochs
    if runtime_configured_steps < 1:
        raise ValueError("Training budget must contain at least one optimizer step")
    canonical_schedule = _canonical_training_schedule(
        None if resume_preview is None else resume_preview.get("training_progress"),
        epochs=config.training.epochs,
        evaluations_per_epoch=config.training.evaluations_per_epoch,
        initial_optimizer_steps_per_epoch=(
            optimizer_steps_per_epoch if resume_preview is None else None
        ),
    )
    coordinate_state = resume_preview or {"global_step": 0}
    coordinates = resume_coordinates(
        coordinate_state,
        schedule=canonical_schedule,
        runtime_optimizer_steps_per_epoch=optimizer_steps_per_epoch,
        runtime_batch_count=len(train_loader),
        gradient_accumulation_steps=batch_plan.gradient_accumulation_steps,
    )
    execution_plan = RuntimeExecutionPlan(
        source=("checkpoint_reuse" if reuse_checkpoint_plan else "runtime_probe"),
        replan_reason=replan_reason,
        hardware=hardware,
        batch_plan=batch_plan,
        worker_plan=worker_plan,
        optimizer_steps_per_epoch=optimizer_steps_per_epoch,
        configured_optimizer_steps=runtime_configured_steps,
        resume_canonical_global_step=coordinates.canonical_global_step,
        resume_runtime_global_step=coordinates.runtime_global_step,
    )
    evaluation_alignment = {
        runtime_step: canonical_step
        for epoch_index in range(config.training.epochs)
        for runtime_step, canonical_step in aligned_epoch_event_steps(
            epoch_index,
            runtime_optimizer_steps_per_epoch=optimizer_steps_per_epoch,
            canonical_optimizer_steps_per_epoch=(
                canonical_schedule.optimizer_steps_per_epoch
            ),
            points_per_epoch=config.training.evaluations_per_epoch,
        )
    }
    maximum_loss_points = min(
        config.training.loss_log_points_per_epoch,
        optimizer_steps_per_epoch,
        canonical_schedule.optimizer_steps_per_epoch,
    )
    actual_loss_points = (
        maximum_loss_points // config.training.evaluations_per_epoch
    ) * config.training.evaluations_per_epoch
    if actual_loss_points < config.training.evaluations_per_epoch:
        raise ValueError("Loss logging cannot cover every validation boundary")
    loss_logging_alignment = {
        runtime_step: canonical_step
        for epoch_index in range(config.training.epochs)
        for runtime_step, canonical_step in aligned_epoch_event_steps(
            epoch_index,
            runtime_optimizer_steps_per_epoch=optimizer_steps_per_epoch,
            canonical_optimizer_steps_per_epoch=(
                canonical_schedule.optimizer_steps_per_epoch
            ),
            points_per_epoch=actual_loss_points,
        )
    }
    loss_logging_plan = {
        "requested_points_per_epoch": config.training.loss_log_points_per_epoch,
        "actual_points_per_epoch": actual_loss_points,
        "runtime_optimizer_steps_per_epoch": optimizer_steps_per_epoch,
        "canonical_optimizer_steps_per_epoch": (
            canonical_schedule.optimizer_steps_per_epoch
        ),
        "runtime_nominal_interval_steps": math.ceil(
            optimizer_steps_per_epoch / actual_loss_points
        ),
    }
    print(
        json.dumps(
            {
                "canonical_training_schedule": canonical_schedule.as_dict(),
                "runtime_execution_plan": execution_plan.as_dict(),
                "loss_logging_plan": loss_logging_plan,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    runtime_warmup_steps = int(
        runtime_configured_steps * config.training.warmup_ratio
    )
    scheduler = _scheduler(
        optimizer,
        runtime_warmup_steps,
        runtime_configured_steps,
    )
    tracking: TrackingRun = start_tracking(config)
    runtime_execution_plan_path = atomic_write_json(
        tracking.directory / "runtime-execution-plan.json",
        {
            "schema_version": RUNTIME_EXECUTION_PLAN_SCHEMA_VERSION,
            "kind": "training-runtime-execution-plan",
            "canonical_training_schedule": canonical_schedule.as_dict(),
            "runtime_execution_plan": execution_plan.as_dict(),
        },
    )
    global_step = coordinates.canonical_global_step
    runtime_global_step = coordinates.runtime_global_step
    starting_epoch = coordinates.starting_epoch
    resume_batch_index = coordinates.resume_batch_index
    last_evaluation_step = -1
    last_checkpoint_step = -1
    last_runtime_evaluation_step = -1
    last_runtime_checkpoint_step = -1
    validation_metrics: dict[str, Any] = {}
    last_validation_flat_metrics: dict[str, float] = {}
    best_checkpoint: Path | None = None
    last_ranking: dict[str, Any] = {}
    accumulated_loss_sum: Tensor | None = None
    accumulated_loss_count = 0
    accumulated_microbatch_samples = 0
    processed_train_samples = 0
    completed_epochs = 0
    early_stopping = EarlyStoppingState()
    stop_reason = "epochs_completed"

    try:
        reconciliation = reconcile_checkpoint_storage(
            tracking.directory,
            monitor=config.training.checkpoint_monitor,
            mode=config.training.checkpoint_mode,
            save_top_k=config.training.checkpoint_save_top_k,
            resume_checkpoint=config.training.resume_checkpoint,
        )
        existing_best = reconciliation.get("best_checkpoint")
        if isinstance(existing_best, str):
            best_checkpoint = Path(existing_best)
        if config.training.resume_checkpoint is not None:
            resume_checkpoint = Path(config.training.resume_checkpoint)
            validate_checkpoint_selection(
                tracking.directory,
                retained_checkpoint=resume_checkpoint.name,
                require_latest_step=True,
            )
            state = load_checkpoint(
                resume_checkpoint,
                bundle.model,
                optimizer,
                scheduler,
                config=config,
            )
            if state.get("training_stage") != config.training.stage:
                raise ValueError("Resume checkpoint belongs to a different training stage")
            if state.get("model_architecture_sha256") != config.model_architecture_digest():
                raise ValueError("Resume checkpoint model architecture digest differs")
            loaded_coordinates = resume_coordinates(
                state,
                schedule=canonical_schedule,
                runtime_optimizer_steps_per_epoch=optimizer_steps_per_epoch,
                runtime_batch_count=len(train_loader),
                gradient_accumulation_steps=(
                    batch_plan.gradient_accumulation_steps
                ),
            )
            if loaded_coordinates != coordinates:
                raise RuntimeError("Resume checkpoint changed during training setup")
            global_step = loaded_coordinates.canonical_global_step
            runtime_global_step = loaded_coordinates.runtime_global_step
            last_checkpoint_step = global_step
            last_evaluation_step = global_step
            last_runtime_checkpoint_step = runtime_global_step
            last_runtime_evaluation_step = runtime_global_step
            starting_epoch = loaded_coordinates.starting_epoch
            resume_batch_index = loaded_coordinates.resume_batch_index
            (
                processed_train_samples,
                completed_epochs,
                early_stopping,
            ) = _restore_training_progress(
                state.get("training_progress"),
                canonical_schedule=canonical_schedule,
            )
            if completed_epochs != starting_epoch:
                raise ValueError(
                    "Checkpoint completed epochs disagree with canonical progress"
                )
            _realign_scheduler(
                scheduler,
                runtime_global_step=runtime_global_step,
                warmup_steps=runtime_warmup_steps,
                total_steps=runtime_configured_steps,
            )
            stored_metrics = state.get("metrics")
            if not isinstance(stored_metrics, dict) or any(
                not isinstance(value, int | float)
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                for value in stored_metrics.values()
            ):
                raise ValueError("Resume checkpoint validation metrics are invalid")
            last_validation_flat_metrics = {
                str(key): float(value) for key, value in stored_metrics.items()
            }
            validation_metrics = _unflatten_metrics(last_validation_flat_metrics)
        if global_step > canonical_schedule.configured_optimizer_steps:
            raise ValueError("Resume checkpoint exceeds the configured training budget")
        if runtime_global_step > runtime_configured_steps:
            raise ValueError("Mapped resume checkpoint exceeds the runtime training budget")

        optimizer.zero_grad(set_to_none=True)
        bundle.model.train()
        stop_training = (
            runtime_global_step >= runtime_configured_steps
            or global_step >= canonical_schedule.configured_optimizer_steps
            or early_stopping.triggered
        )
        if early_stopping.triggered:
            stop_reason = "early_stopping"
        for epoch in range(starting_epoch, config.training.epochs):
            if stop_training:
                break
            epoch_start_batch = resume_batch_index if epoch == starting_epoch else 0
            train_batch_sampler.set_epoch(
                epoch,
                start_batch_index=epoch_start_batch,
            )
            for batch_index, batch in enumerate(
                iter_device_batches(train_loader, device),
                start=epoch_start_batch,
            ):
                with _autocast_context(config, device):
                    output = forward_batch(bundle, batch, config, device)
                    if output.loss is None:
                        raise RuntimeError("Training forward pass did not produce a loss")
                    divisor = _gradient_divisor(
                        batch_index,
                        len(train_loader),
                        batch_plan.gradient_accumulation_steps,
                    )
                    loss = output.loss / divisor
                    detached_loss = output.loss.detach()
                    accumulated_loss_sum = (
                        detached_loss
                        if accumulated_loss_sum is None
                        else accumulated_loss_sum + detached_loss
                    )
                    accumulated_loss_count += 1
                    accumulated_microbatch_samples += int(batch["target_alpha"].shape[0])
                loss.backward()
                should_step = (
                    batch_index + 1
                ) % batch_plan.gradient_accumulation_steps == 0 or batch_index + 1 == len(
                    train_loader
                )
                if not should_step:
                    continue
                torch.nn.utils.clip_grad_norm_(trainable, config.training.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                runtime_global_step += 1
                processed_train_samples += accumulated_microbatch_samples
                accumulated_microbatch_samples = 0
                if runtime_global_step in loss_logging_alignment:
                    global_step = loss_logging_alignment[runtime_global_step]
                    if accumulated_loss_sum is None or accumulated_loss_count < 1:
                        raise RuntimeError("Loss logging cadence has no accumulated loss")
                    logged_loss = float(
                        (accumulated_loss_sum / accumulated_loss_count).cpu()
                    )
                    tracking.log(
                        {
                            "train/loss": logged_loss,
                            "train/pinball_loss": logged_loss,
                            "train/epoch": float(epoch),
                            "train/stage_fraction": config.data.train_fraction,
                            "train/task_learning_rate": float(
                                optimizer.param_groups[0]["lr"]
                            ),
                            "train/lora_learning_rate": float(
                                optimizer.param_groups[-1]["lr"]
                                if len(optimizer.param_groups) > 1
                                else 0.0
                            ),
                        },
                        step=global_step,
                    )
                    accumulated_loss_sum = None
                    accumulated_loss_count = 0
                if runtime_global_step in evaluation_alignment:
                    global_step = evaluation_alignment[runtime_global_step]
                    validation_metrics = evaluate_loader(
                        bundle,
                        validation_loader,
                        config,
                        device,
                    )
                    last_evaluation_step = global_step
                    last_runtime_evaluation_step = runtime_global_step
                    selection = _validation_monitor_value(
                        validation_metrics,
                        config.training.checkpoint_monitor,
                    )
                    early_stopping.observe(
                        selection,
                        mode=config.training.checkpoint_mode,
                        min_delta=config.training.early_stopping_min_delta,
                        patience=config.training.early_stopping_patience_evaluations,
                        epoch_number=epoch + 1,
                        start_epoch=config.training.early_stopping_start_epoch,
                        enabled=config.training.early_stopping_enabled,
                    )
                    last_validation_flat_metrics = _flatten_metrics(validation_metrics)
                    tracking.log(
                        {
                            **{
                                f"validation/{key}": value
                                for key, value in last_validation_flat_metrics.items()
                            },
                            "validation/early_stopping/best_value": (early_stopping.best_value),
                            "validation/early_stopping/stale_evaluations": (
                                early_stopping.stale_evaluations
                            ),
                            "validation/early_stopping/triggered": int(early_stopping.triggered),
                            "validation/evaluation_index": early_stopping.evaluation_count,
                            "validation/epoch_number": epoch + 1,
                        },
                        step=global_step,
                    )
                    completed_epochs_at_step = (
                        epoch + 1
                        if global_step
                        == (epoch + 1) * canonical_schedule.optimizer_steps_per_epoch
                        else epoch
                    )
                    checkpoint, last_ranking = save_ranked_checkpoint(
                        model=bundle.model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        config=config,
                        run_directory=tracking.directory,
                        global_step=global_step,
                        epoch=epoch,
                        batch_index=batch_index,
                        selection_metric_name=config.training.checkpoint_monitor,
                        selection_metric_value=selection,
                        selection_metric_mode=config.training.checkpoint_mode,
                        save_top_k=config.training.checkpoint_save_top_k,
                        metrics=last_validation_flat_metrics,
                        training_progress=_training_progress(
                            processed_train_samples=processed_train_samples,
                            completed_epochs=completed_epochs_at_step,
                            early_stopping=early_stopping,
                            canonical_schedule=canonical_schedule,
                            runtime_execution_plan=execution_plan,
                        ),
                    )
                    best_path = last_ranking.get("best_checkpoint")
                    if isinstance(best_path, str):
                        best_checkpoint = Path(best_path)
                    elif checkpoint is not None:
                        best_checkpoint = checkpoint
                    last_checkpoint_step = global_step
                    last_runtime_checkpoint_step = runtime_global_step
                    completed_epochs = completed_epochs_at_step
                    if early_stopping.triggered:
                        stop_reason = "early_stopping"
                        stop_training = True
                if runtime_global_step >= runtime_configured_steps:
                    stop_training = True
                    break
                if early_stopping.triggered:
                    break
            if stop_training:
                break

        if (
            last_evaluation_step != global_step
            or last_checkpoint_step != global_step
            or last_runtime_evaluation_step != runtime_global_step
            or last_runtime_checkpoint_step != runtime_global_step
        ):
            raise RuntimeError(
                "Training stopped outside the configured epoch-relative validation schedule"
            )
        if best_checkpoint is None:
            raise RuntimeError("Training completed without a validation-ranked checkpoint")
        if not last_validation_flat_metrics:
            raise RuntimeError("Training completed without finite validation metrics")
        if stop_reason == "epochs_completed":
            if (
                global_step != canonical_schedule.configured_optimizer_steps
                or runtime_global_step != runtime_configured_steps
                or completed_epochs != config.training.epochs
            ):
                raise RuntimeError("Training ended before every configured epoch completed")
        elif not early_stopping.triggered:
            raise RuntimeError("Early-stopping completion has no triggered stopping state")

        architecture_digest = config.model_architecture_digest()
        planned_train_samples = selected_train_samples * config.training.epochs
        completion_result = save_training_completion_result(
            model=bundle.model,
            config=config,
            run_directory=tracking.directory,
            global_step=global_step,
            completed_epochs=completed_epochs,
            processed_train_samples=processed_train_samples,
            planned_train_samples=planned_train_samples,
            validation_evaluations=early_stopping.evaluation_count,
            stop_reason=stop_reason,
            early_stopping_state=early_stopping.as_dict(),
            metrics=last_validation_flat_metrics,
        )
        tracking.update_summary(
            {
                "training_stage": config.training.stage,
                "training_fraction": config.data.train_fraction,
                "training_max_samples": config.data.max_samples,
                "dataset_profile": config.data.dataset_profile,
                "selected_datasets": config.data.selected_datasets,
                "selected_train_samples_per_epoch": selected_train_samples,
                "planned_train_samples": planned_train_samples,
                "processed_train_samples": processed_train_samples,
                "configured_optimizer_steps": (
                    canonical_schedule.configured_optimizer_steps
                ),
                "completed_optimizer_steps": global_step,
                "optimizer_step_coverage_ratio": (
                    global_step / canonical_schedule.configured_optimizer_steps
                ),
                "runtime_configured_optimizer_steps": runtime_configured_steps,
                "runtime_completed_optimizer_steps": runtime_global_step,
                "completed_epochs": completed_epochs,
                "validation_evaluations": early_stopping.evaluation_count,
                "stop_reason": stop_reason,
                "early_stopped": stop_reason == "early_stopping",
                "early_stopping": early_stopping.as_dict(),
                "training_batch_padding_per_epoch": (train_batch_sampler.padded_sample_count),
                "runtime_batch_plan": batch_plan.as_dict(),
                "runtime_execution_plan": execution_plan.as_dict(),
                "runtime_execution_plan_path": str(runtime_execution_plan_path),
                "canonical_training_schedule": canonical_schedule.as_dict(),
                "loss_logging_plan": loss_logging_plan,
                "runtime_label_calibration": {
                    "source_split": "train",
                    "sample_count": robust_scale_result.sample_count,
                    "seed": config.training.seed + 17,
                    "horizons": config.data.alpha_horizons,
                    "method": "max(iqr,mad_x_1.4826,1e-4)",
                    "robust_scales": list(robust_scales),
                    "parallel_backend": "pytorch_dataloader_processes",
                    "effective_workers": worker_plan.effective_workers,
                    "cache_hit": robust_scale_result.cache_hit,
                    "cache_path": str(robust_scale_result.cache_path),
                    "cache_identity_sha256": robust_scale_result.identity_sha256,
                },
                "dataloader_worker_plan": worker_plan.as_dict(),
                "model_architecture_sha256": architecture_digest,
                "lora_module_names": list(bundle.lora_module_names),
                "lora_parameter_names": list(bundle.lora_parameter_names),
                "checkpoint_policy": {
                    "selection_source": "validation",
                    "monitor": config.training.checkpoint_monitor,
                    "mode": config.training.checkpoint_mode,
                    "save_top_k": config.training.checkpoint_save_top_k,
                    "evaluations_per_epoch": config.training.evaluations_per_epoch,
                },
                "best_checkpoint": str(best_checkpoint),
                "completion_result": str(completion_result),
                "global_step": global_step,
                **{
                    f"validation/{key}": value
                    for key, value in last_validation_flat_metrics.items()
                },
            }
        )
        if config.wandb.log_model_artifact:
            tracking.log_model_artifact(
                best_checkpoint,
                aliases=["best", "validation-selected", config.training.stage],
            )
        tracking.finish(exit_code=0)
        return TrainingResult(
            run_directory=tracking.directory,
            final_checkpoint=best_checkpoint,
            global_step=global_step,
            stage=config.training.stage,
            model_architecture_sha256=architecture_digest,
            dataset_profile=config.data.dataset_profile,
            selected_datasets=tuple(config.data.selected_datasets),
            selected_train_samples_per_epoch=selected_train_samples,
            processed_train_samples=processed_train_samples,
            configured_optimizer_steps=(
                canonical_schedule.configured_optimizer_steps
            ),
            completed_epochs=completed_epochs,
            validation_evaluations=early_stopping.evaluation_count,
            stop_reason=stop_reason,
            completion_result=completion_result,
            robust_scales=robust_scales,
            validation_metrics=validation_metrics,
        )
    except BaseException:
        with suppress(BaseException):
            tracking.update_summary(
                {
                    "failed": True,
                    "global_step": global_step,
                    "runtime_global_step": runtime_global_step,
                    "runtime_execution_plan": execution_plan.as_dict(),
                    "runtime_execution_plan_path": str(runtime_execution_plan_path),
                    "training_stage": config.training.stage,
                }
            )
        with suppress(BaseException):
            tracking.finish(exit_code=1)
        raise


def write_training_result(result: TrainingResult) -> None:
    print(
        json.dumps(
            {
                "run_directory": str(result.run_directory),
                "final_checkpoint": str(result.final_checkpoint),
                "completion_result": str(result.completion_result),
                "global_step": result.global_step,
                "training_stage": result.stage,
                "model_architecture_sha256": result.model_architecture_sha256,
                "dataset_profile": result.dataset_profile,
                "selected_datasets": list(result.selected_datasets),
                "selected_train_samples_per_epoch": (result.selected_train_samples_per_epoch),
                "processed_train_samples": result.processed_train_samples,
                "configured_optimizer_steps": result.configured_optimizer_steps,
                "completed_epochs": result.completed_epochs,
                "validation_evaluations": result.validation_evaluations,
                "stop_reason": result.stop_reason,
                "early_stopped": result.stop_reason == "early_stopping",
                "robust_scales": list(result.robust_scales),
                "validation_metrics": result.validation_metrics,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
