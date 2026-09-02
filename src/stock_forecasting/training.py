"""Single-GPU quant-only Kronos LoRA training and evaluation."""

from __future__ import annotations

import json
import math
import os
import random
from contextlib import nullcontext, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

from stock_forecasting.checkpointing import (
    load_checkpoint,
    reconcile_checkpoint_storage,
    save_ranked_checkpoint,
    save_training_completion_result,
    validate_checkpoint_selection,
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

DATALOADER_AUTO_MAX_WORKERS = 8
DATALOADER_CPU_RESERVE = 2
DATALOADER_ACTIVE_PERSISTENT_POOLS = 2
DATALOADER_MEMORY_FRACTION = 0.25
DATALOADER_MEMORY_BYTES_PER_WORKER = 512 * 1024**2
DATALOADER_PARENT_MEMORY_RESERVE_BYTES = 4 * 1024**3
DATALOADER_TOTAL_SYMBOL_CACHE_ENTRIES = 128
DATALOADER_PREFETCH_FACTOR = 2
ROBUST_SCALE_BATCH_SIZE = 256
ROBUST_SCALE_SELECTION_BLOCK_SIZE = 16
ROBUST_SCALE_CACHE_SCHEMA_VERSION = "1.0"
ROBUST_SCALE_ALGORITHM = "parallel-blockwise-runtime-labels-v1"
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
            "native_threads_per_worker": 1,
        }


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


def _requested_dataloader_workers(config: ExperimentConfig) -> tuple[int, str]:
    value = os.environ.get("FIN_TS_DATALOADER_WORKERS", "").strip().lower()
    if not value:
        return config.training.num_workers, "training_config"
    if value == "auto":
        return DATALOADER_AUTO_MAX_WORKERS, "runpod_auto"
    if not value.isascii() or not value.isdigit() or int(value) > 32:
        raise ValueError("FIN_TS_DATALOADER_WORKERS must be auto or an integer from 0 to 32")
    return int(value), "environment"


def plan_dataloader_workers(
    requested_workers: int,
    *,
    source: str = "explicit",
    visible_cpu_count: int | None = None,
    available_memory_bytes: int | None = None,
) -> DataLoaderWorkerPlan:
    """Choose a conservative process count shared by loading and calibration."""

    if isinstance(requested_workers, bool) or requested_workers < 0 or requested_workers > 32:
        raise ValueError("requested_workers must be between 0 and 32")
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
        prefetch_factor=DATALOADER_PREFETCH_FACTOR if effective_workers else None,
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


def build_dataloaders(
    config: ExperimentConfig,
    *,
    worker_plan: DataLoaderWorkerPlan | None = None,
) -> tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any]]:
    if worker_plan is None:
        requested_workers, source = _requested_dataloader_workers(config)
        worker_plan = plan_dataloader_workers(requested_workers, source=source)
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
    train_sampler = BlockwisePermutationSampler(
        len(train_source),
        fraction=config.data.train_fraction,
        max_samples=config.data.max_samples,
        seed=config.training.seed,
        block_size=max(1, config.training.batch_size // 2),
    )
    validation_sampler = BlockwisePermutationSampler(
        len(validation_dataset),
        max_samples=config.training.evaluation_max_samples,
        seed=config.training.seed + 1,
        block_size=max(1, config.training.evaluation_batch_size // 2),
    )
    test_sampler = BlockwisePermutationSampler(
        len(test_dataset),
        max_samples=config.training.evaluation_max_samples,
        seed=config.training.seed + 2,
        block_size=max(1, config.training.evaluation_batch_size // 2),
    )
    if len(train_sampler) < config.training.batch_size:
        raise ValueError("Selected training samples do not fill one fixed-size batch")
    train_batch_sampler = FixedSizeBatchSampler(
        train_sampler,
        batch_size=config.training.batch_size,
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
        batch_size=config.training.evaluation_batch_size,
        sampler=validation_sampler,
        shuffle=False,
        collate_fn=collator,
        pin_memory=pin_memory,
        **_loader_process_options(worker_plan, persistent=True),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.evaluation_batch_size,
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


def forward_batch(
    bundle: ModelBundle,
    batch: dict[str, Any],
    config: ExperimentConfig,
    device: torch.device,
) -> Any:
    del config
    non_blocking = device.type == "cuda"
    return bundle.model(
        batch["asset_series"].to(device, non_blocking=non_blocking),
        batch["benchmark_series"].to(device, non_blocking=non_blocking),
        asset_attention_mask=batch["asset_attention_mask"].to(device, non_blocking=non_blocking),
        benchmark_attention_mask=batch["benchmark_attention_mask"].to(
            device, non_blocking=non_blocking
        ),
        asset_timestamps=batch["asset_timestamps"].to(device, non_blocking=non_blocking),
        benchmark_timestamps=batch["benchmark_timestamps"].to(device, non_blocking=non_blocking),
        target_alpha=batch["target_alpha"].to(device, non_blocking=non_blocking),
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


@torch.no_grad()
def evaluate_loader(
    bundle: ModelBundle,
    loader: DataLoader[Any],
    config: ExperimentConfig,
    device: torch.device,
) -> dict[str, Any]:
    was_training = bundle.model.training
    bundle.model.eval()
    targets: list[NDArray[np.float32]] = []
    quantile_predictions: list[NDArray[np.float32]] = []
    losses: list[float] = []
    cutoff_dates: list[str] = []
    years: list[str] = []
    symbols: list[str] = []
    asset_types: list[str] = []
    markets: list[str] = []
    providers: list[str] = []

    for batch in loader:
        with _autocast_context(config, device):
            output = forward_batch(bundle, batch, config, device)
        if output.loss is not None:
            losses.append(float(output.loss.detach().cpu()))
        targets.append(batch["target_alpha"].numpy())
        quantile_predictions.append(output.alpha_quantiles.float().cpu().numpy())
        batch_cutoffs = [str(value) for value in batch["cutoff_at"]]
        cutoff_dates.extend(batch_cutoffs)
        years.extend(value[:4] for value in batch_cutoffs)
        symbols.extend(str(value) for value in batch["symbols"])
        asset_types.extend(str(value) for value in batch["asset_types"])
        markets.extend(str(value) for value in batch["markets"])
        providers.extend(str(value) for value in batch["providers"])

    target_array = np.concatenate(targets).astype(np.float32)
    quantile_array = np.concatenate(quantile_predictions).astype(np.float32)
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
        "loss": float(np.mean(losses)) if losses else None,
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


def _scheduler(optimizer: AdamW, warmup_steps: int, total_steps: int) -> LambdaLR:
    def schedule(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        remaining = max(total_steps - warmup_steps, 1)
        progress = min(max(step - warmup_steps, 0) / remaining, 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, schedule)


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


def _resume_coordinates(state: dict[str, Any], batch_count: int) -> tuple[int, int]:
    epoch = int(state["epoch"])
    batch_index = int(state["batch_index"]) + 1
    if batch_index >= batch_count:
        return epoch + 1, 0
    return epoch, batch_index


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
    configured_optimizer_steps: int,
    early_stopping: EarlyStoppingState,
) -> dict[str, Any]:
    return {
        "processed_train_samples": processed_train_samples,
        "completed_epochs": completed_epochs,
        "configured_optimizer_steps": configured_optimizer_steps,
        "early_stopping": early_stopping.as_dict(),
    }


def _restore_training_progress(
    payload: Any,
    *,
    configured_optimizer_steps: int,
) -> tuple[int, int, EarlyStoppingState]:
    if not isinstance(payload, dict) or set(payload) != {
        "processed_train_samples",
        "completed_epochs",
        "configured_optimizer_steps",
        "early_stopping",
    }:
        raise ValueError("Checkpoint training progress is incomplete")
    processed_train_samples = payload["processed_train_samples"]
    completed_epochs = payload["completed_epochs"]
    stored_optimizer_steps = payload["configured_optimizer_steps"]
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in (
            processed_train_samples,
            completed_epochs,
            stored_optimizer_steps,
        )
    ):
        raise ValueError("Checkpoint training progress counters are invalid")
    if stored_optimizer_steps != configured_optimizer_steps:
        raise ValueError("Checkpoint optimizer-step budget differs from the selected config")
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
    requested_workers, worker_source = _requested_dataloader_workers(config)
    worker_plan = plan_dataloader_workers(
        requested_workers,
        source=worker_source,
    )
    print(
        json.dumps(
            {"dataloader_worker_plan": worker_plan.as_dict()},
            sort_keys=True,
        ),
        flush=True,
    )
    train_loader, validation_loader, _test_loader = build_dataloaders(
        config,
        worker_plan=worker_plan,
    )
    train_dataset = cast(LazyFinancialWindowDataset, train_loader.dataset)
    train_batch_sampler = cast(FixedSizeBatchSampler, train_loader.batch_sampler)
    selected_train_samples = len(train_batch_sampler.sampler)
    robust_scale_result = resolve_runtime_robust_scales(
        train_dataset,
        sample_count=config.data.label_scale_calibration_samples,
        seed=config.training.seed + 17,
        worker_plan=worker_plan,
    )
    robust_scales = robust_scale_result.scales
    bundle = build_model_bundle(config, device, robust_scales=robust_scales)
    parameter_groups, trainable = _optimizer_parameter_groups(bundle, config)
    optimizer = AdamW(
        parameter_groups,
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    optimizer_steps_per_epoch = math.ceil(
        len(train_loader) / config.training.gradient_accumulation_steps
    )
    configured_steps = optimizer_steps_per_epoch * config.training.epochs
    if configured_steps < 1:
        raise ValueError("Training budget must contain at least one optimizer step")
    evaluation_steps = {
        step
        for epoch_index in range(config.training.epochs)
        for step in epoch_evaluation_steps(
            epoch_index,
            optimizer_steps_per_epoch,
            config.training.evaluations_per_epoch,
        )
    }
    scheduler = _scheduler(
        optimizer,
        int(configured_steps * config.training.warmup_ratio),
        configured_steps,
    )
    tracking: TrackingRun = start_tracking(config)
    global_step = 0
    starting_epoch = 0
    resume_batch_index = 0
    last_evaluation_step = -1
    last_checkpoint_step = -1
    validation_metrics: dict[str, Any] = {}
    last_validation_flat_metrics: dict[str, float] = {}
    best_checkpoint: Path | None = None
    last_ranking: dict[str, Any] = {}
    accumulated_microbatch_losses: list[float] = []
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
            global_step = int(state["global_step"])
            last_checkpoint_step = global_step
            last_evaluation_step = global_step
            starting_epoch, resume_batch_index = _resume_coordinates(
                state,
                len(train_loader),
            )
            (
                processed_train_samples,
                completed_epochs,
                early_stopping,
            ) = _restore_training_progress(
                state.get("training_progress"),
                configured_optimizer_steps=configured_steps,
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
        if global_step > configured_steps:
            raise ValueError("Resume checkpoint exceeds the configured training budget")

        optimizer.zero_grad(set_to_none=True)
        bundle.model.train()
        stop_training = global_step >= configured_steps or early_stopping.triggered
        if early_stopping.triggered:
            stop_reason = "early_stopping"
        for epoch in range(starting_epoch, config.training.epochs):
            if stop_training:
                break
            train_batch_sampler.set_epoch(epoch)
            for batch_index, batch in enumerate(train_loader):
                if epoch == starting_epoch and batch_index < resume_batch_index:
                    continue
                with _autocast_context(config, device):
                    output = forward_batch(bundle, batch, config, device)
                    if output.loss is None:
                        raise RuntimeError("Training forward pass did not produce a loss")
                    divisor = _gradient_divisor(
                        batch_index,
                        len(train_loader),
                        config.training.gradient_accumulation_steps,
                    )
                    loss = output.loss / divisor
                    accumulated_microbatch_losses.append(float(output.loss.detach().cpu()))
                    accumulated_microbatch_samples += int(batch["target_alpha"].shape[0])
                loss.backward()
                should_step = (
                    batch_index + 1
                ) % config.training.gradient_accumulation_steps == 0 or batch_index + 1 == len(
                    train_loader
                )
                if not should_step:
                    continue
                torch.nn.utils.clip_grad_norm_(trainable, config.training.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                processed_train_samples += accumulated_microbatch_samples
                accumulated_microbatch_samples = 0
                optimizer_step_loss = float(np.mean(accumulated_microbatch_losses))
                accumulated_microbatch_losses.clear()
                tracking.log(
                    {
                        "train/loss": optimizer_step_loss,
                        "train/pinball_loss": optimizer_step_loss,
                        "train/epoch": float(epoch),
                        "train/stage_fraction": config.data.train_fraction,
                        "train/task_learning_rate": float(optimizer.param_groups[0]["lr"]),
                        "train/lora_learning_rate": float(
                            optimizer.param_groups[-1]["lr"]
                            if len(optimizer.param_groups) > 1
                            else 0.0
                        ),
                    },
                    step=global_step,
                )
                if global_step in evaluation_steps:
                    validation_metrics = evaluate_loader(
                        bundle,
                        validation_loader,
                        config,
                        device,
                    )
                    last_evaluation_step = global_step
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
                        epoch + 1 if batch_index + 1 == len(train_loader) else epoch
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
                            configured_optimizer_steps=configured_steps,
                            early_stopping=early_stopping,
                        ),
                    )
                    best_path = last_ranking.get("best_checkpoint")
                    if isinstance(best_path, str):
                        best_checkpoint = Path(best_path)
                    elif checkpoint is not None:
                        best_checkpoint = checkpoint
                    last_checkpoint_step = global_step
                    completed_epochs = completed_epochs_at_step
                    if early_stopping.triggered:
                        stop_reason = "early_stopping"
                        stop_training = True
                if global_step >= configured_steps:
                    stop_training = True
                    break
                if early_stopping.triggered:
                    break
            if stop_training:
                break

        if last_evaluation_step != global_step or last_checkpoint_step != global_step:
            raise RuntimeError(
                "Training stopped outside the configured epoch-relative validation schedule"
            )
        if best_checkpoint is None:
            raise RuntimeError("Training completed without a validation-ranked checkpoint")
        if not last_validation_flat_metrics:
            raise RuntimeError("Training completed without finite validation metrics")
        if stop_reason == "epochs_completed":
            if global_step != configured_steps or completed_epochs != config.training.epochs:
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
                "configured_optimizer_steps": configured_steps,
                "completed_optimizer_steps": global_step,
                "optimizer_step_coverage_ratio": global_step / configured_steps,
                "completed_epochs": completed_epochs,
                "validation_evaluations": early_stopping.evaluation_count,
                "stop_reason": stop_reason,
                "early_stopped": stop_reason == "early_stopping",
                "early_stopping": early_stopping.as_dict(),
                "training_batch_padding_per_epoch": (train_batch_sampler.padded_sample_count),
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
            configured_optimizer_steps=configured_steps,
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
