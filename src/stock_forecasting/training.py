"""Single-GPU quant-only Kronos LoRA training and evaluation."""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from collections.abc import Sized
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, Subset

from stock_forecasting.checkpointing import (
    load_checkpoint,
    reconcile_checkpoint_storage,
    save_ranked_checkpoint,
    validate_checkpoint_selection,
)
from stock_forecasting.config import ExperimentConfig
from stock_forecasting.data import (
    FinancialBatchCollator,
    FinancialWindowDataset,
    read_processed_records,
)
from stock_forecasting.factory import ModelBundle, build_model_bundle
from stock_forecasting.metrics import (
    POSTPROCESS_SIGNAL_NAMES,
    cross_sectional_metrics,
    multi_horizon_alpha_metrics,
    postprocess_alpha_signal,
)
from stock_forecasting.preflight import run_preflight
from stock_forecasting.tracking import (
    TrackingRun,
    start_tracking,
    training_run_lease,
    validate_tracking_run_contract,
)
from stock_forecasting.training_paths import resolve_processed_dataset_path


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_processed_dataset(path: Path) -> Path:
    return resolve_processed_dataset_path(path)


def deterministic_stratified_indices(
    dataset: FinancialWindowDataset,
    *,
    fraction: float,
    max_samples: int | None = None,
) -> list[int]:
    """Select an exact, deterministic fraction stratified by market and asset type."""

    count = len(dataset)
    target = count if fraction >= 1.0 else max(1, int(count * fraction))
    if max_samples is not None:
        target = min(target, max_samples)
    if target >= count:
        return list(range(count))

    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, record in enumerate(dataset.records):
        metadata = record.get("metadata", {})
        key = (
            str(metadata.get("market", "unknown")),
            str(record.get("asset_type", "unknown")),
        )
        groups[key].append(index)

    effective_fraction = target / count
    quotas: dict[tuple[str, str], int] = {}
    fractional: list[tuple[float, tuple[str, str]]] = []
    for key, indices in sorted(groups.items()):
        exact = len(indices) * effective_fraction
        quota = min(len(indices), math.floor(exact))
        quotas[key] = quota
        fractional.append((exact - quota, key))
    remaining = target - sum(quotas.values())
    for _fractional_part, key in sorted(fractional, key=lambda item: (-item[0], item[1])):
        if remaining == 0:
            break
        if quotas[key] < len(groups[key]):
            quotas[key] += 1
            remaining -= 1
    if remaining:
        for key in sorted(groups):
            while remaining and quotas[key] < len(groups[key]):
                quotas[key] += 1
                remaining -= 1
    if remaining:
        raise RuntimeError("Could not allocate the requested deterministic subset")

    selected: list[int] = []
    for key, indices in sorted(groups.items()):
        ordered = sorted(
            indices,
            key=lambda index: (
                str(dataset.records[index]["cutoff_at"]),
                str(dataset.records[index]["symbol"]),
                index,
            ),
        )
        quota = quotas[key]
        if quota == 0:
            continue
        positions = np.linspace(0, len(ordered) - 1, num=quota, dtype=np.int64)
        selected.extend(ordered[int(position)] for position in positions)
    selected = sorted(set(selected))
    if len(selected) != target:
        raise RuntimeError(
            f"Deterministic subset selected {len(selected)} rows; expected exactly {target}"
        )
    return selected


def build_dataloaders(
    config: ExperimentConfig,
) -> tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any]]:
    dataset_path = resolve_processed_dataset(config.data.processed_path)
    records = read_processed_records(dataset_path)
    train_source = FinancialWindowDataset(records, split="train")
    validation_dataset = FinancialWindowDataset(records, split="validation")
    test_dataset = FinancialWindowDataset(records, split="test")
    indices = deterministic_stratified_indices(
        train_source,
        fraction=config.data.train_fraction,
        max_samples=config.data.max_samples,
    )
    train_dataset: Dataset[Any] = (
        train_source if len(indices) == len(train_source) else Subset(train_source, indices)
    )
    collator = FinancialBatchCollator()
    pin_memory = torch.cuda.is_available()
    generator = torch.Generator().manual_seed(config.training.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collator,
        num_workers=config.training.num_workers,
        pin_memory=pin_memory,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.training.evaluation_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=config.training.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.evaluation_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=config.training.num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, validation_loader, test_loader


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
    return bundle.model(
        batch["asset_series"].to(device),
        batch["benchmark_series"].to(device),
        asset_attention_mask=batch["asset_attention_mask"].to(device),
        benchmark_attention_mask=batch["benchmark_attention_mask"].to(device),
        asset_timestamps=batch["asset_timestamps"].to(device),
        benchmark_timestamps=batch["benchmark_timestamps"].to(device),
        target_alpha=batch["target_alpha"].to(device),
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


@dataclass(frozen=True)
class TrainingResult:
    run_directory: Path
    final_checkpoint: Path
    global_step: int
    stage: str
    model_architecture_sha256: str
    dataset_profile: str
    selected_datasets: tuple[str, ...]
    train_samples: int
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
    train_loader, validation_loader, _test_loader = build_dataloaders(config)
    bundle = build_model_bundle(config, device)
    parameter_groups, trainable = _optimizer_parameter_groups(bundle, config)
    optimizer = AdamW(
        parameter_groups,
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    batches_per_epoch = math.ceil(len(train_loader) / config.training.gradient_accumulation_steps)
    configured_steps = batches_per_epoch * config.training.epochs
    total_steps = min(config.training.max_steps or configured_steps, configured_steps)
    if total_steps < 1:
        raise ValueError("Training budget must contain at least one optimizer step")
    scheduler = _scheduler(
        optimizer,
        int(total_steps * config.training.warmup_ratio),
        total_steps,
    )
    tracking: TrackingRun = start_tracking(config)
    global_step = 0
    starting_epoch = 0
    resume_batch_index = 0
    last_epoch = 0
    last_batch_index = -1
    last_evaluation_step = -1
    last_checkpoint_step = -1
    validation_metrics: dict[str, Any] = {}
    best_checkpoint: Path | None = None
    last_ranking: dict[str, Any] = {}

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
            if state.get("model_architecture_sha256") != config.model.architecture_digest():
                raise ValueError("Resume checkpoint model architecture digest differs")
            global_step = int(state["global_step"])
            last_checkpoint_step = global_step
            starting_epoch, resume_batch_index = _resume_coordinates(
                state,
                len(train_loader),
            )
        if global_step > total_steps:
            raise ValueError("Resume checkpoint exceeds the configured training budget")

        optimizer.zero_grad(set_to_none=True)
        bundle.model.train()
        stop_training = global_step >= total_steps
        for epoch in range(starting_epoch, config.training.epochs):
            if stop_training:
                break
            loader_generator = getattr(train_loader, "generator", None)
            if loader_generator is not None:
                loader_generator.manual_seed(config.training.seed + epoch)
            for batch_index, batch in enumerate(train_loader):
                if epoch == starting_epoch and batch_index < resume_batch_index:
                    continue
                last_epoch = epoch
                last_batch_index = batch_index
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

                if global_step % config.training.log_every_steps == 0:
                    tracking.log(
                        {
                            "train/loss": float(output.loss.detach().cpu()),
                            "train/pinball_loss": float(output.loss.detach().cpu()),
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
                if (
                    global_step % config.training.evaluate_every_steps == 0
                    or global_step >= total_steps
                ):
                    validation_metrics = evaluate_loader(
                        bundle,
                        validation_loader,
                        config,
                        device,
                    )
                    last_evaluation_step = global_step
                    tracking.log(
                        _flatten_metrics(validation_metrics, "validation"),
                        step=global_step,
                    )
                    if (
                        global_step % config.training.checkpoint_every_steps == 0
                        or global_step >= total_steps
                    ):
                        selection = _validation_monitor_value(
                            validation_metrics,
                            config.training.checkpoint_monitor,
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
                            metrics=_flatten_metrics(validation_metrics),
                        )
                        best_path = last_ranking.get("best_checkpoint")
                        if isinstance(best_path, str):
                            best_checkpoint = Path(best_path)
                        elif checkpoint is not None:
                            best_checkpoint = checkpoint
                        last_checkpoint_step = global_step
                if global_step >= total_steps:
                    stop_training = True
                    break
            if stop_training:
                break

        if last_evaluation_step != global_step:
            validation_metrics = evaluate_loader(
                bundle,
                validation_loader,
                config,
                device,
            )
            tracking.log(
                _flatten_metrics(validation_metrics, "validation"),
                step=global_step,
            )
        if last_checkpoint_step != global_step:
            selection = _validation_monitor_value(
                validation_metrics,
                config.training.checkpoint_monitor,
            )
            checkpoint, last_ranking = save_ranked_checkpoint(
                model=bundle.model,
                optimizer=optimizer,
                scheduler=scheduler,
                config=config,
                run_directory=tracking.directory,
                global_step=global_step,
                epoch=last_epoch,
                batch_index=last_batch_index,
                selection_metric_name=config.training.checkpoint_monitor,
                selection_metric_value=selection,
                selection_metric_mode=config.training.checkpoint_mode,
                save_top_k=config.training.checkpoint_save_top_k,
                metrics=_flatten_metrics(validation_metrics),
            )
            best_path = last_ranking.get("best_checkpoint")
            if isinstance(best_path, str):
                best_checkpoint = Path(best_path)
            elif checkpoint is not None:
                best_checkpoint = checkpoint
        if best_checkpoint is None:
            raise RuntimeError("Training completed without a validation-ranked checkpoint")

        architecture_digest = config.model.architecture_digest()
        tracking.update_summary(
            {
                "training_stage": config.training.stage,
                "training_fraction": config.data.train_fraction,
                "dataset_profile": config.data.dataset_profile,
                "selected_datasets": config.data.selected_datasets,
                "train_samples": len(cast(Sized, train_loader.dataset)),
                "model_architecture_sha256": architecture_digest,
                "lora_module_names": list(bundle.lora_module_names),
                "lora_parameter_names": list(bundle.lora_parameter_names),
                "checkpoint_policy": {
                    "selection_source": "validation",
                    "monitor": config.training.checkpoint_monitor,
                    "mode": config.training.checkpoint_mode,
                    "save_top_k": config.training.checkpoint_save_top_k,
                },
                "best_checkpoint": str(best_checkpoint),
                "global_step": global_step,
                **_flatten_metrics(validation_metrics, "validation"),
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
            train_samples=len(cast(Sized, train_loader.dataset)),
            validation_metrics=validation_metrics,
        )
    except BaseException:
        tracking.update_summary(
            {
                "failed": True,
                "global_step": global_step,
                "training_stage": config.training.stage,
            }
        )
        tracking.finish(exit_code=1)
        raise


def write_training_result(result: TrainingResult) -> None:
    print(
        json.dumps(
            {
                "run_directory": str(result.run_directory),
                "final_checkpoint": str(result.final_checkpoint),
                "global_step": result.global_step,
                "training_stage": result.stage,
                "model_architecture_sha256": result.model_architecture_sha256,
                "dataset_profile": result.dataset_profile,
                "selected_datasets": list(result.selected_datasets),
                "train_samples": result.train_samples,
                "validation_metrics": result.validation_metrics,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
