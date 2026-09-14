"""Fail-closed preflight checks for offline quant-only training."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import torch

from stock_forecasting.config import ExperimentConfig
from stock_forecasting.data.manifest import (
    validate_dataset_storage_contract,
    validate_training_dataset_manifest,
)
from stock_forecasting.dataset_identity import MIN_FIXED_EVALUATION_DATES
from stock_forecasting.factory import verify_kronos_source_revision
from stock_forecasting.training_paths import resolve_bar_store_path
from stock_forecasting.training_stage_contract import PRODUCTION_STAGE_SAMPLE_CONTRACTS


@dataclass
class PreflightReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    facts: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def require_success(self) -> None:
        if self.errors:
            raise RuntimeError("Preflight failed:\n- " + "\n- ".join(self.errors))


def _is_ephemeral_runpod_path(path: Path) -> bool:
    resolved = path.expanduser().resolve(strict=False)
    return resolved == Path("/workspace") or Path("/workspace") in resolved.parents


def _persistent_paths(config: ExperimentConfig) -> list[Path]:
    return [
        config.data.raw_path,
        config.data.bar_store_path,
        config.data.resolved_manifest_path,
        config.training.output_root,
        config.validation.output_root,
        config.runtime.log_root,
        config.wandb.directory,
        config.model.kronos_source_root,
    ]


def run_preflight(
    config: ExperimentConfig,
    *,
    require_data: bool = True,
    enforce_runtime_limit: bool = True,
    allow_historical_inference: bool = False,
) -> PreflightReport:
    """Validate paths, immutable Parquet provenance, and numerical model scope."""

    del enforce_runtime_limit
    report = PreflightReport()
    on_runpod = bool(os.environ.get("RUNPOD_POD_ID"))
    report.facts.update(
        {
            "environment": "runpod" if on_runpod else "local",
            "training_stage": config.training.stage,
            "training_fraction": str(config.data.train_fraction),
            "training_max_samples": (
                "unbounded"
                if config.data.max_samples is None
                else str(config.data.max_samples)
            ),
            "dataset_profile": config.data.dataset_profile,
            "selected_datasets": ",".join(config.data.selected_datasets),
            "model_architecture_sha256": config.model_architecture_digest(),
        }
    )

    if on_runpod:
        for path in _persistent_paths(config):
            if _is_ephemeral_runpod_path(path):
                report.errors.append(f"Persistent artifact path uses /workspace: {path}")

    if require_data:
        try:
            bar_store = resolve_bar_store_path(config.data.bar_store_path)
            report.facts["bar_store"] = str(bar_store)
        except (FileNotFoundError, ValueError) as error:
            report.errors.append(str(error))
            bar_store = config.data.bar_store_path
        if config.data.require_ready_manifest:
            try:
                manifest = validate_training_dataset_manifest(
                    config.data.resolved_manifest_path,
                    profile=config.data.dataset_profile,
                    raw_path=config.data.raw_path,
                    bar_store_path=bar_store,
                    require_current_pipeline=not (
                        allow_historical_inference and config.data.fixed_split is None
                        and config.model.feature_mode == "baseline"
                    ),
                )
            except (FileNotFoundError, ValueError) as error:
                report.errors.append(str(error))
            else:
                try:
                    storage_spec = validate_dataset_storage_contract(
                        manifest,
                        input_length=config.data.input_length,
                        max_horizon=config.data.max_horizon,
                        benchmark_mapping_path=config.data.benchmark_mapping_path,
                        effective_embargo_trading_days=(
                            config.data.effective_embargo_trading_days
                        ),
                        max_abs_log_return=config.data.max_abs_log_return,
                        fixed_split=config.data.fixed_split,
                        minimum_evaluation_dates=(
                            MIN_FIXED_EVALUATION_DATES
                            if config.model.time_series_backend == "kronos" else 0
                        ),
                    )
                except ValueError as error:
                    report.errors.append(str(error))
                else:
                    report.facts["dataset_manifest"] = str(config.data.resolved_manifest_path)
                    report.facts["split_counts"] = str(manifest["split_counts"])
                    report.facts["raw_rows"] = str(manifest["artifacts"]["raw"]["row_count"])
                    report.facts["valid_cutoffs"] = str(
                        sum(manifest["split_counts"].values())
                    )
                    report.facts["storage_preparation_spec_sha256"] = str(
                        manifest["storage_preparation_spec_sha256"]
                    )
                    report.facts["input_length"] = str(storage_spec["window_size"])
        elif not config.data.raw_path.is_file():
            report.errors.append(f"Raw Parquet is missing: {config.data.raw_path}")

    if config.model.time_series_backend == "kronos":
        revision = config.model.kronos_source_revision
        assert revision is not None
        try:
            verified_revision = verify_kronos_source_revision(
                config.model.kronos_source_root,
                revision,
            )
        except (FileNotFoundError, RuntimeError, ValueError) as error:
            report.errors.append(str(error))
        else:
            report.facts["kronos_source_revision"] = verified_revision
            report.facts["time_series_model_revision"] = str(
                config.model.time_series_model_revision
            )
            report.facts["time_series_tokenizer_revision"] = str(
                config.model.time_series_tokenizer_revision
            )
        if not config.model.lora.enabled:
            report.errors.append("Production Kronos quant fine-tuning requires predictor LoRA")
        if on_runpod and not torch.cuda.is_available():
            report.errors.append("RunPod Kronos training requires a visible CUDA GPU")
    elif config.model.lora.enabled:
        report.errors.append("Mock backbone cannot enable Kronos LoRA")

    if torch.cuda.is_available():
        report.facts["gpu"] = torch.cuda.get_device_name(0)
        report.facts["gpu_memory_gib"] = (
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}"
        )
        if config.training.mixed_precision == "bf16" and not torch.cuda.is_bf16_supported():
            report.errors.append("Configured BF16 training is not supported by this GPU")
    elif config.model.time_series_backend == "kronos":
        report.warnings.append("Kronos will run on CPU outside RunPod and may be slow")

    sample_contract = PRODUCTION_STAGE_SAMPLE_CONTRACTS[config.training.stage]
    if config.data.train_fraction != sample_contract["train_fraction"]:
        report.errors.append(
            f"{config.training.stage} must use train_fraction="
            f"{sample_contract['train_fraction']}"
        )
    if (
        config.model.time_series_backend == "kronos"
        and config.data.max_samples != sample_contract["max_samples"]
    ):
        report.errors.append(
            f"{config.training.stage} must use max_samples="
            f"{sample_contract['max_samples']}"
        )
    return report
