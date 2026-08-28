"""Fail-closed preflight checks for offline quant-only training."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import torch

from stock_forecasting.config import ExperimentConfig
from stock_forecasting.data.manifest import (
    validate_dataset_preparation_contract,
    validate_training_dataset_manifest,
)
from stock_forecasting.factory import verify_kronos_source_revision
from stock_forecasting.training_paths import resolve_processed_dataset_path


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
        config.data.processed_path,
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
            processed = resolve_processed_dataset_path(config.data.processed_path)
            report.facts["processed_dataset"] = str(processed)
        except (FileNotFoundError, ValueError) as error:
            report.errors.append(str(error))
            processed = config.data.processed_path
        if config.data.require_ready_manifest:
            try:
                manifest = validate_training_dataset_manifest(
                    config.data.resolved_manifest_path,
                    profile=config.data.dataset_profile,
                    raw_path=config.data.raw_path,
                    processed_path=processed,
                )
            except (FileNotFoundError, ValueError) as error:
                report.errors.append(str(error))
            else:
                try:
                    preparation_spec = validate_dataset_preparation_contract(
                        manifest,
                        input_length=config.data.input_length,
                        h_start=config.data.h_start,
                        max_horizon=config.data.max_horizon,
                        alpha_horizons=config.data.alpha_horizons,
                        benchmark_mapping_path=config.data.benchmark_mapping_path,
                        sample_stride=config.data.sample_stride,
                        effective_embargo_trading_days=(
                            config.data.effective_embargo_trading_days
                        ),
                        forecast_horizon=config.data.forecast_horizon,
                        diagnostic_horizons=config.data.diagnostic_horizons,
                        stride=config.data.stride,
                        flat_volatility_multiplier=config.data.flat_volatility_multiplier,
                        max_abs_log_return=config.data.max_abs_log_return,
                        embargo_trading_days=config.data.embargo_trading_days,
                    )
                except ValueError as error:
                    report.errors.append(str(error))
                else:
                    report.facts["dataset_manifest"] = str(config.data.resolved_manifest_path)
                    report.facts["split_counts"] = str(manifest["split_counts"])
                    report.facts["raw_rows"] = str(manifest["artifacts"]["raw"]["row_count"])
                    report.facts["processed_rows"] = str(
                        manifest["artifacts"]["processed"]["row_count"]
                    )
                    report.facts["preparation_spec_sha256"] = str(
                        manifest["preparation_spec_sha256"]
                    )
                    report.facts["input_length"] = str(preparation_spec["window_size"])
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

    if config.training.stage == "stage1" and config.data.train_fraction != 0.15:
        report.errors.append("Stage 1 must use exactly 15% of the training split")
    if config.training.stage == "stage2" and config.data.train_fraction != 1.0:
        report.errors.append("Stage 2 must use the complete training split")
    return report
