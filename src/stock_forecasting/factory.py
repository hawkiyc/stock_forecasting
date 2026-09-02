"""Construct production or download-free quant-only model bundles."""

from __future__ import annotations

import hashlib
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from stock_forecasting.config import ExperimentConfig
from stock_forecasting.models import (
    CausalPerceiverResampler,
    DeterministicTimeSeriesBackbone,
    GatedBenchmarkConditioner,
    KronosBackbone,
    MultiHorizonAlphaHead,
    QuantForecastModel,
    lora_parameter_names,
)

PINNED_KRONOS_SOURCE_REVISION = "67b630e67f6a18c9e9be918d9b4337c960db1e9a"
PINNED_KRONOS_SOURCE_FILES = {
    "LICENSE": "acb2d194d378204e5f2be4dcd24d39ecac437903620c790c3315a96dab388fdc",
    "model/__init__.py": "7bada2fa83c8c3df045caf06a92050a3ab631964fee8ffadcad218f81c3e696e",
    "model/kronos.py": "107abe371db5d17f80bb1342ae1c381935d53db65e217778f7afb917b39a4c2f",
    "model/module.py": "8b0c1d535b07e667295ee2daba6e04474dfefef898d0581ba8105d3143d7f312",
}


@dataclass(frozen=True)
class ModelBundle:
    model: QuantForecastModel
    time_series_model_id: str
    time_series_tokenizer_id: str
    lora_module_names: tuple[str, ...]
    lora_parameter_names: tuple[str, ...]


def _dtype(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp32": torch.float32}[name]


def verify_kronos_source_revision(source_root: Path, expected_revision: str) -> str:
    """Verify the bundled Kronos source without invoking a remote Git client."""

    if not source_root.is_dir():
        raise FileNotFoundError(f"Bundled Kronos source is missing: {source_root}")
    if source_root.is_symlink():
        raise ValueError(f"Bundled Kronos source must not be a symlink: {source_root}")
    if expected_revision != PINNED_KRONOS_SOURCE_REVISION:
        raise ValueError(
            "Kronos source revision mismatch: "
            f"config requests {expected_revision}, bundled source is "
            f"{PINNED_KRONOS_SOURCE_REVISION}"
        )
    resolved_root = source_root.resolve(strict=True)
    for relative_path, expected_sha256 in PINNED_KRONOS_SOURCE_FILES.items():
        source_path = source_root / relative_path
        if not source_path.is_file() or source_path.is_symlink():
            raise FileNotFoundError(f"Bundled Kronos source file is missing: {source_path}")
        resolved_path = source_path.resolve(strict=True)
        if not resolved_path.is_relative_to(resolved_root):
            raise ValueError(f"Bundled Kronos source escapes its root: {source_path}")
        actual_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
        if actual_sha256 != expected_sha256:
            raise ValueError(
                "Bundled Kronos source digest mismatch: "
                f"{relative_path} expected {expected_sha256}, found {actual_sha256}"
            )
    return PINNED_KRONOS_SOURCE_REVISION


def _prepare_kronos_import(source_root: Path, expected_revision: str) -> None:
    verify_kronos_source_revision(source_root, expected_revision)
    root_text = str(source_root.resolve())
    if root_text not in sys.path:
        sys.path.insert(0, root_text)


def _promote_trainable_parameters_to_fp32(model: torch.nn.Module) -> None:
    """Keep optimizer master parameters in FP32 while frozen weights use compact storage."""

    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad and parameter.is_floating_point():
                parameter.data = parameter.data.to(dtype=torch.float32)


def build_model_bundle(
    config: ExperimentConfig,
    device: torch.device,
    *,
    robust_scales: Sequence[float] | None = None,
) -> ModelBundle:
    """Build frozen tokenizer/base weights, predictor LoRA, resampler, and quant head."""

    lora_modules: tuple[str, ...] = ()
    if config.model.time_series_backend == "mock":
        backbone = DeterministicTimeSeriesBackbone(
            input_dim=5,
            hidden_size=config.model.encoder_dim,
            max_context=config.data.input_length,
        )
    else:
        revision = config.model.kronos_source_revision
        assert revision is not None
        _prepare_kronos_import(config.model.kronos_source_root, revision)
        kronos = KronosBackbone.from_pretrained(
            model_name_or_path=config.model.time_series_model_id,
            tokenizer_name_or_path=config.model.time_series_tokenizer_id,
            model_revision=config.model.time_series_model_revision,
            tokenizer_revision=config.model.time_series_tokenizer_revision,
            local_files_only=config.model.local_files_only,
            max_context=config.data.input_length,
        )
        if config.model.lora.enabled:
            lora_modules = kronos.enable_lora(
                target_modules=config.model.lora.target_modules,
                rank=config.model.lora.rank,
                alpha=config.model.lora.alpha,
                dropout=config.model.lora.dropout,
            )
        if config.model.gradient_checkpointing:
            checkpointing_enable = getattr(kronos.model, "gradient_checkpointing_enable", None)
            if not callable(checkpointing_enable):
                raise ValueError(
                    "Configured Kronos implementation does not support gradient checkpointing"
                )
            checkpointing_enable()
        backbone = kronos

    resampler = CausalPerceiverResampler(
        input_dim=int(backbone.hidden_size),
        output_dim=config.model.encoder_dim,
        num_soft_tokens=config.model.latent_tokens,
        latent_dim=config.model.encoder_dim,
        depth=config.model.resampler_layers,
        num_heads=config.model.resampler_heads,
        projector_hidden_multiplier=config.model.projector_hidden_multiplier,
        dropout=config.model.resampler_dropout,
    )
    benchmark_conditioner = GatedBenchmarkConditioner(
        config.model.encoder_dim,
        num_heads=config.model.benchmark_conditioner_heads,
        dropout=config.model.benchmark_conditioner_dropout,
    )
    resolved_scales = (
        tuple(float(value) for value in robust_scales)
        if robust_scales is not None
        else tuple(1.0 for _ in config.data.alpha_horizons)
    )
    if len(resolved_scales) != len(config.data.alpha_horizons):
        raise ValueError("Runtime robust scales must match the configured alpha horizons")
    alpha_head = MultiHorizonAlphaHead(
        config.model.encoder_dim,
        horizons=tuple(config.data.alpha_horizons),
        hidden_dim=config.model.alpha_head_hidden_dim,
        quantiles=tuple(config.model.alpha_quantiles),
        robust_scales=resolved_scales,
        dropout=config.model.alpha_head_dropout,
    )
    model = QuantForecastModel(
        backbone,
        resampler,
        benchmark_conditioner,
        alpha_head,
    )
    target_dtype = _dtype(config.model.dtype) if device.type == "cuda" else torch.float32
    model.to(device=device, dtype=target_dtype)
    _promote_trainable_parameters_to_fp32(model)
    return ModelBundle(
        model=model,
        time_series_model_id=config.model.time_series_model_id,
        time_series_tokenizer_id=config.model.time_series_tokenizer_id,
        lora_module_names=lora_modules,
        lora_parameter_names=lora_parameter_names(model),
    )
