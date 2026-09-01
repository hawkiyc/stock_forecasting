"""Construct production or download-free quant-only model bundles."""

from __future__ import annotations

import subprocess
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
    """Require the configured Kronos source tree to be the exact approved commit."""

    if not source_root.is_dir():
        raise FileNotFoundError(f"Pinned Kronos source is missing: {source_root}")
    try:
        result = subprocess.run(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except FileNotFoundError as error:
        raise RuntimeError("git is required to verify the pinned Kronos source") from error
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("Timed out while verifying the pinned Kronos source") from error
    actual_revision = result.stdout.strip()
    if result.returncode != 0 or len(actual_revision) != 40:
        raise ValueError(f"Kronos source is not a readable Git checkout: {source_root}")
    if actual_revision != expected_revision:
        raise ValueError(
            "Kronos source revision mismatch: "
            f"expected {expected_revision}, found {actual_revision}"
        )
    return actual_revision


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
