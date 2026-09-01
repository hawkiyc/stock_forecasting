"""Typed configuration for quant-only financial time-series experiments."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Literal, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from stock_forecasting.data.horizons import (
    DEFAULT_H_START,
    MAX_ALPHA_HORIZON,
    alpha_horizons_from_start,
)

_ENV_DEFAULT_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*):-([^}]*)}")
DatasetProfile = Literal[
    "tw_only",
    "us_only_eodhd",
    "us_tw_eodhd",
    "us_tw_massive",
]
TrainingStage = Literal["stage1", "stage2"]


def _expand_environment(value: Any) -> Any:
    if isinstance(value, str):
        expanded = _ENV_DEFAULT_PATTERN.sub(
            lambda match: os.environ.get(match.group(1), match.group(2)), value
        )
        return os.path.expandvars(os.path.expanduser(expanded))
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_environment(item) for key, item in value.items()}
    return value


class StrictModel(BaseModel):
    """Reject unknown configuration keys to prevent silent experiment drift."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class DataConfig(StrictModel):
    """Offline paired-OHLCV dataset and next-open alpha contract."""

    raw_path: Path
    bar_store_path: Path
    manifest_path: Path | None = None
    dataset_profile: DatasetProfile = "tw_only"
    input_length: int = Field(default=128, ge=32, le=512)
    h_start: int = Field(default=DEFAULT_H_START, ge=1, le=3)
    benchmark_mapping_path: Path | None = None
    # These two fields are retained only because the stable RunPod readiness
    # scripts validate them. They do not define the new model target.
    forecast_horizon: int = 5
    diagnostic_horizons: list[int] = Field(default_factory=lambda: [1, 20])
    # Stable RunPod CPU-prepare sentinels. Effective values are separate so the
    # established readiness interface and lifecycle paths remain compatible.
    stride: int = 5
    sample_stride: int = Field(default=1, ge=1)
    train_fraction: float = Field(default=1.0, gt=0.0, le=1.0)
    flat_volatility_multiplier: float = Field(default=0.25, gt=0.0)
    max_abs_log_return: float = Field(default=0.5, gt=0.0)
    embargo_trading_days: int = 5
    effective_embargo_trading_days: int = Field(default=14, ge=14)
    train_end: str | None = None
    validation_end: str | None = None
    test_end: str | None = None
    max_samples: int | None = Field(default=None, ge=1)
    label_scale_calibration_samples: int = Field(default=50_000, ge=4)
    require_ready_manifest: bool = True

    @field_validator("h_start", mode="before")
    @classmethod
    def reject_boolean_h_start(cls, value: Any) -> Any:
        if isinstance(value, bool):
            raise ValueError("h_start must be 1, 2, or 3")
        return value

    @model_validator(mode="after")
    def validate_horizons(self) -> DataConfig:
        if self.forecast_horizon != 5:
            raise ValueError("RunPod readiness compatibility requires forecast_horizon=5")
        if self.diagnostic_horizons != [1, 20]:
            raise ValueError("RunPod readiness compatibility requires diagnostic_horizons=[1, 20]")
        if self.stride != 5 or self.embargo_trading_days != 5:
            raise ValueError("Stable RunPod readiness requires stride=5 and embargo=5 sentinels")
        if self.sample_stride != 1:
            raise ValueError("Conditional alpha windows require sample_stride=1")
        return self

    @property
    def alpha_horizons(self) -> list[int]:
        """Return the configured contiguous holding-day forecast horizons."""

        return list(alpha_horizons_from_start(self.h_start))

    @property
    def max_horizon(self) -> int:
        """Return the fixed maximum holding-day forecast horizon."""

        return MAX_ALPHA_HORIZON

    @property
    def selected_datasets(self) -> list[str]:
        """Return stable human-readable dataset identifiers for reports."""

        profiles = {
            "tw_only": ["twse_official", "tpex_official"],
            "us_only_eodhd": ["eodhd_us"],
            "us_tw_eodhd": ["eodhd_us", "twse_official", "tpex_official"],
            "us_tw_massive": ["massive_us", "twse_official", "tpex_official"],
        }
        return sorted(profiles[self.dataset_profile])

    @property
    def resolved_manifest_path(self) -> Path:
        if self.manifest_path is not None:
            return self.manifest_path
        base = self.raw_path if self.raw_path.is_dir() else self.raw_path.parent
        return base / "dataset-manifest.json"


class KronosLoRAConfig(StrictModel):
    """Low-rank adaptation applied only to the Kronos predictor."""

    enabled: bool = True
    rank: int = Field(default=8, ge=1)
    alpha: float = Field(default=16.0, gt=0.0)
    dropout: float = Field(default=0.05, ge=0.0, lt=1.0)
    target_modules: list[str] = Field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "out_proj",
            "w1",
            "w2",
            "w3",
        ],
        min_length=1,
    )

    @field_validator("target_modules")
    @classmethod
    def validate_targets(cls, values: list[str]) -> list[str]:
        targets = [target.strip() for target in values]
        if any(not target for target in targets) or len(set(targets)) != len(targets):
            raise ValueError("lora.target_modules must contain unique non-empty names")
        return targets


class ModelConfig(StrictModel):
    """Shared Kronos encoder, benchmark conditioner, and alpha quantile head."""

    time_series_backend: Literal["kronos", "mock"] = "kronos"
    time_series_model_id: str = "NeoQuasar/Kronos-base"
    time_series_tokenizer_id: str = "NeoQuasar/Kronos-Tokenizer-base"
    time_series_model_revision: str | None = None
    time_series_tokenizer_revision: str | None = None
    kronos_source_root: Path = Path("/runpod-volume/third_party/Kronos")
    kronos_source_revision: str | None = None
    local_files_only: bool = True
    dtype: Literal["bf16", "fp32"] = "bf16"
    gradient_checkpointing: bool = False
    encoder_dim: int = Field(default=512, ge=64)
    latent_tokens: int = Field(default=32, ge=1)
    resampler_layers: int = Field(default=2, ge=1)
    resampler_heads: int = Field(default=8, ge=1)
    projector_hidden_multiplier: int = Field(default=2, ge=1)
    resampler_dropout: float = Field(default=0.05, ge=0.0, lt=1.0)
    benchmark_conditioner_heads: int = Field(default=8, ge=1)
    benchmark_conditioner_dropout: float = Field(default=0.05, ge=0.0, lt=1.0)
    alpha_head_hidden_dim: int = Field(default=512, ge=32)
    alpha_head_dropout: float = Field(default=0.05, ge=0.0, lt=1.0)
    alpha_quantiles: list[float] = Field(default_factory=lambda: [0.1, 0.5, 0.9])
    postprocess_alpha_threshold: float = Field(default=0.0, ge=0.0)
    lora: KronosLoRAConfig = Field(default_factory=KronosLoRAConfig)

    @model_validator(mode="after")
    def validate_model_contract(self) -> ModelConfig:
        if self.encoder_dim % self.resampler_heads != 0:
            raise ValueError("encoder_dim must be divisible by resampler_heads")
        if self.encoder_dim % self.benchmark_conditioner_heads != 0:
            raise ValueError("encoder_dim must be divisible by benchmark_conditioner_heads")
        if self.alpha_quantiles != [0.1, 0.5, 0.9]:
            raise ValueError("alpha_quantiles are fixed at [0.1, 0.5, 0.9]")
        if self.time_series_backend == "mock" and self.lora.enabled:
            raise ValueError("The mock backbone does not expose Kronos LoRA targets")
        if self.time_series_backend == "kronos":
            if "Kronos-base" not in self.time_series_model_id:
                raise ValueError("The approved production architecture requires Kronos-base")
            revisions = {
                "time_series_model_revision": self.time_series_model_revision,
                "time_series_tokenizer_revision": self.time_series_tokenizer_revision,
                "kronos_source_revision": self.kronos_source_revision,
            }
            invalid_revisions = [
                name
                for name, revision in revisions.items()
                if revision is None or re.fullmatch(r"[0-9a-f]{40}", revision) is None
            ]
            if invalid_revisions:
                raise ValueError(
                    "Production Kronos requires pinned 40-character lowercase revisions for: "
                    + ", ".join(invalid_revisions)
                )
        elif any(
            revision is not None
            for revision in (
                self.time_series_model_revision,
                self.time_series_tokenizer_revision,
                self.kronos_source_revision,
            )
        ):
            raise ValueError("Mock experiments must not declare Kronos or model revisions")
        return self

    def architecture_digest(self) -> str:
        """Hash only model semantics so Stage 1 and Stage 2 can be compared."""

        payload = json.dumps(
            json.loads(self.model_dump_json()),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


class TrainingConfig(StrictModel):
    """Single quant-finetuning procedure used by both data-volume stages."""

    stage: TrainingStage
    seed: int = 42
    epochs: int = Field(default=1, ge=1)
    max_steps: int | None = Field(default=None, ge=1)
    batch_size: int = Field(default=8, ge=1)
    evaluation_batch_size: int = Field(default=16, ge=1)
    gradient_accumulation_steps: int = Field(default=1, ge=1)
    learning_rate: float = Field(default=1e-4, gt=0.0)
    lora_learning_rate: float = Field(default=1e-5, gt=0.0)
    weight_decay: float = Field(default=0.01, ge=0.0)
    warmup_ratio: float = Field(default=0.03, ge=0.0, lt=1.0)
    max_grad_norm: float = Field(default=1.0, gt=0.0)
    log_every_steps: Literal[1] = 1
    evaluate_every_steps: int = Field(default=100, ge=1)
    checkpoint_every_steps: int = Field(default=100, ge=1)
    checkpoint_save_top_k: int = Field(default=3, ge=1, le=10)
    checkpoint_monitor: str = "primary_5d/selection_score"
    checkpoint_mode: Literal["min", "max"] = "min"
    output_root: Path = Path("/runpod-volume/savedModel")
    resume_checkpoint: Path | None = None
    mixed_precision: Literal["no", "bf16"] = "bf16"
    num_workers: int = Field(default=2, ge=0)
    evaluation_max_samples: int = Field(default=20_000, ge=32)

    @model_validator(mode="after")
    def validate_training_contract(self) -> TrainingConfig:
        if not self.checkpoint_monitor.startswith("primary_5d/"):
            raise ValueError("checkpoint_monitor must reference a validation primary_5d metric")
        if self.lora_learning_rate > self.learning_rate:
            raise ValueError("lora_learning_rate cannot exceed learning_rate")
        return self


class WandbConfig(StrictModel):
    enabled: bool = True
    project: str = "fin-ts-quant"
    entity: str | None = None
    group: str | None = None
    name: str | None = None
    tags: list[str] = Field(default_factory=list)
    # W&B appends its own ``wandb/`` directory below this SDK root.
    directory: Path = Path("/runpod-volume")
    mode: Literal["online", "offline", "disabled"] = "online"
    allow_offline_fallback: bool = True
    log_model_artifact: bool = True

    @model_validator(mode="after")
    def validate_directory_contract(self) -> WandbConfig:
        if self.directory.name == "wandb":
            raise ValueError("wandb.directory is the SDK root and must not end with /wandb")
        return self


class ValidationConfig(StrictModel):
    enabled: bool = True
    auto_run_after_training: bool = True
    output_root: Path = Path("/runpod-volume/evaluations")
    models: list[
        Literal[
            "always_buy",
            "zero_return",
            "momentum_5d",
            "reversal_5d",
            "ma_crossover",
            "rsi",
            "macd",
            "volatility_scaled",
            "gbdt",
            "gru",
            "dlinear",
            "patchtst",
            "kronos_full",
        ]
    ] = Field(
        default_factory=lambda: [
            "always_buy",
            "zero_return",
            "momentum_5d",
            "reversal_5d",
            "ma_crossover",
            "rsi",
            "macd",
            "volatility_scaled",
            "gbdt",
            "gru",
            "dlinear",
            "patchtst",
            "kronos_full",
        ]
    )
    seeds: list[int] = Field(default_factory=lambda: [42])
    neural_epochs: int = Field(default=30, ge=1)
    neural_patience: int = Field(default=5, ge=1)
    neural_batch_size: int = Field(default=64, ge=1)
    neural_learning_rate: float = Field(default=1e-3, gt=0.0)
    recompute_full_model: bool = False
    resume_completed_models: bool = True
    baseline_max_samples_per_split: int = Field(default=20_000, ge=32)

    @model_validator(mode="after")
    def validate_benchmark_plan(self) -> ValidationConfig:
        if not self.models or len(set(self.models)) != len(self.models):
            raise ValueError("validation.models must be non-empty and unique")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("validation.seeds must be non-empty and unique")
        return self


class RuntimeConfig(StrictModel):
    max_runtime_seconds: int = Field(default=21600, ge=60)
    auto_terminate_pod: bool = True
    termination_retry_seconds: int = Field(default=15, ge=1)
    termination_max_attempts: int = Field(default=20, ge=1)
    log_root: Path = Path("/runpod-volume/logs")


class ExperimentConfig(StrictModel):
    experiment_name: str
    description: str = ""
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    wandb: WandbConfig
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    runtime: RuntimeConfig

    @model_validator(mode="after")
    def validate_stage_contract(self) -> ExperimentConfig:
        expected_fraction = 0.15 if self.training.stage == "stage1" else 1.0
        if abs(self.data.train_fraction - expected_fraction) > 1e-12:
            raise ValueError(
                f"{self.training.stage} requires data.train_fraction={expected_fraction}"
            )
        if self.model.time_series_backend == "kronos" and self.data.max_samples is not None:
            raise ValueError(
                "Production Stage 1/2 cannot cap max_samples; use the exact 15% or full train split"
            )
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as stream:
            payload = yaml.safe_load(stream)
        if not isinstance(payload, dict):
            raise ValueError(f"Configuration must be a mapping: {config_path}")
        return cls.model_validate(_expand_environment(payload))

    def as_dict(self) -> dict[str, Any]:
        return cast(dict[str, Any], json.loads(self.model_dump_json()))

    def model_architecture_digest(self) -> str:
        """Hash model parameters together with the variable output horizon contract."""

        payload = {
            "model_sha256": self.model.architecture_digest(),
            "alpha_horizons": self.data.alpha_horizons,
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def save_resolved(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(self.as_dict(), stream, sort_keys=False, allow_unicode=True)
