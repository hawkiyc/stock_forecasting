"""Configuration tests for the approved two-stage quant-only architecture."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from stock_forecasting.config import DataConfig, ExperimentConfig, KronosLoRAConfig
from stock_forecasting.models import MODEL_OUTPUT_SCHEMA_VERSION
from stock_forecasting.run_contract import (
    TRAINING_IMPLEMENTATION_PATHS,
    TRAINING_RESUME_CONTRACT_VERSION,
    training_implementation_contract,
    training_resume_contract,
    training_resume_contract_digest,
)

ROOT = Path(__file__).resolve().parents[1]


def test_config_import_is_cycle_safe_in_a_fresh_interpreter() -> None:
    environment = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from stock_forecasting.config import ExperimentConfig; "
                "from stock_forecasting.data import FinancialWindowDataset; "
                "assert ExperimentConfig.__name__ == 'ExperimentConfig'; "
                "assert FinancialWindowDataset.__name__ == 'FinancialWindowDataset'"
            ),
        ],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr


def test_lora_targets_normalize_without_recursive_assignment_validation() -> None:
    config = KronosLoRAConfig(target_modules=[" q_proj ", "v_proj "])

    assert config.target_modules == ["q_proj", "v_proj"]
    with pytest.raises(ValidationError, match="unique non-empty names"):
        config.target_modules = ["q_proj", " q_proj "]


def test_stage_configs_share_one_model_architecture_and_start_fresh() -> None:
    stage1 = ExperimentConfig.from_yaml(ROOT / "configs/stage1_kronos_base_lora.yaml")
    stage2 = ExperimentConfig.from_yaml(ROOT / "configs/stage2_kronos_base_lora.yaml")

    assert stage1.training.stage == "stage1"
    assert stage1.data.train_fraction == pytest.approx(0.05)
    assert stage1.data.max_samples == 500_000
    assert stage1.training.epochs == 2
    assert stage1.training.early_stopping_start_epoch == 2
    assert stage2.training.stage == "stage2"
    assert stage2.data.train_fraction == pytest.approx(1.0)
    assert stage2.data.max_samples is None
    assert stage2.training.epochs == 5
    assert stage2.training.early_stopping_start_epoch == 1
    assert stage1.training.evaluations_per_epoch == 5
    assert stage2.training.evaluations_per_epoch == 5
    assert stage1.training.batch_size == "auto"
    assert stage2.training.batch_size == "auto"
    assert stage1.training.evaluation_batch_size == "auto"
    assert stage2.training.evaluation_batch_size == "auto"
    assert stage1.training.gradient_accumulation_steps == "auto"
    assert stage2.training.gradient_accumulation_steps == "auto"
    assert stage1.training.num_workers == "auto"
    assert stage2.training.num_workers == "auto"
    assert stage1.training.target_effective_batch_size == 256
    assert stage2.training.target_effective_batch_size == 256
    assert stage1.training.dataloader_max_prefetch_factor == 16
    assert stage2.training.dataloader_max_prefetch_factor == 16
    assert stage1.training.loss_log_points_per_epoch == 250
    assert stage2.training.loss_log_points_per_epoch == 250
    assert stage1.training.checkpoint_save_top_k == 5
    assert stage2.training.checkpoint_save_top_k == 5
    assert stage1.training.early_stopping_enabled is True
    assert stage2.training.early_stopping_enabled is True
    assert stage1.model == stage2.model
    assert stage1.model_architecture_digest() == stage2.model_architecture_digest()
    assert stage1.training.resume_checkpoint is None
    assert stage2.training.resume_checkpoint is None
    assert training_resume_contract_digest(stage1) != training_resume_contract_digest(stage2)


@pytest.mark.parametrize(
    ("profile", "selected"),
    [
        ("tw_only", ["tpex_official", "twse_official"]),
        ("us_only_eodhd", ["eodhd_us"]),
        ("us_tw_eodhd", ["eodhd_us", "tpex_official", "twse_official"]),
        ("us_tw_massive", ["massive_us", "tpex_official", "twse_official"]),
    ],
)
def test_dataset_profiles_report_exact_selected_sources(
    profile: str,
    selected: list[str],
) -> None:
    config = DataConfig(
        raw_path=Path("raw.parquet"),
        bar_store_path=Path("prepared/bar-store"),
        dataset_profile=profile,
    )
    assert config.selected_datasets == selected


def test_quant_output_dimensions_are_fixed() -> None:
    path = ROOT / "configs/local_mock.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["model"]["alpha_quantiles"] = [0.05, 0.5, 0.95]

    with pytest.raises(ValidationError, match="alpha_quantiles are fixed"):
        ExperimentConfig.model_validate(payload)


@pytest.mark.parametrize(
    ("h_start", "expected_horizons"),
    [
        (1, list(range(1, 15))),
        (2, list(range(2, 15))),
        (3, list(range(3, 15))),
    ],
)
def test_h_start_derives_contiguous_horizons_and_keeps_runpod_sentinels(
    h_start: int,
    expected_horizons: list[int],
) -> None:
    path = ROOT / "configs/local_mock.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["data"]["h_start"] = h_start

    config = ExperimentConfig.model_validate(payload)
    assert config.data.h_start == h_start
    assert config.data.alpha_horizons == expected_horizons
    assert config.data.max_horizon == 14
    assert config.data.stride == 5
    assert config.data.embargo_trading_days == 5
    assert config.data.sample_stride == 1
    assert config.data.effective_embargo_trading_days == 14


@pytest.mark.parametrize("h_start", [0, 4, True])
def test_h_start_rejects_values_outside_one_through_three(h_start: int | bool) -> None:
    path = ROOT / "configs/local_mock.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["data"]["h_start"] = h_start

    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(payload)


def test_h_start_changes_model_and_training_contract_identity() -> None:
    first_day = ExperimentConfig.from_yaml(ROOT / "configs/local_mock.yaml")
    third_day = ExperimentConfig.from_yaml(ROOT / "configs/local_mock.yaml")
    first_day.data.h_start = 1

    assert first_day.model == third_day.model
    assert first_day.model.architecture_digest() == third_day.model.architecture_digest()
    assert first_day.model_architecture_digest() != third_day.model_architecture_digest()
    assert training_resume_contract_digest(first_day) != training_resume_contract_digest(
        third_day
    )


def test_stage_fraction_contract_fails_closed() -> None:
    path = ROOT / "configs/local_mock.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["data"]["train_fraction"] = 1.0

    with pytest.raises(ValidationError, match="stage1 requires"):
        ExperimentConfig.model_validate(payload)


@pytest.mark.parametrize(
    ("config_name", "replacement", "message"),
    [
        ("stage1_kronos_base_lora.yaml", None, "data.max_samples=500000"),
        ("stage1_kronos_base_lora.yaml", 499_999, "data.max_samples=500000"),
        ("stage2_kronos_base_lora.yaml", 500_000, "data.max_samples=None"),
    ],
)
def test_production_stage_sample_caps_fail_closed(
    tmp_path: Path,
    config_name: str,
    replacement: int | None,
    message: str,
) -> None:
    path = ROOT / "configs" / config_name
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["data"]["max_samples"] = replacement
    modified_path = tmp_path / config_name
    modified_path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match=message):
        ExperimentConfig.from_yaml(modified_path)


@pytest.mark.parametrize(
    "obsolete_field",
    [
        "max_steps",
        "evaluate_every_steps",
        "checkpoint_every_steps",
        "log_every_steps",
    ],
)
def test_training_config_rejects_obsolete_step_caps_and_cadence(
    obsolete_field: str,
) -> None:
    path = ROOT / "configs/stage1_kronos_base_lora.yaml"
    payload = ExperimentConfig.from_yaml(path).as_dict()
    payload["training"][obsolete_field] = 200

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ExperimentConfig.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("batch_size", 0, "positive integers"),
        ("num_workers", -1, "non-negative integer"),
        ("auto_evaluation_batch_max_size", 12, "powers of two"),
        (
            "auto_batch_max_size",
            512,
            "cannot exceed target_effective_batch_size",
        ),
    ],
)
def test_automatic_training_batch_contract_fails_closed(
    field: str,
    value: object,
    message: str,
) -> None:
    payload = ExperimentConfig.from_yaml(ROOT / "configs/local_mock.yaml").as_dict()
    payload["training"][field] = value

    with pytest.raises(ValidationError, match=message):
        ExperimentConfig.model_validate(payload)


def test_automatic_evaluation_batch_can_cover_the_training_search_range() -> None:
    payload = ExperimentConfig.from_yaml(ROOT / "configs/local_mock.yaml").as_dict()
    payload["training"]["batch_size"] = "auto"
    payload["training"]["gradient_accumulation_steps"] = "auto"
    payload["training"]["target_effective_batch_size"] = 16
    payload["training"]["auto_batch_max_size"] = 16
    payload["training"]["auto_evaluation_batch_max_size"] = 8

    with pytest.raises(ValidationError, match="maximum training batch size"):
        ExperimentConfig.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("epochs", 1, "training.epochs=2"),
        ("evaluations_per_epoch", 4, "five validations per epoch"),
        ("checkpoint_save_top_k", 4, "best five checkpoints"),
        (
            "checkpoint_monitor",
            "primary_5d/median_correlation",
            "normalized pinball loss",
        ),
        ("checkpoint_mode", "max", "normalized pinball loss"),
        ("early_stopping_enabled", False, "requires validation-loss early stopping"),
        (
            "early_stopping_patience_evaluations",
            4,
            "five consecutive non-improving validations",
        ),
        (
            "early_stopping_min_delta",
            0.001,
            "five consecutive non-improving validations",
        ),
        ("early_stopping_start_epoch", 1, "early_stopping_start_epoch=2"),
    ],
)
def test_stage1_training_control_contract_fails_closed(
    field: str,
    value: object,
    message: str,
) -> None:
    path = ROOT / "configs/stage1_kronos_base_lora.yaml"
    payload = ExperimentConfig.from_yaml(path).as_dict()
    payload["training"][field] = value
    if field == "epochs":
        payload["training"]["early_stopping_start_epoch"] = 1

    with pytest.raises(ValidationError, match=message):
        ExperimentConfig.model_validate(payload)


def test_model_config_has_no_language_or_fact_branch() -> None:
    config = ExperimentConfig.from_yaml(ROOT / "configs/stage1_kronos_base_lora.yaml")
    serialized = config.model.model_dump()
    forbidden = {"llm", "language", "text", "fact", "tokenizer_chat"}

    assert not any(token in key.lower() for key in serialized for token in forbidden)
    training = config.training.model_dump()
    assert not ({"classification_loss_weight", "quantile_loss_weight"} & set(training))
    assert config.model.alpha_quantiles == [0.1, 0.5, 0.9]


def test_kronos_production_config_requires_a_pinned_source_revision() -> None:
    path = ROOT / "configs/stage1_kronos_base_lora.yaml"
    payload = ExperimentConfig.from_yaml(path).as_dict()
    payload["model"]["kronos_source_revision"] = "main"

    with pytest.raises(ValidationError, match="pinned 40-character"):
        ExperimentConfig.model_validate(payload)


@pytest.mark.parametrize(
    "field",
    ["time_series_model_revision", "time_series_tokenizer_revision"],
)
def test_kronos_production_config_requires_pinned_weight_revisions(field: str) -> None:
    path = ROOT / "configs/stage1_kronos_base_lora.yaml"
    payload = ExperimentConfig.from_yaml(path).as_dict()
    payload["model"][field] = "main"

    with pytest.raises(ValidationError, match="pinned 40-character"):
        ExperimentConfig.model_validate(payload)


def test_training_resume_contract_binds_bounded_implementation_sources() -> None:
    config = ExperimentConfig.from_yaml(ROOT / "configs/local_mock.yaml")
    implementation = training_implementation_contract()
    contract = training_resume_contract(config)

    assert contract["schema_version"] == TRAINING_RESUME_CONTRACT_VERSION == "6.0"
    assert contract["model_output_schema_version"] == MODEL_OUTPUT_SCHEMA_VERSION == "5.0"
    assert contract["training_implementation"] == implementation
    assert set(implementation["files"]) == set(TRAINING_IMPLEMENTATION_PATHS)
    assert len(implementation["sha256"]) == 64
    assert all(
        len(digest) == 64 and set(digest) <= set("0123456789abcdef")
        for digest in implementation["files"].values()
    )
