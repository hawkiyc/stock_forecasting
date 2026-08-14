"""Configuration tests for the approved two-stage quant-only architecture."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from fin_ts_multimodal.config import DataConfig, ExperimentConfig
from fin_ts_multimodal.models import MODEL_OUTPUT_SCHEMA_VERSION
from fin_ts_multimodal.run_contract import (
    TRAINING_IMPLEMENTATION_PATHS,
    TRAINING_RESUME_CONTRACT_VERSION,
    training_implementation_contract,
    training_resume_contract,
    training_resume_contract_digest,
)

ROOT = Path(__file__).resolve().parents[1]


def test_stage_configs_share_one_model_architecture_and_start_fresh() -> None:
    stage1 = ExperimentConfig.from_yaml(ROOT / "configs/stage1_kronos_base_lora.yaml")
    stage2 = ExperimentConfig.from_yaml(ROOT / "configs/stage2_kronos_base_lora.yaml")

    assert stage1.training.stage == "stage1"
    assert stage1.data.train_fraction == pytest.approx(0.15)
    assert stage2.training.stage == "stage2"
    assert stage2.data.train_fraction == pytest.approx(1.0)
    assert stage1.model == stage2.model
    assert stage1.model.architecture_digest() == stage2.model.architecture_digest()
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
        processed_path=Path("windows.parquet"),
        dataset_profile=profile,
    )
    assert config.selected_datasets == selected


def test_quant_output_dimensions_are_fixed() -> None:
    path = ROOT / "configs/local_mock.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["model"]["alpha_quantiles"] = [0.05, 0.5, 0.95]

    with pytest.raises(ValidationError, match="alpha_quantiles are fixed"):
        ExperimentConfig.model_validate(payload)


def test_alpha_horizons_and_runpod_compatibility_sentinels_are_fixed() -> None:
    path = ROOT / "configs/local_mock.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["data"]["alpha_horizons"] = [5]

    with pytest.raises(ValidationError, match="trading days 3 through 14"):
        ExperimentConfig.model_validate(payload)

    config = ExperimentConfig.from_yaml(path)
    assert config.data.stride == 5
    assert config.data.embargo_trading_days == 5
    assert config.data.sample_stride == 1
    assert config.data.effective_embargo_trading_days == 14


def test_stage_fraction_contract_fails_closed() -> None:
    path = ROOT / "configs/local_mock.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["data"]["train_fraction"] = 1.0

    with pytest.raises(ValidationError, match="stage1 requires"):
        ExperimentConfig.model_validate(payload)


def test_production_stages_cannot_hide_a_sample_cap() -> None:
    path = ROOT / "configs/stage1_kronos_base_lora.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["data"]["max_samples"] = 100

    with pytest.raises(ValidationError, match="cannot cap max_samples"):
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
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["model"]["kronos_source_revision"] = "main"

    with pytest.raises(ValidationError, match="pinned 40-character"):
        ExperimentConfig.model_validate(payload)


@pytest.mark.parametrize(
    "field",
    ["time_series_model_revision", "time_series_tokenizer_revision"],
)
def test_kronos_production_config_requires_pinned_weight_revisions(field: str) -> None:
    path = ROOT / "configs/stage1_kronos_base_lora.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["model"][field] = "main"

    with pytest.raises(ValidationError, match="pinned 40-character"):
        ExperimentConfig.model_validate(payload)


def test_training_resume_contract_binds_bounded_implementation_sources() -> None:
    config = ExperimentConfig.from_yaml(ROOT / "configs/local_mock.yaml")
    implementation = training_implementation_contract()
    contract = training_resume_contract(config)

    assert contract["schema_version"] == TRAINING_RESUME_CONTRACT_VERSION == "4.0"
    assert contract["model_output_schema_version"] == MODEL_OUTPUT_SCHEMA_VERSION == "4.0"
    assert contract["training_implementation"] == implementation
    assert set(implementation["files"]) == set(TRAINING_IMPLEMENTATION_PATHS)
    assert len(implementation["sha256"]) == 64
    assert all(
        len(digest) == 64 and set(digest) <= set("0123456789abcdef")
        for digest in implementation["files"].values()
    )
