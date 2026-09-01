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
    assert stage1.data.train_fraction == pytest.approx(0.15)
    assert stage2.training.stage == "stage2"
    assert stage2.data.train_fraction == pytest.approx(1.0)
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


def test_production_stages_cannot_hide_a_sample_cap(tmp_path: Path) -> None:
    path = ROOT / "configs/stage1_kronos_base_lora.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["data"]["max_samples"] = 100
    modified_path = tmp_path / "stage1_with_sample_cap.yaml"
    modified_path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="cannot cap max_samples"):
        ExperimentConfig.from_yaml(modified_path)


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

    assert contract["schema_version"] == TRAINING_RESUME_CONTRACT_VERSION == "5.0"
    assert contract["model_output_schema_version"] == MODEL_OUTPUT_SCHEMA_VERSION == "5.0"
    assert contract["training_implementation"] == implementation
    assert set(implementation["files"]) == set(TRAINING_IMPLEMENTATION_PATHS)
    assert len(implementation["sha256"]) == 64
    assert all(
        len(digest) == 64 and set(digest) <= set("0123456789abcdef")
        for digest in implementation["files"].values()
    )
