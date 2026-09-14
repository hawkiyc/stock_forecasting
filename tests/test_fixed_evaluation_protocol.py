"""Fixed-date membership, exact label guards, and paired holdout scoring."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from stock_forecasting.config import ExperimentConfig
from stock_forecasting.data.bar_store import _split_boundaries, _split_codes, build_symbol_bar_store
from stock_forecasting.data.dataset import FinancialBatchCollator, LazyFinancialWindowDataset
from stock_forecasting.data.manifest import artifact_metadata
from stock_forecasting.dataset_identity import (
    DEFAULT_DATASET_STORAGE_PREPARATION,
    FIXED_EVALUATION_SPLIT,
    storage_preparation_spec,
    validate_fixed_split_audit,
    validated_fixed_split,
)
from stock_forecasting.evaluation_protocol import (
    daily_normalized_pinball,
    evaluation_sampler,
    paired_block_comparison,
    sample_membership,
)
from stock_forecasting.factory import build_model_bundle
from stock_forecasting.run_contract import (
    _canonical_payload_digest,
    readonly_checkpoint_contract_digest,
    training_resume_contract,
)
from stock_forecasting.scale_calibration import resolve_scale_feature_statistics

ROOT = Path(__file__).resolve().parents[1]


def test_production_stages_share_dates_calibration_and_full_horizons() -> None:
    configs = [
        ExperimentConfig.from_yaml(ROOT / "configs" / f"{stage}_kronos_base_lora.yaml")
        for stage in ("stage1", "stage2")
    ]
    for config in configs:
        assert config.data.fixed_split == FIXED_EVALUATION_SPLIT
        assert config.data.alpha_horizons == list(range(1, 15))
        assert config.data.label_scale_calibration_samples == 50_000
        assert config.data.calibration_seed == 59
        assert config.model.feature_mode == "combined"
        assert config.model.alpha_head_fp32
    assert list(evaluation_sampler(100_000, configs[0], "test")) == list(
        evaluation_sampler(100_000, configs[1], "test")
    )


def test_fixed_split_changes_storage_identity_but_legacy_is_unchanged() -> None:
    legacy = storage_preparation_spec(DEFAULT_DATASET_STORAGE_PREPARATION)
    assert legacy == DEFAULT_DATASET_STORAGE_PREPARATION
    fixed = storage_preparation_spec({**legacy, "fixed_split": FIXED_EVALUATION_SPLIT})
    assert fixed != legacy and fixed["fixed_split"] == FIXED_EVALUATION_SPLIT
    for payload in ({}, {**FIXED_EVALUATION_SPLIT, "train_end": "2026-01-01"}):
        with pytest.raises(ValueError):
            validated_fixed_split(payload)


def test_bar_store_enforces_actual_label_end_and_retains_historical_context(
    tmp_path: Path, market_frame: pd.DataFrame,
) -> None:
    frame = market_frame.copy()
    days = sorted(frame["timestamp"].unique())
    replacement = pd.bdate_range("2025-01-01", periods=len(days), tz="UTC")
    mapping = dict(zip(days, replacement, strict=True))
    frame["timestamp"] = frame["timestamp"].map(mapping)
    raw = tmp_path / "raw.parquet"
    frame.to_parquet(raw, index=False)
    result = build_symbol_bar_store(
        raw_path=raw, output_root=tmp_path / "bar-store",
        download_manifest={
            "artifacts": {"raw": artifact_metadata(raw, root=tmp_path, row_count=len(frame))},
        },
        window_size=64, bucket_count=2, batch_rows=200, fixed_split=FIXED_EVALUATION_SPLIT,
    )
    audit = result.split_audit
    validate_fixed_split_audit(audit, FIXED_EVALUATION_SPLIT, minimum_evaluation_dates=80)
    assert audit["additional_purge_embargo_applied"] is False
    for split, lower, upper in (
        ("train", None, "2025-06-01"),
        ("validation", "2025-06-01", "2025-12-01"),
        ("test", "2025-12-01", "2026-06-01"),
    ):
        dataset = LazyFinancialWindowDataset(tmp_path / "bar-store", split=split, window_size=64)
        records = [dataset.record_at(index) for index in (0, len(dataset) - 1)]
        for record in records:
            assert pd.Timestamp(record["cutoff_at"]) < pd.Timestamp(upper, tz="UTC")
            label_end = max(pd.Timestamp(value) for value in record["label"]["end_at"].values())
            assert label_end < pd.Timestamp(upper, tz="UTC")
            if lower is not None:
                assert pd.Timestamp(record["cutoff_at"]) >= pd.Timestamp(lower, tz="UTC")
        if split == "validation":
            assert pd.Timestamp(records[0]["window_start_at"]) < pd.Timestamp(lower, tz="UTC")
    corrupted = copy.deepcopy(audit)
    first_market = next(iter(corrupted["dates_by_market"]))
    corrupted["dates_by_market"][first_market]["test"]["unique_cutoff_count"] += 1
    with pytest.raises(ValueError, match="date count"):
        validate_fixed_split_audit(corrupted, FIXED_EVALUATION_SPLIT)
    with pytest.raises(ValueError, match="effective dates"):
        validate_fixed_split_audit(audit, FIXED_EVALUATION_SPLIT, minimum_evaluation_dates=200)
    train = LazyFinancialWindowDataset(tmp_path / "bar-store", split="train", window_size=64)
    statistics = resolve_scale_feature_statistics(
        train, sample_count=16, seed=59, loader_options={"num_workers": 0},
    )
    assert resolve_scale_feature_statistics(
        train, sample_count=16, seed=59, loader_options={"num_workers": 0},
    ) == statistics
    config = ExperimentConfig.from_yaml(ROOT / "configs/local_mock.yaml")
    config.data.input_length = 64
    config.model.feature_mode = "combined"
    bundle = build_model_bundle(config, torch.device("cpu"), scale_feature_statistics=statistics)
    batch = FinancialBatchCollator()([train[0], train[1]])
    output = bundle.model(
        batch["asset_series"], batch["benchmark_series"], target_alpha=batch["target_alpha"],
    )
    assert output.alpha_quantiles.shape == (2, 12, 3)
    assert output.loss is not None and torch.isfinite(output.loss)
    output.loss.backward()
    assert bundle.model.alpha_head.numeric_branch.fusion[-1].weight.grad is not None


def test_date_paired_comparisons_are_bounded_reproducible_and_directional() -> None:
    reference = {str(day): 0.3 for day in range(100)}
    candidate = {day: value - 0.02 for day, value in reference.items()}
    result = paired_block_comparison(candidate, reference)
    assert result == paired_block_comparison(candidate, reference)
    assert result["mean_daily_loss_difference"] == pytest.approx(-0.02)
    assert result["confidence_interval_95"] == pytest.approx([-0.02, -0.02])
    with pytest.raises(ValueError, match="identical"):
        paired_block_comparison({"missing": 0.2}, reference)
    assert paired_block_comparison({"1": 0.2}, {"1": 0.3})["confidence_interval_95"] is None
    with pytest.raises(ValueError, match="duplicated"):
        sample_membership(["A", "A"], ["2026-01-01", "2026-01-01"])
    values = daily_normalized_pinball(
        np.ones((2, 14)), np.zeros((2, 14, 3)), [2.0] * 14, ["a", "b"],
    )
    assert values == pytest.approx({"a": 0.25, "b": 0.25})


def test_each_exclusive_boundary_rejects_labels_ending_on_or_after_it() -> None:
    timestamps = pd.date_range("2025-01-01", "2026-08-31", tz="UTC")
    boundaries = _split_boundaries(
        list(timestamps), list(timestamps), train_fraction=0.7, validation_fraction=0.15,
        purge_bars=20, embargo_bars=14, fixed_split=FIXED_EVALUATION_SPLIT,
    )
    indices = np.arange(len(timestamps) - 14)
    codes, dropped = _split_codes(
        timestamp_ns=timestamps.asi8, indices=indices, observed_ns=timestamps.asi8,
        boundaries=boundaries, max_horizon=14,
    )
    for boundary in FIXED_EVALUATION_SPLIT.values():
        position = timestamps.get_loc(pd.Timestamp(boundary, tz="UTC"))
        assert codes[position - 14] == 0
        assert codes[position - 15] != 0
    assert (codes[timestamps[indices] >= pd.Timestamp("2026-06-01", tz="UTC")] == 0).all()
    assert dropped["after_test_end"] > 0


def test_historical_readonly_does_not_accept_changed_data_or_new_feature_mode() -> None:
    config = ExperimentConfig.from_yaml(ROOT / "configs/local_mock.yaml")
    historical = training_resume_contract(config)
    files = historical["training_implementation"]["files"]
    files["models/forecast.py"] = "0" * 64
    historical["training_implementation"]["sha256"] = _canonical_payload_digest(files)
    historical["data"].pop("calibration_seed")
    historical["model"].pop("feature_mode")
    historical["model"].pop("alpha_head_fp32")
    digest = _canonical_payload_digest(historical)
    manifest = {"training_resume_contract": historical, "training_resume_contract_sha256": digest}
    assert readonly_checkpoint_contract_digest(config, manifest) == digest
    changed = config.model_copy(deep=True)
    changed.data.input_length += 1
    with pytest.raises(ValueError, match="settings differ"):
        readonly_checkpoint_contract_digest(changed, manifest)
