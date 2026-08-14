"""Offline adjusted-OHLCV, alpha-label, split, subset, and manifest contracts."""

from __future__ import annotations

import copy
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from stock_forecasting.baselines import baseline_arrays
from stock_forecasting.data.adjustments import asof_adjusted_window
from stock_forecasting.data.dataset import FinancialBatchCollator, FinancialWindowDataset
from stock_forecasting.data.manifest import (
    artifact_metadata,
    atomic_write_json,
    canonical_json_sha256,
    validate_dataset_preparation_contract,
    validate_training_dataset_manifest,
)
from stock_forecasting.data.schema import MarketDataValidationError, normalize_ohlcv_frame
from stock_forecasting.data.windows import DEFAULT_ALPHA_HORIZONS, build_causal_windows
from stock_forecasting.training import deterministic_stratified_indices


def test_causal_records_have_paired_historical_inputs_and_future_labels_only(
    window_records: list[dict[str, object]],
) -> None:
    record = window_records[0]
    context = record["context"]
    benchmark_context = record["benchmark_context"]
    label = record["label"]
    assert isinstance(context, dict)
    assert isinstance(benchmark_context, dict)
    assert isinstance(label, dict)
    expected_context_fields = {"timestamp", "open", "high", "low", "close", "volume"}
    assert set(context) == expected_context_fields
    assert set(benchmark_context) == expected_context_fields
    cutoff = pd.Timestamp(str(record["cutoff_at"]))
    assert max(pd.to_datetime(context["timestamp"], utc=True)) <= cutoff
    assert max(pd.to_datetime(benchmark_context["timestamp"], utc=True)) <= cutoff
    assert tuple(label["horizons"]) == DEFAULT_ALPHA_HORIZONS
    assert pd.Timestamp(str(label["entry_at"])) > cutoff
    assert all(pd.Timestamp(value) > cutoff for value in label["end_at"].values())
    assert set(label["alpha_log_returns"]) == {
        f"{horizon}d" for horizon in DEFAULT_ALPHA_HORIZONS
    }
    assert label["entry_day_counts_as_holding_day_one"] is True
    assert not ({"text", "facts", "description", "future_benchmark"} & set(record))


def test_globally_back_adjusted_vendor_scale_cancels_at_each_cutoff(
    market_frame: pd.DataFrame,
) -> None:
    rescaled = market_frame.copy()
    rescaled["adjusted_close"] = rescaled["adjusted_close"] * 0.01

    original_records = build_causal_windows(market_frame, window_size=32, stride=20)
    rescaled_records = build_causal_windows(rescaled, window_size=32, stride=20)

    assert len(rescaled_records) == len(original_records)
    for original, adjusted in zip(original_records, rescaled_records, strict=True):
        assert adjusted["sample_id"] == original["sample_id"]
        for context_name in ("context", "benchmark_context"):
            assert adjusted[context_name]["timestamp"] == original[context_name][
                "timestamp"
            ]
            for field in ("open", "high", "low", "close", "volume"):
                assert adjusted[context_name][field] == pytest.approx(
                    original[context_name][field]
                )
        assert adjusted["label"]["alpha_log_returns"] == pytest.approx(
            original["label"]["alpha_log_returns"]
        )
    assert "adjusted_close" not in rescaled_records[0]["context"]


def test_future_benchmark_values_change_labels_but_never_the_model_context(
    market_frame: pd.DataFrame,
) -> None:
    cutoff = pd.Timestamp(
        market_frame[market_frame["symbol"] == "AAPL.US"].iloc[200]["timestamp"]
    )
    changed = market_frame.copy()
    benchmark_mask = (changed["symbol"] == "VTI.US") & (changed["timestamp"] > cutoff)
    steps = np.arange(int(benchmark_mask.sum()), dtype=np.float64) + 1.0
    changed.loc[benchmark_mask, "adjusted_close"] *= np.exp(0.002 * steps)

    original = build_causal_windows(market_frame, window_size=32, stride=1)
    modified = build_causal_windows(changed, window_size=32, stride=1)
    original_record = next(
        record
        for record in original
        if record["symbol"] == "AAPL.US" and pd.Timestamp(record["cutoff_at"]) == cutoff
    )
    modified_record = next(
        record
        for record in modified
        if record["sample_id"] == original_record["sample_id"]
    )

    assert modified_record["context"] == original_record["context"]
    assert modified_record["benchmark_context"] == original_record["benchmark_context"]
    assert (
        modified_record["label"]["alpha_log_returns"]
        != original_record["label"]["alpha_log_returns"]
    )


def test_label_holding_interval_requires_every_benchmark_trading_date(
    market_frame: pd.DataFrame,
) -> None:
    asset_rows = (
        market_frame[market_frame["symbol"] == "AAPL.US"]
        .sort_values("timestamp", kind="stable")
        .reset_index(drop=True)
    )
    cutoff = pd.Timestamp(asset_rows.loc[200, "timestamp"])
    missing_day_two = pd.Timestamp(asset_rows.loc[202, "timestamp"])
    incomplete = market_frame[
        ~(
            (market_frame["symbol"] == "VTI.US")
            & (market_frame["timestamp"] == missing_day_two)
        )
    ].copy()

    original = build_causal_windows(market_frame, window_size=32, stride=1)
    modified = build_causal_windows(incomplete, window_size=32, stride=1)

    assert any(
        record["symbol"] == "AAPL.US" and pd.Timestamp(record["cutoff_at"]) == cutoff
        for record in original
    )
    assert not any(
        record["symbol"] == "AAPL.US" and pd.Timestamp(record["cutoff_at"]) == cutoff
        for record in modified
    )


def test_point_in_time_adjustment_removes_split_jump_from_ohlcv() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-02", "2026-01-05"], utc=True),
            "symbol": ["TEST.US", "TEST.US"],
            "asset_type": ["stock", "stock"],
            "open": [100.0, 50.0],
            "high": [102.0, 52.0],
            "low": [98.0, 49.0],
            "close": [100.0, 50.0],
            "volume": [100.0, 200.0],
            "adjusted_close": [50.0, 50.0],
            "split_adjusted_volume": [200.0, 200.0],
        }
    )

    adjusted = asof_adjusted_window(frame)

    assert adjusted["close"].tolist() == pytest.approx([50.0, 50.0])
    assert adjusted["open"].tolist() == pytest.approx([50.0, 50.0])
    assert adjusted["volume"].tolist() == pytest.approx([200.0, 200.0])


def test_dataset_collator_keeps_paired_numeric_inputs_and_continuous_targets(
    window_records: list[dict[str, object]],
) -> None:
    dataset = FinancialWindowDataset(window_records, split="train", series_mode="relative")
    batch = FinancialBatchCollator()([dataset[0], dataset[1]])

    assert batch["asset_series"].shape == (2, 32, 5)
    assert batch["benchmark_series"].shape == (2, 32, 5)
    assert batch["asset_timestamps"].shape == (2, 32, 5)
    assert batch["benchmark_timestamps"].shape == (2, 32, 5)
    assert batch["asset_attention_mask"].all()
    assert batch["benchmark_attention_mask"].all()
    assert batch["target_alpha"].shape == (2, 12)
    assert len(batch["benchmark_symbols"]) == 2
    assert "target_classes" not in batch
    assert "input_ids" not in batch
    assert "fact_targets" not in batch


def test_dataset_uses_only_serialized_adjusted_context_fields(
    window_records: list[dict[str, object]],
) -> None:
    original = copy.deepcopy(window_records[0])
    with_extra = copy.deepcopy(original)
    context = with_extra["context"]
    assert isinstance(context, dict)
    context["adjusted_close"] = [0.01] * len(context["close"])

    original_item = FinancialWindowDataset([original])[0]
    extra_item = FinancialWindowDataset([with_extra])[0]

    torch_equal = original_item["asset_series"].equal(extra_item["asset_series"])
    assert torch_equal


def test_baseline_arrays_use_exact_asset_and_benchmark_pair(
    window_records: list[dict[str, object]],
) -> None:
    arrays = baseline_arrays(cast_records(window_records))

    assert arrays.sequences.shape[1] == 2
    assert arrays.instrument_mask.shape[1] == 2
    assert arrays.instrument_mask.all()
    assert arrays.targets.shape[1] == 12


def cast_records(records: list[dict[str, object]]) -> list[dict[str, Any]]:
    return [dict(record) for record in records]


def test_baselines_ignore_non_contract_context_fields(
    window_records: list[dict[str, object]],
) -> None:
    raw_records = copy.deepcopy(window_records)
    extra_records = copy.deepcopy(window_records)
    for record in extra_records:
        context = record["context"]
        benchmark_context = record["benchmark_context"]
        assert isinstance(context, dict)
        assert isinstance(benchmark_context, dict)
        context["adjusted_close"] = [0.01] * len(context["close"])
        benchmark_context["adjusted_close"] = [0.01] * len(benchmark_context["close"])

    raw = baseline_arrays(cast_records(raw_records))
    extra = baseline_arrays(cast_records(extra_records))

    assert (raw.features == extra.features).all()
    assert (raw.sequences == extra.sequences).all()


def test_stage1_subset_is_exact_deterministic_and_stratified(
    window_records: list[dict[str, object]],
) -> None:
    dataset = FinancialWindowDataset(window_records, split="train")
    first = deterministic_stratified_indices(dataset, fraction=0.15)
    second = deterministic_stratified_indices(dataset, fraction=0.15)

    assert first == second
    assert len(first) == max(1, int(len(dataset) * 0.15))
    selected_groups = Counter(
        (
            str(dataset.records[index]["metadata"]["market"]),
            str(dataset.records[index]["asset_type"]),
        )
        for index in first
    )
    assert set(selected_groups) == {("TWSE", "etf"), ("US", "stock")}


def test_schema_rejects_duplicate_bars_and_invalid_ranges(
    market_frame: pd.DataFrame,
) -> None:
    duplicate = pd.concat([market_frame.iloc[:1], market_frame.iloc[:1]], ignore_index=True)
    with pytest.raises(MarketDataValidationError, match="Duplicate"):
        normalize_ohlcv_frame(duplicate)

    invalid = market_frame.iloc[:1].copy()
    invalid["high"] = invalid["low"] - 1.0
    with pytest.raises(MarketDataValidationError, match="high must"):
        normalize_ohlcv_frame(invalid)


def test_ready_manifest_binds_conditional_alpha_contract_and_artifacts(tmp_path: Path) -> None:
    raw = tmp_path / "raw" / "market.parquet"
    processed = tmp_path / "processed" / "windows.parquet"
    raw.parent.mkdir()
    processed.parent.mkdir()
    raw.write_bytes(b"immutable raw parquet fixture")
    processed.write_bytes(b"immutable processed parquet fixture")
    manifest = tmp_path / "dataset-manifest.json"
    preparation_spec = {
        "processed_schema_version": "3.0",
        "window_size": 128,
        "target_horizon": 5,
        "diagnostic_horizons": [1, 20],
        "stride": 5,
        "effective_sample_stride": 1,
        "alpha_horizons": list(DEFAULT_ALPHA_HORIZONS),
        "label_kind": "benchmark_relative_adjusted_log_return",
        "signal_timing": "after_close_t",
        "entry_timing": "regular_session_open_t_plus_1",
        "entry_day_counts_as_holding_day_one": True,
        "exit_timing": "regular_session_close_t_plus_h",
        "input_adjustment": "point_in_time_total_return_ohlc_split_adjusted_volume",
        "benchmark_mapping_sha256": canonical_json_sha256({}),
        "flat_volatility_multiplier": 0.25,
        "max_abs_log_return": 0.5,
        "train_fraction": 0.70,
        "validation_fraction": 0.15,
        "purge_bars": 20,
        "embargo_bars": 5,
        "effective_embargo_bars": 14,
    }
    label_statistics = {
        "source_split": "train",
        "horizons": list(DEFAULT_ALPHA_HORIZONS),
        "robust_scale_method": "max(iqr,mad_x_1.4826,1e-4)",
        "robust_scales": [0.01] * len(DEFAULT_ALPHA_HORIZONS),
        "prediction_units": "benchmark_relative_log_return",
    }
    payload = {
        "schema_version": "2.0",
        "kind": "ohlcv-dataset",
        "state": "ready",
        "dataset_profile": "tw_only",
        "selected_datasets": ["tpex_official", "twse_official"],
        "split_counts": {"test": 1, "train": 3, "validation": 1},
        "preparation_spec": preparation_spec,
        "preparation_spec_sha256": canonical_json_sha256(preparation_spec),
        "label_statistics": label_statistics,
        "artifacts": {
            "raw": artifact_metadata(raw, root=tmp_path, row_count=10),
            "processed": artifact_metadata(processed, root=tmp_path, row_count=5),
        },
    }
    atomic_write_json(manifest, payload)

    validated = validate_training_dataset_manifest(
        manifest,
        profile="tw_only",
        raw_path=raw,
        processed_path=processed,
    )
    assert (
        validate_dataset_preparation_contract(
            validated,
            input_length=128,
            alpha_horizons=list(DEFAULT_ALPHA_HORIZONS),
            benchmark_mapping_path=None,
            sample_stride=1,
            effective_embargo_trading_days=14,
            forecast_horizon=5,
            diagnostic_horizons=[1, 20],
            stride=5,
            flat_volatility_multiplier=0.25,
            max_abs_log_return=0.5,
            embargo_trading_days=5,
        )
        == preparation_spec
    )

    with pytest.raises(ValueError, match="window_size"):
        validate_dataset_preparation_contract(
            validated,
            input_length=64,
            alpha_horizons=list(DEFAULT_ALPHA_HORIZONS),
            benchmark_mapping_path=None,
            sample_stride=1,
            effective_embargo_trading_days=14,
            forecast_horizon=5,
            diagnostic_horizons=[1, 20],
            stride=5,
            flat_volatility_multiplier=0.25,
            max_abs_log_return=0.5,
            embargo_trading_days=5,
        )

    raw.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="integrity mismatch"):
        validate_training_dataset_manifest(
            manifest,
            profile="tw_only",
            raw_path=raw,
            processed_path=processed,
        )


def test_manifest_rejects_secret_like_fields(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Secret-like key"):
        atomic_write_json(
            tmp_path / "manifest.json",
            {"schema_version": "2.0", "api_token": "must-not-be-written"},
        )
