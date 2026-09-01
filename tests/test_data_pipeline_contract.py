"""Offline adjusted-OHLCV, alpha-label, split, subset, and manifest contracts."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from stock_forecasting.baselines import baseline_arrays
from stock_forecasting.data.adjustments import asof_adjusted_window
from stock_forecasting.data.bar_store import bar_store_preparation_spec
from stock_forecasting.data.benchmarks import resolve_benchmark
from stock_forecasting.data.dataset import FinancialBatchCollator, FinancialWindowDataset
from stock_forecasting.data.manifest import (
    artifact_metadata,
    atomic_write_json,
    canonical_json_sha256,
    validate_dataset_preparation_contract,
    validate_download_manifest,
    validate_training_dataset_manifest,
)
from stock_forecasting.data.schema import (
    TRAINING_SECURITY_SCOPE,
    TRAINING_TARGET_ASSET_TYPES,
    MarketDataValidationError,
    normalize_ohlcv_frame,
)
from stock_forecasting.data.splits import SPLIT_POLICY, chronological_split
from stock_forecasting.data.windows import DEFAULT_ALPHA_HORIZONS, build_causal_windows


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
    assert set(label["alpha_log_returns"]) == {f"{horizon}d" for horizon in DEFAULT_ALPHA_HORIZONS}
    assert label["entry_day_counts_as_holding_day_one"] is True
    assert not ({"text", "facts", "description", "future_benchmark"} & set(record))


@pytest.mark.parametrize("h_start", [1, 2, 3])
def test_h_start_builds_contiguous_labels_through_fixed_day_14(
    market_frame: pd.DataFrame,
    h_start: int,
) -> None:
    records = build_causal_windows(
        market_frame,
        window_size=32,
        stride=20,
        h_start=h_start,
    )
    label = records[0]["label"]
    assert isinstance(label, dict)

    expected = list(range(h_start, 15))
    assert label["horizons"] == expected
    assert list(label["alpha_log_returns"]) == [f"{horizon}d" for horizon in expected]
    assert list(label["end_at"])[-1] == "14d"


def test_earlier_h_start_preserves_existing_day_3_through_14_labels(
    market_frame: pd.DataFrame,
) -> None:
    first_day = build_causal_windows(
        market_frame,
        window_size=32,
        stride=20,
        h_start=1,
    )
    third_day = build_causal_windows(
        market_frame,
        window_size=32,
        stride=20,
        h_start=3,
    )

    assert [record["sample_id"] for record in first_day] == [
        record["sample_id"] for record in third_day
    ]
    for first_record, third_record in zip(first_day, third_day, strict=True):
        assert first_record["context"] == third_record["context"]
        assert first_record["benchmark_context"] == third_record["benchmark_context"]
        first_labels = first_record["label"]["alpha_log_returns"]
        third_labels = third_record["label"]["alpha_log_returns"]
        assert {key: first_labels[key] for key in third_labels} == third_labels


def test_training_targets_include_common_stock_depositary_receipts_and_equity_etfs(
    market_frame: pd.DataFrame,
    window_records: list[dict[str, object]],
) -> None:
    assert {"stock", "etf"} == TRAINING_TARGET_ASSET_TYPES
    assert {str(record["asset_type"]) for record in window_records} == {"stock", "etf"}

    adr = market_frame[market_frame["symbol"] == "AAPL.US"].copy()
    adr["symbol"] = "BABA.US"
    adr["source_symbol"] = "BABA"
    tdr = market_frame[market_frame["symbol"] == "0050.TW"].copy()
    tdr["symbol"] = "9105.TW"
    tdr["asset_type"] = "stock"
    tdr["source_symbol"] = "9105"
    expanded = pd.concat([market_frame, adr, tdr], ignore_index=True)

    records = build_causal_windows(expanded, window_size=32, stride=20)

    assert {"AAPL.US", "BABA.US", "0050.TW", "9105.TW"} <= {
        str(record["symbol"]) for record in records
    }
    assert {
        str(record["asset_type"])
        for record in records
        if record["symbol"] in {"BABA.US", "9105.TW"}
    } == {"stock"}

    unsupported = market_frame[market_frame["symbol"].isin({"0050.TW", "TAIEX.TW"})].copy()
    unsupported.loc[unsupported["symbol"] == "0050.TW", "symbol"] = "2847B.TW"
    unsupported.loc[unsupported["symbol"] == "2847B.TW", "asset_type"] = "other"
    audit: dict[str, Any] = {}

    excluded = build_causal_windows(
        unsupported,
        window_size=32,
        stride=20,
        benchmark_mapping={"2847B.TW": "TAIEX.TW"},
        audit=audit,
    )

    assert excluded == []
    assert audit["excluded_counts_by_reason"]["unsupported_asset_type"] == 1


@pytest.mark.parametrize(
    ("symbol", "market", "benchmark"),
    (
        ("TQQQ.US", "US", "VTI.US"),
        ("SQQQ.US", "US", "VTI.US"),
        ("00631L.TW", "TWSE", "TAIEX.TW"),
        ("00632R.TW", "TWSE", "TAIEX.TW"),
    ),
)
def test_leveraged_and_inverse_etfs_cannot_bypass_the_allowlist_with_mapping(
    symbol: str,
    market: str,
    benchmark: str,
) -> None:
    decision = resolve_benchmark(
        symbol=symbol,
        asset_type="etf",
        market=market,
        explicit_mapping={symbol: benchmark},
    )

    assert decision.eligible is False
    assert decision.reason == "etf_not_in_audited_unleveraged_equity_allowlist"


def test_taiwan_preferred_share_cannot_be_mislabeled_as_stock_to_bypass_scope() -> None:
    decision = resolve_benchmark(
        symbol="2847B.TW",
        asset_type="stock",
        market="TWSE",
        explicit_mapping={"2847B.TW": "TAIEX.TW"},
    )

    assert decision.eligible is False
    assert decision.reason == "security_not_in_common_stock_or_depositary_receipt_scope"


def test_training_dataset_rejects_targets_outside_the_security_scope(
    window_records: list[dict[str, object]],
) -> None:
    invalid = copy.deepcopy(window_records[0])
    invalid["asset_type"] = "index"

    with pytest.raises(ValueError, match="common stock/ADR/TDR"):
        FinancialWindowDataset([invalid])

    leveraged = copy.deepcopy(window_records[0])
    leveraged["symbol"] = "TQQQ.US"
    leveraged["asset_type"] = "etf"
    leveraged["metadata"]["market"] = "US"
    with pytest.raises(ValueError, match="outside the common-stock/ADR/TDR"):
        FinancialWindowDataset([leveraged])

    preferred = copy.deepcopy(window_records[0])
    preferred["symbol"] = "2847B.TW"
    preferred["asset_type"] = "stock"
    preferred["metadata"]["market"] = "TWSE"
    with pytest.raises(ValueError, match="outside the common-stock/ADR/TDR"):
        FinancialWindowDataset([preferred])


def test_chronological_split_is_global_and_labels_never_cross_boundaries(
    market_frame: pd.DataFrame,
) -> None:
    windows = build_causal_windows(market_frame, window_size=32, stride=2)
    audit: dict[str, Any] = {}

    assigned = chronological_split(
        windows,
        train_fraction=0.70,
        validation_fraction=0.15,
        purge_bars=0,
        embargo_bars=0,
        audit=audit,
    )

    grouped = {
        split: [record for record in assigned if record["split"] == split]
        for split in ("train", "validation", "test")
    }
    assert max(pd.Timestamp(record["cutoff_at"]) for record in grouped["train"]) < min(
        pd.Timestamp(record["cutoff_at"]) for record in grouped["validation"]
    )
    assert max(
        pd.Timestamp(record["cutoff_at"]) for record in grouped["validation"]
    ) < min(pd.Timestamp(record["cutoff_at"]) for record in grouped["test"])
    train_boundary = pd.Timestamp(audit["train_boundary_exclusive"])
    validation_boundary = pd.Timestamp(audit["validation_boundary_exclusive"])
    assert max(
        pd.Timestamp(value)
        for record in grouped["train"]
        for value in record["label"]["end_at"].values()
    ) < train_boundary
    assert max(
        pd.Timestamp(value)
        for record in grouped["validation"]
        for value in record["label"]["end_at"].values()
    ) < validation_boundary
    assert audit["policy"] == SPLIT_POLICY
    assert audit["dropped_counts_by_reason"]["label_crosses_train_boundary"] > 0
    assert audit["dropped_counts_by_reason"]["label_crosses_validation_boundary"] > 0


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
            assert adjusted[context_name]["timestamp"] == original[context_name]["timestamp"]
            for field in ("open", "high", "low", "close", "volume"):
                assert adjusted[context_name][field] == pytest.approx(original[context_name][field])
        assert adjusted["label"]["alpha_log_returns"] == pytest.approx(
            original["label"]["alpha_log_returns"]
        )
    assert "adjusted_close" not in rescaled_records[0]["context"]


def test_parallel_window_build_matches_serial_output(market_frame: pd.DataFrame) -> None:
    serial_audit: dict[str, Any] = {}
    parallel_audit: dict[str, Any] = {}

    serial = build_causal_windows(
        market_frame,
        window_size=32,
        stride=20,
        workers=1,
        audit=serial_audit,
    )
    parallel = build_causal_windows(
        market_frame,
        window_size=32,
        stride=20,
        workers=4,
        audit=parallel_audit,
    )

    assert parallel == serial
    assert parallel_audit == {**serial_audit, "execution_workers": 4}


def test_future_benchmark_values_change_labels_but_never_the_model_context(
    market_frame: pd.DataFrame,
) -> None:
    cutoff = pd.Timestamp(market_frame[market_frame["symbol"] == "AAPL.US"].iloc[200]["timestamp"])
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
        record for record in modified if record["sample_id"] == original_record["sample_id"]
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
        ~((market_frame["symbol"] == "VTI.US") & (market_frame["timestamp"] == missing_day_two))
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
    bar_store = tmp_path / "prepared" / "bar-store"
    raw.parent.mkdir()
    bar_store.mkdir(parents=True)
    raw.write_bytes(b"immutable raw parquet fixture")
    symbol_index = bar_store / "symbol-index.parquet"
    cutoff_ranges = bar_store / "cutoff-ranges.parquet"
    symbol_index.write_bytes(b"immutable symbol index fixture")
    cutoff_ranges.write_bytes(b"immutable cutoff ranges fixture")
    bar_store_manifest = bar_store / "bar-store.json"
    atomic_write_json(
        bar_store_manifest,
        {
            "schema_version": "1.0",
            "kind": "symbol-oriented-ohlcv-bar-store",
            "state": "ready",
            "split_counts": {"test": 1, "train": 3, "validation": 1},
        },
    )
    manifest = tmp_path / "dataset-manifest.json"
    preparation_spec = bar_store_preparation_spec(
        window_size=128,
        max_horizon=14,
        benchmark_mapping_sha256=canonical_json_sha256({}),
        max_abs_log_return=0.5,
        train_fraction=0.70,
        validation_fraction=0.15,
        purge_bars=20,
        compatibility_stride=5,
        effective_sample_stride=1,
        compatibility_embargo_bars=5,
        effective_embargo_bars=14,
        target_horizon=5,
        diagnostic_horizons=[1, 20],
        flat_volatility_multiplier=0.25,
    )
    label_statistics = {
        "state": "runtime_calibration_required",
        "source_split": "train",
        "horizons_available": list(range(1, 15)),
        "robust_scale_method": "max(iqr,mad_x_1.4826,1e-4)",
        "prediction_units": "benchmark_relative_log_return",
    }
    split_audit = {
        "schema_version": "causal-split-audit-v1",
        "policy": SPLIT_POLICY,
        "comparison": "label.end_at < next_split_start",
        "violations": 0,
        "train_boundary_exclusive": "2025-01-01T00:00:00+00:00",
        "validation_boundary_exclusive": "2025-06-01T00:00:00+00:00",
        "validation_start": "2025-01-20T00:00:00+00:00",
        "test_start": "2025-06-20T00:00:00+00:00",
        "label_end_counts": {"train": 3, "validation": 1, "test": 1},
        "maximum_label_end": {
            "train": "2024-12-20T00:00:00+00:00",
            "validation": "2025-05-20T00:00:00+00:00",
            "test": "2025-07-10T00:00:00+00:00",
        },
        "dropped_counts_by_reason": {"purge_or_embargo": 2},
        "splits": {
            "train": {
                "records": 3,
                "cutoff_start_at": "2020-01-01T00:00:00+00:00",
                "cutoff_end_at": "2024-12-01T00:00:00+00:00",
                "label_end_max_at": "2024-12-20T00:00:00+00:00",
            },
            "validation": {
                "records": 1,
                "cutoff_start_at": "2025-01-20T00:00:00+00:00",
                "cutoff_end_at": "2025-05-01T00:00:00+00:00",
                "label_end_max_at": "2025-05-20T00:00:00+00:00",
            },
            "test": {
                "records": 1,
                "cutoff_start_at": "2025-06-20T00:00:00+00:00",
                "cutoff_end_at": "2025-06-20T00:00:00+00:00",
                "label_end_max_at": "2025-07-10T00:00:00+00:00",
            },
        },
    }
    payload = {
        "schema_version": "3.0",
        "kind": "ohlcv-bar-store-dataset",
        "state": "ready",
        "training_security_scope": TRAINING_SECURITY_SCOPE,
        "dataset_profile": "tw_only",
        "selected_datasets": ["tpex_official", "twse_official"],
        "split_counts": {"test": 1, "train": 3, "validation": 1},
        "split_audit": split_audit,
        "preparation_spec": preparation_spec,
        "preparation_spec_sha256": canonical_json_sha256(preparation_spec),
        "label_statistics": label_statistics,
        "artifacts": {
            "raw": artifact_metadata(raw, root=tmp_path, row_count=10),
            "bar_store_manifest": artifact_metadata(
                bar_store_manifest, root=tmp_path, row_count=1
            ),
            "symbol_index": artifact_metadata(symbol_index, root=tmp_path, row_count=2),
            "cutoff_ranges": artifact_metadata(cutoff_ranges, root=tmp_path, row_count=3),
        },
    }
    atomic_write_json(manifest, payload)

    validated = validate_training_dataset_manifest(
        manifest,
        profile="tw_only",
        raw_path=raw,
        bar_store_path=bar_store,
    )
    assert (
        validate_dataset_preparation_contract(
            validated,
            input_length=128,
            h_start=3,
            max_horizon=14,
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

    invalid_boundary = copy.deepcopy(validated)
    crossing_at = invalid_boundary["split_audit"]["train_boundary_exclusive"]
    invalid_boundary["split_audit"]["splits"]["train"]["label_end_max_at"] = crossing_at
    invalid_boundary["split_audit"]["maximum_label_end"]["train"] = crossing_at
    with pytest.raises(ValueError, match="train labels cross"):
        validate_dataset_preparation_contract(
            invalid_boundary,
            input_length=128,
            h_start=3,
            max_horizon=14,
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

    with pytest.raises(ValueError, match="window_size"):
        validate_dataset_preparation_contract(
            validated,
            input_length=64,
            h_start=3,
            max_horizon=14,
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
            bar_store_path=bar_store,
        )


def test_downloaded_checkpoint_binds_raw_and_request_log_integrity(tmp_path: Path) -> None:
    raw = tmp_path / "raw" / "market.parquet"
    request_log = tmp_path / "manifests" / "api-request-log.jsonl"
    raw.parent.mkdir()
    request_log.parent.mkdir()
    raw.write_bytes(b"immutable raw parquet fixture")
    request_log.write_text('{"provider":"fixture"}\n', encoding="utf-8")
    manifest = tmp_path / "download-manifest.json"
    atomic_write_json(
        manifest,
        {
            "schema_version": "2.0",
            "kind": "ohlcv-dataset",
            "state": "downloaded",
            "training_security_scope": TRAINING_SECURITY_SCOPE,
            "dataset_profile": "tw_only",
            "artifacts": {
                "raw": artifact_metadata(raw, root=tmp_path, row_count=10),
                "request_log": artifact_metadata(request_log, root=tmp_path, row_count=1),
            },
        },
    )

    validated = validate_download_manifest(manifest, input_path=raw)
    assert validated["state"] == "downloaded"

    obsolete = json.loads(manifest.read_text(encoding="utf-8"))
    obsolete["training_security_scope"] = "obsolete"
    atomic_write_json(manifest, obsolete)
    with pytest.raises(ValueError, match="security scope"):
        validate_download_manifest(manifest, input_path=raw)
    obsolete["training_security_scope"] = TRAINING_SECURITY_SCOPE
    atomic_write_json(manifest, obsolete)

    request_log.write_text('{"provider":"tampered"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="integrity mismatch"):
        validate_download_manifest(manifest, input_path=raw)


def test_manifest_rejects_secret_like_fields(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Secret-like key"):
        atomic_write_json(
            tmp_path / "manifest.json",
            {"schema_version": "2.0", "api_token": "must-not-be-written"},
        )
