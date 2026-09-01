"""Contracts for bounded, resumable bar storage and lazy label construction."""

from __future__ import annotations

from itertools import islice
from pathlib import Path

import pandas as pd
import pytest
import torch

import stock_forecasting.data.bar_store as bar_store_module
from stock_forecasting.data.bar_store import PreparationPaused, build_symbol_bar_store
from stock_forecasting.data.dataset import (
    BlockwisePermutationSampler,
    FixedSizeBatchSampler,
    LazyFinancialWindowDataset,
)
from stock_forecasting.data.manifest import artifact_metadata, sha256_file
from stock_forecasting.data.windows import CONTEXT_FIELDS, build_causal_windows
from stock_forecasting.training_paths import resolve_bar_store_path


def _download_contract(raw: Path, root: Path, row_count: int) -> dict[str, object]:
    return {
        "artifacts": {
            "raw": artifact_metadata(raw, root=root, row_count=row_count),
        }
    }


def test_bar_store_persists_bars_once_and_builds_labels_lazily(
    tmp_path: Path,
    market_frame: pd.DataFrame,
) -> None:
    raw = tmp_path / "raw" / "market.parquet"
    raw.parent.mkdir()
    market_frame.to_parquet(raw, compression="zstd", index=False)
    store = tmp_path / "prepared" / "bar-store"
    result = build_symbol_bar_store(
        raw_path=raw,
        output_root=store,
        download_manifest=_download_contract(raw, tmp_path, len(market_frame)),
        window_size=32,
        bucket_count=4,
        batch_rows=200,
        purge_bars=20,
        embargo_bars=14,
    )

    assert result.success_path.is_file()
    assert not list(tmp_path.rglob("windows.parquet"))
    assert not list(store.rglob("*label*.parquet"))
    assert not list((store / ".work").iterdir())
    symbol_index = pd.read_parquet(result.symbol_index_path)
    assert int(symbol_index["row_count"].sum()) == len(market_frame)
    assert symbol_index["symbol"].is_unique

    first_day = LazyFinancialWindowDataset(
        store,
        split="train",
        window_size=32,
        h_start=1,
    )
    third_day = LazyFinancialWindowDataset(
        store,
        split="train",
        window_size=32,
        h_start=3,
    )
    with pytest.raises(ValueError, match="h_start must be 1, 2, or 3"):
        LazyFinancialWindowDataset(
            store,
            split="train",
            window_size=32,
            h_start=4,
        )
    first_item = first_day[0]
    third_item = third_day[0]
    assert first_item["sample_id"] == third_item["sample_id"]
    assert torch.equal(first_item["asset_series"], third_item["asset_series"])
    assert first_item["target_alpha"].shape == (14,)
    assert third_item["target_alpha"].shape == (12,)
    assert torch.allclose(first_item["target_alpha"][2:], third_item["target_alpha"])
    assert first_item["asset_series"].shape == (32, 5)
    assert first_item["benchmark_series"].shape == (32, 5)
    assert torch.equal(first_day.target_at(0), first_item["target_alpha"])

    lazy_record = first_day.record_at(0)
    legacy_by_id = {
        record["sample_id"]: record
        for record in build_causal_windows(
            market_frame,
            window_size=32,
            stride=1,
            h_start=1,
        )
    }
    legacy_record = legacy_by_id[lazy_record["sample_id"]]
    assert lazy_record["window_start_at"] == legacy_record["window_start_at"]
    assert lazy_record["cutoff_at"] == legacy_record["cutoff_at"]
    for label_key in (
        "kind",
        "horizons",
        "entry_at",
        "entry_price_field",
        "entry_day_counts_as_holding_day_one",
        "end_at",
    ):
        assert lazy_record["label"][label_key] == legacy_record["label"][label_key]
    for label_key in (
        "alpha_log_returns",
        "asset_total_returns",
        "benchmark_total_returns",
    ):
        assert lazy_record["label"][label_key] == pytest.approx(
            legacy_record["label"][label_key]
        )
    for context_name in ("context", "benchmark_context"):
        assert lazy_record[context_name]["timestamp"] == legacy_record[context_name][
            "timestamp"
        ]
        for field in CONTEXT_FIELDS:
            assert lazy_record[context_name][field] == pytest.approx(
                legacy_record[context_name][field]
            )

    assert resolve_bar_store_path(store) == store.resolve()
    first_shard = next((store / "shards").glob("bucket-*/shard.parquet"))
    with first_shard.open("r+b") as stream:
        original = stream.read(1)
        stream.seek(0)
        stream.write(bytes([original[0] ^ 0x01]))
    with pytest.raises(ValueError, match="shard integrity mismatch"):
        resolve_bar_store_path(store)


def test_bar_store_resume_reuses_completed_outputs(
    tmp_path: Path,
    market_frame: pd.DataFrame,
) -> None:
    raw = tmp_path / "raw" / "market.parquet"
    raw.parent.mkdir()
    market_frame.to_parquet(raw, compression="zstd", index=False)
    store = tmp_path / "prepared" / "bar-store"
    arguments = {
        "raw_path": raw,
        "output_root": store,
        "download_manifest": _download_contract(raw, tmp_path, len(market_frame)),
        "window_size": 32,
        "bucket_count": 4,
        "batch_rows": 200,
        "purge_bars": 20,
        "embargo_bars": 14,
    }
    first = build_symbol_bar_store(**arguments)
    before = sha256_file(first.bar_store_manifest_path)
    second = build_symbol_bar_store(**arguments)

    assert sha256_file(second.bar_store_manifest_path) == before
    assert second.split_counts == first.split_counts


def test_expired_deadline_preserves_identity_and_next_attempt_resumes(
    tmp_path: Path,
    market_frame: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = tmp_path / "raw" / "market.parquet"
    raw.parent.mkdir()
    market_frame.to_parquet(
        raw,
        compression="zstd",
        index=False,
        row_group_size=200,
    )
    store = tmp_path / "prepared" / "bar-store"
    arguments = {
        "raw_path": raw,
        "output_root": store,
        "download_manifest": _download_contract(raw, tmp_path, len(market_frame)),
        "window_size": 32,
        "bucket_count": 4,
        "batch_rows": 100,
        "purge_bars": 20,
        "embargo_bars": 14,
    }

    original_check = bar_store_module._Deadline.check

    def pause_after_first_row_group(
        deadline: bar_store_module._Deadline,
        operation: str,
    ) -> None:
        if operation == "raw segment 2":
            raise PreparationPaused("test pause; durable checkpoints remain")
        original_check(deadline, operation)

    monkeypatch.setattr(
        bar_store_module._Deadline,
        "check",
        pause_after_first_row_group,
    )
    with pytest.raises(PreparationPaused, match="durable checkpoints remain"):
        build_symbol_bar_store(**arguments)

    assert (store / ".work" / "build-state.json").is_file()
    assert (store / ".work" / "segments" / "row-group-000000.json").is_file()
    assert (store / ".work" / "segments" / "segment-000000").is_dir()
    assert (store / ".work" / "segments" / "segment-000001").is_dir()
    assert not (store / "_SUCCESS.json").exists()
    monkeypatch.setattr(bar_store_module._Deadline, "check", original_check)
    resumed = build_symbol_bar_store(**arguments)
    assert resumed.success_path.is_file()
    assert not list((store / ".work").iterdir())


def test_blockwise_sampler_is_exact_deterministic_and_does_not_store_window_indices() -> None:
    first = BlockwisePermutationSampler(1_000_003, fraction=0.15, seed=19)
    second = BlockwisePermutationSampler(1_000_003, fraction=0.15, seed=19)

    assert len(first) == int(1_000_003 * 0.15)
    assert list(islice(first, 1024)) == list(islice(second, 1024))
    assert not any(isinstance(value, list) for value in vars(first).values())

    complete = BlockwisePermutationSampler(1_003, seed=23, block_size=64)
    first_epoch = list(complete)
    complete.set_epoch(1)
    second_epoch = list(complete)
    assert len(first_epoch) == len(set(first_epoch)) == 1_003
    assert set(first_epoch) == set(range(1_003))
    assert second_epoch != first_epoch
    assert set(second_epoch) == set(first_epoch)

    fixed_batches = list(FixedSizeBatchSampler(complete, batch_size=128))
    assert all(len(batch) == 128 for batch in fixed_batches)
    assert len(fixed_batches) == 8
    assert len({index for batch in fixed_batches for index in batch}) == 1_003
