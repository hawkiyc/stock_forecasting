"""Contracts for bounded, resumable bar storage and lazy label construction."""

from __future__ import annotations

import json
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
from stock_forecasting.training import (
    plan_dataloader_workers,
    resolve_runtime_robust_scales,
)
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
        assert lazy_record["label"][label_key] == pytest.approx(legacy_record["label"][label_key])
    for context_name in ("context", "benchmark_context"):
        assert lazy_record[context_name]["timestamp"] == legacy_record[context_name]["timestamp"]
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


def test_runtime_robust_scales_are_cached_without_materializing_labels(
    tmp_path: Path,
    market_frame: pd.DataFrame,
) -> None:
    raw = tmp_path / "raw" / "market.parquet"
    raw.parent.mkdir()
    market_frame.to_parquet(raw, compression="zstd", index=False)
    store = tmp_path / "prepared" / "bar-store"
    build_symbol_bar_store(
        raw_path=raw,
        output_root=store,
        download_manifest=_download_contract(raw, tmp_path, len(market_frame)),
        window_size=32,
        bucket_count=4,
        batch_rows=200,
        purge_bars=20,
        embargo_bars=14,
    )
    dataset = LazyFinancialWindowDataset(
        store,
        split="train",
        window_size=32,
        h_start=3,
    )
    worker_plan = plan_dataloader_workers(
        2,
        source="test",
        visible_cpu_count=4,
        available_memory_bytes=16 * 1024**3,
    )
    assert worker_plan.effective_workers == 2
    cache_root = tmp_path / "training-cache" / "robust-scales"

    first = resolve_runtime_robust_scales(
        dataset,
        sample_count=16,
        seed=59,
        worker_plan=worker_plan,
        cache_root=cache_root,
    )
    second = resolve_runtime_robust_scales(
        dataset,
        sample_count=16,
        seed=59,
        worker_plan=worker_plan,
        cache_root=cache_root,
    )

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert second.scales == pytest.approx(first.scales)
    assert first.cache_path == second.cache_path
    assert first.cache_path.parent == cache_root.resolve()
    assert not list(store.rglob("windows.parquet"))
    assert not list(store.rglob("*label*.parquet"))


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
    second = build_symbol_bar_store(**arguments, workers=4)

    assert sha256_file(second.bar_store_manifest_path) == before
    assert second.split_counts == first.split_counts


def test_memory_planner_reduces_workers_and_rejects_unsafe_single_task() -> None:
    available = 16 * bar_store_module.GIB
    budget = bar_store_module._safe_worker_memory_budget(available, None)
    task_memory = 3 * bar_store_module.GIB
    plan = bar_store_module._plan_phase_workers(
        phase="test_compaction",
        requested_workers=16,
        detected_cpu_count=16,
        task_memory_bytes=[task_memory] * 8,
        task_count=8,
        reused_tasks=0,
        available_memory_bytes=available,
        worker_memory_budget_bytes=budget,
    )

    assert plan.effective_workers == 3
    assert plan.estimated_peak_worker_memory_bytes <= budget
    assert plan.effective_workers < plan.requested_workers
    assert plan.as_dict()["native_threads_per_worker"] == 1

    with pytest.raises(
        bar_store_module.PreparationMemoryLimitExceeded,
        match="completed checkpoints remain reusable",
    ):
        bar_store_module._plan_phase_workers(
            phase="oversized_bucket",
            requested_workers=4,
            detected_cpu_count=4,
            task_memory_bytes=[budget + 1],
            task_count=1,
            reused_tasks=0,
            available_memory_bytes=available,
            worker_memory_budget_bytes=budget,
        )


def test_parallel_bar_store_matches_serial_output(
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
    contract = _download_contract(raw, tmp_path, len(market_frame))
    common = {
        "raw_path": raw,
        "download_manifest": contract,
        "window_size": 32,
        "bucket_count": 4,
        "batch_rows": 100,
        "purge_bars": 20,
        "embargo_bars": 14,
        "memory_budget_bytes": 8 * bar_store_module.GIB,
    }
    monkeypatch.setattr(bar_store_module, "_visible_cpu_count", lambda: 4)
    monkeypatch.setattr(
        bar_store_module,
        "_detect_available_memory_bytes",
        lambda: 16 * bar_store_module.GIB,
    )

    serial = build_symbol_bar_store(
        **common,
        output_root=tmp_path / "serial" / "bar-store",
        workers=1,
    )
    parallel = build_symbol_bar_store(
        **common,
        output_root=tmp_path / "parallel" / "bar-store",
        workers=4,
    )

    serial_manifest = json.loads(serial.bar_store_manifest_path.read_text(encoding="utf-8"))
    parallel_manifest = json.loads(parallel.bar_store_manifest_path.read_text(encoding="utf-8"))
    assert serial_manifest["identity_sha256"] == parallel_manifest["identity_sha256"]
    assert serial.split_counts == parallel.split_counts
    assert serial.split_audit == parallel.split_audit
    assert serial.quality == parallel.quality
    pd.testing.assert_frame_equal(
        pd.read_parquet(serial.symbol_index_path),
        pd.read_parquet(parallel.symbol_index_path),
    )
    pd.testing.assert_frame_equal(
        pd.read_parquet(serial.cutoff_ranges_path),
        pd.read_parquet(parallel.cutoff_ranges_path),
    )
    parallelism = parallel.execution["parallelism"]
    assert parallelism["backend"] == "process_pool_spawn"
    assert parallelism["native_threads_per_worker"] == 1
    assert parallelism["phases"]["raw_scan"]["effective_workers"] > 1
    assert parallelism["phases"]["candidate_ranges"]["effective_workers"] > 1


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

    def pause_after_first_partition(
        deadline: bar_store_module._Deadline,
        operation: str,
    ) -> None:
        if operation == "raw scan partition 1 batch 0":
            raise PreparationPaused("test pause; durable checkpoints remain")
        original_check(deadline, operation)

    monkeypatch.setattr(
        bar_store_module._Deadline,
        "check",
        pause_after_first_partition,
    )
    with pytest.raises(PreparationPaused, match="durable checkpoints remain"):
        build_symbol_bar_store(**arguments)

    assert (store / ".work" / "build-state.json").is_file()
    scan_index = json.loads((store / ".work" / "scan-index.json").read_text(encoding="utf-8"))
    assert scan_index["state"] == "building"
    assert len(scan_index["completed_partitions"]) == 1
    first_partition = store / ".work" / "scan-partitions" / "partition-0000"
    assert (first_partition / "checkpoint.json").is_file()
    assert (first_partition / "partition.parquet").is_file()
    assert not (store / ".work" / "segments").exists()
    execution_plan = json.loads(
        (store / ".work" / "execution-plan.json").read_text(encoding="utf-8")
    )
    assert execution_plan["phases"]["raw_scan"]["effective_workers"] == 1
    assert not (store / "_SUCCESS.json").exists()
    monkeypatch.setattr(bar_store_module._Deadline, "check", original_check)
    resumed = build_symbol_bar_store(**arguments)
    assert resumed.success_path.is_file()
    assert not list((store / ".work").iterdir())


def test_fragmented_raw_row_groups_are_coalesced_without_materializing_windows(
    tmp_path: Path,
    market_frame: pd.DataFrame,
) -> None:
    raw = tmp_path / "raw" / "market.parquet"
    raw.parent.mkdir()
    market_frame.to_parquet(
        raw,
        compression="zstd",
        index=False,
        row_group_size=5,
    )
    result = build_symbol_bar_store(
        raw_path=raw,
        output_root=tmp_path / "prepared" / "bar-store",
        download_manifest=_download_contract(raw, tmp_path, len(market_frame)),
        window_size=32,
        bucket_count=4,
        batch_rows=100,
        purge_bars=20,
        embargo_bars=14,
    )

    execution = result.execution
    assert execution["raw_scan_source_row_groups"] == 288
    assert execution["raw_scan_partitions"] == 4
    assert execution["raw_scan_partitions"] < execution["raw_scan_source_row_groups"]
    assert execution["raw_scan_target_partition_rows"] == 400
    assert execution["window_materialized"] is False
    assert execution["labels_materialized"] is False
    assert not list(tmp_path.rglob("windows.parquet"))
    assert not list((tmp_path / "prepared" / "bar-store").rglob("*label*.parquet"))


def test_scan_layout_upgrade_quarantines_only_incomplete_derived_work(
    tmp_path: Path,
    market_frame: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = tmp_path / "raw" / "market.parquet"
    raw.parent.mkdir()
    market_frame.to_parquet(raw, compression="zstd", index=False, row_group_size=200)
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

    def pause_immediately(
        deadline: bar_store_module._Deadline,
        operation: str,
    ) -> None:
        if operation == "raw scan partition 0 batch 0":
            raise PreparationPaused("create incomplete work")
        original_check(deadline, operation)

    monkeypatch.setattr(bar_store_module._Deadline, "check", pause_immediately)
    with pytest.raises(PreparationPaused, match="create incomplete work"):
        build_symbol_bar_store(**arguments)

    state_path = store / ".work" / "build-state.json"
    previous_state = json.loads(state_path.read_text(encoding="utf-8"))
    previous_state["identity"]["raw_scan_algorithm"] = "parquet-row-group-checkpoints-v2"
    previous_state["identity"].pop("raw_scan_target_partition_rows")
    previous_state["identity_sha256"] = bar_store_module.canonical_json_sha256(
        previous_state["identity"]
    )
    state_path.write_text(
        json.dumps(previous_state, sort_keys=True),
        encoding="utf-8",
    )
    legacy_sentinel = store / ".work" / "segments" / "segment-000000" / "checkpoint.json"
    legacy_sentinel.parent.mkdir(parents=True)
    legacy_sentinel.write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(bar_store_module._Deadline, "check", original_check)
    result = build_symbol_bar_store(**arguments)

    assert result.success_path.is_file()
    quarantines = list((tmp_path / "prepared").glob(".bar-store-obsolete-scan-*"))
    assert len(quarantines) == 1
    assert (quarantines[0] / "work" / "segments" / "segment-000000").is_dir()
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

    fractional = BlockwisePermutationSampler(
        10_003,
        fraction=0.03,
        seed=29,
        block_size=64,
    )
    fractional_first_epoch = list(fractional)
    fractional.set_epoch(1)
    fractional_second_epoch = list(fractional)
    assert (
        len(fractional_first_epoch)
        == len(set(fractional_first_epoch))
        == int(10_003 * 0.03)
    )
    assert fractional_second_epoch != fractional_first_epoch
    assert set(fractional_second_epoch) == set(fractional_first_epoch)

    fixed_batches = list(FixedSizeBatchSampler(complete, batch_size=128))
    assert all(len(batch) == 128 for batch in fixed_batches)
    assert len(fixed_batches) == 8
    assert len({index for batch in fixed_batches for index in batch}) == 1_003
