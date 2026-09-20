"""Exhaustive, independent window enumeration and bounded evaluation resource checks."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
from torch.utils.data import DataLoader

from stock_forecasting.config import ExperimentConfig
from stock_forecasting.data.bar_store import build_symbol_bar_store
from stock_forecasting.data.dataset import FinancialBatchCollator, LazyFinancialWindowDataset
from stock_forecasting.data.manifest import artifact_metadata
from stock_forecasting.dataset_identity import FIXED_EVALUATION_SPLIT
from stock_forecasting.evaluation_protocol import evaluation_sampler, sample_membership
from stock_forecasting.training import (
    BatchProbeMeasurement,
    _explicit_runtime_batch_plan,
    _loader_process_options,
    plan_dataloader_workers,
    plan_runtime_prefetch,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def config_path():
    return ROOT / "configs/local_mock.yaml"


@pytest.fixture
def fixed_store(tmp_path, market_frame):
    frame = market_frame.copy()
    dates = sorted(frame["timestamp"].unique())
    frame["timestamp"] = frame["timestamp"].map(
        dict(zip(dates, pd.bdate_range("2025-01-01", periods=len(dates), tz="UTC"), strict=True))
    )
    # A late listing and a missing benchmark bar create unequal, noncontiguous histories.
    frame = frame.loc[~((frame.symbol == "AAPL.US") & (frame.timestamp < "2025-05-01"))]
    frame = frame.loc[~((frame.symbol == "VTI.US") & (frame.timestamp == "2025-09-01"))]
    raw = tmp_path / "raw.parquet"
    frame.to_parquet(raw, index=False)
    root = tmp_path / "bars"
    build_symbol_bar_store(
        raw_path=raw,
        output_root=root,
        download_manifest={"artifacts": {"raw": artifact_metadata(raw, root=tmp_path)}},
        window_size=32,
        bucket_count=2,
        batch_rows=200,
        fixed_split=FIXED_EVALUATION_SPLIT,
    )
    return root


@pytest.mark.parametrize(
    "split,lower,upper",
    [
        ("validation", "2025-06-01", "2025-12-01"),
        ("test", "2025-12-01", "2026-06-01"),
    ],
)
def test_every_instrument_every_valid_cutoff_independent_of_loader(
    fixed_store,
    config_path,
    split,
    lower,
    upper,
):
    source = LazyFinancialWindowDataset(fixed_store, split=split, window_size=32, h_start=1)
    expected = []
    # Independent exhaustive oracle: do not use the prepared ranges or preparation mask.
    for symbol, row in sorted(source._index.items()):
        if not row["eligible"]:
            continue
        frame = source._load_symbol(symbol)
        benchmark_dates = set(source._load_symbol(row["benchmark_symbol"])["timestamp"])
        for cutoff in range(31, len(frame) - 14):
            day = frame.iloc[cutoff].timestamp
            if not pd.Timestamp(lower, tz="UTC") <= day < pd.Timestamp(upper, tz="UTC"):
                continue
            if frame.iloc[cutoff + 14].timestamp >= pd.Timestamp(upper, tz="UTC"):
                continue
            if frame.iloc[cutoff - 30 : cutoff + 15].adjusted_transition_extreme.any():
                continue
            if not set(frame.iloc[cutoff - 31 : cutoff + 15].timestamp) <= benchmark_dates:
                continue
            expected.append((symbol, day.isoformat()))
    assert len(expected) == len(source) > 100
    assert len(source.ranges) < len(source)  # Only compact ranges, never expanded inputs.
    config = ExperimentConfig.from_yaml(config_path)
    hashes = []
    for seed, batch, workers, prefetch in ((42, 7, 1, 2), (901, 31, 2, 3)):
        config.training.seed = seed
        plan = replace(
            plan_dataloader_workers(
                workers,
                visible_cpu_count=8,
                available_memory_bytes=32 * 1024**3,
            ),
            prefetch_factor=prefetch,
        )
        loader = DataLoader(
            source,
            batch_size=batch,
            sampler=evaluation_sampler(len(source), config, split),
            drop_last=False,
            collate_fn=FinancialBatchCollator(),
            **_loader_process_options(plan, persistent=False),
        )
        actual, sizes = [], []
        for values in loader:
            actual.extend(zip(values["symbols"], values["cutoff_at"], strict=True))
            sizes.append(len(values["symbols"]))
        assert actual == expected and len(set(actual)) == len(actual)
        assert sizes[-1] == (len(source) % batch or batch)
        symbols, dates = zip(*actual, strict=True)
        hashes.append(sample_membership(list(symbols), list(dates)))
    assert hashes[0] == hashes[1]


def test_legacy_sample_cap_cannot_silently_sample_evaluation(config_path):
    config = ExperimentConfig.from_yaml(config_path)
    config.training.evaluation_max_samples = 32
    with pytest.warns(UserWarning, match="retired"):
        sampler = evaluation_sampler(73, config, "test")
    assert list(sampler) == list(range(73))


def test_prefetch_uses_inference_speed_and_shared_memory_budget(config_path):
    config = ExperimentConfig.from_yaml(config_path)
    workers = plan_dataloader_workers(4, visible_cpu_count=8, available_memory_bytes=32 * 1024**3)
    plan = replace(_explicit_runtime_batch_plan(config), seconds_per_training_batch=2.0)
    measured = BatchProbeMeasurement(
        batch_size=plan.evaluation_batch_size,
        seconds_per_batch=0.01,
        samples_per_second=100,
        peak_allocated_bytes=1024,
        projected_peak_bytes=1024,
        accepted=True,
        outcome="accepted",
    )
    args = dict(config=config, largest_host_batch_bytes=1024**2, shared_memory_bytes=1024**3)
    slow = plan_runtime_prefetch(workers, batch_plan=plan, **args)
    fast = plan_runtime_prefetch(
        workers, batch_plan=replace(plan, evaluation_probe=(measured,)), **args
    )
    assert fast.prefetch_factor > slow.prefetch_factor
    limited = plan_runtime_prefetch(
        workers,
        batch_plan=plan,
        config=config,
        largest_host_batch_bytes=1024**2,
        shared_memory_bytes=16 * 1024**2,
    )
    assert limited.estimated_peak_prefetch_memory_bytes <= 8 * 1024**2
    with pytest.raises(ValueError, match="Insufficient"):
        plan_runtime_prefetch(
            workers,
            batch_plan=plan,
            config=config,
            largest_host_batch_bytes=1024**2,
            shared_memory_bytes=1,
        )


def test_duplicate_ranges_fail_before_loading_windows(fixed_store):
    path = fixed_store / "cutoff-ranges.parquet"
    ranges = pd.read_parquet(path)
    position = ranges.index[ranges.split == "validation"][0]
    ranges.loc[position, "stop_index"] += 1
    duplicate = ranges.loc[[position]].copy()
    ranges = pd.concat([ranges, duplicate]).sort_values(["split", "symbol", "start_index"])
    ranges.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="overlap"):
        LazyFinancialWindowDataset(fixed_store, split="validation", window_size=32)
