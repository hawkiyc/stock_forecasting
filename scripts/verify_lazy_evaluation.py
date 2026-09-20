#!/usr/bin/env python3
"""Audit all real evaluation cutoffs without materializing windows or calling providers."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from stock_forecasting.baseline_storage import worker_init
from stock_forecasting.config import ExperimentConfig
from stock_forecasting.data.dataset import LazyFinancialWindowDataset
from stock_forecasting.runtime_resources import detect_available_memory, detect_visible_cpu_count
from stock_forecasting.training_paths import resolve_bar_store_path

_SOURCE = None
_RANGES = None
_BOUNDARIES = None


def initialize(root, window_size, boundaries):
    global _SOURCE, _RANGES, _BOUNDARIES
    worker_init(0)
    _SOURCE = LazyFinancialWindowDataset(
        root, split="validation", window_size=window_size, h_start=1, symbol_cache_size=4
    )
    ranges = pd.read_parquet(Path(root) / "cutoff-ranges.parquet")
    _RANGES = {key: rows for key, rows in ranges.groupby(["symbol", "split"], sort=False)}
    _BOUNDARIES = boundaries


def audit_symbol(symbol):
    source = _SOURCE
    frame = source._load_symbol(symbol)
    benchmark = source._load_symbol(source._index[symbol]["benchmark_symbol"])
    dates = pd.DatetimeIndex(frame.timestamp)
    n, width = len(frame), source.window_size + 14
    valid = np.zeros(n, dtype=bool)
    if n >= width:
        # Independent sliding sums, not the preparation mask or stored sample counts.
        missing = ~dates.isin(pd.DatetimeIndex(benchmark.timestamp))
        coverage = np.convolve(missing.astype("int64"), np.ones(width, dtype="int64"), "valid")
        bad = frame.adjusted_transition_extreme.to_numpy(dtype="int64")
        transitions = np.convolve(bad[1:], np.ones(width - 1, dtype="int64"), "valid")
        valid[source.window_size - 1 : n - 14] = (coverage == 0) & (transitions == 0)
    result = {}
    for split, lower, upper in _BOUNDARIES:
        accepted = valid & (dates >= pd.Timestamp(lower, tz="UTC"))
        indices = np.flatnonzero(accepted)
        expected = indices[dates[indices + 14] < pd.Timestamp(upper, tz="UTC")]
        rows = _RANGES.get((symbol, split))
        actual = (
            np.concatenate(
                [np.arange(int(row.start_index), int(row.stop_index)) for row in rows.itertuples()]
            )
            if rows is not None
            else np.array([])
        )
        if not np.array_equal(actual, expected):
            raise AssertionError(f"Incomplete or invalid lazy cutoffs: {symbol}/{split}")
        result[split] = [dates[int(index)].isoformat() for index in expected]
    return symbol, result


def main():
    if not os.environ.get("RUNPOD_POD_ID"):
        raise ValueError("Real-data verification is cloud-only")
    config = ExperimentConfig.from_yaml("configs/stage2_kronos_base_lora.yaml")
    root = resolve_bar_store_path(config.data.bar_store_path)
    manifest_path = root / "bar-store.json"
    before = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    manifest = json.loads(manifest_path.read_text())
    index = pd.read_parquet(root / "symbol-index.parquet")
    symbols = sorted(index.loc[index.eligible, "symbol"].astype(str))
    fixed = config.data.fixed_split
    boundaries = [
        ("validation", fixed["train_end"], fixed["validation_end"]),
        ("test", fixed["validation_end"], fixed["test_end"]),
    ]
    workers = min(
        int(os.environ.get("FIN_TS_AUDIT_WORKERS", "4")),
        max(1, detect_visible_cpu_count() - 2),
        int(detect_available_memory().available_bytes * 0.25) // (512 * 1024**2),
    )
    if workers < 1:
        raise MemoryError("Insufficient audit worker memory")
    started = time.monotonic()
    digests = {split: hashlib.sha256(b"[") for split, _, _ in boundaries}
    counts = dict.fromkeys(digests, 0)
    products = dict.fromkeys(digests, 0)
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=initialize,
        initargs=(str(root), config.data.input_length, boundaries),
    ) as pool:
        pending, remaining = deque(), iter(symbols)
        for _ in range(workers * 2):
            symbol = next(remaining, None)
            if symbol is not None:
                pending.append(pool.submit(audit_symbol, symbol))
        processed = 0
        while pending:
            symbol, result = pending.popleft().result(timeout=180)
            for split, dates in result.items():
                products[split] += bool(dates)
                for day in dates:
                    if counts[split]:
                        digests[split].update(b",")
                    digests[split].update(
                        json.dumps(
                            [symbol, day], separators=(",", ":"), ensure_ascii=False
                        ).encode()
                    )
                    counts[split] += 1
            processed += 1
            if processed % 1000 == 0:
                print(f"Audited instruments: {processed}/{len(symbols)}", flush=True)
            symbol = next(remaining, None)
            if symbol is not None:
                pending.append(pool.submit(audit_symbol, symbol))
    for split, digest in digests.items():
        digest.update(b"]")
        assert counts[split] == manifest["split_counts"][split]
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == before
    summary = {
        "counts": counts,
        "instruments": products,
        "audited_eligible_instruments": len(symbols),
        "membership_sha256": {name: value.hexdigest() for name, value in digests.items()},
        "workers": workers,
        "seconds": time.monotonic() - started,
        "bar_store_manifest_sha256": before,
        "bar_store_unchanged": True,
        "windows_materialized": False,
    }
    output = Path(os.environ["NETWORK_VOLUME_ROOT"]) / "diagnostics/full-workflow"
    output = output / os.environ["WANDB_RUN_ID"] / "lazy-evaluation-audit.json"
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
