"""Reusable memory-mapped tabular inputs; neural windows stay lazy in the bar store."""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from stock_forecasting.baselines import (
    BaselineBatchDataset,
    _relative_rule_signals,
    collate_baseline_batch,
)
from stock_forecasting.data.dataset import LazyFinancialWindowDataset
from stock_forecasting.data.manifest import atomic_write_json, sha256_file
from stock_forecasting.evaluation_store import META_DTYPE
from stock_forecasting.training_paths import resolve_bar_store_path

SIGNAL_NAMES = ("momentum_5d", "reversal_5d", "ma_crossover", "rsi", "macd", "volatility_scaled")


def worker_init(_worker_id):
    torch.set_num_threads(1)
    import pyarrow

    pyarrow.set_cpu_count(1)
    pyarrow.set_io_thread_count(1)


def loader_options(workers: int, prefetch: int = 2, *, persistent: bool = False) -> dict:
    if workers < 1:
        raise ValueError("Production baseline loaders require at least one worker")
    return {
        "num_workers": workers,
        "prefetch_factor": prefetch,
        "persistent_workers": persistent,
        "multiprocessing_context": "spawn",
        "worker_init_fn": worker_init,
        "timeout": 300,
    }


def lazy_dataset(config, split: str, *, relative: bool = False, validated_root: Path | None = None):
    root = (
        validated_root
        if validated_root is not None
        else resolve_bar_store_path(config.data.bar_store_path)
    )
    return LazyFinancialWindowDataset(
        root,
        split=split,
        window_size=config.data.input_length,
        h_start=config.data.h_start,
        series_mode="relative" if relative else "raw",
        symbol_cache_size=4,
    )


def build_tabular_cache(
    config,
    root: Path,
    *,
    workers: int,
    batch_size: int = 256,
    prefetch: int = 2,
    validated_root: Path | None = None,
    checkpoint_seconds: float = 300,
    progress_seconds: float = 30,
) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    source_root = (
        validated_root
        if validated_root is not None
        else resolve_bar_store_path(config.data.bar_store_path)
    )
    manifest_sha = sha256_file(source_root / "bar-store.json")
    identity = {
        "manifest_sha256": manifest_sha,
        "horizons": list(config.data.alpha_horizons),
        "schema": 2,
    }
    done = root / "complete.json"
    if done.is_file():
        existing = json.loads(done.read_text())
        if existing["identity"] != identity:
            raise ValueError("Baseline input cache does not match the immutable bar store")
        for split, count in existing["counts"].items():
            validate_tabular(root, split, count, len(config.data.alpha_horizons))
        return existing
    counts = {}
    for split in ("train", "validation", "test"):
        source = lazy_dataset(config, split, validated_root=source_root)
        count = len(source)
        counts[split] = count
        directory = root / split
        directory.mkdir(exist_ok=True)
        split_done = directory / "complete.json"
        if split_done.is_file() and json.loads(split_done.read_text()) == {
            "identity": identity,
            "count": count,
        }:
            validate_tabular(root, split, count, len(source.horizons))
            continue
        shapes = {
            "features": (count, 30),
            "targets": (count, len(source.horizons)),
            "signals": (count, 6),
        }
        progress_path = directory / "resume.json"
        offset = 0
        if progress_path.is_file():
            progress = json.loads(progress_path.read_text())
            if progress["identity"] != identity or progress["count"] != count:
                raise ValueError("Baseline partial input cache has a different data contract")
            offset = progress["committed_rows"]
            if type(offset) is not int or not 0 <= offset <= count:
                raise ValueError("Invalid baseline input resume cursor")
            validate_tabular(root, split, count, len(source.horizons))
        required = sum(np.prod(shape) * 4 for shape in shapes.values())
        if split != "train":
            required += count * META_DTYPE.itemsize
        if (
            shutil.disk_usage(root).free
            < (0 if progress_path.is_file() else required) + 2 * 1024**3
        ):
            raise OSError(f"Baseline {split} disk cache needs {required} bytes plus 2 GiB headroom")
        arrays = {
            name: np.lib.format.open_memmap(
                directory / f"{name}.npy",
                mode="r+" if progress_path.is_file() else "w+",
                dtype="float32",
                shape=shape,
            )
            for name, shape in shapes.items()
        }
        if split != "train":
            arrays["metadata"] = np.lib.format.open_memmap(
                directory / "metadata.npy",
                mode="r+" if progress_path.is_file() else "w+",
                dtype=META_DTYPE,
                shape=(count,),
            )
        _commit_tabular_progress(directory, arrays, identity, count, offset)
        loader = DataLoader(
            BaselineBatchDataset(source),
            batch_size=batch_size,
            sampler=range(offset, count),
            collate_fn=collate_baseline_batch,
            **loader_options(workers, prefetch),
        )
        iterator = iter(loader)
        saved_at = reported_at = time.monotonic()
        reported_offset = offset
        print(
            f"Baseline input cache {split}: resume={offset}/{count} batch={batch_size}", flush=True
        )
        try:
            for batch in iterator:
                stop = offset + len(batch.targets)
                arrays["features"][offset:stop] = batch.features
                arrays["targets"][offset:stop] = batch.targets
                signals = _relative_rule_signals(batch)
                arrays["signals"][offset:stop] = np.column_stack(
                    [signals[name] for name in SIGNAL_NAMES]
                )
                if "metadata" in arrays:
                    for name, values in (
                        ("symbol", batch.symbols),
                        ("date", batch.dates),
                        ("market", batch.markets),
                        ("asset_type", batch.asset_types),
                        ("provider", batch.providers),
                    ):
                        if any(len(str(v)) > META_DTYPE[name].itemsize // 4 for v in values):
                            raise ValueError(f"Baseline metadata would be truncated: {name}")
                        arrays["metadata"][name][offset:stop] = values
                offset = stop
                now = time.monotonic()
                if now - reported_at >= progress_seconds:
                    rate = (offset - reported_offset) / max(now - reported_at, 1e-9)
                    print(
                        f"Baseline input cache {split}: {offset}/{count} windows/s={rate:.1f}",
                        flush=True,
                    )
                    reported_at, reported_offset = now, offset
                if now - saved_at >= checkpoint_seconds:
                    _commit_tabular_progress(directory, arrays, identity, count, offset)
                    saved_at = time.monotonic()
            if offset != count:
                raise ValueError("Baseline input cache did not visit the entire split")
            _commit_tabular_progress(directory, arrays, identity, count, offset)
        finally:
            iterator._shutdown_workers()
            for array in arrays.values():
                array.flush()
                array._mmap.close()
        atomic_write_json(split_done, {"identity": identity, "count": count})
    payload = {"identity": identity, "counts": counts}
    atomic_write_json(done, payload)
    return payload


def _commit_tabular_progress(directory, arrays, identity, count, offset):
    """Publish a cursor only after every corresponding mmap is durable."""
    for name, array in arrays.items():
        array.flush()
        with (directory / f"{name}.npy").open("rb") as stream:
            os.fsync(stream.fileno())
    atomic_write_json(
        directory / "resume.json",
        {
            "identity": identity,
            "count": count,
            "committed_rows": offset,
        },
    )


def open_tabular(root: Path, split: str) -> dict:
    names = ["features", "targets", "signals"] + ([] if split == "train" else ["metadata"])
    return {name: np.load(root / split / f"{name}.npy", mmap_mode="r") for name in names}


def validate_tabular(root: Path, split: str, count: int, horizons: int) -> None:
    shapes = {"features": (count, 30), "targets": (count, horizons), "signals": (count, 6)}
    if split != "train":
        shapes["metadata"] = (count,)
    arrays = open_tabular(root, split)
    try:
        for name, shape in shapes.items():
            expected_dtype = META_DTYPE if name == "metadata" else np.dtype("float32")
            if arrays[name].shape != shape or arrays[name].dtype != expected_dtype:
                raise ValueError(f"Invalid baseline cache: {split}/{name}")
    finally:
        for array in arrays.values():
            array._mmap.close()
