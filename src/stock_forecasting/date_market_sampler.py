"""Runtime-only date/market index; the prepared bar-store remains immutable."""

from __future__ import annotations

import multiprocessing
import os
from pathlib import Path

import numpy as np
import pandas as pd

from stock_forecasting.data.dataset import BlockwisePermutationSampler
from stock_forecasting.data.manifest import atomic_write_json, canonical_json_sha256, sha256_file
from stock_forecasting.runtime_resources import detect_available_memory, detect_visible_cpu_count

_SOURCE = None
INDEX_DTYPE = np.dtype([("group", "i8"), ("ordinal", "i8")])


def _initialize(source):
    global _SOURCE
    import pyarrow
    import torch

    torch.set_num_threads(1)
    pyarrow.set_cpu_count(1)
    pyarrow.set_io_thread_count(1)
    source.symbol_cache_size = min(source.symbol_cache_size, 4)
    _SOURCE = source


def _range_rows(task):
    row, offset, size, market_code = task
    frame = _SOURCE._load_symbol(str(row["symbol"]))
    start = int(row["start_index"])
    days = (
        pd.DatetimeIndex(frame["timestamp"].iloc[start : start + size]).as_unit("ns").asi8
        // 86_400_000_000_000
    )
    result = np.empty(size, dtype=INDEX_DTYPE)
    result["group"] = days * 256 + market_code
    result["ordinal"] = np.arange(offset, offset + size)
    return offset, result


def date_market_index(source, *, requested_workers: int) -> Path:
    identity = {
        "version": 1,
        "manifest": sha256_file(source.root / "bar-store.json"),
        "split": source.split,
        "count": len(source),
        "window": source.window_size,
        "horizons": list(source.horizons),
    }
    root = source.root.parent / "training-cache" / "date-market"
    root.mkdir(parents=True, exist_ok=True)
    key = canonical_json_sha256(identity)
    path, marker = root / f"{key}.npy", root / f"{key}.json"
    if marker.is_file() and path.is_file():
        import json

        if json.loads(marker.read_text()) == identity:
            existing = np.load(path, mmap_mode="r")
            try:
                if existing.shape != (len(source),) or existing.dtype != INDEX_DTYPE:
                    raise ValueError("Runtime ranking index is truncated or incompatible")
            finally:
                existing._mmap.close()
            return path
    memory = detect_available_memory().available_bytes
    workers = min(
        requested_workers,
        max(1, detect_visible_cpu_count() - 2),
        max(1, int(memory * 0.2) // (512 * 1024**2)),
    )
    if workers < 2:
        print("Date/market index: memory/CPU budget permits only one worker", flush=True)
    import shutil

    if shutil.disk_usage(root).free < len(source) * INDEX_DTYPE.itemsize + 1024**3:
        raise OSError("Insufficient disk for the bounded runtime ranking index")
    temporary = path.with_suffix(f".pending-{os.getpid()}.npy")
    index = np.lib.format.open_memmap(temporary, mode="w+", dtype=INDEX_DTYPE, shape=(len(source),))
    market_names = sorted({str(v["market"]) for v in source._index.values()})
    if len(market_names) > 256:
        raise ValueError("Ranking index supports at most 256 market identities")

    def tasks():
        previous = 0
        for position, row in enumerate(source.ranges.to_dict("records")):
            end = int(source._range_ends[position])
            yield (
                row,
                previous,
                end - previous,
                market_names.index(str(source._index[str(row["symbol"])]["market"])),
            )
            previous = end

    pending = []
    iterator = iter(tasks())
    try:
        # Pool context termination cancels and joins workers even after a timeout.
        with multiprocessing.get_context("spawn").Pool(
            workers, initializer=_initialize, initargs=(source,)
        ) as pool:
            exhausted = False
            while pending or not exhausted:
                while not exhausted and len(pending) < 2 * workers:
                    task = next(iterator, None)
                    if task is None:
                        exhausted = True
                    else:
                        pending.append(pool.apply_async(_range_rows, (task,)))
                if not pending:
                    break
                offset, rows = pending.pop(0).get(timeout=300)
                index[offset : offset + len(rows)] = rows
        # In-place native sort avoids a second full permutation in each worker.
        index.sort(order=["group", "ordinal"], kind="quicksort")
        index.flush()
    finally:
        index._mmap.close()
    temporary.replace(path)
    atomic_write_json(marker, identity)
    return path


class DateMarketSampler(BlockwisePermutationSampler):
    def __init__(self, source, *, requested_workers: int, **kwargs):
        self.index_path = date_market_index(source, requested_workers=requested_workers)
        super().__init__(len(source), **kwargs)

    def __iter__(self):
        index = np.load(self.index_path, mmap_mode="r")
        try:
            for position in super().__iter__():
                yield int(index["ordinal"][position])
        finally:
            index._mmap.close()
