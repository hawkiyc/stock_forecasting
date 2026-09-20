#!/usr/bin/env python3
"""Cloud-only synthetic full-population reduction; never reads market data."""

from __future__ import annotations

import argparse
import json
import os
import resource
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from stock_forecasting.evaluation_store import CHUNK_ROWS, EvaluationStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=1_264_861)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("RUNPOD_POD_ID") or not 1 <= args.rows <= 2_000_000:
        raise ValueError("Only a bounded authorized cloud capacity test is supported")
    rng = np.random.default_rng(43)
    started = time.monotonic()
    store = EvaluationStore(args.output, args.rows, list(range(1, 15)), [0.03] * 14)
    days = [(date(2025, 12, 1) + timedelta(days=i)).isoformat() for i in range(200)]
    try:
        for start in range(0, args.rows, CHUNK_ROWS):
            stop = min(start + CHUNK_ROWS, args.rows)
            count = stop - start
            target = rng.normal(0, 0.03, (count, 14)).astype("float32")
            prediction = np.sort(rng.normal(0, 0.03, (count, 14, 3)), axis=-1).astype("float32")
            # Canonical symbol-major order mirrors the real lazy bar store.
            store.append(
                target,
                prediction,
                symbols=[f"S{i // 125}" for i in range(start, stop)],
                dates=[days[i % 125] for i in range(start, stop)],
                markets=["US"] * count,
                asset_types=["stock"] * count,
                providers=["synthetic"] * count,
            )
        metrics = store.finish()
        assert metrics["samples"] == args.rows
        summary = {
            "samples": args.rows,
            "seconds": time.monotonic() - started,
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "disk_bytes": sum(path.stat().st_size for path in args.output.glob("*.npy")),
            "membership": metrics["sample_membership"],
            "aggregate": metrics["aggregate"],
        }
        (args.output / "capacity.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary), flush=True)
    finally:
        store.close()


if __name__ == "__main__":
    main()
