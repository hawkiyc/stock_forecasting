#!/usr/bin/env python3
"""Bounded real-bar throughput/equivalence probes; never build production baselines."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import resource
import time
from itertools import islice
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from stock_forecasting.baseline_build import _close_loader, _sequence, resource_plan
from stock_forecasting.baseline_runtime import SampleCursorBatchSampler, tune_baseline_runtime
from stock_forecasting.baseline_storage import lazy_dataset, loader_options, worker_init
from stock_forecasting.baselines import (
    BaselineBatchDataset,
    BaselineRecordDataset,
    CausalGRUBaseline,
    DLinearBaseline,
    PatchTSTBaseline,
    baseline_arrays,
    baseline_arrays_from_windows,
    collate_baseline_batch,
)
from stock_forecasting.config import ExperimentConfig
from stock_forecasting.data.dataset import BlockwisePermutationSampler, FinancialBatchCollator
from stock_forecasting.data.manifest import atomic_write_json, sha256_file
from stock_forecasting.runtime_resources import detect_cpu_quota, detect_visible_cpu_count
from stock_forecasting.training_paths import resolve_bar_store_path


def input_probe(config, root, plan, output, *, legacy, rows, barrier=None):
    worker_init(0)
    source = lazy_dataset(config, "train", validated_root=root)
    dataset = BaselineRecordDataset(source) if legacy else BaselineBatchDataset(source)
    loader = DataLoader(
        dataset,
        batch_size=256,
        sampler=range(rows),
        collate_fn=baseline_arrays if legacy else collate_baseline_batch,
        **loader_options(plan["input_workers"], plan["prefetch_factor"]),
    )
    iterator = iter(loader)
    count = 0
    start = measured_start = time.monotonic()
    try:
        for batch in iterator:
            if not count:
                if barrier is not None:
                    barrier.wait(timeout=180)
                measured_start = time.monotonic()
            count += len(batch.targets)
        elapsed = time.monotonic() - measured_start
    finally:
        iterator._shutdown_workers()
    assert count == rows
    payload = {
        "rows": rows,
        "seconds": time.monotonic() - start,
        "steady_windows_per_second": (rows - min(256, rows)) / max(elapsed, 1e-9),
        "legacy": legacy,
        "workers": plan["input_workers"],
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    }
    atomic_write_json(output, payload)
    return payload


def neural_probe(config_payload, root, parameters, plan, name, output, barrier, rows):
    worker_init(0)
    config = ExperimentConfig.model_validate(config_payload)
    torch.cuda.set_per_process_memory_fraction(plan["gpu_fraction_per_job"])
    source = lazy_dataset(config, "train", relative=True, validated_root=Path(root))
    model = {
        "gru": lambda: CausalGRUBaseline(horizons=source.horizons),
        "dlinear": lambda: DLinearBaseline(source.window_size, horizons=source.horizons),
        "patchtst": lambda: PatchTSTBaseline(horizons=source.horizons),
    }[name]().cuda()
    runtime = tune_baseline_runtime(model, source, parameters, plan)
    atomic_write_json(Path(output).with_suffix(".runtime.json"), runtime)
    sampler = SampleCursorBatchSampler(len(source), runtime["training_batch_size"], 42, 5)
    bounded = list(islice(iter(sampler), max(1, rows // runtime["training_batch_size"])))
    loader = DataLoader(
        source,
        batch_sampler=bounded,
        collate_fn=FinancialBatchCollator(),
        pin_memory=True,
        **loader_options(plan["loader_workers"], runtime["prefetch_factor"], persistent=True),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    iterator = iter(loader)
    first = next(iterator)
    barrier.wait(timeout=180)
    started, wait_seconds, compute_seconds, count = time.monotonic(), 0.0, 0.0, 0
    model.train()
    try:
        for i in range(len(bounded)):
            begin = time.monotonic()
            batch = first if i == 0 else next(iterator)
            wait_seconds += time.monotonic() - begin
            begin = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            prediction = model(_sequence(batch, "cuda"))
            loss = (
                (prediction[:, :, 1] - batch["target_alpha"].cuda(non_blocking=True))
                .square()
                .mean()
            )
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            compute_seconds += time.monotonic() - begin
            count += len(prediction)
        ended = time.monotonic()
        atomic_write_json(
            Path(output),
            {
                "name": name,
                "pid": os.getpid(),
                "rows": count,
                "started_monotonic": started,
                "ended_monotonic": ended,
                "seconds": ended - started,
                "windows_per_second": count / (ended - started),
                "input_wait_seconds": wait_seconds,
                "compute_seconds": compute_seconds,
                "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
                "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                "runtime": runtime,
            },
        )
    finally:
        _close_loader(loader)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=16384)
    args = parser.parse_args()
    if not os.environ.get("RUNPOD_POD_ID") or not 1024 <= args.rows <= 65536:
        raise ValueError("Only bounded authorized cloud probes are supported")
    worker_init(0)
    args.output.mkdir(parents=True, exist_ok=True)
    config = ExperimentConfig.from_yaml(
        os.environ.get("RUNPOD_CONFIG", "configs/stage2_kronos_base_lora.yaml")
    )
    root = resolve_bar_store_path(config.data.bar_store_path)
    before = sha256_file(root / "bar-store.json")
    source = lazy_dataset(config, "train", validated_root=root)
    parameters = json.loads(Path("configs/baseline.json").read_text())
    plan = resource_plan(parameters, len(source), list(source.horizons), source.window_size)
    if plan["gpu_slots"] < 2:
        raise MemoryError("The concurrency probe requires admission for two GPU experiments")
    atomic_write_json(args.output / "resource-plan.json", plan)
    indices = list(
        islice(iter(BlockwisePermutationSampler(len(source), seed=42, block_size=16)), 128)
    )
    oracle = baseline_arrays(source.record_at(i) for i in indices)
    actual = baseline_arrays_from_windows(source.array_batch(indices))
    np.testing.assert_array_equal(actual.targets, oracle.targets)
    np.testing.assert_allclose(actual.features, oracle.features, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(actual.sequences, oracle.sequences, rtol=1e-5, atol=2e-6)
    assert actual.symbols == oracle.symbols and actual.dates == oracle.dates
    old = input_probe(
        config, root, plan, args.output / "input-before.json", legacy=True, rows=args.rows
    )
    new = input_probe(
        config, root, plan, args.output / "input-after.json", legacy=False, rows=args.rows
    )
    print(json.dumps({"input_before": old, "input_after": new}), flush=True)
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(3)
    jobs = [
        context.Process(
            target=neural_probe,
            args=(
                config.model_dump(mode="json"),
                str(root),
                parameters,
                plan,
                name,
                str(args.output / f"{name}.json"),
                barrier,
                args.rows * 2,
            ),
        )
        for name in ("gru", "dlinear")
    ]
    # Input production runs concurrently with the two admitted GPU experiments.
    jobs.append(
        context.Process(
            target=input_probe,
            args=(config, root, plan, args.output / "input-concurrent.json"),
            kwargs={"legacy": False, "rows": args.rows * 2, "barrier": barrier},
        )
    )
    try:
        for job in jobs:
            job.start()
        deadline = time.monotonic() + 360
        for job in jobs:
            job.join(timeout=max(0, deadline - time.monotonic()))
            if job.is_alive() or job.exitcode != 0:
                raise RuntimeError(
                    f"Concurrent throughput job failed: pid={job.pid} exit={job.exitcode}"
                )
    finally:
        for job in jobs:
            if job.is_alive():
                job.terminate()
            if job.pid:
                job.join(timeout=15)
    # Cover the transformer auto-tuner as well without admitting a third GPU job.
    neural_probe(
        config.model_dump(mode="json"),
        str(root),
        parameters,
        plan,
        "patchtst",
        str(args.output / "patchtst.json"),
        context.Barrier(1),
        args.rows,
    )
    assert sha256_file(root / "bar-store.json") == before
    summary = {
        "quota_cpus": detect_cpu_quota(),
        "usable_cpus": detect_visible_cpu_count(),
        "input_speedup": new["steady_windows_per_second"] / old["steady_windows_per_second"],
        "numeric_equivalence": True,
        "prepared_data_unchanged": True,
        "manifest_sha256": before,
        "jobs": {
            name: json.loads((args.output / f"{name}.json").read_text())
            for name in ("gru", "dlinear", "patchtst")
        },
    }
    atomic_write_json(args.output / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
