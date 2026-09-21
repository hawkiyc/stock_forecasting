"""Bounded per-experiment CUDA tuning and sample-exact baseline resumption."""

from __future__ import annotations

import math
import time
from itertools import islice

import torch
from torch.utils.data import DataLoader

from stock_forecasting.baseline_storage import loader_options
from stock_forecasting.data.dataset import BlockwisePermutationSampler, FinancialBatchCollator


class SampleCursorBatchSampler:
    """Keep random epoch order and validation boundaries independent of batch size."""

    def __init__(self, count, batch_size, seed, evaluations):
        if min(count, batch_size, evaluations) < 1:
            raise ValueError("Sample count, batch size and evaluations must be positive")
        self.sampler = BlockwisePermutationSampler(count, seed=seed, block_size=128)
        self.batch_size = batch_size
        self.boundaries = sorted(
            {math.ceil(count * i / evaluations) for i in range(1, evaluations + 1)}
        )
        self.count, self.cursor = count, 0

    def set_epoch(self, epoch, cursor=0):
        if not 0 <= cursor <= self.count:
            raise ValueError("Baseline resume cursor is outside the training population")
        self.sampler.set_epoch(epoch)
        self.cursor = cursor

    def __iter__(self):
        position = self.cursor
        iterator = islice(iter(self.sampler), position, None)
        for boundary in self.boundaries:
            while position < boundary:
                size = min(self.batch_size, boundary - position)
                batch = list(islice(iterator, size))
                if len(batch) != size:
                    raise RuntimeError("Baseline sampler exhausted before its exact boundary")
                yield batch
                position += size

    def __len__(self):
        previous, batches = self.cursor, 0
        for boundary in self.boundaries:
            if boundary > previous:
                batches += math.ceil((boundary - previous) / self.batch_size)
                previous = boundary
        return batches


def bounded_prefetch(*, workers, batch_bytes, host_bytes, shared_bytes, desired, maximum):
    """Account for two live pools, IPC copies and pinned host batches."""
    per_depth = 2 * workers * batch_bytes * 4
    capacity = min(int(host_bytes * 0.5), int(shared_bytes * 0.5)) // max(1, per_depth)
    if capacity < 1:
        raise MemoryError("Baseline batch cannot fit one safely bounded prefetch per worker")
    return max(1, min(maximum, desired, capacity))


def _cuda_probe(model, context, horizons, *, training, maximum, budget, repetitions):
    rows = []
    size = min(32, maximum)
    optimizer_reserve = sum(p.numel() * p.element_size() * 2 for p in model.parameters())
    original_mode = model.training
    # Tuning does not update weights, optimizer state or training RNG streams.
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        try:
            model.train(training)
            while size <= maximum:
                if (
                    rows
                    and rows[-1].get("projected_peak_bytes", 0) * size / rows[-1]["batch_size"]
                    > budget
                ):
                    break
                x = prediction = loss = None
                try:
                    torch.cuda.reset_peak_memory_stats()
                    x = torch.randn(size, 2, context, 5, device="cuda")
                    timings = []
                    for iteration in range(repetitions + 1):
                        model.zero_grad(set_to_none=True)
                        begin, end = (
                            torch.cuda.Event(enable_timing=True),
                            torch.cuda.Event(enable_timing=True),
                        )
                        begin.record()
                        with torch.set_grad_enabled(training):
                            prediction = model(x)
                            if training:
                                loss = prediction.square().mean()
                                loss.backward()
                        end.record()
                        end.synchronize()
                        if iteration:
                            timings.append(begin.elapsed_time(end) / 1000)
                        prediction = loss = None
                    peak = torch.cuda.max_memory_allocated() + optimizer_reserve
                    seconds = sum(timings) / len(timings)
                    rows.append(
                        {
                            "batch_size": size,
                            "seconds_per_batch": seconds,
                            "samples_per_second": size / max(seconds, 1e-9),
                            "projected_peak_bytes": peak,
                            "accepted": peak <= budget,
                        }
                    )
                    if peak > budget:
                        break
                except torch.cuda.OutOfMemoryError:
                    rows.append({"batch_size": size, "accepted": False, "outcome": "oom"})
                    break
                finally:
                    x = prediction = loss = None
                    model.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                if size == maximum:
                    break
                size = min(size * 2, maximum)
        finally:
            model.train(original_mode)
    accepted = [row for row in rows if row["accepted"]]
    if not accepted:
        raise MemoryError("No baseline batch fits the assigned concurrent GPU memory budget")
    best = max(row["samples_per_second"] for row in accepted)
    selected = next(row for row in accepted if row["samples_per_second"] >= best * 0.95)
    return selected, rows


def tune_baseline_runtime(model, source, parameters, plan):
    settings = parameters["resources"]
    workers = plan["loader_workers"]
    manual = parameters["batch_size"]
    result = {
        "loader_workers": workers,
        "training_batch_size": manual,
        "evaluation_batch_size": manual,
        "prefetch_factor": plan["prefetch_factor"],
        "auto_batch": settings.get("auto_batch", True),
    }
    if not result["auto_batch"]:
        return result
    slots = plan.get("gpu_slots", 1)
    host = plan.get("gpu_job_host_bytes", 4 * 1024**3) - 2 * workers * settings["worker_bytes"]
    shared = plan.get("shared_memory_bytes")
    if shared is None:
        import shutil

        shared = shutil.disk_usage("/dev/shm").free
    shared //= slots
    per_sample = 2 * source.window_size * 5 * (4 + 8) + len(source.horizons) * 4
    # Pre-admit batch memory before allocating CUDA tensors or spawning workers.
    host_maximum = max(0, min(int(host * 0.5), int(shared * 0.5)) // (8 * workers * per_sample))
    if host_maximum < 1:
        raise MemoryError("No host/shared-memory budget remains for baseline auto batching")
    gpu_budget = int(
        torch.cuda.get_device_properties(0).total_memory * plan["gpu_fraction_per_job"] * 0.8
    )
    selected = {}
    for mode, training, limit in (
        ("training", True, "max_training_batch_size"),
        ("evaluation", False, "max_evaluation_batch_size"),
    ):
        chosen, probes = _cuda_probe(
            model,
            source.window_size,
            source.horizons,
            training=training,
            maximum=min(
                settings.get(limit, 2048 if training else 4096),
                host_maximum,
                len(source) if training else settings.get(limit, 4096),
            ),
            budget=gpu_budget,
            repetitions=settings.get("batch_probe_repetitions", 3),
        )
        result[f"{mode}_batch_size"] = chosen["batch_size"]
        result[f"{mode}_probe"] = probes
        selected[mode] = chosen
    # Measure the real, dynamically sampled input path in a bounded worker pool.
    sampler = SampleCursorBatchSampler(len(source), result["training_batch_size"], 59, 1)
    batches = list(islice(iter(sampler), settings.get("loader_probe_batches", 5)))
    loader = DataLoader(
        source,
        batch_sampler=batches,
        collate_fn=FinancialBatchCollator(),
        **loader_options(workers, 1),
    )
    iterator = iter(loader)
    timings = []
    try:
        for i in range(len(batches)):
            start = time.monotonic()
            next(iterator)
            if i:
                timings.append(time.monotonic() - start)
    finally:
        iterator._shutdown_workers()
    input_seconds = sum(timings) / max(1, len(timings))
    compute_seconds = min(v["seconds_per_batch"] for v in selected.values())
    maximum_prefetch = settings.get("max_prefetch_factor", 4)
    desired = max(
        plan["prefetch_factor"], math.ceil(input_seconds / max(compute_seconds, 1e-6)) + 1
    )
    result["prefetch_factor"] = bounded_prefetch(
        workers=workers,
        batch_bytes=max(result["training_batch_size"], result["evaluation_batch_size"])
        * per_sample,
        host_bytes=host,
        shared_bytes=shared,
        desired=desired,
        maximum=maximum_prefetch,
    )
    result.update(
        input_seconds_per_batch=input_seconds,
        gpu_budget_bytes=gpu_budget,
        host_buffer_budget_bytes=host,
        shared_memory_budget_bytes=shared,
    )
    return result
