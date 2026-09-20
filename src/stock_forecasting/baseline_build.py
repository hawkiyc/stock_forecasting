"""Full-data baseline jobs with persistent artifacts and bounded experiment parallelism."""

from __future__ import annotations

import json
import math
import multiprocessing
import os
import pickle
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
from torch.utils.data import DataLoader

from stock_forecasting.baseline_contract import runtime_contract, validate_complete
from stock_forecasting.baseline_storage import (
    SIGNAL_NAMES,
    build_tabular_cache,
    lazy_dataset,
    loader_options,
    open_tabular,
)
from stock_forecasting.baselines import (
    RULE_BASELINE_NAMES,
    CausalGRUBaseline,
    DLinearBaseline,
    PatchTSTBaseline,
    _normalized_pinball_torch,
    _seed_baseline,
)
from stock_forecasting.config import ExperimentConfig
from stock_forecasting.data.dataset import (
    BlockwisePermutationSampler,
    FinancialBatchCollator,
    FixedSizeBatchSampler,
)
from stock_forecasting.data.manifest import atomic_write_json, sha256_file
from stock_forecasting.evaluation_store import CHUNK_ROWS, META_DTYPE, EvaluationStore, Moments
from stock_forecasting.optimization_policy import ValidationPlateauScheduler
from stock_forecasting.runtime_resources import detect_available_memory, detect_visible_cpu_count


def _save_torch(path: Path, payload):
    temporary = path.with_suffix(".pending")
    torch.save(payload, temporary)
    temporary.replace(path)


def _save_pickle(path: Path, payload):
    temporary = path.with_suffix(".pending")
    with temporary.open("wb") as stream:
        pickle.dump(payload, stream, protocol=5)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _sequence(batch, device):
    return torch.stack((batch["asset_series"], batch["benchmark_series"]), dim=1).to(
        device, non_blocking=True
    )


def _loader(config, split, workers, batch_size, *, sampler=None, prefetch=2, validated_root=None):
    source = lazy_dataset(config, split, relative=True, validated_root=validated_root)
    options = loader_options(workers, prefetch)
    if sampler is not None:
        return DataLoader(
            source,
            batch_sampler=sampler,
            collate_fn=FinancialBatchCollator(),
            pin_memory=True,
            **options,
        )
    return DataLoader(
        source,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=FinancialBatchCollator(),
        pin_memory=True,
        **options,
    )


@torch.inference_mode()
def _neural_validation(model, loader, horizons, scales):
    model.eval()
    moments = Moments(horizons, scales)
    for batch in loader:
        prediction = model(_sequence(batch, "cuda")).float().cpu().numpy()
        moments.add(batch["target_alpha"].numpy(), prediction)
    model.train()
    return moments.result()["aggregate"]["selection_score"]


@torch.inference_mode()
def _neural_test(model, loader, output, horizons, scales):
    model.eval()
    store = EvaluationStore(output, len(loader.dataset), horizons, scales)
    try:
        for batch in loader:
            store.append(
                batch["target_alpha"].numpy(),
                model(_sequence(batch, "cuda")).float().cpu().numpy(),
                symbols=batch["symbols"],
                dates=batch["cutoff_at"],
                markets=batch["markets"],
                asset_types=batch["asset_types"],
                providers=batch["providers"],
            )
        return store.finish()
    finally:
        store.close()


def _train_neural(config, directory, name, seed, parameters, scales, plan):
    from stock_forecasting.training import ResumableFixedSizeBatchSampler

    _seed_baseline(seed)
    torch.cuda.set_per_process_memory_fraction(plan["gpu_fraction_per_job"])
    horizons = tuple(config.data.alpha_horizons)
    model = {
        "gru": lambda: CausalGRUBaseline(horizons=horizons),
        "dlinear": lambda: DLinearBaseline(config.data.input_length, horizons=horizons),
        "patchtst": lambda: PatchTSTBaseline(horizons=horizons),
    }[name]().cuda()
    from stock_forecasting.training_paths import resolve_bar_store_path

    source_root = (
        Path(plan["validated_bar_store_root"])
        if plan.get("validated_bar_store_root")
        else resolve_bar_store_path(config.data.bar_store_path)
    )
    count = len(lazy_dataset(config, "train", validated_root=source_root))
    sampler = ResumableFixedSizeBatchSampler(
        FixedSizeBatchSampler(
            BlockwisePermutationSampler(count, seed=seed, block_size=128),
            batch_size=parameters["batch_size"],
        )
    )
    train = _loader(
        config,
        "train",
        plan["loader_workers"],
        parameters["batch_size"],
        sampler=sampler,
        prefetch=plan["prefetch_factor"],
        validated_root=source_root,
    )
    validation = _loader(
        config,
        "validation",
        plan["loader_workers"],
        parameters["batch_size"],
        prefetch=plan["prefetch_factor"],
        validated_root=source_root,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=parameters["learning_rate"], weight_decay=parameters["weight_decay"]
    )
    schedule = ValidationPlateauScheduler(
        optimizer,
        math.ceil(len(train) * parameters["epochs"] * parameters["warmup_ratio"]),
        patience=parameters["plateau_patience_evaluations"],
        factor=parameters["plateau_factor"],
        min_ratio=parameters["plateau_min_ratio"],
        low_lr_evaluations=parameters["plateau_min_low_lr_evaluations"],
    )
    best, stale, start_epoch, start_batch, finished = math.inf, 0, 0, 0, False
    last = directory / "resume.pt"
    if last.is_file():
        # Only artifacts created in this hash-scoped, private network-volume directory are loaded.
        state = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        schedule.load_state_dict(state["scheduler"])
        best, stale, start_epoch, start_batch, finished = (
            state[k] for k in ("best", "stale", "epoch", "next_batch", "finished")
        )
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state(state["cuda_rng"])
    scale_tensor = torch.tensor(scales, device="cuda")
    boundaries = {
        math.ceil(len(train) * i / parameters["evaluations_per_epoch"])
        for i in range(1, parameters["evaluations_per_epoch"] + 1)
    }
    for epoch in range(start_epoch, parameters["epochs"]):
        if finished:
            break
        first = start_batch if epoch == start_epoch else 0
        sampler.set_epoch(epoch, start_batch_index=first)
        model.train()
        for position, batch in enumerate(train, first + 1):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(_sequence(batch, "cuda"))
            loss = _normalized_pinball_torch(
                prediction, batch["target_alpha"].cuda(non_blocking=True), scale_tensor
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            schedule.step()
            if position not in boundaries:
                continue
            score = _neural_validation(model, validation, list(horizons), scales)
            improved = score < best - parameters["early_stopping_min_delta"]
            best, stale = (score, 0) if improved else (best, stale + 1)
            if improved:
                _save_torch(
                    directory / "model.pt",
                    {
                        "state_dict": model.state_dict(),
                        "name": name,
                        "horizons": horizons,
                        "context_length": config.data.input_length,
                        "scales": scales,
                        "seed": seed,
                    },
                )
            schedule.observe(score, parameters["early_stopping_min_delta"])
            finished = (
                stale >= parameters["early_stopping_patience_evaluations"]
                and epoch + 1 >= parameters["early_stopping_start_epoch"]
                and schedule.permits_early_stop
            )
            next_epoch, next_batch = (epoch + 1, 0) if position == len(train) else (epoch, position)
            _save_torch(
                last,
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": schedule.state_dict(),
                    "best": best,
                    "stale": stale,
                    "epoch": next_epoch,
                    "next_batch": next_batch,
                    "finished": finished,
                    "python_rng": random.getstate(),
                    "numpy_rng": np.random.get_state(),
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state(),
                },
            )
            atomic_write_json(
                directory / "progress.json",
                {
                    "epoch": epoch + 1,
                    "batch": position,
                    "full_validation_score": score,
                    "best_score": best,
                    "stale_evaluations": stale,
                    "learning_rates": schedule.get_last_lr(),
                    "early_stopped": finished,
                },
            )
            print(
                f"{name}/{seed} epoch={epoch + 1} batch={position} full_validation={score:.8f}",
                flush=True,
            )
            if finished:
                break
    best_state = torch.load(directory / "model.pt", map_location="cuda", weights_only=True)
    model.load_state_dict(best_state["state_dict"])
    validation_metrics = _neural_test(
        model, validation, directory / "validation", list(horizons), scales
    )
    atomic_write_json(directory / "validation-metrics.json", validation_metrics)
    test = _loader(
        config,
        "test",
        plan["loader_workers"],
        parameters["batch_size"],
        prefetch=plan["prefetch_factor"],
        validated_root=source_root,
    )
    metrics = _neural_test(model, test, directory / "test", list(horizons), scales)
    return metrics


def _tabular_score(arrays, predict, horizons, scales, output: Path | None = None):
    moments = Moments(horizons, scales)
    store = (
        None
        if output is None
        else EvaluationStore(output, len(arrays["targets"]), horizons, scales)
    )
    try:
        for start in range(0, len(arrays["targets"]), CHUNK_ROWS):
            stop = min(start + CHUNK_ROWS, len(arrays["targets"]))
            predictions = np.sort(predict(start, stop), axis=-1)
            targets = arrays["targets"][start:stop]
            if store is None:
                moments.add(targets, predictions)
            else:
                meta = arrays["metadata"][start:stop]
                store.append(
                    targets,
                    predictions,
                    symbols=meta["symbol"],
                    dates=meta["date"],
                    markets=meta["market"],
                    asset_types=meta["asset_type"],
                    providers=meta["provider"],
                )
        return moments.result() if store is None else store.finish()
    finally:
        if store is not None:
            store.close()


def _train_gbdt(root, directory, seed, parameters, horizons, scales):
    arrays = {
        split: open_tabular(root / "inputs", split) for split in ("train", "validation", "test")
    }
    settings = parameters["gbdt"]
    models = [
        [
            HistGradientBoostingRegressor(
                loss="quantile",
                quantile=q,
                learning_rate=settings["learning_rate"],
                max_iter=1,
                max_leaf_nodes=settings["max_leaf_nodes"],
                l2_regularization=settings["l2_regularization"],
                early_stopping=False,
                warm_start=True,
                random_state=seed,
            )
            for q in (0.1, 0.5, 0.9)
        ]
        for _ in horizons
    ]
    best, stale, start_iteration, finished = math.inf, 0, 0, False
    resume = directory / "resume.pkl"
    if resume.is_file():
        with resume.open("rb") as stream:
            models, best, stale, start_iteration, finished = pickle.load(stream)

    def prediction(split, start, stop):
        return np.stack(
            [
                np.column_stack([m.predict(arrays[split]["features"][start:stop]) for m in row])
                for row in models
            ],
            axis=1,
        )

    for iteration in range(
        start_iteration + settings["validation_every"],
        settings["max_iter"] + 1,
        settings["validation_every"],
    ):
        if finished:
            break
        # Native OpenMP uses only this job's assigned CPU budget; no nested process fan-out.
        for h, row in enumerate(models):
            for model in row:
                model.set_params(max_iter=iteration)
                model.fit(arrays["train"]["features"], arrays["train"]["targets"][:, h])
        score = _tabular_score(
            arrays["validation"], lambda a, b: prediction("validation", a, b), horizons, scales
        )["aggregate"]["selection_score"]
        improved = score < best - parameters["early_stopping_min_delta"]
        best, stale = (score, 0) if improved else (best, stale + 1)
        if improved:
            _save_pickle(
                directory / "model.pkl",
                {"models": models, "horizons": horizons, "scales": scales, "seed": seed},
            )
        finished = stale >= parameters["early_stopping_patience_evaluations"]
        _save_pickle(resume, (models, best, stale, iteration, finished))
        atomic_write_json(
            directory / "progress.json",
            {
                "iteration": iteration,
                "full_validation_score": score,
                "best_score": best,
                "stale_evaluations": stale,
                "early_stopped": finished,
            },
        )
        print(f"gbdt/{seed} trees={iteration} full_validation={score:.8f}", flush=True)
    with (directory / "model.pkl").open("rb") as stream:
        models = pickle.load(stream)["models"]
    validation_metrics = _tabular_score(
        arrays["validation"],
        lambda a, b: prediction("validation", a, b),
        horizons,
        scales,
        directory / "validation",
    )
    atomic_write_json(directory / "validation-metrics.json", validation_metrics)
    return _tabular_score(
        arrays["test"], lambda a, b: prediction("test", a, b), horizons, scales, directory / "test"
    )


def _train_rules(root, directory, horizons, scales):
    train = open_tabular(root / "inputs", "train")
    validation = open_tabular(root / "inputs", "validation")
    test = open_tabular(root / "inputs", "test")
    result = {}
    # Quantiles use the complete train column, bounded to one horizon/rule at a time.
    for name in RULE_BASELINE_NAMES:
        child = directory / name
        child.mkdir(exist_ok=True)
        artifact = child / "result.json"
        if artifact.is_file():
            result[name] = json.loads(artifact.read_text())
            continue
        offsets = np.empty((len(horizons), 3))
        locations = np.zeros(len(horizons))
        for i, horizon in enumerate(horizons):
            targets = np.asarray(train["targets"][:, i], dtype=np.float64)
            if name == "always_buy":
                locations[i] = max(float(np.median(targets)), 0.25 * scales[i])
                residual = targets - locations[i]
            elif name == "zero_return":
                residual = targets
            else:
                residual = targets - train["signals"][:, SIGNAL_NAMES.index(name)] * (horizon / 5)
            offsets[i] = np.quantile(residual, [0.1, 0.5, 0.9], overwrite_input=True)
        atomic_write_json(
            child / "model.json",
            {
                "rule": name,
                "horizons": horizons,
                "locations": locations.tolist(),
                "residual_quantiles": offsets.tolist(),
                "scales": scales,
            },
        )

        def prediction(arrays, start, stop, name=name, locations=locations, offsets=offsets):
            if name in ("always_buy", "zero_return"):
                medians = np.broadcast_to(locations, (stop - start, len(horizons)))
            else:
                medians = arrays["signals"][start:stop, SIGNAL_NAMES.index(name), None] * (
                    np.asarray(horizons)[None, :] / 5
                )
            return medians[..., None] + offsets

        validation_metrics = _tabular_score(
            validation,
            lambda a, b: prediction(validation, a, b),
            horizons,
            scales,
            child / "validation",
        )
        atomic_write_json(child / "validation-metrics.json", validation_metrics)
        metrics = _tabular_score(
            test, lambda a, b: prediction(test, a, b), horizons, scales, child / "test"
        )
        result[name] = {
            "state": "complete",
            "source": "prebuilt_full_train_rule",
            "metrics": metrics,
        }
        atomic_write_json(artifact, result[name])
    return result


def run_job(config_payload, root_string, name, seed, parameters, scales, plan):
    root = Path(root_string)
    directory = root / "jobs" / (name if seed is None else f"{name}-{seed}")
    directory.mkdir(parents=True, exist_ok=True)
    execution = {"pid": os.getpid(), "started_at": time.time(), "job": name, "seed": seed}
    atomic_write_json(directory / "execution.json", execution)
    try:
        torch.set_num_threads(plan["cpu_threads"] if name == "gbdt" else 1)
        config = ExperimentConfig.model_validate(config_payload)
        with threadpool_limits(limits=plan["cpu_threads"] if name == "gbdt" else 1):
            if name == "inputs":
                payload = build_tabular_cache(
                    config,
                    root / "inputs",
                    workers=plan["input_workers"],
                    prefetch=plan["prefetch_factor"],
                    validated_root=Path(plan["validated_bar_store_root"]),
                )
            elif name == "rules":
                payload = _train_rules(root, directory, list(config.data.alpha_horizons), scales)
            elif name == "gbdt":
                payload = _train_gbdt(
                    root, directory, seed, parameters, list(config.data.alpha_horizons), scales
                )
            else:
                payload = _train_neural(config, directory, name, seed, parameters, scales, plan)
        atomic_write_json(
            directory / "complete.json",
            {"state": "complete", "name": name, "seed": seed, "result": payload},
        )
    except BaseException as error:
        atomic_write_json(directory / "failure.json", {"error": f"{type(error).__name__}: {error}"})
        raise
    finally:
        execution["finished_at"] = time.time()
        execution["peak_gpu_bytes"] = (
            torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0
        )
        atomic_write_json(directory / "execution.json", execution)


def resource_plan(parameters, train_count, horizons, context_length=128):
    settings = parameters["resources"]
    for key in (
        "max_gpu_experiments",
        "max_cpu_experiments",
        "gpu_bytes_per_experiment",
        "host_bytes_per_gpu_experiment",
        "worker_bytes",
        "max_loader_workers_per_experiment",
        "prefetch_factor",
        "timeout_seconds",
    ):
        if type(settings[key]) is not int or settings[key] < 1:
            raise ValueError(f"Baseline resource limit must be a positive integer: {key}")
    for key in ("gpu_memory_fraction", "host_memory_fraction"):
        if not 0 < settings[key] <= 0.85:
            raise ValueError(
                f"Baseline resource fraction must reserve at least 15% headroom: {key}"
            )
    memory = detect_available_memory().available_bytes
    cpus = detect_visible_cpu_count()
    free_gpu, total_gpu = torch.cuda.mem_get_info()
    gpu_slots = min(
        settings["max_gpu_experiments"],
        int(free_gpu * settings["gpu_memory_fraction"]) // settings["gpu_bytes_per_experiment"],
        # Reserve two control cores plus an input parent and at least one worker.
        max(0, (cpus - 4) // 3),
        int(memory * 0.3) // settings["host_bytes_per_gpu_experiment"],
    )
    if gpu_slots < 1:
        raise MemoryError("Insufficient CPU/host/GPU budget for one safely bounded neural baseline")
    if gpu_slots < 2:
        print(
            "Baseline admission: hardware budget allows only one GPU experiment concurrently",
            flush=True,
        )
    host_budget = int(memory * settings["host_memory_fraction"])
    gbdt_bytes = train_count * (30 * 12 + len(horizons) * 8 + 64) * 2 + 2 * 1024**3
    if gbdt_bytes > host_budget:
        raise MemoryError(
            f"Full GBDT requires estimated {gbdt_bytes} host bytes; budget is {host_budget}. "
            "Choose a larger-memory Pod; never subsample."
        )
    workers = min(
        settings["max_loader_workers_per_experiment"],
        max(1, (cpus - 4 - gpu_slots) // (2 * gpu_slots)),
        max(1, (host_budget // (gpu_slots * 2)) // settings["worker_bytes"]),
    )
    shared_memory = shutil.disk_usage("/dev/shm").free
    batch_bytes = parameters["batch_size"] * 2 * context_length * 5 * 4
    if workers * gpu_slots * settings["prefetch_factor"] * batch_bytes * 4 > shared_memory * 0.5:
        workers = int(shared_memory * 0.5) // (
            gpu_slots * settings["prefetch_factor"] * batch_bytes * 4
        )
    if workers < 1:
        raise MemoryError("Insufficient shared memory for baseline multiprocessing loaders")
    if workers == 1:
        print(
            f"Baseline loader admission: one worker per loader; cpus={cpus}, "
            f"host_budget={host_budget}, shm_free={shared_memory}",
            flush=True,
        )
    cpu_budget = cpus - gpu_slots * (2 * workers + 1) - 2
    cpu_slots = min(settings["max_cpu_experiments"], cpu_budget)
    input_workers = min(workers, cpu_budget - 1)
    if cpu_slots < 1 or input_workers < 1:
        raise MemoryError("Insufficient CPU budget for concurrent neural and input workers")
    return {
        "gpu_slots": gpu_slots,
        "cpu_slots": cpu_slots,
        "visible_cpus": cpus,
        "available_host_bytes": memory,
        "available_gpu_bytes": free_gpu,
        "shared_memory_bytes": shared_memory,
        "input_workers": input_workers,
        "input_job_host_bytes": input_workers * settings["worker_bytes"] + 2 * 1024**3,
        "prefetch_factor": settings["prefetch_factor"],
        "loader_workers": workers,
        "cpu_threads": cpu_budget // cpu_slots,
        "host_budget": host_budget,
        "gbdt_bytes": gbdt_bytes,
        "gpu_job_host_bytes": max(
            settings["host_bytes_per_gpu_experiment"],
            2 * workers * settings["worker_bytes"] + 2 * 1024**3,
        ),
        "gpu_fraction_per_job": settings["gpu_memory_fraction"] * free_gpu / total_gpu / gpu_slots,
        "timeout_seconds": settings["timeout_seconds"],
    }


def build_baselines(config):
    from stock_forecasting.training import plan_dataloader_workers, resolve_runtime_robust_scales
    from stock_forecasting.validation_benchmark import _aggregate_numeric

    project, identity = runtime_contract()
    parameters = json.loads((project / "configs/baseline.json").read_text())
    from stock_forecasting.baseline_contract import validate_optimization_alignment

    validate_optimization_alignment(config, parameters)
    root = (
        Path(os.environ.get("NETWORK_VOLUME_ROOT", "/runpod-volume"))
        / "baselines"
        / identity["baseline_id"]
    )
    root.mkdir(parents=True, exist_ok=True)
    complete = root / "complete.json"
    if complete.is_file():
        payload = json.loads(complete.read_text())
        validate_complete(payload, identity)
        return payload
    train = lazy_dataset(config, "train")
    plan = resource_plan(
        parameters, len(train), config.data.alpha_horizons, config.data.input_length
    )
    plan["validated_bar_store_root"] = str(train.root)
    counts = json.loads((train.root / "bar-store.json").read_text())["split_counts"]
    horizon_count = len(config.data.alpha_horizons)
    evaluation_count = counts["validation"] + counts["test"]
    learned_count = 4 * len(parameters["seeds"])
    expected_disk = (
        sum(counts.values()) * (36 + horizon_count) * 4
        + evaluation_count * META_DTYPE.itemsize
        + evaluation_count
        * (horizon_count * 16 + META_DTYPE.itemsize)
        * (len(RULE_BASELINE_NAMES) + learned_count)
        + (learned_count + 2) * 1024**3
    )
    existing_disk = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    remaining_disk = max(0, expected_disk - existing_disk)
    if shutil.disk_usage(root).free < remaining_disk:
        raise OSError(f"Complete baseline artifacts need approximately {remaining_disk} more bytes")
    plan["estimated_remaining_disk_bytes"] = remaining_disk
    atomic_write_json(root / "resource-plan.json", plan)
    calibration = resolve_runtime_robust_scales(
        train,
        sample_count=parameters["label_scale_calibration_samples"],
        seed=parameters["calibration_seed"],
        worker_plan=plan_dataloader_workers(plan["loader_workers"], source="baseline"),
    )
    scales = np.asarray(calibration.scales, dtype=np.float32).tolist()
    jobs = [
        ("inputs", None),
        ("rules", None),
        *(
            (name, seed)
            for seed in parameters["seeds"]
            for name in ("gbdt", "gru", "dlinear", "patchtst")
        ),
    ]
    waiting = [
        job
        for job in jobs
        if not (
            root / "jobs" / (job[0] if job[1] is None else f"{job[0]}-{job[1]}") / "complete.json"
        ).is_file()
    ]
    running = []
    context = multiprocessing.get_context("spawn")
    # Bound BLAS in spawned imports, before each library constructs thread pools.
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"
    try:
        while waiting or running:
            for job in list(waiting):
                if (
                    job[0] in ("rules", "gbdt")
                    and not (root / "jobs/inputs/complete.json").is_file()
                ):
                    continue
                gpu = job[0] not in ("inputs", "rules", "gbdt")
                memory = (
                    plan["gpu_job_host_bytes"]
                    if gpu
                    else plan["gbdt_bytes"]
                    if job[0] == "gbdt"
                    else plan["input_job_host_bytes"]
                    if job[0] == "inputs"
                    else max(2 * 1024**3, len(train) * 24)
                )
                if sum(r[2] == gpu for r in running) >= plan["gpu_slots" if gpu else "cpu_slots"]:
                    continue
                if sum(r[3] for r in running) + memory > plan["host_budget"]:
                    continue
                process = context.Process(
                    target=run_job,
                    args=(config.as_dict(), str(root), *job, parameters, scales, plan),
                    name=f"baseline-{job[0]}-{job[1]}",
                )
                process.start()
                running.append((process, time.monotonic(), gpu, memory))
                waiting.remove(job)
            if waiting and not running:
                raise MemoryError("No pending baseline job fits the resource plan")
            for row in list(running):
                process, started, _, _ = row
                if time.monotonic() - started > plan["timeout_seconds"]:
                    raise TimeoutError(
                        f"Baseline job exceeded its bounded deadline: {process.name}"
                    )
                if process.exitcode is not None:
                    process.join()
                    if process.exitcode:
                        raise RuntimeError(
                            f"Baseline child failed: {process.name}, exit={process.exitcode}"
                        )
                    process.close()
                    running.remove(row)
            time.sleep(1)
    finally:
        for process, *_ in running:
            if process.is_alive():
                process.terminate()
            process.join(timeout=10)
            if process.is_alive():
                process.kill()
                process.join(timeout=10)
            process.close()
    models = {}
    for name, seed in jobs:
        path = root / "jobs" / (name if seed is None else f"{name}-{seed}") / "complete.json"
        result = json.loads(path.read_text())["result"]
        if name == "inputs":
            cache = result
            continue
        if name == "rules":
            models.update(result)
        else:
            models.setdefault(
                name, {"state": "complete", "source": "prebuilt_full_data", "seed_results": {}}
            )["seed_results"][str(seed)] = result
    for result in models.values():
        if "seed_results" in result:
            result["aggregate"] = _aggregate_numeric(list(result["seed_results"].values()))
    membership = models["zero_return"]["metrics"]["sample_membership"]
    validation_membership = None
    for name in models:
        directories = (
            [root / "jobs" / f"{name}-{seed}" for seed in parameters["seeds"]]
            if name not in RULE_BASELINE_NAMES
            else [root / "jobs/rules" / name]
        )
        for directory in directories:
            metrics = json.loads((directory / "validation-metrics.json").read_text())
            current = metrics["sample_membership"]
            if (
                metrics["samples"] != cache["counts"]["validation"]
                or metrics["evaluation_robust_scales"] != scales
            ):
                raise ValueError("Baseline validation population or calibration is incomplete")
            if validation_membership is not None and current != validation_membership:
                raise ValueError("Baseline jobs used different validation populations")
            validation_membership = current
    for result in models.values():
        for metrics in (result.get("seed_results") or {"single": result["metrics"]}).values():
            if (
                metrics["sample_membership"] != membership
                or metrics["samples"] != cache["counts"]["test"]
            ):
                raise ValueError("Baseline jobs used different or incomplete test populations")
    # Persist weights, calibration and predictions; publish completion only after all jobs succeed.
    from concurrent.futures import ThreadPoolExecutor

    files = sorted(
        path
        for path in (root / "jobs").rglob("*")
        if path.is_file() and path.name not in ("resume.pt", "resume.pkl")
    )

    def artifact(path):
        return path.relative_to(root).as_posix(), {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    with ThreadPoolExecutor(max_workers=min(4, max(1, detect_visible_cpu_count() // 2))) as pool:
        artifacts = dict(pool.map(artifact, files))
    payload = {
        "state": "complete",
        "identity": identity,
        "models": models,
        "sample_counts": cache["counts"],
        "evaluation_membership": membership,
        "robust_scales": scales,
        "data_identity": cache["identity"],
        "validation_membership": validation_membership,
        "calibration_identity_sha256": calibration.identity_sha256,
        "artifacts": artifacts,
    }
    validate_complete(payload, identity)
    atomic_write_json(complete, payload)
    return payload
