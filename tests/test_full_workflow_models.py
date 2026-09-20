"""Small synthetic regression checks for the full-data performance release."""

from __future__ import annotations

import json
import multiprocessing
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from stock_forecasting.baseline_build import _train_gbdt, _train_neural, _train_rules
from stock_forecasting.baseline_storage import SIGNAL_NAMES, build_tabular_cache, validate_tabular
from stock_forecasting.config import ExperimentConfig
from stock_forecasting.evaluation_protocol import evaluation_sampler, sample_membership
from stock_forecasting.evaluation_store import META_DTYPE, EvaluationStore, Moments
from stock_forecasting.metrics import multi_horizon_alpha_metrics
from stock_forecasting.models.forecast import MultiHorizonAlphaHead
from stock_forecasting.models.ranking import market_ids, ranking_groups, same_date_ranking_loss
from stock_forecasting.models.scale_features import (
    fit_scale_feature_statistics,
    historical_scale_features,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("cpus", [7, 8, 12, 16, 32])
def test_baseline_cpu_budget_includes_parents_and_concurrent_inputs(monkeypatch, cpus):
    from types import SimpleNamespace

    import stock_forecasting.baseline_build as build

    parameters = json.loads((ROOT / "configs/baseline.json").read_text())
    monkeypatch.setattr(build, "detect_visible_cpu_count", lambda: cpus)
    monkeypatch.setattr(
        build, "detect_available_memory", lambda: SimpleNamespace(available_bytes=64 * 1024**3)
    )
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (24 * 1024**3, 24 * 1024**3))
    monkeypatch.setattr(build.shutil, "disk_usage", lambda path: SimpleNamespace(free=16 * 1024**3))
    plan = build.resource_plan(parameters, 1_000, list(range(1, 15)))
    gpu_cores = plan["gpu_slots"] * (2 * plan["loader_workers"] + 1)
    assert gpu_cores + 2 + plan["cpu_slots"] * plan["cpu_threads"] <= cpus
    assert gpu_cores + 2 + plan["input_workers"] + 1 <= cpus
    assert plan["input_workers"] >= 1 and plan["loader_workers"] >= 1


@pytest.fixture
def small_lazy_config(tmp_path, market_frame):
    from stock_forecasting.data.bar_store import build_symbol_bar_store
    from stock_forecasting.data.manifest import artifact_metadata

    raw = tmp_path / "market.parquet"
    market_frame.to_parquet(raw, index=False)
    store = tmp_path / "bar-store"
    build_symbol_bar_store(
        raw_path=raw,
        output_root=store,
        download_manifest={
            "artifacts": {"raw": artifact_metadata(raw, root=tmp_path, row_count=len(market_frame))}
        },
        window_size=32,
        bucket_count=4,
        batch_rows=200,
        purge_bars=20,
        embargo_bars=14,
    )
    config = ExperimentConfig.from_yaml(ROOT / "configs/local_mock.yaml")
    config.data.bar_store_path = store
    config.data.input_length = 32
    return config


def test_runtime_date_index_preserves_full_population(small_lazy_config):
    from stock_forecasting.baseline_storage import lazy_dataset
    from stock_forecasting.date_market_sampler import DateMarketSampler, date_market_index

    source = lazy_dataset(small_lazy_config, "train")
    path = date_market_index(source, requested_workers=2)
    rows = np.load(path)
    assert sorted(rows["ordinal"].tolist()) == list(range(len(source)))
    assert np.all(rows["group"][1:] >= rows["group"][:-1])
    for group in np.unique(rows["group"])[::23]:
        records = [source.record_at(int(i)) for i in rows["ordinal"][rows["group"] == group]]
        assert len({(r["cutoff_at"], r["metadata"]["market"]) for r in records}) == 1
    sampler = DateMarketSampler(source, requested_workers=2, seed=42, block_size=128)
    assert sorted(sampler) == list(range(len(source)))
    assert date_market_index(source, requested_workers=2) == path


def test_full_tabular_preparation_reuses_without_touching_bar_store(small_lazy_config, tmp_path):
    from stock_forecasting.data.manifest import sha256_file

    manifest = small_lazy_config.data.bar_store_path / "bar-store.json"
    before = sha256_file(manifest)
    root = tmp_path / "inputs"
    result = build_tabular_cache(small_lazy_config, root, workers=2, batch_size=16)
    assert set(result["counts"]) == {"train", "validation", "test"}
    assert all(value > 0 for value in result["counts"].values())
    assert build_tabular_cache(small_lazy_config, root, workers=2) == result
    assert sha256_file(manifest) == before


def test_neural_validation_reuses_and_closes_its_worker_pool(small_lazy_config):
    from stock_forecasting.baseline_build import _close_loader, _loader

    loader = _loader(small_lazy_config, "validation", 1, 16, persistent=True)
    try:
        first = sum(len(batch["symbols"]) for batch in loader)
        processes = list(loader._iterator._workers)
        second = sum(len(batch["symbols"]) for batch in loader)
        assert first == second == len(loader.dataset)
        assert [p.pid for p in processes] == [p.pid for p in loader._iterator._workers]
    finally:
        _close_loader(loader)
    assert loader._iterator is None
    assert all(not p.is_alive() for p in processes)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires authorized cloud CUDA")
def test_neural_baseline_full_splits_and_resume(small_lazy_config, tmp_path):
    parameters = json.loads((ROOT / "configs/baseline.json").read_text())
    parameters.update(epochs=1, batch_size=16, evaluations_per_epoch=2)
    plan = {"loader_workers": 2, "prefetch_factor": 2, "gpu_fraction_per_job": 0.35}
    directory = tmp_path / "neural"
    directory.mkdir()
    scales = [0.03] * len(small_lazy_config.data.alpha_horizons)
    first = _train_neural(small_lazy_config, directory, "dlinear", 42, parameters, scales, plan)
    second = _train_neural(small_lazy_config, directory, "dlinear", 42, parameters, scales, plan)
    assert first == second
    assert (directory / "model.pt").is_file() and (directory / "resume.pt").is_file()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires authorized cloud CUDA")
def test_complete_baseline_builder_and_cache_reuse(small_lazy_config, tmp_path, monkeypatch):
    import stock_forecasting.baseline_build as build
    from stock_forecasting.baseline_contract import baseline_contract, validate_complete
    from stock_forecasting.baseline_storage import lazy_dataset

    config = small_lazy_config
    parameters = json.loads((ROOT / "configs/baseline.json").read_text())
    parameters.update(epochs=1, batch_size=32, label_scale_calibration_samples=32)
    parameters["gbdt"].update(max_iter=2, validation_every=1, max_leaf_nodes=3)
    config.data.label_scale_calibration_samples = 32
    config.training.learning_rate_schedule = "validation_plateau"
    config.training.evaluation_max_samples = None
    config.validation.baseline_max_samples_per_split = None
    config.validation.models = parameters["models"] + ["kronos_full"]
    for name in (
        "evaluations_per_epoch",
        "early_stopping_patience_evaluations",
        "early_stopping_min_delta",
        "early_stopping_start_epoch",
        "plateau_patience_evaluations",
        "plateau_factor",
        "plateau_min_ratio",
        "plateau_min_low_lr_evaluations",
    ):
        setattr(config.training, name, parameters[name])
    project = tmp_path / "project"
    (project / "configs").mkdir(parents=True)
    (project / "configs/baseline.json").write_text(json.dumps(parameters))
    identity = baseline_contract(ROOT, {"dataset_request": {"synthetic": True}})
    identity["contract"]["parameters"] = {k: v for k, v in parameters.items() if k != "resources"}
    identity["baseline_id"] = "baseline-synthetic-integration"
    monkeypatch.setattr(build, "runtime_contract", lambda: (project, identity))
    monkeypatch.setenv("NETWORK_VOLUME_ROOT", str(tmp_path))
    first = build.build_baselines(config)
    validate_complete(first, identity)
    assert first["sample_counts"] == {
        split: len(lazy_dataset(config, split)) for split in ("train", "validation", "test")
    }
    assert first["validation_membership"]["samples"] == first["sample_counts"]["validation"]
    root = tmp_path / "baselines" / identity["baseline_id"]
    execution = [json.loads(path.read_text()) for path in (root / "jobs").glob("*/execution.json")]
    neural = [row for row in execution if row["job"] in ("gru", "dlinear", "patchtst")]
    assert len({row["pid"] for row in neural}) == 3
    assert all(row["peak_gpu_bytes"] > 0 for row in neural)

    # A completed cache must not initialize another dataset, CUDA plan or worker.
    def forbidden(*args, **kwargs):
        pytest.fail("A complete baseline was rebuilt")

    monkeypatch.setattr(build, "lazy_dataset", forbidden)
    monkeypatch.setattr(build, "resource_plan", forbidden)
    assert build.build_baselines(config) == first


def test_main_testing_reuses_baseline_metrics_without_fit_or_inference(tmp_path, monkeypatch):
    import stock_forecasting.baseline_contract as contract
    import stock_forecasting.validation_benchmark as benchmark

    runner = benchmark.ValidationBenchmark.__new__(benchmark.ValidationBenchmark)
    runner.config = ExperimentConfig.from_yaml(ROOT / "configs/local_mock.yaml")
    runner.checkpoint = tmp_path
    runner.models = ["zero_return", "kronos_full"]
    runner.resume = True
    runner.recompute_full_model = True
    runner.defer_terminal_lifecycle = False
    runner.payload = {"models": {}}
    metrics = {
        "samples": 12,
        "sample_membership": {"samples": 12, "ordered_symbol_dates_sha256": "fixture"},
        "evaluation_robust_scales": [0.03],
        "daily_normalized_pinball": {"2026-01-02": 0.4},
        "aggregate": {"normalized_pinball": 0.4},
        "primary_5d": {"selection_score": 0.4},
    }
    cached = {
        "robust_scales": [0.03],
        "identity": {"baseline_id": "fixture"},
        "sample_counts": {"train": 30, "validation": 10, "test": 12},
        "evaluation_membership": metrics["sample_membership"],
        "models": {"zero_return": {"state": "complete", "metrics": metrics}},
    }
    (tmp_path / "trainer-state.json").write_text(json.dumps({"runtime_robust_scales": [0.03]}))
    monkeypatch.setattr(contract, "require_baselines", lambda config: cached)
    monkeypatch.setattr(runner, "_publish", lambda: None)
    monkeypatch.setattr(runner, "_publish_lifecycle", lambda *args: None)

    def forbidden(*args, **kwargs):
        pytest.fail("Baseline fitting or data preparation ran during main testing")

    monkeypatch.setattr(runner, "_run_learned_baseline", forbidden)
    monkeypatch.setattr(benchmark, "_lazy_baseline_arrays", forbidden)
    calls = []

    def evaluate(config, checkpoint, *, split):
        calls.append(split)
        return {"metrics": metrics}

    monkeypatch.setattr(benchmark, "evaluate_checkpoint", evaluate)
    assert runner._run_prebuilt()["state"] == "ready"
    assert calls == ["test"]
    runner._run_prebuilt()
    assert calls == ["test"]
    runner.payload["models"]["kronos_full"]["metrics"] = {
        **metrics,
        "sample_membership": {"samples": 11},
    }
    with pytest.raises(ValueError, match="different holdout samples"):
        runner._run_prebuilt()


def test_full_evaluation_visits_every_row_independent_of_seed():
    config = ExperimentConfig.from_yaml(ROOT / "configs/stage2_kronos_base_lora.yaml")
    for seed in (42, 97):
        config.training.seed = seed
        for split in ("validation", "test"):
            assert list(evaluation_sampler(20017, config, split)) == list(range(20017))


def test_streaming_metrics_equal_dense_metrics_and_membership(tmp_path):
    rng = np.random.default_rng(41)
    horizons, scales = list(range(1, 15)), [0.03] * 14
    y = rng.normal(0, 0.02, (61, 14)).astype("float32")
    q = np.sort(rng.normal(0, 0.03, (61, 14, 3)), axis=-1).astype("float32")
    symbols = [f"S{i % 9}" for i in range(61)]
    dates = [f"2026-01-{1 + i // 9:02d}" for i in range(61)]
    store = EvaluationStore(tmp_path, 61, horizons, scales)
    try:
        for start, stop in ((0, 3), (3, 29), (29, 61)):
            n = stop - start
            store.append(
                y[start:stop],
                q[start:stop],
                symbols=symbols[start:stop],
                dates=dates[start:stop],
                markets=["US"] * n,
                asset_types=["stock"] * n,
                providers=["fixture"] * n,
            )
        result = store.finish()
        dense = multi_horizon_alpha_metrics(
            targets=y,
            quantile_predictions=q,
            horizons=horizons,
            quantiles=[0.1, 0.5, 0.9],
            robust_scales=scales,
        )
        assert result["aggregate"] == pytest.approx(dense["aggregate"], abs=1e-12)
        for horizon in dense["per_horizon"]:
            assert result["per_horizon"][horizon] == pytest.approx(
                dense["per_horizon"][horizon], abs=1e-12
            )
        assert result["sample_membership"] == sample_membership(symbols, dates)
        assert result["subgroups"]["market"]["US"]["samples"] == 61
    finally:
        store.close()


def test_incomplete_duplicate_and_invalid_evaluation_fail_closed(tmp_path):
    with pytest.raises(ValueError, match="positive scale"):
        Moments([5], [0])
    store = EvaluationStore(tmp_path, 2, [5], [0.03])
    try:
        with pytest.raises(ValueError, match="Incomplete"):
            store.finish()
        store.append(
            np.zeros((2, 1)),
            np.zeros((2, 1, 3)),
            symbols=["A", "A"],
            dates=["2026-01-02"] * 2,
            markets=["US"] * 2,
            asset_types=["stock"] * 2,
            providers=["fixture"] * 2,
        )
        with pytest.raises(ValueError, match="Duplicate"):
            store.finish()
    finally:
        store.close()


def test_extended_features_masking_causality_and_output_scale():
    torch.manual_seed(43)
    history = torch.exp(torch.randn(8, 128, 5).cumsum(1) * 0.01 + 4)
    benchmark = torch.exp(torch.randn(8, 128, 5).cumsum(1) * 0.005 + 5)
    features = historical_scale_features(history, benchmark, extended=True)
    assert features.shape == (8, 20) and torch.isfinite(features).all()
    mask = torch.ones(8, 128, dtype=torch.bool)
    mask[:, -3:] = False
    history[:, -3:] = float("nan")
    benchmark[:, -3:] = float("nan")
    masked = historical_scale_features(history, benchmark, extended=True, mask=mask)
    torch.testing.assert_close(
        masked, historical_scale_features(history[:, :-3], benchmark[:, :-3], extended=True)
    )
    statistics = fit_scale_feature_statistics(features.numpy(), {"split": "train"})
    head = MultiHorizonAlphaHead(
        32,
        horizons=tuple(range(1, 15)),
        feature_mode="combined",
        market_aware=True,
        explicit_output_scale=True,
        robust_scales=[0.03] * 14,
    )
    head.numeric_branch.set_statistics(statistics)
    tokens = torch.randn(1, 4, 32).expand(2, -1, -1)
    scales = features[:1].expand(2, -1).clone()
    scales[1, 2] = scales[0, 2] * 2
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = head(
            tokens,
            scale_features=scales,
            benchmark_tokens=tokens,
            market_ids=market_ids(["US", "US"], "cpu"),
        )
    assert output.dtype == torch.float32
    torch.testing.assert_close(output[1], output[0] * 2)
    assert (output[..., 0] < output[..., 1]).all() and (output[..., 1] < output[..., 2]).all()
    output.square().mean().backward()
    assert head.scale_gate.weight.grad.abs().sum() > 0
    assert head.market_embedding.weight.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="market IDs"):
        head(tokens, scale_features=scales, benchmark_tokens=tokens)


def test_ranking_never_crosses_date_market_or_duplicate_security():
    groups, securities = ranking_groups(
        ["d1", "d2", "d1", "d1"], ["US", "US", "TWSE", "US"], ["A", "B", "C", "A"], "cpu"
    )
    prediction = torch.randn(4, 14, 3, requires_grad=True)
    target = torch.randn(4, 14)
    loss = same_date_ranking_loss(prediction, target, torch.ones(14), groups, securities, 256)
    assert loss.item() == 0
    groups, securities = ranking_groups(["d1", "d1"], ["US", "US"], ["A", "B"], "cpu")
    truth = torch.stack([torch.ones(14), -torch.ones(14)])
    good = truth[..., None].expand(-1, -1, 3).clone().requires_grad_()
    right = same_date_ranking_loss(good, truth, torch.ones(14), groups, securities, 256)
    wrong = same_date_ranking_loss(-good, truth, torch.ones(14), groups, securities, 256)
    assert right < wrong
    right.backward()
    assert torch.isfinite(good.grad).all()


def _tabular_fixture(root):
    rng = np.random.default_rng(4)
    for split, count in (("train", 35), ("validation", 11), ("test", 13)):
        directory = root / "inputs" / split
        directory.mkdir(parents=True)
        np.save(directory / "features.npy", rng.normal(size=(count, 30)).astype("float32"))
        np.save(directory / "targets.npy", rng.normal(0, 0.03, (count, 1)).astype("float32"))
        np.save(
            directory / "signals.npy",
            rng.normal(0, 0.01, (count, len(SIGNAL_NAMES))).astype("float32"),
        )
        if split != "train":
            meta = np.zeros(count, dtype=META_DTYPE)
            meta["symbol"] = [f"S{i}" for i in range(count)]
            meta["date"], meta["market"], meta["asset_type"], meta["provider"] = (
                "2026-01-02",
                "US",
                "stock",
                "fixture",
            )
            np.save(directory / "metadata.npy", meta)


def test_full_cpu_baselines_persist_weights_and_resume(tmp_path):
    _tabular_fixture(tmp_path)
    parameters = json.loads((ROOT / "configs/baseline.json").read_text())
    parameters["gbdt"].update(max_iter=2, validation_every=1, max_leaf_nodes=3)
    directory = tmp_path / "gbdt"
    directory.mkdir()
    result = _train_gbdt(tmp_path, directory, 42, parameters, [5], [0.03])
    assert result["samples"] == 13
    assert (directory / "model.pkl").is_file() and (directory / "resume.pkl").is_file()
    resumed = _train_gbdt(tmp_path, directory, 42, parameters, [5], [0.03])
    assert resumed == result
    rules = tmp_path / "rules"
    rules.mkdir()
    results = _train_rules(tmp_path, rules, [5], [0.03])
    assert all(v["metrics"]["samples"] == 13 for v in results.values())
    assert all((rules / name / "model.json").is_file() for name in results)
    validate_tabular(tmp_path / "inputs", "test", 13, 1)
    with pytest.raises(ValueError, match="Invalid baseline"):
        validate_tabular(tmp_path / "inputs", "test", 12, 1)


def _cuda_experiment(name, output, barrier):
    from stock_forecasting.baselines import CausalGRUBaseline, DLinearBaseline

    torch.set_num_threads(1)
    torch.manual_seed(42)
    torch.cuda.set_per_process_memory_fraction(0.35)
    model = (CausalGRUBaseline(hidden_dim=16) if name == "gru" else DLinearBaseline(128)).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    x = torch.randn(16, 2, 128, 5, device="cuda")
    barrier.wait(timeout=90)
    losses = []
    for _ in range(4):
        optimizer.zero_grad(set_to_none=True)
        loss = model(x).square().mean()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    Path(output).write_text(
        json.dumps(
            {"pid": os.getpid(), "losses": losses, "peak_bytes": torch.cuda.max_memory_allocated()}
        )
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Requires the authorized cloud CUDA smoke test"
)
def test_two_experiments_share_one_cuda_device(tmp_path):
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(2)
    processes = [
        context.Process(
            target=_cuda_experiment, args=(name, str(tmp_path / f"{name}.json"), barrier)
        )
        for name in ("gru", "dlinear")
    ]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=120)
            assert process.exitcode == 0
        outputs = [
            json.loads((tmp_path / f"{name}.json").read_text()) for name in ("gru", "dlinear")
        ]
        assert outputs[0]["pid"] != outputs[1]["pid"]
        assert all(np.isfinite(o["losses"]).all() and o["peak_bytes"] > 0 for o in outputs)
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
            if process.is_alive():
                process.kill()
                process.join(timeout=10)
            process.close()
