"""Past-only, train-only and read-only contracts for checkpoint scale diagnostics."""

from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from stock_forecasting.checkpointing import load_checkpoint, save_ranked_checkpoint
from stock_forecasting.cli.evaluate import resolve_checkpoint
from stock_forecasting.cli.probe_scales import build_parser, render_summary, run_probe
from stock_forecasting.config import ExperimentConfig
from stock_forecasting.data.bar_store import build_symbol_bar_store
from stock_forecasting.data.dataset import LazyFinancialWindowDataset
from stock_forecasting.data.manifest import artifact_metadata, sha256_file
from stock_forecasting.factory import build_model_bundle
from stock_forecasting.models.outputs import QuantForecastOutput
from stock_forecasting.representation_scale_probe import (
    READOUT_DESCRIPTIONS,
    TARGET_NAMES,
    HistoricalScaleDataset,
    ProbeSplit,
    ScaleProbeSettings,
    extract_probe_split,
    fit_scale_probes,
    historical_scale_targets,
    regression_metrics,
    representation_readouts,
    sampled_ordinals,
)
from stock_forecasting.run_contract import (
    CHECKPOINT_ARTIFACT_SCHEMA_VERSION,
    training_resume_contract_fingerprint,
)

ROOT = Path(__file__).resolve().parents[1]


def test_historical_scales_match_known_returns_and_are_price_unit_invariant() -> None:
    index = np.arange(127, dtype=float)
    asset_returns = 0.002 + 0.02 * np.sin(index)
    benchmark_returns = 0.001 + 0.01 * np.cos(index)
    asset = 100 * np.exp(np.r_[0.0, np.cumsum(asset_returns)])
    benchmark = 200 * np.exp(np.r_[0.0, np.cumsum(benchmark_returns)])
    expected = [
        returns[-window:].std(ddof=0)
        for window in (20, 60)
        for returns in (asset_returns, benchmark_returns, asset_returns - benchmark_returns)
    ] + [asset.std() / asset.mean(), benchmark.std() / benchmark.mean()]
    actual = historical_scale_targets(asset, benchmark)
    np.testing.assert_allclose(actual, expected, rtol=1e-12)
    np.testing.assert_allclose(historical_scale_targets(asset * 1e3, benchmark * 7), actual)
    assert len(actual) == len(TARGET_NAMES) == 8


def test_historical_scales_handle_constant_prices_and_padding() -> None:
    np.testing.assert_array_equal(historical_scale_targets(np.ones(128), np.ones(128)), 0)
    asset = np.linspace(100, 110, 61)
    benchmark = np.linspace(200, 205, 61)
    mask = np.r_[np.zeros(3, dtype=bool), np.ones(61, dtype=bool)]
    np.testing.assert_allclose(
        historical_scale_targets(
            np.r_[np.nan, 0, -1, asset], np.r_[0, np.inf, -1, benchmark], valid_mask=mask
        ),
        historical_scale_targets(asset, benchmark),
    )
    with pytest.raises(ValueError, match="61"):
        historical_scale_targets(asset[:-1], benchmark[:-1])
    with pytest.raises(ValueError, match="finite and positive"):
        historical_scale_targets(np.r_[0, asset], np.r_[1, benchmark])
    with pytest.raises(ValueError, match="aligned"):
        historical_scale_targets(asset, benchmark[:-1])
    with pytest.raises(ValueError, match="boolean"):
        historical_scale_targets(asset, benchmark, valid_mask=np.ones(61))


@pytest.fixture
def scale_store(tmp_path: Path, market_frame: pd.DataFrame) -> Path:
    raw = tmp_path / "raw.parquet"
    market_frame.to_parquet(raw, index=False)
    store = tmp_path / "bar-store"
    build_symbol_bar_store(
        raw_path=raw,
        output_root=store,
        download_manifest={
            "artifacts": {"raw": artifact_metadata(raw, root=tmp_path, row_count=len(market_frame))}
        },
        window_size=64,
        bucket_count=2,
        batch_rows=200,
        purge_bars=20,
        embargo_bars=14,
    )
    return store


def test_dataset_uses_exact_inputs_without_constructing_future_labels(
    scale_store: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = LazyFinancialWindowDataset(scale_store, split="train", window_size=64)
    expected = source[3]
    dataset = HistoricalScaleDataset(source)

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("The historical probe must never construct forward labels")

    monkeypatch.setattr(source, "_sample_frames", forbidden)
    monkeypatch.setattr(source, "_target_components", forbidden)
    monkeypatch.setattr(source, "_item", forbidden)
    before = dataset[3]
    for key in ("asset_series", "benchmark_series"):
        assert torch.equal(before[key], expected[key])
    assert before["sample_id"] == expected["sample_id"]
    assert "target_alpha" not in before
    assert next(iter(DataLoader(dataset, batch_size=2)))["scale_targets"].shape == (2, 8)

    _, cutoff_index = source._resolve_cutoff(3)
    cutoff_at = pd.Timestamp(before["cutoff_at"])
    for symbol in (before["symbol"], before["benchmark_symbol"]):
        frame = source._load_symbol(symbol)
        future = frame["timestamp"] > cutoff_at
        assert future.any()
        frame.loc[future, ["open", "high", "low", "close", "adjusted_close"]] = np.nan
    assert cutoff_index >= 63
    after = dataset[3]
    for key in ("asset_series", "benchmark_series", "scale_targets"):
        assert torch.equal(before[key], after[key])


def test_dataset_refuses_test_split(scale_store: Path) -> None:
    with pytest.raises(ValueError, match="exclude the test"):
        HistoricalScaleDataset(
            LazyFinancialWindowDataset(scale_store, split="test", window_size=64)
        )


def test_sampling_is_deterministic_bounded_unique_and_not_prefix_only() -> None:
    first = sampled_ordinals(100_000, 127, 42)
    assert first == sampled_ordinals(100_000, 127, 42)
    assert first != sampled_ordinals(100_000, 127, 43)
    assert len(first) == len(set(first)) == 127
    assert first == sorted(first)
    assert min(first) < 10_000 and max(first) > 90_000
    assert sampled_ordinals(5, 10, 42) == list(range(5))
    with pytest.raises(ValueError, match="at least two"):
        sampled_ordinals(1, 10, 42)


def _output(hidden: torch.Tensor, mask: torch.Tensor) -> QuantForecastOutput:
    latents = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2)
    return QuantForecastOutput(
        loss=None,
        pinball_loss=None,
        alpha_quantiles=torch.zeros(2, 14, 3),
        asset_last_hidden_state=hidden,
        benchmark_last_hidden_state=hidden + 10,
        asset_attention_mask=mask,
        benchmark_attention_mask=mask,
        asset_latent_tokens=latents,
        benchmark_latent_tokens=latents + 10,
        conditioned_latent_tokens=latents,
        conditioning_gate=latents,
    )


def test_readouts_obey_masks_and_match_the_head_pool_exactly() -> None:
    hidden = torch.tensor([[[99.0], [1.0], [3.0]], [[2.0], [4.0], [99.0]]])
    output = _output(hidden, torch.tensor([[False, True, True], [True, True, False]]))
    readouts = representation_readouts(output)
    assert set(readouts) == set(READOUT_DESCRIPTIONS)
    torch.testing.assert_close(
        readouts["kronos_pair_mean"], torch.tensor([[2.0, 12.0], [3.0, 13.0]])
    )
    torch.testing.assert_close(
        readouts["kronos_pair_last"], torch.tensor([[3.0, 13.0], [4.0, 14.0]])
    )
    torch.testing.assert_close(
        readouts["conditioned_mean"], output.conditioned_latent_tokens.mean(dim=1)
    )
    bf16_output = replace(
        output, conditioned_latent_tokens=(output.conditioned_latent_tokens * 0.013).bfloat16()
    )
    assert torch.equal(
        representation_readouts(bf16_output)["conditioned_mean"],
        bf16_output.conditioned_latent_tokens.mean(dim=1).float(),
    )
    with pytest.raises(ValueError, match="valid token"):
        representation_readouts(_output(hidden, torch.zeros(2, 3, dtype=torch.bool)))
    hidden[0, 1] = float("nan")
    with pytest.raises(ValueError, match="Non-finite"):
        representation_readouts(_output(hidden, output.asset_attention_mask))


def _linear_splits() -> tuple[ProbeSplit, ProbeSplit]:
    rng = np.random.default_rng(10)
    coefficients = rng.normal(size=(3, 8))
    splits = []
    for name, year, count in (("train", 2020, 160), ("validation", 2021, 80)):
        features = rng.normal(size=(count, 3))
        targets = 10 + features @ coefficients
        samples = [
            {
                "sample_id": f"{name}-{index}",
                "market": "US" if index % 2 else "TWSE",
                "symbol": "fixture",
                "cutoff_at": f"{year}-01-01T00:00:00+00:00",
            }
            for index in range(count)
        ]
        splits.append(ProbeSplit({"conditioned_mean": features}, targets, samples, count))
    return splits[0], splits[1]


def test_linear_probe_recovers_signal_and_shuffled_control_does_not() -> None:
    train, validation = _linear_splits()
    result = fit_scale_probes(train, validation, ScaleProbeSettings(ridge_alpha=0.001))
    metrics = result.metrics["readouts"]["conditioned_mean"]
    for name in TARGET_NAMES:
        assert metrics["validation"][name]["r2"] > 0.999
        assert metrics["validation"][name]["mse_skill_vs_train_mean"] > 0.999
        assert metrics["shuffled_label_validation"][name]["r2"] < 0.4
        assert result.metrics["train_mean_baseline"][name]["mse_skill_vs_train_mean"] == 0
    assert set(metrics["validation_by_market"]) == {"US", "TWSE"}
    assert metrics["feature_dim"] == 3
    assert metrics["train_samples_per_feature"] == pytest.approx(160 / 3)


def test_validation_values_cannot_change_scalers_or_probe_weights() -> None:
    train, validation = _linear_splits()
    settings = ScaleProbeSettings()
    first = fit_scale_probes(train, validation, settings)
    validation.targets += 10_000
    validation.features["conditioned_mean"] *= 10_000
    second = fit_scale_probes(train, validation, settings)
    for key, value in first.fitted_arrays.items():
        np.testing.assert_array_equal(value, second.fitted_arrays[key])
    np.testing.assert_allclose(first.fitted_arrays["target_mean"], train.targets.mean(axis=0))
    np.testing.assert_allclose(
        first.fitted_arrays["conditioned_mean__feature_mean"],
        train.features["conditioned_mean"].mean(axis=0),
    )


@pytest.mark.parametrize("violation", ["overlap", "reversed_dates", "duplicate", "nan"])
def test_probe_rejects_invalid_split_membership_or_values(violation: str) -> None:
    train, validation = _linear_splits()
    if violation == "overlap":
        validation.samples[0]["sample_id"] = train.samples[0]["sample_id"]
    elif violation == "reversed_dates":
        validation.samples[0]["cutoff_at"] = "2019-01-01T00:00:00+00:00"
    elif violation == "duplicate":
        validation.samples[0]["sample_id"] = validation.samples[1]["sample_id"]
    else:
        train.targets[0, 0] = np.nan
    with pytest.raises(ValueError):
        fit_scale_probes(train, validation, ScaleProbeSettings())


def test_constant_targets_and_predictions_are_not_reported_as_valid_correlations() -> None:
    truth = np.ones((8, 8))
    score = regression_metrics(truth, truth, np.ones(8))
    for value in score.values():
        assert value["r2"] is None
        assert value["pearson_r"] is None
        assert value["mse_skill_vs_train_mean"] is None
        assert value["mae"] == value["rmse"] == 0
    variable = np.arange(64).reshape(8, 8).astype(float)
    score = regression_metrics(variable, truth, np.ones(8))
    assert all(value["pearson_r"] is None for value in score.values())
    assert all(value["r2"] < 0 for value in score.values())
    json.dumps(score, allow_nan=False)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"train_samples": 1},
        {"validation_samples": 0},
        {"batch_size": 0},
        {"ridge_alpha": 0},
        {"ridge_alpha": float("nan")},
        {"num_workers": 17},
        {"seed": -1},
    ],
)
def test_invalid_settings_fail_before_execution(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        ScaleProbeSettings(**kwargs)


def test_mock_checkpoint_extraction_preserves_weights_and_checkpoint_files(
    tmp_path: Path,
    scale_store: Path,
) -> None:
    config = ExperimentConfig.from_yaml(ROOT / "configs/local_mock.yaml")
    config.data.input_length = 64
    config.data.bar_store_path = scale_store
    config.data.manifest_path = tmp_path / "not-required-mock-manifest.json"
    config.training.output_root = tmp_path / "savedModel"
    run = config.training.output_root / "run-scale-probe-test"
    run.mkdir(parents=True)
    contract, digest = training_resume_contract_fingerprint(config)
    (run / "run-manifest.json").write_text(
        json.dumps(
            {
                "schema_version": CHECKPOINT_ARTIFACT_SCHEMA_VERSION,
                "run_id": run.name,
                "run_key": run.name,
                "training_resume_contract": contract,
                "training_resume_contract_sha256": digest,
            }
        ),
        encoding="utf-8",
    )
    torch.manual_seed(8)
    model = build_model_bundle(config, torch.device("cpu")).model
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    checkpoint, _ = save_ranked_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=LambdaLR(optimizer, lambda _step: 1.0),
        config=config,
        run_directory=run,
        global_step=1,
        epoch=0,
        batch_index=1,
        selection_metric_name="primary_5d/selection_score",
        selection_metric_value=0.1,
        selection_metric_mode="min",
        save_top_k=1,
        metrics={"primary_5d/selection_score": 0.1},
    )
    assert checkpoint is not None
    assert resolve_checkpoint(run, saved_model_root=config.training.output_root) == checkpoint
    before_files = {str(path): sha256_file(path) for path in run.rglob("*") if path.is_file()}
    torch.manual_seed(8)
    restored = build_model_bundle(config, torch.device("cpu")).model
    load_checkpoint(checkpoint, restored, config=config)
    restored.requires_grad_(False)
    before_weights = {key: value.clone() for key, value in restored.state_dict().items()}
    settings = ScaleProbeSettings(train_samples=8, validation_samples=4, batch_size=2)
    splits = {}
    for split in ("train", "validation"):
        source = HistoricalScaleDataset(
            LazyFinancialWindowDataset(scale_store, split=split, window_size=64)
        )
        splits[split] = extract_probe_split(
            restored,
            source,
            device=torch.device("cpu"),
            settings=settings,
        )
        assert len(splits[split].samples) == (8 if split == "train" else 4)
        assert set(splits[split].features) == set(READOUT_DESCRIPTIONS)
        assert all(
            len(values) == len(splits[split].samples) for values in splits[split].features.values()
        )
    result = fit_scale_probes(splits["train"], splits["validation"], settings)
    assert set(result.metrics["readouts"]) == set(READOUT_DESCRIPTIONS)
    for key, value in restored.state_dict().items():
        assert torch.equal(before_weights[key], value)
    assert not any(parameter.grad is not None for parameter in restored.parameters())
    assert before_files == {
        str(path): sha256_file(path) for path in run.rglob("*") if path.is_file()
    }
    report = {
        "checkpoint": {"path": str(checkpoint)},
        "splits": {key: value.summary() for key, value in splits.items()},
        "metrics": result.metrics,
    }
    summary = render_summary(report)
    assert summary.index("## 中文") < summary.index("## English")
    assert "Stage 1" in summary and "N/A" in summary
    assert "feature_dim" in json.dumps(result.metrics, allow_nan=False)


def test_cli_has_no_test_or_config_override_and_refuses_local_execution(tmp_path: Path) -> None:
    parser = build_parser()
    args = parser.parse_args(["--checkpoint", str(tmp_path)])
    assert args.train_samples == 16_384
    with pytest.raises(SystemExit):
        parser.parse_args(["--checkpoint", str(tmp_path), "--split", "test"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--checkpoint", str(tmp_path), "--config", "new-config.yaml"])
    with pytest.raises(RuntimeError, match="existing CUDA RunPod"):
        run_probe(tmp_path, ScaleProbeSettings())
    assert not list(tmp_path.iterdir())


def test_direct_cli_cannot_bypass_the_script_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RUNPOD_POD_ID", "probe-fixture")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    with pytest.raises(RuntimeError, match="verify the mount and GPU lease"):
        run_probe(tmp_path, ScaleProbeSettings())
    assert not list(tmp_path.iterdir())


def test_workflow_routes_help_and_blocks_local_checkpoint_execution(tmp_path: Path) -> None:
    command = ["bash", str(ROOT / "scripts/runpod_workflow.sh"), "probe-scales"]
    help_result = subprocess.run([*command, "--help"], capture_output=True, text=True, check=False)
    assert help_result.returncode == 0
    assert "existing CUDA RunPod" in help_result.stdout
    assert "does not create/terminate" in help_result.stdout
    result = subprocess.run(
        [*command, "--checkpoint", str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "no local model execution" in result.stderr
    assert not list(tmp_path.iterdir())
