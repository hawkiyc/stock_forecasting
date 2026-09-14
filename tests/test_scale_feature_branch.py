"""Past-only scale features, residual fusion, and checkpoint-bound calibration."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
from torch import nn

from stock_forecasting.checkpointing import (
    _model_runtime_scale_features,
    _restore_runtime_scale_features,
)
from stock_forecasting.models.forecast import MultiHorizonAlphaHead
from stock_forecasting.models.scale_features import (
    SCALE_FEATURE_NAMES,
    NumericalResidualBranch,
    fit_scale_feature_statistics,
    historical_scale_features,
    validate_scale_feature_statistics,
)
from stock_forecasting.representation_scale_probe import historical_scale_targets


def _statistics() -> dict:
    values = np.random.default_rng(5).uniform(0.001, 0.15, (100, 8))
    return fit_scale_feature_statistics(values, {"split": "train", "sample_count": 100})


def _history(close: np.ndarray) -> torch.Tensor:
    return torch.tensor(close, dtype=torch.float32)[None, :, None].expand(1, -1, 5).clone()


def test_vectorized_features_match_the_frozen_diagnostic_and_price_units() -> None:
    index = np.arange(128)
    asset = 100 * np.exp(np.cumsum(0.002 + 0.02 * np.sin(index)))
    benchmark = 200 * np.exp(np.cumsum(0.001 + 0.01 * np.cos(index)))
    actual = historical_scale_features(_history(asset), _history(benchmark))
    assert actual.shape == (1, len(SCALE_FEATURE_NAMES))
    np.testing.assert_allclose(
        actual.numpy()[0], historical_scale_targets(asset, benchmark), rtol=3e-5, atol=1e-7,
    )
    torch.testing.assert_close(
        actual, historical_scale_features(_history(asset * 100), _history(benchmark * 7)),
        rtol=3e-5, atol=1e-7,
    )


def test_features_ignore_masked_rows_including_internal_gaps() -> None:
    close = np.linspace(100, 110, 64)
    asset = _history(close)
    benchmark = _history(close * 2)
    mask = torch.ones(1, 64, dtype=torch.bool)
    mask[:, [0, 20, 63]] = False
    asset[:, ~mask[0]] = float("nan")
    benchmark[:, ~mask[0]] = -1
    actual = historical_scale_features(asset, benchmark, mask=mask)
    expected = historical_scale_features(_history(close[mask[0]]), _history(close[mask[0]] * 2))
    torch.testing.assert_close(actual, expected)
    with pytest.raises(ValueError, match="61"):
        historical_scale_features(asset[:, :60], benchmark[:, :60])
    with pytest.raises(ValueError, match="finite and positive"):
        historical_scale_features(asset, benchmark)
    assert historical_scale_features(_history(np.ones(64)), _history(np.ones(64))).eq(0).all()


@pytest.mark.parametrize("mode", ["baseline", "scales", "benchmark", "combined"])
def test_all_feature_modes_preserve_initial_forecasts_and_ordered_quantiles(mode: str) -> None:
    torch.manual_seed(4)
    baseline = MultiHorizonAlphaHead(32, horizons=range(1, 15), fp32_head=True)
    torch.manual_seed(4)
    head = MultiHorizonAlphaHead(32, horizons=range(1, 15), feature_mode=mode, fp32_head=True)
    if mode in ("scales", "combined"):
        head.numeric_branch.set_statistics(_statistics())
    inputs = torch.randn(2, 4, 32)
    scales = torch.full((2, 8), 0.02)
    benchmark = torch.randn(2, 4, 32)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = head(inputs, scale_features=scales, benchmark_tokens=benchmark)
    assert actual.dtype == torch.float32
    assert actual.shape == (2, 14, 3)
    torch.testing.assert_close(actual, baseline(inputs))
    assert (actual[..., 0] < actual[..., 1]).all()
    assert (actual[..., 1] < actual[..., 2]).all()
    loss = head.pinball_loss(actual, torch.zeros(2, 14))
    assert loss.dtype == torch.float32 and torch.isfinite(loss)
    loss.backward()
    if head.numeric_branch is not None:
        assert head.numeric_branch.fusion[-1].weight.grad.abs().sum() > 0


def test_combined_branch_reads_scales_and_benchmark_after_zero_initialization() -> None:
    branch = NumericalResidualBranch(512, 512, "combined")
    assert sum(parameter.numel() for parameter in branch.parameters()) == 18899
    branch.set_statistics(_statistics())
    hidden = torch.randn(2, 14, 512, requires_grad=True)
    scales = torch.full((2, 8), 0.02, requires_grad=True)
    benchmark = torch.randn(2, 3, 512, requires_grad=True)
    with torch.no_grad():
        branch.fusion[-1].weight.fill_(0.02)
    branch(hidden, scales, benchmark).square().mean().backward()
    assert scales.grad is not None and scales.grad.abs().sum() > 0
    assert benchmark.grad is not None and benchmark.grad.abs().sum() > 0


def test_statistics_are_train_only_validated_and_restore_identical_forecasts() -> None:
    statistics = _statistics()
    assert validate_scale_feature_statistics(statistics) == statistics
    with pytest.raises(ValueError, match="train-only"):
        fit_scale_feature_statistics(np.ones((5, 8)), {"split": "test"})
    corrupted = copy.deepcopy(statistics)
    corrupted["center"][0] += 1
    with pytest.raises(ValueError, match="digest mismatch"):
        validate_scale_feature_statistics(corrupted)
    model = nn.Module()
    model.alpha_head = MultiHorizonAlphaHead(32, feature_mode="combined")
    with pytest.raises(ValueError, match="missing or malformed"):
        _model_runtime_scale_features(model)
    model.alpha_head.numeric_branch.set_statistics(statistics)
    with torch.no_grad():
        model.alpha_head.numeric_branch.fusion[-1].weight.fill_(0.01)
    inputs = torch.randn(2, 3, 32)
    features = torch.full((2, 8), 0.02)
    benchmark = torch.randn(2, 3, 32)
    expected = model.alpha_head(inputs, scale_features=features, benchmark_tokens=benchmark)
    state = {"runtime_scale_features": _model_runtime_scale_features(model)}
    restored = copy.deepcopy(model)
    restored.alpha_head.numeric_branch.statistics = None
    restored.alpha_head.numeric_branch.feature_center.fill_(100)
    restored.alpha_head.numeric_branch.feature_scale.fill_(100)
    _restore_runtime_scale_features(restored, state, require_match=False)
    assert restored.alpha_head.numeric_branch.statistics == statistics
    actual = restored.alpha_head(inputs, scale_features=features, benchmark_tokens=benchmark)
    torch.testing.assert_close(actual, expected)
    with pytest.raises(ValueError, match="missing or malformed"):
        _restore_runtime_scale_features(restored, {}, require_match=False)
    changed = fit_scale_feature_statistics(np.full((5, 8), 0.3), {"split": "train"})
    with pytest.raises(ValueError, match="differ from train calibration"):
        _restore_runtime_scale_features(
            restored, {"runtime_scale_features": changed}, require_match=True,
        )
