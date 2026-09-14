"""Baseline selection never consumes holdout and uses shared training scales."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from torch import nn

import stock_forecasting.baselines as baseline_module
from stock_forecasting.baselines import (
    CausalGRUBaseline,
    _fit_neural,
    baseline_arrays,
    concatenate_baseline_batches,
    evaluate_causal_gru,
)


def test_neural_early_stopping_reads_validation_then_scores_holdout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train = SimpleNamespace(
        sequences=np.ones((4, 2, 2, 5), dtype=np.float32),
        targets=np.zeros((4, 12), dtype=np.float64),
    )
    validation = object()
    holdout = object()
    scales = [0.125] * 12
    calls = []

    def evaluate(_model, arrays, actual_scales, **_kwargs):
        assert actual_scales == scales
        calls.append(arrays)
        return {"primary_5d": {"selection_score": 0.2}, "scored_holdout": arrays is holdout}

    def forbidden(_targets):
        raise AssertionError("Shared scales must not be refitted on a smaller baseline sample")

    monkeypatch.setattr(baseline_module, "_evaluate_neural", evaluate)
    monkeypatch.setattr(baseline_module, "_robust_scales", forbidden)
    model = nn.Sequential(nn.Flatten(), nn.Linear(20, 36), nn.Unflatten(1, (12, 3)))
    _, metrics = _fit_neural(
        model, train, validation, seed=42, epochs=3, patience=1, batch_size=2,
        learning_rate=1e-3, robust_scales=scales, evaluation=holdout,
    )
    assert calls == [validation, validation, holdout]
    assert metrics["scored_holdout"] is True


def test_evaluation_without_train_scales_cannot_fit_them_on_holdout() -> None:
    with pytest.raises(ValueError, match="train-calibrated"):
        evaluate_causal_gru(CausalGRUBaseline(), object())


def test_worker_batch_concatenation_preserves_membership_and_values(window_records) -> None:
    records = window_records[:8]
    expected = baseline_arrays(records)
    actual = concatenate_baseline_batches(
        (baseline_arrays(records[:3]), baseline_arrays(records[3:])),
    )
    assert actual.symbols == expected.symbols
    assert actual.dates == expected.dates
    assert actual.markets == expected.markets
    assert actual.horizons == expected.horizons
    np.testing.assert_array_equal(actual.targets, expected.targets)
    np.testing.assert_array_equal(actual.features, expected.features)
    np.testing.assert_array_equal(actual.sequences, expected.sequences)
