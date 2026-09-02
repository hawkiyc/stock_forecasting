"""Contracts for reusing numerical validation metrics from checkpoints."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from stock_forecasting.training import _flatten_metrics
from stock_forecasting.validation_benchmark import (
    _unflatten_validation_metrics,
    checkpoint_validation_snapshot,
)


def _validation_metrics() -> dict[str, object]:
    return {
        "loss": 0.25,
        "primary_5d": {
            "selection_score": 0.20,
            "normalized_pinball": 0.20,
        },
        "cross_sectional_5d": {
            "rank_ic_mean": 0.12,
            "net_long_short_sharpe": 0.75,
        },
        "per_horizon": {"5d": {"median_mae": 0.10}},
    }


def test_checkpoint_snapshot_round_trips_trainer_metric_namespace(
    tmp_path: Path,
) -> None:
    metrics = _validation_metrics()
    state = {
        "global_step": 200,
        "created_at": "2026-09-02T08:18:00+00:00",
        "metrics": _flatten_metrics(metrics),
    }
    (tmp_path / "trainer-state.json").write_text(
        json.dumps(state),
        encoding="utf-8",
    )

    snapshot = checkpoint_validation_snapshot(tmp_path)

    assert snapshot == {
        "metrics": metrics,
        "source": "checkpoint_validation_snapshot",
        "global_step": 200,
        "created_at": "2026-09-02T08:18:00+00:00",
    }


def test_checkpoint_snapshot_accepts_legacy_validation_prefix() -> None:
    metrics = _validation_metrics()

    assert _unflatten_validation_metrics(
        _flatten_metrics(metrics, "validation")
    ) == metrics


def test_checkpoint_snapshot_rejects_mixed_metric_namespaces() -> None:
    metrics = _validation_metrics()
    mixed = {
        **_flatten_metrics(metrics),
        **_flatten_metrics(metrics, "validation"),
    }

    with pytest.raises(ValueError, match="mixes validation metric namespaces"):
        _unflatten_validation_metrics(mixed)


def test_checkpoint_snapshot_requires_primary_and_cross_sectional_metrics() -> None:
    with pytest.raises(ValueError, match="complete numerical validation metric snapshot"):
        _unflatten_validation_metrics({"primary_5d/selection_score": 0.2})
