"""Causal inference and immutable conditional-alpha JSON contract tests."""

from __future__ import annotations

import pandas as pd
import pytest
import torch

from stock_forecasting.cli.infer import _forecast_payload, prepare_causal_observation
from stock_forecasting.data.windows import DEFAULT_ALPHA_HORIZONS


def test_inference_observation_stops_at_explicit_cutoff_and_pairs_benchmark(
    market_frame: pd.DataFrame,
) -> None:
    symbol_frame = market_frame[market_frame["symbol"] == "AAPL.US"]
    cutoff = pd.Timestamp(symbol_frame.iloc[199]["timestamp"])
    observation = prepare_causal_observation(
        market_frame,
        input_length=128,
        symbol="AAPL.US",
        as_of=cutoff,
    )

    assert pd.Timestamp(observation.cutoff_at) == cutoff
    assert observation.symbol == "AAPL.US"
    assert observation.benchmark_symbol == "VTI.US"
    assert observation.asset_series.shape == (1, 128, 5)
    assert observation.benchmark_series.shape == (1, 128, 5)
    assert observation.asset_timestamp_features.shape == (1, 128, 5)
    assert observation.benchmark_timestamp_features.shape == (1, 128, 5)
    assert observation.asset_attention_mask.all()
    assert observation.benchmark_attention_mask.all()
    assert observation.metadata["provider"] == "eodhd"
    assert observation.metadata["dataset_profile"] == "us_tw_eodhd"
    assert observation.metadata["benchmark_policy"] == "us_vti"


def test_inference_is_invariant_to_vendor_global_adjustment_scale(
    market_frame: pd.DataFrame,
) -> None:
    rescaled = market_frame.copy()
    rescaled["adjusted_close"] = rescaled["adjusted_close"] * 0.01

    original = prepare_causal_observation(
        market_frame,
        input_length=128,
        symbol="AAPL.US",
    )
    adjusted = prepare_causal_observation(
        rescaled,
        input_length=128,
        symbol="AAPL.US",
    )

    torch.testing.assert_close(adjusted.asset_series, original.asset_series)
    torch.testing.assert_close(adjusted.benchmark_series, original.benchmark_series)


def test_inference_rejects_missing_adjustment_anchors(
    market_frame: pd.DataFrame,
) -> None:
    raw_only = market_frame.drop(columns=["adjusted_close", "split_adjusted_volume"])

    with pytest.raises(ValueError, match="adjustment anchors"):
        prepare_causal_observation(
            raw_only,
            input_length=128,
            symbol="AAPL.US",
        )


def test_forecast_payload_is_multi_horizon_continuous_and_classifier_free() -> None:
    patterns = torch.tensor(
        [
            [-0.06, -0.04, -0.02],
            [-0.04, -0.02, 0.00],
            [-0.02, 0.00, 0.02],
            [0.00, 0.02, 0.04],
            [0.02, 0.04, 0.06],
        ],
        dtype=torch.float32,
    )
    values = torch.stack(
        [patterns[index % len(patterns)] for index in range(len(DEFAULT_ALPHA_HORIZONS))]
    ).unsqueeze(0)
    payload = _forecast_payload(
        values,
        horizons=list(DEFAULT_ALPHA_HORIZONS),
        quantile_levels=[0.1, 0.5, 0.9],
        signal_threshold=0.01,
    )

    assert set(payload) == {"units", "signal_threshold", "by_horizon"}
    assert payload["units"] == "benchmark_relative_adjusted_log_return"
    assert payload["signal_threshold"] == pytest.approx(0.01)
    assert list(payload["by_horizon"]) == [
        f"{horizon}d" for horizon in DEFAULT_ALPHA_HORIZONS
    ]
    assert set(payload["by_horizon"]["3d"]) == {
        "alpha_quantiles",
        "central_interval",
        "derived_signal",
    }
    assert payload["by_horizon"]["3d"]["alpha_quantiles"] == pytest.approx(
        {"q10": -0.06, "q50": -0.04, "q90": -0.02}
    )
    assert payload["by_horizon"]["3d"]["central_interval"] == pytest.approx(
        {"nominal_coverage": 0.8, "lower": -0.06, "upper": -0.02}
    )
    assert {
        horizon["derived_signal"] for horizon in payload["by_horizon"].values()
    } == {
        "strong_bearish",
        "bearish",
        "neutral",
        "bullish",
        "strong_bullish",
    }
    rendered = repr(payload).lower()
    assert "logit" not in rendered
    assert "probabilit" not in rendered
    assert "class" not in rendered
