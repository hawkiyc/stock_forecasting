"""Run paired causal OHLCV histories through the conditional alpha model."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from torch import Tensor

from stock_forecasting.checkpointing import load_checkpoint
from stock_forecasting.cli.evaluate import resolve_checkpoint
from stock_forecasting.config import ExperimentConfig
from stock_forecasting.data.adjustments import asof_adjusted_window, ensure_adjustment_columns
from stock_forecasting.data.benchmarks import resolve_benchmark
from stock_forecasting.data.manifest import provenance_summary
from stock_forecasting.data.schema import normalize_ohlcv_frame, read_market_data
from stock_forecasting.factory import build_model_bundle
from stock_forecasting.metrics import POSTPROCESS_SIGNAL_NAMES, postprocess_alpha_signal
from stock_forecasting.models import MODEL_OUTPUT_SCHEMA_VERSION
from stock_forecasting.preflight import run_preflight
from stock_forecasting.training import _autocast_context, set_global_seed

Float32Array = NDArray[np.float32]
Int64Array = NDArray[np.int64]


@dataclass(frozen=True)
class CausalObservation:
    """Paired model inputs whose rows end exactly at one declared cutoff."""

    asset_series: Tensor
    asset_timestamp_features: Tensor
    asset_attention_mask: Tensor
    benchmark_series: Tensor
    benchmark_timestamp_features: Tensor
    benchmark_attention_mask: Tensor
    symbol: str
    benchmark_symbol: str
    asset_type: str
    window_start_at: str
    cutoff_at: str
    metadata: dict[str, Any]
    benchmark_metadata: dict[str, Any]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--symbol", help="Required when input contains multiple target symbols.")
    parser.add_argument(
        "--as-of",
        help="Optional inclusive UTC cutoff; later rows are never read by the model.",
    )
    parser.add_argument("--output", type=Path)
    return parser


def _utc_timestamp(value: str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _load_mapping(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("benchmark_mapping_path must contain a JSON object")
    mapping = {
        str(symbol).strip().upper(): str(benchmark).strip().upper()
        for symbol, benchmark in payload.items()
    }
    if any(not symbol or not benchmark for symbol, benchmark in mapping.items()):
        raise ValueError("Benchmark mapping identifiers must be non-empty")
    return mapping


def _select_symbol(frame: pd.DataFrame, requested_symbol: str | None) -> str:
    candidates = frame[frame["asset_type"] != "index"]
    available = sorted(str(symbol) for symbol in candidates["symbol"].unique())
    if requested_symbol is None:
        if len(available) != 1:
            raise ValueError(
                "--symbol is required when input contains multiple non-index symbols: "
                + ", ".join(available)
            )
        return available[0]
    symbol = requested_symbol.upper()
    if symbol not in available:
        raise ValueError(
            f"Symbol {symbol} is not present as a target; available: {', '.join(available)}"
        )
    return symbol


def _ohlcv_values(window: pd.DataFrame) -> Float32Array:
    return cast(
        Float32Array,
        window[["open", "high", "low", "close", "volume"]].to_numpy(dtype=np.float32),
    )


def _kronos_timestamps(window: pd.DataFrame) -> Int64Array:
    timestamps = pd.DatetimeIndex(window["timestamp"])
    return cast(
        Int64Array,
        np.column_stack(
            [
                timestamps.minute,
                timestamps.hour,
                timestamps.dayofweek,
                timestamps.day,
                timestamps.month,
            ]
        ).astype(np.int64),
    )


def _metadata(window: pd.DataFrame) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for field in (
        "provider",
        "market",
        "currency",
        "source_symbol",
        "dataset_profile",
        "adjustment_source",
    ):
        if field in window:
            values = window[field].dropna().unique().tolist()
            if len(values) == 1:
                output[field] = str(values[0])
    return output


def prepare_causal_observation(
    frame: pd.DataFrame,
    *,
    input_length: int,
    symbol: str | None = None,
    as_of: str | pd.Timestamp | None = None,
    benchmark_mapping: dict[str, str] | None = None,
) -> CausalObservation:
    """Select exact aligned histories and apply cutoff-causal adjustments."""

    required_anchors = {"adjusted_close", "split_adjusted_volume"}
    missing_anchors = sorted(required_anchors.difference(frame.columns))
    if missing_anchors:
        raise ValueError(
            "Inference input must contain point-in-time adjustment anchors: "
            + ", ".join(missing_anchors)
        )
    normalized = ensure_adjustment_columns(normalize_ohlcv_frame(frame))
    if as_of is not None:
        normalized = normalized[normalized["timestamp"] <= _utc_timestamp(as_of)].copy()
    selected_symbol = _select_symbol(normalized, symbol)
    selected = normalized[normalized["symbol"] == selected_symbol].copy()
    selected = selected.sort_values("timestamp", kind="stable")
    if len(selected) < input_length:
        raise ValueError(
            f"{selected_symbol} has {len(selected)} causal rows; input_length={input_length}"
        )
    asset_types = sorted(str(value) for value in selected["asset_type"].unique())
    markets = sorted(str(value) for value in selected.get("market", pd.Series([""])).unique())
    if len(asset_types) != 1 or len(markets) != 1:
        raise ValueError("Asset type and market must remain stable within the selected history")
    decision = resolve_benchmark(
        symbol=selected_symbol,
        asset_type=asset_types[0],
        market=markets[0],
        explicit_mapping=benchmark_mapping,
    )
    if not decision.eligible or decision.benchmark_symbol is None:
        raise ValueError(f"Symbol is not inference-eligible: {decision.reason}")
    benchmark_symbol = decision.benchmark_symbol
    benchmark = normalized[normalized["symbol"] == benchmark_symbol].copy()
    if benchmark.empty:
        raise ValueError(f"Required benchmark is absent from inference input: {benchmark_symbol}")

    raw_window = selected.tail(input_length).reset_index(drop=True)
    benchmark_index = benchmark.set_index("timestamp", drop=False)
    requested_dates = pd.DatetimeIndex(raw_window["timestamp"])
    if not requested_dates.isin(benchmark_index.index).all():
        raise ValueError("Benchmark history has a calendar gap within the asset input window")
    benchmark_raw_window = benchmark_index.loc[requested_dates].reset_index(drop=True)
    if len(benchmark_raw_window) != input_length:
        raise ValueError("Benchmark timestamps must be unique and exactly aligned")

    asset_window = asof_adjusted_window(raw_window)
    benchmark_window = asof_adjusted_window(benchmark_raw_window)
    start_at = pd.Timestamp(cast(Any, raw_window.loc[0, "timestamp"])).isoformat()
    cutoff_at = pd.Timestamp(
        cast(Any, raw_window.loc[len(raw_window) - 1, "timestamp"])
    ).isoformat()
    mask = torch.ones((1, input_length), dtype=torch.bool)
    return CausalObservation(
        asset_series=torch.from_numpy(_ohlcv_values(asset_window)).unsqueeze(0),
        asset_timestamp_features=torch.from_numpy(_kronos_timestamps(asset_window)).unsqueeze(0),
        asset_attention_mask=mask.clone(),
        benchmark_series=torch.from_numpy(_ohlcv_values(benchmark_window)).unsqueeze(0),
        benchmark_timestamp_features=torch.from_numpy(
            _kronos_timestamps(benchmark_window)
        ).unsqueeze(0),
        benchmark_attention_mask=mask.clone(),
        symbol=selected_symbol,
        benchmark_symbol=benchmark_symbol,
        asset_type=asset_types[0],
        window_start_at=start_at,
        cutoff_at=cutoff_at,
        metadata={**_metadata(raw_window), "benchmark_policy": decision.policy},
        benchmark_metadata=_metadata(benchmark_raw_window),
    )


def _forecast_payload(
    alpha_quantiles: Tensor,
    *,
    horizons: list[int],
    quantile_levels: list[float],
    signal_threshold: float,
) -> dict[str, Any]:
    values = np.asarray(alpha_quantiles.detach().float().cpu()[0].tolist(), dtype=np.float64)
    if values.shape != (len(horizons), len(quantile_levels)):
        raise ValueError("Model output does not match the configured horizon/quantile contract")
    signals = postprocess_alpha_signal(values, threshold=signal_threshold)
    by_horizon: dict[str, Any] = {}
    for index, horizon in enumerate(horizons):
        quantiles = {
            f"q{round(level * 100):02d}": float(value)
            for level, value in zip(quantile_levels, values[index], strict=True)
        }
        by_horizon[f"{horizon}d"] = {
            "alpha_quantiles": quantiles,
            "central_interval": {
                "nominal_coverage": float(quantile_levels[-1] - quantile_levels[0]),
                "lower": float(values[index, 0]),
                "upper": float(values[index, -1]),
            },
            "derived_signal": POSTPROCESS_SIGNAL_NAMES[int(signals[index])],
        }
    return {
        "units": "benchmark_relative_adjusted_log_return",
        "signal_threshold": signal_threshold,
        "by_horizon": by_horizon,
    }


@torch.inference_mode()
def infer_observation(
    config: ExperimentConfig,
    checkpoint: str | Path,
    observation: CausalObservation,
) -> dict[str, Any]:
    report = run_preflight(config, require_data=True)
    report.require_success()
    set_global_seed(config.training.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = build_model_bundle(config, device)
    resolved_checkpoint = resolve_checkpoint(
        checkpoint,
        saved_model_root=config.training.output_root,
    )
    checkpoint_state = load_checkpoint(
        resolved_checkpoint,
        bundle.model,
        config=config,
    )
    bundle.model.eval()
    with _autocast_context(config, device):
        output = bundle.model(
            observation.asset_series.to(device),
            observation.benchmark_series.to(device),
            asset_attention_mask=observation.asset_attention_mask.to(device),
            benchmark_attention_mask=observation.benchmark_attention_mask.to(device),
            asset_timestamps=observation.asset_timestamp_features.to(device),
            benchmark_timestamps=observation.benchmark_timestamp_features.to(device),
        )
    training_data = provenance_summary(config.data.resolved_manifest_path)
    return {
        "model_output_schema_version": MODEL_OUTPUT_SCHEMA_VERSION,
        "symbol": observation.symbol,
        "benchmark_symbol": observation.benchmark_symbol,
        "asset_type": observation.asset_type,
        "window_start_at": observation.window_start_at,
        "cutoff_at": observation.cutoff_at,
        "execution_contract": {
            "signal": "after_close_t",
            "entry": "regular_session_open_t_plus_1",
            "entry_day_counts_as_holding_day_one": True,
            "exit": "regular_session_close_t_plus_h",
        },
        "forecast": _forecast_payload(
            output.alpha_quantiles,
            horizons=config.data.alpha_horizons,
            quantile_levels=config.model.alpha_quantiles,
            signal_threshold=config.model.postprocess_alpha_threshold,
        ),
        "data_provenance": {
            "training_dataset": training_data,
            "inference_asset": observation.metadata,
            "inference_benchmark": observation.benchmark_metadata,
        },
        "encoder_contract": {
            "asset_last_hidden_state_shape": list(output.asset_last_hidden_state.shape),
            "benchmark_last_hidden_state_shape": list(output.benchmark_last_hidden_state.shape),
            "asset_latent_tokens_shape": list(output.asset_latent_tokens.shape),
            "benchmark_latent_tokens_shape": list(output.benchmark_latent_tokens.shape),
            "conditioned_latent_tokens_shape": list(output.conditioned_latent_tokens.shape),
            "reusable_method": "encode_ohlcv",
            "future_multimodal_interface": "conditioned_latent_tokens",
        },
        "checkpoint": {
            "path": str(resolved_checkpoint),
            "global_step": checkpoint_state.get("global_step"),
            "epoch": checkpoint_state.get("epoch"),
            "training_stage": checkpoint_state.get("training_stage"),
            "model_architecture_sha256": checkpoint_state.get("model_architecture_sha256"),
        },
        "device": str(device),
    }


def infer_file(
    config: ExperimentConfig,
    checkpoint: str | Path,
    input_path: str | Path,
    *,
    symbol: str | None = None,
    as_of: str | None = None,
) -> dict[str, Any]:
    observation = prepare_causal_observation(
        read_market_data(input_path),
        input_length=config.data.input_length,
        symbol=symbol,
        as_of=as_of,
        benchmark_mapping=_load_mapping(config.data.benchmark_mapping_path),
    )
    return infer_observation(config, checkpoint, observation)


def _write_payload(payload: dict[str, Any], output: Path | None) -> None:
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(f"{rendered}\n", encoding="utf-8")
    print(rendered)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = ExperimentConfig.from_yaml(args.config)
    payload = infer_file(
        config,
        args.checkpoint,
        args.input,
        symbol=args.symbol,
        as_of=args.as_of,
    )
    _write_payload(payload, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
