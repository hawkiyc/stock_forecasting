"""Run or resume the complete validation benchmark on a GPU Pod."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from stock_forecasting.config import ExperimentConfig
from stock_forecasting.run_paths import (
    canonical_network_volume_root,
    validate_validation_lifecycle_path,
    validate_wandb_directory,
)
from stock_forecasting.validation_benchmark import (
    ALL_VALIDATION_MODELS,
    ValidationBenchmark,
    discover_checkpoint,
    discover_run_id,
    log_validation_to_wandb,
    validated_persistent_path,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--lifecycle", type=Path)
    parser.add_argument("--models", nargs="+", choices=ALL_VALIDATION_MODELS)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--recompute-full-model", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--disable-wandb", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.allow_cpu and not torch.cuda.is_available():
        raise SystemExit("Complete validation must run inside a CUDA-enabled GPU Pod")
    config = ExperimentConfig.from_yaml(args.config)
    if not args.disable_wandb and config.wandb.enabled and config.wandb.mode != "disabled":
        validate_wandb_directory(config.wandb.directory)
    volume_root = canonical_network_volume_root()
    training_lifecycle = volume_root / "lifecycle" / "stage1" / "training.json"
    run_id = discover_run_id(args.run_id, training_lifecycle=training_lifecycle)
    checkpoint = discover_checkpoint(
        args.checkpoint,
        saved_model_root=Path(os.environ.get("SAVED_MODEL_ROOT", str(volume_root / "savedModel"))),
        run_id=run_id,
    )
    checkpoint = validated_persistent_path(checkpoint, volume_root, "checkpoint")
    output = validated_persistent_path(
        args.output or config.validation.output_root / run_id / "validation-benchmark.json",
        volume_root,
        "validation output",
    )
    lifecycle = validate_validation_lifecycle_path(
        args.lifecycle or volume_root / "lifecycle" / "stage1" / "validation.json",
        network_volume_root=volume_root,
    )
    models = list(args.models or config.validation.models)
    seeds = list(args.seeds or config.validation.seeds)
    resume = config.validation.resume_completed_models if args.resume is None else args.resume
    validator = ValidationBenchmark(
        config,
        run_id=run_id,
        checkpoint=checkpoint,
        output=output,
        lifecycle=lifecycle,
        models=models,
        seeds=seeds,
        resume=resume,
        recompute_full_model=(args.recompute_full_model or config.validation.recompute_full_model),
        defer_terminal_lifecycle=bool(
            os.environ.get("RUNPOD_TMUX_LOG_FILE") and os.environ.get("RUNPOD_LAUNCH_ID")
        ),
    )
    payload = validator.run()
    if not args.disable_wandb:
        log_validation_to_wandb(config, payload, output)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
