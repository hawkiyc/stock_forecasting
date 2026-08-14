"""Evaluate a trained adapter on a chronological validation or test split."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sized
from pathlib import Path
from typing import Any, Literal, cast

import torch

from stock_forecasting.checkpointing import load_checkpoint, validate_checkpoint_selection
from stock_forecasting.config import ExperimentConfig
from stock_forecasting.data.manifest import provenance_summary
from stock_forecasting.evaluation_paths import (
    checkpoint_run_id,
    validate_checkpoint_evaluation_output,
)
from stock_forecasting.factory import build_model_bundle
from stock_forecasting.preflight import run_preflight
from stock_forecasting.run_paths import validate_checkpoint_path, validate_run_id
from stock_forecasting.training import build_dataloaders, evaluate_loader, set_global_seed

EvaluationSplit = Literal["validation", "test"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Resolved experiment YAML configuration.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help=(
            "Exact <run_id>/checkpoint-NNNNNN directory or a canonical run directory "
            "containing validation-selected best-checkpoint.json."
        ),
    )
    parser.add_argument(
        "--split",
        choices=("validation", "test"),
        default="validation",
        help="Chronological split to evaluate. Train is intentionally excluded.",
    )
    parser.add_argument(
        "--confirm-test",
        action="store_true",
        help="Required with --split test to reduce accidental repeated test-set evaluation.",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON report destination.")
    return parser


def resolve_checkpoint(path: str | Path, *, saved_model_root: str | Path) -> Path:
    """Resolve only a canonical exact or validation-selected checkpoint path."""

    source = Path(path).expanduser().resolve(strict=False)
    expected_saved_model_root = Path(saved_model_root).expanduser().resolve(strict=False)
    if (source / "adapter.safetensors").is_file():
        run_id = validate_run_id(source.parent.name)
        resolved = validate_checkpoint_path(
            source,
            saved_model_root=expected_saved_model_root,
            run_id=run_id,
        )
        validate_checkpoint_selection(source.parent, retained_checkpoint=source.name)
        return resolved
    run_id = validate_run_id(source.name)
    if source.parent != expected_saved_model_root:
        raise ValueError("Checkpoint run directory must be stored directly under SAVED_MODEL_ROOT")
    best_pointer = source / "best-checkpoint.json"
    if best_pointer.is_file():
        checkpoint_name = validate_checkpoint_selection(source)
        candidate = validate_checkpoint_path(
            source / checkpoint_name,
            saved_model_root=expected_saved_model_root,
            run_id=run_id,
        )
        if (candidate / "adapter.safetensors").is_file():
            return candidate
        raise FileNotFoundError(f"Best-checkpoint pointer targets a missing directory: {candidate}")
    if not source.is_dir():
        raise FileNotFoundError(f"Checkpoint path does not exist: {source}")
    raise FileNotFoundError(
        f"Canonical validation pointer is missing below checkpoint run directory: {source}"
    )


def _checkpoint_summary(checkpoint: Path, state: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": str(checkpoint),
        "global_step": state.get("global_step"),
        "epoch": state.get("epoch"),
        "created_at": state.get("created_at"),
    }


def evaluate_checkpoint(
    config: ExperimentConfig,
    checkpoint: str | Path,
    *,
    split: EvaluationSplit = "test",
) -> dict[str, Any]:
    """Load trainable weights and evaluate without changing split membership."""

    if split not in ("validation", "test"):
        raise ValueError("Evaluation split must be 'validation' or 'test'.")
    report = run_preflight(
        config,
        require_data=True,
        enforce_runtime_limit=False,
    )
    report.require_success()
    set_global_seed(config.training.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _train_loader, validation_loader, test_loader = build_dataloaders(config)
    loader = validation_loader if split == "validation" else test_loader
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
    metrics = evaluate_loader(
        bundle,
        loader,
        config,
        device,
    )
    return {
        "run_id": checkpoint_run_id(resolved_checkpoint),
        "split": split,
        "samples": len(cast(Sized, loader.dataset)),
        "device": str(device),
        "checkpoint": _checkpoint_summary(resolved_checkpoint, checkpoint_state),
        "data_provenance": provenance_summary(config.data.resolved_manifest_path),
        "model_architecture_sha256": config.model.architecture_digest(),
        "metrics": metrics,
    }


def _write_payload(payload: dict[str, Any], output: Path | None) -> None:
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(f"{rendered}\n", encoding="utf-8")
    print(rendered)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.split == "test" and not args.confirm_test:
        raise SystemExit("--split test requires --confirm-test")
    config = ExperimentConfig.from_yaml(args.config)
    resolved_checkpoint = resolve_checkpoint(
        args.checkpoint,
        saved_model_root=config.training.output_root,
    )
    output = None
    if args.output is not None:
        output, _run_id = validate_checkpoint_evaluation_output(
            args.output,
            validation_output_root=config.validation.output_root,
            checkpoint=resolved_checkpoint,
            kind=args.split,
        )
    payload = evaluate_checkpoint(config, resolved_checkpoint, split=args.split)
    _write_payload(payload, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
