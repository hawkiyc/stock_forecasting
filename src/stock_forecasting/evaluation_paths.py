"""Canonical destinations for evaluation CLI artifacts."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from stock_forecasting.run_paths import (
    CHECKPOINT_NAME_PATTERN,
    canonical_network_volume_root,
    validate_evaluation_path,
    validate_run_id,
)

CheckpointEvaluationKind = Literal["validation", "test"]

CHECKPOINT_EVALUATION_FILENAMES: dict[CheckpointEvaluationKind, str] = {
    "validation": "checkpoint-validation-evaluation.json",
    "test": "checkpoint-test-evaluation.json",
}
STANDALONE_BASELINE_FILENAME = "standalone-baseline-benchmark.json"
OFFICIAL_PER_RUN_FILENAMES = frozenset(
    {
        "validation-benchmark.json",
        *CHECKPOINT_EVALUATION_FILENAMES.values(),
    }
)


def _validated_evaluation_root(validation_output_root: str | Path) -> Path:
    configured_root = Path(validation_output_root).expanduser().resolve(strict=False)
    if not os.environ.get("RUNPOD_POD_ID"):
        return configured_root
    volume_root = canonical_network_volume_root()
    canonical_root = (volume_root / "evaluations").resolve(strict=False)
    if configured_root != canonical_root:
        raise ValueError(
            "validation.output_root must equal the canonical RunPod evaluations directory"
        )
    return canonical_root


def checkpoint_run_id(checkpoint: str | Path) -> str:
    """Derive exactly one run ID from a canonical checkpoint directory."""

    resolved = Path(checkpoint).expanduser().resolve(strict=False)
    if CHECKPOINT_NAME_PATTERN.fullmatch(resolved.name) is None:
        raise ValueError("Evaluation requires a canonical checkpoint-NNNNNN directory")
    return validate_run_id(resolved.parent.name)


def validate_checkpoint_evaluation_output(
    output: str | Path,
    *,
    validation_output_root: str | Path,
    checkpoint: str | Path,
    kind: CheckpointEvaluationKind,
) -> tuple[Path, str]:
    """Bind a checkpoint report to evaluations/<run_id>/<fixed filename>."""

    run_id = checkpoint_run_id(checkpoint)
    evaluation_root = _validated_evaluation_root(validation_output_root)
    destination = validate_evaluation_path(
        output,
        evaluation_root=evaluation_root,
        run_id=run_id,
        filename=CHECKPOINT_EVALUATION_FILENAMES[kind],
    )
    return destination, run_id


def validate_standalone_baseline_output(
    output: str | Path,
    *,
    validation_output_root: str | Path,
) -> Path:
    """Keep baselines without training identity outside every official per-run directory."""

    destination = Path(output).expanduser().resolve(strict=False)
    configured_root = _validated_evaluation_root(validation_output_root)
    on_runpod = bool(os.environ.get("RUNPOD_POD_ID"))
    if on_runpod:
        expected = configured_root / STANDALONE_BASELINE_FILENAME
        if destination != expected:
            raise ValueError(
                "RunPod standalone baseline output must equal "
                "validation.output_root/standalone-baseline-benchmark.json"
            )
        return expected

    if destination.name in OFFICIAL_PER_RUN_FILENAMES:
        raise ValueError("Standalone baseline output cannot use an official per-run filename")
    try:
        relative = destination.relative_to(configured_root)
    except ValueError:
        return destination
    if len(relative.parts) != 1:
        raise ValueError("Standalone baseline output cannot be stored under evaluations/<run_id>")
    return destination
