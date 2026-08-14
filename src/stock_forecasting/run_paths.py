"""Canonical persistent paths for one training run."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,119}$")
CHECKPOINT_NAME_PATTERN = re.compile(r"^checkpoint-[0-9]{6,}$")
LIFECYCLE_SCHEMA_VERSION = 1


def validate_run_id(run_id: str) -> str:
    """Return a safe run ID without silently rewriting it."""

    if not isinstance(run_id, str) or RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise ValueError("Run ID must be a safe 1-120 character directory name")
    if "--" in run_id:
        raise ValueError("Run ID must not contain the duplicated-component separator '--'")
    return run_id


def canonical_network_volume_root() -> Path:
    """Resolve one unambiguous persistent RunPod volume root."""

    network_root_value = os.environ.get("NETWORK_VOLUME_ROOT")
    runpod_root_value = os.environ.get("RUNPOD_VOLUME_ROOT")
    if network_root_value and runpod_root_value:
        network_root = Path(network_root_value).expanduser().resolve(strict=False)
        runpod_root = Path(runpod_root_value).expanduser().resolve(strict=False)
        if network_root != runpod_root:
            raise ValueError(
                "NETWORK_VOLUME_ROOT and RUNPOD_VOLUME_ROOT must identify the same path"
            )
    root = (
        Path(network_root_value or runpod_root_value or "/runpod-volume")
        .expanduser()
        .resolve(strict=False)
    )
    workspace = Path("/workspace")
    if root == workspace or workspace in root.parents:
        raise ValueError("RunPod persistent paths must never use ephemeral /workspace")
    return root


def validate_run_environment_ids(
    wandb_run_id: str | None,
    runpod_run_key: str | None,
) -> str | None:
    """Require both runtime identity variables to describe the same run."""

    wandb_value = wandb_run_id or None
    runpod_value = runpod_run_key or None
    if wandb_value is not None:
        validate_run_id(wandb_value)
    if runpod_value is not None:
        validate_run_id(runpod_value)
    if (wandb_value is None) != (runpod_value is None):
        raise ValueError("WANDB_RUN_ID and RUNPOD_RUN_KEY must be provided together")
    if wandb_value is not None and wandb_value != runpod_value:
        raise ValueError("WANDB_RUN_ID and RUNPOD_RUN_KEY must identify the same run")
    return wandb_value


def validate_training_output_root(output_root: str | Path) -> Path:
    """Fail before writes unless RunPod uses the one canonical persistent root."""

    configured = Path(output_root).expanduser().resolve(strict=False)
    if not os.environ.get("RUNPOD_POD_ID"):
        return configured
    network_root = canonical_network_volume_root()
    expected = (network_root / "savedModel").resolve(strict=False)
    saved_model_root = (
        Path(os.environ.get("SAVED_MODEL_ROOT", str(expected))).expanduser().resolve(strict=False)
    )
    workspace = Path("/workspace")
    if configured == workspace or workspace in configured.parents:
        raise ValueError("Training output paths must never use ephemeral /workspace")
    if saved_model_root != expected:
        raise ValueError("SAVED_MODEL_ROOT must equal NETWORK_VOLUME_ROOT/savedModel")
    if configured != expected:
        raise ValueError("training.output_root must equal the canonical RunPod SAVED_MODEL_ROOT")
    return configured


def validate_wandb_directory(directory: str | Path) -> Path:
    """Require W&B's SDK root to be the RunPod volume root, never its wandb child."""

    configured = Path(directory).expanduser().resolve(strict=False)
    if not os.environ.get("RUNPOD_POD_ID"):
        return configured
    network_root = canonical_network_volume_root()
    configured_wandb_root = (
        Path(os.environ.get("WANDB_DIR", str(network_root))).expanduser().resolve(strict=False)
    )
    workspace = Path("/workspace")
    if configured == workspace or workspace in configured.parents:
        raise ValueError("W&B paths must never use ephemeral /workspace")
    if configured_wandb_root != network_root:
        raise ValueError("WANDB_DIR must equal NETWORK_VOLUME_ROOT")
    if configured != network_root:
        raise ValueError("wandb.directory must equal NETWORK_VOLUME_ROOT on RunPod")
    return configured


def validate_selected_run_environment(
    selected_run_id: str,
    wandb_run_id: str | None,
    runpod_run_key: str | None,
) -> str:
    """Bind an explicitly selected run to any active runtime identity."""

    selected = validate_run_id(selected_run_id)
    active = validate_run_environment_ids(wandb_run_id, runpod_run_key)
    if active is not None and active != selected:
        raise ValueError(
            "Selected validation run must match active WANDB_RUN_ID and RUNPOD_RUN_KEY"
        )
    return selected


def checkpoint_run_directory(saved_model_root: str | Path, run_id: str) -> Path:
    """Return the only supported checkpoint run directory."""

    return Path(saved_model_root) / validate_run_id(run_id)


def evaluation_run_directory(evaluation_root: str | Path, run_id: str) -> Path:
    """Return the only supported evaluation run directory."""

    return Path(evaluation_root) / validate_run_id(run_id)


def log_run_directory(log_root: str | Path, run_id: str) -> Path:
    """Return the only supported per-run log directory."""

    return Path(log_root) / validate_run_id(run_id)


def validate_checkpoint_path(
    checkpoint: str | Path,
    *,
    saved_model_root: str | Path,
    run_id: str,
) -> Path:
    """Require savedModel/<run_id>/checkpoint-NNNNNN exactly."""

    candidate = Path(checkpoint).expanduser().resolve(strict=False)
    expected_run_directory = checkpoint_run_directory(saved_model_root, run_id).resolve(
        strict=False
    )
    if candidate.parent != expected_run_directory:
        raise ValueError("Checkpoint must be stored directly under SAVED_MODEL_ROOT/<WANDB_RUN_ID>")
    if CHECKPOINT_NAME_PATTERN.fullmatch(candidate.name) is None:
        raise ValueError("Checkpoint directory must be named checkpoint-NNNNNN or longer")
    return candidate


def validate_training_resume_path(
    checkpoint: str | Path,
    *,
    saved_model_root: str | Path,
    run_id: str,
) -> Path:
    """Accept a ranked numerical-model checkpoint."""

    candidate = Path(checkpoint).expanduser().resolve(strict=False)
    expected_run_directory = checkpoint_run_directory(saved_model_root, run_id).resolve(
        strict=False
    )
    if candidate.parent != expected_run_directory:
        raise ValueError(
            "Resume checkpoint must be stored directly under SAVED_MODEL_ROOT/<WANDB_RUN_ID>"
        )
    if CHECKPOINT_NAME_PATTERN.fullmatch(candidate.name) is None:
        raise ValueError("Resume checkpoint must be a ranked checkpoint")
    return candidate


def validate_evaluation_path(
    path: str | Path,
    *,
    evaluation_root: str | Path,
    run_id: str,
    filename: str,
) -> Path:
    """Require evaluations/<run_id>/<filename> exactly."""

    candidate = Path(path).expanduser().resolve(strict=False)
    expected = (evaluation_run_directory(evaluation_root, run_id) / filename).resolve(strict=False)
    if candidate != expected:
        raise ValueError(f"Evaluation artifact must equal evaluations/<run_id>/{filename}")
    return candidate


def validate_validation_lifecycle_path(
    path: str | Path,
    *,
    network_volume_root: str | Path,
) -> Path:
    """Require the single lifecycle marker monitored by the validation guard."""

    candidate = Path(path).expanduser().resolve(strict=False)
    expected = (Path(network_volume_root) / "lifecycle" / "stage1" / "validation.json").resolve(
        strict=False
    )
    if candidate != expected:
        raise ValueError("Validation lifecycle must equal lifecycle/stage1/validation.json")
    return candidate


def validate_training_lifecycle_path(
    path: str | Path,
    *,
    network_volume_root: str | Path,
) -> Path:
    """Require the single lifecycle marker monitored by the training guard."""

    candidate = Path(path).expanduser().resolve(strict=False)
    expected = (Path(network_volume_root) / "lifecycle" / "stage1" / "training.json").resolve(
        strict=False
    )
    if candidate != expected:
        raise ValueError("Training lifecycle must equal lifecycle/stage1/training.json")
    return candidate


def validate_run_lifecycle_payload(
    payload: Mapping[str, Any],
    *,
    network_volume_root: str | Path,
    expected_kind: str,
) -> str:
    """Bind every persisted run-scoped lifecycle path to one canonical run ID."""

    if payload.get("schema_version") != LIFECYCLE_SCHEMA_VERSION:
        raise ValueError(f"Lifecycle schema_version must equal {LIFECYCLE_SCHEMA_VERSION}")
    if payload.get("kind") != expected_kind:
        raise ValueError(f"Lifecycle kind must equal {expected_kind}")
    run_id = validate_run_id(payload.get("wandb_run_id"))
    volume_root = Path(network_volume_root).expanduser().resolve(strict=False)
    checkpoint = payload.get("checkpoint")
    if checkpoint not in (None, ""):
        validate_checkpoint_path(
            str(checkpoint),
            saved_model_root=volume_root / "savedModel",
            run_id=run_id,
        )
    evaluation_root = volume_root / "evaluations"
    expected_evaluations = {
        "result_path": "validation-benchmark.json",
    }
    for key, filename in expected_evaluations.items():
        value = payload.get(key)
        if value not in (None, ""):
            validate_evaluation_path(
                str(value),
                evaluation_root=evaluation_root,
                run_id=run_id,
                filename=filename,
            )
    expected_log_root = (volume_root / "logs" / run_id).resolve(strict=False)
    for key in ("log_path", "recovery_path"):
        value = payload.get(key)
        if value in (None, ""):
            continue
        candidate = Path(str(value)).expanduser().resolve(strict=False)
        try:
            candidate.relative_to(expected_log_root)
        except ValueError as error:
            raise ValueError(f"Lifecycle {key} must be stored below logs/<run_id>") from error
        if key == "recovery_path" and candidate.name != "recovery.json":
            raise ValueError("Lifecycle recovery_path must identify recovery.json")
    return run_id
