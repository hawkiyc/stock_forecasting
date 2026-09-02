"""Checkpoint only the trainable numerical adapter and forecast components."""

from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import random
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from stock_forecasting.config import ExperimentConfig
from stock_forecasting.models import MODEL_OUTPUT_SCHEMA_VERSION
from stock_forecasting.run_contract import (
    CHECKPOINT_ARTIFACT_SCHEMA_VERSION,
    training_resume_contract_digest,
)
from stock_forecasting.run_paths import (
    CHECKPOINT_NAME_PATTERN,
    validate_run_id,
)

CHECKPOINT_MANIFEST = "checkpoint-leaderboard.json"
BEST_CHECKPOINT_POINTER = "best-checkpoint.json"
CHECKPOINT_TRANSACTION = ".checkpoint-transaction.json"
COMPLETION_RESULT_DIRECTORY = "completion-result"
COMPLETION_RESULT_FILE = "training-result.json"
REQUIRED_CHECKPOINT_FILES = (
    "adapter.safetensors",
    "optimizer.pt",
    "scheduler.pt",
    "trainer-state.json",
    "resolved-config.yaml",
)
HASHED_CHECKPOINT_FILES = (
    "adapter.safetensors",
    "optimizer.pt",
    "scheduler.pt",
    "resolved-config.yaml",
)


def _is_non_negative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _runtime_robust_scales(payload: dict[str, Any], label: str) -> list[float]:
    values = payload.get("runtime_robust_scales")
    if (
        not isinstance(values, list)
        or not 12 <= len(values) <= 14
        or any(
            not isinstance(value, int | float)
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) <= 0.0
            for value in values
        )
    ):
        raise ValueError(f"{label} has invalid runtime robust scales")
    return [float(value) for value in values]


def _model_runtime_robust_scales(model: torch.nn.Module) -> list[float]:
    alpha_head = getattr(model, "alpha_head", None)
    values = getattr(alpha_head, "robust_scales", None)
    if not isinstance(values, torch.Tensor) or values.ndim != 1:
        raise ValueError("Quant model has no one-dimensional runtime robust scales")
    scales = [float(value) for value in values.detach().float().cpu().tolist()]
    return _runtime_robust_scales(
        {"runtime_robust_scales": scales},
        "Quant model",
    )


def _restore_runtime_robust_scales(
    model: torch.nn.Module,
    trainer_state: dict[str, Any],
    *,
    require_match: bool,
) -> None:
    stored = _runtime_robust_scales(trainer_state, "Trainer state")
    alpha_head = getattr(model, "alpha_head", None)
    current = getattr(alpha_head, "robust_scales", None)
    if not isinstance(current, torch.Tensor) or current.ndim != 1:
        raise ValueError("Quant model has no one-dimensional runtime robust scales")
    restored = current.new_tensor(stored)
    if restored.shape != current.shape:
        raise ValueError("Checkpoint runtime robust scales do not match model horizons")
    if require_match and not torch.allclose(current, restored, rtol=0.0, atol=1e-8):
        raise ValueError("Checkpoint runtime robust scales differ from train calibration")
    current.copy_(restored)


def _file_integrity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size_bytes = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            size_bytes += len(chunk)
    return {"sha256": digest.hexdigest(), "size_bytes": size_bytes}


def _validate_checkpoint_file_integrity(source: Path, state: dict[str, Any]) -> None:
    artifact_files = state.get("artifact_files")
    if not isinstance(artifact_files, dict) or set(artifact_files) != set(HASHED_CHECKPOINT_FILES):
        raise ValueError("Trainer state has an incomplete checkpoint integrity manifest")
    for name in HASHED_CHECKPOINT_FILES:
        expected = artifact_files.get(name)
        if (
            not isinstance(expected, dict)
            or set(expected) != {"sha256", "size_bytes"}
            or not isinstance(expected.get("sha256"), str)
            or len(expected["sha256"]) != 64
            or any(character not in "0123456789abcdef" for character in expected["sha256"])
            or not isinstance(expected.get("size_bytes"), int)
            or isinstance(expected["size_bytes"], bool)
            or expected["size_bytes"] <= 0
            or _file_integrity(source / name) != expected
        ):
            raise ValueError(f"Checkpoint file integrity mismatch: {name}")


def _validate_validation_selection(payload: Any, label: str) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != {
        "source",
        "metric",
        "value",
        "mode",
    }:
        raise ValueError(f"{label} has no complete validation selection")
    value = payload["value"]
    if (
        payload["source"] != "validation"
        or not isinstance(payload["metric"], str)
        or not payload["metric"].startswith("primary_5d/")
        or payload["mode"] not in {"min", "max"}
        or not isinstance(value, int | float)
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{label} was not selected from a validation primary_5d metric")
    return payload


def _capture_rng_state() -> dict[str, Any]:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "state": numpy_state[1].tolist(),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state().tolist(),
        "torch_cuda": (
            [state.tolist() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else []
        ),
    }


def _nested_tuple(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_nested_tuple(item) for item in value)
    return value


def _validate_rng_state_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != {
        "python",
        "numpy",
        "torch_cpu",
        "torch_cuda",
    }:
        raise ValueError("Checkpoint RNG state is missing or invalid")
    numpy_state = payload["numpy"]
    python_state = payload["python"]
    torch_cpu = payload["torch_cpu"]
    torch_cuda = payload["torch_cuda"]
    if (
        not isinstance(python_state, list | tuple)
        or len(python_state) != 3
        or not isinstance(numpy_state, dict)
        or set(numpy_state)
        != {"bit_generator", "state", "position", "has_gauss", "cached_gaussian"}
        or not isinstance(numpy_state["bit_generator"], str)
        or not isinstance(numpy_state["state"], list)
        or not all(_is_non_negative_int(value) for value in numpy_state["state"])
        or not _is_non_negative_int(numpy_state["position"])
        or numpy_state["has_gauss"] not in {0, 1}
        or not isinstance(numpy_state["cached_gaussian"], int | float)
        or isinstance(numpy_state["cached_gaussian"], bool)
        or not isinstance(torch_cpu, list)
        or not torch_cpu
        or not all(_is_non_negative_int(value) and value <= 255 for value in torch_cpu)
        or not isinstance(torch_cuda, list)
        or not all(
            isinstance(device_state, list)
            and device_state
            and all(_is_non_negative_int(value) and value <= 255 for value in device_state)
            for device_state in torch_cuda
        )
    ):
        raise ValueError("Checkpoint RNG state is incomplete")
    try:
        python_validator = random.Random()
        python_validator.setstate(_nested_tuple(python_state))
        numpy_validator = np.random.RandomState()
        numpy_validator.set_state(
            (
                str(numpy_state["bit_generator"]),
                np.asarray(numpy_state["state"], dtype=np.uint32),
                int(numpy_state["position"]),
                int(numpy_state["has_gauss"]),
                float(numpy_state["cached_gaussian"]),
            )
        )
        torch_validator = torch.Generator(device="cpu")
        torch_validator.set_state(torch.tensor(torch_cpu, dtype=torch.uint8))
    except (RuntimeError, TypeError, ValueError) as error:
        raise ValueError("Checkpoint RNG state is invalid") from error
    if torch.cuda.is_available() and len(torch_cuda) != torch.cuda.device_count():
        raise ValueError("Checkpoint CUDA RNG state does not match the visible devices")
    return payload


def _restore_rng_state(payload: Any) -> None:
    state = _validate_rng_state_payload(payload)
    numpy_state = state["numpy"]
    python_state = state["python"]
    torch_cpu = state["torch_cpu"]
    torch_cuda = state["torch_cuda"]
    try:
        random.setstate(_nested_tuple(python_state))
        np.random.set_state(
            (
                str(numpy_state["bit_generator"]),
                np.asarray(numpy_state["state"], dtype=np.uint32),
                int(numpy_state["position"]),
                int(numpy_state["has_gauss"]),
                float(numpy_state["cached_gaussian"]),
            )
        )
        torch.set_rng_state(torch.tensor(torch_cpu, dtype=torch.uint8))
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(
                [torch.tensor(state, dtype=torch.uint8) for state in torch_cuda]
            )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Checkpoint RNG state is invalid") from error


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"{label} is missing: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload


def _artifact_contract_digest(payload: dict[str, Any], label: str) -> str:
    if payload.get("schema_version") != CHECKPOINT_ARTIFACT_SCHEMA_VERSION:
        raise ValueError(f"{label} does not use the current checkpoint artifact schema")
    digest = payload.get("training_resume_contract_sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(f"{label} has an invalid training resume contract digest")
    return digest


def validate_checkpoint_trainer_state(
    checkpoint_dir: str | Path,
) -> dict[str, Any]:
    """Bind trainer state identity and global step to its checkpoint directory."""

    source = Path(checkpoint_dir)
    run_id = validate_run_id(source.parent.name)
    if CHECKPOINT_NAME_PATTERN.fullmatch(source.name) is None:
        raise ValueError("Checkpoint directory has a non-canonical name")
    state = _read_json_object(source / "trainer-state.json", "Trainer state")
    if state.get("run_id") != run_id or state.get("run_key") != run_id:
        raise ValueError("Trainer state identity does not match its canonical run directory")
    _artifact_contract_digest(state, "Trainer state")
    if state.get("model_output_schema_version") != MODEL_OUTPUT_SCHEMA_VERSION:
        raise ValueError("Trainer state does not use the conditional alpha output schema")
    global_step = state.get("global_step")
    epoch = state.get("epoch")
    batch_index = state.get("batch_index")
    if (
        not _is_non_negative_int(global_step)
        or f"checkpoint-{global_step:06d}" != source.name
        or not _is_non_negative_int(epoch)
        or not _is_non_negative_int(batch_index)
        or state.get("training_stage") not in {"stage1", "stage2"}
    ):
        raise ValueError("Trainer state step or batch position is not canonical")
    _validate_rng_state_payload(state.get("rng_state"))
    _validate_validation_selection(state.get("selection"), "Trainer state")
    _runtime_robust_scales(state, "Trainer state")
    _validate_checkpoint_file_integrity(source, state)
    return state


def validate_checkpoint_selection(
    run_directory: str | Path,
    *,
    retained_checkpoint: str | None = None,
    require_latest_step: bool = False,
) -> str:
    """Validate one run's manifest, leaderboard, and best-checkpoint pointer."""

    root = Path(run_directory)
    run_id = validate_run_id(root.name)
    transaction_path = root / CHECKPOINT_TRANSACTION
    if transaction_path.exists():
        pending = _read_json_object(transaction_path, "Checkpoint transaction")
        pending_monitor = pending.get("monitor")
        pending_mode = pending.get("mode")
        pending_save_top_k = pending.get("save_top_k")
        if not isinstance(pending_monitor, str) or not isinstance(pending_mode, str):
            raise ValueError("Pending checkpoint transaction has an invalid policy")
        _validate_checkpoint_policy(
            monitor=pending_monitor,
            mode=pending_mode,
            save_top_k=pending_save_top_k,
        )
        _apply_checkpoint_transaction(
            root,
            monitor=pending_monitor,
            mode=pending_mode,
            save_top_k=pending_save_top_k,
        )
    run_manifest = _read_json_object(root / "run-manifest.json", "Run manifest")
    leaderboard = _read_json_object(root / CHECKPOINT_MANIFEST, "Checkpoint leaderboard")
    pointer = _read_json_object(root / BEST_CHECKPOINT_POINTER, "Best-checkpoint pointer")
    for label, payload in (
        ("Run manifest", run_manifest),
        ("Checkpoint leaderboard", leaderboard),
        ("Best-checkpoint pointer", pointer),
    ):
        if payload.get("run_id") != run_id or payload.get("run_key") != run_id:
            raise ValueError(f"{label} identity does not match its canonical run directory")
    contract_digests = {
        _artifact_contract_digest(run_manifest, "Run manifest"),
        _artifact_contract_digest(leaderboard, "Checkpoint leaderboard"),
        _artifact_contract_digest(pointer, "Best-checkpoint pointer"),
    }
    if len(contract_digests) != 1:
        raise ValueError("Checkpoint artifacts use different training resume contracts")
    contract_digest = next(iter(contract_digests))
    if (
        pointer.get("selection_source") != "validation"
        or leaderboard.get("selection_source") != "validation"
    ):
        raise ValueError("Checkpoint selection must use validation metrics")
    monitor = pointer.get("monitor")
    mode = pointer.get("mode")
    if monitor != leaderboard.get("monitor") or mode != leaderboard.get("mode"):
        raise ValueError("Best-checkpoint pointer and leaderboard selection policy disagree")
    transaction_id = leaderboard.get("transaction_id")
    if (
        not isinstance(transaction_id, str)
        or len(transaction_id) != 32
        or any(character not in "0123456789abcdef" for character in transaction_id)
        or pointer.get("transaction_id") != transaction_id
    ):
        raise ValueError("Best-checkpoint pointer and leaderboard transaction disagree")
    _validate_checkpoint_policy(
        monitor=str(monitor),
        mode=str(mode),
        save_top_k=leaderboard.get("save_top_k"),
    )
    checkpoints = leaderboard.get("checkpoints")
    if not isinstance(checkpoints, list) or not checkpoints:
        raise ValueError("Checkpoint leaderboard has no retained checkpoints")
    if len(checkpoints) > leaderboard["save_top_k"]:
        raise ValueError("Checkpoint leaderboard exceeds its validation retention cap")
    names: list[str] = []
    for expected_rank, row in enumerate(checkpoints, start=1):
        if not isinstance(row, dict):
            raise ValueError("Checkpoint leaderboard contains an invalid row")
        name = row.get("path")
        value = row.get("value")
        global_step = row.get("global_step")
        if (
            not isinstance(name, str)
            or CHECKPOINT_NAME_PATTERN.fullmatch(name) is None
            or row.get("rank") != expected_rank
            or not isinstance(global_step, int)
            or isinstance(global_step, bool)
            or global_step < 0
            or f"checkpoint-{global_step:06d}" != name
            or not isinstance(value, int | float)
            or isinstance(value, bool)
            or not math.isfinite(float(value))
        ):
            raise ValueError("Checkpoint leaderboard contains a non-canonical row")
        names.append(name)
    if len(names) != len(set(names)):
        raise ValueError("Checkpoint leaderboard repeats a checkpoint")
    expected_order = sorted(checkpoints, key=lambda row: _checkpoint_sort_key(row, str(mode)))
    if names != [str(row["path"]) for row in expected_order]:
        raise ValueError("Checkpoint leaderboard ranks do not follow the validation policy")
    best = pointer.get("path")
    if (
        not isinstance(best, str)
        or CHECKPOINT_NAME_PATTERN.fullmatch(best) is None
        or leaderboard.get("best_checkpoint") != best
        or names[0] != best
    ):
        raise ValueError("Best-checkpoint pointer and leaderboard disagree")
    pointer_value = pointer.get("value")
    if (
        not isinstance(pointer_value, int | float)
        or isinstance(pointer_value, bool)
        or not math.isfinite(float(pointer_value))
        or float(pointer_value) != float(checkpoints[0]["value"])
    ):
        raise ValueError("Best-checkpoint pointer and leaderboard values disagree")
    if retained_checkpoint is not None and (
        CHECKPOINT_NAME_PATTERN.fullmatch(retained_checkpoint) is None
        or retained_checkpoint not in names
    ):
        raise ValueError("Requested checkpoint is not retained by the validation leaderboard")
    if retained_checkpoint is not None and require_latest_step:
        latest = max(checkpoints, key=lambda row: int(row["global_step"]))
        if retained_checkpoint != latest["path"]:
            raise ValueError("Training resume requires the retained checkpoint with maximum step")
    for row in checkpoints:
        checkpoint_name = str(row["path"])
        checkpoint_directory = root / checkpoint_name
        missing_or_empty = [
            name
            for name in REQUIRED_CHECKPOINT_FILES
            if not (checkpoint_directory / name).is_file()
            or (checkpoint_directory / name).stat().st_size <= 0
        ]
        if missing_or_empty:
            raise ValueError(
                f"Checkpoint {checkpoint_name} is incomplete: {', '.join(missing_or_empty)}"
            )
        state = validate_checkpoint_trainer_state(checkpoint_directory)
        selection = state["selection"]
        selection_value = selection.get("value")
        if (
            state.get("training_resume_contract_sha256") != contract_digest
            or state.get("global_step") != row.get("global_step")
            or selection.get("metric") != monitor
            or selection.get("mode") != mode
            or not isinstance(selection_value, int | float)
            or isinstance(selection_value, bool)
            or not math.isfinite(float(selection_value))
            or float(selection_value) != float(row["value"])
        ):
            raise ValueError(
                f"Trainer state and validation leaderboard disagree for {checkpoint_name}"
            )
    return best


def trainable_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Return CPU tensors for parameters that are intentionally trainable."""

    trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    return {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in model.state_dict().items()
        if name in trainable_names
    }


def _stage_checkpoint(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    config: ExperimentConfig,
    run_directory: Path,
    global_step: int,
    epoch: int,
    batch_index: int,
    selection_metric_name: str,
    selection_metric_value: float,
    selection_metric_mode: str,
    metrics: dict[str, float] | None = None,
    training_progress: dict[str, Any] | None = None,
) -> tuple[Path, Path, dict[str, Any]]:
    """Write and validate a checkpoint without publishing its canonical directory."""

    if not all(_is_non_negative_int(value) for value in (global_step, epoch, batch_index)):
        raise ValueError("global_step, epoch, and batch_index must be non-negative integers")
    if scheduler is None:
        raise ValueError("Canonical checkpoints require scheduler state")
    run_id = validate_run_id(run_directory.name)
    contract_digest = training_resume_contract_digest(config)
    run_manifest = _read_json_object(run_directory / "run-manifest.json", "Run manifest")
    if run_manifest.get("run_id") != run_id or run_manifest.get("run_key") != run_id:
        raise ValueError("Run manifest identity does not match its canonical run directory")
    if _artifact_contract_digest(run_manifest, "Run manifest") != contract_digest:
        raise ValueError("Run manifest does not match the current training resume contract")
    if not selection_metric_name.startswith("primary_5d/"):
        raise ValueError("Checkpoint selection must use a validation primary_5d metric")
    if (
        not isinstance(selection_metric_value, int | float)
        or isinstance(selection_metric_value, bool)
        or not math.isfinite(float(selection_metric_value))
    ):
        raise ValueError("A finite validation selection metric is required for checkpointing")
    if selection_metric_mode not in {"min", "max"}:
        raise ValueError("selection_metric_mode must be min or max")
    selection = {
        "source": "validation",
        "metric": selection_metric_name,
        "value": float(selection_metric_value),
        "mode": selection_metric_mode,
    }
    checkpoint_dir = run_directory / f"checkpoint-{global_step:06d}"
    if checkpoint_dir.exists():
        raise FileExistsError(f"Checkpoint already exists: {checkpoint_dir}")
    staging_dir = run_directory / f".{checkpoint_dir.name}.staging-{uuid.uuid4().hex}"
    staging_dir.mkdir(parents=True, exist_ok=False)
    trainer_state = {
        "schema_version": CHECKPOINT_ARTIFACT_SCHEMA_VERSION,
        "run_id": run_id,
        "run_key": run_id,
        "training_resume_contract_sha256": contract_digest,
        "model_output_schema_version": MODEL_OUTPUT_SCHEMA_VERSION,
        "global_step": global_step,
        "epoch": epoch,
        "batch_index": batch_index,
        "training_stage": config.training.stage,
        "model_architecture_sha256": config.model_architecture_digest(),
        "metrics": metrics or {},
        "training_progress": training_progress or {},
        "created_at": datetime.now(UTC).isoformat(),
        "time_series_model_id": config.model.time_series_model_id,
        "time_series_tokenizer_id": config.model.time_series_tokenizer_id,
        "time_series_model_revision": config.model.time_series_model_revision,
        "time_series_tokenizer_revision": config.model.time_series_tokenizer_revision,
        "kronos_source_revision": config.model.kronos_source_revision,
        "runtime_robust_scales": _model_runtime_robust_scales(model),
        "selection": selection,
        "rng_state": _capture_rng_state(),
    }
    try:
        save_file(trainable_state_dict(model), staging_dir / "adapter.safetensors")
        torch.save(optimizer.state_dict(), staging_dir / "optimizer.pt")
        torch.save(scheduler.state_dict(), staging_dir / "scheduler.pt")
        config.save_resolved(staging_dir / "resolved-config.yaml")
        trainer_state["artifact_files"] = {
            name: _file_integrity(staging_dir / name) for name in HASHED_CHECKPOINT_FILES
        }
        (staging_dir / "trainer-state.json").write_text(
            json.dumps(trainer_state, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        for name in REQUIRED_CHECKPOINT_FILES:
            with (staging_dir / name).open("rb") as stream:
                _fsync_descriptor(stream.fileno())
        _fsync_directory(staging_dir)
    except BaseException:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise
    row = {
        "path": checkpoint_dir.name,
        "global_step": global_step,
        "value": float(selection_metric_value),
        "created_at": trainer_state["created_at"],
    }
    return staging_dir, checkpoint_dir, row


def save_checkpoint(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    config: ExperimentConfig,
    run_directory: Path,
    global_step: int,
    epoch: int,
    batch_index: int,
    selection_metric_name: str,
    selection_metric_value: float,
    selection_metric_mode: str,
    metrics: dict[str, float] | None = None,
    training_progress: dict[str, Any] | None = None,
) -> Path:
    """Stage a self-describing checkpoint and atomically publish it."""

    staging_dir, checkpoint_dir, _row = _stage_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        run_directory=run_directory,
        global_step=global_step,
        epoch=epoch,
        batch_index=batch_index,
        selection_metric_name=selection_metric_name,
        selection_metric_value=selection_metric_value,
        selection_metric_mode=selection_metric_mode,
        metrics=metrics,
        training_progress=training_progress,
    )
    try:
        staging_dir.replace(checkpoint_dir)
        _fsync_directory(run_directory)
    except BaseException:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise
    return checkpoint_dir


def save_training_completion_result(
    *,
    model: torch.nn.Module,
    config: ExperimentConfig,
    run_directory: Path,
    global_step: int,
    completed_epochs: int,
    processed_train_samples: int,
    planned_train_samples: int,
    validation_evaluations: int,
    stop_reason: str,
    early_stopping_state: dict[str, Any],
    metrics: dict[str, float],
) -> Path:
    """Atomically store final trainable weights without another optimizer copy."""

    integer_values = (
        global_step,
        completed_epochs,
        processed_train_samples,
        planned_train_samples,
        validation_evaluations,
    )
    if not all(_is_non_negative_int(value) for value in integer_values):
        raise ValueError("Training completion counters must be non-negative integers")
    if stop_reason not in {"epochs_completed", "early_stopping"}:
        raise ValueError("Training completion stop_reason is invalid")
    if not isinstance(early_stopping_state, dict):
        raise ValueError("Training completion early-stopping state must be a mapping")
    if not isinstance(metrics, dict) or any(
        not isinstance(value, int | float)
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        for value in metrics.values()
    ):
        raise ValueError("Training completion metrics must be finite numerical values")

    run_id = validate_run_id(run_directory.name)
    contract_digest = training_resume_contract_digest(config)
    run_manifest = _read_json_object(run_directory / "run-manifest.json", "Run manifest")
    if run_manifest.get("run_id") != run_id or run_manifest.get("run_key") != run_id:
        raise ValueError("Run manifest identity does not match its canonical run directory")
    if _artifact_contract_digest(run_manifest, "Run manifest") != contract_digest:
        raise ValueError("Run manifest does not match the current training resume contract")

    result_directory = run_directory / COMPLETION_RESULT_DIRECTORY
    if result_directory.exists():
        raise FileExistsError(f"Training completion result already exists: {result_directory}")
    staging_directory = run_directory / (
        f".{COMPLETION_RESULT_DIRECTORY}.staging-{uuid.uuid4().hex}"
    )
    staging_directory.mkdir(parents=True, exist_ok=False)
    try:
        adapter_path = staging_directory / "adapter.safetensors"
        config_path = staging_directory / "resolved-config.yaml"
        save_file(trainable_state_dict(model), adapter_path)
        config.save_resolved(config_path)
        payload = {
            "schema_version": CHECKPOINT_ARTIFACT_SCHEMA_VERSION,
            "kind": "training-completion-result",
            "run_id": run_id,
            "run_key": run_id,
            "training_resume_contract_sha256": contract_digest,
            "model_output_schema_version": MODEL_OUTPUT_SCHEMA_VERSION,
            "training_stage": config.training.stage,
            "model_architecture_sha256": config.model_architecture_digest(),
            "global_step": global_step,
            "completed_epochs": completed_epochs,
            "processed_train_samples": processed_train_samples,
            "planned_train_samples": planned_train_samples,
            "validation_evaluations": validation_evaluations,
            "stop_reason": stop_reason,
            "early_stopped": stop_reason == "early_stopping",
            "early_stopping": early_stopping_state,
            "metrics": metrics,
            "runtime_robust_scales": _model_runtime_robust_scales(model),
            "created_at": datetime.now(UTC).isoformat(),
            "artifact_files": {
                "adapter.safetensors": _file_integrity(adapter_path),
                "resolved-config.yaml": _file_integrity(config_path),
            },
        }
        result_path = staging_directory / COMPLETION_RESULT_FILE
        result_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        for path in (adapter_path, config_path, result_path):
            with path.open("rb") as stream:
                _fsync_descriptor(stream.fileno())
        _fsync_directory(staging_directory)
        staging_directory.replace(result_directory)
        _fsync_directory(run_directory)
    except BaseException:
        shutil.rmtree(staging_directory, ignore_errors=True)
        raise
    return result_directory


def _fsync_descriptor(descriptor: int) -> None:
    try:
        os.fsync(descriptor)
    except OSError as error:
        if error.errno not in {errno.EINVAL, errno.ENOTSUP, errno.EROFS}:
            raise


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        _fsync_descriptor(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False, sort_keys=True)
            stream.flush()
            _fsync_descriptor(stream.fileno())
        temporary.replace(path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _ranked_checkpoint_rows(
    run_directory: Path,
    *,
    monitor: str,
    mode: str,
) -> tuple[list[dict[str, Any]], list[Path]]:
    run_id = validate_run_id(run_directory.name)
    run_manifest = _read_json_object(run_directory / "run-manifest.json", "Run manifest")
    if run_manifest.get("run_id") != run_id or run_manifest.get("run_key") != run_id:
        raise ValueError("Run manifest identity does not match its canonical run directory")
    contract_digest = _artifact_contract_digest(run_manifest, "Run manifest")
    ranked: list[dict[str, Any]] = []
    unranked: list[Path] = []
    for checkpoint in sorted(run_directory.glob("checkpoint-*")):
        if not checkpoint.is_dir():
            continue
        if CHECKPOINT_NAME_PATTERN.fullmatch(checkpoint.name) is None:
            raise ValueError(f"Checkpoint directory has a non-canonical name: {checkpoint.name}")
        state_path = checkpoint / "trainer-state.json"
        required_paths = tuple(checkpoint / name for name in REQUIRED_CHECKPOINT_FILES)
        if any(not path.is_file() or path.stat().st_size <= 0 for path in required_paths):
            unranked.append(checkpoint)
            continue
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            unranked.append(checkpoint)
            continue
        if not isinstance(state, dict):
            raise ValueError(f"Checkpoint {checkpoint} trainer state must be a JSON object")
        _validate_checkpoint_file_integrity(checkpoint, state)
        global_step = state.get("global_step")
        epoch = state.get("epoch")
        batch_index = state.get("batch_index")
        if (
            state.get("schema_version") != CHECKPOINT_ARTIFACT_SCHEMA_VERSION
            or state.get("model_output_schema_version") != MODEL_OUTPUT_SCHEMA_VERSION
            or state.get("training_resume_contract_sha256") != contract_digest
            or state.get("run_id") != run_id
            or state.get("run_key") != run_id
            or not _is_non_negative_int(global_step)
            or f"checkpoint-{global_step:06d}" != checkpoint.name
            or not _is_non_negative_int(epoch)
            or not _is_non_negative_int(batch_index)
            or state.get("training_stage") not in {"stage1", "stage2"}
        ):
            raise ValueError(f"Checkpoint {checkpoint} has inconsistent run identity or step")
        _runtime_robust_scales(state, f"Checkpoint {checkpoint}")
        selection = _validate_validation_selection(
            state.get("selection"),
            f"Checkpoint {checkpoint}",
        )
        if selection.get("metric") != monitor or selection.get("mode") != mode:
            raise ValueError(f"Checkpoint {checkpoint} uses a different ranking policy")
        selection_value = selection.get("value")
        value = float(selection_value)
        ranked.append(
            {
                "path": checkpoint.name,
                "global_step": global_step,
                "value": value,
                "created_at": state.get("created_at"),
            }
        )
    ranked.sort(
        key=lambda row: (
            row["value"] if mode == "min" else -row["value"],
            -row["global_step"],
        )
    )
    return ranked, unranked


def _validate_checkpoint_policy(*, monitor: str, mode: str, save_top_k: Any) -> None:
    if not monitor.startswith("primary_5d/"):
        raise ValueError("Checkpoint monitor must be a validation primary_5d metric")
    if (
        mode not in {"min", "max"}
        or not isinstance(save_top_k, int)
        or isinstance(save_top_k, bool)
        or save_top_k < 1
        or save_top_k > 10
    ):
        raise ValueError("Checkpoint policy requires mode=min/max and 1 <= save_top_k <= 10")


def _checkpoint_sort_key(row: dict[str, Any], mode: str) -> tuple[float, int]:
    return (
        row["value"] if mode == "min" else -row["value"],
        -row["global_step"],
    )


def _publish_checkpoint_leaderboard(
    root: Path,
    *,
    retained: list[dict[str, Any]],
    monitor: str,
    mode: str,
    save_top_k: int,
    transaction_id: str,
) -> None:
    if not retained:
        (root / CHECKPOINT_MANIFEST).unlink(missing_ok=True)
        (root / BEST_CHECKPOINT_POINTER).unlink(missing_ok=True)
        return
    best = retained[0]
    run_id = validate_run_id(root.name)
    run_manifest = _read_json_object(root / "run-manifest.json", "Run manifest")
    if run_manifest.get("run_id") != run_id or run_manifest.get("run_key") != run_id:
        raise ValueError("Run manifest identity does not match its canonical run directory")
    contract_digest = _artifact_contract_digest(run_manifest, "Run manifest")
    manifest = {
        "schema_version": CHECKPOINT_ARTIFACT_SCHEMA_VERSION,
        "run_id": run_id,
        "run_key": run_id,
        "training_resume_contract_sha256": contract_digest,
        "selection_source": "validation",
        "monitor": monitor,
        "mode": mode,
        "save_top_k": save_top_k,
        "transaction_id": transaction_id,
        "best_checkpoint": best["path"],
        "checkpoints": [{**row, "rank": rank} for rank, row in enumerate(retained, start=1)],
        "updated_at": datetime.now(UTC).isoformat(),
    }
    _atomic_json(root / CHECKPOINT_MANIFEST, manifest)
    _atomic_json(
        root / BEST_CHECKPOINT_POINTER,
        {
            "schema_version": CHECKPOINT_ARTIFACT_SCHEMA_VERSION,
            "run_id": run_id,
            "run_key": run_id,
            "training_resume_contract_sha256": contract_digest,
            "path": best["path"],
            "selection_source": "validation",
            "monitor": monitor,
            "mode": mode,
            "value": best["value"],
            "transaction_id": transaction_id,
        },
    )


def _checkpoint_transaction_payload(
    root: Path,
    *,
    retained: list[dict[str, Any]],
    remove_paths: list[Path],
    monitor: str,
    mode: str,
    save_top_k: int,
    candidate_staging: Path | None = None,
) -> dict[str, Any]:
    run_id = validate_run_id(root.name)
    run_manifest = _read_json_object(root / "run-manifest.json", "Run manifest")
    contract_digest = _artifact_contract_digest(run_manifest, "Run manifest")
    candidate = None
    if candidate_staging is not None:
        candidate_name = candidate_staging.name.removeprefix(".").split(".staging-", 1)[0]
        candidate = {
            "path": candidate_name,
            "staging_path": candidate_staging.name,
        }
    return {
        "schema_version": CHECKPOINT_ARTIFACT_SCHEMA_VERSION,
        "kind": "checkpoint-selection-transaction",
        "transaction_id": uuid.uuid4().hex,
        "run_id": run_id,
        "run_key": run_id,
        "training_resume_contract_sha256": contract_digest,
        "monitor": monitor,
        "mode": mode,
        "save_top_k": save_top_k,
        "candidate": candidate,
        "retained": retained,
        "remove": sorted({path.name for path in remove_paths}),
        "created_at": datetime.now(UTC).isoformat(),
    }


def _validate_transaction_checkpoint(
    directory: Path,
    *,
    canonical_name: str,
    row: dict[str, Any],
    run_id: str,
    contract_digest: str,
    monitor: str,
    mode: str,
) -> None:
    missing_or_empty = [
        name
        for name in REQUIRED_CHECKPOINT_FILES
        if not (directory / name).is_file() or (directory / name).stat().st_size <= 0
    ]
    if missing_or_empty:
        raise ValueError(
            f"Checkpoint transaction references an incomplete checkpoint: {canonical_name}"
        )
    state = _read_json_object(directory / "trainer-state.json", "Trainer state")
    selection = _validate_validation_selection(state.get("selection"), "Trainer state")
    state_global_step = state.get("global_step")
    if (
        state.get("schema_version") != CHECKPOINT_ARTIFACT_SCHEMA_VERSION
        or state.get("model_output_schema_version") != MODEL_OUTPUT_SCHEMA_VERSION
        or state.get("run_id") != run_id
        or state.get("run_key") != run_id
        or state.get("training_resume_contract_sha256") != contract_digest
        or not _is_non_negative_int(state_global_step)
        or state.get("training_stage") not in {"stage1", "stage2"}
        or state_global_step != row.get("global_step")
        or f"checkpoint-{state_global_step:06d}" != canonical_name
        or selection.get("metric") != monitor
        or selection.get("mode") != mode
        or float(selection["value"]) != float(row.get("value", math.nan))
    ):
        raise ValueError(f"Checkpoint transaction metadata disagrees with {canonical_name}")
    _validate_rng_state_payload(state.get("rng_state"))
    _runtime_robust_scales(state, "Trainer state")
    _validate_checkpoint_file_integrity(directory, state)


def _is_canonical_staging_name(name: str, checkpoint_name: str | None = None) -> bool:
    if "/" in name or not name.startswith(".checkpoint-") or ".staging-" not in name:
        return False
    staged_checkpoint, suffix = name.removeprefix(".").split(".staging-", 1)
    return (
        CHECKPOINT_NAME_PATTERN.fullmatch(staged_checkpoint) is not None
        and (checkpoint_name is None or staged_checkpoint == checkpoint_name)
        and len(suffix) == 32
        and all(character in "0123456789abcdef" for character in suffix)
    )


def _validate_checkpoint_transaction(
    root: Path,
    payload: dict[str, Any],
    *,
    monitor: str,
    mode: str,
    save_top_k: int,
) -> tuple[list[dict[str, Any]], dict[str, str] | None, list[str], str]:
    run_id = validate_run_id(root.name)
    run_manifest = _read_json_object(root / "run-manifest.json", "Run manifest")
    contract_digest = _artifact_contract_digest(run_manifest, "Run manifest")
    transaction_id = payload.get("transaction_id")
    if (
        payload.get("schema_version") != CHECKPOINT_ARTIFACT_SCHEMA_VERSION
        or payload.get("kind") != "checkpoint-selection-transaction"
        or payload.get("run_id") != run_id
        or payload.get("run_key") != run_id
        or payload.get("training_resume_contract_sha256") != contract_digest
        or payload.get("monitor") != monitor
        or payload.get("mode") != mode
        or payload.get("save_top_k") != save_top_k
        or not isinstance(transaction_id, str)
        or len(transaction_id) != 32
        or any(character not in "0123456789abcdef" for character in transaction_id)
    ):
        raise ValueError("Checkpoint transaction does not match the current run and policy")
    retained = payload.get("retained")
    if not isinstance(retained, list) or len(retained) > save_top_k:
        raise ValueError("Checkpoint transaction has an invalid retained set")
    names: list[str] = []
    for row in retained:
        if not isinstance(row, dict):
            raise ValueError("Checkpoint transaction contains an invalid ranking row")
        name = row.get("path")
        global_step = row.get("global_step")
        value = row.get("value")
        if (
            not isinstance(name, str)
            or CHECKPOINT_NAME_PATTERN.fullmatch(name) is None
            or not _is_non_negative_int(global_step)
            or f"checkpoint-{global_step:06d}" != name
            or not isinstance(value, int | float)
            or isinstance(value, bool)
            or not math.isfinite(float(value))
        ):
            raise ValueError("Checkpoint transaction contains a non-canonical ranking row")
        names.append(name)
    if len(names) != len(set(names)) or retained != sorted(
        retained, key=lambda row: _checkpoint_sort_key(row, mode)
    ):
        raise ValueError("Checkpoint transaction retained set is not uniquely ranked")
    candidate = payload.get("candidate")
    if candidate is not None:
        if not isinstance(candidate, dict) or set(candidate) != {"path", "staging_path"}:
            raise ValueError("Checkpoint transaction candidate is invalid")
        candidate_path = candidate.get("path")
        staging_path = candidate.get("staging_path")
        if (
            not isinstance(candidate_path, str)
            or candidate_path not in names
            or not isinstance(staging_path, str)
            or not _is_canonical_staging_name(staging_path, candidate_path)
        ):
            raise ValueError("Checkpoint transaction candidate path is invalid")
    remove = payload.get("remove")
    if not isinstance(remove, list) or not all(isinstance(name, str) for name in remove):
        raise ValueError("Checkpoint transaction removal set is invalid")
    for name in remove:
        if CHECKPOINT_NAME_PATTERN.fullmatch(name) is None and not _is_canonical_staging_name(name):
            raise ValueError("Checkpoint transaction removal path is invalid")
    if set(remove) & set(names):
        raise ValueError("Checkpoint transaction cannot remove a retained checkpoint")
    return retained, candidate, remove, transaction_id


def _promote_staged_checkpoint(staging: Path, final_path: Path, root: Path) -> None:
    staging.replace(final_path)
    _fsync_directory(root)


def _apply_checkpoint_transaction(
    root: Path,
    *,
    monitor: str,
    mode: str,
    save_top_k: int,
) -> None:
    transaction_path = root / CHECKPOINT_TRANSACTION
    payload = _read_json_object(transaction_path, "Checkpoint transaction")
    retained, candidate, remove, transaction_id = _validate_checkpoint_transaction(
        root,
        payload,
        monitor=monitor,
        mode=mode,
        save_top_k=save_top_k,
    )
    run_id = validate_run_id(root.name)
    contract_digest = str(payload["training_resume_contract_sha256"])
    candidate_name = candidate["path"] if candidate is not None else None
    candidate_staging = root / candidate["staging_path"] if candidate is not None else None
    for row in retained:
        name = str(row["path"])
        directory = root / name
        if name == candidate_name and not directory.exists():
            assert candidate_staging is not None
            directory = candidate_staging
        _validate_transaction_checkpoint(
            directory,
            canonical_name=name,
            row=row,
            run_id=run_id,
            contract_digest=contract_digest,
            monitor=monitor,
            mode=mode,
        )
    if candidate is not None:
        final_path = root / candidate["path"]
        assert candidate_staging is not None
        if final_path.exists() and candidate_staging.exists():
            raise ValueError("Checkpoint transaction has both staged and published candidates")
        if not final_path.exists():
            for name in remove:
                path = root / name
                if path.exists() and CHECKPOINT_NAME_PATTERN.fullmatch(name) is not None:
                    shutil.rmtree(path)
            _promote_staged_checkpoint(candidate_staging, final_path, root)
    _publish_checkpoint_leaderboard(
        root,
        retained=retained,
        monitor=monitor,
        mode=mode,
        save_top_k=save_top_k,
        transaction_id=transaction_id,
    )
    for name in remove:
        path = root / name
        if path.exists():
            shutil.rmtree(path)
    transaction_path.unlink()
    _fsync_directory(root)


def reconcile_checkpoint_storage(
    run_directory: str | Path,
    *,
    monitor: str,
    mode: str,
    save_top_k: int,
    current_checkpoint: str | Path | None = None,
    resume_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    """Recover pending transactions, then publish one hard-capped checkpoint set."""

    _validate_checkpoint_policy(monitor=monitor, mode=mode, save_top_k=save_top_k)
    root = Path(run_directory)
    validate_run_id(root.name)
    root.mkdir(parents=True, exist_ok=True)
    transaction_path = root / CHECKPOINT_TRANSACTION
    if transaction_path.exists():
        _apply_checkpoint_transaction(
            root,
            monitor=monitor,
            mode=mode,
            save_top_k=save_top_k,
        )
    for target_name in (
        CHECKPOINT_MANIFEST,
        BEST_CHECKPOINT_POINTER,
        CHECKPOINT_TRANSACTION,
    ):
        for temporary in root.glob(f".{target_name}.tmp-*"):
            temporary.unlink(missing_ok=True)
    ranked, unranked = _ranked_checkpoint_rows(root, monitor=monitor, mode=mode)
    staging = sorted(path for path in root.glob(".checkpoint-*.staging-*") if path.is_dir())
    retained = ranked[:save_top_k]
    latest_completed = max(ranked, key=lambda row: int(row["global_step"])) if ranked else None
    if resume_checkpoint is not None:
        requested = Path(resume_checkpoint)
        if (
            requested.parent.resolve(strict=False) != root.resolve(strict=False)
            or CHECKPOINT_NAME_PATTERN.fullmatch(requested.name) is None
        ):
            raise ValueError("Resume checkpoint must be stored directly under its canonical run")
        if latest_completed is None or requested.name != latest_completed["path"]:
            raise ValueError(
                "Training resume checkpoint must have the maximum completed global_step"
            )
        if requested.name not in {row["path"] for row in retained}:
            raise ValueError(
                "Latest completed checkpoint is not retained by the validation top-k policy"
            )
    removed_paths = [root / row["path"] for row in ranked[save_top_k:]] + unranked + staging
    transaction = _checkpoint_transaction_payload(
        root,
        retained=retained,
        remove_paths=removed_paths,
        monitor=monitor,
        mode=mode,
        save_top_k=save_top_k,
    )
    _atomic_json(transaction_path, transaction)
    _apply_checkpoint_transaction(
        root,
        monitor=monitor,
        mode=mode,
        save_top_k=save_top_k,
    )
    current_name = Path(current_checkpoint).name if current_checkpoint is not None else None
    return {
        "best_checkpoint": str(root / retained[0]["path"]) if retained else None,
        "current_retained": current_name in {row["path"] for row in retained},
        "current_rank": next(
            (rank for rank, row in enumerate(retained, start=1) if row["path"] == current_name),
            None,
        ),
        "retained_count": len(retained),
        "latest_completed_checkpoint": (
            str(root / latest_completed["path"]) if latest_completed is not None else None
        ),
        "latest_completed_global_step": (
            int(latest_completed["global_step"]) if latest_completed is not None else None
        ),
        "removed": sorted({path.name for path in removed_paths}),
        "manifest": str(root / CHECKPOINT_MANIFEST),
    }


def prepare_checkpoint_save(
    run_directory: str | Path,
    *,
    monitor: str,
    mode: str,
    save_top_k: int,
    global_step: int,
    selection_value: float,
) -> bool:
    """Decide whether a candidate qualifies without deleting a retained checkpoint."""

    if not _is_non_negative_int(global_step):
        raise ValueError("global_step must be non-negative")
    if (
        not isinstance(selection_value, int | float)
        or isinstance(selection_value, bool)
        or not math.isfinite(float(selection_value))
    ):
        raise ValueError("A finite validation selection metric is required for checkpointing")
    root = Path(run_directory)
    reconcile_checkpoint_storage(
        root,
        monitor=monitor,
        mode=mode,
        save_top_k=save_top_k,
    )
    ranked, _ = _ranked_checkpoint_rows(root, monitor=monitor, mode=mode)
    candidate = {
        "path": f"checkpoint-{global_step:06d}",
        "global_step": global_step,
        "value": float(selection_value),
        "created_at": None,
    }
    if (root / candidate["path"]).exists():
        raise FileExistsError(f"Checkpoint already exists: {root / candidate['path']}")
    combined = sorted([*ranked, candidate], key=lambda row: _checkpoint_sort_key(row, mode))
    retained_names = {row["path"] for row in combined[:save_top_k]}
    return candidate["path"] in retained_names


def save_ranked_checkpoint(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    config: ExperimentConfig,
    run_directory: Path,
    global_step: int,
    epoch: int,
    batch_index: int,
    selection_metric_name: str,
    selection_metric_value: float,
    selection_metric_mode: str,
    save_top_k: int,
    metrics: dict[str, float] | None = None,
    training_progress: dict[str, Any] | None = None,
) -> tuple[Path | None, dict[str, Any]]:
    """Commit a fully staged checkpoint and its validation ranking as one transaction."""

    _validate_checkpoint_policy(
        monitor=selection_metric_name,
        mode=selection_metric_mode,
        save_top_k=save_top_k,
    )
    if not prepare_checkpoint_save(
        run_directory,
        monitor=selection_metric_name,
        mode=selection_metric_mode,
        save_top_k=save_top_k,
        global_step=global_step,
        selection_value=selection_metric_value,
    ):
        ranked, _unranked = _ranked_checkpoint_rows(
            run_directory,
            monitor=selection_metric_name,
            mode=selection_metric_mode,
        )
        retained = ranked[:save_top_k]
        return None, {
            "best_checkpoint": str(run_directory / retained[0]["path"]),
            "current_retained": False,
            "current_rank": None,
            "retained_count": len(retained),
            "removed": [],
            "manifest": str(run_directory / CHECKPOINT_MANIFEST),
        }
    staging_dir, checkpoint_dir, candidate = _stage_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        run_directory=run_directory,
        global_step=global_step,
        epoch=epoch,
        batch_index=batch_index,
        selection_metric_name=selection_metric_name,
        selection_metric_value=selection_metric_value,
        selection_metric_mode=selection_metric_mode,
        metrics=metrics,
        training_progress=training_progress,
    )
    transaction_path = run_directory / CHECKPOINT_TRANSACTION
    transaction_published = False
    try:
        ranked, unranked = _ranked_checkpoint_rows(
            run_directory,
            monitor=selection_metric_name,
            mode=selection_metric_mode,
        )
        retained = sorted(
            [*ranked, candidate],
            key=lambda row: _checkpoint_sort_key(row, selection_metric_mode),
        )[:save_top_k]
        if candidate["path"] not in {row["path"] for row in retained}:
            raise RuntimeError("Checkpoint ranking changed while the candidate was staged")
        retained_names = {str(row["path"]) for row in retained}
        removed_paths = [
            run_directory / str(row["path"]) for row in ranked if row["path"] not in retained_names
        ]
        removed_paths.extend(unranked)
        transaction = _checkpoint_transaction_payload(
            run_directory,
            retained=retained,
            remove_paths=removed_paths,
            monitor=selection_metric_name,
            mode=selection_metric_mode,
            save_top_k=save_top_k,
            candidate_staging=staging_dir,
        )
        _atomic_json(transaction_path, transaction)
        transaction_published = True
        _apply_checkpoint_transaction(
            run_directory,
            monitor=selection_metric_name,
            mode=selection_metric_mode,
            save_top_k=save_top_k,
        )
    except BaseException:
        transaction_published = transaction_published or transaction_path.exists()
        if not transaction_published:
            shutil.rmtree(staging_dir, ignore_errors=True)
        raise
    current_rank = next(
        (rank for rank, row in enumerate(retained, start=1) if row["path"] == checkpoint_dir.name),
        None,
    )
    ranking = {
        "best_checkpoint": str(run_directory / retained[0]["path"]),
        "current_retained": current_rank is not None,
        "current_rank": current_rank,
        "retained_count": len(retained),
        "removed": sorted({path.name for path in removed_paths}),
        "manifest": str(run_directory / CHECKPOINT_MANIFEST),
    }
    return checkpoint_dir, ranking


def update_checkpoint_leaderboard(
    run_directory: str | Path,
    *,
    monitor: str,
    mode: str,
    save_top_k: int,
    current_checkpoint: str | Path,
) -> dict[str, Any]:
    """Atomically rank validation checkpoints and remove every non-top-k directory."""

    root = Path(run_directory)
    current = Path(current_checkpoint)
    if (
        current.parent.resolve(strict=False) != root.resolve(strict=False)
        or CHECKPOINT_NAME_PATTERN.fullmatch(current.name) is None
    ):
        raise ValueError("Current checkpoint must be stored directly under its canonical run")
    result = reconcile_checkpoint_storage(
        root,
        monitor=monitor,
        mode=mode,
        save_top_k=save_top_k,
        current_checkpoint=current,
    )
    if result["retained_count"] == 0:
        raise ValueError("No validation-ranked checkpoints are available")
    return result


def load_checkpoint(
    checkpoint_dir: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    *,
    config: ExperimentConfig | None = None,
) -> dict[str, Any]:
    """Load trainable state and optional optimizer/scheduler state."""

    source = Path(checkpoint_dir)
    if (optimizer is None) != (scheduler is None):
        raise ValueError("Training resume requires both optimizer and scheduler state")
    trainer_state = validate_checkpoint_trainer_state(source)
    _restore_runtime_robust_scales(
        model,
        trainer_state,
        require_match=optimizer is not None,
    )
    missing = _load_trainable_model_state(
        model,
        load_file(source / "adapter.safetensors"),
    )
    if optimizer is not None:
        optimizer.load_state_dict(
            torch.load(source / "optimizer.pt", map_location="cpu", weights_only=True)
        )
        assert scheduler is not None
        scheduler.load_state_dict(
            torch.load(source / "scheduler.pt", map_location="cpu", weights_only=True)
        )
        _restore_rng_state(trainer_state.get("rng_state"))
    if config is not None:
        _validate_loaded_model_contract(source, trainer_state, config)
    trainer_state["missing_frozen_keys"] = missing
    return trainer_state


def _validate_loaded_model_contract(
    checkpoint_dir: Path,
    trainer_state: dict[str, Any],
    config: ExperimentConfig,
) -> None:
    """Reject evaluation or inference under a different model or training stage."""

    expected_architecture = config.model_architecture_digest()
    expected_resume_contract = training_resume_contract_digest(config)
    if trainer_state.get("model_output_schema_version") != MODEL_OUTPUT_SCHEMA_VERSION:
        raise ValueError("Checkpoint predates the conditional alpha output contract")
    if trainer_state.get("training_resume_contract_sha256") != expected_resume_contract:
        raise ValueError("Checkpoint training implementation or dataset contract differs")
    if trainer_state.get("model_architecture_sha256") != expected_architecture:
        raise ValueError("Checkpoint model architecture differs from the selected config")
    if trainer_state.get("training_stage") != config.training.stage:
        raise ValueError("Checkpoint training stage differs from the selected config")
    if trainer_state.get("time_series_model_id") != config.model.time_series_model_id:
        raise ValueError("Checkpoint time-series model ID differs from the selected config")
    if trainer_state.get("time_series_tokenizer_id") != config.model.time_series_tokenizer_id:
        raise ValueError("Checkpoint time-series tokenizer ID differs from the selected config")
    if trainer_state.get("time_series_model_revision") != config.model.time_series_model_revision:
        raise ValueError("Checkpoint time-series model revision differs from the selected config")
    if (
        trainer_state.get("time_series_tokenizer_revision")
        != config.model.time_series_tokenizer_revision
    ):
        raise ValueError(
            "Checkpoint time-series tokenizer revision differs from the selected config"
        )
    if trainer_state.get("kronos_source_revision") != config.model.kronos_source_revision:
        raise ValueError("Checkpoint Kronos source revision differs from the selected config")
    stored_config = ExperimentConfig.from_yaml(checkpoint_dir / "resolved-config.yaml")
    if (
        stored_config.model_architecture_digest() != expected_architecture
        or stored_config.training.stage != config.training.stage
    ):
        raise ValueError("Checkpoint resolved config differs from its selected model contract")


def _load_trainable_model_state(
    model: torch.nn.Module,
    state: dict[str, torch.Tensor],
) -> list[str]:
    """Strictly restore the quant model's trainable parameter union."""

    expected_trainable = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    checkpoint_names = set(state)
    missing_checkpoint = sorted(expected_trainable - checkpoint_names)
    if missing_checkpoint:
        raise ValueError(f"Missing trainable checkpoint keys: {missing_checkpoint}")
    unexpected_checkpoint = sorted(checkpoint_names - expected_trainable)
    if unexpected_checkpoint:
        raise ValueError(f"Unexpected checkpoint keys: {unexpected_checkpoint}")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise ValueError(f"Unexpected checkpoint keys: {sorted(unexpected)}")
    return sorted(missing)
