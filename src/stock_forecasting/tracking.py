"""Weights & Biases integration with a durable local fallback."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import platform
import re
import shutil
import socket
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from stock_forecasting.checkpointing import validate_checkpoint_selection
from stock_forecasting.config import ExperimentConfig
from stock_forecasting.data.manifest import provenance_summary
from stock_forecasting.run_contract import (
    CHECKPOINT_ARTIFACT_SCHEMA_VERSION,
    training_resume_contract_fingerprint,
    validate_training_resume_contract,
)
from stock_forecasting.run_paths import (
    canonical_network_volume_root,
    checkpoint_run_directory,
    validate_run_environment_ids,
    validate_run_id,
    validate_training_output_root,
    validate_training_resume_path,
    validate_wandb_directory,
)


_SELECTION_ID_PATTERN = re.compile(r"selection-[0-9a-f]{16}")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def collect_system_metadata() -> dict[str, Any]:
    """Collect non-secret runtime metadata for experiment provenance."""

    metadata: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
    }
    try:
        import torch

        metadata["torch"] = torch.__version__
        metadata["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            metadata["cuda"] = torch.version.cuda
            metadata["gpu_count"] = torch.cuda.device_count()
            metadata["gpus"] = [
                {
                    "name": torch.cuda.get_device_name(index),
                    "total_memory_bytes": torch.cuda.get_device_properties(index).total_memory,
                }
                for index in range(torch.cuda.device_count())
            ]
    except ImportError:
        metadata["torch"] = "not-installed"
    return metadata


def collect_dataset_provenance(
    config: ExperimentConfig | None = None,
) -> dict[str, Any]:
    """Read the CPU-approved dataset contract without copying raw data or secrets."""

    if config is not None:
        try:
            return {"status": "ready", **provenance_summary(config.data.resolved_manifest_path)}
        except (FileNotFoundError, ValueError) as error:
            return {
                "status": "manifest-unavailable",
                "path": str(config.data.resolved_manifest_path),
                "error": str(error),
            }
    volume_root = canonical_network_volume_root()
    marker = Path(
        os.environ.get(
            "DATASET_READINESS_MANIFEST",
            str(volume_root / "lifecycle/stage1/dataset.json"),
        )
    )
    if not marker.is_file():
        return {"status": "manifest-unavailable", "path": str(marker)}
    payload = json.loads(marker.read_text(encoding="utf-8"))
    keys = (
        "dataset_profile",
        "selected_datasets",
        "providers",
        "markets",
        "date_range",
        "data_pipeline_digest",
        "preparation_spec_sha256",
        "preparation_spec",
        "universe_sha256",
        "symbols",
        "split_counts",
        "model_repositories",
        "model_manifest",
        "model_manifest_sha256",
        "raw",
        "processed",
        "dataset_manifest",
        "download_manifest",
        "request_log",
    )
    return {"status": "ready", "path": str(marker), **{key: payload.get(key) for key in keys}}


def collect_selection_provenance() -> dict[str, Any]:
    """Collect the non-secret selection identity already checked by the mounted gate."""

    selection_id = os.environ.get("RUNPOD_SELECTION_ID", "")
    selection_sha256 = os.environ.get("RUNPOD_SELECTION_SHA256", "")
    dataset_request_sha256 = os.environ.get("RUNPOD_DATASET_REQUEST_SHA256", "")
    if not selection_id and not os.environ.get("RUNPOD_POD_ID"):
        return {"status": "unavailable"}
    if _SELECTION_ID_PATTERN.fullmatch(selection_id) is None:
        raise ValueError("RunPod selection ID is unavailable or invalid")
    if _SHA256_PATTERN.fullmatch(selection_sha256) is None:
        raise ValueError("RunPod selection digest is unavailable or invalid")
    if _SHA256_PATTERN.fullmatch(dataset_request_sha256) is None:
        raise ValueError("RunPod dataset request digest is unavailable or invalid")

    volume_root = canonical_network_volume_root()
    marker = Path(
        os.environ.get(
            "DATASET_READINESS_MANIFEST",
            str(volume_root / "lifecycle/stage1/dataset.json"),
        )
    )
    if not marker.is_file():
        raise FileNotFoundError(f"RunPod dataset readiness marker is missing: {marker}")
    payload = json.loads(marker.read_text(encoding="utf-8"))
    expected = {
        "selection_id": selection_id,
        "selection_sha256": selection_sha256,
        "dataset_request_sha256": dataset_request_sha256,
        "selected_stage": os.environ.get("RUNPOD_STAGE", ""),
        "stage_config_sha256": os.environ.get("RUNPOD_STAGE_CONFIG_SHA256", ""),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"RunPod selection provenance disagrees on {key}")
    return {
        "status": "ready",
        "path": str(marker),
        **expected,
        "stage_config_path": payload.get("stage_config_path"),
        "requested_dataset": payload.get("requested_dataset"),
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _source_config_sha256() -> str | None:
    value = os.environ.get("RUNPOD_SOURCE_CONFIG_SHA256", "").strip()
    if not value:
        if os.environ.get("RUNPOD_POD_ID"):
            raise ValueError("RunPod training requires RUNPOD_SOURCE_CONFIG_SHA256")
        return None
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("RUNPOD_SOURCE_CONFIG_SHA256 must be a lowercase SHA-256 digest")
    return value


def validate_tracking_run_contract(
    config: ExperimentConfig,
) -> tuple[str | None, Path | None]:
    """Validate run identity and resume artifacts without starting W&B or loading models."""

    validate_training_output_root(config.training.output_root)
    validate_wandb_directory(config.wandb.directory)
    expected_run_id = validate_run_environment_ids(
        os.environ.get("WANDB_RUN_ID"),
        os.environ.get("RUNPOD_RUN_KEY"),
    )
    resume_checkpoint = config.training.resume_checkpoint
    if resume_checkpoint is not None and (
        not os.environ.get("WANDB_RUN_ID") or not os.environ.get("RUNPOD_RUN_KEY")
    ):
        raise ValueError("Training resume requires WANDB_RUN_ID and RUNPOD_RUN_KEY")
    if expected_run_id is None:
        if resume_checkpoint is not None:
            raise ValueError("Training resume requires WANDB_RUN_ID and RUNPOD_RUN_KEY")
        return None, None
    run_directory = checkpoint_run_directory(config.training.output_root, expected_run_id)
    if resume_checkpoint is None:
        if run_directory.exists():
            raise FileExistsError(f"Fresh run directory already exists: {run_directory}")
        return expected_run_id, run_directory
    if not run_directory.is_dir():
        raise FileNotFoundError(f"Resume run directory is missing: {run_directory}")
    validated_checkpoint = validate_training_resume_path(
        resume_checkpoint,
        saved_model_root=config.training.output_root,
        run_id=expected_run_id,
    )
    validate_checkpoint_selection(
        run_directory,
        retained_checkpoint=validated_checkpoint.name,
        require_latest_step=True,
    )
    validate_training_resume_contract(
        config,
        run_directory=run_directory,
        checkpoint_directory=validated_checkpoint,
    )
    return expected_run_id, run_directory


@contextmanager
def training_run_lease(config: ExperimentConfig) -> Iterator[None]:
    """Hold a non-blocking OS lease for the entire mutation window of one run."""

    validate_training_output_root(config.training.output_root)
    run_id = validate_run_environment_ids(
        os.environ.get("WANDB_RUN_ID"),
        os.environ.get("RUNPOD_RUN_KEY"),
    )
    if run_id is None:
        yield
        return
    lease_root = config.training.output_root / ".run-leases"
    lease_root.mkdir(parents=True, exist_ok=True)
    lease_path = lease_root / f"{run_id}.lock"
    descriptor = os.open(lease_path, os.O_CREAT | os.O_RDWR, 0o600)
    stream = os.fdopen(descriptor, "r+", encoding="utf-8")
    try:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Training run is already active: {run_id}") from error
        stream.seek(0)
        stream.truncate()
        json.dump(
            {
                "run_id": run_id,
                "pod_id": os.environ.get("RUNPOD_POD_ID", ""),
                "launch_id": os.environ.get("RUNPOD_LAUNCH_ID", ""),
                "pid": os.getpid(),
                "acquired_at": datetime.now(UTC).isoformat(),
            },
            stream,
            ensure_ascii=False,
            sort_keys=True,
        )
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
        yield
    finally:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        finally:
            stream.close()


@dataclass(frozen=True)
class _FreshRunReservation:
    final_directory: Path
    staging_directory: Path


def _reserve_fresh_run_directory(final_directory: Path) -> _FreshRunReservation:
    if final_directory.exists():
        raise FileExistsError(f"Fresh run directory already exists: {final_directory}")
    final_directory.parent.mkdir(parents=True, exist_ok=True)
    staging_directory = final_directory.parent / f".run-staging-{uuid.uuid4().hex}"
    staging_directory.mkdir(parents=False, exist_ok=False)
    return _FreshRunReservation(
        final_directory=final_directory,
        staging_directory=staging_directory,
    )


def _prepare_expected_run_directory(
    config: ExperimentConfig,
    *,
    expected_run_id: str | None,
    validated_run_directory: Path | None,
) -> _FreshRunReservation | None:
    """Reserve a fresh fixed-ID directory only after its contract passed preflight."""

    if expected_run_id is None or config.training.resume_checkpoint is not None:
        return None
    if validated_run_directory is None:
        raise RuntimeError("Expected run ID has no validated checkpoint directory")
    return _reserve_fresh_run_directory(validated_run_directory)


def _rollback_fresh_run_reservation(reservation: _FreshRunReservation | None) -> None:
    """Delete only this process's unpublished staging directory."""

    if reservation is None:
        return
    shutil.rmtree(reservation.staging_directory, ignore_errors=True)


@dataclass
class TrackingRun:
    """Small adapter around W&B that also works when tracking is disabled."""

    id: str
    name: str
    directory: Path
    backend: Any | None
    mode: str

    @property
    def key(self) -> str:
        """Use the durable W&B run ID as the sole checkpoint directory key."""

        return validate_run_id(self.id)

    def log(self, payload: dict[str, Any], step: int | None = None) -> None:
        if self.backend is not None:
            self.backend.log(payload, step=step)
        local_log = self.directory / "metrics.jsonl"
        record = {"step": step, "time": datetime.now(UTC).isoformat(), **payload}
        with local_log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def update_summary(self, payload: dict[str, Any]) -> None:
        if self.backend is not None:
            for key, value in payload.items():
                self.backend.summary[key] = value
        summary_path = self.directory / "summary.json"
        current: dict[str, Any] = {}
        if summary_path.exists():
            current = json.loads(summary_path.read_text(encoding="utf-8"))
        current.update(payload)
        summary_path.write_text(
            json.dumps(current, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
        )

    def log_model_artifact(self, checkpoint_dir: Path, aliases: list[str]) -> None:
        if self.backend is None:
            return
        import wandb

        artifact = wandb.Artifact(name=f"adapter-{self.id}", type="model")
        artifact.add_dir(str(checkpoint_dir))
        self.backend.log_artifact(artifact, aliases=aliases)

    def finish(self, exit_code: int = 0) -> None:
        if self.backend is not None:
            self.backend.finish(exit_code=exit_code)


def _start_wandb(config: ExperimentConfig) -> Any:
    validate_wandb_directory(config.wandb.directory)
    import wandb

    run_id = validate_run_environment_ids(
        os.environ.get("WANDB_RUN_ID"),
        os.environ.get("RUNPOD_RUN_KEY"),
    )
    resume_policy = "must" if config.training.resume_checkpoint is not None else "never"
    return wandb.init(
        project=config.wandb.project,
        entity=config.wandb.entity or None,
        group=config.wandb.group,
        name=config.wandb.name,
        id=run_id,
        resume=resume_policy if run_id else None,
        tags=config.wandb.tags,
        config={
            **config.as_dict(),
            "system": collect_system_metadata(),
            "dataset_provenance": collect_dataset_provenance(config),
            "selection_provenance": collect_selection_provenance(),
        },
        dir=str(config.wandb.directory),
        mode=config.wandb.mode,
    )


def start_tracking(config: ExperimentConfig) -> TrackingRun:
    """Initialize W&B and create a unique, durable checkpoint directory."""

    expected_run_id, validated_run_directory = validate_tracking_run_contract(config)
    config.wandb.directory.mkdir(parents=True, exist_ok=True)
    reservation = _prepare_expected_run_directory(
        config,
        expected_run_id=expected_run_id,
        validated_run_directory=validated_run_directory,
    )
    backend: Any | None = None
    selected_mode = config.wandb.mode

    if config.wandb.enabled and config.wandb.mode != "disabled":
        try:
            backend = _start_wandb(config)
        except Exception:
            if not config.wandb.allow_offline_fallback:
                if config.training.resume_checkpoint is None:
                    _rollback_fresh_run_reservation(reservation)
                raise
            import wandb

            selected_mode = "offline"
            run_id = expected_run_id
            resume_policy = "must" if config.training.resume_checkpoint is not None else "never"
            try:
                backend = wandb.init(
                    project=config.wandb.project,
                    entity=config.wandb.entity or None,
                    group=config.wandb.group,
                    name=config.wandb.name,
                    id=run_id,
                    resume=resume_policy if run_id else None,
                    tags=[*config.wandb.tags, "offline-fallback"],
                    config={
                        **config.as_dict(),
                        "system": collect_system_metadata(),
                        "dataset_provenance": collect_dataset_provenance(config),
                        "selection_provenance": collect_selection_provenance(),
                    },
                    dir=str(config.wandb.directory),
                    mode="offline",
                )
            except BaseException:
                if config.training.resume_checkpoint is None:
                    _rollback_fresh_run_reservation(reservation)
                raise
        except BaseException:
            if config.training.resume_checkpoint is None:
                _rollback_fresh_run_reservation(reservation)
            raise

    if backend is None:
        run_id = expected_run_id or uuid.uuid4().hex[:8]
        run_name = config.wandb.name or config.experiment_name
    else:
        run_id = str(backend.id)
        run_name = str(backend.name or config.experiment_name)
    try:
        run_id = validate_run_id(run_id)
    except BaseException:
        if backend is not None:
            backend.finish(exit_code=1)
        if config.training.resume_checkpoint is None:
            _rollback_fresh_run_reservation(reservation)
        raise
    if expected_run_id and run_id != expected_run_id:
        if backend is not None:
            backend.finish(exit_code=1)
        if config.training.resume_checkpoint is None:
            _rollback_fresh_run_reservation(reservation)
        raise RuntimeError("W&B returned a run ID different from WANDB_RUN_ID")

    provisional = TrackingRun(
        id=run_id,
        name=run_name,
        directory=config.training.output_root,
        backend=backend,
        mode=selected_mode,
    )
    run_directory = checkpoint_run_directory(config.training.output_root, provisional.id)
    resume_checkpoint = config.training.resume_checkpoint
    try:
        if resume_checkpoint is None:
            if reservation is None:
                reservation = _reserve_fresh_run_directory(run_directory)
            elif reservation.final_directory != run_directory:
                raise RuntimeError("Prepared checkpoint directory does not match the W&B run ID")
            staging_directory = reservation.staging_directory
            provisional.directory = staging_directory
            config.save_resolved(staging_directory / "resolved-config.yaml")
            resume_contract, resume_contract_sha256 = training_resume_contract_fingerprint(config)
            _atomic_json(
                staging_directory / "run-manifest.json",
                {
                    "schema_version": CHECKPOINT_ARTIFACT_SCHEMA_VERSION,
                    "run_id": run_id,
                    "run_name": run_name,
                    "run_key": provisional.key,
                    "source_config_sha256": _source_config_sha256(),
                    "resolved_config_sha256": _sha256(staging_directory / "resolved-config.yaml"),
                    "training_resume_contract": resume_contract,
                    "training_resume_contract_sha256": resume_contract_sha256,
                    "tracking_mode": selected_mode,
                    "created_at": datetime.now(UTC).isoformat(),
                    "system": collect_system_metadata(),
                    "dataset_provenance": collect_dataset_provenance(config),
                    "selection_provenance": collect_selection_provenance(),
                },
            )
            if run_directory.exists():
                raise FileExistsError(f"Fresh run directory already exists: {run_directory}")
            staging_directory.replace(run_directory)
        provisional.directory = run_directory
    except BaseException:
        if backend is not None:
            backend.finish(exit_code=1)
        if resume_checkpoint is None:
            _rollback_fresh_run_reservation(reservation)
        raise
    return provisional
