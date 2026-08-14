"""Immutable contracts for quant-only training runs and resumable checkpoints."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from stock_forecasting.config import ExperimentConfig
from stock_forecasting.data.manifest import load_dataset_manifest, sha256_file
from stock_forecasting.models import MODEL_OUTPUT_SCHEMA_VERSION
from stock_forecasting.run_paths import validate_run_id, validate_training_resume_path

CHECKPOINT_ARTIFACT_SCHEMA_VERSION = "3.0"
TRAINING_RESUME_CONTRACT_VERSION = "4.0"
TRAINING_IMPLEMENTATION_PATHS = (
    "checkpointing.py",
    "config.py",
    "data/dataset.py",
    "data/io.py",
    "data/manifest.py",
    "factory.py",
    "metrics.py",
    "models/backbones.py",
    "models/forecast.py",
    "models/lora.py",
    "models/outputs.py",
    "models/projector.py",
    "models/quant.py",
    "preflight.py",
    "run_contract.py",
    "run_paths.py",
    "tracking.py",
    "training.py",
    "training_paths.py",
)


def training_implementation_contract() -> dict[str, Any]:
    """Fingerprint the bounded source set that defines quant training semantics."""

    package_root = Path(__file__).resolve().parent
    files: dict[str, str] = {}
    for relative_path in TRAINING_IMPLEMENTATION_PATHS:
        source = package_root / relative_path
        if not source.is_file():
            raise FileNotFoundError(f"Training implementation file is missing: {source}")
        files[relative_path] = sha256_file(source)
    encoded = json.dumps(
        files,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return {
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "files": files,
    }


def _dataset_resume_contract(config: ExperimentConfig) -> dict[str, Any]:
    manifest_path = config.data.resolved_manifest_path
    if not manifest_path.is_file():
        return {
            "status": "unavailable",
            "path": str(manifest_path),
            "dataset_profile": config.data.dataset_profile,
        }
    payload = load_dataset_manifest(manifest_path, required_state="ready")
    return {
        "status": "ready",
        "path": str(manifest_path),
        "sha256": sha256_file(manifest_path),
        "dataset_profile": payload["dataset_profile"],
        "selected_datasets": payload["selected_datasets"],
        "data_pipeline_digest": payload["data_pipeline_digest"],
        "preparation_spec_sha256": payload["preparation_spec_sha256"],
        "universe_sha256": payload["universe_sha256"],
        "split_counts": payload["split_counts"],
        "artifacts": payload["artifacts"],
    }


def training_resume_contract(config: ExperimentConfig) -> dict[str, Any]:
    """Return semantics that must remain identical within one interrupted run."""

    payload = config.as_dict()
    training = dict(payload["training"])
    training.pop("resume_checkpoint", None)
    return {
        "schema_version": TRAINING_RESUME_CONTRACT_VERSION,
        "model_output_schema_version": MODEL_OUTPUT_SCHEMA_VERSION,
        "data": payload["data"],
        "model": payload["model"],
        "model_architecture_sha256": config.model.architecture_digest(),
        "training": training,
        "training_implementation": training_implementation_contract(),
        "dataset_artifacts": _dataset_resume_contract(config),
    }


def training_resume_contract_digest(config: ExperimentConfig) -> str:
    _, digest = training_resume_contract_fingerprint(config)
    return digest


def training_resume_contract_fingerprint(
    config: ExperimentConfig,
) -> tuple[dict[str, Any], str]:
    contract = training_resume_contract(config)
    encoded = json.dumps(
        contract,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return contract, hashlib.sha256(encoded).hexdigest()


def validate_training_resume_contract(
    config: ExperimentConfig,
    *,
    run_directory: str | Path,
    checkpoint_directory: str | Path,
) -> str:
    """Require run and checkpoint artifacts to match current quant semantics."""

    root = Path(run_directory).expanduser().resolve(strict=False)
    run_id = validate_run_id(root.name)
    checkpoint = validate_training_resume_path(
        checkpoint_directory,
        saved_model_root=root.parent,
        run_id=run_id,
    )
    expected_digest = training_resume_contract_digest(config)
    artifacts = (
        (root / "run-manifest.json", "Run manifest"),
        (checkpoint / "trainer-state.json", "Trainer state"),
    )
    for path, label in artifacts:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise FileNotFoundError(f"{label} is missing: {path}") from error
        if not isinstance(payload, dict):
            raise ValueError(f"{label} must contain a JSON object")
        if payload.get("run_id") != run_id or payload.get("run_key") != run_id:
            raise ValueError(f"{label} identity does not match its canonical run directory")
        if payload.get("schema_version") != CHECKPOINT_ARTIFACT_SCHEMA_VERSION:
            raise ValueError(f"{label} does not use the current checkpoint artifact schema")
        if payload.get("training_resume_contract_sha256") != expected_digest:
            raise ValueError(f"{label} does not match the current training resume contract")
    for path, label in (
        (root / "resolved-config.yaml", "Run resolved config"),
        (checkpoint / "resolved-config.yaml", "Checkpoint resolved config"),
    ):
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"{label} is missing or empty: {path}")
        stored_config = ExperimentConfig.from_yaml(path)
        if training_resume_contract_digest(stored_config) != expected_digest:
            raise ValueError(f"{label} does not match the current training resume contract")
    return expected_digest
