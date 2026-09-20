"""Immutable contracts for quant-only training runs and resumable checkpoints."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from stock_forecasting.checkpoint_resume_migrations import CHECKPOINT_RETENTION_MIGRATIONS
from stock_forecasting.config import DataConfig, ExperimentConfig, ModelConfig
from stock_forecasting.data.manifest import load_dataset_manifest, sha256_file
from stock_forecasting.models import MODEL_OUTPUT_SCHEMA_VERSION
from stock_forecasting.run_paths import validate_run_id, validate_training_resume_path

CHECKPOINT_ARTIFACT_SCHEMA_VERSION = "4.0"
TRAINING_RESUME_CONTRACT_VERSION = "6.0"
TRAINING_IMPLEMENTATION_PATHS = (
    "checkpointing.py",
    "config.py",
    "data/adjustments.py",
    "data/bar_store.py",
    "data/dataset.py",
    "data/horizons.py",
    "data/manifest.py",
    "factory.py",
    "metrics.py",
    "models/backbones.py",
    "models/forecast.py",
    "models/lora.py",
    "models/outputs.py",
    "models/projector.py",
    "models/quant.py",
    "models/scale_features.py",
    "preflight.py",
    "run_contract.py",
    "run_paths.py",
    "tracking.py",
    "training.py",
    "training_paths.py",
    "training_stage_contract.py",
    "scale_calibration.py",
    "dataset_identity.py",
    "evaluation_protocol.py",
    "evaluation_store.py",
    "optimization_policy.py",
    "date_market_sampler.py",
    "models/ranking.py",
    "baseline_contract.py",
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
        "storage_preparation_spec_sha256": payload["storage_preparation_spec_sha256"],
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
        "model_architecture_sha256": config.model_architecture_digest(),
        "training": training,
        "training_implementation": training_implementation_contract(),
        "dataset_artifacts": _dataset_resume_contract(config),
    }


def training_resume_contract_digest(config: ExperimentConfig) -> str:
    _, digest = training_resume_contract_fingerprint(config)
    return digest


def _canonical_payload_digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validated_implementation_files(
    implementation: Any,
    *,
    label: str,
) -> dict[str, str]:
    if not isinstance(implementation, dict) or set(implementation) != {"sha256", "files"}:
        raise ValueError(f"{label} has an invalid training implementation contract")
    files = implementation.get("files")
    if (
        not isinstance(files, dict)
        or set(files) != set(TRAINING_IMPLEMENTATION_PATHS)
        or any(
            not isinstance(path, str)
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            for path, digest in files.items()
        )
    ):
        raise ValueError(f"{label} has invalid training implementation file digests")
    expected_digest = _canonical_payload_digest(files)
    if implementation.get("sha256") != expected_digest:
        raise ValueError(f"{label} training implementation digest is inconsistent")
    return dict(files)


def _checkpoint_retention_migration_matches(
    stored_contract: dict[str, Any],
    current_contract: dict[str, Any],
) -> bool:
    stored_semantics = {
        key: value for key, value in stored_contract.items() if key != "training_implementation"
    }
    current_semantics = {
        key: value for key, value in current_contract.items() if key != "training_implementation"
    }
    if stored_semantics != current_semantics:
        return False
    stored_files = _validated_implementation_files(
        stored_contract.get("training_implementation"),
        label="Stored training resume contract",
    )
    current_files = _validated_implementation_files(
        current_contract.get("training_implementation"),
        label="Current training resume contract",
    )
    changed_files = {path for path in stored_files if stored_files[path] != current_files[path]}
    for migration in CHECKPOINT_RETENTION_MIGRATIONS:
        from_files = migration["from_files"]
        to_files = migration["to_files"]
        if (
            changed_files == set(from_files) == set(to_files)
            and all(stored_files[path] == digest for path, digest in from_files.items())
            and all(current_files[path] == digest for path, digest in to_files.items())
        ):
            return True
    return False


def compatible_training_resume_contract_digest(
    config: ExperimentConfig,
    run_manifest: dict[str, Any],
) -> str:
    """Return the stored digest after an exact, allowlisted retention-only migration."""

    stored_digest = run_manifest.get("training_resume_contract_sha256")
    if (
        not isinstance(stored_digest, str)
        or len(stored_digest) != 64
        or any(character not in "0123456789abcdef" for character in stored_digest)
    ):
        raise ValueError("Run manifest has an invalid training resume contract digest")
    current_contract = training_resume_contract(config)
    current_digest = _canonical_payload_digest(current_contract)
    if stored_digest == current_digest:
        return stored_digest
    stored_contract = run_manifest.get("training_resume_contract")
    if not isinstance(stored_contract, dict):
        raise ValueError("Run manifest has no training resume contract snapshot")
    if _canonical_payload_digest(stored_contract) != stored_digest:
        raise ValueError("Run manifest training resume contract snapshot is inconsistent")
    if not _checkpoint_retention_migration_matches(stored_contract, current_contract):
        raise ValueError("Run manifest does not match the current training resume contract")
    return stored_digest


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


def readonly_checkpoint_contract_digest(
    config: ExperimentConfig,
    run_manifest: dict[str, Any],
) -> str:
    """Permit legacy inference/probes without authorizing a training-code migration.

    The original model, data, stage, artifacts, and resolved settings must still
    match. Only a legacy baseline can use this compatibility path; new fixed-date
    runs retain the strict implementation contract.
    """

    if config.data.fixed_split is not None or config.model.feature_mode != "baseline":
        return compatible_training_resume_contract_digest(config, run_manifest)
    stored = run_manifest.get("training_resume_contract")
    digest = run_manifest.get("training_resume_contract_sha256")
    if not isinstance(stored, dict) or _canonical_payload_digest(stored) != digest:
        raise ValueError("Historical checkpoint has an inconsistent run contract")
    implementation = stored.get("training_implementation")
    files = implementation.get("files") if isinstance(implementation, dict) else None
    if (
        not isinstance(files, dict)
        or not files
        or any(
            not isinstance(path, str)
            or not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for path, value in files.items()
        )
        or implementation.get("sha256") != _canonical_payload_digest(files)
    ):
        raise ValueError("Historical checkpoint implementation fingerprint is invalid")
    current = training_resume_contract(config)
    semantics = {key: value for key, value in stored.items() if key != "training_implementation"}
    semantics["data"] = DataConfig.model_validate(semantics.get("data")).model_dump(mode="json")
    semantics["model"] = ModelConfig.model_validate(semantics.get("model")).model_dump(mode="json")
    expected = {key: value for key, value in current.items() if key != "training_implementation"}
    if semantics != expected:
        raise ValueError("Historical checkpoint model, data, or training settings differ")
    return str(digest)


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
    run_manifest_path = root / "run-manifest.json"
    try:
        run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Run manifest is missing: {run_manifest_path}") from error
    if not isinstance(run_manifest, dict):
        raise ValueError("Run manifest must contain a JSON object")
    if run_manifest.get("run_id") != run_id or run_manifest.get("run_key") != run_id:
        raise ValueError("Run manifest identity does not match its canonical run directory")
    if run_manifest.get("schema_version") != CHECKPOINT_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("Run manifest does not use the current checkpoint artifact schema")
    expected_digest = compatible_training_resume_contract_digest(config, run_manifest)
    trainer_state_path = checkpoint / "trainer-state.json"
    for path, label in ((trainer_state_path, "Trainer state"),):
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
    current_contract = training_resume_contract(config)
    for path, label in (
        (root / "resolved-config.yaml", "Run resolved config"),
        (checkpoint / "resolved-config.yaml", "Checkpoint resolved config"),
    ):
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"{label} is missing or empty: {path}")
        stored_config = ExperimentConfig.from_yaml(path)
        if training_resume_contract(stored_config) != current_contract:
            raise ValueError(f"{label} does not match the current training resume contract")
    return expected_digest
