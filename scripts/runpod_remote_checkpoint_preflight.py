#!/usr/bin/env python3
"""Validate ranked numerical-model checkpoints before paid GPU work."""

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import runpod_readiness as readiness

MAX_JSON_BYTES = 8 * 1024 * 1024
CHECKPOINT_FILES = (
    "adapter.safetensors",
    "optimizer.pt",
    "scheduler.pt",
    "trainer-state.json",
    "resolved-config.yaml",
)
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class RemoteCheckpointPreflightError(RuntimeError):
    """Report a fail-closed remote checkpoint validation error."""


def _safe_error_text(stderr):
    message = stderr.decode("utf-8", errors="replace").strip()
    if not message:
        return "the S3 wrapper returned no diagnostic output"
    return message[-2000:]


class RunPodS3Reader:
    """Read small JSON objects and HEAD large artifacts through the project wrapper."""

    def __init__(self, wrapper, bucket):
        resolved_wrapper = wrapper.expanduser().resolve(strict=False)
        if not resolved_wrapper.is_file():
            raise RemoteCheckpointPreflightError(
                f"RunPod S3 wrapper is unavailable: {resolved_wrapper}"
            )
        if not bucket or not all(character.isalnum() or character in "_-" for character in bucket):
            raise RemoteCheckpointPreflightError("RunPod network-volume ID is not canonical")
        self.wrapper = resolved_wrapper
        self.bucket = bucket

    def _run(self, arguments, label):
        completed = subprocess.run(  # noqa: UP022 - local control Python may be 3.6.
            ["bash", str(self.wrapper), *arguments],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if completed.returncode != 0:
            raise RemoteCheckpointPreflightError(
                f"Unable to verify {label}: {_safe_error_text(completed.stderr)}"
            )
        return completed.stdout

    def json_object(self, key, label):
        content = self.bytes_object(key, label)
        try:
            payload = json.loads(content.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RemoteCheckpointPreflightError(f"{label} is not valid UTF-8 JSON") from error
        if not isinstance(payload, dict):
            raise RemoteCheckpointPreflightError(f"{label} must contain a JSON object")
        return payload

    def bytes_object(self, key, label):
        content = self._run(
            ["s3", "cp", f"s3://{self.bucket}/{key}", "-", "--only-show-errors"],
            label,
        )
        if not content or len(content) > MAX_JSON_BYTES:
            raise RemoteCheckpointPreflightError(
                f"{label} is empty or exceeds the {MAX_JSON_BYTES}-byte safety limit"
            )
        return content

    def content_length(self, key, label):
        output = self._run(
            [
                "s3api",
                "head-object",
                "--bucket",
                self.bucket,
                "--key",
                key,
                "--query",
                "ContentLength",
                "--output",
                "text",
            ],
            label,
        )
        try:
            size_bytes = int(output.decode("ascii").strip())
        except (UnicodeDecodeError, ValueError) as error:
            raise RemoteCheckpointPreflightError(
                f"{label} returned an invalid ContentLength"
            ) from error
        if size_bytes < 1:
            raise RemoteCheckpointPreflightError(f"{label} is missing or empty")
        return size_bytes


def _validate_run_manifest(
    manifest,
    run_id,
    leaderboard_contract_digest,
    config_sha256,
    resolved_config,
    dataset_readiness,
    dataset_manifest,
    dataset_manifest_sha256,
):
    if manifest.get("run_id") != run_id or manifest.get("run_key") != run_id:
        raise ValueError("Run manifest identity does not match its canonical run directory")
    manifest_contract_digest = readiness._checkpoint_artifact_contract_digest(
        manifest,
        "Run manifest",
    )
    if manifest_contract_digest != leaderboard_contract_digest:
        raise ValueError("Run manifest and checkpoint leaderboard contracts disagree")
    if manifest.get("source_config_sha256") != config_sha256:
        raise ValueError("Run manifest was created from a different RUNPOD_CONFIG")
    resolved_config_sha256 = hashlib.sha256(resolved_config).hexdigest()
    if manifest.get("resolved_config_sha256") != resolved_config_sha256:
        raise ValueError("Run resolved config does not match its manifest digest")
    resume_contract = manifest.get("training_resume_contract")
    if not isinstance(resume_contract, dict):
        raise ValueError("Run manifest has no current training resume contract snapshot")
    encoded_contract = json.dumps(
        resume_contract,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    if hashlib.sha256(encoded_contract).hexdigest() != manifest_contract_digest:
        raise ValueError("Run manifest training contract snapshot has a different digest")
    current_dataset_contract = _dataset_contract(
        dataset_readiness,
        dataset_manifest,
        dataset_manifest_sha256,
    )
    if resume_contract.get("dataset_artifacts") != current_dataset_contract:
        raise ValueError("Run checkpoint was created from different numerical dataset artifacts")


def _required_sha256(payload, key):
    value = payload.get(key)
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"Dataset readiness manifest has an invalid {key}")
    return value


def _dataset_artifact(payload, name):
    value = payload.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"Dataset readiness manifest has no valid {name} artifact")
    relative_path = value.get("relative_path")
    if (
        not isinstance(relative_path, str)
        or not relative_path
        or "\\" in relative_path
        or Path(relative_path).is_absolute()
        or ".." in Path(relative_path).parts
    ):
        raise ValueError(f"Dataset readiness manifest has an unsafe {name} artifact path")
    sha256 = value.get("sha256")
    size_bytes = value.get("size_bytes")
    row_count = value.get("row_count")
    if not isinstance(sha256, str) or SHA256_PATTERN.fullmatch(sha256) is None:
        raise ValueError(f"Dataset readiness manifest has an invalid {name} artifact digest")
    if (
        not isinstance(size_bytes, int)
        or isinstance(size_bytes, bool)
        or size_bytes <= 0
        or not isinstance(row_count, int)
        or isinstance(row_count, bool)
        or row_count <= 0
    ):
        raise ValueError(f"Dataset readiness manifest has invalid {name} artifact bounds")
    return {
        "relative_path": relative_path,
        "sha256": sha256,
        "size_bytes": size_bytes,
        "row_count": row_count,
    }


def _manifest_artifact(payload, name):
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("Numerical dataset manifest has no artifacts")
    value = artifacts.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"Numerical dataset manifest has no valid {name} artifact")
    relative_path = value.get("relative_path")
    sha256 = value.get("sha256")
    size_bytes = value.get("size_bytes")
    row_count = value.get("row_count")
    if (
        not isinstance(relative_path, str)
        or not relative_path
        or "\\" in relative_path
        or Path(relative_path).is_absolute()
        or ".." in Path(relative_path).parts
        or not isinstance(sha256, str)
        or SHA256_PATTERN.fullmatch(sha256) is None
        or not isinstance(size_bytes, int)
        or isinstance(size_bytes, bool)
        or size_bytes <= 0
        or not isinstance(row_count, int)
        or isinstance(row_count, bool)
        or row_count <= 0
    ):
        raise ValueError(f"Numerical dataset manifest has invalid {name} artifact metadata")
    return {
        "relative_path": relative_path,
        "sha256": sha256,
        "size_bytes": size_bytes,
        "row_count": row_count,
    }


def _dataset_contract(readiness_payload, dataset_payload, dataset_manifest_sha256):
    if (
        not isinstance(readiness_payload, dict)
        or readiness_payload.get("schema_version") != 2
        or readiness_payload.get("kind") != "stage1-dataset"
        or readiness_payload.get("state") != "ready"
    ):
        raise ValueError("Dataset readiness manifest must contain a ready JSON object")
    if (
        not isinstance(dataset_payload, dict)
        or dataset_payload.get("schema_version") != "3.0"
        or dataset_payload.get("kind") != "ohlcv-bar-store-dataset"
        or dataset_payload.get("state") != "ready"
    ):
        raise ValueError("Numerical dataset manifest must contain a ready JSON object")
    readiness_dataset_manifest = _dataset_artifact(readiness_payload, "dataset_manifest")
    if readiness_dataset_manifest["sha256"] != dataset_manifest_sha256:
        raise ValueError("Dataset readiness marker and dataset manifest digest disagree")
    raw = _manifest_artifact(dataset_payload, "raw")
    lazy_artifacts = {
        name: _manifest_artifact(dataset_payload, name)
        for name in ("bar_store_manifest", "symbol_index", "cutoff_ranges")
    }
    if raw["sha256"] != _dataset_artifact(readiness_payload, "raw")["sha256"] or any(
        artifact["sha256"]
        != _dataset_artifact(readiness_payload, name)["sha256"]
        for name, artifact in lazy_artifacts.items()
    ):
        raise ValueError("Dataset readiness marker and numerical artifacts disagree")
    for key in (
        "dataset_profile",
        "selected_datasets",
        "data_pipeline_digest",
        "universe_sha256",
        "split_counts",
    ):
        if readiness_payload.get(key) != dataset_payload.get(key):
            raise ValueError(f"Dataset readiness marker and dataset manifest disagree on {key}")
    if readiness_payload.get("storage_preparation_spec_sha256") != dataset_payload.get(
        "preparation_spec_sha256"
    ):
        raise ValueError("Dataset readiness marker and storage preparation contract disagree")
    return {
        "status": "ready",
        "path": f"/runpod-volume/{readiness_dataset_manifest['relative_path']}",
        "sha256": dataset_manifest_sha256,
        "dataset_profile": dataset_payload["dataset_profile"],
        "selected_datasets": dataset_payload["selected_datasets"],
        "data_pipeline_digest": _required_sha256(
            dataset_payload,
            "data_pipeline_digest",
        ),
        "preparation_spec_sha256": _required_sha256(
            dataset_payload,
            "preparation_spec_sha256",
        ),
        "universe_sha256": _required_sha256(dataset_payload, "universe_sha256"),
        "split_counts": dataset_payload["split_counts"],
        "artifacts": {"raw": raw, **lazy_artifacts},
    }


def _best_checkpoint_from_pointer(
    pointer,
    leaderboard,
    checkpoints,
    run_id,
    contract_digest,
    monitor,
    mode,
):
    if pointer.get("run_id") != run_id or pointer.get("run_key") != run_id:
        raise ValueError("Best-checkpoint pointer identity does not match its run directory")
    if pointer.get("selection_source") != "validation":
        raise ValueError("Best checkpoint must be selected from validation metrics")
    pointer_contract_digest = readiness._checkpoint_artifact_contract_digest(
        pointer,
        "Best-checkpoint pointer",
    )
    if pointer_contract_digest != contract_digest:
        raise ValueError("Best-checkpoint pointer and leaderboard contracts disagree")
    checkpoint_name = pointer.get("path")
    if (
        not isinstance(checkpoint_name, str)
        or readiness.CHECKPOINT_NAME_PATTERN.fullmatch(checkpoint_name) is None
    ):
        raise ValueError("Best checkpoint pointer contains a non-canonical checkpoint name")
    if (
        leaderboard.get("best_checkpoint") != checkpoint_name
        or checkpoints[0]["path"] != checkpoint_name
    ):
        raise ValueError("Best-checkpoint pointer and leaderboard disagree")
    if pointer.get("monitor") != monitor or pointer.get("mode") != mode:
        raise ValueError("Best-checkpoint pointer and leaderboard selection policy disagree")
    pointer_value = pointer.get("value")
    if (
        not isinstance(pointer_value, (int, float))
        or isinstance(pointer_value, bool)
        or float(pointer_value) != float(checkpoints[0]["value"])
    ):
        raise ValueError("Best-checkpoint pointer and leaderboard values disagree")
    return checkpoint_name


def _verify_checkpoint_artifacts(
    reader,
    run_id,
    checkpoint_name,
    trainer_state,
):
    artifact_files = trainer_state["artifact_files"]
    for filename in CHECKPOINT_FILES:
        key = f"savedModel/{run_id}/{checkpoint_name}/{filename}"
        if filename == "resolved-config.yaml":
            content = reader.bytes_object(key, f"{checkpoint_name}/{filename}")
            expected = artifact_files[filename]
            if (
                len(content) != expected["size_bytes"]
                or hashlib.sha256(content).hexdigest() != expected["sha256"]
            ):
                raise RemoteCheckpointPreflightError(
                    "Remote checkpoint digest does not match trainer-state integrity metadata: "
                    f"{run_id}/{checkpoint_name}/{filename}"
                )
            continue
        remote_size = reader.content_length(key, f"{checkpoint_name}/{filename}")
        if filename == "trainer-state.json":
            continue
        expected_size = artifact_files[filename]["size_bytes"]
        if remote_size != expected_size:
            raise RemoteCheckpointPreflightError(
                "Remote checkpoint size does not match trainer-state integrity metadata: "
                f"{run_id}/{checkpoint_name}/{filename}"
            )


def validate_remote_checkpoint_run(
    reader,
    run_id,
    checkpoint_name,
    selection_policy,
    config_path,
):
    """Validate the requested resume or validation artifact before Pod creation."""

    canonical_run_id = readiness._validate_run_id(run_id, "Remote checkpoint run ID")
    run_root = f"savedModel/{canonical_run_id}"
    manifest = reader.json_object(f"{run_root}/run-manifest.json", "run manifest")
    resolved_config = reader.bytes_object(
        f"{run_root}/resolved-config.yaml",
        "run resolved config",
    )
    dataset_readiness = reader.json_object(
        "lifecycle/stage1/dataset.json",
        "dataset readiness manifest",
    )
    dataset_manifest_artifact = _dataset_artifact(
        dataset_readiness,
        "dataset_manifest",
    )
    dataset_manifest_bytes = reader.bytes_object(
        dataset_manifest_artifact["relative_path"],
        "numerical dataset manifest",
    )
    try:
        dataset_manifest = json.loads(dataset_manifest_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Numerical dataset manifest is not valid UTF-8 JSON") from error
    leaderboard = reader.json_object(
        f"{run_root}/checkpoint-leaderboard.json",
        "checkpoint leaderboard",
    )
    monitor, mode, checkpoints, contract_digest = readiness._validated_checkpoint_leaderboard(
        leaderboard,
        canonical_run_id,
    )
    _validate_run_manifest(
        manifest,
        canonical_run_id,
        contract_digest,
        hashlib.sha256(config_path.read_bytes()).hexdigest(),
        resolved_config,
        dataset_readiness,
        dataset_manifest,
        hashlib.sha256(dataset_manifest_bytes).hexdigest(),
    )
    pointer = reader.json_object(
        f"{run_root}/best-checkpoint.json",
        "best-checkpoint pointer",
    )
    best_checkpoint = _best_checkpoint_from_pointer(
        pointer,
        leaderboard,
        checkpoints,
        canonical_run_id,
        contract_digest,
        monitor,
        mode,
    )
    retained_names = [str(row["path"]) for row in checkpoints]
    if selection_policy == "latest":
        latest_row = max(checkpoints, key=lambda row: int(row["global_step"]))
        selected_checkpoint = checkpoint_name or str(latest_row["path"])
        if selected_checkpoint != latest_row["path"]:
            raise ValueError(
                "Training resume checkpoint must be the latest retained global_step: "
                f"requested {selected_checkpoint}, latest {latest_row['path']}"
            )
    elif selection_policy == "best":
        if checkpoint_name:
            raise ValueError("Best-checkpoint selection must not supply --checkpoint-name")
        selected_checkpoint = best_checkpoint
    else:
        if not checkpoint_name:
            raise ValueError("Retained-checkpoint selection requires --checkpoint-name")
        selected_checkpoint = checkpoint_name

    if (
        readiness.CHECKPOINT_NAME_PATTERN.fullmatch(selected_checkpoint) is None
        or selected_checkpoint not in retained_names
    ):
        raise ValueError("Selected checkpoint is not retained by the validation leaderboard")

    for row in checkpoints:
        retained_name = str(row["path"])
        trainer_state = reader.json_object(
            f"{run_root}/{retained_name}/trainer-state.json",
            f"{retained_name}/trainer-state.json",
        )
        readiness._validate_trainer_state_identity(
            trainer_state,
            run_id=canonical_run_id,
            checkpoint_name=retained_name,
            row=row,
            monitor=monitor,
            mode=mode,
            contract_digest=contract_digest,
        )
        _verify_checkpoint_artifacts(
            reader,
            canonical_run_id,
            retained_name,
            trainer_state,
        )

    return selected_checkpoint


def build_parser():
    parser = argparse.ArgumentParser(
        description="Fail closed before creating a paid GPU Pod for resume or validation."
    )
    parser.add_argument("--s3-wrapper", type=Path, required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--selection-policy",
        choices=("latest", "best", "retained"),
        required=True,
    )
    parser.add_argument("--checkpoint-name")
    return parser


def main():
    arguments = build_parser().parse_args()
    try:
        reader = RunPodS3Reader(arguments.s3_wrapper, arguments.bucket)
        checkpoint_name = validate_remote_checkpoint_run(
            reader,
            run_id=arguments.run_id,
            checkpoint_name=arguments.checkpoint_name,
            selection_policy=arguments.selection_policy,
            config_path=arguments.config.expanduser().resolve(strict=False),
        )
    except (OSError, ValueError, RemoteCheckpointPreflightError) as error:
        print(f"Remote checkpoint preflight failed: {error}", file=sys.stderr)
        return 3
    print(checkpoint_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
