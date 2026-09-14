#!/usr/bin/env python3
"""Create and validate RunPod lifecycle manifests without external packages."""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import runpy
import sys
import uuid
from collections import namedtuple
from contextlib import suppress
from datetime import datetime, timedelta, timezone
from pathlib import Path

SCHEMA_VERSION = 1
LIFECYCLE_SCHEMA_VERSION = 1
CHECKPOINT_ARTIFACT_SCHEMA_VERSION = "4.0"
TRAINING_COMPLETION_SCHEMA_VERSION = "1.0"
SYMBOL_PATTERN = re.compile(r"^[A-Z0-9.^_-]+$")
SOURCE_ROOT_NAMES = ("src", "configs", "scripts", "tests")
SOURCE_SUFFIXES = frozenset({".py", ".sh", ".yaml", ".yml"})
ISO_TIMESTAMP_PATTERN = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})[T ](?P<time>\d{2}:\d{2}:\d{2})"
    r"(?P<fraction>\.\d{1,6})?(?P<zone>Z|[+-]\d{2}:\d{2})$"
)
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,119}$")
POD_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
CHECKPOINT_NAME_PATTERN = re.compile(r"^checkpoint-[0-9]{6,}$")
CPU_PREPARATION_RESUMABLE_STATES = frozenset(
    {
        "waiting_for_provider",
        "waiting_for_budget",
        "waiting_for_resume",
        "waiting_for_preparation",
        "downloaded",
    }
)
GUARD_LIFECYCLE_KINDS = frozenset(
    {
        "stage1-cpu-preparation",
        "stage1-mixed-finalization",
        "stage1-training",
        "stage1-validation",
    }
)
GUARD_LIFECYCLE_STATES = frozenset(
    {
        "preparing",
        "finalizing",
        "ready",
        "failed",
        "timed_out",
        *CPU_PREPARATION_RESUMABLE_STATES,
    }
)
GPU_WORKFLOW_ACTIVE_STATES = frozenset({"preparing", "finalizing"})
GPU_WORKFLOW_TERMINAL_STATES = frozenset({"ready", "failed", "timed_out"})
ORPHANED_GPU_WORKFLOW_REASON = "runpod_pod_not_found"

_CONTENT_IDENTITY_CONTRACT = runpy.run_path(
    str(
        Path(__file__).resolve().parents[1]
        / "src"
        / "stock_forecasting"
        / "data"
        / "content_identity.py"
    )
)
CONTENT_IDENTITY_SCHEMA_VERSION = _CONTENT_IDENTITY_CONTRACT[
    "CONTENT_IDENTITY_SCHEMA_VERSION"
]

_DATASET_IDENTITY_CONTRACT = runpy.run_path(
    str(
        Path(__file__).resolve().parents[1]
        / "src"
        / "stock_forecasting"
        / "dataset_identity.py"
    )
)
_DATASET_PROFILE_CONTRACT = runpy.run_path(
    str(
        Path(__file__).resolve().parents[1]
        / "src"
        / "stock_forecasting"
        / "dataset_profiles.py"
    )
)
_TRAINING_STAGE_CONTRACT = runpy.run_path(
    str(
        Path(__file__).resolve().parents[1]
        / "src"
        / "stock_forecasting"
        / "training_stage_contract.py"
    )
)
DATASET_REQUEST_SCHEMA_VERSION = _DATASET_IDENTITY_CONTRACT[
    "DATASET_REQUEST_SCHEMA_VERSION"
]
MIN_FIXED_EVALUATION_DATES = _DATASET_IDENTITY_CONTRACT[
    "MIN_FIXED_EVALUATION_DATES"
]
DEFAULT_DATASET_STORAGE_PREPARATION = _DATASET_IDENTITY_CONTRACT[
    "DEFAULT_DATASET_STORAGE_PREPARATION"
]
storage_preparation_spec = _DATASET_IDENTITY_CONTRACT["storage_preparation_spec"]
validate_fixed_split_audit = _DATASET_IDENTITY_CONTRACT["validate_fixed_split_audit"]
BAR_STORE_SCHEMA_VERSION = _DATASET_IDENTITY_CONTRACT["BAR_STORE_SCHEMA_VERSION"]
BAR_STORE_KIND = _DATASET_IDENTITY_CONTRACT["BAR_STORE_KIND"]
dataset_request_identity_payload = _DATASET_IDENTITY_CONTRACT[
    "dataset_request_identity_payload"
]
PROFILE_DATASETS = {
    profile: sorted(datasets)
    for profile, datasets in _DATASET_PROFILE_CONTRACT["PROFILE_DATASETS"].items()
}
PRODUCTION_STAGE_SAMPLE_CONTRACTS = _TRAINING_STAGE_CONTRACT[
    "PRODUCTION_STAGE_SAMPLE_CONTRACTS"
]


StageContract = namedtuple(
    "StageContract",
    (
        "dataset_profile",
        "training_stage",
        "train_fraction",
        "max_samples",
        "embargo_trading_days",
        "required_model_repositories",
    ),
)


def _strip_yaml_scalar(value):
    scalar = value.strip()
    if scalar[:1] in {'"', "'"}:
        if len(scalar) < 2 or scalar[-1] != scalar[0]:
            raise ValueError("Training config contains a malformed quoted scalar")
        scalar = scalar[1:-1]
    return scalar


def _resolve_environment_default(value):
    match = re.fullmatch(r"\$\{([A-Za-z_][A-Za-z0-9_]*):-([^}]*)\}", value)
    if match is None:
        return value
    return os.environ.get(match.group(1), match.group(2))


def _load_stage_contract(config_path):
    """Read the small scalar readiness contract from top-level YAML blocks."""

    path = Path(config_path)
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Training config is missing or is a symlink: {path}")
    data_values = {}
    model_values = {}
    training_values = {}
    active_values = None
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indentation = len(raw_line) - len(raw_line.lstrip(" "))
        stripped = raw_line.strip()
        if indentation == 0:
            if stripped == "data:":
                active_values = data_values
            elif stripped == "model:":
                active_values = model_values
            elif stripped == "training:":
                active_values = training_values
            else:
                active_values = None
            continue
        if active_values is None or indentation != 2 or ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        if key in active_values:
            raise ValueError(f"Training config repeats a readiness key at line {line_number}")
        active_values[key] = _resolve_environment_default(_strip_yaml_scalar(value))

    required = {
        "dataset_profile",
        "train_fraction",
        "max_samples",
        "embargo_trading_days",
    }
    missing = sorted(required.difference(data_values))
    if missing:
        raise ValueError("Training config has no readiness values: " + ", ".join(missing))
    dataset_profile = data_values["dataset_profile"]
    if dataset_profile not in PROFILE_DATASETS:
        raise ValueError("Training config data.dataset_profile is unsupported")
    if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", data_values["train_fraction"]):
        raise ValueError("Training config data.train_fraction must be numeric")
    train_fraction = float(data_values["train_fraction"])
    raw_max_samples = data_values["max_samples"].lower()
    if raw_max_samples in {"", "null", "~"}:
        max_samples = None
    elif re.fullmatch(r"[1-9][0-9]*", raw_max_samples):
        max_samples = int(raw_max_samples)
    else:
        raise ValueError(
            "Training config data.max_samples must be null or a positive integer"
        )
    if not re.fullmatch(r"[0-9]+", data_values["embargo_trading_days"]):
        raise ValueError("Training config data.embargo_trading_days must be an integer")
    embargo_trading_days = int(data_values["embargo_trading_days"])

    required_model_keys = {
        "time_series_backend",
        "time_series_model_id",
        "time_series_tokenizer_id",
    }
    missing_model_keys = sorted(required_model_keys.difference(model_values))
    if missing_model_keys:
        raise ValueError(
            "Training config has no model readiness values: " + ", ".join(missing_model_keys)
        )
    repositories = []
    if model_values["time_series_backend"] != "mock":
        repositories.extend(
            [
                model_values["time_series_model_id"],
                model_values["time_series_tokenizer_id"],
            ]
        )
    repository_pattern = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")
    if any(not repository_pattern.fullmatch(repository) for repository in repositories):
        raise ValueError("Training config contains an invalid Hugging Face repository ID")
    training_stage = training_values.get("stage")
    if training_stage not in {"stage1", "stage2"}:
        raise ValueError("Training config training.stage must be stage1 or stage2")
    sample_contract = PRODUCTION_STAGE_SAMPLE_CONTRACTS[training_stage]
    expected_fraction = float(sample_contract["train_fraction"])
    if abs(train_fraction - expected_fraction) > 1e-12:
        raise ValueError("Training config stage and train_fraction disagree")
    if (
        model_values["time_series_backend"] == "kronos"
        and max_samples != sample_contract["max_samples"]
    ):
        raise ValueError("Training config stage and max_samples disagree")
    return StageContract(
        dataset_profile=dataset_profile,
        training_stage=training_stage,
        train_fraction=train_fraction,
        max_samples=max_samples,
        embargo_trading_days=embargo_trading_days,
        required_model_repositories=tuple(dict.fromkeys(repositories)),
    )


def _utc_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative_path(value):
    candidate = Path(value)
    if candidate.is_absolute() or ".." in candidate.parts or not candidate.parts:
        raise ValueError("Manifest paths must be safe paths relative to the project root")
    return candidate


def _file_record(project_root, relative_path):
    safe_relative = _safe_relative_path(relative_path)
    candidate_path = project_root / safe_relative
    if candidate_path.is_symlink():
        raise ValueError(f"Manifest file is missing or is a symlink: {relative_path}")
    absolute_path = candidate_path.resolve()
    try:
        absolute_path.relative_to(project_root.resolve())
    except ValueError as error:
        raise ValueError(f"Manifest file escapes the project root: {relative_path}") from error
    if not absolute_path.is_file() or absolute_path.is_symlink():
        raise ValueError(f"Manifest file is missing or is a symlink: {relative_path}")
    return {
        "path": safe_relative.as_posix(),
        "sha256": _sha256(absolute_path),
        "size_bytes": absolute_path.stat().st_size,
    }


def _records_digest(records):
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: item["path"]):
        line = "{sha256} {size_bytes} {path}\n".format(**record)
        digest.update(line.encode("utf-8"))
    return digest.hexdigest()


def _content_identity_module(project_root):
    source = (
        Path(project_root)
        / "src"
        / "stock_forecasting"
        / "data"
        / "content_identity.py"
    )
    if not source.is_file() or source.is_symlink():
        raise ValueError("Data content identity helper is missing or is a symlink")
    specification = importlib.util.spec_from_file_location(
        "runpod_data_content_identity",
        source,
    )
    if specification is None or specification.loader is None:
        raise ValueError("Data content identity helper cannot be loaded")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _project_content_identity(project_root):
    module = _content_identity_module(project_root)
    package_root = Path(project_root) / "src" / "stock_forecasting"
    return (
        module.code_content_identity(package_root=package_root),
        module.semantic_source_paths(),
    )


def _validate_content_identity_digest(value, label):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _validate_code_content_identity(identity):
    if (
        not isinstance(identity, dict)
        or set(identity)
        != {
            "schema_version",
            "provider_materialization_digests",
            "raw_materialization_digest",
            "bar_store_materialization_digest",
        }
        or identity.get("schema_version") != CONTENT_IDENTITY_SCHEMA_VERSION
    ):
        raise ValueError("Code manifest data content identity is invalid")
    providers = identity.get("provider_materialization_digests")
    expected_providers = {
        "eodhd_us",
        "massive_us",
        "tpex_official",
        "twse_official",
    }
    if not isinstance(providers, dict) or set(providers) != expected_providers:
        raise ValueError("Code manifest provider content identities are invalid")
    for provider, digest in providers.items():
        _validate_content_identity_digest(digest, f"Code {provider} content identity")
    _validate_content_identity_digest(
        identity.get("raw_materialization_digest"),
        "Code raw materialization identity",
    )
    _validate_content_identity_digest(
        identity.get("bar_store_materialization_digest"),
        "Code bar-store content identity",
    )
    return identity


def _dataset_content_identity_from_code(code_payload, selected_datasets):
    code_identity = _validate_code_content_identity(code_payload.get("data_content_identity"))
    if (
        not isinstance(selected_datasets, list)
        or selected_datasets != sorted(set(selected_datasets))
    ):
        raise ValueError("Selected datasets for content identity are invalid")
    providers = code_identity["provider_materialization_digests"]
    missing = sorted(set(selected_datasets).difference(providers))
    if missing:
        raise ValueError("Code has no provider content identity for: " + ", ".join(missing))
    return {
        "schema_version": CONTENT_IDENTITY_SCHEMA_VERSION,
        "selected_datasets": selected_datasets,
        "provider_materialization_digests": {
            provider: providers[provider] for provider in selected_datasets
        },
        "raw_materialization_digest": code_identity["raw_materialization_digest"],
        "bar_store_materialization_digest": code_identity[
            "bar_store_materialization_digest"
        ],
    }


def _load_json(source):
    if source == "-":
        return json.load(sys.stdin)
    with Path(source).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _unique_temporary_path(path):
    return path.parent / f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"


def _atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _unique_temporary_path(path)
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()


def _write_immutable_json(path, payload):
    """Atomically publish one immutable JSON object without replacing an existing file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _unique_temporary_path(path)
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(str(temporary), str(path))
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()


def _require_string(payload, key):
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Manifest field {key!r} must be a non-empty string")
    return value


def _validate_code_payload(payload, project_root=None):
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("kind") != "code":
        raise ValueError("Unsupported code manifest schema")
    if payload.get("state") != "ready":
        raise ValueError("Code manifest is not ready: {}".format(payload.get("state", "missing")))
    records = payload.get("files")
    if not isinstance(records, list) or not records:
        raise ValueError("Code manifest has no files")
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Code manifest contains an invalid file record")
        _safe_relative_path(_require_string(record, "path"))
        _require_string(record, "sha256")
        if not isinstance(record.get("size_bytes"), int) or record["size_bytes"] < 0:
            raise ValueError("Code manifest contains an invalid file size")
    if _records_digest(records) != payload.get("release_digest"):
        raise ValueError("Code manifest release digest is inconsistent")

    pipeline_paths = payload.get("data_pipeline_paths")
    if (
        not isinstance(pipeline_paths, list)
        or not pipeline_paths
        or pipeline_paths != sorted(set(pipeline_paths))
    ):
        raise ValueError("Code manifest has no data pipeline scope")
    pipeline_set = set(pipeline_paths)
    pipeline_records = [record for record in records if record["path"] in pipeline_set]
    if len(pipeline_records) != len(pipeline_set):
        raise ValueError("Code manifest data pipeline scope is incomplete")
    content_identity = _validate_code_content_identity(payload.get("data_content_identity"))
    if _payload_sha256(content_identity) != payload.get("data_pipeline_digest"):
        raise ValueError("Code manifest data pipeline digest is inconsistent")
    removed_fields = {
        "compatible_data_pipeline_digests",
        "data_pipeline_compatibility_base_digest",
    }
    if removed_fields.intersection(payload):
        raise ValueError("Code manifest uses removed data pipeline compatibility fields")

    if project_root is not None:
        manifest_paths = _validate_manifest_project_files(payload, project_root)
        expected_identity, expected_paths = _project_content_identity(project_root)
        if content_identity != expected_identity or pipeline_paths != expected_paths:
            raise ValueError(
                "Code manifest data content identity differs from mounted source"
            )
        extra_sources = [
            relative
            for relative, _candidate in _find_unlisted_sources(project_root, manifest_paths)
        ]
        if extra_sources:
            raise ValueError(
                "Mounted project contains stale source files outside the release manifest: "
                + ", ".join(sorted(extra_sources))
            )
    return payload


def _validate_manifest_project_files(payload, project_root):
    root = Path(project_root)
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"Project root is missing, invalid, or a symlink: {root}")
    records = payload["files"]
    actual_records = [
        _file_record(root, record["path"])
        for record in sorted(records, key=lambda item: item["path"])
    ]
    if actual_records != sorted(records, key=lambda item: item["path"]):
        raise ValueError("Local project does not match the uploaded code manifest")
    return {record["path"] for record in records}


def _find_unlisted_sources(project_root, manifest_paths):
    """Return regular source files outside the manifest without following symlinks."""

    root = Path(project_root).resolve()
    unlisted = []
    for root_name in SOURCE_ROOT_NAMES:
        source_root = root / root_name
        if source_root.is_symlink():
            raise ValueError(f"Scoped source root is a symlink: {root_name}")
        if not source_root.exists():
            continue
        if not source_root.is_dir():
            raise ValueError(f"Scoped source root is not a directory: {root_name}")
        for current_root, directory_names, file_names in os.walk(
            source_root, topdown=True, followlinks=False
        ):
            current = Path(current_root)
            for directory_name in list(directory_names):
                directory = current / directory_name
                relative = directory.relative_to(root).as_posix()
                if directory.is_symlink():
                    raise ValueError(f"Scoped source tree contains a symlink: {relative}")
                try:
                    directory.resolve().relative_to(root)
                except ValueError as error:
                    raise ValueError(
                        f"Scoped source directory escapes the project root: {relative}"
                    ) from error
            for file_name in file_names:
                candidate = current / file_name
                if candidate.suffix not in SOURCE_SUFFIXES:
                    continue
                relative = candidate.relative_to(root).as_posix()
                if candidate.is_symlink():
                    raise ValueError(f"Scoped source file is a symlink: {relative}")
                try:
                    resolved = candidate.resolve(strict=True)
                    resolved.relative_to(root)
                except (OSError, ValueError) as error:
                    raise ValueError(
                        f"Scoped source file escapes the project root: {relative}"
                    ) from error
                if not resolved.is_file():
                    raise ValueError(f"Scoped source path is not a regular file: {relative}")
                if relative not in manifest_paths:
                    unlisted.append((relative, candidate))
    return sorted(unlisted, key=lambda item: item[0])


def _ensure_safe_directory(path, trusted_root):
    """Create a directory tree while rejecting existing symlink components."""

    trusted = Path(trusted_root).resolve()
    target = Path(path)
    try:
        relative = target.relative_to(trusted)
    except ValueError as error:
        raise ValueError("Quarantine directory escapes NETWORK_VOLUME_ROOT") from error
    current = trusted
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"Quarantine directory contains a symlink: {current}")
        if current.exists():
            if not current.is_dir():
                raise ValueError(f"Quarantine directory collides with a file: {current}")
            continue
        current.mkdir()


def command_quarantine_stale_code(arguments):
    payload = _validate_code_payload(_load_json(arguments.marker))
    volume_argument = arguments.network_volume_root
    project_argument = arguments.project_root
    if volume_argument.is_symlink() or not volume_argument.is_dir():
        raise ValueError("NETWORK_VOLUME_ROOT is missing, invalid, or a symlink")
    if project_argument.is_symlink() or not project_argument.is_dir():
        raise ValueError("Project root is missing, invalid, or a symlink")
    volume_root = volume_argument.resolve()
    project_root = project_argument.resolve()
    if volume_root == Path("/workspace") or Path("/workspace") in volume_root.parents:
        raise ValueError("Quarantine must never use /workspace")
    try:
        lexical_relative = project_argument.absolute().relative_to(volume_argument.absolute())
        project_root.relative_to(volume_root)
    except ValueError as error:
        raise ValueError("Project root must be stored on NETWORK_VOLUME_ROOT") from error
    lexical_component = volume_argument.absolute()
    for part in lexical_relative.parts:
        lexical_component = lexical_component / part
        if lexical_component.is_symlink():
            raise ValueError(f"Project path contains a symlink component: {lexical_component}")

    manifest_paths = _validate_manifest_project_files(payload, project_root)
    stale_sources = _find_unlisted_sources(project_root, manifest_paths)
    if not stale_sources:
        print("No stale source files require quarantine")
        return 0

    quarantine_id = arguments.quarantine_id
    if quarantine_id is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        quarantine_id = f"{timestamp}-{payload['release_digest'][:12]}-{os.getpid()}"
    if not re.fullmatch(r"[A-Za-z0-9._-]+", quarantine_id):
        raise ValueError("Quarantine ID contains unsupported characters")

    quarantine_root = volume_root / "lifecycle" / "quarantine"
    batch_root = quarantine_root / quarantine_id
    if batch_root.exists() or batch_root.is_symlink():
        raise ValueError(f"Quarantine destination already exists: {batch_root}")

    planned_moves = []
    for relative, source in stale_sources:
        safe_relative = _safe_relative_path(relative)
        destination = batch_root / safe_relative
        try:
            destination.relative_to(batch_root)
        except ValueError as error:
            raise ValueError(f"Quarantine destination escapes its batch: {relative}") from error
        if destination.exists() or destination.is_symlink():
            raise ValueError(f"Quarantine destination collision: {relative}")
        planned_moves.append(
            {
                "path": relative,
                "source": source,
                "destination": destination,
                "sha256": _sha256(source),
                "size_bytes": source.stat().st_size,
            }
        )

    _ensure_safe_directory(quarantine_root, volume_root)
    batch_root.mkdir()
    audit_path = batch_root / "quarantine.json"
    audit_payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": "code-quarantine",
        "state": "moving",
        "generated_at": _utc_now(),
        "release_digest": payload["release_digest"],
        "file_count": len(planned_moves),
        "files": [
            {
                "path": move["path"],
                "sha256": move["sha256"],
                "size_bytes": move["size_bytes"],
            }
            for move in planned_moves
        ],
    }
    with audit_path.open("x", encoding="utf-8") as stream:
        json.dump(audit_payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")

    for move in planned_moves:
        _ensure_safe_directory(move["destination"].parent, batch_root)
        if move["destination"].exists() or move["destination"].is_symlink():
            raise ValueError(f"Quarantine destination collision: {move['path']}")
        if (
            move["source"].is_symlink()
            or not move["source"].is_file()
            or move["source"].stat().st_size != move["size_bytes"]
            or _sha256(move["source"]) != move["sha256"]
        ):
            raise ValueError(f"Stale source changed before quarantine: {move['path']}")
        move["source"].rename(move["destination"])

    audit_payload["state"] = "ready"
    completed_audit = batch_root / "quarantine.ready.json.tmp"
    with completed_audit.open("x", encoding="utf-8") as stream:
        json.dump(audit_payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    completed_audit.replace(audit_path)
    print(f"Quarantined {len(planned_moves)} stale source files to {batch_root}")
    return 0


def command_code_manifest(arguments):
    project_root = arguments.project_root.resolve()
    records = sorted(
        [_file_record(project_root, path) for path in arguments.paths],
        key=lambda item: item["path"],
    )
    if len({record["path"] for record in records}) != len(records):
        raise ValueError("Code manifest contains duplicate paths")
    content_identity, pipeline_paths = _project_content_identity(project_root)
    requested_pipeline_paths = sorted(
        {_safe_relative_path(path).as_posix() for path in arguments.pipeline_path}
    )
    if requested_pipeline_paths and requested_pipeline_paths != pipeline_paths:
        raise ValueError(
            "Explicit data pipeline paths differ from the content identity boundary"
        )
    record_paths = {record["path"] for record in records}
    if not set(pipeline_paths).issubset(record_paths):
        raise ValueError("Every data pipeline path must also be in the upload manifest")
    pipeline_digest = _payload_sha256(content_identity)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": "code",
        "state": arguments.state,
        "generated_at": _utc_now(),
        "remote_project_dir": arguments.remote_project_dir,
        "file_count": len(records),
        "total_bytes": sum(record["size_bytes"] for record in records),
        "release_digest": _records_digest(records),
        "data_pipeline_digest": pipeline_digest,
        "data_pipeline_paths": pipeline_paths,
        "data_content_identity": content_identity,
        "files": records,
    }
    json.dump(payload, sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


def command_check_code(arguments):
    payload = _validate_code_payload(_load_json(arguments.marker), arguments.project_root)
    print(
        "Code ready: {} files, release {}".format(
            payload["file_count"], payload["release_digest"][:12]
        )
    )
    return 0


def _validate_artifact(payload, name):
    artifact = payload.get(name)
    if not isinstance(artifact, dict):
        raise ValueError(f"Dataset manifest is missing {name!r}")
    _require_string(artifact, "relative_path")
    _require_string(artifact, "sha256")
    if not isinstance(artifact.get("size_bytes"), int) or artifact["size_bytes"] <= 0:
        raise ValueError(f"Dataset artifact {name!r} is empty")
    if not isinstance(artifact.get("row_count"), int) or artifact["row_count"] <= 0:
        raise ValueError(f"Dataset artifact {name!r} has no rows")
    return artifact


def _universe_sha256(symbols):
    rendered = "\n".join(sorted(symbols)) + "\n"
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _payload_sha256(payload):
    canonical = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _dataset_request_sha256(payload):
    if not isinstance(payload, dict):
        return ""
    return _payload_sha256(dataset_request_identity_payload(payload))


def _parse_utc_timestamp(value, label):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} is not a timestamp")
    match = ISO_TIMESTAMP_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError(f"{label} is not a valid ISO timestamp")

    fraction = match.group("fraction") or ""
    timestamp_format = "%Y-%m-%dT%H:%M:%S.%f" if fraction else "%Y-%m-%dT%H:%M:%S"
    try:
        timestamp = datetime.strptime(
            "{}T{}{}".format(match.group("date"), match.group("time"), fraction),
            timestamp_format,
        )
        zone = match.group("zone")
        if zone == "Z":
            timestamp_zone = timezone.utc
        else:
            offset_hours = int(zone[1:3])
            offset_minutes = int(zone[4:6])
            if offset_hours > 23 or offset_minutes > 59:
                raise ValueError("invalid timezone offset")
            offset = timedelta(hours=offset_hours, minutes=offset_minutes)
            if zone.startswith("-"):
                offset = -offset
            timestamp_zone = timezone(offset)
    except ValueError as error:
        raise ValueError(f"{label} is not a valid ISO timestamp") from error
    return timestamp.replace(tzinfo=timestamp_zone).astimezone(timezone.utc)


def _validate_causal_split_audit(audit, split_counts, label):
    if not isinstance(audit, dict):
        raise ValueError(f"{label} has no causal split audit")
    if audit.get("schema_version") != "causal-split-audit-v1":
        raise ValueError(f"{label} causal split audit schema is invalid")
    if audit.get("comparison") != "label.end_at < next_split_start":
        raise ValueError(f"{label} causal split comparison is not strict")
    if audit.get("violations") != 0:
        raise ValueError(f"{label} causal split audit contains violations")

    validation_start = _parse_utc_timestamp(
        audit.get("validation_start"), f"{label} validation_start"
    )
    test_start = _parse_utc_timestamp(audit.get("test_start"), f"{label} test_start")
    if validation_start >= test_start:
        raise ValueError(f"{label} validation boundary does not precede test")

    train_boundary = _parse_utc_timestamp(
        audit.get("train_boundary_exclusive"),
        f"{label} train_boundary_exclusive",
    )
    validation_boundary = _parse_utc_timestamp(
        audit.get("validation_boundary_exclusive"),
        f"{label} validation_boundary_exclusive",
    )
    if not train_boundary <= validation_start < validation_boundary <= test_start:
        raise ValueError(f"{label} causal split boundaries are inconsistent")

    counts = audit.get("label_end_counts")
    maxima = audit.get("maximum_label_end")
    if not isinstance(counts, dict) or not isinstance(maxima, dict):
        raise ValueError(f"{label} causal split audit is incomplete")
    boundaries = {"train": train_boundary, "validation": validation_boundary}
    for split, boundary in boundaries.items():
        if counts.get(split) != split_counts[split]:
            raise ValueError(f"{label} {split} label-end count is inconsistent")
        maximum = _parse_utc_timestamp(maxima.get(split), f"{label} maximum {split} label.end_at")
        if maximum >= boundary:
            raise ValueError(f"{label} {split} label crosses the next split boundary")


def _numerical_pipeline_digest_from_code_payload(code_payload, selected_datasets=None):
    identity = _validate_code_content_identity(code_payload.get("data_content_identity"))
    if selected_datasets is None:
        return _payload_sha256(identity)
    return _payload_sha256(
        _dataset_content_identity_from_code(code_payload, selected_datasets)
    )


def command_code_numerical_pipeline_digest(arguments):
    payload = _validate_code_payload(_load_json(arguments.marker))
    selected_datasets = (
        PROFILE_DATASETS[arguments.dataset_profile]
        if arguments.dataset_profile is not None
        else None
    )
    sys.stdout.write(
        _numerical_pipeline_digest_from_code_payload(payload, selected_datasets) + "\n"
    )
    return 0


def _validate_dataset_code_compatibility(
    payload,
    *,
    code_payload=None,
    expected_numerical_pipeline_digest=None,
):
    """Validate data semantics separately from full source-release provenance."""

    pipeline_digest = payload.get("data_pipeline_digest")
    if not isinstance(pipeline_digest, str) or not re.fullmatch(
        r"[0-9a-f]{64}", pipeline_digest
    ):
        raise ValueError("Dataset readiness numerical pipeline digest is invalid")
    content_identity = payload.get("data_content_identity")
    selected_datasets = payload.get("selected_datasets")
    if (
        not isinstance(content_identity, dict)
        or set(content_identity)
        != {
            "schema_version",
            "selected_datasets",
            "provider_materialization_digests",
            "raw_materialization_digest",
            "bar_store_materialization_digest",
        }
        or content_identity.get("schema_version")
        != CONTENT_IDENTITY_SCHEMA_VERSION
        or content_identity.get("selected_datasets") != selected_datasets
        or _payload_sha256(content_identity) != pipeline_digest
    ):
        raise ValueError("Dataset readiness data content identity is invalid")
    providers = content_identity.get("provider_materialization_digests")
    if (
        not isinstance(providers, dict)
        or not isinstance(selected_datasets, list)
        or set(providers) != set(selected_datasets)
        or any(
            not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
            for value in providers.values()
        )
    ):
        raise ValueError("Dataset readiness provider content identities are invalid")
    _validate_content_identity_digest(
        content_identity.get("raw_materialization_digest"),
        "Dataset raw materialization identity",
    )
    _validate_content_identity_digest(
        content_identity.get("bar_store_materialization_digest"),
        "Dataset bar-store content identity",
    )
    if code_payload is not None:
        expected_identity = _dataset_content_identity_from_code(
            code_payload,
            selected_datasets,
        )
        if content_identity != expected_identity:
            raise ValueError(
                "Dataset was prepared with a different numerical pipeline revision"
            )
    if expected_numerical_pipeline_digest is not None and (
        not isinstance(expected_numerical_pipeline_digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected_numerical_pipeline_digest) is None
    ):
        raise ValueError("Expected numerical pipeline digest is invalid")
    if (
        expected_numerical_pipeline_digest is not None
        and pipeline_digest != expected_numerical_pipeline_digest
    ):
        raise ValueError("Dataset was prepared with a different numerical pipeline revision")

    code_release_digest = payload.get("code_release_digest")
    if not isinstance(code_release_digest, str) or not re.fullmatch(
        r"[0-9a-f]{64}", code_release_digest
    ):
        raise ValueError("Dataset readiness code release digest is invalid")


def _validate_quant_dataset_payload(
    payload,
    code_payload=None,
    expected_numerical_pipeline_digest=None,
    stage_contract=None,
):
    if payload.get("schema_version") != 2 or payload.get("kind") != "stage1-dataset":
        raise ValueError("Unsupported numerical dataset readiness schema")
    if payload.get("state") != "ready":
        raise ValueError(
            "Stage 1 dataset manifest is not ready: {}".format(payload.get("state", "missing"))
        )
    selection_id = payload.get("selection_id")
    selection_sha256 = payload.get("selection_sha256")
    dataset_request_sha256 = payload.get("dataset_request_sha256")
    if not isinstance(selection_id, str) or not re.fullmatch(
        r"selection-[0-9a-f]{16}", selection_id
    ):
        raise ValueError("Dataset readiness selection_id is invalid")
    if not isinstance(selection_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", selection_sha256):
        raise ValueError("Dataset readiness selection digest is invalid")
    if not isinstance(dataset_request_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", dataset_request_sha256
    ):
        raise ValueError("Dataset readiness request digest is invalid")
    if selection_id != f"selection-{selection_sha256[:16]}":
        raise ValueError("Dataset readiness selection ID disagrees with its digest")
    requested_dataset = payload.get("requested_dataset")
    if (
        not isinstance(requested_dataset, dict)
        or _dataset_request_sha256(requested_dataset) != dataset_request_sha256
    ):
        raise ValueError("Dataset readiness requested dataset contract is invalid")
    expected_data_root = f"datasets/{dataset_request_sha256}"
    if payload.get("data_root_relative") != expected_data_root:
        raise ValueError("Dataset readiness namespace disagrees with its request digest")
    selected_stage = payload.get("selected_stage")
    if selected_stage not in {"stage1", "stage2"}:
        raise ValueError("Dataset readiness selected stage is invalid")
    stage_config_path = payload.get("stage_config_path")
    stage_config_sha256 = payload.get("stage_config_sha256")
    if (
        not isinstance(stage_config_path, str)
        or _safe_relative_path(stage_config_path).as_posix() != stage_config_path
        or not isinstance(stage_config_sha256, str)
        or not re.fullmatch(r"[0-9a-f]{64}", stage_config_sha256)
    ):
        raise ValueError("Dataset readiness stage config identity is invalid")

    approved_paths = {
        "raw": f"{expected_data_root}/raw/market.parquet",
        "bar_store_manifest": (
            f"{expected_data_root}/prepared/bar-store/bar-store.json"
        ),
        "symbol_index": f"{expected_data_root}/prepared/bar-store/symbol-index.parquet",
        "cutoff_ranges": f"{expected_data_root}/prepared/bar-store/cutoff-ranges.parquet",
        "dataset_manifest": f"{expected_data_root}/dataset-manifest.json",
        "download_manifest": f"{expected_data_root}/download-manifest.json",
        "request_log": f"{expected_data_root}/manifests/api-request-log.jsonl",
        "model_manifest": "cache/hf-models.json",
    }
    artifacts = {}
    for name, approved_path in approved_paths.items():
        artifact = _validate_artifact(payload, name)
        if artifact["relative_path"] != approved_path:
            raise ValueError(f"Numerical artifact path is not approved: {name}")
        if not re.fullmatch(r"[0-9a-f]{64}", artifact["sha256"]):
            raise ValueError(f"Numerical artifact digest is invalid: {name}")
        artifacts[name] = artifact

    profile = payload.get("dataset_profile")
    if profile not in PROFILE_DATASETS:
        raise ValueError("Dataset readiness profile is unsupported")
    if payload.get("selected_datasets") != PROFILE_DATASETS[profile]:
        raise ValueError("Dataset readiness providers disagree with its profile")
    if requested_dataset.get("profile") != profile:
        raise ValueError("Dataset readiness requested profile disagrees with prepared data")
    if requested_dataset.get("selected_datasets") != payload.get("selected_datasets"):
        raise ValueError("Dataset readiness requested providers disagree with prepared data")
    if requested_dataset.get("date_range") != payload.get("date_range"):
        raise ValueError("Dataset readiness requested date range disagrees with prepared data")
    if stage_contract is not None:
        if profile != stage_contract.dataset_profile:
            raise ValueError("Dataset profile differs from the selected training config")
        if sorted(payload.get("model_repositories", [])) != sorted(
            stage_contract.required_model_repositories
        ):
            raise ValueError("Offline model cache does not cover the selected model config")

    symbols = payload.get("symbols")
    if (
        not isinstance(symbols, dict)
        or not isinstance(symbols.get("count"), int)
        or isinstance(symbols.get("count"), bool)
        or symbols["count"] < 1
        or not isinstance(symbols.get("values"), list)
        or symbols["values"] != sorted(set(symbols["values"]))
        or symbols["count"] != len(symbols["values"])
        or any(
            not isinstance(symbol, str) or not SYMBOL_PATTERN.fullmatch(symbol)
            for symbol in symbols["values"]
        )
    ):
        raise ValueError("Dataset readiness symbol universe is invalid")
    if payload.get("universe_sha256") != _payload_sha256(symbols):
        raise ValueError("Dataset readiness universe digest is inconsistent")

    preparation_provenance = payload.get("preparation_provenance")
    if not isinstance(preparation_provenance, dict):
        raise ValueError("Dataset readiness preparation provenance is invalid")
    storage_preparation = payload.get("storage_preparation_spec")
    if (
        not isinstance(storage_preparation, dict)
        or set(storage_preparation) != set(storage_preparation_spec(preparation_provenance))
        or any(
            preparation_provenance.get(field) != storage_preparation.get(field)
            for field in storage_preparation
        )
    ):
        raise ValueError("Dataset readiness storage preparation provenance is invalid")
    storage_preparation_sha256 = payload.get("storage_preparation_spec_sha256")
    if (
        not isinstance(storage_preparation_sha256, str)
        or not re.fullmatch(r"[0-9a-f]{64}", storage_preparation_sha256)
        or storage_preparation_sha256 != _payload_sha256(storage_preparation)
    ):
        raise ValueError("Dataset readiness storage preparation spec is invalid")
    if (
        storage_preparation.get("bar_store_schema_version")
        != BAR_STORE_SCHEMA_VERSION
        or storage_preparation.get("storage_kind") != BAR_STORE_KIND
        or storage_preparation.get("window_materialized") is not False
        or storage_preparation.get("labels_materialized") is not False
        or storage_preparation.get("max_horizon")
        != DEFAULT_DATASET_STORAGE_PREPARATION["max_horizon"]
    ):
        raise ValueError("Quant dataset must use the supported lazy bar-store contract")
    window_size = storage_preparation.get("window_size")
    purge_bars = storage_preparation.get("purge_bars")
    effective_embargo_bars = storage_preparation.get("effective_embargo_bars")
    if (
        not isinstance(window_size, int)
        or isinstance(window_size, bool)
        or window_size < 2
        or not isinstance(purge_bars, int)
        or isinstance(purge_bars, bool)
        or purge_bars < 0
        or not isinstance(effective_embargo_bars, int)
        or isinstance(effective_embargo_bars, bool)
        or effective_embargo_bars
        < DEFAULT_DATASET_STORAGE_PREPARATION["max_horizon"]
        or re.fullmatch(
            r"[0-9a-f]{64}",
            str(storage_preparation.get("benchmark_mapping_sha256", "")),
        )
        is None
    ):
        raise ValueError("Dataset readiness storage preparation values are invalid")
    numeric_storage_values = (
        storage_preparation.get("max_abs_log_return"),
        storage_preparation.get("train_fraction"),
        storage_preparation.get("validation_fraction"),
    )
    if any(
        not isinstance(value, int | float)
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) <= 0.0
        for value in numeric_storage_values
    ):
        raise ValueError("Dataset readiness numeric storage values are invalid")
    if (
        float(storage_preparation["train_fraction"])
        + float(storage_preparation["validation_fraction"])
        >= 1.0
    ):
        raise ValueError("Dataset readiness split fractions are invalid")
    requested_preparation = requested_dataset.get("preparation")
    if not isinstance(requested_preparation, dict):
        raise ValueError("Dataset readiness requested preparation contract is invalid")
    for field in storage_preparation:
        if storage_preparation.get(field) != requested_preparation.get(field):
            raise ValueError(
                f"Dataset readiness storage field {field} disagrees with its request"
            )
    validate_fixed_split_audit(
        payload.get("split_audit"), storage_preparation.get("fixed_split"),
        minimum_evaluation_dates=MIN_FIXED_EVALUATION_DATES,
    )
    training_security_scope = payload.get("training_security_scope")
    if (
        not isinstance(training_security_scope, str)
        or not training_security_scope
        or preparation_provenance.get("training_security_scope")
        != training_security_scope
    ):
        raise ValueError("Dataset readiness training security scope is inconsistent")
    split_counts = payload.get("split_counts")
    if (
        not isinstance(split_counts, dict)
        or set(split_counts) != {"train", "validation", "test"}
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 1
            for value in split_counts.values()
        )
        or sum(split_counts.values()) != payload.get("valid_cutoff_count")
    ):
        raise ValueError("Dataset readiness split counts are invalid")
    _validate_causal_split_audit(
        payload.get("split_audit"),
        split_counts,
        "Dataset readiness",
    )
    repositories = payload.get("model_repositories")
    if (
        not isinstance(repositories, list)
        or not repositories
        or repositories != sorted(set(repositories))
        or artifacts["model_manifest"]["row_count"] != len(repositories)
    ):
        raise ValueError("Dataset readiness model repositories are invalid")
    smoke = payload.get("time_series_smoke_test")
    if (
        payload.get("models_local_files_only_verified") is not True
        or not isinstance(smoke, dict)
        or smoke.get("passed") is not True
        or smoke.get("local_files_only") is not True
        or smoke.get("backend") != "kronos"
    ):
        raise ValueError("Kronos cache has not passed offline hidden-state verification")

    _validate_dataset_code_compatibility(
        payload,
        code_payload=code_payload,
        expected_numerical_pipeline_digest=expected_numerical_pipeline_digest,
    )
    return payload


def _guard_lifecycle_state(
    payload,
    *,
    expected_kind,
    expected_pod_id,
    active_run_id="",
):
    """Validate one lifecycle marker before an external guard may terminate a Pod."""

    if expected_kind not in GUARD_LIFECYCLE_KINDS:
        raise ValueError("Guard lifecycle kind is unsupported")
    if not isinstance(expected_pod_id, str) or not re.fullmatch(
        r"[A-Za-z0-9_-]+", expected_pod_id
    ):
        raise ValueError("Guard expected Pod ID is invalid")
    if active_run_id and (
        not isinstance(active_run_id, str)
        or RUN_ID_PATTERN.fullmatch(active_run_id) is None
        or "--" in active_run_id
    ):
        raise ValueError("Guard active run ID is invalid")
    if not isinstance(payload, dict):
        raise ValueError("Guard lifecycle marker must contain a JSON object")
    if payload.get("kind") != expected_kind or payload.get("pod_id") != expected_pod_id:
        raise ValueError("Guard lifecycle identity does not match the monitored Pod")

    state = payload.get("state")
    if state not in GUARD_LIFECYCLE_STATES:
        raise ValueError("Guard lifecycle state is unsupported")

    if payload.get("schema_version") != LIFECYCLE_SCHEMA_VERSION:
        raise ValueError("Guard lifecycle marker has an unsupported schema")

    if (
        state in CPU_PREPARATION_RESUMABLE_STATES
        and expected_kind != "stage1-cpu-preparation"
    ):
        raise ValueError(
            "Resumable acquisition states are valid only for CPU preparation"
        )
    if (
        expected_kind == "stage1-cpu-preparation"
        and state == "downloaded"
        and payload.get("exit_code") is None
    ):
        return "downloaded_active"
    if state in CPU_PREPARATION_RESUMABLE_STATES and (
        type(payload.get("exit_code")) is not int or payload.get("exit_code") != 75
    ):
        raise ValueError("Terminal resumable CPU preparation state requires exit_code=75")
    if state == "ready":
        if expected_kind == "stage1-training" and payload.get("training_completed") is not True:
            raise ValueError("Ready training lifecycle is incomplete")
        if expected_kind == "stage1-validation" and payload.get("validation_completed") is not True:
            raise ValueError("Ready validation lifecycle is incomplete")

    run_id = payload.get("wandb_run_id", "")
    if expected_kind in {"stage1-training", "stage1-validation"} and (
        not isinstance(run_id, str)
        or RUN_ID_PATTERN.fullmatch(run_id) is None
        or "--" in run_id
    ):
        raise ValueError("Guard lifecycle run ID is invalid")
    if active_run_id and run_id != active_run_id:
        raise ValueError("Guard lifecycle run ID does not match the active run")
    return state


def command_guard_lifecycle_state(arguments):
    try:
        payload = json.load(sys.stdin)
    except (TypeError, ValueError) as error:
        raise ValueError("Guard lifecycle input is not valid JSON") from error
    state = _guard_lifecycle_state(
        payload,
        expected_kind=arguments.expected_kind,
        expected_pod_id=arguments.expected_pod_id,
        active_run_id=arguments.active_run_id,
    )
    print(state)
    return 0


def _verify_dataset_artifacts(payload, network_volume_root):
    volume_root = network_volume_root.resolve()
    artifact_names = (
        "raw",
        "bar_store_manifest",
        "symbol_index",
        "cutoff_ranges",
        "dataset_manifest",
        "download_manifest",
        "request_log",
        "model_manifest",
    )
    paths = {}
    for name in artifact_names:
        artifact = payload[name]
        path = (volume_root / _safe_relative_path(artifact["relative_path"])).resolve()
        try:
            path.relative_to(volume_root)
        except ValueError as error:
            raise ValueError("Dataset artifact escapes NETWORK_VOLUME_ROOT") from error
        if not path.is_file():
            raise ValueError(f"Dataset artifact is missing: {path}")
        if path.stat().st_size != artifact["size_bytes"] or _sha256(path) != artifact["sha256"]:
            raise ValueError(f"Dataset artifact integrity changed: {path}")
        paths[name] = path

    dataset_payload = _load_json(str(paths["dataset_manifest"]))
    if (
        dataset_payload.get("schema_version") != "4.0"
        or dataset_payload.get("kind") != "ohlcv-bar-store-dataset"
        or dataset_payload.get("state") != "ready"
        or dataset_payload.get("dataset_profile") != payload.get("dataset_profile")
        or dataset_payload.get("selected_datasets") != payload.get("selected_datasets")
        or dataset_payload.get("training_security_scope")
        != payload.get("training_security_scope")
        or dataset_payload.get("data_content_identity")
        != payload.get("data_content_identity")
        or dataset_payload.get("data_pipeline_digest") != payload.get("data_pipeline_digest")
        or dataset_payload.get("storage_preparation_spec")
        != payload.get("storage_preparation_spec")
        or dataset_payload.get("storage_preparation_spec_sha256")
        != payload.get("storage_preparation_spec_sha256")
        or dataset_payload.get("universe_sha256") != payload.get("universe_sha256")
        or dataset_payload.get("split_counts") != payload.get("split_counts")
        or dataset_payload.get("split_audit") != payload.get("split_audit")
    ):
        raise ValueError("Dataset readiness marker disagrees with dataset-manifest.json")
    dataset_artifacts = dataset_payload.get("artifacts")
    if not isinstance(dataset_artifacts, dict):
        raise ValueError("Dataset manifest artifacts are invalid")
    for name in ("raw", "bar_store_manifest", "symbol_index", "cutoff_ranges"):
        source = dataset_artifacts.get(name)
        if (
            not isinstance(source, dict)
            or source.get("sha256") != payload[name]["sha256"]
            or source.get("size_bytes") != payload[name]["size_bytes"]
            or source.get("row_count") != payload[name]["row_count"]
        ):
            raise ValueError(f"Dataset readiness {name} artifact disagrees with its manifest")

    model_payload = _load_json(str(paths["model_manifest"]))
    repositories = model_payload.get("repositories")
    if (
        model_payload.get("local_files_only_verified") is not True
        or not isinstance(repositories, dict)
        or sorted(repositories) != payload.get("model_repositories")
        or model_payload.get("time_series_smoke_test") != payload.get("time_series_smoke_test")
    ):
        raise ValueError("Dataset and Kronos cache manifests disagree")
    for repository, snapshot_value in repositories.items():
        if not isinstance(repository, str) or not isinstance(snapshot_value, str):
            raise ValueError("Kronos cache manifest contains invalid entries")
        snapshot = Path(snapshot_value).resolve()
        try:
            snapshot.relative_to(volume_root)
        except ValueError as error:
            raise ValueError("Kronos snapshot escapes NETWORK_VOLUME_ROOT") from error
        if not snapshot.is_dir():
            raise ValueError(f"Kronos snapshot is missing: {repository}")


def command_check_dataset(arguments):
    code_payload = None
    if arguments.code_marker:
        code_payload = _validate_code_payload(_load_json(arguments.code_marker))
    stage_contract = (
        _load_stage_contract(arguments.stage_config) if arguments.stage_config else None
    )
    payload = _validate_quant_dataset_payload(
        _load_json(arguments.marker),
        code_payload=code_payload,
        expected_numerical_pipeline_digest=(
            arguments.expected_numerical_pipeline_digest
        ),
        stage_contract=stage_contract,
    )
    if arguments.network_volume_root is not None:
        _verify_dataset_artifacts(payload, arguments.network_volume_root)
    print(
        "Stage 1 data ready: {} raw rows, {} lazy valid cutoffs".format(
            payload["raw"]["row_count"], payload["valid_cutoff_count"]
        )
    )
    return 0


def command_stage_contract(arguments):
    contract = _load_stage_contract(arguments.config)
    json.dump(
        {
            "dataset_profile": contract.dataset_profile,
            "training_stage": contract.training_stage,
            "train_fraction": contract.train_fraction,
            "max_samples": contract.max_samples,
            "embargo_trading_days": contract.embargo_trading_days,
            "required_model_repositories": list(contract.required_model_repositories),
        },
        sys.stdout,
        ensure_ascii=False,
        sort_keys=True,
    )
    sys.stdout.write("\n")
    return 0


def _require_path_inside(path, root, label):
    resolved = Path(str(path)).resolve(strict=False)
    expected_root = root.resolve(strict=False)
    try:
        resolved.relative_to(expected_root)
    except ValueError as error:
        raise ValueError(f"{label} must be stored below {expected_root}") from error
    return resolved


def _validate_run_id(run_id, label="Run ID"):
    if not isinstance(run_id, str) or RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise ValueError(f"{label} must be a safe 1-120 character directory name")
    if "--" in run_id:
        raise ValueError(f"{label} must not contain the duplicated-component separator '--'")
    return run_id


def _checkpoint_artifact_contract_digest(payload, label):
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


def _validate_checkpoint_artifact_files(payload):
    expected_names = {
        "adapter.safetensors",
        "optimizer.pt",
        "scheduler.pt",
        "resolved-config.yaml",
    }
    artifact_files = payload.get("artifact_files")
    if not isinstance(artifact_files, dict) or set(artifact_files) != expected_names:
        raise ValueError("Trainer state has an incomplete checkpoint integrity manifest")
    for name, row in artifact_files.items():
        digest = row.get("sha256") if isinstance(row, dict) else None
        size_bytes = row.get("size_bytes") if isinstance(row, dict) else None
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or not isinstance(size_bytes, int)
            or isinstance(size_bytes, bool)
            or size_bytes < 1
        ):
            raise ValueError(f"Trainer state has invalid integrity metadata for {name}")


def _validate_runtime_robust_scales(payload):
    values = payload.get("runtime_robust_scales")
    if (
        not isinstance(values, list)
        or not 12 <= len(values) <= 14
        or any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) <= 0.0
            for value in values
        )
    ):
        raise ValueError("Trainer state has invalid runtime robust scales")


def _validate_run_lifecycle_paths(payload, volume_root):
    if payload.get("kind") not in {"stage1-training", "stage1-validation"}:
        return
    if payload.get("schema_version") != LIFECYCLE_SCHEMA_VERSION:
        raise ValueError(
            "Training and validation lifecycle schema_version must equal "
            f"{LIFECYCLE_SCHEMA_VERSION}"
        )
    run_id = _validate_run_id(
        payload.get("wandb_run_id"),
        "Training and validation lifecycle wandb_run_id",
    )

    checkpoint_value = payload.get("checkpoint")
    if checkpoint_value not in (None, ""):
        checkpoint = Path(str(checkpoint_value)).resolve(strict=False)
        expected_parent = (volume_root / "savedModel" / run_id).resolve(strict=False)
        if (
            checkpoint.parent != expected_parent
            or CHECKPOINT_NAME_PATTERN.fullmatch(checkpoint.name) is None
        ):
            raise ValueError(
                "Lifecycle checkpoint must equal savedModel/<wandb_run_id>/checkpoint-NNNNNN"
            )

    expected_evaluation_root = (volume_root / "evaluations" / run_id).resolve(strict=False)
    expected_validation_paths = {
        "result_path": expected_evaluation_root / "validation-benchmark.json",
    }
    for key, expected in expected_validation_paths.items():
        value = payload.get(key)
        if value not in (None, "") and Path(str(value)).resolve(strict=False) != expected:
            raise ValueError(f"Lifecycle {key} does not match wandb_run_id")

    log_path = payload.get("log_path")
    if log_path not in (None, ""):
        _require_path_inside(log_path, volume_root / "logs" / run_id, "Lifecycle log_path")
    recovery_path = payload.get("recovery_path")
    if recovery_path not in (None, ""):
        recovery = _require_path_inside(
            recovery_path,
            volume_root / "logs" / run_id,
            "Lifecycle recovery_path",
        )
        if recovery.name != "recovery.json":
            raise ValueError("Lifecycle recovery_path must identify recovery.json")


def command_completed_training_run(arguments):
    payload = _load_json(arguments.marker)
    if payload.get("kind") != "stage1-training":
        raise ValueError("Training lifecycle marker has the wrong kind")
    if payload.get("state") not in {"ready", "failed", "timed_out"}:
        raise ValueError(
            "Training lifecycle is not terminal; automatic validation may still be running"
        )
    if payload.get("training_completed") is not True:
        raise ValueError("Training lifecycle does not identify a completed training run")
    volume_root = arguments.network_volume_root.resolve(strict=False)
    _validate_run_lifecycle_paths(payload, volume_root)
    run_id = _validate_run_id(payload.get("wandb_run_id"), "Training lifecycle run ID")
    sys.stdout.write(run_id + "\n")
    return 0


def command_resumable_training_run(arguments):
    payload = _load_json(arguments.marker)
    if payload.get("kind") != "stage1-training":
        raise ValueError("Training lifecycle marker has the wrong kind")
    if payload.get("state") not in {"failed", "timed_out"}:
        raise ValueError("Training lifecycle is not failed or timed out")
    if payload.get("training_completed") is True:
        raise ValueError("Training already completed; resume standalone validation instead")
    volume_root = arguments.network_volume_root.resolve(strict=False)
    _validate_run_lifecycle_paths(payload, volume_root)
    run_id = _validate_run_id(payload.get("wandb_run_id"), "Training lifecycle run ID")
    sys.stdout.write(run_id + "\n")
    return 0


def _validate_training_completion_payload(payload, volume_root, expected_run_id):
    if payload.get("schema_version") != TRAINING_COMPLETION_SCHEMA_VERSION:
        raise ValueError("Training completion marker does not use the current immutable schema")
    if payload.get("kind") != "stage1-training-completion":
        raise ValueError("Training completion marker has the wrong kind")
    if payload.get("state") != "ready" or payload.get("training_completed") is not True:
        raise ValueError("Training completion marker does not confirm successful training")
    run_id = _validate_run_id(payload.get("run_id"), "Training completion run ID")
    if payload.get("run_key") != run_id:
        raise ValueError("Training completion run_key does not match run_id")
    if run_id != _validate_run_id(expected_run_id, "Expected training run ID"):
        raise ValueError("Training completion marker belongs to a different run")
    expected_manifest = (volume_root / "savedModel" / run_id / "run-manifest.json").resolve(
        strict=False
    )
    if Path(str(payload.get("run_manifest_path", ""))).resolve(strict=False) != expected_manifest:
        raise ValueError("Training completion marker has a non-canonical run manifest path")
    digest = payload.get("run_manifest_sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError("Training completion marker has an invalid run manifest digest")
    return run_id


def command_check_training_completion(arguments):
    volume_root = arguments.network_volume_root.resolve(strict=False)
    payload = _load_json(arguments.marker)
    run_id = _validate_training_completion_payload(payload, volume_root, arguments.run_id)
    if payload.get("run_manifest_sha256") != arguments.run_manifest_sha256:
        raise ValueError("Training completion marker does not match the remote run manifest")
    sys.stdout.write(run_id + "\n")
    return 0


def command_write_training_completion(arguments):
    volume_root = arguments.network_volume_root.resolve(strict=False)
    run_id = _validate_run_id(arguments.run_id, "Training completion run ID")
    output = arguments.output.resolve(strict=False)
    expected_output = (
        volume_root / "lifecycle" / "runs" / run_id / "training-completed.json"
    ).resolve(strict=False)
    if output != expected_output:
        raise ValueError(
            "Immutable training completion marker must equal "
            "lifecycle/runs/<run_id>/training-completed.json"
        )
    run_manifest = arguments.run_manifest.resolve(strict=False)
    expected_manifest = (volume_root / "savedModel" / run_id / "run-manifest.json").resolve(
        strict=False
    )
    if run_manifest != expected_manifest or not run_manifest.is_file():
        raise ValueError("Training completion requires the canonical run manifest")
    manifest_payload = _load_json(str(run_manifest))
    if manifest_payload.get("run_id") != run_id or manifest_payload.get("run_key") != run_id:
        raise ValueError("Run manifest identity does not match the completed run")
    _checkpoint_artifact_contract_digest(manifest_payload, "Run manifest")
    payload = {
        "schema_version": TRAINING_COMPLETION_SCHEMA_VERSION,
        "kind": "stage1-training-completion",
        "state": "ready",
        "training_completed": True,
        "completed_at": _utc_now(),
        "run_id": run_id,
        "run_key": run_id,
        "run_manifest_path": str(run_manifest),
        "run_manifest_sha256": _sha256(run_manifest),
        "pod_id": os.environ.get("RUNPOD_POD_ID", ""),
        "launch_id": arguments.launch_id,
    }
    if output.exists():
        existing = _load_json(str(output))
        _validate_training_completion_payload(existing, volume_root, run_id)
        if existing.get("run_manifest_sha256") != payload["run_manifest_sha256"]:
            raise ValueError("Immutable training completion marker conflicts with run manifest")
        return 0
    try:
        _write_immutable_json(output, payload)
    except FileExistsError as error:
        existing = _load_json(str(output))
        _validate_training_completion_payload(existing, volume_root, run_id)
        if existing.get("run_manifest_sha256") != payload["run_manifest_sha256"]:
            raise ValueError(
                "Immutable training completion marker conflicts with run manifest"
            ) from error
    return 0


def _validated_gpu_workflow_status(payload, *, expected_kind, volume_root):
    if expected_kind not in {"stage1-training", "stage1-validation"}:
        raise ValueError("GPU workflow lifecycle kind is unsupported")
    if payload.get("kind") != expected_kind:
        raise ValueError("GPU workflow lifecycle marker has the wrong kind")
    _validate_run_lifecycle_paths(payload, volume_root)
    state = payload.get("state")
    if state not in GPU_WORKFLOW_ACTIVE_STATES | GPU_WORKFLOW_TERMINAL_STATES:
        raise ValueError("GPU workflow lifecycle marker has an unsupported state")
    pod_id = payload.get("pod_id")
    if state in GPU_WORKFLOW_ACTIVE_STATES and (
        not isinstance(pod_id, str) or POD_ID_PATTERN.fullmatch(pod_id) is None
    ):
        raise ValueError("Active GPU workflow lifecycle marker has an invalid Pod ID")
    return state, pod_id or ""


def command_gpu_workflow_status(arguments):
    """Print a validated GPU lifecycle state and its owning Pod ID."""

    state, pod_id = _validated_gpu_workflow_status(
        _load_json(arguments.marker),
        expected_kind=arguments.kind,
        volume_root=arguments.network_volume_root.resolve(strict=False),
    )
    sys.stdout.write(f"{state}\t{pod_id}\n")
    return 0


def _runpod_pod_probe_state(payload, *, expected_pod_id, command_exit_code):
    """Classify only an explicit Pod object or an explicit RunPod 404 response."""

    if not isinstance(expected_pod_id, str) or POD_ID_PATTERN.fullmatch(
        expected_pod_id
    ) is None:
        raise ValueError("Expected RunPod Pod ID is invalid")
    if not isinstance(command_exit_code, int) or isinstance(command_exit_code, bool):
        raise ValueError("RunPod Pod lookup exit code is invalid")
    if not isinstance(payload, dict):
        raise ValueError("RunPod Pod lookup did not return a JSON object")
    if command_exit_code == 0:
        if payload.get("id") != expected_pod_id:
            raise ValueError("RunPod Pod lookup returned a different Pod")
        return "present"
    if payload.get("status") == 404 and payload.get("code") == "not_found":
        return "absent"
    code = payload.get("code", "unknown_error")
    status = payload.get("status", "unknown")
    raise ValueError(
        "RunPod Pod lookup is indeterminate "
        f"(exit_code={command_exit_code}, status={status}, code={code})"
    )


def command_runpod_pod_probe_state(arguments):
    state = _runpod_pod_probe_state(
        _load_json(arguments.response),
        expected_pod_id=arguments.expected_pod_id,
        command_exit_code=arguments.command_exit_code,
    )
    sys.stdout.write(state + "\n")
    return 0


def _orphaned_gpu_workflow_payload(
    payload,
    *,
    expected_kind,
    expected_pod_id,
    volume_root,
    minimum_age_seconds,
    now=None,
):
    """Build a terminal audit record after an exact RunPod Pod lookup returned 404."""

    if (
        not isinstance(minimum_age_seconds, int)
        or isinstance(minimum_age_seconds, bool)
        or minimum_age_seconds < 0
    ):
        raise ValueError("GPU lifecycle orphan minimum age is invalid")
    state, pod_id = _validated_gpu_workflow_status(
        payload,
        expected_kind=expected_kind,
        volume_root=volume_root,
    )
    if state not in GPU_WORKFLOW_ACTIVE_STATES:
        raise ValueError("Only an active GPU workflow lifecycle can be reconciled")
    if pod_id != expected_pod_id:
        raise ValueError("GPU workflow lifecycle Pod ID changed during reconciliation")

    observed_at = now or datetime.now(timezone.utc)
    generated_at = _parse_utc_timestamp(
        payload.get("generated_at"), "GPU workflow lifecycle generated_at"
    )
    age_seconds = (observed_at - generated_at).total_seconds()
    if age_seconds < minimum_age_seconds:
        raise ValueError(
            "GPU workflow lifecycle is too recent to reconcile as orphaned "
            f"(age_seconds={max(0, int(age_seconds))}, "
            f"minimum_seconds={minimum_age_seconds})"
        )

    reconciled_at = observed_at.astimezone(timezone.utc).isoformat()
    reconciled = {
        **payload,
        "state": "failed",
        "generated_at": reconciled_at,
        "exit_code": 1,
        "reason": ORPHANED_GPU_WORKFLOW_REASON,
        "timed_out": False,
        "reconciliation": {
            "previous_state": state,
            "pod_lookup": "not_found",
            "reconciled_at": reconciled_at,
        },
    }
    if expected_kind == "stage1-training":
        training_completed = payload.get("training_completed") is True
        reconciled["training_completed"] = training_completed
        reconciled["resume_discovery_required"] = not training_completed
    else:
        reconciled["validation_completed"] = False
        reconciled["resume_validation_required"] = True
    _validate_run_lifecycle_paths(reconciled, volume_root)
    return reconciled


def command_reconcile_orphaned_gpu_workflow(arguments):
    payload = _orphaned_gpu_workflow_payload(
        _load_json(arguments.marker),
        expected_kind=arguments.kind,
        expected_pod_id=arguments.expected_pod_id,
        volume_root=arguments.network_volume_root.resolve(strict=False),
        minimum_age_seconds=arguments.minimum_age_seconds,
    )
    json.dump(payload, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


def command_gpu_workflow_available(arguments):
    """Fail closed when a canonical singleton GPU lifecycle is still active."""

    state, _ = _validated_gpu_workflow_status(
        _load_json(arguments.marker),
        expected_kind=arguments.kind,
        volume_root=arguments.network_volume_root.resolve(strict=False),
    )
    if state in GPU_WORKFLOW_ACTIVE_STATES:
        raise ValueError("Another GPU workflow is active; refusing to create a competing paid Pod")
    sys.stdout.write(str(state) + "\n")
    return 0


def command_training_phase_completed(arguments):
    """Confirm that this exact launch completed training before post-training validation."""

    payload = _load_json(arguments.marker)
    if payload.get("kind") != "stage1-training":
        raise ValueError("Training lifecycle marker has the wrong kind")
    if payload.get("state") != "finalizing":
        raise ValueError("Training phase marker must be in finalizing state")
    if payload.get("training_completed") is not True:
        raise ValueError("Training phase marker does not confirm training completion")
    volume_root = arguments.network_volume_root.resolve(strict=False)
    _validate_run_lifecycle_paths(payload, volume_root)
    run_id = _validate_run_id(payload.get("wandb_run_id"), "Training lifecycle run ID")
    if run_id != _validate_run_id(arguments.run_id, "Expected training run ID"):
        raise ValueError("Training phase marker belongs to a different run")
    pod_id = os.environ.get("RUNPOD_POD_ID", "")
    if not pod_id or payload.get("pod_id") != pod_id:
        raise ValueError("Training phase marker belongs to a different Pod")
    if payload.get("launch_id") != arguments.launch_id:
        raise ValueError("Training phase marker belongs to a different launch")
    sys.stdout.write(run_id + "\n")
    return 0


def _validate_active_run_lifecycle(arguments, allowed_states):
    payload = _load_json(arguments.marker)
    if payload.get("kind") != arguments.kind:
        raise ValueError("Run lifecycle marker has the wrong kind")
    if payload.get("state") not in allowed_states:
        raise ValueError("Run lifecycle marker is not active for this finalizer")
    volume_root = arguments.network_volume_root.resolve(strict=False)
    _validate_run_lifecycle_paths(payload, volume_root)
    run_id = _validate_run_id(payload.get("wandb_run_id"), "Lifecycle run ID")
    if run_id != _validate_run_id(arguments.run_id, "Expected lifecycle run ID"):
        raise ValueError("Run lifecycle marker belongs to a different run")
    pod_id = os.environ.get("RUNPOD_POD_ID", "")
    if not pod_id or payload.get("pod_id") != pod_id:
        raise ValueError("Run lifecycle marker belongs to a different Pod")
    if payload.get("launch_id") != arguments.launch_id:
        raise ValueError("Run lifecycle marker belongs to a different launch")
    sys.stdout.write(run_id + "\n")
    return 0


def command_finalizing_run_lifecycle(arguments):
    """Validate one exact finalizing lifecycle before making it terminal."""

    return _validate_active_run_lifecycle(arguments, {"finalizing"})


def command_active_run_lifecycle(arguments):
    """Validate one exact preparing or finalizing lifecycle before termination."""

    return _validate_active_run_lifecycle(arguments, {"preparing", "finalizing"})


def command_resumable_cpu_preparation_lifecycle(arguments):
    """Validate a worker-published resumable CPU marker before tmux preserves it."""

    payload = _load_json(arguments.marker)
    state = payload.get("state")
    if (
        payload.get("schema_version") != LIFECYCLE_SCHEMA_VERSION
        or payload.get("kind") != "stage1-cpu-preparation"
        or state not in CPU_PREPARATION_RESUMABLE_STATES
        or payload.get("exit_code") != 75
    ):
        raise ValueError("CPU preparation lifecycle is not a resumable terminal state")
    volume_root = arguments.network_volume_root.resolve(strict=False)
    _validate_run_lifecycle_paths(payload, volume_root)
    pod_id = os.environ.get("RUNPOD_POD_ID", "")
    if not pod_id or payload.get("pod_id") != pod_id:
        raise ValueError("CPU preparation lifecycle belongs to a different Pod")
    if payload.get("launch_id") != arguments.launch_id:
        raise ValueError("CPU preparation lifecycle belongs to a different launch")

    progress_path = Path(str(payload.get("progress_path", ""))).resolve(strict=False)
    datasets_root = (volume_root / "datasets").resolve(strict=False)
    try:
        relative = progress_path.relative_to(datasets_root)
    except ValueError as error:
        raise ValueError("Dataset progress escapes the datasets root") from error
    if (
        len(relative.parts) != 2
        or re.fullmatch(r"[0-9a-f]{64}", relative.parts[0]) is None
        or relative.parts[1] != "download-progress.json"
        or not progress_path.is_file()
        or progress_path.is_symlink()
    ):
        raise ValueError("Dataset progress path is not canonical")
    progress = _load_json(progress_path)
    identity = progress.get("identity")
    context = progress.get("context")
    expected_progress_state = (
        "downloaded" if state == "waiting_for_preparation" else state
    )
    if (
        progress.get("schema_version") != 1
        or progress.get("kind") != "ohlcv-download-progress"
        or progress.get("state") != expected_progress_state
        or not isinstance(identity, dict)
        or not isinstance(context, dict)
        or progress.get("identity_sha256") != _payload_sha256(identity)
        or context.get("dataset_request_sha256") != relative.parts[0]
    ):
        raise ValueError("Dataset progress does not match the resumable lifecycle")
    sys.stdout.write(str(state) + "\n")
    return 0


def _validated_checkpoint_leaderboard(leaderboard, run_id):
    if leaderboard.get("run_id") != run_id or leaderboard.get("run_key") != run_id:
        raise ValueError(
            "Checkpoint leaderboard identity does not match its canonical run directory"
        )
    if leaderboard.get("selection_source") != "validation":
        raise ValueError("Checkpoint leaderboard must be selected from validation metrics")
    contract_digest = _checkpoint_artifact_contract_digest(
        leaderboard,
        "Checkpoint leaderboard",
    )
    monitor = leaderboard.get("monitor")
    mode = leaderboard.get("mode")
    if not isinstance(monitor, str) or not monitor.startswith("primary_5d/"):
        raise ValueError("Checkpoint leaderboard has an invalid validation monitor")
    if mode not in {"min", "max"}:
        raise ValueError("Checkpoint leaderboard has an invalid selection mode")
    checkpoints = leaderboard.get("checkpoints")
    save_top_k = leaderboard.get("save_top_k")
    if (
        not isinstance(checkpoints, list)
        or not checkpoints
        or not isinstance(save_top_k, int)
        or isinstance(save_top_k, bool)
        or save_top_k < 1
        or save_top_k > 10
        or len(checkpoints) > save_top_k
    ):
        raise ValueError("Checkpoint leaderboard contains an invalid retained set")
    checkpoint_names = []
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
            or not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
        ):
            raise ValueError("Checkpoint leaderboard contains a non-canonical checkpoint row")
        checkpoint_names.append(name)
    if len(checkpoint_names) != len(set(checkpoint_names)):
        raise ValueError("Checkpoint leaderboard repeats a checkpoint")
    expected_order = sorted(
        checkpoints,
        key=lambda row: (
            float(row["value"]) if mode == "min" else -float(row["value"]),
            -int(row["global_step"]),
        ),
    )
    if checkpoint_names != [row["path"] for row in expected_order]:
        raise ValueError("Checkpoint leaderboard ranks do not follow the validation policy")
    return monitor, mode, checkpoints, contract_digest


def _validated_best_checkpoint_pointer(
    pointer,
    leaderboard,
    *,
    run_id,
    monitor,
    mode,
    checkpoints,
    contract_digest,
):
    if pointer.get("run_id") != run_id or pointer.get("run_key") != run_id:
        raise ValueError("Best-checkpoint pointer identity does not match its run directory")
    if pointer.get("selection_source") != "validation":
        raise ValueError("Best checkpoint must be selected from validation metrics")
    if _checkpoint_artifact_contract_digest(pointer, "Best-checkpoint pointer") != contract_digest:
        raise ValueError("Best-checkpoint pointer and leaderboard contracts disagree")
    checkpoint_name = pointer.get("path")
    checkpoint_names = [row["path"] for row in checkpoints]
    if (
        not isinstance(checkpoint_name, str)
        or CHECKPOINT_NAME_PATTERN.fullmatch(checkpoint_name) is None
        or leaderboard.get("best_checkpoint") != checkpoint_name
        or checkpoint_names[0] != checkpoint_name
    ):
        raise ValueError("Best-checkpoint pointer and leaderboard disagree")
    if pointer.get("monitor") != monitor or pointer.get("mode") != mode:
        raise ValueError("Best-checkpoint pointer and leaderboard selection policy disagree")
    pointer_value = pointer.get("value")
    if (
        not isinstance(pointer_value, (int, float))
        or isinstance(pointer_value, bool)
        or not math.isfinite(float(pointer_value))
        or float(pointer_value) != float(checkpoints[0]["value"])
    ):
        raise ValueError("Best-checkpoint pointer and leaderboard values disagree")
    transaction_id = leaderboard.get("transaction_id")
    if (
        not isinstance(transaction_id, str)
        or re.fullmatch(r"[0-9a-f]{32}", transaction_id) is None
        or pointer.get("transaction_id") != transaction_id
    ):
        raise ValueError("Best-checkpoint pointer and leaderboard transactions disagree")
    return checkpoint_name


def _validate_trainer_state_identity(
    trainer_state,
    *,
    run_id,
    checkpoint_name,
    row,
    monitor,
    mode,
    contract_digest,
):
    _validate_checkpoint_artifact_files(trainer_state)
    _validate_runtime_robust_scales(trainer_state)
    if _checkpoint_artifact_contract_digest(trainer_state, "Trainer state") != contract_digest:
        raise ValueError("Trainer state and checkpoint leaderboard contracts disagree")
    if trainer_state.get("run_id") != run_id or trainer_state.get("run_key") != run_id:
        raise ValueError("Trainer state identity does not match its canonical run directory")
    global_step = trainer_state.get("global_step")
    epoch = trainer_state.get("epoch")
    batch_index = trainer_state.get("batch_index")
    selection = trainer_state.get("selection")
    selection_value = selection.get("value") if isinstance(selection, dict) else None
    if (
        global_step != row.get("global_step")
        or not isinstance(global_step, int)
        or isinstance(global_step, bool)
        or f"checkpoint-{global_step:06d}" != checkpoint_name
        or trainer_state.get("training_stage") not in {"stage1", "stage2"}
        or not isinstance(epoch, int)
        or isinstance(epoch, bool)
        or epoch < 0
        or not isinstance(batch_index, int)
        or isinstance(batch_index, bool)
        or batch_index < 0
        or not isinstance(trainer_state.get("rng_state"), dict)
        or not isinstance(selection, dict)
        or selection.get("source") != "validation"
        or selection.get("metric") != monitor
        or selection.get("mode") != mode
        or not isinstance(selection_value, (int, float))
        or isinstance(selection_value, bool)
        or not math.isfinite(float(selection_value))
        or float(selection_value) != float(row["value"])
    ):
        raise ValueError("Trainer state and validation leaderboard disagree")


def command_best_checkpoint_name(arguments):
    run_id = _validate_run_id(arguments.run_id)
    pointer = _load_json(arguments.pointer)
    leaderboard = _load_json(arguments.leaderboard)
    trainer_state = _load_json(arguments.trainer_state)
    monitor, mode, checkpoints, contract_digest = _validated_checkpoint_leaderboard(
        leaderboard,
        run_id,
    )
    checkpoint_name = _validated_best_checkpoint_pointer(
        pointer,
        leaderboard,
        run_id=run_id,
        monitor=monitor,
        mode=mode,
        checkpoints=checkpoints,
        contract_digest=contract_digest,
    )
    _validate_trainer_state_identity(
        trainer_state,
        run_id=run_id,
        checkpoint_name=checkpoint_name,
        row=checkpoints[0],
        monitor=monitor,
        mode=mode,
        contract_digest=contract_digest,
    )
    sys.stdout.write(checkpoint_name + "\n")
    return 0


def command_checkpoint_download_names(arguments):
    """Print the manifest-authoritative checkpoint names selected for download."""

    run_id = _validate_run_id(arguments.run_id)
    pointer = _load_json(arguments.pointer)
    leaderboard = _load_json(arguments.leaderboard)
    monitor, mode, checkpoints, contract_digest = _validated_checkpoint_leaderboard(
        leaderboard,
        run_id,
    )
    best_checkpoint = _validated_best_checkpoint_pointer(
        pointer,
        leaderboard,
        run_id=run_id,
        monitor=monitor,
        mode=mode,
        checkpoints=checkpoints,
        contract_digest=contract_digest,
    )
    selected = (
        [row["path"] for row in checkpoints] if arguments.scope == "all" else [best_checkpoint]
    )
    sys.stdout.write("\n".join(selected) + "\n")
    return 0


def command_checkpoint_pointer_name(arguments):
    run_id = _validate_run_id(arguments.run_id)
    pointer = _load_json(arguments.pointer)
    if pointer.get("run_id") != run_id or pointer.get("run_key") != run_id:
        raise ValueError("Best-checkpoint pointer identity does not match its run directory")
    checkpoint_name = pointer.get("path")
    _checkpoint_artifact_contract_digest(pointer, "Best-checkpoint pointer")
    if (
        pointer.get("selection_source") != "validation"
        or not isinstance(checkpoint_name, str)
        or CHECKPOINT_NAME_PATTERN.fullmatch(checkpoint_name) is None
    ):
        raise ValueError("Best-checkpoint pointer is not canonical validation output")
    sys.stdout.write(checkpoint_name + "\n")
    return 0


def command_check_retained_checkpoint(arguments):
    run_id = _validate_run_id(arguments.run_id)
    checkpoint_name = arguments.checkpoint_name
    if CHECKPOINT_NAME_PATTERN.fullmatch(checkpoint_name) is None:
        raise ValueError("Retained checkpoint has a non-canonical name")
    leaderboard = _load_json(arguments.leaderboard)
    trainer_state = _load_json(arguments.trainer_state)
    monitor, mode, checkpoints, contract_digest = _validated_checkpoint_leaderboard(
        leaderboard,
        run_id,
    )
    matching = [row for row in checkpoints if row["path"] == checkpoint_name]
    if len(matching) != 1:
        raise ValueError("Requested checkpoint is not retained by the validation leaderboard")
    _validate_trainer_state_identity(
        trainer_state,
        run_id=run_id,
        checkpoint_name=checkpoint_name,
        row=matching[0],
        monitor=monitor,
        mode=mode,
        contract_digest=contract_digest,
    )
    sys.stdout.write(checkpoint_name + "\n")
    return 0


def command_check_run_manifest(arguments):
    run_id = _validate_run_id(arguments.run_id)
    payload = _load_json(arguments.manifest)
    if payload.get("run_id") != run_id or payload.get("run_key") != run_id:
        raise ValueError("Run manifest identity does not match its canonical run directory")
    _checkpoint_artifact_contract_digest(payload, "Run manifest")
    sys.stdout.write(run_id + "\n")
    return 0


def command_write_state(arguments):
    output = arguments.output.resolve()
    volume_root = arguments.network_volume_root.resolve()
    try:
        output.relative_to(volume_root)
    except ValueError as error:
        raise ValueError("Lifecycle state must be stored on NETWORK_VOLUME_ROOT") from error
    if output == Path("/workspace") or Path("/workspace") in output.parents:
        raise ValueError("Lifecycle state must never use /workspace")
    canonical_outputs = {
        "stage1-cpu-preparation": (
            volume_root / "lifecycle" / "stage1" / "cpu-preparation.json"
        ),
        "stage1-mixed-finalization": (
            volume_root / "lifecycle" / "stage1" / "mixed-finalization.json"
        ),
        "stage1-training": volume_root / "lifecycle" / "stage1" / "training.json",
        "stage1-validation": volume_root / "lifecycle" / "stage1" / "validation.json",
    }
    expected_output = canonical_outputs.get(arguments.kind)
    if expected_output is None:
        raise ValueError("Lifecycle kind is unsupported")
    if output != expected_output.resolve(strict=False):
        raise ValueError(f"{arguments.kind} lifecycle must equal {expected_output}")
    if (
        arguments.state in CPU_PREPARATION_RESUMABLE_STATES
        and arguments.kind != "stage1-cpu-preparation"
    ):
        raise ValueError(
            f"{arguments.state} is valid only for stage1-cpu-preparation"
        )
    if (
        arguments.state in CPU_PREPARATION_RESUMABLE_STATES - {"downloaded"}
        and arguments.exit_code != 75
    ):
        raise ValueError(f"{arguments.state} lifecycle requires exit_code=75")
    if arguments.state == "downloaded" and arguments.exit_code not in {None, 75}:
        raise ValueError("downloaded lifecycle accepts only no exit code or exit_code=75")
    resolved_progress_path = None
    if arguments.progress_path:
        if arguments.kind != "stage1-cpu-preparation":
            raise ValueError(
                "Only stage1-cpu-preparation lifecycle may reference download progress"
            )
        progress_path = Path(arguments.progress_path)
        if not progress_path.is_file() or progress_path.is_symlink():
            raise ValueError("Download progress must be a regular file")
        resolved_progress_path = progress_path.resolve()
        datasets_root = (volume_root / "datasets").resolve(strict=False)
        try:
            progress_relative = resolved_progress_path.relative_to(datasets_root)
        except ValueError as error:
            raise ValueError("Download progress must be stored below datasets/") from error
        if (
            len(progress_relative.parts) != 2
            or re.fullmatch(r"[0-9a-f]{64}", progress_relative.parts[0]) is None
            or progress_relative.parts[1] != "download-progress.json"
        ):
            raise ValueError(
                "Download progress must equal datasets/<dataset-request-sha256>/"
                "download-progress.json"
            )
        progress_payload = _load_json(resolved_progress_path)
        if not isinstance(progress_payload, dict):
            raise ValueError("Download progress must contain a JSON object")
        progress_identity = progress_payload.get("identity")
        progress_context = progress_payload.get("context")
        if (
            progress_payload.get("schema_version") != 1
            or progress_payload.get("kind") != "ohlcv-download-progress"
            or not isinstance(progress_identity, dict)
            or not isinstance(progress_context, dict)
            or progress_payload.get("identity_sha256") != _payload_sha256(progress_identity)
            or progress_context.get("dataset_request_sha256")
            != progress_relative.parts[0]
        ):
            raise ValueError("Download progress identity does not match its dataset namespace")
        expected_progress_state = (
            "downloaded"
            if arguments.state == "waiting_for_preparation"
            else arguments.state
        )
        if (
            arguments.state in CPU_PREPARATION_RESUMABLE_STATES
            and progress_payload.get("state") != expected_progress_state
        ):
            raise ValueError(f"{arguments.state} lifecycle requires matching download progress")
    elif arguments.state in CPU_PREPARATION_RESUMABLE_STATES:
        raise ValueError(f"{arguments.state} lifecycle requires --progress-path")
    inherited = {}
    if arguments.inherit_existing and output.is_file():
        with output.open("r", encoding="utf-8") as stream:
            existing = json.load(stream)
        if not isinstance(existing, dict):
            raise ValueError("Existing lifecycle state must be a JSON object")
        _validate_run_lifecycle_paths(existing, volume_root)
        if existing.get("kind") != arguments.kind:
            raise ValueError("Existing lifecycle kind does not match this finalizer")
        existing_state = existing.get("state")
        active_states = {"preparing", "finalizing"}
        terminal_states = {
            "ready",
            "failed",
            "timed_out",
            "waiting_for_provider",
            "waiting_for_budget",
            "waiting_for_resume",
            "waiting_for_preparation",
            "downloaded",
        }
        if existing_state not in active_states | terminal_states:
            raise ValueError("Existing lifecycle state is unsupported for finalization")
        current_pod_id = os.environ.get("RUNPOD_POD_ID", "")
        same_launch = (
            bool(current_pod_id)
            and existing.get("pod_id") == current_pod_id
            and existing.get("launch_id") == arguments.launch_id
        )
        if existing_state in active_states and not same_launch:
            if not current_pod_id or existing.get("pod_id") != current_pod_id:
                raise ValueError("Existing lifecycle pod_id does not match this Pod")
            raise ValueError("Existing lifecycle launch_id does not match this launch")
        if same_launch:
            requested_run_id = arguments.wandb_run_id or ""
            existing_run_id = existing.get("wandb_run_id") or ""
            if (requested_run_id or existing_run_id) and requested_run_id != existing_run_id:
                raise ValueError("Existing lifecycle wandb_run_id does not match this run")
            for key in (
                "wandb_run_id",
                "training_completed",
                "recovery_path",
                "progress_path",
                "log_path",
                "max_runtime_seconds",
                "checkpoint",
                "result_path",
            ):
                if existing.get(key) not in (None, ""):
                    inherited[key] = existing[key]
    payload = {
        **inherited,
        "schema_version": LIFECYCLE_SCHEMA_VERSION,
        "kind": arguments.kind,
        "state": arguments.state,
        "generated_at": _utc_now(),
        "pod_id": os.environ.get("RUNPOD_POD_ID", ""),
        "launch_id": arguments.launch_id,
    }
    if arguments.exit_code is not None:
        payload["exit_code"] = arguments.exit_code
    if arguments.log_path:
        payload["log_path"] = arguments.log_path
    if arguments.wandb_run_id:
        payload["wandb_run_id"] = arguments.wandb_run_id
    if arguments.recovery_path:
        payload["recovery_path"] = arguments.recovery_path
    if resolved_progress_path is not None:
        payload["progress_path"] = str(resolved_progress_path)
    if arguments.max_runtime_seconds is not None:
        payload["max_runtime_seconds"] = arguments.max_runtime_seconds
    if arguments.kind == "stage1-training" and arguments.state in {
        "finalizing",
        "ready",
        "failed",
        "timed_out",
    }:
        payload["training_completed"] = (
            bool(arguments.training_completed)
            if arguments.training_completed is not None
            else bool(inherited.get("training_completed", arguments.state == "ready"))
        )
        payload["timed_out"] = arguments.state == "timed_out"
    if arguments.kind == "stage1-validation" and arguments.state in {
        "ready",
        "failed",
        "timed_out",
    }:
        payload["validation_completed"] = arguments.state == "ready"
        payload["timed_out"] = arguments.state == "timed_out"
    _validate_run_lifecycle_paths(payload, volume_root)
    _atomic_json(output, payload)
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True

    code = subparsers.add_parser("code-manifest")
    code.add_argument("--project-root", type=Path, required=True)
    code.add_argument("--remote-project-dir", required=True)
    code.add_argument("--state", choices=("syncing", "ready"), required=True)
    code.add_argument("--pipeline-path", action="append", default=[])
    code.add_argument("paths", nargs="+")
    code.set_defaults(handler=command_code_manifest)

    check_code = subparsers.add_parser("check-code")
    check_code.add_argument("--marker", required=True)
    check_code.add_argument("--project-root", type=Path)
    check_code.set_defaults(handler=command_check_code)

    numerical_pipeline = subparsers.add_parser("code-numerical-pipeline-digest")
    numerical_pipeline.add_argument("--marker", required=True)
    numerical_pipeline.add_argument(
        "--dataset-profile",
        choices=tuple(PROFILE_DATASETS),
    )
    numerical_pipeline.set_defaults(handler=command_code_numerical_pipeline_digest)

    quarantine = subparsers.add_parser("quarantine-stale-code")
    quarantine.add_argument("--marker", required=True)
    quarantine.add_argument("--project-root", type=Path, required=True)
    quarantine.add_argument("--network-volume-root", type=Path, required=True)
    quarantine.add_argument("--quarantine-id")
    quarantine.set_defaults(handler=command_quarantine_stale_code)

    check_dataset = subparsers.add_parser("check-dataset")
    check_dataset.add_argument("--marker", required=True)
    check_dataset.add_argument("--code-marker")
    check_dataset.add_argument("--expected-numerical-pipeline-digest")
    check_dataset.add_argument("--network-volume-root", type=Path)
    check_dataset.add_argument("--stage-config", type=Path)
    check_dataset.set_defaults(handler=command_check_dataset)

    stage_contract = subparsers.add_parser("stage-contract")
    stage_contract.add_argument("--config", type=Path, required=True)
    stage_contract.set_defaults(handler=command_stage_contract)

    completed_training = subparsers.add_parser("completed-training-run")
    completed_training.add_argument("--marker", required=True)
    completed_training.add_argument("--network-volume-root", type=Path, required=True)
    completed_training.set_defaults(handler=command_completed_training_run)

    resumable_training = subparsers.add_parser("resumable-training-run")
    resumable_training.add_argument("--marker", required=True)
    resumable_training.add_argument("--network-volume-root", type=Path, required=True)
    resumable_training.set_defaults(handler=command_resumable_training_run)

    training_completion = subparsers.add_parser("check-training-completion")
    training_completion.add_argument("--marker", required=True)
    training_completion.add_argument("--network-volume-root", type=Path, required=True)
    training_completion.add_argument("--run-id", required=True)
    training_completion.add_argument("--run-manifest-sha256", required=True)
    training_completion.set_defaults(handler=command_check_training_completion)

    write_training_completion = subparsers.add_parser("write-training-completion")
    write_training_completion.add_argument("--output", type=Path, required=True)
    write_training_completion.add_argument("--network-volume-root", type=Path, required=True)
    write_training_completion.add_argument("--run-id", required=True)
    write_training_completion.add_argument("--run-manifest", type=Path, required=True)
    write_training_completion.add_argument("--launch-id", required=True)
    write_training_completion.set_defaults(handler=command_write_training_completion)

    gpu_workflow = subparsers.add_parser("gpu-workflow-available")
    gpu_workflow.add_argument("--marker", required=True)
    gpu_workflow.add_argument("--network-volume-root", type=Path, required=True)
    gpu_workflow.add_argument(
        "--kind",
        choices=("stage1-training", "stage1-validation"),
        required=True,
    )
    gpu_workflow.set_defaults(handler=command_gpu_workflow_available)

    gpu_workflow_status = subparsers.add_parser("gpu-workflow-status")
    gpu_workflow_status.add_argument("--marker", required=True)
    gpu_workflow_status.add_argument("--network-volume-root", type=Path, required=True)
    gpu_workflow_status.add_argument(
        "--kind",
        choices=("stage1-training", "stage1-validation"),
        required=True,
    )
    gpu_workflow_status.set_defaults(handler=command_gpu_workflow_status)

    pod_probe = subparsers.add_parser("runpod-pod-probe-state")
    pod_probe.add_argument("--response", required=True)
    pod_probe.add_argument("--expected-pod-id", required=True)
    pod_probe.add_argument("--command-exit-code", type=int, required=True)
    pod_probe.set_defaults(handler=command_runpod_pod_probe_state)

    orphaned_gpu_workflow = subparsers.add_parser(
        "reconcile-orphaned-gpu-workflow"
    )
    orphaned_gpu_workflow.add_argument("--marker", required=True)
    orphaned_gpu_workflow.add_argument(
        "--network-volume-root", type=Path, required=True
    )
    orphaned_gpu_workflow.add_argument(
        "--kind",
        choices=("stage1-training", "stage1-validation"),
        required=True,
    )
    orphaned_gpu_workflow.add_argument("--expected-pod-id", required=True)
    orphaned_gpu_workflow.add_argument(
        "--minimum-age-seconds", type=int, default=60
    )
    orphaned_gpu_workflow.set_defaults(
        handler=command_reconcile_orphaned_gpu_workflow
    )

    completed_training_phase = subparsers.add_parser("training-phase-completed")
    completed_training_phase.add_argument("--marker", required=True)
    completed_training_phase.add_argument("--network-volume-root", type=Path, required=True)
    completed_training_phase.add_argument("--run-id", required=True)
    completed_training_phase.add_argument("--launch-id", required=True)
    completed_training_phase.set_defaults(handler=command_training_phase_completed)

    finalizing_lifecycle = subparsers.add_parser("finalizing-run-lifecycle")
    finalizing_lifecycle.add_argument("--marker", required=True)
    finalizing_lifecycle.add_argument("--network-volume-root", type=Path, required=True)
    finalizing_lifecycle.add_argument(
        "--kind",
        choices=("stage1-training", "stage1-validation"),
        required=True,
    )
    finalizing_lifecycle.add_argument("--run-id", required=True)
    finalizing_lifecycle.add_argument("--launch-id", required=True)
    finalizing_lifecycle.set_defaults(handler=command_finalizing_run_lifecycle)

    active_lifecycle = subparsers.add_parser("active-run-lifecycle")
    active_lifecycle.add_argument("--marker", required=True)
    active_lifecycle.add_argument("--network-volume-root", type=Path, required=True)
    active_lifecycle.add_argument(
        "--kind",
        choices=("stage1-training", "stage1-validation"),
        required=True,
    )
    active_lifecycle.add_argument("--run-id", required=True)
    active_lifecycle.add_argument("--launch-id", required=True)
    active_lifecycle.set_defaults(handler=command_active_run_lifecycle)

    resumable_cpu_preparation = subparsers.add_parser(
        "resumable-cpu-preparation-lifecycle"
    )
    resumable_cpu_preparation.add_argument("--marker", type=Path, required=True)
    resumable_cpu_preparation.add_argument(
        "--network-volume-root", type=Path, required=True
    )
    resumable_cpu_preparation.add_argument("--launch-id", required=True)
    resumable_cpu_preparation.set_defaults(
        handler=command_resumable_cpu_preparation_lifecycle
    )

    guard_lifecycle = subparsers.add_parser("guard-lifecycle-state")
    guard_lifecycle.add_argument(
        "--expected-kind",
        choices=tuple(sorted(GUARD_LIFECYCLE_KINDS)),
        required=True,
    )
    guard_lifecycle.add_argument("--expected-pod-id", required=True)
    guard_lifecycle.add_argument("--active-run-id", default="")
    guard_lifecycle.set_defaults(handler=command_guard_lifecycle_state)

    best_checkpoint = subparsers.add_parser("best-checkpoint-name")
    best_checkpoint.add_argument("--pointer", required=True)
    best_checkpoint.add_argument("--leaderboard", required=True)
    best_checkpoint.add_argument("--trainer-state", required=True)
    best_checkpoint.add_argument("--run-id", required=True)
    best_checkpoint.set_defaults(handler=command_best_checkpoint_name)

    checkpoint_download = subparsers.add_parser("checkpoint-download-names")
    checkpoint_download.add_argument("--pointer", required=True)
    checkpoint_download.add_argument("--leaderboard", required=True)
    checkpoint_download.add_argument("--run-id", required=True)
    checkpoint_download.add_argument("--scope", choices=("all", "best"), required=True)
    checkpoint_download.set_defaults(handler=command_checkpoint_download_names)

    checkpoint_pointer = subparsers.add_parser("checkpoint-pointer-name")
    checkpoint_pointer.add_argument("--pointer", required=True)
    checkpoint_pointer.add_argument("--run-id", required=True)
    checkpoint_pointer.set_defaults(handler=command_checkpoint_pointer_name)

    retained_checkpoint = subparsers.add_parser("check-retained-checkpoint")
    retained_checkpoint.add_argument("--leaderboard", required=True)
    retained_checkpoint.add_argument("--trainer-state", required=True)
    retained_checkpoint.add_argument("--run-id", required=True)
    retained_checkpoint.add_argument("--checkpoint-name", required=True)
    retained_checkpoint.set_defaults(handler=command_check_retained_checkpoint)

    run_manifest = subparsers.add_parser("check-run-manifest")
    run_manifest.add_argument("--manifest", required=True)
    run_manifest.add_argument("--run-id", required=True)
    run_manifest.set_defaults(handler=command_check_run_manifest)

    state = subparsers.add_parser("write-state")
    state.add_argument("--output", type=Path, required=True)
    state.add_argument("--network-volume-root", type=Path, required=True)
    state.add_argument("--kind", required=True)
    state.add_argument(
        "--state",
        choices=(
            "preparing",
            "finalizing",
            "ready",
            "failed",
            "timed_out",
            "waiting_for_provider",
            "waiting_for_budget",
            "waiting_for_resume",
            "waiting_for_preparation",
            "downloaded",
        ),
        required=True,
    )
    state.add_argument("--launch-id", required=True)
    state.add_argument("--exit-code", type=int)
    state.add_argument("--log-path")
    state.add_argument("--wandb-run-id")
    state.add_argument("--recovery-path")
    state.add_argument("--progress-path")
    state.add_argument("--max-runtime-seconds", type=int)
    state.add_argument("--training-completed", type=int, choices=(0, 1))
    state.add_argument("--inherit-existing", action="store_true")
    state.set_defaults(handler=command_write_state)
    return parser


def main():
    arguments = build_parser().parse_args()
    try:
        return arguments.handler(arguments)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"RunPod readiness check failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
