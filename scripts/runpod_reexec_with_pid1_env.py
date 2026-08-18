#!/usr/bin/env python3
"""Re-exec an SSH-started workflow with an allowlisted RunPod PID 1 environment."""

import argparse
import os
import sys
from pathlib import Path

ALLOWED_NAMES = frozenset(
    {
        "CACHE_ROOT",
        "DATA_ROOT",
        "EODHD_API_TOKEN",
        "FIN_TS_DATASET_PROFILE",
        "HF_TOKEN",
        "LOG_ROOT",
        "MAX_RUNTIME_SECONDS",
        "NETWORK_VOLUME_ROOT",
        "PROJECT_ROOT",
        "RESUME_CHECKPOINT",
        "RUNPOD_CONFIG",
        "RUNPOD_CPU_COUNT",
        "RUNPOD_CPU_EODHD_QPS",
        "RUNPOD_CPU_MAX_API_CALLS",
        "RUNPOD_CPU_MAX_RUNTIME_SECONDS",
        "RUNPOD_CPU_PREPARE_RESERVE_SECONDS",
        "RUNPOD_CPU_TAIWAN_QPS",
        "RUNPOD_EXPECTED_VOLUME_ID",
        "RUNPOD_EXPECTED_CUDA_PREFIX",
        "RUNPOD_EXPECTED_TORCH_VERSION",
        "RUNPOD_EXPECTED_UBUNTU_VERSION",
        "RUNPOD_IMAGE",
        "RUNPOD_POD_ID",
        "RUNPOD_PROVIDER_MAX_BACKOFF_SECONDS",
        "RUNPOD_DATASET_REQUEST_SHA256",
        "RUNPOD_REMOTE_SELECTION_PATH",
        "RUNPOD_REQUESTED_CPU_COUNT",
        "RUNPOD_ROLE",
        "RUNPOD_SELECTION_ID",
        "RUNPOD_SELECTION_SHA256",
        "RUNPOD_SHUTDOWN_ACTION",
        "RUNPOD_STAGE",
        "RUNPOD_STAGE_CONFIG_SHA256",
        "RUNPOD_VOLUME_ROOT",
        "RUNPOD_VOLUME_ID",
        "SAVED_MODEL_ROOT",
        "STAGE1_DATA_END",
        "STAGE1_DATA_START",
        "STAGE1_SYMBOL_LIMIT",
        "STAGE1_US_ETF_SYMBOLS",
        "STAGE1_US_SYMBOLS",
        "VALIDATION_CHECKPOINT",
        "VALIDATION_DISABLE_WANDB",
        "VALIDATION_RECOMPUTE_FULL_MODEL",
        "VALIDATION_RESUME",
        "VALIDATION_FORCE_RECOMPUTE",
        "VALIDATION_RUN_ID",
        "WANDB_ARTIFACT_DIR",
        "WANDB_API_KEY",
        "WANDB_CACHE_DIR",
        "WANDB_DIR",
        "WANDB_ENTITY",
        "WANDB_PROJECT",
        "WANDB_RUN_ID",
    }
)
CLEAR_ONLY_NAMES = frozenset(
    {
        "CHECKPOINT_ROOT",
        "OUTPUT_DIR",
        "POETRY_VERSION",
        "RUNPOD_HF_HOME",
        "RUNPOD_LAUNCH_ID",
        "RUNPOD_POETRY_BIN",
        "RUNPOD_POETRY_CACHE_DIR",
        "RUNPOD_PIP_CACHE_DIR",
        "RUNPOD_PYTHON_BIN",
        "RUNPOD_RUN_KEY",
        "RUNPOD_SHUTDOWN_DIR",
        "RUNPOD_SHUTDOWN_MARKER",
        "RUNPOD_SOURCE_CONFIG_SHA256",
        "RUNPOD_TMPDIR",
        "RUNPOD_TMUX_LOG_FILE",
        "RUNPOD_TORCH_HOME",
        "RUNPOD_TRAINING_COMPLETED_MARKER",
        "RUNPOD_TRAINING_LIFECYCLE_MARKER",
        "RUNPOD_VALIDATION_LIFECYCLE_MARKER",
        "RUNPOD_WANDB_ARTIFACT_DIR",
        "RUNPOD_WANDB_CACHE_DIR",
        "RUNPOD_XDG_CACHE_HOME",
        "HF_HOME",
        "PIP_CACHE_DIR",
        "TRANSFORMERS_CACHE",
        "TMPDIR",
        "XDG_CACHE_HOME",
        "TORCH_HOME",
        "POETRY_CACHE_DIR",
    }
)
FORBIDDEN_NAMES = frozenset(
    {
        "RUNPOD_API_KEY",
        "RUNPOD_S3_ACCESS_KEY_ID",
        "RUNPOD_S3_SECRET_ACCESS_KEY",
    }
)
ROLE_SECRET_NAMES = frozenset({"EODHD_API_TOKEN", "HF_TOKEN", "WANDB_API_KEY"})


def read_allowlisted_environment(path):
    """Read only approved names from a NUL-delimited process environment."""

    selected = {}
    for entry in path.read_bytes().split(b"\0"):
        if not entry or b"=" not in entry:
            continue
        raw_name, raw_value = entry.split(b"=", maxsplit=1)
        name = raw_name.decode("utf-8", errors="strict")
        if name in ALLOWED_NAMES:
            selected[name] = raw_value.decode("utf-8", errors="strict")
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("/proc/1/environ"))
    parser.add_argument("command", nargs=argparse.REMAINDER)
    arguments = parser.parse_args()
    if arguments.source != Path("/proc/1/environ") and os.environ.get("RUNPOD_TEST_MODE") != "1":
        parser.error("A custom environment source is allowed only in test mode")
    command = list(arguments.command)
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        parser.error("A command is required after --")

    imported = read_allowlisted_environment(arguments.source)
    for required in ("RUNPOD_POD_ID", "RUNPOD_ROLE"):
        if not imported.get(required):
            raise ValueError(f"PID 1 environment is missing {required}")
    environment = dict(os.environ)
    # PID 1 is authoritative for every job-scoped value. Clear stale SSH values
    # before importing the allowlist so an old path cannot leak into a new Pod.
    for name in FORBIDDEN_NAMES | ROLE_SECRET_NAMES | ALLOWED_NAMES | CLEAR_ONLY_NAMES:
        environment.pop(name, None)
    role = imported["RUNPOD_ROLE"]
    if role == "cpu-prep":
        imported.pop("WANDB_API_KEY", None)
    elif role in {"gpu-train", "gpu-validation"}:
        imported.pop("EODHD_API_TOKEN", None)
        imported.pop("HF_TOKEN", None)
    else:
        raise ValueError(f"Unsupported RUNPOD_ROLE in PID 1 environment: {role}")
    environment.update(imported)
    environment["RUNPOD_SSH_ENV_IMPORTED"] = "1"
    os.execvpe(command[0], command, environment)
    return 70


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, UnicodeError, ValueError) as error:
        print(f"Unable to import RunPod PID 1 environment: {error}", file=sys.stderr)
        raise SystemExit(2) from error
