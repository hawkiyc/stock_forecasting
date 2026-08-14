#!/usr/bin/env bash

# Create a GPU Pod for the latest completed run or an explicitly selected historical run.
set -Eeuo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/runpod_paths.sh
source "${SCRIPT_DIR}/lib/runpod_paths.sh"

VALIDATION_VOLUME_ROOT="${RUNPOD_VOLUME_MOUNT_PATH:-/runpod-volume}"
runpod_validate_absolute_path "${VALIDATION_VOLUME_ROOT}" RUNPOD_VOLUME_MOUNT_PATH
case "${VALIDATION_VOLUME_ROOT}" in
    /workspace|/workspace/*)
        echo "Network volumes must not be mounted under /workspace" >&2
        exit 2
        ;;
esac
if [[ -n "${PROJECT_ROOT:-}" ]]; then
    runpod_validate_path_in_root \
        "${PROJECT_ROOT}" "${VALIDATION_VOLUME_ROOT}" \
        PROJECT_ROOT RUNPOD_VOLUME_MOUNT_PATH
fi
if [[ -n "${SAVED_MODEL_ROOT:-}" ]]; then
    runpod_validate_path_in_root \
        "${SAVED_MODEL_ROOT}" "${VALIDATION_VOLUME_ROOT}" \
        SAVED_MODEL_ROOT RUNPOD_VOLUME_MOUNT_PATH
fi

VALIDATION_TARGET_RUN_ID=""
VALIDATION_RESUME="${VALIDATION_RESUME:-1}"
VALIDATION_FORCE_RECOMPUTE="${VALIDATION_FORCE_RECOMPUTE:-0}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --force)
            VALIDATION_FORCE_RECOMPUTE=1
            VALIDATION_RESUME=0
            ;;
        --no-resume)
            VALIDATION_RESUME=0
            ;;
        --resume)
            VALIDATION_RESUME=1
            ;;
        --*)
            echo "Unknown validation option: $1" >&2
            exit 2
            ;;
        *)
            if [[ -n "${VALIDATION_TARGET_RUN_ID}" ]]; then
                echo "Usage: bash scripts/create_runpod_validation_pod.sh [--force|--no-resume|--resume] [RUN_ID]" >&2
                exit 2
            fi
            VALIDATION_TARGET_RUN_ID="$1"
            ;;
    esac
    shift
done

if [[ -n "${VALIDATION_TARGET_RUN_ID}" ]]; then
    if [[ ! "${VALIDATION_TARGET_RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,119}$ \
        || "${VALIDATION_TARGET_RUN_ID}" == *--* ]]; then
        echo "RUN_ID must be a safe 1-120 character directory name" >&2
        exit 2
    fi
    export VALIDATION_RUN_ID="${VALIDATION_TARGET_RUN_ID}"
    export VALIDATION_CHECKPOINT=""
else
    # Clear stale shell values so zero arguments always selects the latest lifecycle.
    export VALIDATION_RUN_ID=""
    export VALIDATION_CHECKPOINT=""
fi

export RUNPOD_GPU_WORKFLOW=validation
export RUNPOD_POD_NAME="${RUNPOD_POD_NAME:-fin-ts-multimodal-validation}"
export WANDB_RUN_ID=""
export RESUME_CHECKPOINT=""
export MAX_RUNTIME_SECONDS="${RUNPOD_VALIDATION_MAX_RUNTIME_SECONDS:-${MAX_RUNTIME_SECONDS:-21600}}"
export RUNPOD_HARD_LIMIT_SECONDS="${RUNPOD_VALIDATION_HARD_LIMIT_SECONDS:-${RUNPOD_HARD_LIMIT_SECONDS:-25200}}"
export RUNPOD_TERMINATE_AFTER="${RUNPOD_VALIDATION_TERMINATE_AFTER:-${RUNPOD_TERMINATE_AFTER:-7h}}"
export VALIDATION_RECOMPUTE_FULL_MODEL="${VALIDATION_RECOMPUTE_FULL_MODEL:-1}"
export VALIDATION_RESUME VALIDATION_FORCE_RECOMPUTE

exec bash "${SCRIPT_DIR}/create_runpod_pod.sh"
