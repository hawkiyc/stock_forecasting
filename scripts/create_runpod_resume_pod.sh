#!/usr/bin/env bash

# Discover the latest retained checkpoint and create a fail-closed resume Pod.
set -Eeuo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
S3_WRAPPER="${SCRIPT_DIR}/runpod_s3_project.sh"
READINESS_HELPER="${SCRIPT_DIR}/runpod_readiness.py"
CHECKPOINT_PREFLIGHT="${SCRIPT_DIR}/runpod_remote_checkpoint_preflight.py"
VOLUME_MOUNT_PATH="${RUNPOD_VOLUME_MOUNT_PATH:-/runpod-volume}"
# shellcheck source=lib/runpod_cli.sh
source "${SCRIPT_DIR}/lib/runpod_cli.sh"
# shellcheck source=lib/runpod_project_env.sh
source "${SCRIPT_DIR}/lib/runpod_project_env.sh"
# shellcheck source=lib/runpod_selection.sh
source "${SCRIPT_DIR}/lib/runpod_selection.sh"

MAX_RUNTIME=12h
GPU_ID="NVIDIA GeForce RTX 5090"
TARGET_RUN_ID=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --maxRuntime|--max-runtime)
            [[ $# -ge 2 ]] || { echo "$1 requires a value" >&2; exit 2; }
            MAX_RUNTIME="$2"
            shift 2
            ;;
        --gpuId|--gpu-id)
            [[ $# -ge 2 ]] || { echo "$1 requires a value" >&2; exit 2; }
            GPU_ID="$2"
            shift 2
            ;;
        --*)
            echo "Unknown resume option: $1" >&2
            exit 2
            ;;
        *)
            if [[ -n "${TARGET_RUN_ID}" ]]; then
                echo "Usage: create_runpod_resume_pod.sh [--maxRuntime DURATION] [--gpuId GPU_ID] [RUN_ID]" >&2
                exit 2
            fi
            TARGET_RUN_ID="$1"
            shift
            ;;
    esac
done

runpod_validate_gpu_id "${GPU_ID}"
MAX_RUNTIME_SECONDS="$(runpod_duration_seconds "${MAX_RUNTIME}" --maxRuntime)"
runpod_load_s3_env "${PROJECT_ROOT}"
runpod_load_active_selection "${PROJECT_ROOT}"

if [[ -z "${TARGET_RUN_ID}" ]]; then
    TRAINING_LIFECYCLE_JSON="$(bash "${S3_WRAPPER}" s3 cp \
        "s3://${RUNPOD_NETWORK_VOLUME_ID}/lifecycle/stage1/training.json" - \
        --only-show-errors)"
    TARGET_RUN_ID="$(printf '%s\n' "${TRAINING_LIFECYCLE_JSON}" \
        | python3 "${READINESS_HELPER}" resumable-training-run \
            --marker - \
            --network-volume-root "${VOLUME_MOUNT_PATH}")"
fi
if [[ ! "${TARGET_RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,119}$ \
    || "${TARGET_RUN_ID}" == *--* ]]; then
    echo "RUN_ID must be a safe 1-120 character directory name" >&2
    exit 2
fi

COMPLETION_KEYS="$(bash "${S3_WRAPPER}" s3api list-objects-v2 \
    --bucket "${RUNPOD_NETWORK_VOLUME_ID}" \
    --prefix "lifecycle/runs/${TARGET_RUN_ID}/training-completed.json" \
    --query 'Contents[].Key' \
    --output text)"
for completion_key in ${COMPLETION_KEYS}; do
    if [[ "${completion_key}" == "lifecycle/runs/${TARGET_RUN_ID}/training-completed.json" ]]; then
        echo "Training is already complete for ${TARGET_RUN_ID}; use the validate command" >&2
        exit 2
    fi
done

LOCAL_CONFIG_PATH="${PROJECT_ROOT}/${RUNPOD_CONFIG}"
CHECKPOINT_NAME="$(python3 "${CHECKPOINT_PREFLIGHT}" \
    --s3-wrapper "${S3_WRAPPER}" \
    --bucket "${RUNPOD_NETWORK_VOLUME_ID}" \
    --run-id "${TARGET_RUN_ID}" \
    --config "${LOCAL_CONFIG_PATH}" \
    --selection-policy latest)"
if [[ ! "${CHECKPOINT_NAME}" =~ ^checkpoint-[0-9]{6,}$ ]]; then
    echo "Remote checkpoint preflight did not return a canonical checkpoint" >&2
    exit 3
fi

export WANDB_RUN_ID="${TARGET_RUN_ID}"
export RESUME_CHECKPOINT="${VOLUME_MOUNT_PATH}/savedModel/${TARGET_RUN_ID}/${CHECKPOINT_NAME}"
export RUNPOD_CLI_MAX_RUNTIME_SECONDS="${MAX_RUNTIME_SECONDS}"
export RUNPOD_CLI_HARD_LIMIT_SECONDS="$((MAX_RUNTIME_SECONDS + 3600))"
export RUNPOD_CLI_TERMINATE_AFTER="$(((MAX_RUNTIME_SECONDS + 3659) / 60))m"
export RUNPOD_CLI_GPU_ID="${GPU_ID}"
export RUNPOD_GPU_WORKFLOW=train
exec bash "${SCRIPT_DIR}/create_runpod_pod.sh"
