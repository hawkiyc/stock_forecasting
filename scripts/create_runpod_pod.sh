#!/usr/bin/env bash

# Create a training Pod locally and arm a second, external termination deadline.
set -Eeuo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
S3_WRAPPER="${SCRIPT_DIR}/runpod_s3_project.sh"
READINESS_HELPER="${SCRIPT_DIR}/runpod_readiness.py"
REMOTE_CHECKPOINT_PREFLIGHT="${SCRIPT_DIR}/runpod_remote_checkpoint_preflight.py"
# shellcheck source=lib/runpod_paths.sh
source "${SCRIPT_DIR}/lib/runpod_paths.sh"
# shellcheck source=lib/runpod_project_env.sh
source "${SCRIPT_DIR}/lib/runpod_project_env.sh"
runpod_load_create_env "${LOCAL_PROJECT_ROOT}"
# shellcheck source=lib/runpod_selection.sh
source "${SCRIPT_DIR}/lib/runpod_selection.sh"
runpod_load_active_selection "${LOCAL_PROJECT_ROOT}"

# Explicit CLI choices take precedence over the immutable selection's legacy runtime defaults.
if [[ -n "${RUNPOD_CLI_MAX_RUNTIME_SECONDS:-}" ]]; then
    MAX_RUNTIME_SECONDS="${RUNPOD_CLI_MAX_RUNTIME_SECONDS}"
    RUNPOD_HARD_LIMIT_SECONDS="${RUNPOD_CLI_HARD_LIMIT_SECONDS}"
    RUNPOD_TERMINATE_AFTER="${RUNPOD_CLI_TERMINATE_AFTER}"
fi

RUNPOD_NETWORK_VOLUME_ID="${RUNPOD_NETWORK_VOLUME_ID:-}"
RUNPOD_VOLUME_MOUNT_PATH="${RUNPOD_VOLUME_MOUNT_PATH:-/runpod-volume}"
PROJECT_ROOT="${PROJECT_ROOT:-${RUNPOD_VOLUME_MOUNT_PATH}/stock_forecasting}"
SAVED_MODEL_ROOT="${SAVED_MODEL_ROOT:-${RUNPOD_VOLUME_MOUNT_PATH}/savedModel}"
RUNPOD_GPU_WORKFLOW="${RUNPOD_GPU_WORKFLOW:-train}"
case "${RUNPOD_GPU_WORKFLOW}" in
    train)
        RUNPOD_POD_NAME="${RUNPOD_POD_NAME:-fin-ts-multimodal-poc}"
        RUNPOD_GPU_ROLE=gpu-train
        RUNPOD_GUARD_LIFECYCLE_KEY=lifecycle/stage1/training.json
        RUNPOD_TMUX_WORKFLOW=stage1-train
        ;;
    validation)
        RUNPOD_POD_NAME="${RUNPOD_POD_NAME:-fin-ts-multimodal-validation}"
        RUNPOD_GPU_ROLE=gpu-validation
        RUNPOD_GUARD_LIFECYCLE_KEY=lifecycle/stage1/validation.json
        RUNPOD_TMUX_WORKFLOW=stage1-validate
        ;;
    *)
        echo "RUNPOD_GPU_WORKFLOW must be train or validation" >&2
        exit 2
        ;;
esac
RUNPOD_GPU_ID="${RUNPOD_CLI_GPU_ID:-${RUNPOD_GPU_ID:-NVIDIA GeForce RTX 5090}}"
RUNPOD_GPU_COUNT="${RUNPOD_GPU_COUNT:-1}"
RUNPOD_IMAGE="${RUNPOD_IMAGE:-runpod/pytorch:1.0.7-cu1281-torch291-ubuntu2404}"
RUNPOD_MIN_CUDA_VERSION="${RUNPOD_MIN_CUDA_VERSION:-12.8}"
RUNPOD_EXPECTED_TORCH_VERSION="${RUNPOD_EXPECTED_TORCH_VERSION:-2.9.1+cu128}"
RUNPOD_EXPECTED_CUDA_PREFIX="${RUNPOD_EXPECTED_CUDA_PREFIX:-12.8}"
RUNPOD_EXPECTED_UBUNTU_VERSION="${RUNPOD_EXPECTED_UBUNTU_VERSION:-24.04}"
RUNPOD_WANDB_SECRET_NAME="${RUNPOD_WANDB_SECRET_NAME:-wandb_api_key}"
WANDB_PROJECT="${WANDB_PROJECT:-fin-ts-quant}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
RUNPOD_CLOUD_TYPE="${RUNPOD_CLOUD_TYPE:-SECURE}"
RUNPOD_DATACENTER_ID="${RUNPOD_DATACENTER_ID:-EU-RO-1}"
RUNPOD_CONTAINER_DISK_GB="${RUNPOD_CONTAINER_DISK_GB:-50}"
DATA_ROOT="${RUNPOD_VOLUME_MOUNT_PATH}/datasets/${RUNPOD_DATASET_REQUEST_SHA256}"
REMOTE_SELECTION_MOUNT_PATH="${RUNPOD_VOLUME_MOUNT_PATH}/${RUNPOD_REMOTE_SELECTION_RELATIVE_PATH}"
if [[ "${RUNPOD_GPU_WORKFLOW}" == "validation" ]]; then
    MAX_RUNTIME_SECONDS="${RUNPOD_VALIDATION_MAX_RUNTIME_SECONDS:-${MAX_RUNTIME_SECONDS}}"
    RUNPOD_HARD_LIMIT_SECONDS="${RUNPOD_VALIDATION_HARD_LIMIT_SECONDS:-${RUNPOD_HARD_LIMIT_SECONDS}}"
    RUNPOD_TERMINATE_AFTER="${RUNPOD_VALIDATION_TERMINATE_AFTER:-${RUNPOD_TERMINATE_AFTER}}"
fi
RUNPOD_GUARD_LOG_DIR="${RUNPOD_GUARD_LOG_DIR:-${HOME:-/tmp}/.local/state/runpod-guards}"
RUNPOD_GUARD_LAUNCHER="${SCRIPT_DIR}/launch_runpod_guard.sh"
WANDB_RUN_ID="${WANDB_RUN_ID:-}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
VALIDATION_RECOMPUTE_FULL_MODEL="${VALIDATION_RECOMPUTE_FULL_MODEL:-0}"
VALIDATION_RESUME="${VALIDATION_RESUME:-1}"
VALIDATION_FORCE_RECOMPUTE="${VALIDATION_FORCE_RECOMPUTE:-0}"
VALIDATION_RUN_ID="${VALIDATION_RUN_ID:-}"
VALIDATION_CHECKPOINT="${VALIDATION_CHECKPOINT:-}"

if [[ "${VALIDATION_FORCE_RECOMPUTE}" == "1" ]]; then
    VALIDATION_RESUME=0
    VALIDATION_RECOMPUTE_FULL_MODEL=1
fi

if [[ "${RUNPOD_GPU_WORKFLOW}" == "validation" ]]; then
    if [[ -n "${WANDB_RUN_ID}" || -n "${RESUME_CHECKPOINT}" ]]; then
        echo "Validation Pods must not carry a training resume identity" >&2
        exit 2
    fi
elif [[ -n "${VALIDATION_RUN_ID}" || -n "${VALIDATION_CHECKPOINT}" ]]; then
    echo "Training Pods must not carry a validation identity" >&2
    exit 2
fi

# Allocate a fresh run identity on the control host so the paid Pod, the external
# guard, W&B, checkpoints, logs, and lifecycle records all share one exact ID.
FRESH_TRAINING_RUN=0
if [[ "${RUNPOD_GPU_WORKFLOW}" == "train" \
    && -z "${WANDB_RUN_ID}" && -z "${RESUME_CHECKPOINT}" ]]; then
    WANDB_RUN_ID="run-$(date -u +%Y%m%dT%H%M%SZ)-${RANDOM}${RANDOM}"
    FRESH_TRAINING_RUN=1
fi

if [[ -n "${RUNPOD_POD_ID:-}" && "${RUNPOD_TEST_MODE:-0}" != "1" ]]; then
    echo "create_runpod_pod.sh must run outside the RunPod Pod" >&2
    exit 2
fi
if [[ -z "${RUNPOD_NETWORK_VOLUME_ID}" ]]; then
    echo "RUNPOD_NETWORK_VOLUME_ID is required" >&2
    exit 2
fi
if [[ ! "${RUNPOD_NETWORK_VOLUME_ID}" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo "RUNPOD_NETWORK_VOLUME_ID contains invalid characters" >&2
    exit 2
fi
runpod_validate_absolute_path "${RUNPOD_VOLUME_MOUNT_PATH}" RUNPOD_VOLUME_MOUNT_PATH
case "${RUNPOD_VOLUME_MOUNT_PATH}" in
    /workspace|/workspace/*)
        echo "Network volumes must not be mounted under /workspace" >&2
        exit 2
        ;;
esac
runpod_validate_path_in_root \
    "${PROJECT_ROOT}" "${RUNPOD_VOLUME_MOUNT_PATH}" \
    PROJECT_ROOT RUNPOD_VOLUME_MOUNT_PATH
runpod_validate_path_in_root \
    "${SAVED_MODEL_ROOT}" "${RUNPOD_VOLUME_MOUNT_PATH}" \
    SAVED_MODEL_ROOT RUNPOD_VOLUME_MOUNT_PATH
if [[ "${SAVED_MODEL_ROOT}" != "${RUNPOD_VOLUME_MOUNT_PATH}/savedModel" ]]; then
    echo "SAVED_MODEL_ROOT must equal RUNPOD_VOLUME_MOUNT_PATH/savedModel" >&2
    exit 2
fi
if [[ ! "${RUNPOD_VOLUME_MOUNT_PATH}" =~ ^/[A-Za-z0-9._/-]+$ \
    || ! "${PROJECT_ROOT}" =~ ^/[A-Za-z0-9._/-]+$ ]]; then
    echo "Volume and project paths contain unsupported characters" >&2
    exit 2
fi
if [[ ! "${RUNPOD_GPU_COUNT}" =~ ^[1-9][0-9]*$ ]]; then
    echo "RUNPOD_GPU_COUNT must be a positive integer" >&2
    exit 2
fi
if [[ ! "${RUNPOD_CONTAINER_DISK_GB}" =~ ^[1-9][0-9]*$ ]]; then
    echo "RUNPOD_CONTAINER_DISK_GB must be a positive integer" >&2
    exit 2
fi
if [[ ! "${RUNPOD_MIN_CUDA_VERSION}" =~ ^[0-9]+\.[0-9]+$ ]]; then
    echo "RUNPOD_MIN_CUDA_VERSION must use a value such as 12.8" >&2
    exit 2
fi
if [[ ! "${RUNPOD_IMAGE}" =~ ^[A-Za-z0-9._/:+-]+$ ]]; then
    echo "RUNPOD_IMAGE contains unsupported characters" >&2
    exit 2
fi
if [[ ! "${RUNPOD_EXPECTED_TORCH_VERSION}" =~ ^[0-9]+\.[0-9]+\.[0-9]+\+cu[0-9]+$ ]]; then
    echo "RUNPOD_EXPECTED_TORCH_VERSION must look like 2.9.1+cu128" >&2
    exit 2
fi
if [[ ! "${RUNPOD_EXPECTED_CUDA_PREFIX}" =~ ^[0-9]+\.[0-9]+$ \
    || ! "${RUNPOD_EXPECTED_UBUNTU_VERSION}" =~ ^[0-9]+\.[0-9]+$ ]]; then
    echo "Expected CUDA and Ubuntu versions must use major.minor format" >&2
    exit 2
fi
if [[ "${RUNPOD_CLOUD_TYPE}" != "SECURE" && "${RUNPOD_CLOUD_TYPE}" != "COMMUNITY" ]]; then
    echo "RUNPOD_CLOUD_TYPE must be SECURE or COMMUNITY" >&2
    exit 2
fi
if [[ ! "${RUNPOD_DATACENTER_ID}" =~ ^[A-Z0-9]+(-[A-Z0-9]+)+$ ]]; then
    echo "RUNPOD_DATACENTER_ID has an invalid format" >&2
    exit 2
fi
if [[ ! "${MAX_RUNTIME_SECONDS}" =~ ^[1-9][0-9]*$ \
    || ! "${RUNPOD_HARD_LIMIT_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Runtime limits must be positive integers" >&2
    exit 2
fi
if [[ "${RUNPOD_HARD_LIMIT_SECONDS}" -le "${MAX_RUNTIME_SECONDS}" ]]; then
    echo "RUNPOD_HARD_LIMIT_SECONDS must exceed MAX_RUNTIME_SECONDS" >&2
    exit 2
fi
if [[ ! "${RUNPOD_TERMINATE_AFTER}" =~ ^[1-9][0-9]*[mhd]$ ]]; then
    echo "RUNPOD_TERMINATE_AFTER must use a duration such as 30m, 7h, or 2d" >&2
    exit 2
fi
if [[ "${VALIDATION_RECOMPUTE_FULL_MODEL}" != "0" \
    && "${VALIDATION_RECOMPUTE_FULL_MODEL}" != "1" ]]; then
    echo "VALIDATION_RECOMPUTE_FULL_MODEL must be 0 or 1" >&2
    exit 2
fi
for boolean_name in VALIDATION_RESUME VALIDATION_FORCE_RECOMPUTE; do
    if [[ "${!boolean_name}" != "0" && "${!boolean_name}" != "1" ]]; then
        echo "${boolean_name} must be 0 or 1" >&2
        exit 2
    fi
done
if [[ -n "${VALIDATION_RUN_ID}" \
    && ( ! "${VALIDATION_RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,119}$ \
        || "${VALIDATION_RUN_ID}" == *--* ) ]]; then
    echo "VALIDATION_RUN_ID must be a safe 1-120 character directory name" >&2
    exit 2
fi
if [[ -n "${VALIDATION_CHECKPOINT}" ]]; then
    if [[ -z "${VALIDATION_RUN_ID}" ]]; then
        echo "VALIDATION_CHECKPOINT requires VALIDATION_RUN_ID" >&2
        exit 2
    fi
    runpod_validate_path_in_root \
        "${VALIDATION_CHECKPOINT}" "${SAVED_MODEL_ROOT}" \
        VALIDATION_CHECKPOINT SAVED_MODEL_ROOT
    if [[ ! "${VALIDATION_CHECKPOINT}" =~ ^/[A-Za-z0-9._/-]+$ ]]; then
        echo "VALIDATION_CHECKPOINT contains unsupported characters" >&2
        exit 2
    fi
    if [[ "$(dirname "${VALIDATION_CHECKPOINT}")" \
        != "${SAVED_MODEL_ROOT}/${VALIDATION_RUN_ID}" \
        || ! "$(basename "${VALIDATION_CHECKPOINT}")" =~ ^checkpoint-[0-9]{6,}$ ]]; then
        echo "VALIDATION_CHECKPOINT must equal SAVED_MODEL_ROOT/VALIDATION_RUN_ID/checkpoint-NNNNNN (six or more digits)" >&2
        exit 2
    fi
fi
TERMINATE_AFTER_VALUE="${RUNPOD_TERMINATE_AFTER%?}"
case "${RUNPOD_TERMINATE_AFTER: -1}" in
    m) TERMINATE_AFTER_SECONDS=$((TERMINATE_AFTER_VALUE * 60)) ;;
    h) TERMINATE_AFTER_SECONDS=$((TERMINATE_AFTER_VALUE * 3600)) ;;
    d) TERMINATE_AFTER_SECONDS=$((TERMINATE_AFTER_VALUE * 86400)) ;;
esac
if [[ "${TERMINATE_AFTER_SECONDS}" -le "${MAX_RUNTIME_SECONDS}" ]]; then
    echo "RUNPOD_TERMINATE_AFTER must exceed MAX_RUNTIME_SECONDS" >&2
    exit 2
fi
if [[ ! "${RUNPOD_CONFIG}" =~ ^[A-Za-z0-9._/-]+$ \
    || "${RUNPOD_CONFIG}" == /* \
    || "/${RUNPOD_CONFIG}/" == *"/../"* \
    || "/${RUNPOD_CONFIG}/" == *"/./"* \
    || "${RUNPOD_CONFIG}" == *"//"* ]]; then
    echo "RUNPOD_CONFIG must be a safe path relative to PROJECT_ROOT" >&2
    exit 2
fi
LOCAL_CONFIG_PATH="${LOCAL_PROJECT_ROOT}/${RUNPOD_CONFIG}"
if [[ "${RUNPOD_TEST_MODE:-0}" != "1" && ! -r "${LOCAL_CONFIG_PATH}" ]]; then
    echo "RUNPOD_CONFIG is not readable in the uploaded local project: ${LOCAL_CONFIG_PATH}" >&2
    exit 2
fi
if [[ ! "${RUNPOD_WANDB_SECRET_NAME}" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo "RUNPOD_WANDB_SECRET_NAME contains unsupported characters" >&2
    exit 2
fi
if [[ ! "${WANDB_PROJECT}" =~ ^[A-Za-z0-9._-]+$ \
    || ! "${WANDB_ENTITY}" =~ ^[A-Za-z0-9._-]*$ ]]; then
    echo "W&B project or entity contains unsupported characters" >&2
    exit 2
fi
if [[ "${RUNPOD_GPU_WORKFLOW}" == "train" ]]; then
    if [[ ! "${WANDB_RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,119}$ \
        || "${WANDB_RUN_ID}" == *--* ]]; then
        echo "WANDB_RUN_ID must be a safe 1-120 character directory name" >&2
        exit 2
    fi
    if [[ "${FRESH_TRAINING_RUN}" == "0" \
        && ( -z "${WANDB_RUN_ID}" || -z "${RESUME_CHECKPOINT}" ) ]]; then
        echo "WANDB_RUN_ID and RESUME_CHECKPOINT must be supplied together" >&2
        exit 2
    fi
fi
if [[ -n "${RESUME_CHECKPOINT}" ]]; then
    runpod_validate_path_in_root \
        "${RESUME_CHECKPOINT}" "${SAVED_MODEL_ROOT}" \
        RESUME_CHECKPOINT SAVED_MODEL_ROOT
    if [[ ! "${RESUME_CHECKPOINT}" =~ ^/[A-Za-z0-9._/-]+$ ]]; then
        echo "RESUME_CHECKPOINT contains unsupported characters" >&2
        exit 2
    fi
    if [[ "$(dirname "${RESUME_CHECKPOINT}")" != "${SAVED_MODEL_ROOT}/${WANDB_RUN_ID}" \
        || ! "$(basename "${RESUME_CHECKPOINT}")" =~ ^checkpoint-[0-9]{6,}$ ]]; then
        echo "RESUME_CHECKPOINT must identify a canonical checkpoint below SAVED_MODEL_ROOT/WANDB_RUN_ID" >&2
        exit 2
    fi
fi
if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 is required for readiness checks and UTC deadline calculation" >&2
    exit 127
fi

if [[ "${RUNPOD_TEST_READINESS_READY:-0}" == "1" ]]; then
    if [[ "${RUNPOD_TEST_MODE:-0}" != "1" ]]; then
        echo "RUNPOD_TEST_READINESS_READY is allowed only in test mode" >&2
        exit 2
    fi
else
    # This gate runs before the paid GPU Pod creation request.
    bash "${SCRIPT_DIR}/verify_runpod_stage_readiness.sh" --gpu
fi

verify_remote_checkpoint_run() {
    local run_id="$1"
    local selection_policy="$2"
    local checkpoint_name="${3:-}"
    local command=(
        python3 "${REMOTE_CHECKPOINT_PREFLIGHT}"
        --s3-wrapper "${S3_WRAPPER}"
        --bucket "${RUNPOD_NETWORK_VOLUME_ID}"
        --run-id "${run_id}"
        --config "${LOCAL_CONFIG_PATH}"
        --selection-policy "${selection_policy}"
    )
    if [[ -n "${checkpoint_name}" ]]; then
        command+=(--checkpoint-name "${checkpoint_name}")
    fi
    "${command[@]}"
}

verify_no_active_gpu_workflow() {
    local lifecycle_keys listed_key lifecycle_key lifecycle_kind lifecycle_json
    lifecycle_keys="$(bash "${S3_WRAPPER}" s3api list-objects-v2 \
        --bucket "${RUNPOD_NETWORK_VOLUME_ID}" \
        --prefix lifecycle/stage1/ \
        --query 'Contents[].Key' \
        --output text)"
    for lifecycle_key in \
        lifecycle/stage1/training.json \
        lifecycle/stage1/validation.json; do
        lifecycle_kind=stage1-training
        if [[ "${lifecycle_key}" == "lifecycle/stage1/validation.json" ]]; then
            lifecycle_kind=stage1-validation
        fi
        for listed_key in ${lifecycle_keys}; do
            if [[ "${listed_key}" != "${lifecycle_key}" ]]; then
                continue
            fi
            lifecycle_json="$(bash "${S3_WRAPPER}" s3 cp \
                "s3://${RUNPOD_NETWORK_VOLUME_ID}/${lifecycle_key}" - \
                --only-show-errors)"
            printf '%s\n' "${lifecycle_json}" \
                | python3 "${READINESS_HELPER}" gpu-workflow-available \
                    --marker - \
                    --network-volume-root "${RUNPOD_VOLUME_MOUNT_PATH}" \
                    --kind "${lifecycle_kind}" >/dev/null
            break
        done
    done
}

verify_completed_training_run() {
    local run_id="$1" completion_json run_manifest_json run_manifest_sha256
    completion_json="$(bash "${S3_WRAPPER}" s3 cp \
        "s3://${RUNPOD_NETWORK_VOLUME_ID}/lifecycle/runs/${run_id}/training-completed.json" - \
        --only-show-errors)"
    run_manifest_json="$(bash "${S3_WRAPPER}" s3 cp \
        "s3://${RUNPOD_NETWORK_VOLUME_ID}/savedModel/${run_id}/run-manifest.json" - \
        --only-show-errors)"
    run_manifest_sha256="$(printf '%s' "${run_manifest_json}" \
        | python3 -c 'import hashlib, sys; print(hashlib.sha256(sys.stdin.buffer.read()).hexdigest())')"
    printf '%s\n' "${completion_json}" \
        | python3 "${READINESS_HELPER}" check-training-completion \
            --marker - \
            --network-volume-root "${RUNPOD_VOLUME_MOUNT_PATH}" \
            --run-id "${run_id}" \
            --run-manifest-sha256 "${run_manifest_sha256}" >/dev/null
}

if [[ "${RUNPOD_TEST_MODE:-0}" != "1" ]]; then
    if [[ ! -r "${S3_WRAPPER}" || ! -r "${READINESS_HELPER}" \
        || ! -r "${REMOTE_CHECKPOINT_PREFLIGHT}" ]]; then
        echo "RunPod checkpoint preflight helpers are unavailable" >&2
        exit 127
    fi
    verify_no_active_gpu_workflow
    if [[ "${RUNPOD_GPU_WORKFLOW}" == "validation" ]]; then
        if [[ -z "${VALIDATION_RUN_ID}" ]]; then
            TRAINING_LIFECYCLE_JSON="$(bash "${S3_WRAPPER}" s3 cp \
                "s3://${RUNPOD_NETWORK_VOLUME_ID}/lifecycle/stage1/training.json" - \
                --only-show-errors)"
            VALIDATION_RUN_ID="$(printf '%s\n' "${TRAINING_LIFECYCLE_JSON}" \
                | python3 "${READINESS_HELPER}" completed-training-run \
                    --marker - \
                    --network-volume-root "${RUNPOD_VOLUME_MOUNT_PATH}")"
        fi
        verify_completed_training_run "${VALIDATION_RUN_ID}"
        if [[ -z "${VALIDATION_CHECKPOINT}" ]]; then
            VALIDATION_CHECKPOINT_NAME="$(verify_remote_checkpoint_run \
                "${VALIDATION_RUN_ID}" best)"
            VALIDATION_CHECKPOINT="${SAVED_MODEL_ROOT}/${VALIDATION_RUN_ID}/${VALIDATION_CHECKPOINT_NAME}"
            runpod_validate_path_in_root \
                "${VALIDATION_CHECKPOINT}" "${SAVED_MODEL_ROOT}" \
                VALIDATION_CHECKPOINT SAVED_MODEL_ROOT
        else
            VALIDATION_CHECKPOINT_NAME="$(basename "${VALIDATION_CHECKPOINT}")"
            verify_remote_checkpoint_run \
                "${VALIDATION_RUN_ID}" retained \
                "${VALIDATION_CHECKPOINT_NAME}" >/dev/null
        fi
        export VALIDATION_RUN_ID VALIDATION_CHECKPOINT
    elif [[ -n "${RESUME_CHECKPOINT}" ]]; then
        RESUME_CHECKPOINT_NAME="$(basename "${RESUME_CHECKPOINT}")"
        verify_remote_checkpoint_run \
            "${WANDB_RUN_ID}" latest "${RESUME_CHECKPOINT_NAME}" >/dev/null
    fi
fi

TERMINATE_AFTER_DEADLINE="$(python3 "${SCRIPT_DIR}/utc_deadline.py" \
    "${RUNPOD_TERMINATE_AFTER}")"
WANDB_API_KEY_REFERENCE="{{ RUNPOD_SECRET_${RUNPOD_WANDB_SECRET_NAME} }}"

POD_ENV_JSON="$(printf \
    '{"NETWORK_VOLUME_ROOT":"%s","RUNPOD_VOLUME_ROOT":"%s","RUNPOD_EXPECTED_VOLUME_ID":"%s","PROJECT_ROOT":"%s","DATA_ROOT":"%s","SAVED_MODEL_ROOT":"%s","WANDB_DIR":"%s","RUNPOD_CONFIG":"%s","RUNPOD_STAGE":"%s","RUNPOD_STAGE_CONFIG_SHA256":"%s","RUNPOD_SELECTION_ID":"%s","RUNPOD_SELECTION_SHA256":"%s","RUNPOD_DATASET_REQUEST_SHA256":"%s","RUNPOD_REMOTE_SELECTION_PATH":"%s","FIN_TS_DATASET_PROFILE":"%s","STAGE1_US_SYMBOLS":"%s","STAGE1_US_ETF_SYMBOLS":"%s","STAGE1_SYMBOL_LIMIT":"%s","STAGE1_DATA_START":"%s","STAGE1_DATA_END":"%s","RUNPOD_ROLE":"%s","MAX_RUNTIME_SECONDS":"%s","RUNPOD_SHUTDOWN_ACTION":"terminate","RUNPOD_IMAGE":"%s","RUNPOD_EXPECTED_TORCH_VERSION":"%s","RUNPOD_EXPECTED_CUDA_PREFIX":"%s","RUNPOD_EXPECTED_UBUNTU_VERSION":"%s","WANDB_API_KEY":"%s","WANDB_PROJECT":"%s","WANDB_ENTITY":"%s","WANDB_RUN_ID":"%s","RESUME_CHECKPOINT":"%s","VALIDATION_RECOMPUTE_FULL_MODEL":"%s","VALIDATION_RESUME":"%s","VALIDATION_FORCE_RECOMPUTE":"%s","VALIDATION_RUN_ID":"%s","VALIDATION_CHECKPOINT":"%s"}' \
    "${RUNPOD_VOLUME_MOUNT_PATH}" "${RUNPOD_VOLUME_MOUNT_PATH}" \
    "${RUNPOD_NETWORK_VOLUME_ID}" "${PROJECT_ROOT}" "${DATA_ROOT}" \
    "${SAVED_MODEL_ROOT}" "${RUNPOD_VOLUME_MOUNT_PATH}" \
    "${RUNPOD_CONFIG}" "${RUNPOD_STAGE}" "${RUNPOD_STAGE_CONFIG_SHA256}" \
    "${RUNPOD_SELECTION_ID}" "${RUNPOD_SELECTION_SHA256}" \
    "${RUNPOD_DATASET_REQUEST_SHA256}" "${REMOTE_SELECTION_MOUNT_PATH}" \
    "${FIN_TS_DATASET_PROFILE}" "${STAGE1_US_SYMBOLS}" \
    "${STAGE1_US_ETF_SYMBOLS}" "${STAGE1_SYMBOL_LIMIT}" \
    "${STAGE1_DATA_START}" "${STAGE1_DATA_END}" "${RUNPOD_GPU_ROLE}" \
    "${MAX_RUNTIME_SECONDS}" \
    "${RUNPOD_IMAGE}" "${RUNPOD_EXPECTED_TORCH_VERSION}" \
    "${RUNPOD_EXPECTED_CUDA_PREFIX}" "${RUNPOD_EXPECTED_UBUNTU_VERSION}" \
    "${WANDB_API_KEY_REFERENCE}" \
    "${WANDB_PROJECT}" "${WANDB_ENTITY}" "${WANDB_RUN_ID}" "${RESUME_CHECKPOINT}" \
    "${VALIDATION_RECOMPUTE_FULL_MODEL}" "${VALIDATION_RESUME}" \
    "${VALIDATION_FORCE_RECOMPUTE}" "${VALIDATION_RUN_ID}" \
    "${VALIDATION_CHECKPOINT}")"

CREATE_COMMAND=(
    bash "${SCRIPT_DIR}/runpodctl_project.sh" pod create
    --name "${RUNPOD_POD_NAME}"
    --gpu-id "${RUNPOD_GPU_ID}"
    --gpu-count "${RUNPOD_GPU_COUNT}"
    --image "${RUNPOD_IMAGE}"
    --min-cuda-version "${RUNPOD_MIN_CUDA_VERSION}"
    --cloud-type "${RUNPOD_CLOUD_TYPE}"
    --data-center-ids "${RUNPOD_DATACENTER_ID}"
    --container-disk-in-gb "${RUNPOD_CONTAINER_DISK_GB}"
    --network-volume-id "${RUNPOD_NETWORK_VOLUME_ID}"
    --volume-mount-path "${RUNPOD_VOLUME_MOUNT_PATH}"
    --terminate-after "${TERMINATE_AFTER_DEADLINE}"
    --env "${POD_ENV_JSON}"
)

if [[ "${RUNPOD_CREATE_DRY_RUN:-0}" == "1" || "${RUNPOD_TEST_MODE:-0}" == "1" ]]; then
    printf 'DRY RUN:'
    printf ' %q' "${CREATE_COMMAND[@]}"
    printf '\n'
    exit 0
fi
if ! command -v runpodctl >/dev/null 2>&1; then
    echo "runpodctl is required on the machine creating the Pod" >&2
    exit 127
fi
if [[ ! -f "${SCRIPT_DIR}/runpodctl_project.sh" \
    || ! -r "${SCRIPT_DIR}/runpodctl_project.sh" ]]; then
    echo "Project runpodctl wrapper is unavailable" >&2
    exit 127
fi
if [[ ! -r "${RUNPOD_GUARD_LAUNCHER}" ]]; then
    echo "External guard launcher is unavailable: ${RUNPOD_GUARD_LAUNCHER}" >&2
    exit 127
fi

CREATE_OUTPUT="$("${CREATE_COMMAND[@]}")"
POD_ID="$(printf '%s' "${CREATE_OUTPUT}" | python3 -c \
    'import json, sys; payload=json.load(sys.stdin); print(payload.get("id", ""))')"
if [[ ! "${POD_ID}" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo "Unable to parse a safe Pod ID from runpodctl output; no guard was armed" >&2
    exit 3
fi

mkdir -p "${RUNPOD_GUARD_LOG_DIR}"
GUARD_LOG="${RUNPOD_GUARD_LOG_DIR}/${POD_ID}.log"

GUARD_PID="$(RUNPOD_GUARD_VOLUME_ROOT="${RUNPOD_VOLUME_MOUNT_PATH}" \
    RUNPOD_GUARD_RUN_ID="${VALIDATION_RUN_ID:-${WANDB_RUN_ID}}" \
    bash "${RUNPOD_GUARD_LAUNCHER}" \
    "${POD_ID}" "${RUNPOD_HARD_LIMIT_SECONDS}" \
    "${RUNPOD_GUARD_LIFECYCLE_KEY}" "${GUARD_LOG}")"

printf 'Created Pod: %s\n' "${POD_ID}"
printf 'External hard-limit guard PID: %s\n' "${GUARD_PID}"
printf 'Guard log: %s\n' "${GUARD_LOG}"
printf 'Guard readiness: %s\n' "${GUARD_LOG%.log}.ready.json"
printf 'After SSH login, run: bash scripts/runpod_tmux_launch.sh %s\n' \
    "${RUNPOD_TMUX_WORKFLOW}"
