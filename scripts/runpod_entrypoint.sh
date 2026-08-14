#!/usr/bin/env bash

# Launch this script from a RunPod Docker start command and pass the training command after --.
set -Eeuo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/runpod_paths.sh
source "${SCRIPT_DIR}/lib/runpod_paths.sh"

NETWORK_VOLUME_ROOT="${NETWORK_VOLUME_ROOT:-${RUNPOD_VOLUME_ROOT:-/runpod-volume}}"
PROJECT_ROOT="${PROJECT_ROOT:-${NETWORK_VOLUME_ROOT}/stock_forecasting}"
DATA_ROOT="${DATA_ROOT:-${NETWORK_VOLUME_ROOT}/data}"
CACHE_ROOT="${CACHE_ROOT:-${NETWORK_VOLUME_ROOT}/cache}"
LOG_ROOT="${LOG_ROOT:-${NETWORK_VOLUME_ROOT}/logs}"
SAVED_MODEL_ROOT="${SAVED_MODEL_ROOT:-${NETWORK_VOLUME_ROOT}/savedModel}"
WANDB_DIR="${WANDB_DIR:-${NETWORK_VOLUME_ROOT}}"
RUNPOD_SHUTDOWN_ACTION="${RUNPOD_SHUTDOWN_ACTION:-terminate}"
MAX_RUNTIME_SECONDS="${MAX_RUNTIME_SECONDS:-21600}"
POETRY_VERSION="${POETRY_VERSION:-2.4.0}"
RUNPOD_CONFIG="${RUNPOD_CONFIG:-configs/stage1_kronos_base_lora.yaml}"
PROJECT_VENV="${PROJECT_ROOT}/.venv"
RUNPOD_PYTHON_BIN="${RUNPOD_PYTHON_BIN:-/usr/local/bin/python}"
export RUNPOD_IMAGE="${RUNPOD_IMAGE:-runpod/pytorch:1.0.7-cu1281-torch291-ubuntu2404}"
RUNPOD_EXPECTED_TORCH_VERSION="${RUNPOD_EXPECTED_TORCH_VERSION:-2.9.1+cu128}"
RUNPOD_EXPECTED_CUDA_PREFIX="${RUNPOD_EXPECTED_CUDA_PREFIX:-12.8}"
RUNPOD_EXPECTED_UBUNTU_VERSION="${RUNPOD_EXPECTED_UBUNTU_VERSION:-24.04}"
RUNPOD_ROLE="${RUNPOD_ROLE:-gpu-train}"
RUNTIME_VERIFIER="${PROJECT_ROOT}/scripts/verify_runpod_runtime.py"
READINESS_HELPER="${PROJECT_ROOT}/scripts/runpod_readiness.py"

if [[ -z "${RUNPOD_POD_ID:-}" \
    && "${RUNPOD_DRY_RUN:-0}" != "1" \
    && "${RUNPOD_TEST_MODE:-0}" != "1" ]]; then
    echo "Training entrypoint must run inside a RunPod Pod" >&2
    exit 2
fi
if [[ "${RUNPOD_ROLE}" != "gpu-train" ]]; then
    echo "Training entrypoint requires RUNPOD_ROLE=gpu-train" >&2
    exit 2
fi

runpod_validate_absolute_path "${NETWORK_VOLUME_ROOT}" NETWORK_VOLUME_ROOT
for path_name in PROJECT_ROOT DATA_ROOT CACHE_ROOT LOG_ROOT SAVED_MODEL_ROOT WANDB_DIR; do
    path_value="${!path_name}"
    case "${path_value}" in
        /workspace|/workspace/*)
            echo "${path_name} must never use ephemeral /workspace" >&2
            exit 2
            ;;
    esac
    runpod_validate_path_in_root \
        "${path_value}" "${NETWORK_VOLUME_ROOT}" "${path_name}" NETWORK_VOLUME_ROOT
done

if [[ "${NETWORK_VOLUME_ROOT}" == "/workspace" || "${NETWORK_VOLUME_ROOT}" == /workspace/* ]]; then
    echo "NETWORK_VOLUME_ROOT must never use ephemeral /workspace" >&2
    exit 2
fi
if [[ "${LOG_ROOT}" != "${NETWORK_VOLUME_ROOT}/logs" ]]; then
    echo "LOG_ROOT must equal NETWORK_VOLUME_ROOT/logs" >&2
    exit 2
fi
if [[ "${SAVED_MODEL_ROOT}" != "${NETWORK_VOLUME_ROOT}/savedModel" ]]; then
    echo "SAVED_MODEL_ROOT must equal NETWORK_VOLUME_ROOT/savedModel" >&2
    exit 2
fi
if [[ "${WANDB_DIR}" != "${NETWORK_VOLUME_ROOT}" ]]; then
    echo "WANDB_DIR is the W&B SDK root and must equal NETWORK_VOLUME_ROOT" >&2
    exit 2
fi

# A single shared lease protects both singleton training and validation lifecycle files.
runpod_acquire_gpu_workflow_lease "${NETWORK_VOLUME_ROOT}"

export NETWORK_VOLUME_ROOT PROJECT_ROOT DATA_ROOT CACHE_ROOT LOG_ROOT SAVED_MODEL_ROOT WANDB_DIR
export RUNPOD_VOLUME_ROOT="${NETWORK_VOLUME_ROOT}"
export RUNPOD_SHUTDOWN_ACTION MAX_RUNTIME_SECONDS
# Ignore image and SSH cache overrides; all runtime caches are persistent and derived from the volume.
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
export HF_HOME="${CACHE_ROOT}/huggingface"
export TORCH_HOME="${CACHE_ROOT}/torch"
export POETRY_CACHE_DIR="${CACHE_ROOT}/pypoetry"
export TMPDIR="${NETWORK_VOLUME_ROOT}/tmp"
export WANDB_CACHE_DIR="${CACHE_ROOT}/wandb"
export WANDB_ARTIFACT_DIR="${NETWORK_VOLUME_ROOT}/wandb-artifacts"

for path_name in XDG_CACHE_HOME HF_HOME TORCH_HOME POETRY_CACHE_DIR TMPDIR WANDB_CACHE_DIR WANDB_ARTIFACT_DIR; do
    path_value="${!path_name}"
    runpod_validate_path_in_root \
        "${path_value}" "${NETWORK_VOLUME_ROOT}" "${path_name}" NETWORK_VOLUME_ROOT
done

mkdir -p \
    "${PROJECT_ROOT}" \
    "${DATA_ROOT}" \
    "${CACHE_ROOT}" \
    "${LOG_ROOT}" \
    "${SAVED_MODEL_ROOT}" \
    "${WANDB_DIR}/wandb" \
    "${XDG_CACHE_HOME}" \
    "${HF_HOME}" \
    "${TORCH_HOME}" \
    "${POETRY_CACHE_DIR}" \
    "${TMPDIR}" \
    "${WANDB_CACHE_DIR}" \
    "${WANDB_ARTIFACT_DIR}"

if [[ -z "${WANDB_RUN_ID:-}" ]]; then
    if [[ -n "${RUNPOD_RUN_KEY:-}" ]]; then
        echo "RUNPOD_RUN_KEY cannot be set when WANDB_RUN_ID is empty" >&2
    else
        echo "WANDB_RUN_ID must be preallocated by create_runpod_pod.sh" >&2
    fi
    exit 2
elif [[ -n "${RUNPOD_RUN_KEY:-}" && "${RUNPOD_RUN_KEY}" != "${WANDB_RUN_ID}" ]]; then
    echo "RUNPOD_RUN_KEY and WANDB_RUN_ID must identify the same run" >&2
    exit 2
fi
if [[ ! "${WANDB_RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,119}$ \
    || "${WANDB_RUN_ID}" == *--* ]]; then
    echo "WANDB_RUN_ID must be a safe 1-120 character directory name" >&2
    exit 2
fi
RUNPOD_RUN_KEY="${WANDB_RUN_ID}"
export WANDB_RUN_ID RUNPOD_RUN_KEY
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
    runpod_validate_path_in_root \
        "${RESUME_CHECKPOINT}" "${SAVED_MODEL_ROOT}" \
        RESUME_CHECKPOINT SAVED_MODEL_ROOT
    if [[ "$(dirname "${RESUME_CHECKPOINT}")" != "${SAVED_MODEL_ROOT}/${WANDB_RUN_ID}" \
        || ! "$(basename "${RESUME_CHECKPOINT}")" =~ ^checkpoint-[0-9]{6,}$ ]]; then
        echo "RESUME_CHECKPOINT must identify a canonical checkpoint below SAVED_MODEL_ROOT/WANDB_RUN_ID" >&2
        exit 2
    fi
fi
RUNPOD_LAUNCH_ID="${RUNPOD_LAUNCH_ID:-launch-$(date -u +%Y%m%dT%H%M%SZ)-${RANDOM}${RANDOM}}"
RUNPOD_LAUNCH_ID="$(printf '%s' "${RUNPOD_LAUNCH_ID}" | tr -c 'A-Za-z0-9._-' '-')"
export RUNPOD_LAUNCH_ID
export RUNPOD_SHUTDOWN_DIR="${LOG_ROOT}/${RUNPOD_RUN_KEY}/launcher/${RUNPOD_LAUNCH_ID}"
export RUNPOD_SHUTDOWN_MARKER="${RUNPOD_SHUTDOWN_DIR}/shutdown.json"
export RUNPOD_TRAINING_LIFECYCLE_MARKER="${NETWORK_VOLUME_ROOT}/lifecycle/stage1/training.json"
export RUNPOD_VALIDATION_LIFECYCLE_MARKER="${NETWORK_VOLUME_ROOT}/lifecycle/stage1/validation.json"
export RUNPOD_TRAINING_COMPLETED_MARKER="${RUNPOD_SHUTDOWN_DIR}/training-completed.json"

for path_name in \
    RUNPOD_SHUTDOWN_DIR RUNPOD_SHUTDOWN_MARKER \
    RUNPOD_TRAINING_LIFECYCLE_MARKER RUNPOD_VALIDATION_LIFECYCLE_MARKER \
    RUNPOD_TRAINING_COMPLETED_MARKER; do
    runpod_validate_path_in_root \
        "${!path_name}" "${NETWORK_VOLUME_ROOT}" "${path_name}" NETWORK_VOLUME_ROOT
done

mkdir -p "${SAVED_MODEL_ROOT}" "${RUNPOD_SHUTDOWN_DIR}"
if [[ -n "${RUNPOD_TMUX_LOG_FILE:-}" ]]; then
    # The tmux runner already relies on Bash process substitution. Mirror launcher
    # output to the pane while retaining separate durable stdout/stderr files.
    exec > >(tee -a "${RUNPOD_SHUTDOWN_DIR}/launcher.stdout.log") \
        2> >(tee -a "${RUNPOD_SHUTDOWN_DIR}/launcher.stderr.log" >&2)
else
    # Keep a file-only fallback for non-tmux container launches.
    exec >> "${RUNPOD_SHUTDOWN_DIR}/launcher.stdout.log" \
        2>> "${RUNPOD_SHUTDOWN_DIR}/launcher.stderr.log"
fi

CHILD_PID=""
LAUNCH_STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
LAUNCH_STARTED_EPOCH="$(date +%s)"

write_training_lifecycle() {
    local lifecycle_state="$1"
    local exit_code="${2:-}"
    local recovery_path="${3:-}"
    local command=(
        "${RUNPOD_PYTHON_BIN}" "${READINESS_HELPER}" write-state
        --output "${RUNPOD_TRAINING_LIFECYCLE_MARKER}"
        --network-volume-root "${NETWORK_VOLUME_ROOT}"
        --kind stage1-training
        --state "${lifecycle_state}"
        --launch-id "${RUNPOD_LAUNCH_ID}"
        --log-path "${RUNPOD_SHUTDOWN_DIR}"
        --wandb-run-id "${WANDB_RUN_ID}"
        --max-runtime-seconds "${MAX_RUNTIME_SECONDS}"
    )
    if [[ -n "${exit_code}" ]]; then
        command+=(--exit-code "${exit_code}")
    fi
    if [[ -n "${recovery_path}" ]]; then
        command+=(--recovery-path "${recovery_path}")
    fi
    if [[ -f "${RUNPOD_TRAINING_COMPLETED_MARKER}" ]]; then
        command+=(--training-completed 1)
    fi
    if [[ ! -x "${RUNPOD_PYTHON_BIN}" || ! -r "${READINESS_HELPER}" ]]; then
        echo "Unable to publish the training lifecycle marker" >&2
        return 1
    fi
    "${command[@]}"
}

latest_recovery_path() {
    find "${LOG_ROOT}/${RUNPOD_RUN_KEY}" -mindepth 2 -maxdepth 2 \
        -type f -name recovery.json -print 2>/dev/null | LC_ALL=C sort | tail -n 1
}

mark_validation_timed_out_if_running() {
    if [[ ! -f "${RUNPOD_VALIDATION_LIFECYCLE_MARKER}" ]]; then
        return 0
    fi
    if ! "${RUNPOD_PYTHON_BIN}" -c \
        'import json, os, sys
with open(sys.argv[1], encoding="utf-8") as stream:
    payload = json.load(stream)
active = payload.get("state") in ("preparing", "finalizing")
same_pod = not os.environ.get("RUNPOD_POD_ID") or payload.get("pod_id") == os.environ.get("RUNPOD_POD_ID")
raise SystemExit(0 if active and same_pod else 1)' \
        "${RUNPOD_VALIDATION_LIFECYCLE_MARKER}"; then
        return 0
    fi
    "${RUNPOD_PYTHON_BIN}" "${READINESS_HELPER}" write-state \
        --output "${RUNPOD_VALIDATION_LIFECYCLE_MARKER}" \
        --network-volume-root "${NETWORK_VOLUME_ROOT}" \
        --kind stage1-validation \
        --state timed_out \
        --launch-id "${RUNPOD_LAUNCH_ID}" \
        --exit-code 124 \
        --log-path "${RUNPOD_SHUTDOWN_DIR}" \
        --wandb-run-id "${WANDB_RUN_ID}" \
        --max-runtime-seconds "${MAX_RUNTIME_SECONDS}" \
        --inherit-existing || true
}

forward_signal() {
    local signal_name="$1"
    if [[ -n "${CHILD_PID}" ]] && kill -0 "${CHILD_PID}" 2>/dev/null; then
        kill "-${signal_name}" "${CHILD_PID}" 2>/dev/null || kill "-${signal_name}" -- "-${CHILD_PID}" 2>/dev/null || true
    fi
}

shutdown_fallback() {
    local training_exit_code=$?
    trap - EXIT INT TERM
    local elapsed_seconds=$(( $(date +%s) - LAUNCH_STARTED_EPOCH ))
    local lifecycle_state="failed"
    local recovery_path=""
    if [[ ${training_exit_code} -eq 0 ]]; then
        lifecycle_state="ready"
    elif [[ ${training_exit_code} -eq 124 \
        || ( ${elapsed_seconds} -ge ${MAX_RUNTIME_SECONDS} \
            && ( ${training_exit_code} -eq 137 || ${training_exit_code} -eq 143 ) ) ]]; then
        lifecycle_state="timed_out"
        training_exit_code=124
    fi
    if [[ "${lifecycle_state}" == "timed_out" ]]; then
        mark_validation_timed_out_if_running
    fi
    printf '{"started_at":"%s","ended_at":"%s","elapsed_seconds":%d,"exit_code":%d,"lifecycle_state":"%s","project_root":"%s","run_key":"%s"}\n' \
        "${LAUNCH_STARTED_AT}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${elapsed_seconds}" \
        "${training_exit_code}" "${lifecycle_state}" "${PROJECT_ROOT}" "${RUNPOD_RUN_KEY}" \
        > "${RUNPOD_SHUTDOWN_DIR}/launcher-metadata.json"
    recovery_path="$(latest_recovery_path)"
    local published_lifecycle_state="${lifecycle_state}"
    local lifecycle_published=0
    if [[ -n "${RUNPOD_TMUX_LOG_FILE:-}" && -n "${RUNPOD_LAUNCH_ID:-}" ]]; then
        published_lifecycle_state="finalizing"
    fi
    if write_training_lifecycle \
        "${published_lifecycle_state}" "${training_exit_code}" "${recovery_path}"; then
        lifecycle_published=1
    else
        echo "Unable to publish ${published_lifecycle_state} training lifecycle" >&2
        if [[ -z "${RUNPOD_TMUX_LOG_FILE:-}" && ${training_exit_code} -eq 0 ]]; then
            training_exit_code=74
        fi
    fi
    if [[ -z "${RUNPOD_TMUX_LOG_FILE:-}" && ${lifecycle_published} -eq 1 ]]; then
        bash "${SCRIPT_DIR}/stop_runpod_pod.sh" || true
    elif [[ -z "${RUNPOD_TMUX_LOG_FILE:-}" ]]; then
        echo "Pod shutdown skipped because terminal lifecycle publication failed" >&2
    fi
    exit "${training_exit_code}"
}

trap 'forward_signal TERM' TERM
trap 'forward_signal INT' INT
trap shutdown_fallback EXIT

write_training_lifecycle preparing

if [[ -z "${WANDB_API_KEY:-}" || "${WANDB_API_KEY}" == *'{{ RUNPOD_SECRET_'* ]]; then
    echo "WANDB_API_KEY RunPod Secret is missing or was not resolved" >&2
    exit 2
fi

# This second gate protects against races, manual Pod creation, and changed volume contents.
if [[ "${RUNPOD_TEST_READINESS_READY:-0}" == "1" \
    && "${RUNPOD_TEST_MODE:-0}" == "1" ]]; then
    :
else
    bash "${SCRIPT_DIR}/verify_runpod_mounted_readiness.sh"
fi

if [[ ! -f "${PROJECT_ROOT}/pyproject.toml" ]]; then
    echo "pyproject.toml not found under PROJECT_ROOT: ${PROJECT_ROOT}" >&2
    exit 2
fi
POETRY_BIN="${RUNPOD_POETRY_BIN:-${NETWORK_VOLUME_ROOT}/tools/poetry/${POETRY_VERSION}/bin/poetry}"
runpod_validate_path_in_root \
    "${POETRY_BIN}" "${NETWORK_VOLUME_ROOT}" POETRY_BIN NETWORK_VOLUME_ROOT
if [[ ! -x "${POETRY_BIN}" ]]; then
    echo "Persistent Poetry is unavailable; run scripts/setup_runpod_environment.sh first" >&2
    exit 127
fi
if [[ "${RUNPOD_PYTHON_BIN}" != /* || ! -x "${RUNPOD_PYTHON_BIN}" ]]; then
    echo "Python executable not found: ${RUNPOD_PYTHON_BIN}" >&2
    exit 127
fi
case "${RUNPOD_PYTHON_BIN}" in
    "${NETWORK_VOLUME_ROOT}"/*|/workspace/*)
        echo "RUNPOD_PYTHON_BIN must come from the RunPod image" >&2
        exit 2
        ;;
esac
if ! "${RUNPOD_PYTHON_BIN}" -c \
    'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)'; then
    echo "The approved RunPod image environment requires Python 3.12" >&2
    exit 2
fi
if [[ "${1:-}" == "--" ]]; then
    shift
fi
USE_DEFAULT_COMMAND=0
if [[ $# -eq 0 ]]; then
    USE_DEFAULT_COMMAND=1
    set -- bash "${SCRIPT_DIR}/runpod_train_then_validate.sh" "${RUNPOD_CONFIG}"
fi

cd "${PROJECT_ROOT}"
if [[ "${RUNPOD_CONFIG}" == /workspace || "${RUNPOD_CONFIG}" == /workspace/* ]]; then
    echo "RUNPOD_CONFIG must never use /workspace" >&2
    exit 2
fi
if [[ ${USE_DEFAULT_COMMAND} -eq 1 ]]; then
    if [[ "${RUNPOD_CONFIG}" == /* ]]; then
        case "${RUNPOD_CONFIG}" in
            "${NETWORK_VOLUME_ROOT}"/*) ;;
            *)
                echo "Absolute RUNPOD_CONFIG must be on NETWORK_VOLUME_ROOT" >&2
                exit 2
                ;;
        esac
        CONFIG_PATH="${RUNPOD_CONFIG}"
    else
        if [[ "/${RUNPOD_CONFIG}/" == *"/../"* ]]; then
            echo "Relative RUNPOD_CONFIG must not traverse outside PROJECT_ROOT" >&2
            exit 2
        fi
        CONFIG_PATH="${PROJECT_ROOT}/${RUNPOD_CONFIG}"
    fi
    runpod_validate_path_in_root \
        "${CONFIG_PATH}" "${NETWORK_VOLUME_ROOT}" CONFIG_PATH NETWORK_VOLUME_ROOT
    if [[ ! -f "${CONFIG_PATH}" ]]; then
        echo "RUNPOD_CONFIG not found: ${CONFIG_PATH}" >&2
        exit 2
    fi
fi
if [[ ! -x "${PROJECT_VENV}/bin/python" ]]; then
    echo "Persistent project .venv is unavailable; run scripts/setup_runpod_environment.sh first" >&2
    exit 127
fi
if ! grep -Eq '^include-system-site-packages = true$' "${PROJECT_VENV}/pyvenv.cfg"; then
    echo "Persistent project .venv does not inherit RunPod image packages" >&2
    exit 3
fi
if [[ "${RUNPOD_TEST_MODE:-0}" != "1" && "${RUNPOD_DRY_RUN:-0}" != "1" ]]; then
    if [[ ! -f "${RUNTIME_VERIFIER}" ]]; then
        echo "RunPod runtime verifier not found: ${RUNTIME_VERIFIER}" >&2
        exit 2
    fi
    IMAGE_RUNTIME_METADATA="${RUNPOD_SHUTDOWN_DIR}/image-runtime.json"
    "${RUNPOD_PYTHON_BIN}" "${RUNTIME_VERIFIER}" \
        --role image \
        --expected-torch "${RUNPOD_EXPECTED_TORCH_VERSION}" \
        --expected-cuda-prefix "${RUNPOD_EXPECTED_CUDA_PREFIX}" \
        --expected-ubuntu "${RUNPOD_EXPECTED_UBUNTU_VERSION}" \
        --network-volume-root "${NETWORK_VOLUME_ROOT}" \
        --require-cuda-device \
        --output "${IMAGE_RUNTIME_METADATA}"
    "${PROJECT_VENV}/bin/python" "${RUNTIME_VERIFIER}" \
        --role venv \
        --expected-torch "${RUNPOD_EXPECTED_TORCH_VERSION}" \
        --expected-cuda-prefix "${RUNPOD_EXPECTED_CUDA_PREFIX}" \
        --expected-ubuntu "${RUNPOD_EXPECTED_UBUNTU_VERSION}" \
        --network-volume-root "${NETWORK_VOLUME_ROOT}" \
        --require-cuda-device \
        --reference "${IMAGE_RUNTIME_METADATA}" \
        --output "${RUNPOD_SHUTDOWN_DIR}/venv-runtime.json"
fi
"${POETRY_BIN}" env use "${PROJECT_VENV}/bin/python"
if ! "${PROJECT_VENV}/bin/python" -c \
    'import numpy as np; print(f"Verified project NumPy: {np.__version__}")'; then
    echo "Persistent Poetry environment is missing NumPy; rerun setup_runpod_environment.sh" >&2
    exit 3
fi

# Invoke the installed modules with the persistent Poetry-managed interpreter.
# This avoids an extra Poetry/Cleo option-parsing layer around supervisor flags.
"${PROJECT_VENV}/bin/python" -m stock_forecasting.runpod.supervisor \
    --max-runtime-seconds "${MAX_RUNTIME_SECONDS}" \
    --no-auto-shutdown \
    -- "$@" &
CHILD_PID=$!
set +e
wait "${CHILD_PID}"
TRAINING_EXIT_CODE=$?
set -e
CHILD_PID=""
exit "${TRAINING_EXIT_CODE}"
