#!/usr/bin/env bash

# Run or resume the complete validation benchmark inside a CUDA-enabled RunPod Pod.
set -Eeuo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/runpod_paths.sh
source "${SCRIPT_DIR}/lib/runpod_paths.sh"

NETWORK_VOLUME_ROOT="${NETWORK_VOLUME_ROOT:-${RUNPOD_VOLUME_ROOT:-/runpod-volume}}"
PROJECT_ROOT="${PROJECT_ROOT:-${NETWORK_VOLUME_ROOT}/stock_forecasting}"
CACHE_ROOT="${CACHE_ROOT:-${NETWORK_VOLUME_ROOT}/cache}"
SAVED_MODEL_ROOT="${SAVED_MODEL_ROOT:-${NETWORK_VOLUME_ROOT}/savedModel}"
WANDB_DIR="${WANDB_DIR:-${NETWORK_VOLUME_ROOT}}"
RUNPOD_CONFIG="${RUNPOD_CONFIG:-configs/stage1_kronos_base_lora.yaml}"
PROJECT_VENV="${PROJECT_ROOT}/.venv"

if [[ -z "${RUNPOD_POD_ID:-}" && "${RUNPOD_TEST_MODE:-0}" != "1" ]]; then
    echo "Validation must run inside a RunPod GPU Pod" >&2
    exit 2
fi
runpod_validate_absolute_path "${NETWORK_VOLUME_ROOT}" NETWORK_VOLUME_ROOT
if [[ "${RUNPOD_ROLE:-gpu-validation}" != "gpu-validation" \
    && "${RUNPOD_ROLE:-}" != "gpu-train" ]]; then
    echo "Validation requires RUNPOD_ROLE=gpu-validation or gpu-train" >&2
    exit 2
fi
for path_name in PROJECT_ROOT CACHE_ROOT SAVED_MODEL_ROOT WANDB_DIR; do
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
if [[ "${SAVED_MODEL_ROOT}" != "${NETWORK_VOLUME_ROOT}/savedModel" ]]; then
    echo "SAVED_MODEL_ROOT must equal NETWORK_VOLUME_ROOT/savedModel" >&2
    exit 2
fi
if [[ "${WANDB_DIR}" != "${NETWORK_VOLUME_ROOT}" ]]; then
    echo "WANDB_DIR is the W&B SDK root and must equal NETWORK_VOLUME_ROOT" >&2
    exit 2
fi

# Auto-validation inherits the training lease; a standalone validation acquires it here.
runpod_acquire_gpu_workflow_lease "${NETWORK_VOLUME_ROOT}"
VALIDATION_TARGET_RUN_ID=""
for run_id_name in VALIDATION_RUN_ID WANDB_RUN_ID RUNPOD_RUN_KEY; do
    run_id_value="${!run_id_name:-}"
    if [[ -n "${run_id_value}" \
        && ( ! "${run_id_value}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,119}$ \
            || "${run_id_value}" == *--* ) ]]; then
        echo "${run_id_name} must be a canonical single run ID" >&2
        exit 2
    fi
    if [[ -n "${run_id_value}" ]]; then
        if [[ -n "${VALIDATION_TARGET_RUN_ID}" \
            && "${run_id_value}" != "${VALIDATION_TARGET_RUN_ID}" ]]; then
            echo "Validation runtime IDs must identify the same run" >&2
            exit 2
        fi
        VALIDATION_TARGET_RUN_ID="${run_id_value}"
    fi
done
if [[ -n "${VALIDATION_TARGET_RUN_ID}" ]]; then
    VALIDATION_RUN_ID="${VALIDATION_TARGET_RUN_ID}"
    WANDB_RUN_ID="${VALIDATION_TARGET_RUN_ID}"
    RUNPOD_RUN_KEY="${VALIDATION_TARGET_RUN_ID}"
    export VALIDATION_RUN_ID WANDB_RUN_ID RUNPOD_RUN_KEY
fi

export NETWORK_VOLUME_ROOT PROJECT_ROOT CACHE_ROOT SAVED_MODEL_ROOT WANDB_DIR
export RUNPOD_VOLUME_ROOT="${NETWORK_VOLUME_ROOT}"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
export HF_HOME="${CACHE_ROOT}/huggingface"
export TORCH_HOME="${CACHE_ROOT}/torch"
export TMPDIR="${NETWORK_VOLUME_ROOT}/tmp"
export WANDB_CACHE_DIR="${CACHE_ROOT}/wandb"
export WANDB_ARTIFACT_DIR="${NETWORK_VOLUME_ROOT}/wandb-artifacts"

for path_name in \
    XDG_CACHE_HOME HF_HOME TORCH_HOME TMPDIR WANDB_CACHE_DIR WANDB_ARTIFACT_DIR; do
    runpod_validate_path_in_root \
        "${!path_name}" "${NETWORK_VOLUME_ROOT}" "${path_name}" NETWORK_VOLUME_ROOT
done
mkdir -p \
    "${CACHE_ROOT}" \
    "${WANDB_DIR}/wandb" \
    "${XDG_CACHE_HOME}" \
    "${HF_HOME}" \
    "${TORCH_HOME}" \
    "${TMPDIR}" \
    "${WANDB_CACHE_DIR}" \
    "${WANDB_ARTIFACT_DIR}"

if [[ ! -x "${PROJECT_VENV}/bin/python" ]]; then
    echo "Persistent project .venv is unavailable" >&2
    exit 127
fi
if ! "${PROJECT_VENV}/bin/python" -c \
    'import numpy as np; print(f"Verified project NumPy: {np.__version__}")'; then
    echo "Persistent Poetry environment is missing NumPy; rerun setup_runpod_environment.sh" >&2
    exit 3
fi
if [[ "${RUNPOD_CONFIG}" == /* ]]; then
    CONFIG_PATH="${RUNPOD_CONFIG}"
else
    if [[ "/${RUNPOD_CONFIG}/" == *"/../"* ]]; then
        echo "RUNPOD_CONFIG must not traverse outside PROJECT_ROOT" >&2
        exit 2
    fi
    CONFIG_PATH="${PROJECT_ROOT}/${RUNPOD_CONFIG}"
fi
runpod_validate_path_in_root \
    "${CONFIG_PATH}" "${NETWORK_VOLUME_ROOT}" CONFIG_PATH NETWORK_VOLUME_ROOT
case "${CONFIG_PATH}" in
    /workspace|/workspace/*)
        echo "RUNPOD_CONFIG must never use ephemeral /workspace" >&2
        exit 2
        ;;
    "${NETWORK_VOLUME_ROOT}"/*) ;;
    *)
        echo "RUNPOD_CONFIG must be stored on NETWORK_VOLUME_ROOT" >&2
        exit 2
        ;;
esac
if [[ -n "${VALIDATION_CHECKPOINT:-}" ]]; then
    runpod_validate_path_in_root \
        "${VALIDATION_CHECKPOINT}" "${SAVED_MODEL_ROOT}" \
        VALIDATION_CHECKPOINT SAVED_MODEL_ROOT
    if [[ -z "${VALIDATION_TARGET_RUN_ID}" \
        || "$(dirname "${VALIDATION_CHECKPOINT}")" \
            != "${SAVED_MODEL_ROOT}/${VALIDATION_TARGET_RUN_ID}" \
        || ! "$(basename "${VALIDATION_CHECKPOINT}")" =~ ^checkpoint-[0-9]{6,}$ ]]; then
        echo "VALIDATION_CHECKPOINT must equal SAVED_MODEL_ROOT/run_id/checkpoint-NNNNNN" >&2
        exit 2
    fi
fi
if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "Validation config not found: ${CONFIG_PATH}" >&2
    exit 2
fi
if [[ "${RUNPOD_TEST_MODE:-0}" != "1" ]]; then
    if [[ -z "${WANDB_API_KEY:-}" || "${WANDB_API_KEY}" == *'{{ RUNPOD_SECRET_'* ]]; then
        echo "WANDB_API_KEY RunPod Secret is missing or was not resolved" >&2
        exit 2
    fi
    bash "${SCRIPT_DIR}/verify_runpod_mounted_readiness.sh"
fi

VALIDATION_RESUME="${VALIDATION_RESUME:-1}"
VALIDATION_FORCE_RECOMPUTE="${VALIDATION_FORCE_RECOMPUTE:-0}"
for boolean_name in VALIDATION_RESUME VALIDATION_FORCE_RECOMPUTE; do
    if [[ "${!boolean_name}" != "0" && "${!boolean_name}" != "1" ]]; then
        echo "${boolean_name} must be 0 or 1" >&2
        exit 2
    fi
done
if [[ "${VALIDATION_FORCE_RECOMPUTE}" == "1" ]]; then
    VALIDATION_RESUME=0
    VALIDATION_RECOMPUTE_FULL_MODEL=1
fi

arguments=(--config "${CONFIG_PATH}")
if [[ -n "${VALIDATION_RUN_ID:-}" ]]; then
    arguments+=(--run-id "${VALIDATION_RUN_ID}")
fi
if [[ -n "${VALIDATION_CHECKPOINT:-}" ]]; then
    arguments+=(--checkpoint "${VALIDATION_CHECKPOINT}")
fi
if [[ "${VALIDATION_RECOMPUTE_FULL_MODEL:-0}" == "1" ]]; then
    arguments+=(--recompute-full-model)
fi
if [[ "${VALIDATION_RESUME}" == "1" ]]; then
    arguments+=(--resume)
else
    arguments+=(--no-resume)
fi
if [[ "${VALIDATION_DISABLE_WANDB:-0}" == "1" ]]; then
    arguments+=(--disable-wandb)
fi
if [[ "$#" -gt 0 ]]; then
    arguments+=("$@")
fi

cd "${PROJECT_ROOT}"
# The validation CLI keeps the lifecycle non-terminal until the numerical
# benchmark has completed and persisted its validation-benchmark.json artifact.
exec "${PROJECT_VENV}/bin/python" -m fin_ts_multimodal.cli.validate_benchmarks \
    "${arguments[@]}"
