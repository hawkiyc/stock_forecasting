#!/usr/bin/env bash

# Prepare the persistent directory layout without downloading or installing anything.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/runpod_paths.sh
source "${SCRIPT_DIR}/lib/runpod_paths.sh"

NETWORK_VOLUME_ROOT="${NETWORK_VOLUME_ROOT:-/runpod-volume}"
PROJECT_ROOT="${PROJECT_ROOT:-${NETWORK_VOLUME_ROOT}/stock_forecasting}"
DATA_ROOT="${DATA_ROOT:-${NETWORK_VOLUME_ROOT}/data}"
CACHE_ROOT="${CACHE_ROOT:-${NETWORK_VOLUME_ROOT}/cache}"
LOG_ROOT="${LOG_ROOT:-${NETWORK_VOLUME_ROOT}/logs}"
SAVED_MODEL_ROOT="${SAVED_MODEL_ROOT:-${NETWORK_VOLUME_ROOT}/savedModel}"
WANDB_DIR="${WANDB_DIR:-${NETWORK_VOLUME_ROOT}}"
LIFECYCLE_ROOT="${LIFECYCLE_ROOT:-${NETWORK_VOLUME_ROOT}/lifecycle}"

if [[ -z "${RUNPOD_POD_ID:-}" && "${RUNPOD_TEST_MODE:-0}" != "1" ]]; then
    echo "Network volume bootstrap must run inside a RunPod Pod" >&2
    exit 2
fi

runpod_validate_absolute_path "${NETWORK_VOLUME_ROOT}" NETWORK_VOLUME_ROOT
case "${NETWORK_VOLUME_ROOT}" in
    /workspace|/workspace/*)
        echo "NETWORK_VOLUME_ROOT must never use ephemeral /workspace" >&2
        exit 2
        ;;
esac

for path_name in PROJECT_ROOT DATA_ROOT CACHE_ROOT LOG_ROOT SAVED_MODEL_ROOT WANDB_DIR LIFECYCLE_ROOT; do
    path_value="${!path_name}"
    runpod_validate_path_in_root \
        "${path_value}" "${NETWORK_VOLUME_ROOT}" "${path_name}" NETWORK_VOLUME_ROOT
done
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
if [[ "${LIFECYCLE_ROOT}" != "${NETWORK_VOLUME_ROOT}/lifecycle" ]]; then
    echo "LIFECYCLE_ROOT must equal NETWORK_VOLUME_ROOT/lifecycle" >&2
    exit 2
fi

XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
HF_HOME="${CACHE_ROOT}/huggingface"
TORCH_HOME="${CACHE_ROOT}/torch"
POETRY_CACHE_DIR="${CACHE_ROOT}/pypoetry"
TMPDIR="${NETWORK_VOLUME_ROOT}/tmp"
WANDB_CACHE_DIR="${CACHE_ROOT}/wandb"
WANDB_ARTIFACT_DIR="${NETWORK_VOLUME_ROOT}/wandb-artifacts"

for path_name in \
    XDG_CACHE_HOME HF_HOME TORCH_HOME POETRY_CACHE_DIR TMPDIR \
    WANDB_CACHE_DIR WANDB_ARTIFACT_DIR; do
    runpod_validate_path_in_root \
        "${!path_name}" "${NETWORK_VOLUME_ROOT}" "${path_name}" NETWORK_VOLUME_ROOT
done

mkdir -p \
    "${PROJECT_ROOT}" \
    "${DATA_ROOT}/raw" \
    "${DATA_ROOT}/prepared" \
    "${CACHE_ROOT}" \
    "${LOG_ROOT}" \
    "${SAVED_MODEL_ROOT}" \
    "${WANDB_DIR}/wandb" \
    "${LIFECYCLE_ROOT}/stage1" \
    "${XDG_CACHE_HOME}" \
    "${HF_HOME}" \
    "${TORCH_HOME}" \
    "${POETRY_CACHE_DIR}" \
    "${TMPDIR}" \
    "${WANDB_CACHE_DIR}" \
    "${WANDB_ARTIFACT_DIR}"

PATHS_FILE="${NETWORK_VOLUME_ROOT}/runpod-paths.env"
PATHS_FILE_TMP="${PATHS_FILE}.tmp"
runpod_validate_path_in_root \
    "${PATHS_FILE}" "${NETWORK_VOLUME_ROOT}" PATHS_FILE NETWORK_VOLUME_ROOT
runpod_validate_path_in_root \
    "${PATHS_FILE_TMP}" "${NETWORK_VOLUME_ROOT}" PATHS_FILE_TMP NETWORK_VOLUME_ROOT
umask 077
{
    printf 'export NETWORK_VOLUME_ROOT=%q\n' "${NETWORK_VOLUME_ROOT}"
    printf 'export RUNPOD_VOLUME_ROOT=%q\n' "${NETWORK_VOLUME_ROOT}"
    printf 'export PROJECT_ROOT=%q\n' "${PROJECT_ROOT}"
    printf 'export DATA_ROOT=%q\n' "${DATA_ROOT}"
    printf 'export CACHE_ROOT=%q\n' "${CACHE_ROOT}"
    printf 'export LOG_ROOT=%q\n' "${LOG_ROOT}"
    printf 'export SAVED_MODEL_ROOT=%q\n' "${SAVED_MODEL_ROOT}"
    printf 'export WANDB_DIR=%q\n' "${WANDB_DIR}"
    printf 'export LIFECYCLE_ROOT=%q\n' "${LIFECYCLE_ROOT}"
    printf 'export XDG_CACHE_HOME=%q\n' "${XDG_CACHE_HOME}"
    printf 'export HF_HOME=%q\n' "${HF_HOME}"
    printf 'export TORCH_HOME=%q\n' "${TORCH_HOME}"
    printf 'export POETRY_CACHE_DIR=%q\n' "${POETRY_CACHE_DIR}"
    printf 'export TMPDIR=%q\n' "${TMPDIR}"
    printf 'export WANDB_CACHE_DIR=%q\n' "${WANDB_CACHE_DIR}"
    printf 'export WANDB_ARTIFACT_DIR=%q\n' "${WANDB_ARTIFACT_DIR}"
} > "${PATHS_FILE_TMP}"
mv "${PATHS_FILE_TMP}" "${PATHS_FILE}"

printf 'Network volume initialized at %s\n' "${NETWORK_VOLUME_ROOT}"
printf 'Project root: %s\n' "${PROJECT_ROOT}"
printf 'Persistent environment: %s\n' "${PATHS_FILE}"
