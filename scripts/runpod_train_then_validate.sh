#!/usr/bin/env bash

# Train first, then automatically run the resumable validation benchmark.
set -Eeuo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/runpod_paths.sh
source "${SCRIPT_DIR}/lib/runpod_paths.sh"

NETWORK_VOLUME_ROOT="${NETWORK_VOLUME_ROOT:-${RUNPOD_VOLUME_ROOT:-/runpod-volume}}"
PROJECT_ROOT="${PROJECT_ROOT:-${NETWORK_VOLUME_ROOT}/stock_forecasting}"
RUNPOD_CONFIG="${1:-${RUNPOD_CONFIG:-configs/stage1_kronos_base_lora.yaml}}"
PROJECT_VENV="${PROJECT_ROOT}/.venv"
VALIDATION_LIFECYCLE_MARKER="${RUNPOD_VALIDATION_LIFECYCLE_MARKER:-${NETWORK_VOLUME_ROOT}/lifecycle/stage1/validation.json}"
READINESS_HELPER="${PROJECT_ROOT}/scripts/runpod_readiness.py"
RUNPOD_PYTHON_BIN="${RUNPOD_PYTHON_BIN:-/usr/local/bin/python}"

runpod_validate_absolute_path "${NETWORK_VOLUME_ROOT}" NETWORK_VOLUME_ROOT
case "${NETWORK_VOLUME_ROOT}" in
    /workspace|/workspace/*)
        echo "NETWORK_VOLUME_ROOT must never use /workspace" >&2
        exit 2
        ;;
esac
runpod_validate_path_in_root \
    "${PROJECT_ROOT}" "${NETWORK_VOLUME_ROOT}" PROJECT_ROOT NETWORK_VOLUME_ROOT

for run_id_name in RUNPOD_RUN_KEY WANDB_RUN_ID; do
    run_id_value="${!run_id_name:-}"
    if [[ ! "${run_id_value}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,119}$ \
        || "${run_id_value}" == *--* ]]; then
        echo "${run_id_name} must be a canonical single run ID" >&2
        exit 2
    fi
done
if [[ "${RUNPOD_RUN_KEY}" != "${WANDB_RUN_ID}" ]]; then
    echo "RUNPOD_RUN_KEY and WANDB_RUN_ID must identify the same run" >&2
    exit 2
fi

DEFAULT_SHUTDOWN_DIR="${NETWORK_VOLUME_ROOT}/logs/${RUNPOD_RUN_KEY}"
EFFECTIVE_SHUTDOWN_DIR="${RUNPOD_SHUTDOWN_DIR:-${DEFAULT_SHUTDOWN_DIR}}"
TRAINING_COMPLETED_MARKER="${RUNPOD_TRAINING_COMPLETED_MARKER:-${EFFECTIVE_SHUTDOWN_DIR}/training-completed.json}"
IMMUTABLE_TRAINING_COMPLETION_MARKER="${NETWORK_VOLUME_ROOT}/lifecycle/runs/${RUNPOD_RUN_KEY}/training-completed.json"
RUN_MANIFEST="${NETWORK_VOLUME_ROOT}/savedModel/${RUNPOD_RUN_KEY}/run-manifest.json"

if [[ "${RUNPOD_CONFIG}" == /* ]]; then
    CONFIG_PATH="${RUNPOD_CONFIG}"
else
    if [[ ! "${RUNPOD_CONFIG}" =~ ^[A-Za-z0-9._/-]+$ \
        || "/${RUNPOD_CONFIG}/" == *"/../"* \
        || "/${RUNPOD_CONFIG}/" == *"/./"* \
        || "${RUNPOD_CONFIG}" == *"//"* ]]; then
        echo "RUNPOD_CONFIG must be a safe path" >&2
        exit 2
    fi
    CONFIG_PATH="${PROJECT_ROOT}/${RUNPOD_CONFIG}"
fi
runpod_validate_path_in_root \
    "${CONFIG_PATH}" "${NETWORK_VOLUME_ROOT}" CONFIG_PATH NETWORK_VOLUME_ROOT

for path_name in \
    PROJECT_VENV EFFECTIVE_SHUTDOWN_DIR TRAINING_COMPLETED_MARKER \
    IMMUTABLE_TRAINING_COMPLETION_MARKER RUN_MANIFEST \
    VALIDATION_LIFECYCLE_MARKER READINESS_HELPER; do
    runpod_validate_path_in_root \
        "${!path_name}" "${NETWORK_VOLUME_ROOT}" "${path_name}" NETWORK_VOLUME_ROOT
done
for required_path_name in RUNPOD_SHUTDOWN_DIR RUNPOD_TRAINING_LIFECYCLE_MARKER; do
    required_path_value="${!required_path_name:-}"
    if [[ -z "${required_path_value}" ]]; then
        echo "${required_path_name} is required" >&2
        exit 2
    fi
    runpod_validate_path_in_root \
        "${required_path_value}" "${NETWORK_VOLUME_ROOT}" \
        "${required_path_name}" NETWORK_VOLUME_ROOT
done

"${PROJECT_VENV}/bin/python" -m fin_ts_multimodal.cli.train --config "${RUNPOD_CONFIG}"

"${RUNPOD_PYTHON_BIN}" "${READINESS_HELPER}" write-training-completion \
    --output "${IMMUTABLE_TRAINING_COMPLETION_MARKER}" \
    --network-volume-root "${NETWORK_VOLUME_ROOT}" \
    --run-id "${WANDB_RUN_ID}" \
    --run-manifest "${RUN_MANIFEST}" \
    --launch-id "${RUNPOD_LAUNCH_ID}"

mkdir -p "$(dirname "${TRAINING_COMPLETED_MARKER}")"
marker_tmp="${TRAINING_COMPLETED_MARKER}.tmp.$$.$RANDOM"
printf '{"training_completed":true,"completed_at":"%s","wandb_run_id":"%s"}\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${WANDB_RUN_ID}" > "${marker_tmp}"
mv "${marker_tmp}" "${TRAINING_COMPLETED_MARKER}"

"${RUNPOD_PYTHON_BIN}" "${READINESS_HELPER}" write-state \
    --output "${RUNPOD_TRAINING_LIFECYCLE_MARKER}" \
    --network-volume-root "${NETWORK_VOLUME_ROOT}" \
    --kind stage1-training \
    --state finalizing \
    --launch-id "${RUNPOD_LAUNCH_ID}" \
    --log-path "${RUNPOD_SHUTDOWN_DIR}" \
    --wandb-run-id "${WANDB_RUN_ID}" \
    --max-runtime-seconds "${MAX_RUNTIME_SECONDS}" \
    --training-completed 1

VALIDATION_AUTO_RUN="$("${PROJECT_VENV}/bin/python" -c \
    'import sys
from fin_ts_multimodal.config import ExperimentConfig
config = ExperimentConfig.from_yaml(sys.argv[1])
print("1" if config.validation.enabled and config.validation.auto_run_after_training else "0")' \
    "${RUNPOD_CONFIG}")"
if [[ "${VALIDATION_AUTO_RUN}" != "1" ]]; then
    printf 'Automatic validation is disabled by config: %s\n' "${RUNPOD_CONFIG}"
    exit 0
fi

"${RUNPOD_PYTHON_BIN}" "${READINESS_HELPER}" write-state \
    --output "${VALIDATION_LIFECYCLE_MARKER}" \
    --network-volume-root "${NETWORK_VOLUME_ROOT}" \
    --kind stage1-validation \
    --state preparing \
    --launch-id "${RUNPOD_LAUNCH_ID}" \
    --log-path "${RUNPOD_SHUTDOWN_DIR}" \
    --wandb-run-id "${WANDB_RUN_ID}" \
    --max-runtime-seconds "${MAX_RUNTIME_SECONDS}"

exec bash "${SCRIPT_DIR}/runpod_validation.sh" --resume
