#!/usr/bin/env bash

# Revalidate the full-data Stage 2 contract on a CPU Pod, then terminate it.
set -Eeuo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/runpod_paths.sh
source "${SCRIPT_DIR}/lib/runpod_paths.sh"

NETWORK_VOLUME_ROOT="${NETWORK_VOLUME_ROOT:-${RUNPOD_VOLUME_ROOT:-/runpod-volume}}"
PROJECT_ROOT="${PROJECT_ROOT:-${NETWORK_VOLUME_ROOT}/stock_forecasting}"
DATA_ROOT="${DATA_ROOT:-${NETWORK_VOLUME_ROOT}/data}"
LOG_ROOT="${LOG_ROOT:-${NETWORK_VOLUME_ROOT}/logs}"
LIFECYCLE_ROOT="${LIFECYCLE_ROOT:-${NETWORK_VOLUME_ROOT}/lifecycle}"
POETRY_VERSION="${POETRY_VERSION:-2.4.0}"
POETRY_BIN="${RUNPOD_POETRY_BIN:-${NETWORK_VOLUME_ROOT}/tools/poetry/${POETRY_VERSION}/bin/poetry}"
RUNPOD_PYTHON_BIN="${RUNPOD_PYTHON_BIN:-/usr/local/bin/python}"
RUNPOD_ROLE="${RUNPOD_ROLE:-cpu-prep}"
RUNPOD_SELECTION_HELPER="${SCRIPT_DIR}/runpod_selection.py"
RUNPOD_REMOTE_SELECTION_PATH="${RUNPOD_REMOTE_SELECTION_PATH:-}"
# Keep the established environment name so existing RunPod launch automation remains compatible.
STAGE2_CONFIG="${RUNPOD_MIXED_CONFIG:-configs/stage2_kronos_base_lora.yaml}"
FIN_TS_DATASET_PROFILE="${FIN_TS_DATASET_PROFILE:-us_tw_eodhd}"
LAUNCH_ID="${RUNPOD_LAUNCH_ID:-launch-$(date -u +%Y%m%dT%H%M%SZ)-${RANDOM}${RANDOM}}"
FINALIZE_DIR="${LOG_ROOT}/cpu-finalize/${LAUNCH_ID}"
FINALIZE_LOG="${RUNPOD_TMUX_LOG_FILE:-${FINALIZE_DIR}/combined.log}"
METADATA_PATH="${FINALIZE_DIR}/metadata.json"
CODE_MARKER="${LIFECYCLE_ROOT}/stage1/code.json"
DATASET_MARKER="${LIFECYCLE_ROOT}/stage1/dataset.json"
# Preserve the existing lifecycle path and kind for deployed guard compatibility.
FINALIZATION_MARKER="${LIFECYCLE_ROOT}/stage1/mixed-finalization.json"
MODEL_MANIFEST="${NETWORK_VOLUME_ROOT}/cache/hf-models.json"
DATASET_MANIFEST="${DATA_ROOT}/dataset-manifest.json"
RUNPOD_SHUTDOWN_DIR="${FINALIZE_DIR}/shutdown"
RUNPOD_SHUTDOWN_MARKER="${RUNPOD_SHUTDOWN_DIR}/shutdown.json"
export RUNPOD_SHUTDOWN_DIR RUNPOD_SHUTDOWN_MARKER
export NETWORK_VOLUME_ROOT RUNPOD_VOLUME_ROOT="${NETWORK_VOLUME_ROOT}"
export PROJECT_ROOT DATA_ROOT LOG_ROOT LIFECYCLE_ROOT RUNPOD_ROLE
export FIN_TS_DATASET_PROFILE

if [[ -z "${RUNPOD_POD_ID:-}" && "${RUNPOD_TEST_MODE:-0}" != "1" ]]; then
    echo "Stage 2 contract finalization must run inside a RunPod Pod" >&2
    exit 2
fi
if [[ "${RUNPOD_ROLE}" != "cpu-prep" ]]; then
    echo "Stage 2 contract finalization requires the CPU-only cpu-prep role" >&2
    exit 2
fi
if [[ "${RUNPOD_STAGE:-}" != "stage2" ]]; then
    echo "Stage 2 finalization requires a stage2 selection created before the CPU Pod" >&2
    exit 2
fi
runpod_validate_absolute_path "${NETWORK_VOLUME_ROOT}" NETWORK_VOLUME_ROOT
if [[ ! "${LAUNCH_ID}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "RUNPOD_LAUNCH_ID contains unsupported characters" >&2
    exit 2
fi
if [[ ! "${STAGE2_CONFIG}" =~ ^[A-Za-z0-9._/-]+$ \
    || "${STAGE2_CONFIG}" == /* \
    || "/${STAGE2_CONFIG}/" == *"/../"* ]]; then
    echo "RUNPOD_MIXED_CONFIG must be a safe path relative to PROJECT_ROOT" >&2
    exit 2
fi
STAGE2_CONFIG_PATH="${PROJECT_ROOT}/${STAGE2_CONFIG}"
for path_name in \
    PROJECT_ROOT DATA_ROOT LOG_ROOT LIFECYCLE_ROOT POETRY_BIN FINALIZE_DIR \
    FINALIZE_LOG METADATA_PATH CODE_MARKER DATASET_MARKER FINALIZATION_MARKER \
    MODEL_MANIFEST DATASET_MANIFEST STAGE2_CONFIG_PATH RUNPOD_SHUTDOWN_DIR \
    RUNPOD_SHUTDOWN_MARKER RUNPOD_SELECTION_HELPER RUNPOD_REMOTE_SELECTION_PATH; do
    path_value="${!path_name}"
    case "${path_value}" in
        /workspace|/workspace/*)
            echo "${path_name} must never use /workspace" >&2
            exit 2
            ;;
    esac
    runpod_validate_path_in_root \
        "${path_value}" "${NETWORK_VOLUME_ROOT}" "${path_name}" NETWORK_VOLUME_ROOT
done
if [[ ! -x "${RUNPOD_PYTHON_BIN}" ]]; then
    echo "RunPod image Python is unavailable: ${RUNPOD_PYTHON_BIN}" >&2
    exit 127
fi
if [[ ! -f "${STAGE2_CONFIG_PATH}" || -L "${STAGE2_CONFIG_PATH}" ]]; then
    echo "Stage 2 config is missing or is a symlink: ${STAGE2_CONFIG_PATH}" >&2
    exit 2
fi
bash "${SCRIPT_DIR}/verify_runpod_mounted_readiness.sh" --mount-only

mkdir -p "${FINALIZE_DIR}" "${LIFECYCLE_ROOT}/stage1" "${RUNPOD_SHUTDOWN_DIR}"
STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
FINALIZE_SUCCEEDED=0

write_finalization_state() {
    local state="$1"
    local exit_code="${2:-}"
    local command=(
        "${RUNPOD_PYTHON_BIN}" "${SCRIPT_DIR}/runpod_readiness.py" write-state
        --output "${FINALIZATION_MARKER}"
        --network-volume-root "${NETWORK_VOLUME_ROOT}"
        --kind stage1-mixed-finalization
        --state "${state}"
        --launch-id "${LAUNCH_ID}"
        --log-path "${FINALIZE_LOG}"
    )
    if [[ -n "${exit_code}" ]]; then
        command+=(--exit-code "${exit_code}")
    fi
    "${command[@]}"
}

finish_cpu_finalize() {
    local finalize_exit_code=$?
    trap - EXIT INT TERM
    if [[ ${FINALIZE_SUCCEEDED} -ne 1 ]]; then
        write_finalization_state failed "${finalize_exit_code}" || true
    fi
    printf '{"started_at":"%s","ended_at":"%s","exit_code":%d,"state":"%s","log_path":"%s"}\n' \
        "${STARTED_AT}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${finalize_exit_code}" \
        "$([[ ${FINALIZE_SUCCEEDED} -eq 1 ]] && printf ready || printf failed)" \
        "${FINALIZE_LOG}" > "${METADATA_PATH}"
    if [[ -z "${RUNPOD_TMUX_LOG_FILE:-}" ]]; then
        bash "${SCRIPT_DIR}/runpod_self_terminate.sh" || true
    fi
    exit "${finalize_exit_code}"
}
trap 'exit 124' TERM
trap 'exit 130' INT
trap finish_cpu_finalize EXIT

write_finalization_state finalizing
if [[ -z "${HF_TOKEN:-}" || "${HF_TOKEN}" == *'{{ RUNPOD_SECRET_'* ]]; then
    echo "HF_TOKEN RunPod Secret is missing or was not resolved" >&2
    exit 2
fi
"${RUNPOD_PYTHON_BIN}" "${SCRIPT_DIR}/runpod_readiness.py" quarantine-stale-code \
    --marker "${CODE_MARKER}" \
    --project-root "${PROJECT_ROOT}" \
    --network-volume-root "${NETWORK_VOLUME_ROOT}"
"${RUNPOD_PYTHON_BIN}" "${SCRIPT_DIR}/runpod_readiness.py" check-code \
    --marker "${CODE_MARKER}" \
    --project-root "${PROJECT_ROOT}"
"${RUNPOD_PYTHON_BIN}" "${RUNPOD_SELECTION_HELPER}" verify-environment \
    --project-root "${PROJECT_ROOT}" \
    --selection "${RUNPOD_REMOTE_SELECTION_PATH}"
"${RUNPOD_PYTHON_BIN}" "${SCRIPT_DIR}/runpod_readiness.py" check-dataset \
    --marker "${DATASET_MARKER}" \
    --code-marker "${CODE_MARKER}" \
    --stage-config "${STAGE2_CONFIG_PATH}" \
    --network-volume-root "${NETWORK_VOLUME_ROOT}"

bash "${SCRIPT_DIR}/bootstrap_network_volume.sh"
RUNPOD_ROLE=cpu-prep bash "${SCRIPT_DIR}/setup_runpod_environment.sh"
if [[ ! -x "${POETRY_BIN}" ]]; then
    echo "Persistent Poetry is unavailable after environment setup" >&2
    exit 127
fi
RUNPOD_ROLE=cpu-prep RUNPOD_CONFIG="${STAGE2_CONFIG}" \
    bash "${SCRIPT_DIR}/prefetch_hf_models.sh" --smoke-time-series-backbone

cd "${PROJECT_ROOT}"
"${POETRY_BIN}" env use "${PROJECT_ROOT}/.venv/bin/python"
"${POETRY_BIN}" run pytest
"${POETRY_BIN}" run fin-ts-verify-stage1-data \
    --dataset-manifest "${DATASET_MANIFEST}" \
    --code-manifest "${CODE_MARKER}" \
    --model-manifest "${MODEL_MANIFEST}" \
    --config "${STAGE2_CONFIG_PATH}" \
    --volume-root "${NETWORK_VOLUME_ROOT}" \
    --launch-id "${LAUNCH_ID}" \
    --output "${DATASET_MARKER}"
"${RUNPOD_PYTHON_BIN}" "${RUNPOD_SELECTION_HELPER}" bind-marker \
    --project-root "${PROJECT_ROOT}" \
    --selection "${RUNPOD_REMOTE_SELECTION_PATH}" \
    --marker "${DATASET_MARKER}" \
    --volume-root "${NETWORK_VOLUME_ROOT}"

write_finalization_state ready
FINALIZE_SUCCEEDED=1
printf 'Stage 2 full-data contract is ready; dataset profile=%s; manifest: %s\n' \
    "${FIN_TS_DATASET_PROFILE}" "${DATASET_MARKER}"
