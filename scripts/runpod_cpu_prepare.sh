#!/usr/bin/env bash

# Prepare the complete numerical dataset and model cache on a CPU Pod, then terminate it.
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
RUNPOD_CONFIG="${RUNPOD_CONFIG:-configs/stage1_kronos_base_lora.yaml}"
RUNPOD_SELECTION_HELPER="${SCRIPT_DIR}/runpod_selection.py"
RUNPOD_REMOTE_SELECTION_PATH="${RUNPOD_REMOTE_SELECTION_PATH:-}"
RUNPOD_ROLE="${RUNPOD_ROLE:-cpu-prep}"
LAUNCH_ID="${RUNPOD_LAUNCH_ID:-launch-$(date -u +%Y%m%dT%H%M%SZ)-${RANDOM}${RANDOM}}"
WORKFLOW_STARTED_EPOCH="$(date +%s)"
PREP_DIR="${LOG_ROOT}/cpu-prep/${LAUNCH_ID}"
PREP_LOG="${RUNPOD_TMUX_LOG_FILE:-${PREP_DIR}/combined.log}"
CODE_MARKER="${LIFECYCLE_ROOT}/stage1/code.json"
DATASET_MARKER="${LIFECYCLE_ROOT}/stage1/dataset.json"
MODEL_MANIFEST="${NETWORK_VOLUME_ROOT}/cache/hf-models.json"
STAGING_ROOT="${NETWORK_VOLUME_ROOT}/tmp/stage1-prep/${LAUNCH_ID}"
DATA_STAGING_ROOT="${STAGING_ROOT}/data"
RAW_STAGING="${DATA_STAGING_ROOT}/raw/market.parquet"
DOWNLOAD_MANIFEST_STAGING="${DATA_STAGING_ROOT}/download-manifest.json"
DATASET_MANIFEST_STAGING="${DATA_STAGING_ROOT}/dataset-manifest.json"
REQUEST_LOG_STAGING="${DATA_STAGING_ROOT}/manifests/api-request-log.jsonl"
RAW_FINAL="${DATA_ROOT}/raw/market.parquet"
BAR_STORE_FINAL="${DATA_ROOT}/prepared/bar-store"
BAR_STORE_SUCCESS="${BAR_STORE_FINAL}/_SUCCESS.json"
DOWNLOAD_MANIFEST_FINAL="${DATA_ROOT}/download-manifest.json"
DATASET_MANIFEST_FINAL="${DATA_ROOT}/dataset-manifest.json"
REQUEST_LOG_FINAL="${DATA_ROOT}/manifests/api-request-log.jsonl"
API_CACHE_ROOT="${DATA_ROOT}/api-cache"
PROVIDER_CHECKPOINT_ROOT="${DATA_ROOT}/provider-checkpoints"
DOWNLOAD_PROGRESS="${DATA_ROOT}/download-progress.json"
FINAL_DATA_FILES=(
    "${RAW_FINAL}"
    "${BAR_STORE_FINAL}"
    "${DOWNLOAD_MANIFEST_FINAL}"
    "${DATASET_MANIFEST_FINAL}"
    "${REQUEST_LOG_FINAL}"
)
ACQUISITION_FINAL_FILES=(
    "${RAW_FINAL}"
    "${DOWNLOAD_MANIFEST_FINAL}"
    "${REQUEST_LOG_FINAL}"
)
PREPARATION_FINAL_FILES=(
    "${BAR_STORE_SUCCESS}"
    "${DATASET_MANIFEST_FINAL}"
)
FIN_TS_DATASET_PROFILE="${FIN_TS_DATASET_PROFILE:-us_tw_eodhd}"
FIN_TS_H_START="${FIN_TS_H_START:-}"
STAGE1_US_SYMBOLS="${STAGE1_US_SYMBOLS:-}"
STAGE1_US_ETF_SYMBOLS="${STAGE1_US_ETF_SYMBOLS:-}"
STAGE1_SYMBOL_LIMIT="${STAGE1_SYMBOL_LIMIT:-}"
STAGE1_DATA_START="${STAGE1_DATA_START:-2005-01-01}"
STAGE1_DATA_END="${STAGE1_DATA_END:-}"
RUNPOD_DATASET_REVISION="${RUNPOD_DATASET_REVISION:-v1}"
RUNPOD_CPU_MAX_API_CALLS="${RUNPOD_CPU_MAX_API_CALLS:-}"
RUNPOD_CPU_EODHD_QPS="${RUNPOD_CPU_EODHD_QPS:-}"
RUNPOD_CPU_TAIWAN_QPS="${RUNPOD_CPU_TAIWAN_QPS:-}"
RUNPOD_PROVIDER_MAX_BACKOFF_SECONDS="${RUNPOD_PROVIDER_MAX_BACKOFF_SECONDS:-60}"
TPEX_PROXY_URL="${TPEX_PROXY_URL:-}"
TPEX_PROXY_TOKEN="${TPEX_PROXY_TOKEN:-}"
RUNPOD_CPU_MAX_RUNTIME_SECONDS="${RUNPOD_CPU_MAX_RUNTIME_SECONDS:-21600}"
RUNPOD_CPU_PREPARE_RESERVE_SECONDS="${RUNPOD_CPU_PREPARE_RESERVE_SECONDS:-}"
RUNPOD_PYTEST_WORKERS="${RUNPOD_PYTEST_WORKERS:-auto}"
RUNPOD_PYTEST_THREADS_PER_WORKER="${RUNPOD_PYTEST_THREADS_PER_WORKER:-1}"
RUNPOD_REQUESTED_CPU_COUNT="${RUNPOD_REQUESTED_CPU_COUNT:-${RUNPOD_CPU_COUNT:-1}}"
METADATA_PATH="${PREP_DIR}/metadata.json"
RUNPOD_SHUTDOWN_DIR="${PREP_DIR}/shutdown"
RUNPOD_SHUTDOWN_MARKER="${RUNPOD_SHUTDOWN_DIR}/shutdown.json"
export RUNPOD_SHUTDOWN_DIR RUNPOD_SHUTDOWN_MARKER
export NETWORK_VOLUME_ROOT RUNPOD_VOLUME_ROOT="${NETWORK_VOLUME_ROOT}"
export PROJECT_ROOT DATA_ROOT LOG_ROOT LIFECYCLE_ROOT RUNPOD_ROLE RUNPOD_CONFIG
export FIN_TS_DATASET_PROFILE FIN_TS_H_START
export RUNPOD_DATASET_REVISION
export RUNPOD_CPU_MAX_API_CALLS RUNPOD_CPU_EODHD_QPS RUNPOD_CPU_TAIWAN_QPS
export TPEX_PROXY_URL TPEX_PROXY_TOKEN
# The image may export cache paths under ephemeral /workspace; never inherit them.
export HF_HOME="${NETWORK_VOLUME_ROOT}/cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export PIP_CACHE_DIR="${NETWORK_VOLUME_ROOT}/cache/pip"
export TMPDIR="${NETWORK_VOLUME_ROOT}/tmp"

if [[ -z "${RUNPOD_POD_ID:-}" && "${RUNPOD_TEST_MODE:-0}" != "1" ]]; then
    echo "CPU preparation must run inside a RunPod Pod" >&2
    exit 2
fi
if [[ "${RUNPOD_ROLE}" != "cpu-prep" ]]; then
    echo "CPU preparation requires RUNPOD_ROLE=cpu-prep" >&2
    exit 2
fi
if [[ ! "${RUNPOD_SELECTION_ID:-}" =~ ^selection-[0-9a-f]{16}$ \
    || ! "${RUNPOD_SELECTION_SHA256:-}" =~ ^[0-9a-f]{64}$ \
    || ! "${RUNPOD_DATASET_REQUEST_SHA256:-}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "CPU preparation requires a valid immutable training selection" >&2
    exit 2
fi
if [[ ! "${RUNPOD_DATASET_REVISION}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$ ]]; then
    echo "RUNPOD_DATASET_REVISION must be a safe 1-64 character label" >&2
    exit 2
fi
if [[ -z "${RUNPOD_REMOTE_SELECTION_PATH}" ]]; then
    echo "RUNPOD_REMOTE_SELECTION_PATH is required" >&2
    exit 2
fi
if [[ ! "${FIN_TS_DATASET_PROFILE}" =~ ^(tw_only|us_only_eodhd|us_tw_eodhd|us_tw_massive)$ ]]; then
    echo "FIN_TS_DATASET_PROFILE is unsupported" >&2
    exit 2
fi
if [[ ! "${FIN_TS_H_START}" =~ ^[1-3]$ ]]; then
    echo "FIN_TS_H_START must be 1, 2, or 3" >&2
    exit 2
fi
if [[ "${FIN_TS_DATASET_PROFILE}" == "us_tw_massive" ]]; then
    echo "The Massive provider interface is reserved but not implemented" >&2
    exit 2
fi
if [[ ! "${RUNPOD_PYTEST_WORKERS}" =~ ^(auto|0|[1-9][0-9]*)$ ]]; then
    echo "RUNPOD_PYTEST_WORKERS must be auto, 0, or a positive integer" >&2
    exit 2
fi
if [[ ! "${RUNPOD_PYTEST_THREADS_PER_WORKER}" =~ ^[1-9][0-9]*$ ]]; then
    echo "RUNPOD_PYTEST_THREADS_PER_WORKER must be a positive integer" >&2
    exit 2
fi
if [[ ! "${RUNPOD_REQUESTED_CPU_COUNT}" =~ ^[1-9][0-9]*$ \
    || "${RUNPOD_REQUESTED_CPU_COUNT}" -gt 32 ]]; then
    echo "RUNPOD_REQUESTED_CPU_COUNT must be an integer from 1 through 32" >&2
    exit 2
fi
if [[ -z "${STAGE1_DATA_END}" ]]; then
    echo "STAGE1_DATA_END is required; rerun configure with an explicit --end date" >&2
    exit 2
fi
if [[ -n "${STAGE1_SYMBOL_LIMIT}" && ! "${STAGE1_SYMBOL_LIMIT}" =~ ^[1-9][0-9]*$ ]]; then
    echo "STAGE1_SYMBOL_LIMIT must be empty or a positive integer" >&2
    exit 2
fi
for symbol_list in "${STAGE1_US_SYMBOLS}" "${STAGE1_US_ETF_SYMBOLS}"; do
    if [[ -n "${symbol_list}" \
        && ! "${symbol_list}" =~ ^[A-Za-z0-9.^_-]+([[:space:]]+[A-Za-z0-9.^_-]+)*$ ]]; then
        echo "Optional US symbol lists contain unsupported characters" >&2
        exit 2
    fi
done
if [[ ! "${RUNPOD_CPU_MAX_RUNTIME_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "RUNPOD_CPU_MAX_RUNTIME_SECONDS must be a positive integer" >&2
    exit 2
fi
if [[ ! "${RUNPOD_CPU_MAX_API_CALLS}" =~ ^[1-9][0-9]*$ \
    || ! "${RUNPOD_CPU_EODHD_QPS}" =~ ^(0|[1-9][0-9]*)([.][0-9]+)?$ \
    || "${RUNPOD_CPU_EODHD_QPS}" =~ ^0([.]0+)?$ \
    || ! "${RUNPOD_CPU_TAIWAN_QPS}" =~ ^(0|[1-9][0-9]*)([.][0-9]+)?$ \
    || "${RUNPOD_CPU_TAIWAN_QPS}" =~ ^0([.]0+)?$ ]]; then
    echo "CPU acquisition budget and QPS settings are invalid" >&2
    exit 2
fi
if [[ ! "${RUNPOD_PROVIDER_MAX_BACKOFF_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "RUNPOD_PROVIDER_MAX_BACKOFF_SECONDS must be a positive integer" >&2
    exit 2
fi
if [[ -z "${RUNPOD_CPU_PREPARE_RESERVE_SECONDS}" ]]; then
    RUNPOD_CPU_PREPARE_RESERVE_SECONDS=$((RUNPOD_CPU_MAX_RUNTIME_SECONDS / 4))
    if [[ ${RUNPOD_CPU_PREPARE_RESERVE_SECONDS} -lt 1 ]]; then
        RUNPOD_CPU_PREPARE_RESERVE_SECONDS=1
    fi
    if [[ ${RUNPOD_CPU_PREPARE_RESERVE_SECONDS} -gt 7200 ]]; then
        RUNPOD_CPU_PREPARE_RESERVE_SECONDS=7200
    fi
elif [[ ! "${RUNPOD_CPU_PREPARE_RESERVE_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "RUNPOD_CPU_PREPARE_RESERVE_SECONDS must be a positive integer" >&2
    exit 2
fi
if [[ ${RUNPOD_CPU_PREPARE_RESERVE_SECONDS} -ge ${RUNPOD_CPU_MAX_RUNTIME_SECONDS} ]]; then
    echo "CPU data-preparation reserve must be shorter than the Pod workflow runtime" >&2
    exit 2
fi
WORKFLOW_DEADLINE_EPOCH=$((WORKFLOW_STARTED_EPOCH + RUNPOD_CPU_MAX_RUNTIME_SECONDS))
ACQUISITION_DEADLINE_EPOCH=$((WORKFLOW_DEADLINE_EPOCH - RUNPOD_CPU_PREPARE_RESERVE_SECONDS))

runpod_validate_absolute_path "${NETWORK_VOLUME_ROOT}" NETWORK_VOLUME_ROOT
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
for path_name in \
    PROJECT_ROOT DATA_ROOT LOG_ROOT LIFECYCLE_ROOT POETRY_BIN CONFIG_PATH \
    PREP_DIR PREP_LOG CODE_MARKER DATASET_MARKER MODEL_MANIFEST STAGING_ROOT \
    DATA_STAGING_ROOT RAW_STAGING DOWNLOAD_MANIFEST_STAGING \
    DATASET_MANIFEST_STAGING REQUEST_LOG_STAGING RAW_FINAL BAR_STORE_FINAL \
    BAR_STORE_SUCCESS \
    DOWNLOAD_MANIFEST_FINAL DATASET_MANIFEST_FINAL REQUEST_LOG_FINAL API_CACHE_ROOT \
    DOWNLOAD_PROGRESS \
    METADATA_PATH RUNPOD_SHUTDOWN_DIR RUNPOD_SHUTDOWN_MARKER HF_HOME \
    RUNPOD_SELECTION_HELPER RUNPOD_REMOTE_SELECTION_PATH \
    TRANSFORMERS_CACHE PIP_CACHE_DIR TMPDIR; do
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
if [[ ! "${LAUNCH_ID}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "RUNPOD_LAUNCH_ID contains unsupported characters" >&2
    exit 2
fi
if [[ ! -x "${RUNPOD_PYTHON_BIN}" ]]; then
    echo "RunPod image Python is unavailable: ${RUNPOD_PYTHON_BIN}" >&2
    exit 127
fi

DETECTED_CPU_COUNT=""
if command -v nproc >/dev/null 2>&1; then
    DETECTED_CPU_COUNT="$(nproc)"
elif command -v getconf >/dev/null 2>&1; then
    DETECTED_CPU_COUNT="$(getconf _NPROCESSORS_ONLN)"
fi
if [[ ! "${DETECTED_CPU_COUNT}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Unable to detect the CPU count available to this Pod" >&2
    exit 2
fi
FIN_TS_CPU_WORKERS="${RUNPOD_REQUESTED_CPU_COUNT}"
if [[ "${DETECTED_CPU_COUNT}" -lt "${FIN_TS_CPU_WORKERS}" ]]; then
    FIN_TS_CPU_WORKERS="${DETECTED_CPU_COUNT}"
fi
if [[ -n "${RUNPOD_CPU_COUNT:-}" ]]; then
    if [[ ! "${RUNPOD_CPU_COUNT}" =~ ^[1-9][0-9]*$ ]]; then
        echo "RunPod provided an invalid RUNPOD_CPU_COUNT" >&2
        exit 2
    fi
    if [[ "${RUNPOD_CPU_COUNT}" -lt "${FIN_TS_CPU_WORKERS}" ]]; then
        FIN_TS_CPU_WORKERS="${RUNPOD_CPU_COUNT}"
    fi
fi
if [[ "${RUNPOD_PYTEST_WORKERS}" == "auto" ]]; then
    RUNPOD_PYTEST_WORKERS="${FIN_TS_CPU_WORKERS}"
elif [[ "${RUNPOD_PYTEST_WORKERS}" != "0" \
    && "${RUNPOD_PYTEST_WORKERS}" -gt "${FIN_TS_CPU_WORKERS}" ]]; then
    RUNPOD_PYTEST_WORKERS="${FIN_TS_CPU_WORKERS}"
fi
export FIN_TS_CPU_WORKERS
bash "${SCRIPT_DIR}/verify_runpod_mounted_readiness.sh" --mount-only

read -r -a US_SYMBOLS <<< "${STAGE1_US_SYMBOLS}"
read -r -a US_ETF_SYMBOLS <<< "${STAGE1_US_ETF_SYMBOLS}"
mkdir -p "${PREP_DIR}" "${DATA_STAGING_ROOT}/raw" \
    "${DATA_STAGING_ROOT}/manifests" \
    "${LIFECYCLE_ROOT}/stage1" "${DATA_ROOT}/raw" "${DATA_ROOT}/prepared" \
    "${DATA_ROOT}/manifests" "${DATA_ROOT}/quarantine" "${API_CACHE_ROOT}" \
    "${RUNPOD_SHUTDOWN_DIR}" "${HF_HOME}"

for final_data_file in "${FINAL_DATA_FILES[@]}"; do
    if [[ -L "${final_data_file}" ]]; then
        echo "Selected dataset namespace contains a symlink: ${final_data_file}" >&2
        exit 2
    fi
done

quarantine_known_files() {
    local reason="$1"
    shift
    local quarantine_root="${DATA_ROOT}/quarantine/${reason}-${LAUNCH_ID}"
    local source relative destination
    runpod_validate_path_in_root \
        "${quarantine_root}" "${DATA_ROOT}" quarantine_root DATA_ROOT
    for source in "$@"; do
        if [[ ! -e "${source}" ]]; then
            continue
        fi
        relative="${source#"${DATA_ROOT}/"}"
        if [[ "${relative}" == "${source}" || -z "${relative}" ]]; then
            echo "Refusing to quarantine a path outside DATA_ROOT: ${source}" >&2
            exit 2
        fi
        destination="${quarantine_root}/${relative}"
        mkdir -p "$(dirname "${destination}")"
        mv "${source}" "${destination}"
    done
    printf 'Quarantined incomplete dataset artifacts without deleting them: %s\n' \
        "${quarantine_root}" >&2
}

ACQUISITION_FINAL_COUNT=0
PREPARATION_FINAL_COUNT=0
for final_data_file in "${ACQUISITION_FINAL_FILES[@]}"; do
    if [[ -e "${final_data_file}" ]]; then
        ACQUISITION_FINAL_COUNT=$((ACQUISITION_FINAL_COUNT + 1))
    fi
done
for final_data_file in "${PREPARATION_FINAL_FILES[@]}"; do
    if [[ -e "${final_data_file}" ]]; then
        PREPARATION_FINAL_COUNT=$((PREPARATION_FINAL_COUNT + 1))
    fi
done
REUSE_READY_DATASET=0
REUSE_DOWNLOADED_DATASET=0
if [[ ${ACQUISITION_FINAL_COUNT} -eq ${#ACQUISITION_FINAL_FILES[@]} ]]; then
    if [[ ${PREPARATION_FINAL_COUNT} -eq ${#PREPARATION_FINAL_FILES[@]} ]]; then
        REUSE_READY_DATASET=1
    else
        if [[ -e "${DATASET_MANIFEST_FINAL}" && ! -e "${BAR_STORE_SUCCESS}" ]]; then
            quarantine_known_files incomplete-preparation "${DATASET_MANIFEST_FINAL}"
        fi
        # A partial or complete bar store without dataset-manifest.json is a
        # durable preparation checkpoint and must remain available for resume.
        REUSE_DOWNLOADED_DATASET=1
    fi
elif [[ ${ACQUISITION_FINAL_COUNT} -eq 0 ]]; then
    if [[ -e "${BAR_STORE_FINAL}" || -e "${DATASET_MANIFEST_FINAL}" ]]; then
        quarantine_known_files orphaned-preparation \
            "${BAR_STORE_FINAL}" "${DATASET_MANIFEST_FINAL}"
    fi
else
    quarantine_known_files incomplete-acquisition "${FINAL_DATA_FILES[@]}"
fi

STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
PREP_SUCCEEDED=0
PREP_RESUMABLE_STATE=""

write_lifecycle_state() {
    local state="$1"
    local exit_code="${2:-}"
    local command=(
        "${RUNPOD_PYTHON_BIN}" "${SCRIPT_DIR}/runpod_readiness.py" write-state
        --output "${DATASET_MARKER}"
        --network-volume-root "${NETWORK_VOLUME_ROOT}"
        --kind stage1-dataset
        --state "${state}"
        --launch-id "${LAUNCH_ID}"
        --log-path "${PREP_LOG}"
    )
    if [[ -n "${exit_code}" ]]; then
        command+=(--exit-code "${exit_code}")
    fi
    if [[ -f "${DOWNLOAD_PROGRESS}" ]]; then
        command+=(--progress-path "${DOWNLOAD_PROGRESS}")
    fi
    "${command[@]}"
}

finish_cpu_prep() {
    local prep_exit_code=$?
    local prep_state=failed
    trap - EXIT INT TERM
    if [[ ${PREP_SUCCEEDED} -eq 1 ]]; then
        prep_state=ready
    elif [[ -n "${PREP_RESUMABLE_STATE}" ]]; then
        prep_state="${PREP_RESUMABLE_STATE}"
    elif [[ ${prep_exit_code} -eq 124 ]]; then
        prep_state=timed_out
    fi
    if [[ ${PREP_SUCCEEDED} -ne 1 ]]; then
        write_lifecycle_state "${prep_state}" "${prep_exit_code}" || true
    fi
    printf '{"started_at":"%s","ended_at":"%s","exit_code":%d,"state":"%s","log_path":"%s","requested_cpu_count":%d,"detected_cpu_count":%d,"effective_workers":%d,"max_runtime_seconds":%d,"preparation_reserve_seconds":%d,"max_api_calls":%s,"eodhd_qps":"%s","taiwan_qps":"%s","provider_max_backoff_seconds":%d,"acquisition_deadline_epoch_seconds":%d}\n' \
        "${STARTED_AT}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${prep_exit_code}" \
        "${prep_state}" \
        "${PREP_LOG}" "${RUNPOD_REQUESTED_CPU_COUNT}" "${DETECTED_CPU_COUNT}" \
        "${FIN_TS_CPU_WORKERS}" "${RUNPOD_CPU_MAX_RUNTIME_SECONDS}" \
        "${RUNPOD_CPU_PREPARE_RESERVE_SECONDS}" \
        "${RUNPOD_CPU_MAX_API_CALLS}" "${RUNPOD_CPU_EODHD_QPS}" \
        "${RUNPOD_CPU_TAIWAN_QPS}" \
        "${RUNPOD_PROVIDER_MAX_BACKOFF_SECONDS}" "${ACQUISITION_DEADLINE_EPOCH}" \
        > "${METADATA_PATH}"
    if [[ -z "${RUNPOD_TMUX_LOG_FILE:-}" ]]; then
        bash "${SCRIPT_DIR}/runpod_self_terminate.sh" || true
    fi
    exit "${prep_exit_code}"
}
trap 'exit 124' TERM
trap 'exit 130' INT
trap finish_cpu_prep EXIT

write_lifecycle_state preparing
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

bash "${SCRIPT_DIR}/bootstrap_network_volume.sh"
RUNPOD_ROLE=cpu-prep bash "${SCRIPT_DIR}/setup_runpod_environment.sh"
if [[ ! -x "${POETRY_BIN}" ]]; then
    echo "Persistent Poetry is unavailable after environment setup" >&2
    exit 127
fi

cd "${PROJECT_ROOT}"
"${POETRY_BIN}" env use "${PROJECT_ROOT}/.venv/bin/python"
"${POETRY_BIN}" run ruff check .
PYTEST_TMPDIR="/tmp/fin-ts-pytest-${LAUNCH_ID}"
PYTEST_DATASET_MARKER="${PYTEST_TMPDIR}/pytest-dataset-readiness.json"
PYTEST_ARGUMENTS=()
if [[ "${RUNPOD_PYTEST_WORKERS}" != "0" ]]; then
    PYTEST_ARGUMENTS=(-n "${RUNPOD_PYTEST_WORKERS}" --dist=load)
    printf 'Running the complete pytest suite with xdist workers: %s\n' \
        "${RUNPOD_PYTEST_WORKERS}"
else
    printf 'Running the complete pytest suite serially\n'
fi
mkdir -p "${PYTEST_TMPDIR}"
OMP_NUM_THREADS="${RUNPOD_PYTEST_THREADS_PER_WORKER}" \
MKL_NUM_THREADS="${RUNPOD_PYTEST_THREADS_PER_WORKER}" \
OPENBLAS_NUM_THREADS="${RUNPOD_PYTEST_THREADS_PER_WORKER}" \
NUMEXPR_NUM_THREADS="${RUNPOD_PYTEST_THREADS_PER_WORKER}" \
TORCH_NUM_THREADS="${RUNPOD_PYTEST_THREADS_PER_WORKER}" \
TMPDIR="${PYTEST_TMPDIR}" \
DATASET_READINESS_MANIFEST="${PYTEST_DATASET_MARKER}" \
    "${POETRY_BIN}" run pytest "${PYTEST_ARGUMENTS[@]}"

RUNPOD_ROLE=cpu-prep RUNPOD_CONFIG="${RUNPOD_CONFIG}" \
    bash "${SCRIPT_DIR}/prefetch_hf_models.sh" --smoke-time-series-backbone

if [[ ${REUSE_READY_DATASET} -eq 1 ]]; then
    if ! "${POETRY_BIN}" run fin-ts-verify-download \
        --manifest "${DOWNLOAD_MANIFEST_FINAL}" \
        --raw "${RAW_FINAL}"; then
        quarantine_known_files obsolete-security-scope "${FINAL_DATA_FILES[@]}"
        REUSE_READY_DATASET=0
        REUSE_DOWNLOADED_DATASET=0
        printf 'The immutable dataset uses an obsolete acquisition security scope; rebuilding from verified provider cache entries.\n' >&2
    fi
fi
if [[ ${REUSE_READY_DATASET} -eq 1 ]]; then
    "${RUNPOD_PYTHON_BIN}" "${SCRIPT_DIR}/runpod_readiness.py" check-code \
        --marker "${CODE_MARKER}" \
        --project-root "${PROJECT_ROOT}"
    "${POETRY_BIN}" run fin-ts-verify-stage1-data \
        --dataset-manifest "${DATASET_MANIFEST_FINAL}" \
        --code-manifest "${CODE_MARKER}" \
        --model-manifest "${MODEL_MANIFEST}" \
        --config "${CONFIG_PATH}" \
        --volume-root "${NETWORK_VOLUME_ROOT}" \
        --launch-id "${LAUNCH_ID}" \
        --output "${DATASET_MARKER}"
    "${RUNPOD_PYTHON_BIN}" "${RUNPOD_SELECTION_HELPER}" bind-marker \
        --project-root "${PROJECT_ROOT}" \
        --selection "${RUNPOD_REMOTE_SELECTION_PATH}" \
        --marker "${DATASET_MARKER}" \
        --volume-root "${NETWORK_VOLUME_ROOT}"
    PREP_SUCCEEDED=1
    printf 'CPU preparation reused immutable data; selection=%s; dataset profile=%s; readiness manifest: %s\n' \
        "${RUNPOD_SELECTION_ID}" "${FIN_TS_DATASET_PROFILE}" "${DATASET_MARKER}"
    exit 0
fi

publish_download_checkpoint() {
    local destination
    for destination in "${ACQUISITION_FINAL_FILES[@]}"; do
        if [[ -e "${destination}" || -L "${destination}" ]]; then
            echo "Refusing to overwrite acquisition checkpoint artifact: ${destination}" >&2
            return 3
        fi
    done
    # All paths are on the same network volume. Hard links publish the durable
    # checkpoint without copying a potentially large raw Parquet file.
    (
        trap '' INT TERM
        ln "${RAW_STAGING}" "${RAW_FINAL}"
        ln "${REQUEST_LOG_STAGING}" "${REQUEST_LOG_FINAL}"
        ln "${DOWNLOAD_MANIFEST_STAGING}" "${DOWNLOAD_MANIFEST_FINAL}"
    )
}

if [[ ${REUSE_DOWNLOADED_DATASET} -eq 1 ]]; then
    if ! "${POETRY_BIN}" run fin-ts-verify-download \
        --manifest "${DOWNLOAD_MANIFEST_FINAL}" \
        --raw "${RAW_FINAL}"; then
        quarantine_known_files invalid-acquisition "${FINAL_DATA_FILES[@]}"
        REUSE_DOWNLOADED_DATASET=0
        printf 'The invalid downloaded checkpoint was quarantined; rebuilding from verified provider cache entries.\n' >&2
    else
        printf 'Reused durable downloaded checkpoint; no provider API calls are required.\n'
    fi
fi
if [[ ${REUSE_DOWNLOADED_DATASET} -eq 0 ]]; then
    # Provider credentials are acquisition-only. A durable raw/download
    # checkpoint must remain preparable even when provider Secrets are absent.
    if [[ "${FIN_TS_DATASET_PROFILE}" == *eodhd* \
        && ( -z "${EODHD_API_TOKEN:-}" \
            || "${EODHD_API_TOKEN}" == *'{{ RUNPOD_SECRET_'* ) ]]; then
        echo "EODHD_API_TOKEN RunPod Secret is missing or was not resolved" >&2
        exit 2
    fi
    if [[ "${FIN_TS_DATASET_PROFILE}" == "tw_only" \
        || "${FIN_TS_DATASET_PROFILE}" == "us_tw_eodhd" \
        || "${FIN_TS_DATASET_PROFILE}" == "us_tw_massive" ]]; then
        if [[ -z "${TPEX_PROXY_TOKEN}" \
            || "${TPEX_PROXY_TOKEN}" == *'{{ RUNPOD_SECRET_'* ]]; then
            echo "TPEX_PROXY_TOKEN RunPod Secret is missing or was not resolved" >&2
            exit 2
        fi
        if [[ ! "${TPEX_PROXY_URL}" =~ ^https://[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?(\.[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?)*\.run\.app/?$ ]]; then
            echo "TPEX_PROXY_URL is missing or is not an approved run.app origin" >&2
            exit 2
        fi
    fi
    DOWNLOAD_ARGUMENTS=(
        --profile "${FIN_TS_DATASET_PROFILE}"
        --start "${STAGE1_DATA_START}"
        --end "${STAGE1_DATA_END}"
        --output "${RAW_STAGING}"
        --manifest-root "${DATA_STAGING_ROOT}"
        --raw-cache-root "${API_CACHE_ROOT}"
        --provider-checkpoint-root "${PROVIDER_CHECKPOINT_ROOT}"
        --cache-revision "${RUNPOD_DATASET_REVISION}"
        --progress-path "${DOWNLOAD_PROGRESS}"
        --dataset-request-sha256 "${RUNPOD_DATASET_REQUEST_SHA256}"
        --selection-id "${RUNPOD_SELECTION_ID}"
        --selection-sha256 "${RUNPOD_SELECTION_SHA256}"
        --launch-id "${LAUNCH_ID}"
        --max-api-calls "${RUNPOD_CPU_MAX_API_CALLS}"
        --eodhd-qps "${RUNPOD_CPU_EODHD_QPS}"
        --taiwan-qps "${RUNPOD_CPU_TAIWAN_QPS}"
        --max-backoff-seconds "${RUNPOD_PROVIDER_MAX_BACKOFF_SECONDS}"
        --workers "${FIN_TS_CPU_WORKERS}"
        --acquisition-deadline-epoch-seconds "${ACQUISITION_DEADLINE_EPOCH}"
        --preparation-reserve-seconds "${RUNPOD_CPU_PREPARE_RESERVE_SECONDS}"
    )
    if [[ ${#US_SYMBOLS[@]} -gt 0 ]]; then
        DOWNLOAD_ARGUMENTS+=(--symbols "${US_SYMBOLS[@]}")
    fi
    if [[ ${#US_ETF_SYMBOLS[@]} -gt 0 ]]; then
        DOWNLOAD_ARGUMENTS+=(--etf-symbols "${US_ETF_SYMBOLS[@]}")
    fi
    if [[ -n "${STAGE1_SYMBOL_LIMIT}" ]]; then
        DOWNLOAD_ARGUMENTS+=(--symbol-limit "${STAGE1_SYMBOL_LIMIT}")
    fi
    if [[ "${FIN_TS_DATASET_PROFILE}" == "tw_only" \
        || "${FIN_TS_DATASET_PROFILE}" == "us_tw_eodhd" \
        || "${FIN_TS_DATASET_PROFILE}" == "us_tw_massive" ]]; then
        bash "${SCRIPT_DIR}/warm_tpex_cloud_run_relay.sh"
    fi
    set +e
    "${POETRY_BIN}" run fin-ts-download "${DOWNLOAD_ARGUMENTS[@]}"
    DOWNLOAD_EXIT_CODE=$?
    set -e
    if [[ ${DOWNLOAD_EXIT_CODE} -eq 75 ]]; then
        PREP_RESUMABLE_STATE="$("${RUNPOD_PYTHON_BIN}" -c \
            'import json, sys
with open(sys.argv[1], encoding="utf-8") as stream:
    state = json.load(stream).get("state")
allowed = {"waiting_for_provider", "waiting_for_budget", "waiting_for_resume"}
if state not in allowed:
    raise SystemExit("Download progress has no supported resumable state")
print(state)' "${DOWNLOAD_PROGRESS}")"
        printf 'Acquisition paused in state %s. Cached responses at %s and completed provider checkpoints at %s are preserved; rerun the same cpu prepare workflow.\n' \
            "${PREP_RESUMABLE_STATE}" "${API_CACHE_ROOT}" \
            "${PROVIDER_CHECKPOINT_ROOT}" >&2
        exit 75
    fi
    if [[ ${DOWNLOAD_EXIT_CODE} -ne 0 ]]; then
        exit "${DOWNLOAD_EXIT_CODE}"
    fi
    publish_download_checkpoint
    "${POETRY_BIN}" run fin-ts-verify-download \
        --manifest "${DOWNLOAD_MANIFEST_FINAL}" \
        --raw "${RAW_FINAL}"
fi

write_lifecycle_state downloaded
REMAINING_WORKFLOW_SECONDS=$((WORKFLOW_DEADLINE_EPOCH - $(date +%s)))
if [[ ${REMAINING_WORKFLOW_SECONDS} -lt ${RUNPOD_CPU_PREPARE_RESERVE_SECONDS} ]]; then
    PREP_RESUMABLE_STATE=downloaded
    printf 'Raw acquisition is durable, but only %s seconds remain versus the %s-second preparation reserve; data preparation is deferred to the next CPU Pod.\n' \
        "${REMAINING_WORKFLOW_SECONDS}" "${RUNPOD_CPU_PREPARE_RESERVE_SECONDS}" >&2
    exit 75
fi
write_lifecycle_state preparing

set +e
"${POETRY_BIN}" run fin-ts-prepare \
    --input "${RAW_FINAL}" \
    --output "${BAR_STORE_FINAL}" \
    --download-manifest "${DOWNLOAD_MANIFEST_FINAL}" \
    --dataset-manifest "${DATASET_MANIFEST_STAGING}" \
    --window-size 128 \
    --stride 5 \
    --sample-stride 1 \
    --h-start "${FIN_TS_H_START}" \
    --target-horizon 5 \
    --diagnostic-horizons 1 20 \
    --flat-volatility-multiplier 0.25 \
    --max-abs-log-return 0.5 \
    --train-fraction 0.70 \
    --validation-fraction 0.15 \
    --purge-bars 20 \
    --embargo-bars 5 \
    --effective-embargo-bars 14 \
    --deadline-epoch-seconds "$((WORKFLOW_DEADLINE_EPOCH - 120))" \
    --workers "${FIN_TS_CPU_WORKERS}"
PREPARE_EXIT_CODE=$?
set -e
if [[ ${PREPARE_EXIT_CODE} -eq 75 ]]; then
    PREP_RESUMABLE_STATE=waiting_for_preparation
    printf 'Bar-store preparation paused at a durable checkpoint; rerun the same CPU prepare workflow to continue without provider API calls.\n' >&2
    exit 75
fi
if [[ ${PREPARE_EXIT_CODE} -ne 0 ]]; then
    exit "${PREPARE_EXIT_CODE}"
fi

# Refuse to publish data if source changed during the CPU preparation run.
"${RUNPOD_PYTHON_BIN}" "${SCRIPT_DIR}/runpod_readiness.py" check-code \
    --marker "${CODE_MARKER}" \
    --project-root "${PROJECT_ROOT}"

"${POETRY_BIN}" run fin-ts-verify-stage1-data \
    --dataset-manifest "${DATASET_MANIFEST_STAGING}" \
    --code-manifest "${CODE_MARKER}" \
    --model-manifest "${MODEL_MANIFEST}" \
    --config "${CONFIG_PATH}" \
    --volume-root "${NETWORK_VOLUME_ROOT}" \
    --launch-id "${LAUNCH_ID}" \
    --verify-only

if [[ ! -f "${BAR_STORE_SUCCESS}" ]]; then
    echo "Bar-store preparation returned success without _SUCCESS.json" >&2
    exit 3
fi
if [[ -e "${DATASET_MANIFEST_FINAL}" || -L "${DATASET_MANIFEST_FINAL}" ]]; then
    echo "Refusing to overwrite an existing immutable dataset manifest: ${DATASET_MANIFEST_FINAL}" >&2
    exit 3
fi
mv "${DATASET_MANIFEST_STAGING}" "${DATASET_MANIFEST_FINAL}"

"${POETRY_BIN}" run fin-ts-verify-stage1-data \
    --dataset-manifest "${DATASET_MANIFEST_FINAL}" \
    --code-manifest "${CODE_MARKER}" \
    --model-manifest "${MODEL_MANIFEST}" \
    --config "${CONFIG_PATH}" \
    --volume-root "${NETWORK_VOLUME_ROOT}" \
    --launch-id "${LAUNCH_ID}" \
    --output "${DATASET_MARKER}"

"${RUNPOD_PYTHON_BIN}" "${RUNPOD_SELECTION_HELPER}" bind-marker \
    --project-root "${PROJECT_ROOT}" \
    --selection "${RUNPOD_REMOTE_SELECTION_PATH}" \
    --marker "${DATASET_MARKER}" \
    --volume-root "${NETWORK_VOLUME_ROOT}"

PREP_SUCCEEDED=1
printf 'CPU preparation completed; selection=%s; dataset profile=%s; readiness manifest: %s\n' \
    "${RUNPOD_SELECTION_ID}" "${FIN_TS_DATASET_PROFILE}" "${DATASET_MARKER}"
