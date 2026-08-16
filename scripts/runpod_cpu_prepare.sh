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
PREP_DIR="${LOG_ROOT}/cpu-prep/${LAUNCH_ID}"
PREP_LOG="${RUNPOD_TMUX_LOG_FILE:-${PREP_DIR}/combined.log}"
CODE_MARKER="${LIFECYCLE_ROOT}/stage1/code.json"
DATASET_MARKER="${LIFECYCLE_ROOT}/stage1/dataset.json"
MODEL_MANIFEST="${NETWORK_VOLUME_ROOT}/cache/hf-models.json"
STAGING_ROOT="${NETWORK_VOLUME_ROOT}/tmp/stage1-prep/${LAUNCH_ID}"
DATA_STAGING_ROOT="${STAGING_ROOT}/data"
RAW_STAGING="${DATA_STAGING_ROOT}/raw/market.parquet"
PROCESSED_STAGING="${DATA_STAGING_ROOT}/processed/windows.parquet"
DOWNLOAD_MANIFEST_STAGING="${DATA_STAGING_ROOT}/download-manifest.json"
DATASET_MANIFEST_STAGING="${DATA_STAGING_ROOT}/dataset-manifest.json"
REQUEST_LOG_STAGING="${DATA_STAGING_ROOT}/manifests/api-request-log.jsonl"
RAW_FINAL="${DATA_ROOT}/raw/market.parquet"
PROCESSED_FINAL="${DATA_ROOT}/processed/windows.parquet"
DOWNLOAD_MANIFEST_FINAL="${DATA_ROOT}/download-manifest.json"
DATASET_MANIFEST_FINAL="${DATA_ROOT}/dataset-manifest.json"
REQUEST_LOG_FINAL="${DATA_ROOT}/manifests/api-request-log.jsonl"
API_CACHE_ROOT="${DATA_ROOT}/api-cache"
DOWNLOAD_PROGRESS="${DATA_ROOT}/download-progress.json"
FINAL_DATA_FILES=(
    "${RAW_FINAL}"
    "${PROCESSED_FINAL}"
    "${DOWNLOAD_MANIFEST_FINAL}"
    "${DATASET_MANIFEST_FINAL}"
    "${REQUEST_LOG_FINAL}"
)
FIN_TS_DATASET_PROFILE="${FIN_TS_DATASET_PROFILE:-us_tw_eodhd}"
STAGE1_US_SYMBOLS="${STAGE1_US_SYMBOLS:-}"
STAGE1_US_ETF_SYMBOLS="${STAGE1_US_ETF_SYMBOLS:-}"
STAGE1_SYMBOL_LIMIT="${STAGE1_SYMBOL_LIMIT:-}"
STAGE1_DATA_START="${STAGE1_DATA_START:-2005-01-01}"
STAGE1_DATA_END="${STAGE1_DATA_END:-}"
STAGE1_MAX_API_CALLS="${STAGE1_MAX_API_CALLS:-100000}"
STAGE1_EODHD_QPS="${STAGE1_EODHD_QPS:-16}"
STAGE1_TAIWAN_QPS="${STAGE1_TAIWAN_QPS:-0.5}"
RUNPOD_PYTEST_WORKERS="${RUNPOD_PYTEST_WORKERS:-auto}"
RUNPOD_PYTEST_THREADS_PER_WORKER="${RUNPOD_PYTEST_THREADS_PER_WORKER:-1}"
RUNPOD_REQUESTED_CPU_COUNT="${RUNPOD_REQUESTED_CPU_COUNT:-${RUNPOD_CPU_COUNT:-1}}"
METADATA_PATH="${PREP_DIR}/metadata.json"
RUNPOD_SHUTDOWN_DIR="${PREP_DIR}/shutdown"
RUNPOD_SHUTDOWN_MARKER="${RUNPOD_SHUTDOWN_DIR}/shutdown.json"
export RUNPOD_SHUTDOWN_DIR RUNPOD_SHUTDOWN_MARKER
export NETWORK_VOLUME_ROOT RUNPOD_VOLUME_ROOT="${NETWORK_VOLUME_ROOT}"
export PROJECT_ROOT DATA_ROOT LOG_ROOT LIFECYCLE_ROOT RUNPOD_ROLE RUNPOD_CONFIG
export FIN_TS_DATASET_PROFILE
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
if [[ -z "${RUNPOD_REMOTE_SELECTION_PATH}" ]]; then
    echo "RUNPOD_REMOTE_SELECTION_PATH is required" >&2
    exit 2
fi
if [[ ! "${FIN_TS_DATASET_PROFILE}" =~ ^(tw_only|us_only_eodhd|us_tw_eodhd|us_tw_massive)$ ]]; then
    echo "FIN_TS_DATASET_PROFILE is unsupported" >&2
    exit 2
fi
if [[ "${FIN_TS_DATASET_PROFILE}" == "us_tw_massive" ]]; then
    echo "The Massive provider interface is reserved but not implemented" >&2
    exit 2
fi
if [[ "${FIN_TS_DATASET_PROFILE}" == *eodhd* \
    && ( -z "${EODHD_API_TOKEN:-}" || "${EODHD_API_TOKEN}" == *'{{ RUNPOD_SECRET_'* ) ]]; then
    echo "EODHD_API_TOKEN RunPod Secret is missing or was not resolved" >&2
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
    DATA_STAGING_ROOT RAW_STAGING PROCESSED_STAGING DOWNLOAD_MANIFEST_STAGING \
    DATASET_MANIFEST_STAGING REQUEST_LOG_STAGING RAW_FINAL PROCESSED_FINAL \
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

EXISTING_FINAL_FILES=0
for final_data_file in "${FINAL_DATA_FILES[@]}"; do
    if [[ -L "${final_data_file}" ]]; then
        echo "Selected dataset namespace contains a symlink: ${final_data_file}" >&2
        exit 2
    fi
    if [[ -e "${final_data_file}" ]]; then
        EXISTING_FINAL_FILES=$((EXISTING_FINAL_FILES + 1))
    fi
done
if [[ ${EXISTING_FINAL_FILES} -eq 0 ]]; then
    REUSE_READY_DATASET=0
elif [[ ${EXISTING_FINAL_FILES} -eq ${#FINAL_DATA_FILES[@]} ]]; then
    REUSE_READY_DATASET=1
else
    echo "Selected dataset namespace is incomplete; choose a new dataset revision" >&2
    exit 3
fi

read -r -a US_SYMBOLS <<< "${STAGE1_US_SYMBOLS}"
read -r -a US_ETF_SYMBOLS <<< "${STAGE1_US_ETF_SYMBOLS}"
mkdir -p "${PREP_DIR}" "${DATA_STAGING_ROOT}/raw" \
    "${DATA_STAGING_ROOT}/processed" "${DATA_STAGING_ROOT}/manifests" \
    "${LIFECYCLE_ROOT}/stage1" "${DATA_ROOT}/raw" "${DATA_ROOT}/processed" \
    "${DATA_ROOT}/manifests" "${API_CACHE_ROOT}" "${RUNPOD_SHUTDOWN_DIR}" "${HF_HOME}"
STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
PREP_SUCCEEDED=0
PREP_WAITING_FOR_PROVIDER=0
DOWNLOAD_ATTEMPTED=0

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
    if [[ ${DOWNLOAD_ATTEMPTED} -eq 1 && -f "${DOWNLOAD_PROGRESS}" ]]; then
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
    elif [[ ${PREP_WAITING_FOR_PROVIDER} -eq 1 ]]; then
        prep_state=waiting_for_provider
    elif [[ ${prep_exit_code} -eq 124 ]]; then
        prep_state=timed_out
    fi
    if [[ ${PREP_SUCCEEDED} -ne 1 ]]; then
        write_lifecycle_state "${prep_state}" "${prep_exit_code}" || true
    fi
    printf '{"started_at":"%s","ended_at":"%s","exit_code":%d,"state":"%s","log_path":"%s","requested_cpu_count":%d,"detected_cpu_count":%d,"effective_workers":%d}\n' \
        "${STARTED_AT}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${prep_exit_code}" \
        "${prep_state}" \
        "${PREP_LOG}" "${RUNPOD_REQUESTED_CPU_COUNT}" "${DETECTED_CPU_COUNT}" \
        "${FIN_TS_CPU_WORKERS}" > "${METADATA_PATH}"
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

DOWNLOAD_ARGUMENTS=(
    --profile "${FIN_TS_DATASET_PROFILE}"
    --start "${STAGE1_DATA_START}"
    --end "${STAGE1_DATA_END}"
    --output "${RAW_STAGING}"
    --manifest-root "${DATA_STAGING_ROOT}"
    --raw-cache-root "${API_CACHE_ROOT}"
    --progress-path "${DOWNLOAD_PROGRESS}"
    --dataset-request-sha256 "${RUNPOD_DATASET_REQUEST_SHA256}"
    --selection-id "${RUNPOD_SELECTION_ID}"
    --selection-sha256 "${RUNPOD_SELECTION_SHA256}"
    --launch-id "${LAUNCH_ID}"
    --max-api-calls "${STAGE1_MAX_API_CALLS}"
    --eodhd-qps "${STAGE1_EODHD_QPS}"
    --taiwan-qps "${STAGE1_TAIWAN_QPS}"
    --workers "${FIN_TS_CPU_WORKERS}"
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
DOWNLOAD_ATTEMPTED=1
set +e
"${POETRY_BIN}" run fin-ts-download "${DOWNLOAD_ARGUMENTS[@]}"
DOWNLOAD_EXIT_CODE=$?
set -e
if [[ ${DOWNLOAD_EXIT_CODE} -eq 75 ]]; then
    PREP_WAITING_FOR_PROVIDER=1
    printf 'Provider quota or temporary availability prevented completion. Progress is saved at %s. Rerun the same cpu prepare workflow after the provider permits requests again.\n' \
        "${DOWNLOAD_PROGRESS}" >&2
    exit 75
fi
if [[ ${DOWNLOAD_EXIT_CODE} -ne 0 ]]; then
    exit "${DOWNLOAD_EXIT_CODE}"
fi

"${POETRY_BIN}" run fin-ts-prepare \
    --input "${RAW_STAGING}" \
    --output "${PROCESSED_STAGING}" \
    --download-manifest "${DOWNLOAD_MANIFEST_STAGING}" \
    --dataset-manifest "${DATASET_MANIFEST_STAGING}" \
    --window-size 128 \
    --stride 5 \
    --sample-stride 1 \
    --alpha-horizons 3 4 5 6 7 8 9 10 11 12 13 14 \
    --target-horizon 5 \
    --diagnostic-horizons 1 20 \
    --flat-volatility-multiplier 0.25 \
    --max-abs-log-return 0.5 \
    --train-fraction 0.70 \
    --validation-fraction 0.15 \
    --purge-bars 20 \
    --embargo-bars 5 \
    --effective-embargo-bars 14 \
    --workers "${FIN_TS_CPU_WORKERS}"

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

for final_data_file in "${FINAL_DATA_FILES[@]}"; do
    if [[ -e "${final_data_file}" || -L "${final_data_file}" ]]; then
        echo "Refusing to overwrite an existing immutable dataset artifact: ${final_data_file}" >&2
        exit 3
    fi
done
mv "${RAW_STAGING}" "${RAW_FINAL}"
mv "${PROCESSED_STAGING}" "${PROCESSED_FINAL}"
mv "${REQUEST_LOG_STAGING}" "${REQUEST_LOG_FINAL}"
mv "${DOWNLOAD_MANIFEST_STAGING}" "${DOWNLOAD_MANIFEST_FINAL}"
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
