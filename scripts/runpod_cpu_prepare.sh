#!/usr/bin/env bash

# Prepare the complete numerical dataset and model cache on a CPU Pod, then terminate it.
set -Eeuo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/runpod_paths.sh
source "${SCRIPT_DIR}/lib/runpod_paths.sh"

NETWORK_VOLUME_ROOT="${NETWORK_VOLUME_ROOT:-${RUNPOD_VOLUME_ROOT:-/runpod-volume}}"
PROJECT_ROOT="${PROJECT_ROOT:-${NETWORK_VOLUME_ROOT}/ts_multimodal_LLM}"
DATA_ROOT="${DATA_ROOT:-${NETWORK_VOLUME_ROOT}/data}"
LOG_ROOT="${LOG_ROOT:-${NETWORK_VOLUME_ROOT}/logs}"
LIFECYCLE_ROOT="${LIFECYCLE_ROOT:-${NETWORK_VOLUME_ROOT}/lifecycle}"
POETRY_VERSION="${POETRY_VERSION:-2.4.0}"
POETRY_BIN="${RUNPOD_POETRY_BIN:-${NETWORK_VOLUME_ROOT}/tools/poetry/${POETRY_VERSION}/bin/poetry}"
RUNPOD_PYTHON_BIN="${RUNPOD_PYTHON_BIN:-/usr/local/bin/python}"
RUNPOD_CONFIG="${RUNPOD_CONFIG:-configs/stage1_kronos_base_lora.yaml}"
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
FIN_TS_DATASET_PROFILE="${FIN_TS_DATASET_PROFILE:-us_tw_eodhd}"
STAGE1_US_SYMBOLS="${STAGE1_US_SYMBOLS:-}"
STAGE1_US_ETF_SYMBOLS="${STAGE1_US_ETF_SYMBOLS:-}"
STAGE1_SYMBOL_LIMIT="${STAGE1_SYMBOL_LIMIT:-}"
STAGE1_DATA_START="${STAGE1_DATA_START:-2010-01-01}"
STAGE1_DATA_END="${STAGE1_DATA_END:-2026-07-27}"
STAGE1_MAX_API_CALLS="${STAGE1_MAX_API_CALLS:-90000}"
STAGE1_EODHD_QPS="${STAGE1_EODHD_QPS:-5}"
STAGE1_TAIWAN_QPS="${STAGE1_TAIWAN_QPS:-0.5}"
RUNPOD_PYTEST_WORKERS="${RUNPOD_PYTEST_WORKERS:-8}"
RUNPOD_PYTEST_THREADS_PER_WORKER="${RUNPOD_PYTEST_THREADS_PER_WORKER:-1}"
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
    METADATA_PATH RUNPOD_SHUTDOWN_DIR RUNPOD_SHUTDOWN_MARKER HF_HOME \
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

read -r -a US_SYMBOLS <<< "${STAGE1_US_SYMBOLS}"
read -r -a US_ETF_SYMBOLS <<< "${STAGE1_US_ETF_SYMBOLS}"
mkdir -p "${PREP_DIR}" "${DATA_STAGING_ROOT}/raw" \
    "${DATA_STAGING_ROOT}/processed" "${DATA_STAGING_ROOT}/manifests" \
    "${LIFECYCLE_ROOT}/stage1" "${DATA_ROOT}/raw" "${DATA_ROOT}/processed" \
    "${DATA_ROOT}/manifests" "${API_CACHE_ROOT}" "${RUNPOD_SHUTDOWN_DIR}" "${HF_HOME}"
STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
PREP_SUCCEEDED=0

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
    "${command[@]}"
}

finish_cpu_prep() {
    local prep_exit_code=$?
    trap - EXIT INT TERM
    if [[ ${PREP_SUCCEEDED} -ne 1 ]]; then
        write_lifecycle_state failed "${prep_exit_code}" || true
    fi
    printf '{"started_at":"%s","ended_at":"%s","exit_code":%d,"state":"%s","log_path":"%s"}\n' \
        "${STARTED_AT}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${prep_exit_code}" \
        "$([[ ${PREP_SUCCEEDED} -eq 1 ]] && printf ready || printf failed)" \
        "${PREP_LOG}" > "${METADATA_PATH}"
    bash "${SCRIPT_DIR}/stop_runpod_pod.sh" || true
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

DOWNLOAD_ARGUMENTS=(
    --profile "${FIN_TS_DATASET_PROFILE}"
    --start "${STAGE1_DATA_START}"
    --end "${STAGE1_DATA_END}"
    --output "${RAW_STAGING}"
    --manifest-root "${DATA_STAGING_ROOT}"
    --raw-cache-root "${API_CACHE_ROOT}"
    --max-api-calls "${STAGE1_MAX_API_CALLS}"
    --eodhd-qps "${STAGE1_EODHD_QPS}"
    --taiwan-qps "${STAGE1_TAIWAN_QPS}"
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
"${POETRY_BIN}" run fin-ts-download "${DOWNLOAD_ARGUMENTS[@]}"

RUNPOD_ROLE=cpu-prep RUNPOD_CONFIG="${RUNPOD_CONFIG}" \
    bash "${SCRIPT_DIR}/prefetch_hf_models.sh" --smoke-time-series-backbone

"${POETRY_BIN}" run fin-ts-prepare \
    --input "${RAW_STAGING}" \
    --output "${PROCESSED_STAGING}" \
    --download-manifest "${DOWNLOAD_MANIFEST_STAGING}" \
    --dataset-manifest "${DATASET_MANIFEST_STAGING}" \
    --window-size 128 \
    --stride 5 \
    --target-horizon 5 \
    --diagnostic-horizons 1 20 \
    --flat-volatility-multiplier 0.25 \
    --max-abs-log-return 0.5 \
    --train-fraction 0.70 \
    --validation-fraction 0.15 \
    --purge-bars 20 \
    --embargo-bars 5

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

PREP_SUCCEEDED=1
printf 'CPU preparation completed; dataset profile=%s; readiness manifest: %s\n' \
    "${FIN_TS_DATASET_PROFILE}" "${DATASET_MARKER}"
