#!/usr/bin/env bash

# Create a short-lived CPU preparation Pod after uploaded source passes its readiness gate.
set -Eeuo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=lib/runpod_paths.sh
source "${SCRIPT_DIR}/lib/runpod_paths.sh"
# shellcheck source=lib/runpod_project_env.sh
source "${SCRIPT_DIR}/lib/runpod_project_env.sh"
runpod_load_create_env "${LOCAL_PROJECT_ROOT}"
# shellcheck source=lib/runpod_selection.sh
source "${SCRIPT_DIR}/lib/runpod_selection.sh"
runpod_load_active_selection "${LOCAL_PROJECT_ROOT}"

RUNPOD_NETWORK_VOLUME_ID="${RUNPOD_NETWORK_VOLUME_ID:-}"
RUNPOD_VOLUME_MOUNT_PATH="${RUNPOD_VOLUME_MOUNT_PATH:-/runpod-volume}"
PROJECT_ROOT="${PROJECT_ROOT:-${RUNPOD_VOLUME_MOUNT_PATH}/stock_forecasting}"
RUNPOD_POD_NAME="${RUNPOD_CPU_POD_NAME:-fin-ts-stage1-cpu-prep}"
RUNPOD_IMAGE="${RUNPOD_IMAGE:-runpod/pytorch:1.0.7-cu1281-torch291-ubuntu2404}"
RUNPOD_EXPECTED_TORCH_VERSION="${RUNPOD_EXPECTED_TORCH_VERSION:-2.9.1+cu128}"
RUNPOD_EXPECTED_CUDA_PREFIX="${RUNPOD_EXPECTED_CUDA_PREFIX:-12.8}"
RUNPOD_EXPECTED_UBUNTU_VERSION="${RUNPOD_EXPECTED_UBUNTU_VERSION:-24.04}"
RUNPOD_HF_SECRET_NAME="${RUNPOD_HF_SECRET_NAME:-huggingface_token}"
RUNPOD_EODHD_SECRET_NAME="${RUNPOD_EODHD_SECRET_NAME:-eodhd_api_token}"
RUNPOD_CLOUD_TYPE="${RUNPOD_CLOUD_TYPE:-SECURE}"
RUNPOD_DATACENTER_ID="${RUNPOD_DATACENTER_ID:-EU-RO-1}"
RUNPOD_CONTAINER_DISK_GB="${RUNPOD_CPU_CONTAINER_DISK_GB:-}"
RUNPOD_API_BASE_URL="${RUNPOD_API_BASE_URL:-https://rest.runpod.io/v1}"
RUNPOD_CPU_FLAVOR_ID="${RUNPOD_CPU_FLAVOR_ID:-cpu3g}"
RUNPOD_CPU_VCPU_COUNT="${RUNPOD_CPU_VCPU_COUNT:-8}"
RUNPOD_CPU_MAX_RUNTIME_SECONDS="${RUNPOD_CPU_MAX_RUNTIME_SECONDS:-21600}"
RUNPOD_CPU_HARD_LIMIT_SECONDS="${RUNPOD_CPU_HARD_LIMIT_SECONDS:-25200}"
RUNPOD_GUARD_LOG_DIR="${RUNPOD_GUARD_LOG_DIR:-${HOME:-/tmp}/.local/state/runpod-guards}"
RUNPOD_GUARD_LAUNCHER="${SCRIPT_DIR}/launch_runpod_guard.sh"
RUNPOD_CPU_TMUX_WORKFLOW=cpu-prepare
if [[ "${RUNPOD_STAGE}" == "stage2" ]]; then
    RUNPOD_CPU_TMUX_WORKFLOW=cpu-finalize
fi

if [[ -n "${RUNPOD_POD_ID:-}" && "${RUNPOD_TEST_MODE:-0}" != "1" ]]; then
    echo "create_runpod_cpu_pod.sh must run outside the RunPod Pod" >&2
    exit 2
fi
if [[ ! "${RUNPOD_NETWORK_VOLUME_ID}" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo "RUNPOD_NETWORK_VOLUME_ID is required and must use safe characters" >&2
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
if [[ "${PROJECT_ROOT}" == "${RUNPOD_VOLUME_MOUNT_PATH}" ]]; then
    echo "PROJECT_ROOT must be a child of RUNPOD_VOLUME_MOUNT_PATH" >&2
    exit 2
fi
if [[ ! "${RUNPOD_VOLUME_MOUNT_PATH}" =~ ^/[A-Za-z0-9._/-]+$ \
    || ! "${PROJECT_ROOT}" =~ ^/[A-Za-z0-9._/-]+$ ]]; then
    echo "Volume and project paths contain unsupported characters" >&2
    exit 2
fi
if [[ ! "${RUNPOD_DATACENTER_ID}" =~ ^[A-Z0-9]+(-[A-Z0-9]+)+$ ]]; then
    echo "RUNPOD_DATACENTER_ID has an invalid format" >&2
    exit 2
fi
if [[ ! "${RUNPOD_API_BASE_URL}" =~ ^https://[A-Za-z0-9._:-]+(/[A-Za-z0-9._/-]*)?$ ]]; then
    echo "RUNPOD_API_BASE_URL must be an HTTPS URL" >&2
    exit 2
fi
case "${RUNPOD_CPU_FLAVOR_ID}" in
    cpu3c|cpu3g|cpu3m|cpu5c|cpu5g|cpu5m) ;;
    *)
        echo "RUNPOD_CPU_FLAVOR_ID must be one of: cpu3c, cpu3g, cpu3m, cpu5c, cpu5g, cpu5m" >&2
        exit 2
        ;;
esac
if [[ ! "${RUNPOD_CPU_VCPU_COUNT}" =~ ^[1-9][0-9]*$ \
    || "${RUNPOD_CPU_VCPU_COUNT}" -gt 32 ]]; then
    echo "RUNPOD_CPU_VCPU_COUNT must be an integer from 1 through 32" >&2
    exit 2
fi
case "${RUNPOD_CPU_FLAVOR_ID}" in
    cpu3c|cpu3g|cpu3m) CONTAINER_DISK_GB_PER_VCPU=10 ;;
    cpu5c|cpu5g|cpu5m) CONTAINER_DISK_GB_PER_VCPU=15 ;;
esac
MAX_CONTAINER_DISK_GB=$((RUNPOD_CPU_VCPU_COUNT * CONTAINER_DISK_GB_PER_VCPU))
if [[ -z "${RUNPOD_CONTAINER_DISK_GB}" ]]; then
    RUNPOD_CONTAINER_DISK_GB=30
    if [[ "${RUNPOD_CONTAINER_DISK_GB}" -gt "${MAX_CONTAINER_DISK_GB}" ]]; then
        RUNPOD_CONTAINER_DISK_GB="${MAX_CONTAINER_DISK_GB}"
    fi
elif [[ ! "${RUNPOD_CONTAINER_DISK_GB}" =~ ^[1-9][0-9]*$ \
    || "${RUNPOD_CONTAINER_DISK_GB}" -gt "${MAX_CONTAINER_DISK_GB}" ]]; then
    printf 'RUNPOD_CPU_CONTAINER_DISK_GB must be an integer from 1 through %d for %s with %s vCPUs\n' \
        "${MAX_CONTAINER_DISK_GB}" "${RUNPOD_CPU_FLAVOR_ID}" \
        "${RUNPOD_CPU_VCPU_COUNT}" >&2
    exit 2
fi
if [[ ! "${RUNPOD_CPU_MAX_RUNTIME_SECONDS}" =~ ^[1-9][0-9]*$ \
    || ! "${RUNPOD_CPU_HARD_LIMIT_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "CPU runtime limits must be positive integers" >&2
    exit 2
fi
if [[ "${RUNPOD_CPU_HARD_LIMIT_SECONDS}" -le "${RUNPOD_CPU_MAX_RUNTIME_SECONDS}" ]]; then
    echo "RUNPOD_CPU_HARD_LIMIT_SECONDS must exceed RUNPOD_CPU_MAX_RUNTIME_SECONDS" >&2
    exit 2
fi
if [[ ! "${RUNPOD_IMAGE}" =~ ^[A-Za-z0-9._/:+-]+$ ]]; then
    echo "RunPod image contains unsupported characters" >&2
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
if [[ ! "${RUNPOD_HF_SECRET_NAME}" =~ ^[A-Za-z0-9_-]+$ \
    || ! "${RUNPOD_EODHD_SECRET_NAME}" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo "RunPod secret names contain unsupported characters" >&2
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
for symbol_list in "${STAGE1_US_SYMBOLS}" "${STAGE1_US_ETF_SYMBOLS}"; do
    if [[ -n "${symbol_list}" \
        && ! "${symbol_list}" =~ ^[A-Za-z0-9.^_-]+([[:space:]]+[A-Za-z0-9.^_-]+)*$ ]]; then
        echo "Optional US symbol lists contain unsupported characters" >&2
        exit 2
    fi
done
if [[ ! "${STAGE1_DATA_START}" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ \
    || ! "${STAGE1_DATA_END}" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
    echo "Stage 1 dates must use YYYY-MM-DD" >&2
    exit 2
fi
if [[ -n "${STAGE1_SYMBOL_LIMIT}" && ! "${STAGE1_SYMBOL_LIMIT}" =~ ^[1-9][0-9]*$ ]]; then
    echo "STAGE1_SYMBOL_LIMIT must be empty or a positive integer" >&2
    exit 2
fi
if [[ ! "${STAGE1_MAX_API_CALLS}" =~ ^[1-9][0-9]*$ \
    || ! "${STAGE1_EODHD_QPS}" =~ ^[0-9]+([.][0-9]+)?$ \
    || ! "${STAGE1_TAIWAN_QPS}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "API budget and QPS settings are invalid" >&2
    exit 2
fi
if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 is required on the local control machine" >&2
    exit 127
fi

if [[ "${RUNPOD_TEST_READINESS_READY:-0}" == "1" ]]; then
    if [[ "${RUNPOD_TEST_MODE:-0}" != "1" ]]; then
        echo "RUNPOD_TEST_READINESS_READY is allowed only in test mode" >&2
        exit 2
    fi
else
    bash "${SCRIPT_DIR}/verify_runpod_stage_readiness.sh" --code-only
fi

REMOTE_SELECTION_MOUNT_PATH="${RUNPOD_VOLUME_MOUNT_PATH}/${RUNPOD_REMOTE_SELECTION_RELATIVE_PATH}"
DATA_ROOT="${RUNPOD_VOLUME_MOUNT_PATH}/datasets/${RUNPOD_DATASET_REQUEST_SHA256}"
if [[ "${RUNPOD_TEST_MODE:-0}" != "1" ]]; then
    bash "${SCRIPT_DIR}/runpod_s3_project.sh" s3 cp \
        "${RUNPOD_SELECTION_FILE}" \
        "s3://${RUNPOD_NETWORK_VOLUME_ID}/${RUNPOD_REMOTE_SELECTION_RELATIVE_PATH}" \
        --only-show-errors
    LOCAL_SELECTION_FILE_SHA256="$(python3 -c \
        'import hashlib, pathlib, sys; print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())' \
        "${RUNPOD_SELECTION_FILE}")"
    REMOTE_SELECTION_FILE_SHA256="$(bash "${SCRIPT_DIR}/runpod_s3_project.sh" s3 cp \
        "s3://${RUNPOD_NETWORK_VOLUME_ID}/${RUNPOD_REMOTE_SELECTION_RELATIVE_PATH}" - \
        --only-show-errors \
        | python3 -c 'import hashlib, sys; print(hashlib.sha256(sys.stdin.buffer.read()).hexdigest())')"
    if [[ "${REMOTE_SELECTION_FILE_SHA256}" != "${LOCAL_SELECTION_FILE_SHA256}" ]]; then
        echo "Uploaded training selection failed its S3 integrity check" >&2
        exit 3
    fi
fi

HF_TOKEN_REFERENCE="{{ RUNPOD_SECRET_${RUNPOD_HF_SECRET_NAME} }}"
EODHD_TOKEN_REFERENCE=""
if [[ "${FIN_TS_DATASET_PROFILE}" == *eodhd* ]]; then
    EODHD_TOKEN_REFERENCE="{{ RUNPOD_SECRET_${RUNPOD_EODHD_SECRET_NAME} }}"
fi
POD_ENV_JSON="$(printf \
    '{"NETWORK_VOLUME_ROOT":"%s","RUNPOD_VOLUME_ROOT":"%s","RUNPOD_EXPECTED_VOLUME_ID":"%s","RUNPOD_REQUESTED_CPU_COUNT":"%s","PROJECT_ROOT":"%s","DATA_ROOT":"%s","RUNPOD_CONFIG":"%s","RUNPOD_STAGE":"%s","RUNPOD_STAGE_CONFIG_SHA256":"%s","RUNPOD_SELECTION_ID":"%s","RUNPOD_SELECTION_SHA256":"%s","RUNPOD_DATASET_REQUEST_SHA256":"%s","RUNPOD_REMOTE_SELECTION_PATH":"%s","RUNPOD_ROLE":"cpu-prep","RUNPOD_CPU_MAX_RUNTIME_SECONDS":"%s","RUNPOD_SHUTDOWN_ACTION":"terminate","RUNPOD_IMAGE":"%s","RUNPOD_EXPECTED_TORCH_VERSION":"%s","RUNPOD_EXPECTED_CUDA_PREFIX":"%s","RUNPOD_EXPECTED_UBUNTU_VERSION":"%s","HF_TOKEN":"%s","EODHD_API_TOKEN":"%s","FIN_TS_DATASET_PROFILE":"%s","STAGE1_US_SYMBOLS":"%s","STAGE1_US_ETF_SYMBOLS":"%s","STAGE1_SYMBOL_LIMIT":"%s","STAGE1_DATA_START":"%s","STAGE1_DATA_END":"%s","STAGE1_MAX_API_CALLS":"%s","STAGE1_EODHD_QPS":"%s","STAGE1_TAIWAN_QPS":"%s"}' \
    "${RUNPOD_VOLUME_MOUNT_PATH}" "${RUNPOD_VOLUME_MOUNT_PATH}" \
    "${RUNPOD_NETWORK_VOLUME_ID}" "${RUNPOD_CPU_VCPU_COUNT}" \
    "${PROJECT_ROOT}" "${DATA_ROOT}" "${RUNPOD_CONFIG}" "${RUNPOD_STAGE}" \
    "${RUNPOD_STAGE_CONFIG_SHA256}" "${RUNPOD_SELECTION_ID}" \
    "${RUNPOD_SELECTION_SHA256}" "${RUNPOD_DATASET_REQUEST_SHA256}" \
    "${REMOTE_SELECTION_MOUNT_PATH}" "${RUNPOD_CPU_MAX_RUNTIME_SECONDS}" \
    "${RUNPOD_IMAGE}" "${RUNPOD_EXPECTED_TORCH_VERSION}" \
    "${RUNPOD_EXPECTED_CUDA_PREFIX}" "${RUNPOD_EXPECTED_UBUNTU_VERSION}" \
    "${HF_TOKEN_REFERENCE}" "${EODHD_TOKEN_REFERENCE}" "${FIN_TS_DATASET_PROFILE}" \
    "${STAGE1_US_SYMBOLS}" "${STAGE1_US_ETF_SYMBOLS}" "${STAGE1_SYMBOL_LIMIT}" \
    "${STAGE1_DATA_START}" "${STAGE1_DATA_END}" "${STAGE1_MAX_API_CALLS}" \
    "${STAGE1_EODHD_QPS}" "${STAGE1_TAIWAN_QPS}")"

POD_CREATE_JSON="$(
    POD_ENV_JSON="${POD_ENV_JSON}" \
    RUNPOD_CLOUD_TYPE="${RUNPOD_CLOUD_TYPE}" \
    RUNPOD_CONTAINER_DISK_GB="${RUNPOD_CONTAINER_DISK_GB}" \
    RUNPOD_CPU_FLAVOR_ID="${RUNPOD_CPU_FLAVOR_ID}" \
    RUNPOD_CPU_VCPU_COUNT="${RUNPOD_CPU_VCPU_COUNT}" \
    RUNPOD_DATACENTER_ID="${RUNPOD_DATACENTER_ID}" \
    RUNPOD_IMAGE="${RUNPOD_IMAGE}" \
    RUNPOD_NETWORK_VOLUME_ID="${RUNPOD_NETWORK_VOLUME_ID}" \
    RUNPOD_POD_NAME="${RUNPOD_POD_NAME}" \
    RUNPOD_VOLUME_MOUNT_PATH="${RUNPOD_VOLUME_MOUNT_PATH}" \
    python3 -c '
import json
import os

payload = {
    "cloudType": os.environ["RUNPOD_CLOUD_TYPE"],
    "computeType": "CPU",
    "containerDiskInGb": int(os.environ["RUNPOD_CONTAINER_DISK_GB"]),
    "cpuFlavorIds": [os.environ["RUNPOD_CPU_FLAVOR_ID"]],
    "cpuFlavorPriority": "custom",
    "dataCenterIds": [os.environ["RUNPOD_DATACENTER_ID"]],
    "env": json.loads(os.environ["POD_ENV_JSON"]),
    "imageName": os.environ["RUNPOD_IMAGE"],
    "name": os.environ["RUNPOD_POD_NAME"],
    "networkVolumeId": os.environ["RUNPOD_NETWORK_VOLUME_ID"],
    "vcpuCount": int(os.environ["RUNPOD_CPU_VCPU_COUNT"]),
    "volumeMountPath": os.environ["RUNPOD_VOLUME_MOUNT_PATH"],
}
print(json.dumps(payload, separators=(",", ":")))
'
)"

if [[ "${RUNPOD_CREATE_DRY_RUN:-0}" == "1" || "${RUNPOD_TEST_MODE:-0}" == "1" ]]; then
    printf 'DRY RUN: POST %s/pods --name %q --compute-type cpu --cloud-type %q --image-name %q --cpu-flavor-id %q --vcpu-count %q --data-center-ids %q --container-disk-in-gb %q --network-volume-id %q --volume-mount-path %q --env %q\n' \
        "${RUNPOD_API_BASE_URL%/}" "${RUNPOD_POD_NAME}" \
        "${RUNPOD_CLOUD_TYPE}" "${RUNPOD_IMAGE}" \
        "${RUNPOD_CPU_FLAVOR_ID}" "${RUNPOD_CPU_VCPU_COUNT}" \
        "${RUNPOD_DATACENTER_ID}" "${RUNPOD_CONTAINER_DISK_GB}" \
        "${RUNPOD_NETWORK_VOLUME_ID}" "${RUNPOD_VOLUME_MOUNT_PATH}" \
        "${POD_ENV_JSON}"
    exit 0
fi
runpod_load_project_env_key "${LOCAL_PROJECT_ROOT}" RUNPOD_API_KEY required
if ! command -v curl >/dev/null 2>&1; then
    echo "curl is required on the local control machine" >&2
    exit 127
fi
if [[ ! -r "${RUNPOD_GUARD_LAUNCHER}" ]]; then
    echo "External guard launcher is unavailable: ${RUNPOD_GUARD_LAUNCHER}" >&2
    exit 127
fi

set +e
CREATE_RESPONSE="$({
    printf 'header = "Authorization: Bearer %s"\n' "${RUNPOD_API_KEY}"
    printf 'header = "Content-Type: application/json"\n'
} | curl --config - \
    --silent --show-error \
    --request POST \
    --url "${RUNPOD_API_BASE_URL%/}/pods" \
    --connect-timeout 30 \
    --max-time 120 \
    --data "${POD_CREATE_JSON}" \
    --write-out '%{http_code}')"
CURL_EXIT_CODE=$?
set -e
if [[ ${CURL_EXIT_CODE} -ne 0 || ${#CREATE_RESPONSE} -lt 3 ]]; then
    echo "RunPod CPU Pod creation request failed" >&2
    exit 4
fi
CREATE_HTTP_CODE="${CREATE_RESPONSE: -3}"
CREATE_BODY="${CREATE_RESPONSE:0:${#CREATE_RESPONSE}-3}"
if [[ ! "${CREATE_HTTP_CODE}" =~ ^2[0-9][0-9]$ ]]; then
    printf 'RunPod CPU Pod creation returned HTTP %s: %s\n' \
        "${CREATE_HTTP_CODE}" "${CREATE_BODY}" >&2
    exit 4
fi
POD_ID="$(printf '%s' "${CREATE_BODY}" | python3 -c \
    'import json, sys; print(json.load(sys.stdin).get("id", ""))')"
if [[ ! "${POD_ID}" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo "Unable to parse a safe CPU Pod ID; no external guard was armed" >&2
    exit 3
fi

mkdir -p "${RUNPOD_GUARD_LOG_DIR}"
GUARD_LOG="${RUNPOD_GUARD_LOG_DIR}/${POD_ID}.log"
GUARD_PID="$(RUNPOD_GUARD_VOLUME_ROOT="${RUNPOD_VOLUME_MOUNT_PATH}" \
    bash "${RUNPOD_GUARD_LAUNCHER}" \
    "${POD_ID}" "${RUNPOD_CPU_HARD_LIMIT_SECONDS}" \
    lifecycle/stage1/dataset.json,lifecycle/stage1/mixed-finalization.json \
    "${GUARD_LOG}")"

printf 'Created CPU preparation Pod: %s\n' "${POD_ID}"
printf 'External guard PID: %s\n' "${GUARD_PID}"
printf 'External guard log: %s\n' "${GUARD_LOG}"
printf 'External guard readiness: %s\n' "${GUARD_LOG%.log}.ready.json"
printf 'After SSH login, run: bash scripts/runpod_tmux_launch.sh %s\n' \
    "${RUNPOD_CPU_TMUX_WORKFLOW}"
