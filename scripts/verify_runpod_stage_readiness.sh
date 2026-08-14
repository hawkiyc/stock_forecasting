#!/usr/bin/env bash

# Verify source and data readiness through the network-volume S3 API before renting compute.
set -Eeuo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
S3_WRAPPER="${SCRIPT_DIR}/runpod_s3_project.sh"
READINESS_HELPER="${SCRIPT_DIR}/runpod_readiness.py"
CODE_MARKER_KEY="lifecycle/stage1/code.json"
DATASET_MARKER_KEY="lifecycle/stage1/dataset.json"

# shellcheck source=lib/runpod_project_env.sh
source "${SCRIPT_DIR}/lib/runpod_project_env.sh"
runpod_load_create_env "${LOCAL_PROJECT_ROOT}"
RUNPOD_CONFIG="${RUNPOD_CONFIG:-configs/stage1_kronos_base_lora.yaml}"

if [[ $# -ne 1 ]]; then
    echo "Usage: verify_runpod_stage_readiness.sh --code-only|--gpu" >&2
    exit 2
fi
case "$1" in
    --code-only) MODE=code-only ;;
    --gpu) MODE=gpu ;;
    *)
        echo "Usage: verify_runpod_stage_readiness.sh --code-only|--gpu" >&2
        exit 2
        ;;
esac

if [[ ! "${RUNPOD_NETWORK_VOLUME_ID:-}" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo "RUNPOD_NETWORK_VOLUME_ID is required" >&2
    exit 2
fi
if [[ ! -f "${S3_WRAPPER}" || ! -r "${S3_WRAPPER}" || ! -f "${READINESS_HELPER}" ]]; then
    echo "RunPod S3 or readiness helper is unavailable" >&2
    exit 127
fi
if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 is required for readiness verification" >&2
    exit 127
fi
if [[ ! "${RUNPOD_CONFIG}" =~ ^[A-Za-z0-9._/-]+$ \
    || "${RUNPOD_CONFIG}" == /* \
    || "/${RUNPOD_CONFIG}/" == *"/../"* \
    || "/${RUNPOD_CONFIG}/" == *"/./"* \
    || "${RUNPOD_CONFIG}" == *"//"* \
    || ! -f "${LOCAL_PROJECT_ROOT}/${RUNPOD_CONFIG}" ]]; then
    echo "RUNPOD_CONFIG must be an existing safe path relative to the local project" >&2
    exit 2
fi

CODE_JSON="$(bash "${S3_WRAPPER}" s3 cp \
    "s3://${RUNPOD_NETWORK_VOLUME_ID}/${CODE_MARKER_KEY}" - \
    --only-show-errors)"
printf '%s\n' "${CODE_JSON}" \
    | python3 "${READINESS_HELPER}" check-code \
        --marker - \
        --project-root "${LOCAL_PROJECT_ROOT}"

if [[ "${MODE}" == "code-only" ]]; then
    printf 'CPU preparation gate passed: uploaded code is ready.\n'
    exit 0
fi

CODE_RELEASE_DIGEST="$(printf '%s\n' "${CODE_JSON}" | python3 -c \
    'import json, sys; print(json.load(sys.stdin)["release_digest"])')"
DATASET_JSON="$(bash "${S3_WRAPPER}" s3 cp \
    "s3://${RUNPOD_NETWORK_VOLUME_ID}/${DATASET_MARKER_KEY}" - \
    --only-show-errors)"
printf '%s\n' "${DATASET_JSON}" \
    | python3 "${READINESS_HELPER}" check-dataset \
        --marker - \
        --expected-code-release-digest "${CODE_RELEASE_DIGEST}" \
        --stage-config "${LOCAL_PROJECT_ROOT}/${RUNPOD_CONFIG}"

RAW_EXPECTED_SIZE="$(printf '%s\n' "${DATASET_JSON}" | python3 -c \
    'import json, sys; print(json.load(sys.stdin)["raw"]["size_bytes"])')"
PROCESSED_EXPECTED_SIZE="$(printf '%s\n' "${DATASET_JSON}" | python3 -c \
    'import json, sys; print(json.load(sys.stdin)["processed"]["size_bytes"])')"
RAW_REMOTE_SIZE="$(bash "${S3_WRAPPER}" s3api head-object \
    --bucket "${RUNPOD_NETWORK_VOLUME_ID}" \
    --key data/raw/market.parquet \
    --query ContentLength \
    --output text)"
PROCESSED_REMOTE_SIZE="$(bash "${S3_WRAPPER}" s3api head-object \
    --bucket "${RUNPOD_NETWORK_VOLUME_ID}" \
    --key data/processed/windows.parquet \
    --query ContentLength \
    --output text)"
if [[ "${RAW_REMOTE_SIZE}" != "${RAW_EXPECTED_SIZE}" ]]; then
    echo "Raw dataset size does not match its readiness manifest" >&2
    exit 3
fi
if [[ "${PROCESSED_REMOTE_SIZE}" != "${PROCESSED_EXPECTED_SIZE}" ]]; then
    echo "Processed dataset size does not match its readiness manifest" >&2
    exit 3
fi
DATASET_MANIFEST_EXPECTED_SIZE="$(printf '%s\n' "${DATASET_JSON}" | python3 -c \
    'import json, sys; print(json.load(sys.stdin)["dataset_manifest"]["size_bytes"])')"
DOWNLOAD_MANIFEST_EXPECTED_SIZE="$(printf '%s\n' "${DATASET_JSON}" | python3 -c \
    'import json, sys; print(json.load(sys.stdin)["download_manifest"]["size_bytes"])')"
REQUEST_LOG_EXPECTED_SIZE="$(printf '%s\n' "${DATASET_JSON}" | python3 -c \
    'import json, sys; print(json.load(sys.stdin)["request_log"]["size_bytes"])')"
MODEL_MANIFEST_EXPECTED_SIZE="$(printf '%s\n' "${DATASET_JSON}" | python3 -c \
    'import json, sys; print(json.load(sys.stdin)["model_manifest"]["size_bytes"])')"

verify_remote_size() {
    local object_key="$1"
    local expected_size="$2"
    local label="$3"
    local remote_size
    remote_size="$(bash "${S3_WRAPPER}" s3api head-object \
        --bucket "${RUNPOD_NETWORK_VOLUME_ID}" \
        --key "${object_key}" \
        --query ContentLength \
        --output text)"
    if [[ "${remote_size}" != "${expected_size}" ]]; then
        printf '%s size does not match its readiness manifest\n' "${label}" >&2
        exit 3
    fi
}

verify_remote_size data/dataset-manifest.json \
    "${DATASET_MANIFEST_EXPECTED_SIZE}" "Dataset manifest"
verify_remote_size data/download-manifest.json \
    "${DOWNLOAD_MANIFEST_EXPECTED_SIZE}" "Download manifest"
verify_remote_size data/manifests/api-request-log.jsonl \
    "${REQUEST_LOG_EXPECTED_SIZE}" "API request log"
verify_remote_size cache/hf-models.json \
    "${MODEL_MANIFEST_EXPECTED_SIZE}" "Hugging Face model manifest"

printf 'GPU creation gate passed: code, dataset, and offline model cache are ready.\n'
