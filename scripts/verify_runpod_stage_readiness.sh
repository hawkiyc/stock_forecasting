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
# shellcheck source=lib/runpod_selection.sh
source "${SCRIPT_DIR}/lib/runpod_selection.sh"
runpod_load_active_selection "${LOCAL_PROJECT_ROOT}"

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
    | python3 "${SCRIPT_DIR}/runpod_selection.py" verify-marker \
        --project-root "${LOCAL_PROJECT_ROOT}" \
        --selection "${RUNPOD_SELECTION_FILE}" \
        --marker -
REMOTE_SELECTION_JSON="$(bash "${S3_WRAPPER}" s3 cp \
    "s3://${RUNPOD_NETWORK_VOLUME_ID}/${RUNPOD_REMOTE_SELECTION_RELATIVE_PATH}" - \
    --only-show-errors)"
printf '%s\n' "${REMOTE_SELECTION_JSON}" \
    | python3 "${SCRIPT_DIR}/runpod_selection.py" verify-selection-copy \
        --project-root "${LOCAL_PROJECT_ROOT}" \
        --selection "${RUNPOD_SELECTION_FILE}" \
        --candidate -
printf '%s\n' "${DATASET_JSON}" \
    | python3 "${READINESS_HELPER}" check-dataset \
        --marker - \
        --expected-code-release-digest "${CODE_RELEASE_DIGEST}" \
        --stage-config "${LOCAL_PROJECT_ROOT}/${RUNPOD_CONFIG}"

verify_remote_size() {
    local artifact_name="$1"
    local label="$2"
    local object_key=""
    local expected_size=""
    local remote_size
    object_key="$(printf '%s\n' "${DATASET_JSON}" | python3 -c \
        'import json, sys; print(json.load(sys.stdin)[sys.argv[1]]["relative_path"])' \
        "${artifact_name}")"
    expected_size="$(printf '%s\n' "${DATASET_JSON}" | python3 -c \
        'import json, sys; print(json.load(sys.stdin)[sys.argv[1]]["size_bytes"])' \
        "${artifact_name}")"
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

verify_remote_size raw "Raw dataset"
verify_remote_size processed "Processed dataset"
verify_remote_size dataset_manifest "Dataset manifest"
verify_remote_size download_manifest "Download manifest"
verify_remote_size request_log "API request log"
verify_remote_size model_manifest "Hugging Face model manifest"

printf 'GPU creation gate passed: code, dataset, and offline model cache are ready.\n'
