#!/usr/bin/env bash

# Recheck lifecycle manifests and artifact checksums from inside a mounted RunPod volume.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/runpod_paths.sh
source "${SCRIPT_DIR}/lib/runpod_paths.sh"

NETWORK_VOLUME_ROOT="${NETWORK_VOLUME_ROOT:-${RUNPOD_VOLUME_ROOT:-/runpod-volume}}"
PROJECT_ROOT="${PROJECT_ROOT:-${NETWORK_VOLUME_ROOT}/stock_forecasting}"
RUNPOD_PYTHON_BIN="${RUNPOD_PYTHON_BIN:-/usr/local/bin/python}"
LIFECYCLE_ROOT="${LIFECYCLE_ROOT:-${NETWORK_VOLUME_ROOT}/lifecycle}"
CODE_MARKER="${LIFECYCLE_ROOT}/stage1/code.json"
DATASET_MARKER="${LIFECYCLE_ROOT}/stage1/dataset.json"
RUNPOD_CONFIG="${RUNPOD_CONFIG:-configs/stage1_kronos_base_lora.yaml}"
RUNPOD_SELECTION_HELPER="${PROJECT_ROOT}/scripts/runpod_selection.py"
RUNPOD_REMOTE_SELECTION_PATH="${RUNPOD_REMOTE_SELECTION_PATH:-}"
RUNPOD_EXPECTED_VOLUME_ID="${RUNPOD_EXPECTED_VOLUME_ID:-}"
RUNPOD_VOLUME_ID="${RUNPOD_VOLUME_ID:-}"
MOUNT_ONLY=0
if [[ $# -gt 1 ]]; then
    echo "Usage: verify_runpod_mounted_readiness.sh [--mount-only]" >&2
    exit 2
fi
if [[ "${1:-}" == "--mount-only" ]]; then
    MOUNT_ONLY=1
elif [[ $# -ne 0 ]]; then
    echo "Usage: verify_runpod_mounted_readiness.sh [--mount-only]" >&2
    exit 2
fi

verify_network_volume_mountpoint() {
    if [[ ! -d "${NETWORK_VOLUME_ROOT}" || -L "${NETWORK_VOLUME_ROOT}" ]]; then
        echo "NETWORK_VOLUME_ROOT must be a real mounted directory, not a symlink" >&2
        return 2
    fi
    if command -v mountpoint >/dev/null 2>&1; then
        if ! mountpoint -q -- "${NETWORK_VOLUME_ROOT}"; then
            echo "NETWORK_VOLUME_ROOT is not an actual mount point: ${NETWORK_VOLUME_ROOT}" >&2
            return 2
        fi
        return 0
    fi
    if command -v findmnt >/dev/null 2>&1; then
        local target
        target="$(findmnt -rn -M "${NETWORK_VOLUME_ROOT}" -o TARGET 2>/dev/null || true)"
        if [[ "${target}" != "${NETWORK_VOLUME_ROOT}" ]]; then
            echo "findmnt could not prove NETWORK_VOLUME_ROOT is an exact mount point" >&2
            return 2
        fi
        return 0
    fi
    if [[ -r /proc/self/mountinfo ]] \
        && awk -v target="${NETWORK_VOLUME_ROOT}" '$5 == target { found=1 } END { exit !found }' \
            /proc/self/mountinfo; then
        return 0
    fi
    echo "Unable to prove NETWORK_VOLUME_ROOT is an exact mount point" >&2
    return 2
}

if [[ -z "${RUNPOD_POD_ID:-}" && "${RUNPOD_TEST_MODE:-0}" != "1" ]]; then
    echo "Mounted readiness verification must run inside a RunPod Pod" >&2
    exit 2
fi
runpod_validate_absolute_path "${NETWORK_VOLUME_ROOT}" NETWORK_VOLUME_ROOT
case "${NETWORK_VOLUME_ROOT}" in
    /workspace|/workspace/*)
        echo "NETWORK_VOLUME_ROOT must never use /workspace" >&2
        exit 2
        ;;
esac
if [[ ! "${RUNPOD_EXPECTED_VOLUME_ID}" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo "RUNPOD_EXPECTED_VOLUME_ID is missing or invalid" >&2
    exit 2
fi
if [[ ! "${RUNPOD_VOLUME_ID}" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo "RunPod did not provide a valid RUNPOD_VOLUME_ID" >&2
    exit 2
fi
if [[ "${RUNPOD_VOLUME_ID}" != "${RUNPOD_EXPECTED_VOLUME_ID}" ]]; then
    printf 'Mounted RunPod volume ID mismatch: expected=%s actual=%s\n' \
        "${RUNPOD_EXPECTED_VOLUME_ID}" "${RUNPOD_VOLUME_ID}" >&2
    exit 2
fi
verify_network_volume_mountpoint
if [[ ${MOUNT_ONLY} -eq 1 ]]; then
    printf 'Verified RunPod network volume mount: path=%s volume_id=%s\n' \
        "${NETWORK_VOLUME_ROOT}" "${RUNPOD_VOLUME_ID}"
    exit 0
fi
for path_name in PROJECT_ROOT LIFECYCLE_ROOT CODE_MARKER DATASET_MARKER \
    RUNPOD_SELECTION_HELPER RUNPOD_REMOTE_SELECTION_PATH; do
    runpod_validate_path_in_root \
        "${!path_name}" "${NETWORK_VOLUME_ROOT}" "${path_name}" NETWORK_VOLUME_ROOT
done
runpod_validate_absolute_path "${RUNPOD_PYTHON_BIN}" RUNPOD_PYTHON_BIN
if [[ ! -x "${RUNPOD_PYTHON_BIN}" ]]; then
    echo "RunPod image Python is unavailable" >&2
    exit 127
fi
if [[ "${RUNPOD_CONFIG}" == /workspace || "${RUNPOD_CONFIG}" == /workspace/* ]]; then
    echo "RUNPOD_CONFIG must never use /workspace" >&2
    exit 2
fi
if [[ "${RUNPOD_CONFIG}" == /* ]]; then
    CONFIG_PATH="${RUNPOD_CONFIG}"
else
    if [[ ! "${RUNPOD_CONFIG}" =~ ^[A-Za-z0-9._/-]+$ \
        || "/${RUNPOD_CONFIG}/" == *"/../"* \
        || "/${RUNPOD_CONFIG}/" == *"/./"* \
        || "${RUNPOD_CONFIG}" == *"//"* ]]; then
        echo "Relative RUNPOD_CONFIG is unsafe" >&2
        exit 2
    fi
    CONFIG_PATH="${PROJECT_ROOT}/${RUNPOD_CONFIG}"
fi
runpod_validate_path_in_root \
    "${CONFIG_PATH}" "${NETWORK_VOLUME_ROOT}" CONFIG_PATH NETWORK_VOLUME_ROOT
if [[ ! -f "${CONFIG_PATH}" || -L "${CONFIG_PATH}" ]]; then
    echo "RUNPOD_CONFIG is missing or is a symlink: ${CONFIG_PATH}" >&2
    exit 2
fi
if [[ ! -f "${RUNPOD_SELECTION_HELPER}" || -L "${RUNPOD_SELECTION_HELPER}" \
    || ! -f "${RUNPOD_REMOTE_SELECTION_PATH}" \
    || -L "${RUNPOD_REMOTE_SELECTION_PATH}" ]]; then
    echo "Mounted immutable training selection is unavailable" >&2
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
"${RUNPOD_PYTHON_BIN}" "${RUNPOD_SELECTION_HELPER}" verify-marker \
    --project-root "${PROJECT_ROOT}" \
    --selection "${RUNPOD_REMOTE_SELECTION_PATH}" \
    --marker "${DATASET_MARKER}"
"${RUNPOD_PYTHON_BIN}" "${SCRIPT_DIR}/runpod_readiness.py" check-dataset \
    --marker "${DATASET_MARKER}" \
    --code-marker "${CODE_MARKER}" \
    --stage-config "${CONFIG_PATH}" \
    --network-volume-root "${NETWORK_VOLUME_ROOT}"
