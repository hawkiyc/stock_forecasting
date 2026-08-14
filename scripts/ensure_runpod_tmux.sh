#!/usr/bin/env bash

# Ensure tmux exists without making the package installation depend on the SSH session.
set -Eeuo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/runpod_paths.sh
source "${SCRIPT_DIR}/lib/runpod_paths.sh"

NETWORK_VOLUME_ROOT="${NETWORK_VOLUME_ROOT:-${RUNPOD_VOLUME_ROOT:-/runpod-volume}}"
LOG_ROOT="${LOG_ROOT:-${NETWORK_VOLUME_ROOT}/logs}"
POD_KEY="${RUNPOD_POD_ID:-test-pod}"
INSTALL_DIR="${LOG_ROOT}/bootstrap/${POD_KEY}/tmux"
INSTALL_LOG="${INSTALL_DIR}/install.log"
INSTALL_STATUS="${INSTALL_DIR}/status.json"
INSTALL_PID_FILE="${INSTALL_DIR}/installer.pid"
WAIT_SECONDS="${RUNPOD_TMUX_INSTALL_WAIT_SECONDS:-600}"

if [[ -z "${RUNPOD_POD_ID:-}" && "${RUNPOD_TEST_MODE:-0}" != "1" ]]; then
    echo "tmux setup must run inside a RunPod Pod" >&2
    exit 2
fi
runpod_validate_absolute_path "${NETWORK_VOLUME_ROOT}" NETWORK_VOLUME_ROOT
case "${NETWORK_VOLUME_ROOT}" in
    /workspace|/workspace/*)
        echo "NETWORK_VOLUME_ROOT must never use /workspace" >&2
        exit 2
        ;;
esac
runpod_validate_path_in_root \
    "${LOG_ROOT}" "${NETWORK_VOLUME_ROOT}" LOG_ROOT NETWORK_VOLUME_ROOT
if [[ ! "${POD_KEY}" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo "RUNPOD_POD_ID must be a canonical single path component" >&2
    exit 2
fi
for path_name in INSTALL_DIR INSTALL_LOG INSTALL_STATUS INSTALL_PID_FILE; do
    runpod_validate_path_in_root \
        "${!path_name}" "${LOG_ROOT}" "${path_name}" LOG_ROOT
done
if [[ "${LOG_ROOT}" == "${NETWORK_VOLUME_ROOT}" ]]; then
    echo "LOG_ROOT must be a child of NETWORK_VOLUME_ROOT" >&2
    exit 2
fi
if [[ ! "${WAIT_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "RUNPOD_TMUX_INSTALL_WAIT_SECONDS must be a positive integer" >&2
    exit 2
fi

if command -v tmux >/dev/null 2>&1; then
    tmux -V
    exit 0
fi
if [[ "${RUNPOD_TMUX_INSTALL_DRY_RUN:-0}" == "1" ]]; then
    printf 'Would install tmux and persist the installer log at %s\n' "${INSTALL_LOG}"
    exit 0
fi
if [[ ! -f "${SCRIPT_DIR}/install_runpod_tmux_worker.sh" ]]; then
    echo "tmux installer worker is missing" >&2
    exit 127
fi

mkdir -p "${INSTALL_DIR}"
INSTALLER_RUNNING=0
if [[ -f "${INSTALL_PID_FILE}" ]]; then
    INSTALL_PID="$(tr -d '[:space:]' < "${INSTALL_PID_FILE}")"
    if [[ "${INSTALL_PID}" =~ ^[1-9][0-9]*$ ]] && kill -0 "${INSTALL_PID}" 2>/dev/null; then
        INSTALLER_RUNNING=1
    fi
fi
if [[ ${INSTALLER_RUNNING} -eq 0 ]]; then
    nohup bash "${SCRIPT_DIR}/install_runpod_tmux_worker.sh" "${INSTALL_STATUS}" \
        >> "${INSTALL_LOG}" 2>&1 &
    INSTALL_PID=$!
    printf '%s\n' "${INSTALL_PID}" > "${INSTALL_PID_FILE}"
fi

elapsed=0
while [[ ${elapsed} -lt ${WAIT_SECONDS} ]]; do
    if command -v tmux >/dev/null 2>&1; then
        tmux -V
        printf 'tmux installation log: %s\n' "${INSTALL_LOG}"
        exit 0
    fi
    if [[ -f "${INSTALL_STATUS}" ]] && grep -q '"state":"failed"' "${INSTALL_STATUS}"; then
        echo "tmux installation failed; inspect ${INSTALL_LOG}" >&2
        exit 3
    fi
    sleep 2
    elapsed=$((elapsed + 2))
done

echo "Timed out waiting for tmux installation; inspect ${INSTALL_LOG}" >&2
exit 124
