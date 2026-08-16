#!/usr/bin/env bash

# Terminate the current Pod after its terminal workflow state has been persisted.
set -u
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HELPER="${SCRIPT_DIR}/runpod_self_terminate.py"
STOP_SCRIPT="${SCRIPT_DIR}/stop_runpod_pod.sh"
RUNPOD_IMAGE_PYTHON="${RUNPOD_PYTHON_BIN:-/usr/local/bin/python}"
ATTEMPTS="${RUNPOD_SELF_TERMINATE_ATTEMPTS:-3}"

if [[ ! "${ATTEMPTS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "RUNPOD_SELF_TERMINATE_ATTEMPTS must be a positive integer" >&2
    exit 2
fi
if [[ ! -x "${RUNPOD_IMAGE_PYTHON}" || ! -r "${HELPER}" || ! -r "${STOP_SCRIPT}" ]]; then
    echo "Pod self-termination helpers are unavailable" >&2
    exit 127
fi

attempt=1
while [[ ${attempt} -le ${ATTEMPTS} ]]; do
    if "${RUNPOD_IMAGE_PYTHON}" "${HELPER}" --stop-script "${STOP_SCRIPT}"; then
        exit 0
    fi
    if [[ ${attempt} -lt ${ATTEMPTS} ]]; then
        sleep "$((attempt * 5))"
    fi
    attempt=$((attempt + 1))
done
echo "Pod self-termination failed after ${ATTEMPTS} attempts; external guard remains armed" >&2
exit 4
