#!/usr/bin/env bash

# Best-effort fallback for failures that happen before the Python supervisor starts.
set -u
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/runpod_paths.sh
source "${SCRIPT_DIR}/lib/runpod_paths.sh"

NETWORK_VOLUME_ROOT="${NETWORK_VOLUME_ROOT:-/runpod-volume}"
LOG_ROOT="${LOG_ROOT:-${NETWORK_VOLUME_ROOT}/logs}"
if [[ "${RUNPOD_RUN_KEY+x}" == x ]]; then
    RUN_KEY="${RUNPOD_RUN_KEY}"
    if [[ ! "${RUN_KEY}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,119}$ \
        || "${RUN_KEY}" == *--* ]]; then
        echo "RUNPOD_RUN_KEY must be a canonical single run ID" >&2
        exit 2
    fi
else
    RUN_KEY=bootstrap
fi
SHUTDOWN_DIR="${RUNPOD_SHUTDOWN_DIR:-${LOG_ROOT}/${RUN_KEY}}"
MARKER_PATH="${RUNPOD_SHUTDOWN_MARKER:-${SHUTDOWN_DIR}/shutdown.json}"
# RunPod requires termination instead of stop for Pods with a network volume.
ACTION="${RUNPOD_SHUTDOWN_ACTION:-terminate}"
API_BASE_URL="${RUNPOD_API_BASE_URL:-https://rest.runpod.io/v1}"

runpod_validate_absolute_path "${NETWORK_VOLUME_ROOT}" NETWORK_VOLUME_ROOT || exit $?
case "${NETWORK_VOLUME_ROOT}" in
    /workspace|/workspace/*)
        echo "NETWORK_VOLUME_ROOT must never use /workspace" >&2
        exit 2
        ;;
esac

runpod_validate_path_in_root \
    "${LOG_ROOT}" "${NETWORK_VOLUME_ROOT}" LOG_ROOT NETWORK_VOLUME_ROOT || exit $?
runpod_validate_path_in_root \
    "${SHUTDOWN_DIR}" "${NETWORK_VOLUME_ROOT}" \
    RUNPOD_SHUTDOWN_DIR NETWORK_VOLUME_ROOT || exit $?
runpod_validate_path_in_root \
    "${MARKER_PATH}" "${NETWORK_VOLUME_ROOT}" \
    RUNPOD_SHUTDOWN_MARKER NETWORK_VOLUME_ROOT || exit $?

if [[ -f "${MARKER_PATH}" ]] \
    && grep -Eq '"success"[[:space:]]*:[[:space:]]*true' "${MARKER_PATH}"; then
    exit 0
fi

mkdir -p "${SHUTDOWN_DIR}" "$(dirname "${MARKER_PATH}")"

if [[ "${RUNPOD_DRY_RUN:-0}" == "1" || "${RUNPOD_TEST_MODE:-0}" == "1" ]]; then
    printf '{"attempted":false,"skipped_reason":"dry-run-or-test-mode"}\n' > "${MARKER_PATH}"
    exit 0
fi

if [[ -z "${RUNPOD_API_KEY:-}" || -z "${RUNPOD_POD_ID:-}" ]]; then
    printf '{"attempted":false,"skipped_reason":"missing-api-key-or-pod-id"}\n' > "${MARKER_PATH}"
    exit 0
fi

if [[ ! "${RUNPOD_POD_ID}" =~ ^[A-Za-z0-9_-]+$ ]]; then
    printf '{"attempted":false,"error":"invalid-pod-id"}\n' > "${MARKER_PATH}"
    exit 2
fi

if ! command -v curl >/dev/null 2>&1; then
    printf '{"attempted":false,"error":"curl-not-found"}\n' > "${MARKER_PATH}"
    exit 3
fi

printf '{"attempted":true,"state":"started","action":"%s"}\n' "${ACTION}" > "${MARKER_PATH}"
RESPONSE_PATH="${SHUTDOWN_DIR}/shutdown-response.txt"
HTTP_CODE_PATH="${SHUTDOWN_DIR}/shutdown-http-code.txt"

case "${ACTION}" in
    terminate)
        HTTP_METHOD="DELETE"
        API_URL="${API_BASE_URL%/}/pods/${RUNPOD_POD_ID}"
        ;;
    stop)
        HTTP_METHOD="POST"
        API_URL="${API_BASE_URL%/}/pods/${RUNPOD_POD_ID}/stop"
        ;;
    *)
        printf '{"attempted":false,"error":"invalid-shutdown-action"}\n' > "${MARKER_PATH}"
        exit 2
        ;;
esac

# Feed the authorization header through stdin so the token never appears in
# curl's argv, process listings, persisted commands, or normal output.
HTTP_CODE="$({
    printf 'header = "Authorization: Bearer %s"\n' "${RUNPOD_API_KEY}"
    printf 'header = "Content-Type: application/json"\n'
} | curl --config - \
    --silent --show-error \
    --request "${HTTP_METHOD}" \
    --url "${API_URL}" \
    --connect-timeout "${RUNPOD_SHUTDOWN_CONNECT_TIMEOUT_SECONDS:-10}" \
    --max-time "${RUNPOD_SHUTDOWN_TIMEOUT_SECONDS:-30}" \
    --output "${RESPONSE_PATH}" \
    --write-out '%{http_code}')"
CURL_EXIT_CODE=$?
printf '%s\n' "${HTTP_CODE}" > "${HTTP_CODE_PATH}"

if [[ ${CURL_EXIT_CODE} -ne 0 || ! "${HTTP_CODE}" =~ ^2[0-9][0-9]$ ]]; then
    printf '{"attempted":true,"success":false,"action":"%s","http_code":"%s","curl_exit_code":%d}\n' \
        "${ACTION}" "${HTTP_CODE}" "${CURL_EXIT_CODE}" > "${MARKER_PATH}"
    exit 4
fi

printf '{"attempted":true,"success":true,"action":"%s","http_code":"%s"}\n' \
    "${ACTION}" "${HTTP_CODE}" > "${MARKER_PATH}"
