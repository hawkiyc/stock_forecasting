#!/usr/bin/env bash

# Start a scale-to-zero Cloud Run relay without issuing a TPEx upstream request.
set -Eeuo pipefail
set +x
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESPONSE_VERIFIER="${SCRIPT_DIR}/verify_tpex_cloud_run_response.py"

if [[ $# -ne 0 ]]; then
    echo "Usage: bash scripts/warm_tpex_cloud_run_relay.sh" >&2
    exit 2
fi
for command_name in curl python3; do
    if ! command -v "${command_name}" >/dev/null 2>&1; then
        echo "${command_name} is required to warm the TPEx relay" >&2
        exit 127
    fi
done
TPEX_PROXY_URL="${TPEX_PROXY_URL:-}"
TPEX_PROXY_TOKEN="${TPEX_PROXY_TOKEN:-}"
if [[ ! "${TPEX_PROXY_URL}" =~ ^https://[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?(\.[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?)*\.run\.app/?$ ]]; then
    echo "TPEX_PROXY_URL is missing or is not an approved run.app origin" >&2
    exit 2
fi
if [[ ${#TPEX_PROXY_TOKEN} -lt 32 || ${#TPEX_PROXY_TOKEN} -gt 512 ]]; then
    echo "TPEX_PROXY_TOKEN has an invalid length" >&2
    exit 2
fi

WARMUP_URL="${TPEX_PROXY_URL%/}/_internal/warmup"
WARMUP_MAX_ATTEMPTS=6
retry_delay_seconds=1
for ((attempt = 1; attempt <= WARMUP_MAX_ATTEMPTS; attempt++)); do
    if warmup_message="$({
        printf 'silent\n'
        printf 'show-error\n'
        printf 'connect-timeout = 15\n'
        printf 'max-time = 60\n'
        printf 'header = "Accept: application/json"\n'
        printf 'header = "X-TPEX-Relay-Token: %s"\n' "${TPEX_PROXY_TOKEN}"
        printf 'url = "%s"\n' "${WARMUP_URL}"
    } | curl --config - --write-out $'\n__FIN_TS_HTTP_STATUS__:%{http_code}\n__FIN_TS_RELAY_VERSION__:%header{x-tpex-relay-version}\n__FIN_TS_RELAY_REGION__:%header{x-tpex-relay-region}\n__FIN_TS_RELAY_REVISION__:%header{x-tpex-relay-revision}' 2>/dev/null | python3 "${RESPONSE_VERIFIER}" \
        --mode warmup \
        --label warmup \
        --expected-region asia-east1)"; then
        unset TPEX_PROXY_TOKEN
        printf 'Cloud Run TPEx relay warmup passed; no TPEx upstream request was sent.\n'
        exit 0
    else
        warmup_status=$?
    fi
    if [[ ${warmup_status} -ne 75 || ${attempt} -eq ${WARMUP_MAX_ATTEMPTS} ]]; then
        printf '%s\n' "${warmup_message}" >&2
        exit 2
    fi
    printf '%s; retrying in %ss (attempt %d/%d)\n' \
        "${warmup_message}" "${retry_delay_seconds}" \
        "${attempt}" "${WARMUP_MAX_ATTEMPTS}" >&2
    sleep "${retry_delay_seconds}"
    retry_delay_seconds=$((retry_delay_seconds * 2))
    if [[ ${retry_delay_seconds} -gt 8 ]]; then
        retry_delay_seconds=8
    fi
done

echo "Cloud Run TPEx relay warmup exhausted its readiness attempts" >&2
exit 2
