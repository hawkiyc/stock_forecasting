#!/usr/bin/env bash

# Verify warmup and every approved TPEx route through the deployed Cloud Run relay.
set -Eeuo pipefail
set +x
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RESPONSE_VERIFIER="${SCRIPT_DIR}/verify_tpex_cloud_run_response.py"
# shellcheck source=lib/runpod_project_env.sh
source "${SCRIPT_DIR}/lib/runpod_project_env.sh"

VERIFY_FROM_ENVIRONMENT=0
if [[ $# -eq 1 && "$1" == "--environment" ]]; then
    VERIFY_FROM_ENVIRONMENT=1
elif [[ $# -ne 0 ]]; then
    echo "Usage: bash scripts/verify_tpex_cloud_run_relay.sh [--environment]" >&2
    exit 2
fi
for command_name in curl python3; do
    if ! command -v "${command_name}" >/dev/null 2>&1; then
        echo "${command_name} is required on the local control machine" >&2
        exit 127
    fi
done
if [[ ${VERIFY_FROM_ENVIRONMENT} -eq 0 ]]; then
    runpod_validate_project_env_file "${PROJECT_ROOT}"
    runpod_load_project_env_key "${PROJECT_ROOT}" TPEX_PROXY_URL required
    runpod_load_project_env_key "${PROJECT_ROOT}" TPEX_PROXY_SHARED_SECRET required
    runpod_load_project_env_key "${PROJECT_ROOT}" GCP_CLOUD_RUN_REGION required
fi
export -n TPEX_PROXY_SHARED_SECRET 2>/dev/null || true
TPEX_PROXY_URL="${TPEX_PROXY_URL:-}"
TPEX_PROXY_SHARED_SECRET="${TPEX_PROXY_SHARED_SECRET:-}"
GCP_CLOUD_RUN_REGION="${GCP_CLOUD_RUN_REGION:-}"
if [[ ! "${TPEX_PROXY_URL}" =~ ^https://[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?(\.[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?)*\.run\.app/?$ ]]; then
    echo "TPEX_PROXY_URL is missing or is not an approved run.app origin" >&2
    exit 2
fi
if [[ ${#TPEX_PROXY_SHARED_SECRET} -lt 32 || ${#TPEX_PROXY_SHARED_SECRET} -gt 512 ]]; then
    echo "TPEX_PROXY_SHARED_SECRET has an invalid length" >&2
    exit 2
fi
if [[ "${GCP_CLOUD_RUN_REGION}" != "asia-east1" ]]; then
    echo "GCP_CLOUD_RUN_REGION must be asia-east1 for the TPEx relay" >&2
    exit 2
fi

VERIFY_MAX_ATTEMPTS=8
VERIFY_TPEX_INTERVAL_SECONDS=2
verify_relay_request() {
    local label="$1"
    local mode="$2"
    local request_url="$3"
    local attempt=0
    local retry_delay_seconds=1
    local verification_message=""
    local verification_status=0

    for ((attempt = 1; attempt <= VERIFY_MAX_ATTEMPTS; attempt++)); do
        if verification_message="$({
            printf 'silent\n'
            printf 'show-error\n'
            printf 'connect-timeout = 15\n'
            printf 'max-time = 60\n'
            printf 'header = "Accept: application/json"\n'
            printf 'header = "X-TPEX-Relay-Token: %s"\n' "${TPEX_PROXY_SHARED_SECRET}"
            printf 'url = "%s"\n' "${request_url}"
        } | curl --config - --write-out $'\n__FIN_TS_HTTP_STATUS__:%{http_code}\n__FIN_TS_RELAY_VERSION__:%header{x-tpex-relay-version}\n__FIN_TS_RELAY_REGION__:%header{x-tpex-relay-region}\n__FIN_TS_RELAY_REVISION__:%header{x-tpex-relay-revision}\n__FIN_TS_UPSTREAM_REDIRECTS__:%header{x-tpex-upstream-redirects}\n__FIN_TS_UPSTREAM_REDIRECT_COOKIES__:%header{x-tpex-upstream-redirect-cookies}' 2>/dev/null | python3 "${RESPONSE_VERIFIER}" \
            --mode "${mode}" \
            --label "${label}" \
            --expected-region "${GCP_CLOUD_RUN_REGION}")"; then
            return 0
        else
            verification_status=$?
        fi

        if [[ ${verification_status} -ne 75 || ${attempt} -eq ${VERIFY_MAX_ATTEMPTS} ]]; then
            printf '%s\n' "${verification_message}" >&2
            return 2
        fi
        printf '%s; retrying in %ss (attempt %d/%d)\n' \
            "${verification_message}" "${retry_delay_seconds}" \
            "${attempt}" "${VERIFY_MAX_ATTEMPTS}" >&2
        sleep "${retry_delay_seconds}"
        retry_delay_seconds=$((retry_delay_seconds * 2))
        if [[ ${retry_delay_seconds} -gt 16 ]]; then
            retry_delay_seconds=16
        fi
    done
    return 2
}

RELAY_ROOT="${TPEX_PROXY_URL%/}"
verify_relay_request "warmup" "warmup" "${RELAY_ROOT}/_internal/warmup"
verify_relay_request "dailyQuotes" "official" \
    "${RELAY_ROOT}/www/zh-tw/afterTrading/dailyQuotes?date=2026%2F04%2F01&id=&response=json"
sleep "${VERIFY_TPEX_INTERVAL_SECONDS}"
verify_relay_request "exDailyQ" "official" \
    "${RELAY_ROOT}/www/zh-tw/bulletin/exDailyQ?startDate=2026%2F04%2F01&endDate=2026%2F04%2F30&response=json"
sleep "${VERIFY_TPEX_INTERVAL_SECONDS}"
verify_relay_request "ROE" "official" \
    "${RELAY_ROOT}/www/zh-tw/indexInfo/ROE?date=2026%2F04%2F01&response=json"
sleep "${VERIFY_TPEX_INTERVAL_SECONDS}"
verify_relay_request "inx" "official" \
    "${RELAY_ROOT}/www/zh-tw/indexInfo/inx?date=2026%2F04%2F01&response=json"

unset TPEX_PROXY_SHARED_SECRET
printf 'Cloud Run TPEx relay verification passed: warmup and all four official routes returned valid JSON.\n'
