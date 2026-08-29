#!/usr/bin/env bash

# Verify the authenticated Worker against the TPEx endpoint that RunPod cannot reach directly.
set -Eeuo pipefail
set +x
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=lib/runpod_project_env.sh
source "${SCRIPT_DIR}/lib/runpod_project_env.sh"

VERIFY_FROM_ENVIRONMENT=0
if [[ $# -eq 1 && "$1" == "--environment" ]]; then
    VERIFY_FROM_ENVIRONMENT=1
elif [[ $# -ne 0 ]]; then
    echo "Usage: bash scripts/verify_tpex_cloudflare_proxy.sh [--environment]" >&2
    exit 2
fi
if ! command -v curl >/dev/null 2>&1; then
    echo "curl is required on the local control machine" >&2
    exit 127
fi
if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 is required on the local control machine" >&2
    exit 127
fi
if [[ ${VERIFY_FROM_ENVIRONMENT} -eq 0 ]]; then
    runpod_validate_project_env_file "${PROJECT_ROOT}"
    runpod_load_project_env_key "${PROJECT_ROOT}" TPEX_PROXY_URL required
    runpod_load_project_env_key "${PROJECT_ROOT}" TPEX_PROXY_SHARED_SECRET required
fi
TPEX_PROXY_URL="${TPEX_PROXY_URL:-}"
TPEX_PROXY_SHARED_SECRET="${TPEX_PROXY_SHARED_SECRET:-}"
if [[ ! "${TPEX_PROXY_URL:-}" =~ ^https://[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?\.[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?\.workers\.dev/?$ ]]; then
    echo "TPEX_PROXY_URL is missing or is not an approved workers.dev origin" >&2
    exit 2
fi
if [[ ${#TPEX_PROXY_SHARED_SECRET} -lt 32 || ${#TPEX_PROXY_SHARED_SECRET} -gt 512 ]]; then
    echo "TPEX_PROXY_SHARED_SECRET has an invalid length" >&2
    exit 2
fi

VERIFY_URL="${TPEX_PROXY_URL%/}/www/zh-tw/bulletin/exDailyQ?startDate=2026%2F08%2F01&endDate=2026%2F08%2F28&response=json"
VERIFY_MAX_ATTEMPTS=8
VERIFY_RETRY_DELAY_SECONDS=1
verification_passed=0
for ((attempt = 1; attempt <= VERIFY_MAX_ATTEMPTS; attempt++)); do
    if verification_message="$({
        printf 'silent\n'
        printf 'show-error\n'
        printf 'connect-timeout = 15\n'
        printf 'max-time = 120\n'
        printf 'header = "Accept: application/json"\n'
        printf 'header = "X-TPEX-Proxy-Token: %s"\n' "${TPEX_PROXY_SHARED_SECRET}"
        printf 'url = "%s"\n' "${VERIFY_URL}"
    } | curl --config - --write-out $'\n__FIN_TS_HTTP_STATUS__:%{http_code}' | python3 -c '
import json
import sys

raw = sys.stdin.buffer.read()
marker = b"\n__FIN_TS_HTTP_STATUS__:"
body, separator, status_raw = raw.rpartition(marker)
if not separator:
    print("TPEx proxy verification did not receive an HTTP status")
    raise SystemExit(75)
try:
    status = int(status_raw.decode("ascii"))
except (UnicodeDecodeError, ValueError):
    print("TPEx proxy verification received an invalid HTTP status")
    raise SystemExit(75)
try:
    payload = json.loads(body)
except json.JSONDecodeError:
    payload = None

if not isinstance(payload, dict):
    if status in {0, 404, 408, 425, 429, 500, 502, 503, 504, 522, 523, 524, 525, 526}:
        print(f"TPEx proxy endpoint is not ready: HTTP {status}, non-JSON response")
        raise SystemExit(75)
    print(f"TPEx proxy verification failed: HTTP {status}, non-JSON response")
    raise SystemExit(2)

tables = payload.get("tables")
legacy_table = isinstance(payload.get("fields"), list) and isinstance(payload.get("data"), list)
if status == 200 and (isinstance(tables, list) or legacy_table):
    raise SystemExit(0)

worker_error = payload.get("error")
worker_error_label = worker_error if isinstance(worker_error, str) else "none"
fatal_worker_errors = {
    "method_not_allowed",
    "unauthorized",
    "unsupported_tpex_request",
    "tpex_upstream_redirect_rejected",
    "tpex_upstream_redirect_limit_exceeded",
}
message = (
    f"TPEx proxy verification failed: HTTP {status}, "
    f"worker_error={worker_error_label}, official_data_table=false"
)
if (
    status in {0, 404, 408, 425, 429, 500, 502, 503, 504, 522, 523, 524, 525, 526}
    and worker_error_label not in fatal_worker_errors
):
    print(message)
    raise SystemExit(75)
print(message)
raise SystemExit(2)
')"; then
        verification_passed=1
        break
    else
        verification_status=$?
    fi

    if [[ ${verification_status} -ne 75 || ${attempt} -eq ${VERIFY_MAX_ATTEMPTS} ]]; then
        printf '%s\n' "${verification_message}" >&2
        exit 2
    fi
    printf '%s; retrying in %ss (attempt %d/%d)\n' \
        "${verification_message}" "${VERIFY_RETRY_DELAY_SECONDS}" \
        "${attempt}" "${VERIFY_MAX_ATTEMPTS}" >&2
    sleep "${VERIFY_RETRY_DELAY_SECONDS}"
    VERIFY_RETRY_DELAY_SECONDS=$((VERIFY_RETRY_DELAY_SECONDS * 2))
    if [[ ${VERIFY_RETRY_DELAY_SECONDS} -gt 16 ]]; then
        VERIFY_RETRY_DELAY_SECONDS=16
    fi
done

if [[ ${verification_passed} -ne 1 ]]; then
    echo "TPEx proxy verification exhausted its readiness attempts" >&2
    exit 2
fi

unset TPEX_PROXY_SHARED_SECRET
printf 'Cloudflare TPEx proxy verification passed: authenticated exDailyQ returned official JSON.\n'
