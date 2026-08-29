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
{
    printf 'silent\n'
    printf 'show-error\n'
    printf 'fail-with-body\n'
    printf 'connect-timeout = 15\n'
    printf 'max-time = 120\n'
    printf 'header = "Accept: application/json"\n'
    printf 'header = "X-TPEX-Proxy-Token: %s"\n' "${TPEX_PROXY_SHARED_SECRET}"
    printf 'url = "%s"\n' "${VERIFY_URL}"
} | curl --config - | python3 -c '
import json
import sys

payload = json.load(sys.stdin)
if not isinstance(payload, dict):
    raise SystemExit("TPEx proxy verification did not return a JSON object")
tables = payload.get("tables")
legacy_table = isinstance(payload.get("fields"), list) and isinstance(payload.get("data"), list)
if not isinstance(tables, list) and not legacy_table:
    raise SystemExit("TPEx proxy verification response has no official data table")
'

unset TPEX_PROXY_SHARED_SECRET
printf 'Cloudflare TPEx proxy verification passed: authenticated exDailyQ returned official JSON.\n'
