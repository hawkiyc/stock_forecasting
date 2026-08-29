#!/usr/bin/env bash

# Configure local-only Cloudflare credentials and an automatically generated relay token.
set -Eeuo pipefail
set +x
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${PROJECT_ROOT}/.env"
ENV_TEMPLATE="${PROJECT_ROOT}/.env.example"
ENV_HELPER="${SCRIPT_DIR}/update_runpod_env.py"
# shellcheck source=lib/runpod_project_env.sh
source "${SCRIPT_DIR}/lib/runpod_project_env.sh"

if [[ $# -ne 0 ]]; then
    echo "Usage: bash scripts/configure_tpex_cloudflare_proxy.sh" >&2
    exit 2
fi
if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 is required on the local control machine" >&2
    exit 127
fi

CLOUDFLARE_API_TOKEN=""
CLOUDFLARE_ACCOUNT_ID=""
CLOUDFLARE_WORKERS_SUBDOMAIN=""
TPEX_PROXY_SHARED_SECRET=""
if [[ -f "${ENV_FILE}" ]]; then
    runpod_validate_project_env_file "${PROJECT_ROOT}"
    runpod_load_project_env_key "${PROJECT_ROOT}" CLOUDFLARE_API_TOKEN optional
    runpod_load_project_env_key "${PROJECT_ROOT}" CLOUDFLARE_ACCOUNT_ID optional
    runpod_load_project_env_key "${PROJECT_ROOT}" CLOUDFLARE_WORKERS_SUBDOMAIN optional
    runpod_load_project_env_key "${PROJECT_ROOT}" TPEX_PROXY_SHARED_SECRET optional
fi

printf 'Configure the restricted Cloudflare TPEx Worker. Blank input preserves an existing value.\n'
IFS= read -r -s -p 'Cloudflare API token (Workers Scripts Edit/Write): ' CLOUDFLARE_API_TOKEN_INPUT
printf '\n'
printf 'Cloudflare account ID [%s]: ' "${CLOUDFLARE_ACCOUNT_ID}"
IFS= read -r CLOUDFLARE_ACCOUNT_ID_INPUT
printf 'Optional workers.dev account subdomain [%s]: ' "${CLOUDFLARE_WORKERS_SUBDOMAIN}"
IFS= read -r CLOUDFLARE_WORKERS_SUBDOMAIN_INPUT

if [[ -z "${TPEX_PROXY_SHARED_SECRET}" ]]; then
    TPEX_PROXY_SHARED_SECRET_INPUT="$(python3 -c 'import secrets; print(secrets.token_urlsafe(48))')"
else
    TPEX_PROXY_SHARED_SECRET_INPUT=""
fi

{
    if [[ -n "${CLOUDFLARE_API_TOKEN_INPUT}" ]]; then
        printf 'CLOUDFLARE_API_TOKEN\0%s\0' "${CLOUDFLARE_API_TOKEN_INPUT}"
    fi
    if [[ -n "${CLOUDFLARE_ACCOUNT_ID_INPUT}" ]]; then
        printf 'CLOUDFLARE_ACCOUNT_ID\0%s\0' "${CLOUDFLARE_ACCOUNT_ID_INPUT}"
    fi
    if [[ -n "${CLOUDFLARE_WORKERS_SUBDOMAIN_INPUT}" ]]; then
        printf 'CLOUDFLARE_WORKERS_SUBDOMAIN\0%s\0' \
            "${CLOUDFLARE_WORKERS_SUBDOMAIN_INPUT}"
    fi
    if [[ -n "${TPEX_PROXY_SHARED_SECRET_INPUT}" ]]; then
        printf 'TPEX_PROXY_SHARED_SECRET\0%s\0' "${TPEX_PROXY_SHARED_SECRET_INPUT}"
    fi
} | python3 "${ENV_HELPER}" apply-null \
    --env-file "${ENV_FILE}" \
    --template "${ENV_TEMPLATE}" \
    --require-key CLOUDFLARE_API_TOKEN \
    --require-key CLOUDFLARE_ACCOUNT_ID \
    --require-key TPEX_PROXY_SHARED_SECRET

unset CLOUDFLARE_API_TOKEN CLOUDFLARE_API_TOKEN_INPUT \
    TPEX_PROXY_SHARED_SECRET TPEX_PROXY_SHARED_SECRET_INPUT
printf 'Cloudflare TPEx proxy configuration completed. Secret values were not printed.\n'
