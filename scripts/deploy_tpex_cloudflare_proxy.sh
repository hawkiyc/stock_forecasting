#!/usr/bin/env bash

# Deploy the restricted Worker, mirror its token into RunPod Secrets, and verify TPEx access.
set -Eeuo pipefail
set +x
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${PROJECT_ROOT}/.env"
ENV_TEMPLATE="${PROJECT_ROOT}/.env.example"
ENV_HELPER="${SCRIPT_DIR}/update_runpod_env.py"
WORKER_SOURCE="${PROJECT_ROOT}/cloudflare/tpex-proxy/src/index.mjs"
# shellcheck source=lib/runpod_project_env.sh
source "${SCRIPT_DIR}/lib/runpod_project_env.sh"

if [[ $# -ne 0 ]]; then
    echo "Usage: bash scripts/deploy_tpex_cloudflare_proxy.sh" >&2
    exit 2
fi
if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 is required on the local control machine" >&2
    exit 127
fi
runpod_load_tpex_proxy_deploy_env "${PROJECT_ROOT}"

TPEX_PROXY_URL="$(python3 "${SCRIPT_DIR}/cloudflare_tpex_worker.py" deploy \
    --source "${WORKER_SOURCE}")"
export TPEX_PROXY_URL
bash "${SCRIPT_DIR}/verify_tpex_cloudflare_proxy.sh" --environment
RUNPOD_TPEX_PROXY_SECRET_NAME="$(
    python3 "${SCRIPT_DIR}/create_runpod_tpex_proxy_secret.py"
)"

{
    printf 'TPEX_PROXY_URL\0%s\0' "${TPEX_PROXY_URL}"
    printf 'RUNPOD_TPEX_PROXY_SECRET_NAME\0%s\0' \
        "${RUNPOD_TPEX_PROXY_SECRET_NAME}"
} | python3 "${ENV_HELPER}" apply-null \
    --env-file "${ENV_FILE}" \
    --template "${ENV_TEMPLATE}" \
    --require-key TPEX_PROXY_URL \
    --require-key RUNPOD_TPEX_PROXY_SECRET_NAME

unset CLOUDFLARE_API_TOKEN RUNPOD_API_KEY TPEX_PROXY_SHARED_SECRET
printf 'Deployed restricted TPEx Worker: %s\n' "${TPEX_PROXY_URL}"
printf 'Created RunPod Secret reference: %s\n' "${RUNPOD_TPEX_PROXY_SECRET_NAME}"
