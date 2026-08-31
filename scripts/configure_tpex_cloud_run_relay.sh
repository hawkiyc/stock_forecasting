#!/usr/bin/env bash

# Configure local Cloud Run relay metadata and generate its shared application token.
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
    echo "Usage: bash scripts/configure_tpex_cloud_run_relay.sh" >&2
    exit 2
fi
if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 is required on the local control machine" >&2
    exit 127
fi

GCP_PROJECT_ID=""
GCP_CLOUD_RUN_REGION="asia-east1"
GCP_TPEX_RELAY_SERVICE="stock-forecasting-tpex-relay"
GCP_TPEX_RELAY_SECRET="stock-forecasting-tpex-relay-token"
TPEX_PROXY_SHARED_SECRET=""
if [[ -f "${ENV_FILE}" ]]; then
    runpod_validate_project_env_file "${PROJECT_ROOT}"
    runpod_load_project_env_key "${PROJECT_ROOT}" GCP_PROJECT_ID optional
    runpod_load_project_env_key "${PROJECT_ROOT}" GCP_CLOUD_RUN_REGION optional
    runpod_load_project_env_key "${PROJECT_ROOT}" GCP_TPEX_RELAY_SERVICE optional
    runpod_load_project_env_key "${PROJECT_ROOT}" GCP_TPEX_RELAY_SECRET optional
    runpod_load_project_env_key "${PROJECT_ROOT}" TPEX_PROXY_SHARED_SECRET optional
fi
export -n TPEX_PROXY_SHARED_SECRET 2>/dev/null || true
GCP_CLOUD_RUN_REGION="${GCP_CLOUD_RUN_REGION:-asia-east1}"
GCP_TPEX_RELAY_SERVICE="${GCP_TPEX_RELAY_SERVICE:-stock-forecasting-tpex-relay}"
GCP_TPEX_RELAY_SECRET="${GCP_TPEX_RELAY_SECRET:-stock-forecasting-tpex-relay-token}"

if [[ -z "${GCP_PROJECT_ID}" ]] && command -v gcloud >/dev/null 2>&1; then
    configured_project="$(gcloud config get-value project 2>/dev/null || true)"
    if [[ "${configured_project}" != "(unset)" ]]; then
        GCP_PROJECT_ID="${configured_project}"
    fi
fi

printf 'Configure the restricted GCP Cloud Run TPEx relay. Blank input preserves the displayed value.\n'
printf 'GCP project ID [%s]: ' "${GCP_PROJECT_ID}"
IFS= read -r GCP_PROJECT_ID_INPUT
printf 'Cloud Run service name [%s]: ' "${GCP_TPEX_RELAY_SERVICE}"
IFS= read -r GCP_TPEX_RELAY_SERVICE_INPUT
printf 'Secret Manager secret name [%s]: ' "${GCP_TPEX_RELAY_SECRET}"
IFS= read -r GCP_TPEX_RELAY_SECRET_INPUT

GCP_PROJECT_ID_INPUT="${GCP_PROJECT_ID_INPUT:-${GCP_PROJECT_ID}}"
GCP_TPEX_RELAY_SERVICE_INPUT="${GCP_TPEX_RELAY_SERVICE_INPUT:-${GCP_TPEX_RELAY_SERVICE}}"
GCP_TPEX_RELAY_SECRET_INPUT="${GCP_TPEX_RELAY_SECRET_INPUT:-${GCP_TPEX_RELAY_SECRET}}"
if [[ -z "${TPEX_PROXY_SHARED_SECRET}" ]]; then
    TPEX_PROXY_SHARED_SECRET_INPUT="$(python3 -c 'import secrets; print(secrets.token_urlsafe(48))')"
else
    TPEX_PROXY_SHARED_SECRET_INPUT=""
fi

{
    printf 'GCP_PROJECT_ID\0%s\0' "${GCP_PROJECT_ID_INPUT}"
    printf 'GCP_CLOUD_RUN_REGION\0asia-east1\0'
    printf 'GCP_TPEX_RELAY_SERVICE\0%s\0' "${GCP_TPEX_RELAY_SERVICE_INPUT}"
    printf 'GCP_TPEX_RELAY_SECRET\0%s\0' "${GCP_TPEX_RELAY_SECRET_INPUT}"
    if [[ -n "${TPEX_PROXY_SHARED_SECRET_INPUT}" ]]; then
        printf 'TPEX_PROXY_SHARED_SECRET\0%s\0' "${TPEX_PROXY_SHARED_SECRET_INPUT}"
    fi
} | python3 "${ENV_HELPER}" apply-null \
    --env-file "${ENV_FILE}" \
    --template "${ENV_TEMPLATE}" \
    --require-key GCP_PROJECT_ID \
    --require-key GCP_CLOUD_RUN_REGION \
    --require-key GCP_TPEX_RELAY_SERVICE \
    --require-key GCP_TPEX_RELAY_SECRET \
    --require-key TPEX_PROXY_SHARED_SECRET

unset TPEX_PROXY_SHARED_SECRET TPEX_PROXY_SHARED_SECRET_INPUT
printf 'Cloud Run TPEx relay configuration completed for asia-east1. Secret values were not printed.\n'
