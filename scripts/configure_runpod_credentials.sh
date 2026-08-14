#!/usr/bin/env bash

# Prompt for local-only credentials and atomically create the project dotenv file.
set -Eeuo pipefail
set +x
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${PROJECT_ROOT}/.env"
ENV_TEMPLATE="${PROJECT_ROOT}/.env.example"
ENV_HELPER="${SCRIPT_DIR}/update_runpod_env.py"

if [[ $# -ne 0 ]]; then
    echo "Usage: bash scripts/configure_runpod_credentials.sh" >&2
    exit 2
fi
if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 is required on the local control machine" >&2
    exit 127
fi

printf 'Enter credentials without echo. Leave a field blank to keep its existing value.\n'
IFS= read -r -s -p 'RunPod API key: ' RUNPOD_API_KEY_INPUT
printf '\n'
IFS= read -r -s -p 'RunPod S3 access key ID: ' RUNPOD_S3_ACCESS_KEY_ID_INPUT
printf '\n'
IFS= read -r -s -p 'RunPod S3 secret access key: ' RUNPOD_S3_SECRET_ACCESS_KEY_INPUT
printf '\n'

{
    if [[ -n "${RUNPOD_API_KEY_INPUT}" ]]; then
        printf 'RUNPOD_API_KEY\0%s\0' "${RUNPOD_API_KEY_INPUT}"
    fi
    if [[ -n "${RUNPOD_S3_ACCESS_KEY_ID_INPUT}" ]]; then
        printf 'RUNPOD_S3_ACCESS_KEY_ID\0%s\0' "${RUNPOD_S3_ACCESS_KEY_ID_INPUT}"
    fi
    if [[ -n "${RUNPOD_S3_SECRET_ACCESS_KEY_INPUT}" ]]; then
        printf 'RUNPOD_S3_SECRET_ACCESS_KEY\0%s\0' "${RUNPOD_S3_SECRET_ACCESS_KEY_INPUT}"
    fi
} | python3 "${ENV_HELPER}" apply-null \
    --env-file "${ENV_FILE}" \
    --template "${ENV_TEMPLATE}" \
    --require-key RUNPOD_API_KEY \
    --require-key RUNPOD_S3_ACCESS_KEY_ID \
    --require-key RUNPOD_S3_SECRET_ACCESS_KEY

unset RUNPOD_API_KEY_INPUT RUNPOD_S3_ACCESS_KEY_ID_INPUT \
    RUNPOD_S3_SECRET_ACCESS_KEY_INPUT
printf 'Credential setup completed. Secret values were not printed.\n'
