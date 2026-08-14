#!/usr/bin/env bash

# Run AWS CLI against this project's RunPod network-volume S3 endpoint.
set -Eeuo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# shellcheck source=lib/runpod_project_env.sh
source "${SCRIPT_DIR}/lib/runpod_project_env.sh"

if [[ $# -lt 2 ]]; then
    echo "Usage: runpod_s3_project.sh AWS_CLI_ARGUMENTS..." >&2
    exit 2
fi

runpod_load_s3_env "${LOCAL_PROJECT_ROOT}"

if [[ ! "${RUNPOD_S3_REGION}" =~ ^[A-Z0-9]+(-[A-Z0-9]+)+$ ]]; then
    echo "RUNPOD_S3_REGION has an invalid format" >&2
    exit 2
fi

REGION_LOWER="$(printf '%s' "${RUNPOD_S3_REGION}" | tr '[:upper:]' '[:lower:]')"
EXPECTED_ENDPOINT="https://s3api-${REGION_LOWER}.runpod.io/"
RUNPOD_S3_ENDPOINT="${RUNPOD_S3_ENDPOINT:-${EXPECTED_ENDPOINT}}"
if [[ "${RUNPOD_S3_ENDPOINT}" != "${EXPECTED_ENDPOINT}" ]]; then
    echo "RUNPOD_S3_ENDPOINT must match the selected RunPod datacenter" >&2
    exit 2
fi

if ! command -v aws >/dev/null 2>&1; then
    echo "AWS CLI is required on the local control machine" >&2
    exit 127
fi

case "$1:$2" in
    s3:cp|s3:ls|s3api:head-bucket|s3api:head-object|s3api:list-objects-v2) ;;
    *)
        echo "Unsupported AWS operation for the RunPod S3 wrapper" >&2
        exit 2
        ;;
esac

for argument in "$@"; do
    case "${argument}" in
        --delete|--debug|--no-verify-ssl|--profile|--profile=*|\
        --endpoint-url|--endpoint-url=*|--region|--region=*)
            echo "Unsafe or wrapper-managed AWS argument: ${argument}" >&2
            exit 2
            ;;
    esac
done

AWS_BIN="$(command -v aws)"
ACCESS_KEY_VALUE="${RUNPOD_S3_ACCESS_KEY_ID}"
SECRET_KEY_VALUE="${RUNPOD_S3_SECRET_ACCESS_KEY}"
REGION_VALUE="${RUNPOD_S3_REGION}"
ENDPOINT_VALUE="${RUNPOD_S3_ENDPOINT}"
PATH_VALUE="${PATH:-/usr/local/bin:/usr/bin:/bin}"
HOME_VALUE="${HOME:-/tmp}"
TMPDIR_VALUE="${TMPDIR:-/tmp}"
CONNECT_TIMEOUT_VALUE="${RUNPOD_S3_CONNECT_TIMEOUT_OVERRIDE:-${RUNPOD_S3_CONNECT_TIMEOUT:-120}}"
READ_TIMEOUT_VALUE="${RUNPOD_S3_READ_TIMEOUT_OVERRIDE:-${RUNPOD_S3_READ_TIMEOUT:-600}}"
MAX_ATTEMPTS_VALUE="${RUNPOD_S3_MAX_ATTEMPTS_OVERRIDE:-10}"
if [[ ! "${CONNECT_TIMEOUT_VALUE}" =~ ^[1-9][0-9]*$ \
    || ! "${READ_TIMEOUT_VALUE}" =~ ^[1-9][0-9]*$ \
    || ! "${MAX_ATTEMPTS_VALUE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "RunPod S3 timeout and retry overrides must be positive integers" >&2
    exit 2
fi

# Give AWS CLI only the S3 credential and runtime values it needs.
runpod_clear_exported_environment
export PATH="${PATH_VALUE}"
export HOME="${HOME_VALUE}"
export TMPDIR="${TMPDIR_VALUE}"
export AWS_ACCESS_KEY_ID="${ACCESS_KEY_VALUE}"
export AWS_SECRET_ACCESS_KEY="${SECRET_KEY_VALUE}"
export AWS_REGION="${REGION_VALUE}"
export AWS_DEFAULT_REGION="${REGION_VALUE}"
export AWS_CONFIG_FILE=/dev/null
export AWS_SHARED_CREDENTIALS_FILE=/dev/null
export AWS_EC2_METADATA_DISABLED=true
export AWS_PAGER=""
export AWS_RETRY_MODE=standard
export AWS_MAX_ATTEMPTS="${MAX_ATTEMPTS_VALUE}"

exec "${AWS_BIN}" "$@" \
    --cli-connect-timeout "${CONNECT_TIMEOUT_VALUE}" \
    --cli-read-timeout "${READ_TIMEOUT_VALUE}" \
    --region "${REGION_VALUE}" \
    --endpoint-url "${ENDPOINT_VALUE}"
