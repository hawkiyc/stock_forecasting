#!/usr/bin/env bash

# Verify project-scoped S3 credentials without uploading or modifying objects.
set -Eeuo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
S3_WRAPPER="${SCRIPT_DIR}/runpod_s3_project.sh"

# shellcheck source=lib/runpod_project_env.sh
source "${SCRIPT_DIR}/lib/runpod_project_env.sh"
runpod_load_s3_env "${LOCAL_PROJECT_ROOT}"

if [[ ! -f "${S3_WRAPPER}" || ! -r "${S3_WRAPPER}" ]]; then
    echo "RunPod S3 wrapper is unavailable: ${S3_WRAPPER}" >&2
    exit 127
fi

bash "${S3_WRAPPER}" s3api head-bucket \
    --bucket "${RUNPOD_NETWORK_VOLUME_ID}" \
    >/dev/null

printf 'RunPod S3 access verified for region %s.\n' "${RUNPOD_S3_REGION}"
