#!/usr/bin/env bash

# Show Cloud Run control-plane readiness without sending a TPEx request.
set -Eeuo pipefail
set +x
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=lib/runpod_project_env.sh
source "${SCRIPT_DIR}/lib/runpod_project_env.sh"

if [[ $# -ne 0 ]]; then
    echo "Usage: bash scripts/show_tpex_cloud_run_relay_status.sh" >&2
    exit 2
fi
if ! command -v gcloud >/dev/null 2>&1; then
    echo "gcloud is required on the local control machine" >&2
    exit 127
fi
runpod_validate_project_env_file "${PROJECT_ROOT}"
runpod_load_project_env_key "${PROJECT_ROOT}" GCP_PROJECT_ID required
runpod_load_project_env_key "${PROJECT_ROOT}" GCP_CLOUD_RUN_REGION required
runpod_load_project_env_key "${PROJECT_ROOT}" GCP_TPEX_RELAY_SERVICE required

gcloud run services describe "${GCP_TPEX_RELAY_SERVICE}" \
    --project="${GCP_PROJECT_ID}" \
    --region="${GCP_CLOUD_RUN_REGION}" \
    --format='yaml(metadata.name,status.url,status.latestCreatedRevisionName,status.latestReadyRevisionName,status.conditions)'
