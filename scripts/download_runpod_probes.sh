#!/usr/bin/env bash

# Download the latest completed diagnostic through the local control plane.
set -Eeuo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

usage() {
    echo "Usage: bash scripts/runpod_workflow.sh download-probes [PROBE_RUN_ID]"
    echo "Omit PROBE_RUN_ID for the latest completed diagnostic across the volume."
    echo "Provide the diagnosed model's run ID to select its latest completed diagnostic."
    echo "Volume, credentials, checkpoint and output paths are resolved automatically."
}

if [[ $# -eq 1 && ( "$1" == --help || "$1" == -h ) ]]; then
    usage
    exit 0
fi
if [[ $# -gt 1 ]] || [[ $# -eq 1 && ( ! "$1" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,119}$ || "$1" == *--* ) ]]; then
    usage >&2
    exit 2
fi

# shellcheck source=lib/runpod_project_env.sh
source "${SCRIPT_DIR}/lib/runpod_project_env.sh"
runpod_load_s3_env "${PROJECT_ROOT}"
if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 is required on the local control machine" >&2
    exit 127
fi
exec python3 "${SCRIPT_DIR}/download_runpod_probes.py" "$@"
