#!/usr/bin/env bash

# Run runpodctl with this project's API key instead of the global CLI key.
set -Eeuo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# shellcheck source=lib/runpod_project_env.sh
source "${SCRIPT_DIR}/lib/runpod_project_env.sh"

if [[ $# -eq 0 ]]; then
    echo "Usage: runpodctl_project.sh RUNPODCTL_ARGUMENTS..." >&2
    exit 2
fi

runpod_load_runpodctl_env "${LOCAL_PROJECT_ROOT}"

if ! command -v runpodctl >/dev/null 2>&1; then
    echo "runpodctl is required on the local control machine" >&2
    exit 127
fi

RUNPODCTL_BIN="$(command -v runpodctl)"
API_KEY_VALUE="${RUNPOD_API_KEY}"
PATH_VALUE="${PATH:-/usr/local/bin:/usr/bin:/bin}"
HOME_VALUE="${HOME:-/tmp}"
TMPDIR_VALUE="${TMPDIR:-/tmp}"

runpod_clear_exported_environment
export PATH="${PATH_VALUE}"
export HOME="${HOME_VALUE}"
export TMPDIR="${TMPDIR_VALUE}"
export RUNPOD_API_KEY="${API_KEY_VALUE}"

exec "${RUNPODCTL_BIN}" "$@"
