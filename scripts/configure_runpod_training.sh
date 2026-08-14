#!/usr/bin/env bash

# Create an immutable stage and dataset selection without editing dotenv or YAML files.
set -Eeuo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SELECTION_HELPER="${SCRIPT_DIR}/runpod_selection.py"

if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 is required on the local control machine" >&2
    exit 127
fi
if [[ ! -f "${SELECTION_HELPER}" || -L "${SELECTION_HELPER}" ]]; then
    echo "RunPod selection helper is unavailable" >&2
    exit 127
fi

if [[ $# -eq 0 ]]; then
    set -- --interactive
fi

exec python3 "${SELECTION_HELPER}" create --project-root "${PROJECT_ROOT}" "$@"
