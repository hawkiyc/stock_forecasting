#!/usr/bin/env bash

# Create one RunPod network volume and register its returned identity atomically.
set -Eeuo pipefail
set +x
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${PROJECT_ROOT}/.env"
ENV_TEMPLATE="${PROJECT_ROOT}/.env.example"
ENV_HELPER="${SCRIPT_DIR}/update_runpod_env.py"
RUNPODCTL_WRAPPER="${SCRIPT_DIR}/runpodctl_project.sh"

VOLUME_NAME=stock-forecasting
VOLUME_SIZE_GB=100
DATACENTER_ID=EU-RO-1
FORCE_NEW=0

usage() {
    echo "Usage: bash scripts/deploy_runpod_network_volume.sh [--name NAME] [--size-gb N] [--datacenter ID] [--force-new]" >&2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --name)
            [[ $# -ge 2 ]] || { usage; exit 2; }
            VOLUME_NAME="$2"
            shift 2
            ;;
        --size-gb)
            [[ $# -ge 2 ]] || { usage; exit 2; }
            VOLUME_SIZE_GB="$2"
            shift 2
            ;;
        --datacenter)
            [[ $# -ge 2 ]] || { usage; exit 2; }
            DATACENTER_ID="$2"
            shift 2
            ;;
        --force-new)
            FORCE_NEW=1
            shift
            ;;
        *)
            usage
            exit 2
            ;;
    esac
done

if [[ ! "${VOLUME_NAME}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,62}$ ]]; then
    echo "Network volume name contains unsupported characters" >&2
    exit 2
fi
if [[ ! "${VOLUME_SIZE_GB}" =~ ^[1-9][0-9]{0,3}$ \
    || "${VOLUME_SIZE_GB}" -gt 4000 ]]; then
    echo "Network volume size must be between 1 and 4000 GB" >&2
    exit 2
fi
if [[ ! "${DATACENTER_ID}" =~ ^[A-Z0-9]+(-[A-Z0-9]+)+$ ]]; then
    echo "RunPod datacenter ID has an invalid format" >&2
    exit 2
fi
if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 is required on the local control machine" >&2
    exit 127
fi

# shellcheck source=lib/runpod_project_env.sh
source "${SCRIPT_DIR}/lib/runpod_project_env.sh"
runpod_validate_project_env_file "${PROJECT_ROOT}"
EXISTING_VOLUME_ID="$(runpod_read_project_env_value "${ENV_FILE}" RUNPOD_NETWORK_VOLUME_ID || true)"
if [[ -n "${EXISTING_VOLUME_ID}" && "${FORCE_NEW}" != "1" ]]; then
    printf 'A network volume is already registered: %s\n' "${EXISTING_VOLUME_ID}"
    printf 'No new billable volume was created. Use --force-new only when a second volume is intentional.\n'
    exit 0
fi

CREATE_OUTPUT="$(bash "${RUNPODCTL_WRAPPER}" network-volume create \
    --name "${VOLUME_NAME}" \
    --size "${VOLUME_SIZE_GB}" \
    --data-center-id "${DATACENTER_ID}" \
    --output json)"

set +e
PARSED_VOLUME="$(printf '%s' "${CREATE_OUTPUT}" | python3 -c '
import json
import re
import sys

payload = json.load(sys.stdin)
if isinstance(payload, dict) and isinstance(payload.get("data"), dict):
    payload = payload["data"]
if not isinstance(payload, dict):
    raise SystemExit("RunPod volume response is not a JSON object")
volume_id = payload.get("id", "")
datacenter = payload.get("dataCenterId", payload.get("data_center_id", ""))
if not isinstance(volume_id, str) or re.fullmatch(r"[A-Za-z0-9_-]+", volume_id) is None:
    raise SystemExit("RunPod volume response has no safe volume ID")
if datacenter and datacenter != sys.argv[1]:
    raise SystemExit("RunPod created the volume in an unexpected datacenter")
print(volume_id)
print(sys.argv[1])
' "${DATACENTER_ID}")"
PARSE_EXIT_CODE=$?
set -e
if [[ ${PARSE_EXIT_CODE} -ne 0 ]]; then
    printf 'The volume may have been created, but its response could not be registered. RunPod response: %s\n' \
        "${CREATE_OUTPUT}" >&2
    exit 3
fi
VOLUME_ID="$(printf '%s\n' "${PARSED_VOLUME}" | sed -n '1p')"
RETURNED_DATACENTER_ID="$(printf '%s\n' "${PARSED_VOLUME}" | sed -n '2p')"
REGION_LOWER="$(printf '%s' "${RETURNED_DATACENTER_ID}" | tr '[:upper:]' '[:lower:]')"
S3_ENDPOINT="https://s3api-${REGION_LOWER}.runpod.io/"

if ! {
    printf 'RUNPOD_NETWORK_VOLUME_ID\0%s\0' "${VOLUME_ID}"
    printf 'RUNPOD_DATACENTER_ID\0%s\0' "${RETURNED_DATACENTER_ID}"
    printf 'RUNPOD_S3_REGION\0%s\0' "${RETURNED_DATACENTER_ID}"
    printf 'RUNPOD_S3_ENDPOINT\0%s\0' "${S3_ENDPOINT}"
} | python3 "${ENV_HELPER}" apply-null \
    --env-file "${ENV_FILE}" \
    --template "${ENV_TEMPLATE}"; then
    printf 'The volume was created but local registration failed. Preserve this ID: %s\n' \
        "${VOLUME_ID}" >&2
    exit 3
fi

printf 'Network volume deployed and registered: id=%s datacenter=%s size_gb=%s\n' \
    "${VOLUME_ID}" "${RETURNED_DATACENTER_ID}" "${VOLUME_SIZE_GB}"
