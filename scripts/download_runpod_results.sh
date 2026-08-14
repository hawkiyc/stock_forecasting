#!/usr/bin/env bash

# Download one completed run's immutable manifests, best checkpoint, and validation result.
set -Eeuo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
S3_WRAPPER="${SCRIPT_DIR}/runpod_s3_project.sh"
DOWNLOAD_ROOT="${PROJECT_ROOT}/artifacts/runpod"
RUN_ID=""
RUN_ID_WAS_EXPLICIT=0
RESUME=0

usage() {
    echo "Usage: bash scripts/download_runpod_results.sh [--resume] [RUN_ID]" >&2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --resume)
            RESUME=1
            shift
            ;;
        *)
            if [[ -n "${RUN_ID}" ]]; then
                usage
                exit 2
            fi
            RUN_ID="$1"
            RUN_ID_WAS_EXPLICIT=1
            shift
            ;;
    esac
done

# shellcheck source=lib/runpod_project_env.sh
source "${SCRIPT_DIR}/lib/runpod_project_env.sh"
runpod_load_s3_env "${PROJECT_ROOT}"
if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 is required on the local control machine" >&2
    exit 127
fi

TRAINING_JSON="$(bash "${S3_WRAPPER}" s3 cp \
    "s3://${RUNPOD_NETWORK_VOLUME_ID}/lifecycle/stage1/training.json" - \
    --only-show-errors)"
LATEST_TRAINING_RUN_ID="$(printf '%s' "${TRAINING_JSON}" | python3 -c '
import json
import re
import sys

payload = json.load(sys.stdin)
run_id = payload.get("wandb_run_id", "")
if not isinstance(run_id, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,119}", run_id) is None:
    raise SystemExit("Latest training lifecycle has no safe run ID")
print(run_id)
')"
if [[ -z "${RUN_ID}" ]]; then
    RUN_ID="$(printf '%s' "${TRAINING_JSON}" | python3 -c '
import json
import re
import sys

payload = json.load(sys.stdin)
run_id = payload.get("wandb_run_id", "")
if payload.get("state") != "ready":
    raise SystemExit("Latest training lifecycle is not ready")
if not isinstance(run_id, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,119}", run_id) is None:
    raise SystemExit("Latest training lifecycle has no safe run ID")
print(run_id)
')"
fi
if [[ ! "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,119}$ \
    || "${RUN_ID}" == *--* ]]; then
    echo "Run ID contains unsupported characters" >&2
    exit 2
fi

RUN_DOWNLOAD_ROOT="${DOWNLOAD_ROOT}/${RUN_ID}"
if [[ -d "${RUN_DOWNLOAD_ROOT}" && "${RESUME}" != "1" ]]; then
    echo "Download target already exists; use --resume to fill or refresh known files" >&2
    exit 2
fi
mkdir -p "${RUN_DOWNLOAD_ROOT}"
if [[ "${LATEST_TRAINING_RUN_ID}" == "${RUN_ID}" ]]; then
    printf '%s\n' "${TRAINING_JSON}" > "${RUN_DOWNLOAD_ROOT}/training-lifecycle.json"
elif [[ "${RUN_ID_WAS_EXPLICIT}" == "1" ]]; then
    printf 'Latest singleton training lifecycle belongs to %s; it will not be attached to older run %s.\n' \
        "${LATEST_TRAINING_RUN_ID}" "${RUN_ID}"
fi

download_file() {
    local remote_key="$1"
    local local_name="$2"
    bash "${S3_WRAPPER}" s3 cp \
        "s3://${RUNPOD_NETWORK_VOLUME_ID}/${remote_key}" \
        "${RUN_DOWNLOAD_ROOT}/${local_name}" \
        --only-show-errors
}

download_file "savedModel/${RUN_ID}/run-manifest.json" run-manifest.json
download_file "savedModel/${RUN_ID}/resolved-config.yaml" resolved-config.yaml
download_file "savedModel/${RUN_ID}/checkpoint-leaderboard.json" checkpoint-leaderboard.json
download_file "savedModel/${RUN_ID}/best-checkpoint.json" best-checkpoint.json

BEST_CHECKPOINT="$(python3 -c '
import json
import re
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    payload = json.load(stream)
value = payload.get("path", "")
if not isinstance(value, str) or re.fullmatch(r"checkpoint-[0-9]{6,}", value) is None:
    raise SystemExit("Best-checkpoint pointer is invalid")
print(value)
' "${RUN_DOWNLOAD_ROOT}/best-checkpoint.json")"
mkdir -p "${RUN_DOWNLOAD_ROOT}/${BEST_CHECKPOINT}"
bash "${S3_WRAPPER}" s3 cp \
    "s3://${RUNPOD_NETWORK_VOLUME_ID}/savedModel/${RUN_ID}/${BEST_CHECKPOINT}/" \
    "${RUN_DOWNLOAD_ROOT}/${BEST_CHECKPOINT}/" \
    --recursive --only-show-errors

download_file "evaluations/${RUN_ID}/validation-benchmark.json" validation-benchmark.json
download_file \
    "lifecycle/runs/${RUN_ID}/training-completed.json" \
    training-completed.json

VALIDATION_JSON="$(bash "${S3_WRAPPER}" s3 cp \
    "s3://${RUNPOD_NETWORK_VOLUME_ID}/lifecycle/stage1/validation.json" - \
    --only-show-errors)"
VALIDATION_RUN_ID="$(printf '%s' "${VALIDATION_JSON}" | python3 -c '
import json
import sys

payload = json.load(sys.stdin)
print(str(payload.get("wandb_run_id", "")))
')"
if [[ "${VALIDATION_RUN_ID}" == "${RUN_ID}" ]]; then
    printf '%s\n' "${VALIDATION_JSON}" > "${RUN_DOWNLOAD_ROOT}/validation-lifecycle.json"
else
    printf 'Latest singleton validation lifecycle belongs to another run; scoped validation artifacts remain authoritative for %s.\n' \
        "${RUN_ID}"
fi

printf 'Downloaded RunPod result: run_id=%s best_checkpoint=%s target=%s\n' \
    "${RUN_ID}" "${BEST_CHECKPOINT}" "${RUN_DOWNLOAD_ROOT}"
