#!/usr/bin/env bash

# Download one completed run's immutable manifests, retained checkpoints, and validation result.
set -Eeuo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
S3_WRAPPER="${SCRIPT_DIR}/runpod_s3_project.sh"
DOWNLOAD_ROOT="${PROJECT_ROOT}/artifacts/runpod"
RUN_ID=""
RUN_ID_WAS_EXPLICIT=0
RESUME=0
CHECKPOINT_SCOPE="all"
CHECKPOINT_SCOPE_WAS_EXPLICIT=0

usage() {
    echo "Usage: bash scripts/download_runpod_results.sh [--resume] [--checkpointScope all|best] [RUN_ID]" >&2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --resume)
            RESUME=1
            shift
            ;;
        --checkpointScope|--checkpoint-scope)
            if [[ $# -lt 2 ]]; then
                echo "$1 requires all or best" >&2
                usage
                exit 2
            fi
            if [[ "${CHECKPOINT_SCOPE_WAS_EXPLICIT}" == "1" ]]; then
                echo "--checkpointScope may be specified only once" >&2
                usage
                exit 2
            fi
            case "$2" in
                all|best)
                    CHECKPOINT_SCOPE="$2"
                    ;;
                *)
                    echo "--checkpointScope must be all or best" >&2
                    usage
                    exit 2
                    ;;
            esac
            CHECKPOINT_SCOPE_WAS_EXPLICIT=1
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --*)
            echo "Unknown download option: $1" >&2
            usage
            exit 2
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

CHECKPOINT_NAMES="$(python3 "${SCRIPT_DIR}/runpod_readiness.py" \
    checkpoint-download-names \
    --leaderboard "${RUN_DOWNLOAD_ROOT}/checkpoint-leaderboard.json" \
    --pointer "${RUN_DOWNLOAD_ROOT}/best-checkpoint.json" \
    --run-id "${RUN_ID}" \
    --scope "${CHECKPOINT_SCOPE}")"
BEST_CHECKPOINT="${CHECKPOINT_NAMES%%$'\n'*}"

CHECKPOINT_COUNT=0
while IFS= read -r checkpoint_name; do
    [[ -n "${checkpoint_name}" ]] || continue
    mkdir -p "${RUN_DOWNLOAD_ROOT}/${checkpoint_name}"
    bash "${S3_WRAPPER}" s3 cp \
        "s3://${RUNPOD_NETWORK_VOLUME_ID}/savedModel/${RUN_ID}/${checkpoint_name}/" \
        "${RUN_DOWNLOAD_ROOT}/${checkpoint_name}/" \
        --recursive --only-show-errors
    CHECKPOINT_COUNT=$((CHECKPOINT_COUNT + 1))
done <<< "${CHECKPOINT_NAMES}"
if [[ "${CHECKPOINT_COUNT}" -lt 1 ]]; then
    echo "No retained checkpoints were selected for download" >&2
    exit 1
fi

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

printf 'Downloaded RunPod result: run_id=%s checkpoint_scope=%s checkpoint_count=%s best_checkpoint=%s target=%s\n' \
    "${RUN_ID}" "${CHECKPOINT_SCOPE}" "${CHECKPOINT_COUNT}" \
    "${BEST_CHECKPOINT}" "${RUN_DOWNLOAD_ROOT}"
