#!/usr/bin/env bash

# Summarize known RunPod lifecycle markers without creating local artifacts.
set -Eeuo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
S3_WRAPPER="${SCRIPT_DIR}/runpod_s3_project.sh"

# shellcheck source=lib/runpod_project_env.sh
source "${SCRIPT_DIR}/lib/runpod_project_env.sh"
runpod_load_s3_env "${PROJECT_ROOT}"
if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 is required on the local control machine" >&2
    exit 127
fi

AVAILABLE_KEYS="$(bash "${S3_WRAPPER}" s3api list-objects-v2 \
    --bucket "${RUNPOD_NETWORK_VOLUME_ID}" \
    --prefix lifecycle/ \
    --query 'Contents[].Key' \
    --output text)"

summarize_marker() {
    local key="$1"
    local label="$2"
    local found=0
    local listed_key=""
    local payload=""

    for listed_key in ${AVAILABLE_KEYS}; do
        if [[ "${listed_key}" == "${key}" ]]; then
            found=1
            break
        fi
    done
    if [[ ${found} -eq 0 ]]; then
        printf '%-12s missing\n' "${label}"
        return 0
    fi
    payload="$(bash "${S3_WRAPPER}" s3 cp \
        "s3://${RUNPOD_NETWORK_VOLUME_ID}/${key}" - \
        --only-show-errors)"
    printf '%s' "${payload}" | python3 -c '
import json
import sys

label = sys.argv[1]
payload = json.load(sys.stdin)
fields = [
    "state=" + str(payload.get("state", "unknown")),
]
for key in (
    "selection_id",
    "dataset_profile",
    "wandb_run_id",
    "launch_id",
    "progress_path",
):
    value = payload.get(key)
    if value:
        fields.append(key + "=" + str(value))
print("{:<12} {}".format(label, " ".join(fields)))
' "${label}"
}

summarize_download_progress() {
    local dataset_payload=""
    local progress_key=""
    local progress_payload=""

    if ! dataset_payload="$(bash "${S3_WRAPPER}" s3 cp \
        "s3://${RUNPOD_NETWORK_VOLUME_ID}/lifecycle/stage1/dataset.json" - \
        --only-show-errors 2>/dev/null)"; then
        return 0
    fi
    progress_key="$(printf '%s' "${dataset_payload}" | python3 -c '
import json
import re
import sys

payload = json.load(sys.stdin)
path = str(payload.get("progress_path", ""))
match = re.fullmatch(
    r"/runpod-volume/(datasets/[0-9a-f]{64}/download-progress[.]json)",
    path,
)
if match is not None:
    print(match.group(1))
')"
    if [[ -z "${progress_key}" ]]; then
        return 0
    fi
    if ! progress_payload="$(bash "${S3_WRAPPER}" s3 cp \
        "s3://${RUNPOD_NETWORK_VOLUME_ID}/${progress_key}" - \
        --only-show-errors 2>/dev/null)"; then
        printf '%-12s unavailable key=%s\n' download "${progress_key}"
        return 0
    fi
    printf '%s' "${progress_payload}" | python3 -c '
import json
import sys

payload = json.load(sys.stdin)
if payload.get("schema_version") != 1 or payload.get("kind") != "ohlcv-download-progress":
    raise SystemExit("Download progress schema is unsupported")
attempt = payload.get("attempt") if isinstance(payload.get("attempt"), dict) else {}
cache = payload.get("cache") if isinstance(payload.get("cache"), dict) else {}
error = payload.get("last_error") if isinstance(payload.get("last_error"), dict) else {}
plan = payload.get("plan") if isinstance(payload.get("plan"), dict) else {}
fields = [
    "state=" + str(payload.get("state", "unknown")),
    "attempt=" + str(attempt.get("number", "unknown")),
    "cached_responses=" + str(cache.get("responses", "unknown")),
    "network_requests=" + str(attempt.get("network_requests", "unknown")),
]
estimated = plan.get("estimated_http_requests")
if estimated is not None:
    fields.append("estimated_requests=" + str(estimated))
for key in ("category", "provider", "status_code", "retry_after_seconds"):
    value = error.get(key)
    if value is not None:
        fields.append(key + "=" + str(value))
print("{:<12} {}".format("download", " ".join(fields)))
'
}

summarize_wandb() {
    local run_id=""
    local lifecycle_payload=""
    local status_key=""
    local status_payload=""

    for lifecycle_key in lifecycle/stage1/validation.json lifecycle/stage1/training.json; do
        if lifecycle_payload="$(bash "${S3_WRAPPER}" s3 cp \
            "s3://${RUNPOD_NETWORK_VOLUME_ID}/${lifecycle_key}" - \
            --only-show-errors 2>/dev/null)"; then
            run_id="$(printf '%s' "${lifecycle_payload}" | python3 -c '
import json
import re
import sys
payload = json.load(sys.stdin)
value = str(payload.get("wandb_run_id", ""))
if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,119}", value) and "--" not in value:
    print(value)
')"
        fi
        [[ -n "${run_id}" ]] && break
    done
    [[ -n "${run_id}" ]] || return 0
    status_key="lifecycle/runs/${run_id}/wandb.json"
    if ! status_payload="$(bash "${S3_WRAPPER}" s3 cp \
        "s3://${RUNPOD_NETWORK_VOLUME_ID}/${status_key}" - \
        --only-show-errors 2>/dev/null)"; then
        printf '%-12s missing run_id=%s\n' wandb "${run_id}"
        return 0
    fi
    printf '%s' "${status_payload}" | python3 -c '
import json
import sys
payload = json.load(sys.stdin)
components = payload.get("components", {})
fields = ["state=" + str(payload.get("state", "unknown")), "run_id=" + str(payload.get("run_id", ""))]
for name in ("training", "validation"):
    details = components.get(name)
    if isinstance(details, dict):
        fields.append(name + "=" + str(details.get("state", "unknown")))
print("{:<12} {}".format("wandb", " ".join(fields)))
'
}

summarize_marker lifecycle/stage1/code.json code
summarize_marker lifecycle/stage1/dataset.json dataset
summarize_download_progress
summarize_marker lifecycle/stage1/training.json training
summarize_marker lifecycle/stage1/validation.json validation
summarize_wandb
