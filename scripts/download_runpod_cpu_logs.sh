#!/usr/bin/env bash

# Download the terminal CPU-preparation lifecycle and its tmux log directory.
set -Eeuo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
S3_WRAPPER="${SCRIPT_DIR}/runpod_s3_project.sh"
LOCAL_CPU_ROOT="${LOCAL_PROJECT_ROOT}/runpod_error_log_temp/cpu-prepare"
DATASET_KEY="lifecycle/stage1/dataset.json"

if [[ $# -ne 0 ]]; then
    printf 'Usage: bash scripts/download_runpod_cpu_logs.sh\n' >&2
    exit 2
fi

# shellcheck source=lib/runpod_project_env.sh
source "${SCRIPT_DIR}/lib/runpod_project_env.sh"
runpod_load_s3_env "${LOCAL_PROJECT_ROOT}"

mkdir -p "${LOCAL_CPU_ROOT}"
bash "${S3_WRAPPER}" s3 cp \
    "s3://${RUNPOD_NETWORK_VOLUME_ID}/${DATASET_KEY}" \
    "${LOCAL_CPU_ROOT}/dataset.json" \
    --only-show-errors

if ! command -v python3 >/dev/null 2>&1; then
    printf 'python3 is required to parse the CPU lifecycle\n' >&2
    exit 127
fi

LIFECYCLE_FIELDS="$(python3 - "${LOCAL_CPU_ROOT}/dataset.json" <<'PY'
import json
import re
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as handle:
    payload = json.load(handle)

state = str(payload.get("state", ""))
if state not in {"ready", "failed", "timed_out", "waiting_for_provider"}:
    raise SystemExit(f"CPU dataset lifecycle is not terminal: {state or '<missing>'}")

launch_id = str(payload.get("launch_id", ""))
if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,119}", launch_id) is None:
    raise SystemExit("CPU dataset lifecycle has no valid launch_id")

log_path = str(payload.get("log_path", ""))
expected_prefix = "/runpod-volume/logs/tmux/fin-ts-cpu-prepare/"
expected_log_path = expected_prefix + launch_id + "/combined.log"
if log_path != expected_log_path:
    raise SystemExit(f"Unexpected CPU log_path: {log_path or '<missing>'}")

print("\t".join((state, launch_id, log_path)))
PY
)"
IFS=$'\t' read -r CPU_STATE CPU_LAUNCH_ID CPU_LOG_PATH <<< "${LIFECYCLE_FIELDS}"

CPU_LOG_DIR="${CPU_LOG_PATH%/combined.log}"
CPU_LOG_KEY="${CPU_LOG_DIR#/runpod-volume/}"
LOCAL_CPU_LAUNCH_DIR="${LOCAL_CPU_ROOT}/${CPU_LAUNCH_ID}"
mkdir -p "${LOCAL_CPU_LAUNCH_DIR}"

bash "${S3_WRAPPER}" s3 cp \
    "s3://${RUNPOD_NETWORK_VOLUME_ID}/${CPU_LOG_KEY}/" \
    "${LOCAL_CPU_LAUNCH_DIR}/" \
    --recursive --only-show-errors

printf 'CPU lifecycle: %s\n' "${LOCAL_CPU_ROOT}/dataset.json"
printf 'CPU log directory: %s\n' "${LOCAL_CPU_LAUNCH_DIR}"
cat "${LOCAL_CPU_ROOT}/dataset.json"
find "${LOCAL_CPU_LAUNCH_DIR}" -maxdepth 3 -type f -print | sort

if [[ -f "${LOCAL_CPU_LAUNCH_DIR}/combined.log" ]]; then
    tail -n 200 "${LOCAL_CPU_LAUNCH_DIR}/combined.log"
fi
