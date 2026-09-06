#!/usr/bin/env bash

# Run an independent frozen-checkpoint diagnostic in an existing GPU Pod.
set -Eeuo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/runpod_paths.sh
source "${SCRIPT_DIR}/lib/runpod_paths.sh"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'EOF'
Usage (inside an existing CUDA RunPod Pod):
  bash scripts/runpod_workflow.sh probe-scales --checkpoint /runpod-volume/savedModel/<run-id> [OPTIONS]

Options:
  --train-samples N       Maximum probe train cutoffs (default: 16384)
  --validation-samples N  Maximum validation cutoffs (default: 4096)
  --batch-size N          Frozen extraction batch size (default: 16)
  --num-workers N         Data loader workers, 0-16 (default: 0)
  --ridge-alpha X         Fixed positive ridge penalty (default: 10)
  --seed N                Reproducible probe seed (default: 42)

Use a canonical run directory (validation-selected best) or exact retained checkpoint.
The resolved checkpoint config is used, never the active stage selection.
Results: NETWORK_VOLUME_ROOT/diagnostics/representation-scales/<run>/<checkpoint>/probe-*/
This foreground command does not create/terminate a Pod or alter lifecycle completion markers.
No test split, W&B run, training optimizer, or checkpoint update is involved.
EOF
    exit 0
fi

# SSH sessions may omit the Pod environment. Reuse the existing allowlisted
# importer, without executing Python on the local macOS control machine.
if [[ "${RUNPOD_TEST_MODE:-0}" != "1" \
    && "${RUNPOD_SSH_ENV_IMPORTED:-0}" != "1" \
    && -r /proc/1/environ ]]; then
    if [[ ! -x /usr/local/bin/python ]]; then
        echo "RunPod PID 1 environment importer is unavailable" >&2
        exit 127
    fi
    exec /usr/local/bin/python "${SCRIPT_DIR}/runpod_reexec_with_pid1_env.py" -- \
        bash "${BASH_SOURCE[0]}" "$@"
fi

if [[ -z "${RUNPOD_POD_ID:-}" ]]; then
    echo "Scale probing must run inside an existing CUDA RunPod Pod; no local model execution" >&2
    exit 2
fi
if [[ $# -lt 2 || "$1" != "--checkpoint" ]]; then
    echo "Pass --checkpoint /runpod-volume/savedModel/<run-id> first; use --help for options" >&2
    exit 2
fi

NETWORK_VOLUME_ROOT="${NETWORK_VOLUME_ROOT:-${RUNPOD_VOLUME_ROOT:-/runpod-volume}}"
PROJECT_ROOT="${PROJECT_ROOT:-${NETWORK_VOLUME_ROOT}/stock_forecasting}"
CACHE_ROOT="${CACHE_ROOT:-${NETWORK_VOLUME_ROOT}/cache}"
SAVED_MODEL_ROOT="${SAVED_MODEL_ROOT:-${NETWORK_VOLUME_ROOT}/savedModel}"
case "${NETWORK_VOLUME_ROOT}" in
    /|/workspace|/workspace/*)
        echo "Scale probing requires a persistent network volume, not / or /workspace" >&2
        exit 2
        ;;
esac
for path_name in PROJECT_ROOT CACHE_ROOT SAVED_MODEL_ROOT; do
    runpod_validate_path_in_root \
        "${!path_name}" "${NETWORK_VOLUME_ROOT}" "${path_name}" NETWORK_VOLUME_ROOT
done
if [[ "${SAVED_MODEL_ROOT}" != "${NETWORK_VOLUME_ROOT}/savedModel" ]]; then
    echo "SAVED_MODEL_ROOT must equal NETWORK_VOLUME_ROOT/savedModel" >&2
    exit 2
fi
runpod_validate_path_in_root "$2" "${SAVED_MODEL_ROOT}" CHECKPOINT SAVED_MODEL_ROOT
export NETWORK_VOLUME_ROOT PROJECT_ROOT CACHE_ROOT SAVED_MODEL_ROOT
export RUNPOD_VOLUME_ROOT="${NETWORK_VOLUME_ROOT}"

# Verify the exact mount and share the existing exclusive GPU lease, but do not run
# training/validation readiness transitions or rewrite their lifecycle markers.
bash "${SCRIPT_DIR}/verify_runpod_mounted_readiness.sh" --mount-only
runpod_acquire_gpu_workflow_lease "${NETWORK_VOLUME_ROOT}"

export HF_HOME="${CACHE_ROOT}/huggingface"
export TORCH_HOME="${CACHE_ROOT}/torch"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
export TMPDIR="${NETWORK_VOLUME_ROOT}/tmp"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
if [[ ! -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
    echo "Persistent project .venv is unavailable; use the existing cloud setup workflow" >&2
    exit 127
fi
cd "${PROJECT_ROOT}"
exec "${PROJECT_ROOT}/.venv/bin/python" -m stock_forecasting.cli.probe_scales "$@"
