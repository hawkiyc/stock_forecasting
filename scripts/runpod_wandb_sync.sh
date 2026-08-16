#!/usr/bin/env bash

# Sync preserved offline or incomplete W&B transactions from the network volume.
set -Eeuo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID1_ENV_HELPER="${SCRIPT_DIR}/runpod_reexec_with_pid1_env.py"
PID1_IMPORT_PYTHON="${RUNPOD_PYTHON_BIN:-/usr/local/bin/python}"
if [[ "${RUNPOD_SSH_ENV_IMPORTED:-0}" != "1" ]]; then
    exec "${PID1_IMPORT_PYTHON}" "${PID1_ENV_HELPER}" -- bash "${BASH_SOURCE[0]}" "$@"
fi

terminate_sync_pod() {
    local sync_exit_code=$?
    trap - EXIT
    bash "${SCRIPT_DIR}/runpod_self_terminate.sh" || true
    exit "${sync_exit_code}"
}
trap terminate_sync_pod EXIT

NETWORK_VOLUME_ROOT="${NETWORK_VOLUME_ROOT:-${RUNPOD_VOLUME_ROOT:-/runpod-volume}}"
PROJECT_ROOT="${PROJECT_ROOT:-${NETWORK_VOLUME_ROOT}/stock_forecasting}"
PROJECT_PYTHON="${PROJECT_ROOT}/.venv/bin/python"
if [[ "${RUNPOD_ROLE:-}" != "gpu-train" && "${RUNPOD_ROLE:-}" != "gpu-validation" ]]; then
    echo "W&B sync must run in a GPU Pod that has the W&B RunPod Secret" >&2
    exit 2
fi
if [[ ! -x "${PROJECT_PYTHON}" ]]; then
    echo "Persistent project Python is unavailable: ${PROJECT_PYTHON}" >&2
    exit 127
fi
bash "${SCRIPT_DIR}/verify_runpod_mounted_readiness.sh" --mount-only
set +e
"${PROJECT_PYTHON}" -m stock_forecasting.cli.sync_wandb "$@"
SYNC_EXIT_CODE=$?
set -e
exit "${SYNC_EXIT_CODE}"
