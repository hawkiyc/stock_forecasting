#!/usr/bin/env bash
# Run the independent baseline workflow inside the existing persistent environment.
set -Eeuo pipefail
umask 077
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/runpod_paths.sh"
NETWORK_VOLUME_ROOT="${NETWORK_VOLUME_ROOT:-/runpod-volume}"
PROJECT_ROOT="${PROJECT_ROOT:-${NETWORK_VOLUME_ROOT}/stock_forecasting}"
[[ "${RUNPOD_ROLE:-}" == "gpu-baseline" ]] || { echo "Baseline workflow requires its dedicated GPU Pod" >&2; exit 2; }
bash "${SCRIPT_DIR}/verify_runpod_mounted_readiness.sh" --mount-only
runpod_acquire_gpu_workflow_lease "${NETWORK_VOLUME_ROOT}"
runpod_validate_path_in_root "${PROJECT_ROOT}" "${NETWORK_VOLUME_ROOT}" PROJECT_ROOT NETWORK_VOLUME_ROOT
CONFIG_PATH="${RUNPOD_CONFIG:?Immutable selection config is required}"
[[ "${CONFIG_PATH}" == /* ]] || CONFIG_PATH="${PROJECT_ROOT}/${CONFIG_PATH}"
runpod_validate_path_in_root "${CONFIG_PATH}" "${PROJECT_ROOT}" CONFIG_PATH PROJECT_ROOT
PROJECT_PYTHON="${PROJECT_ROOT}/.venv/bin/python"
[[ -x "${PROJECT_PYTHON}" ]] || { echo "Persistent project Poetry environment is missing" >&2; exit 127; }
bash "${SCRIPT_DIR}/verify_runpod_mounted_readiness.sh"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export RUNPOD_SELECTION_FILE="${RUNPOD_REMOTE_SELECTION_PATH:?Immutable remote selection is required}"
export TMPDIR="${NETWORK_VOLUME_ROOT}/tmp"
mkdir -p "${TMPDIR}"
cd "${PROJECT_ROOT}"
exec "${PROJECT_PYTHON}" -m stock_forecasting.cli.build_baselines --config "${CONFIG_PATH}"
