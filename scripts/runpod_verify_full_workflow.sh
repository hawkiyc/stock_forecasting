#!/usr/bin/env bash
# Authorized synthetic verification only: no production training or market-data calls.
set -Eeuo pipefail
umask 077
main() {
# Parse the full body before executing; never hot-edit a running shell workflow.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/runpod_paths.sh"
NETWORK_VOLUME_ROOT="${NETWORK_VOLUME_ROOT:-/runpod-volume}"
PROJECT_ROOT="${PROJECT_ROOT:-${NETWORK_VOLUME_ROOT}/stock_forecasting}"
[[ "${RUNPOD_ROLE:-}" == "gpu-baseline" ]] || exit 2
[[ "${MAX_RUNTIME_SECONDS:?}" -le 2700 ]] || { echo "Verification requires a workload limit of at most 45 minutes" >&2; exit 2; }
bash "${SCRIPT_DIR}/verify_runpod_mounted_readiness.sh"
runpod_acquire_gpu_workflow_lease "${NETWORK_VOLUME_ROOT}"
cd "${PROJECT_ROOT}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export HF_HOME="${NETWORK_VOLUME_ROOT}/cache/huggingface"
OUTPUT="${NETWORK_VOLUME_ROOT}/diagnostics/full-workflow/${WANDB_RUN_ID:?}"
mkdir -p "${OUTPUT}"
# Short, Pod-local scratch avoids AF_UNIX path limits and repeated fixture-copy
# traffic on the network volume. Persistent evidence stays in OUTPUT.
export TMPDIR="$(mktemp -d /tmp/fin-ts-qa.XXXXXXXX)"
# No completion manifest is written under baselines/. This is not a baseline build.
set +e
timeout --kill-after=15s 1200 .venv/bin/python -m pytest tests \
    -o faulthandler_timeout=120 --junitxml="${OUTPUT}/pytest.xml" >"${OUTPUT}/pytest.log" 2>&1
test_exit=$?
timeout --kill-after=15s 120 .venv/bin/ruff check . >"${OUTPUT}/ruff.log" 2>&1
lint_exit=$?
timeout --kill-after=15s 420 .venv/bin/python scripts/verify_kronos_full_workflow.py \
    >"${OUTPUT}/kronos-smoke.log" 2>&1
kronos_exit=$?
timeout --kill-after=15s 300 .venv/bin/python scripts/verify_full_evaluation_capacity.py \
    --output "${OUTPUT}/capacity" >"${OUTPUT}/capacity.log" 2>&1
capacity_exit=$?
set -e
printf '{"pytest":%s,"ruff":%s,"kronos":%s,"capacity":%s}\n' \
    "${test_exit}" "${lint_exit}" "${kronos_exit}" "${capacity_exit}" >"${OUTPUT}/acceptance-status.json"
tail -n 65 "${OUTPUT}/pytest.log"
tail -n 35 "${OUTPUT}/ruff.log"
tail -n 30 "${OUTPUT}/kronos-smoke.log"
tail -n 30 "${OUTPUT}/capacity.log"
printf 'Synthetic verification: pytest=%s ruff=%s output=%s\n' "${test_exit}" "${lint_exit}" "${OUTPUT}"
# Keep the authorized debugging window bounded by tmux timeout and the local
# hard-limit guard. The control host publishes the terminal lifecycle after QA.
printf 'Awaiting control-host QA completion; the hard deadline remains armed.\n'
while sleep 10; do :; done
}

main "$@"
