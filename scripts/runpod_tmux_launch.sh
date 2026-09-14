#!/usr/bin/env bash

# Launch an allowlisted RunPod workflow in a detached tmux session with persistent logs.
set -Eeuo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Help must not import a Pod environment, acquire a lease, or start a session.
if [[ "${1:-}" == "probe-scales" ]]; then
    for argument in "$@"; do
        if [[ "${argument}" == "--help" || "${argument}" == "-h" ]]; then
            exec bash "${SCRIPT_DIR}/runpod_probe_scales.sh" --help
        fi
    done
fi
PID1_ENV_HELPER="${SCRIPT_DIR}/runpod_reexec_with_pid1_env.py"
PID1_IMPORT_PYTHON=/usr/local/bin/python
if [[ "${RUNPOD_TEST_MODE:-0}" == "1" \
    && -n "${RUNPOD_TEST_PID1_IMPORT_PYTHON:-}" ]]; then
    PID1_IMPORT_PYTHON="${RUNPOD_TEST_PID1_IMPORT_PYTHON}"
fi
if [[ ( "${RUNPOD_TEST_MODE:-0}" != "1" \
        || "${RUNPOD_TEST_FORCE_PID1_REEXEC:-0}" == "1" ) \
    && "${RUNPOD_SSH_ENV_IMPORTED:-0}" != "1" ]]; then
    if [[ ! -x "${PID1_IMPORT_PYTHON}" || ! -r "${PID1_ENV_HELPER}" ]]; then
        echo "RunPod PID 1 environment importer is unavailable" >&2
        exit 127
    fi
    exec "${PID1_IMPORT_PYTHON}" "${PID1_ENV_HELPER}" -- \
        bash "${BASH_SOURCE[0]}" "$@"
fi

# shellcheck source=lib/runpod_paths.sh
source "${SCRIPT_DIR}/lib/runpod_paths.sh"

NETWORK_VOLUME_ROOT="${NETWORK_VOLUME_ROOT:-${RUNPOD_VOLUME_ROOT:-/runpod-volume}}"
PROJECT_ROOT="${PROJECT_ROOT:-${NETWORK_VOLUME_ROOT}/stock_forecasting}"
LOG_ROOT="${LOG_ROOT:-${NETWORK_VOLUME_ROOT}/logs}"
READINESS_HELPER="${SCRIPT_DIR}/runpod_readiness.py"
PROBE_LIFECYCLE_HELPER="${SCRIPT_DIR}/runpod_probe_lifecycle.py"
PROBE_OWNER_RUN_ID=""
SELF_TERMINATE_SCRIPT="${SCRIPT_DIR}/runpod_self_terminate.sh"
RUNPOD_IMAGE_PYTHON="${RUNPOD_PYTHON_BIN:-/usr/local/bin/python}"
FINALIZE_LIFECYCLE_ON_EXIT=0
PROVIDER_WAIT_EXIT_ALLOWED=0
RUN_DIRECTORY_ID=""
PRESERVE_POD_ON_LAUNCH_ERROR=0
if [[ "${1:-}" == "probe-scales" ]]; then
    # A rejected diagnostic launch must not shut down an existing GPU workload.
    PRESERVE_POD_ON_LAUNCH_ERROR=1
fi

terminate_failed_launch() {
    local launch_exit_code=$?
    trap - EXIT
    if [[ ${launch_exit_code} -ne 0 \
        && ${PRESERVE_POD_ON_LAUNCH_ERROR} -ne 1 \
        && "${RUNPOD_POD_ID:-}" =~ ^[A-Za-z0-9_-]+$ \
        && "${RUNPOD_TEST_MODE:-0}" != "1" ]]; then
        if ! bash "${SELF_TERMINATE_SCRIPT}"; then
            echo "Early workflow launch failed and Pod self-termination also failed" >&2
            echo "External lifecycle guard remains armed" >&2
        fi
    fi
    exit "${launch_exit_code}"
}
trap terminate_failed_launch EXIT

runpod_validate_absolute_path "${NETWORK_VOLUME_ROOT}" NETWORK_VOLUME_ROOT
for path_name in PROJECT_ROOT LOG_ROOT; do
    path_value="${!path_name}"
    case "${path_value}" in
        /workspace|/workspace/*)
            echo "${path_name} must never use /workspace" >&2
            exit 2
            ;;
    esac
    runpod_validate_path_in_root \
        "${path_value}" "${NETWORK_VOLUME_ROOT}" "${path_name}" NETWORK_VOLUME_ROOT
done

if [[ $# -lt 1 || ( "$1" != "probe-scales" && $# -ne 1 ) ]]; then
    echo "Usage: runpod_tmux_launch.sh cpu-prepare|cpu-finalize|stage1-train|stage1-validate|probe-scales [PROBE OPTIONS]" >&2
    exit 2
fi
if [[ "${LOG_ROOT}" != "${NETWORK_VOLUME_ROOT}/logs" ]]; then
    echo "LOG_ROOT must equal NETWORK_VOLUME_ROOT/logs" >&2
    exit 2
fi
bash "${SCRIPT_DIR}/verify_runpod_mounted_readiness.sh" --mount-only
case "$1" in
    cpu-prepare)
        SESSION_NAME=fin-ts-cpu-prepare
        JOB_SCRIPT="${SCRIPT_DIR}/runpod_cpu_prepare.sh"
        JOB_ROLE=cpu-prep
        MAX_RUNTIME_SECONDS="${RUNPOD_CPU_MAX_RUNTIME_SECONDS:-21600}"
        JOB_TIMEOUT_GRACE_SECONDS=0
        FAILURE_LIFECYCLE_MARKER="${NETWORK_VOLUME_ROOT}/lifecycle/stage1/cpu-preparation.json"
        FAILURE_LIFECYCLE_KIND=stage1-cpu-preparation
        PROVIDER_WAIT_EXIT_ALLOWED=1
        ;;
    cpu-finalize)
        SESSION_NAME=fin-ts-cpu-finalize
        JOB_SCRIPT="${SCRIPT_DIR}/runpod_cpu_finalize.sh"
        JOB_ROLE=cpu-prep
        MAX_RUNTIME_SECONDS="${RUNPOD_CPU_FINALIZE_MAX_RUNTIME_SECONDS:-${RUNPOD_CPU_MAX_RUNTIME_SECONDS:-21600}}"
        JOB_TIMEOUT_GRACE_SECONDS=0
        FAILURE_LIFECYCLE_MARKER="${NETWORK_VOLUME_ROOT}/lifecycle/stage1/mixed-finalization.json"
        FAILURE_LIFECYCLE_KIND=stage1-mixed-finalization
        ;;
    stage1-train)
        SESSION_NAME=fin-ts-stage1-train
        JOB_SCRIPT="${SCRIPT_DIR}/runpod_entrypoint.sh"
        JOB_ROLE=gpu-train
        MAX_RUNTIME_SECONDS="${MAX_RUNTIME_SECONDS:-21600}"
        JOB_TIMEOUT_GRACE_SECONDS=300
        FAILURE_LIFECYCLE_MARKER="${NETWORK_VOLUME_ROOT}/lifecycle/stage1/training.json"
        FAILURE_LIFECYCLE_KIND=stage1-training
        FINALIZE_LIFECYCLE_ON_EXIT=1
        if [[ -z "${WANDB_RUN_ID:-}" ]]; then
            echo "WANDB_RUN_ID must be preallocated by create_runpod_pod.sh" >&2
            exit 2
        fi
        if [[ ! "${WANDB_RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,119}$ \
            || "${WANDB_RUN_ID}" == *--* ]]; then
            echo "WANDB_RUN_ID must be a safe 1-120 character directory name" >&2
            exit 2
        fi
        for run_id_name in RUNPOD_RUN_KEY VALIDATION_RUN_ID; do
            run_id_value="${!run_id_name:-}"
            if [[ -n "${run_id_value}" && "${run_id_value}" != "${WANDB_RUN_ID}" ]]; then
                echo "Training runtime IDs must identify the same run" >&2
                exit 2
            fi
        done
        RUNPOD_RUN_KEY="${WANDB_RUN_ID}"
        RUN_DIRECTORY_ID="${WANDB_RUN_ID}"
        ;;
    stage1-validate)
        SESSION_NAME=fin-ts-stage1-validate
        JOB_SCRIPT="${SCRIPT_DIR}/runpod_validation.sh"
        JOB_ROLE=gpu-validation
        MAX_RUNTIME_SECONDS="${RUNPOD_VALIDATION_MAX_RUNTIME_SECONDS:-${MAX_RUNTIME_SECONDS:-21600}}"
        JOB_TIMEOUT_GRACE_SECONDS=300
        FAILURE_LIFECYCLE_MARKER="${NETWORK_VOLUME_ROOT}/lifecycle/stage1/validation.json"
        FAILURE_LIFECYCLE_KIND=stage1-validation
        FINALIZE_LIFECYCLE_ON_EXIT=1
        for run_id_name in VALIDATION_RUN_ID WANDB_RUN_ID RUNPOD_RUN_KEY; do
            run_id_value="${!run_id_name:-}"
            if [[ -n "${run_id_value}" ]]; then
                if [[ ! "${run_id_value}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,119}$ \
                    || "${run_id_value}" == *--* ]]; then
                    echo "${run_id_name} must be a canonical single run ID" >&2
                    exit 2
                fi
                if [[ -n "${RUN_DIRECTORY_ID}" \
                    && "${run_id_value}" != "${RUN_DIRECTORY_ID}" ]]; then
                    echo "Validation runtime IDs must identify the same run" >&2
                    exit 2
                fi
                RUN_DIRECTORY_ID="${run_id_value}"
            fi
        done
        if [[ -z "${RUN_DIRECTORY_ID}" ]]; then
            TRAINING_LIFECYCLE_MARKER="${NETWORK_VOLUME_ROOT}/lifecycle/stage1/training.json"
            if [[ ! -r "${TRAINING_LIFECYCLE_MARKER}" ]]; then
                echo "Completed training lifecycle is unavailable for validation" >&2
                exit 2
            fi
            RUN_DIRECTORY_ID="$("${RUNPOD_IMAGE_PYTHON}" "${READINESS_HELPER}" \
                completed-training-run \
                --marker "${TRAINING_LIFECYCLE_MARKER}" \
                --network-volume-root "${NETWORK_VOLUME_ROOT}")"
        fi
        if [[ ! "${RUN_DIRECTORY_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,119}$ \
            || "${RUN_DIRECTORY_ID}" == *--* ]]; then
            echo "Validation run ID must be a safe 1-120 character directory name" >&2
            exit 2
        fi
        ;;
    probe-scales)
        SESSION_NAME=fin-ts-probe-scales
        JOB_SCRIPT="${SCRIPT_DIR}/runpod_probe_scales.sh"
        JOB_ROLE=gpu-probe
        MAX_RUNTIME_SECONDS="${MAX_RUNTIME_SECONDS:-21600}"
        JOB_TIMEOUT_GRACE_SECONDS=0
        FAILURE_LIFECYCLE_MARKER=""
        FAILURE_LIFECYCLE_KIND=""
        for run_id_name in VALIDATION_RUN_ID WANDB_RUN_ID RUNPOD_RUN_KEY; do
            run_id_value="${!run_id_name:-}"
            if [[ -n "${run_id_value}" ]]; then
                if [[ ! "${run_id_value}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,119}$ \
                    || "${run_id_value}" == *--* \
                    || ( -n "${PROBE_OWNER_RUN_ID}" && "${PROBE_OWNER_RUN_ID}" != "${run_id_value}" ) ]]; then
                    echo "Diagnostic Pod owner IDs must identify the same canonical run" >&2
                    exit 2
                fi
                PROBE_OWNER_RUN_ID="${run_id_value}"
            fi
        done
        if [[ -z "${PROBE_OWNER_RUN_ID}" || ! -r "${PROBE_LIFECYCLE_HELPER}" ]]; then
            echo "Diagnostic requires the Pod owner run ID and local-guard lifecycle helper" >&2
            exit 2
        fi
        ;;
    *)
        echo "Usage: runpod_tmux_launch.sh cpu-prepare|cpu-finalize|stage1-train|stage1-validate|probe-scales [PROBE OPTIONS]" >&2
        exit 2
        ;;
esac
shift

if [[ -z "${RUNPOD_POD_ID:-}" && "${RUNPOD_TEST_MODE:-0}" != "1" ]]; then
    echo "tmux workflows must run inside a RunPod Pod" >&2
    exit 2
fi
runpod_validate_path_in_root \
    "${JOB_SCRIPT}" "${PROJECT_ROOT}" JOB_SCRIPT PROJECT_ROOT
if [[ -n "${FAILURE_LIFECYCLE_MARKER}" ]]; then
    runpod_validate_path_in_root \
        "${FAILURE_LIFECYCLE_MARKER}" "${NETWORK_VOLUME_ROOT}" \
        FAILURE_LIFECYCLE_MARKER NETWORK_VOLUME_ROOT
fi
if [[ ! "${MAX_RUNTIME_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Workflow runtime limit must be a positive integer" >&2
    exit 2
fi
JOB_TIMEOUT_SECONDS=$((MAX_RUNTIME_SECONDS + JOB_TIMEOUT_GRACE_SECONDS))
if [[ ! -f "${JOB_SCRIPT}" ]]; then
    echo "RunPod job script is missing: ${JOB_SCRIPT}" >&2
    exit 127
fi
if ! command -v timeout >/dev/null 2>&1; then
    echo "GNU timeout is required in the selected RunPod image" >&2
    exit 127
fi

bash "${SCRIPT_DIR}/ensure_runpod_tmux.sh"

TMUX_SOCKET="${SESSION_NAME}"
if tmux -L "${TMUX_SOCKET}" has-session -t "${SESSION_NAME}" 2>/dev/null; then
    PRESERVE_POD_ON_LAUNCH_ERROR=1
    echo "tmux session already exists: ${SESSION_NAME}" >&2
    echo "Attach with: tmux -L ${TMUX_SOCKET} attach -t ${SESSION_NAME}" >&2
    exit 3
fi

LAUNCH_ID="launch-$(date -u +%Y%m%dT%H%M%SZ)-${RANDOM}${RANDOM}"
if [[ -n "${RUN_DIRECTORY_ID}" ]]; then
    JOB_DIR="${LOG_ROOT}/${RUN_DIRECTORY_ID}/tmux/${SESSION_NAME}/${LAUNCH_ID}"
else
    JOB_DIR="${LOG_ROOT}/tmux/${SESSION_NAME}/${LAUNCH_ID}"
fi
POD_TERMINAL_ROOT=""
if [[ "${JOB_ROLE}" == "gpu-train" || "${JOB_ROLE}" == "gpu-validation" ]]; then
    POD_TERMINAL_ROOT="${NETWORK_VOLUME_ROOT}/lifecycle/runs/${RUN_DIRECTORY_ID}/pods"
fi
JOB_LOG="${JOB_DIR}/combined.log"
JOB_STATUS="${JOB_DIR}/status.json"
RUNNER="${JOB_DIR}/runner.sh"
RUNNER_TMP="${RUNNER}.tmp.$$.$RANDOM"
for path_name in JOB_DIR JOB_LOG JOB_STATUS RUNNER RUNNER_TMP; do
    runpod_validate_path_in_root \
        "${!path_name}" "${LOG_ROOT}" "${path_name}" LOG_ROOT
done
if [[ -n "${POD_TERMINAL_ROOT}" ]]; then
    runpod_validate_path_in_root \
        "${POD_TERMINAL_ROOT}" "${NETWORK_VOLUME_ROOT}" \
        POD_TERMINAL_ROOT NETWORK_VOLUME_ROOT
fi
mkdir -p "${JOB_DIR}"

{
    printf '#!/usr/bin/env bash\n'
    printf 'set -uo pipefail\n'
    printf 'umask 077\n'
    printf 'export RUNPOD_ROLE=%q\n' "${JOB_ROLE}"
    printf 'export RUNPOD_LAUNCH_ID=%q\n' "${LAUNCH_ID}"
    printf 'export RUNPOD_TMUX_LOG_FILE=%q\n' "${JOB_LOG}"
    if [[ "${JOB_ROLE}" == "gpu-train" ]]; then
        printf 'export WANDB_RUN_ID=%q\n' "${RUN_DIRECTORY_ID}"
        printf 'export RUNPOD_RUN_KEY=%q\n' "${RUN_DIRECTORY_ID}"
    elif [[ "${JOB_ROLE}" == "gpu-validation" ]]; then
        printf 'export VALIDATION_RUN_ID=%q\n' "${RUN_DIRECTORY_ID}"
        printf 'export WANDB_RUN_ID=%q\n' "${RUN_DIRECTORY_ID}"
        printf 'export RUNPOD_RUN_KEY=%q\n' "${RUN_DIRECTORY_ID}"
    elif [[ "${JOB_ROLE}" == "gpu-probe" ]]; then
        printf 'export RUNPOD_SCALE_PROBE_TMUX_WORKER=1\n'
    fi
    printf 'exec > >(tee -a %q) 2>&1\n' "${JOB_LOG}"
    printf 'printf '\''[%%s] tmux workflow started\\n'\'' "$(date -u +%%Y-%%m-%%dT%%H:%%M:%%SZ)"\n'
    printf 'set +e\n'
    if [[ "${JOB_ROLE}" == "gpu-probe" ]]; then
        # Hold the lease in the runner, not only in its model subprocess. It must
        # survive model exit until status publication and Pod termination finish.
        printf 'source %q\n' "${SCRIPT_DIR}/lib/runpod_paths.sh"
        printf 'publish_probe_state() {\n'
        printf '  local attempt=1\n'
        printf '  while [[ ${attempt} -le 3 ]]; do\n'
        printf '    if %q %q write-state --network-volume-root %q --pod-id %q --owner-run-id %q --launch-id %q --state "$1" --exit-code "$2"; then return 0; fi\n' \
            "${RUNPOD_IMAGE_PYTHON}" "${PROBE_LIFECYCLE_HELPER}" "${NETWORK_VOLUME_ROOT}" \
            "${RUNPOD_POD_ID:-}" "${PROBE_OWNER_RUN_ID}" "${LAUNCH_ID}"
        printf '    attempt=$((attempt + 1))\n'
        printf '  done\n'
        printf '  return 74\n'
        printf '}\n'
        printf 'unset RUNPOD_GPU_WORKFLOW_LEASE_HELD\n'
        printf 'diagnostic_lease_acquired=0\n'
        printf 'runpod_acquire_gpu_workflow_lease %q\n' "${NETWORK_VOLUME_ROOT}"
        printf 'job_exit_code=$?\n'
        printf 'if [[ ${job_exit_code} -eq 0 ]]; then\n'
        printf '  diagnostic_lease_acquired=1\n'
        printf '  if publish_probe_state running 0; then\n'
    fi
    printf 'timeout --signal=TERM --kill-after=60 %qs bash %q' \
        "${JOB_TIMEOUT_SECONDS}" "${JOB_SCRIPT}"
    if [[ $# -gt 0 ]]; then
        printf ' %q' "$@"
    fi
    printf '\n'
    printf 'job_exit_code=$?\n'
    if [[ "${JOB_ROLE}" == "gpu-probe" ]]; then
        printf '  else job_exit_code=74; fi\n'
        printf 'fi\n'
    fi
    printf 'finalization_exit_code=0\n'
    printf 'terminal_lifecycle_allowed=1\n'
    printf 'if [[ ${job_exit_code} -eq 75 ]]; then terminal_lifecycle_allowed=0; fi\n'
    printf 'training_phase_completed=0\n'
    printf 'finalization_retry_delay=1\n'
    printf 'if [[ "${RUNPOD_TEST_MODE:-0}" == 1 ]]; then finalization_retry_delay=0; fi\n'
    printf 'status_state=failed\n'
    printf 'if [[ ${job_exit_code} -eq 0 ]]; then status_state=succeeded; fi\n'
    printf 'if [[ ${job_exit_code} -eq 124 ]]; then status_state=timed_out; fi\n'
    if [[ ${PROVIDER_WAIT_EXIT_ALLOWED} -eq 1 ]]; then
        printf 'if [[ ${job_exit_code} -eq 75 ]]; then status_state=resumable; fi\n'
    fi
    printf 'status_tmp=%q\n' "${JOB_STATUS}.tmp.$$.$RANDOM"
    printf 'publish_status() {\n'
    printf '  local state="$1" exit_code="$2" finalization_code="$3"\n'
    printf '  local finalization_attempt=1\n'
    printf '  while [[ ${finalization_attempt} -le 3 ]]; do\n'
    printf '    if printf '\''{"state":"%%s","exit_code":%%d,"ended_at":"%%s","log_path":"%%s","finalization_exit_code":%%d}\\n'\'' "${state}" "${exit_code}" "$(date -u +%%Y-%%m-%%dT%%H:%%M:%%SZ)" %q "${finalization_code}" > "${status_tmp}" && mv "${status_tmp}" %q; then\n' \
        "${JOB_LOG}" "${JOB_STATUS}"
    printf '      return 0\n'
    printf '    fi\n'
    printf '    sleep "$((finalization_attempt * finalization_retry_delay))"\n'
    printf '    finalization_attempt=$((finalization_attempt + 1))\n'
    printf '  done\n'
    printf '  return 1\n'
    printf '}\n'
    printf 'publish_failed_status() {\n'
    printf '  local effective_exit_code=${job_exit_code}\n'
    printf '  if [[ ${effective_exit_code} -eq 0 ]]; then effective_exit_code=74; fi\n'
    printf '  if publish_status failed "${effective_exit_code}" 74; then return 0; fi\n'
    printf '  printf '\''Unable to publish failed workflow status after three attempts\\n'\'' >&2\n'
    printf '  terminal_lifecycle_allowed=0\n'
    printf '  return 1\n'
    printf '}\n'
    if [[ "${JOB_ROLE}" == "gpu-train" || "${JOB_ROLE}" == "gpu-validation" ]]; then
        printf 'publish_lease_contention_signal() {\n'
        printf '  if [[ ${job_exit_code} -ne 75 ]]; then return 0; fi\n'
        printf '  local pod_id="${RUNPOD_POD_ID:-}"\n'
        printf '  if [[ ! "${pod_id}" =~ ^[A-Za-z0-9_-]+$ ]]; then\n'
        printf '    printf '\''Unable to publish GPU lease-contention signal: invalid Pod ID\\n'\'' >&2\n'
        printf '    return 1\n'
        printf '  fi\n'
        printf '  local signal_dir=%q\n' "${POD_TERMINAL_ROOT}"
        printf '  local signal_path="${signal_dir}/${pod_id}/terminal.json"\n'
        printf '  local signal_tmp="${signal_path}.tmp.$$.$RANDOM"\n'
        printf '  local signal_attempt=1\n'
        printf '  while [[ ${signal_attempt} -le 3 ]]; do\n'
        printf '    if mkdir -p "$(dirname "${signal_path}")" && printf '\''{"schema_version":1,"kind":"gpu-workflow-terminal","lifecycle_kind":"%%s","state":"failed","reason":"gpu_workflow_lease_contended","exit_code":75,"pod_id":"%%s","wandb_run_id":"%%s","launch_id":"%%s","generated_at":"%%s"}\\n'\'' %q "${pod_id}" %q %q "$(date -u +%%Y-%%m-%%dT%%H:%%M:%%SZ)" > "${signal_tmp}" && mv "${signal_tmp}" "${signal_path}"; then\n' \
            "${FAILURE_LIFECYCLE_KIND}" "${RUN_DIRECTORY_ID}" "${LAUNCH_ID}"
        printf '      printf '\''Published GPU lease-contention signal: %%s\\n'\'' "${signal_path}"\n'
        printf '      return 0\n'
        printf '    fi\n'
        printf '    sleep "$((signal_attempt * finalization_retry_delay))"\n'
        printf '    signal_attempt=$((signal_attempt + 1))\n'
        printf '  done\n'
        printf '  return 1\n'
        printf '}\n'
    fi
    printf 'publish_lifecycle() {\n'
    printf '  local marker="$1" kind="$2" state="$3" lifecycle_exit_code="${4:-${job_exit_code}}"\n'
    printf '  local finalization_attempt=1\n'
    printf '  while [[ ${finalization_attempt} -le 3 ]]; do\n'
    printf '    if %q %q write-state --output "${marker}" --network-volume-root %q --kind "${kind}" --state "${state}" --launch-id %q --exit-code "${lifecycle_exit_code}" --log-path %q --max-runtime-seconds %q --wandb-run-id %q --inherit-existing; then\n' \
        "${RUNPOD_IMAGE_PYTHON}" "${READINESS_HELPER}" \
        "${NETWORK_VOLUME_ROOT}" "${LAUNCH_ID}" "${JOB_LOG}" \
        "${MAX_RUNTIME_SECONDS}" "${RUN_DIRECTORY_ID}"
    printf '      return 0\n'
    printf '    fi\n'
    printf '    sleep "$((finalization_attempt * finalization_retry_delay))"\n'
    printf '    finalization_attempt=$((finalization_attempt + 1))\n'
    printf '  done\n'
    printf '  return 1\n'
    printf '}\n'
    printf 'finalize_lifecycle() {\n'
    printf '  local marker="$1" kind="$2" intended_state="$3" label="$4"\n'
    printf '  if publish_lifecycle "${marker}" "${kind}" "${intended_state}"; then return 0; fi\n'
    printf '  printf '\''Unable to publish %%s lifecycle after three attempts\\n'\'' "${label}" >&2\n'
    printf '  finalization_exit_code=74\n'
    printf '  if ! publish_failed_status; then return 1; fi\n'
    printf '  if publish_lifecycle "${marker}" "${kind}" failed 74; then return 1; fi\n'
    printf '  printf '\''Unable to publish failed %%s lifecycle after three attempts\\n'\'' "${label}" >&2\n'
    printf '  terminal_lifecycle_allowed=0\n'
    printf '  return 1\n'
    printf '}\n'
    if [[ "${JOB_ROLE}" == "gpu-train" ]]; then
        printf 'if %q %q training-phase-completed --marker %q --network-volume-root %q --run-id %q --launch-id %q >/dev/null 2>&1; then\n' \
            "${RUNPOD_IMAGE_PYTHON}" "${READINESS_HELPER}" \
            "${FAILURE_LIFECYCLE_MARKER}" "${NETWORK_VOLUME_ROOT}" \
            "${RUN_DIRECTORY_ID}" "${LAUNCH_ID}"
        printf '  training_phase_completed=1\n'
        printf 'fi\n'
    fi
    printf 'if ! publish_status "${status_state}" "${job_exit_code}" 0; then\n'
    printf '  printf '\''Unable to publish initial workflow status after three attempts\\n'\'' >&2\n'
    printf '  finalization_exit_code=74\n'
    printf '  if ! publish_failed_status; then :; fi\n'
    printf 'fi\n'
    if [[ "${JOB_ROLE}" == "gpu-train" || "${JOB_ROLE}" == "gpu-validation" ]]; then
        printf 'if [[ ${job_exit_code} -eq 75 ]]; then\n'
        printf '  if ! publish_lease_contention_signal; then\n'
        printf '    printf '\''Unable to persist GPU lease-contention signal\\n'\'' >&2\n'
        printf '  fi\n'
        printf 'fi\n'
    fi
    if [[ "${JOB_ROLE}" == "gpu-train" ]]; then
        printf 'validation_marker=%q\n' \
            "${NETWORK_VOLUME_ROOT}/lifecycle/stage1/validation.json"
        printf 'if [[ -f "${validation_marker}" ]] && %q %q active-run-lifecycle --marker "${validation_marker}" --network-volume-root %q --kind stage1-validation --run-id %q --launch-id %q >/dev/null 2>&1; then\n' \
            "${RUNPOD_IMAGE_PYTHON}" "${READINESS_HELPER}" \
            "${NETWORK_VOLUME_ROOT}" "${RUN_DIRECTORY_ID}" "${LAUNCH_ID}"
        printf '  validation_lifecycle_state=failed\n'
        printf '  if [[ ${job_exit_code} -eq 0 ]]; then validation_lifecycle_state=ready; fi\n'
        printf '  if [[ ${job_exit_code} -eq 124 ]]; then validation_lifecycle_state=timed_out; fi\n'
        printf '  if [[ ${finalization_exit_code} -ne 0 ]]; then validation_lifecycle_state=failed; fi\n'
        printf '  if [[ ${terminal_lifecycle_allowed} -eq 1 ]]; then\n'
        printf '    if ! finalize_lifecycle "${validation_marker}" stage1-validation "${validation_lifecycle_state}" validation; then :; fi\n'
        printf '  fi\n'
        printf 'fi\n'
    fi
    printf 'cpu_resumable_lifecycle_valid=0\n'
    if [[ ${PROVIDER_WAIT_EXIT_ALLOWED} -eq 1 ]]; then
        printf 'if [[ ${job_exit_code} -eq 75 ]] && %q %q resumable-cpu-preparation-lifecycle --marker %q --network-volume-root %q --launch-id %q >/dev/null 2>&1; then\n' \
            "${RUNPOD_IMAGE_PYTHON}" "${READINESS_HELPER}" \
            "${FAILURE_LIFECYCLE_MARKER}" "${NETWORK_VOLUME_ROOT}" "${LAUNCH_ID}"
        printf '  cpu_resumable_lifecycle_valid=1\n'
        printf 'fi\n'
    fi
    if [[ -n "${FAILURE_LIFECYCLE_MARKER:-}" ]]; then
        if [[ ${PROVIDER_WAIT_EXIT_ALLOWED} -eq 1 ]]; then
            # The CPU worker publishes the precise waiting state and progress path.
            printf 'if [[ ${cpu_resumable_lifecycle_valid} -ne 1 && ( ${job_exit_code} -ne 0 || %q == 1 ) && ${terminal_lifecycle_allowed} -eq 1 ]]; then\n' \
                "${FINALIZE_LIFECYCLE_ON_EXIT}"
        else
            printf 'if [[ ( ${job_exit_code} -ne 0 || %q == 1 ) && ${terminal_lifecycle_allowed} -eq 1 ]]; then\n' \
                "${FINALIZE_LIFECYCLE_ON_EXIT}"
        fi
        printf '  lifecycle_state=failed\n'
        printf '  if [[ ${job_exit_code} -eq 0 ]]; then lifecycle_state=ready; fi\n'
        printf '  if [[ ${job_exit_code} -eq 124 ]]; then lifecycle_state=timed_out; fi\n'
        if [[ "${FAILURE_LIFECYCLE_KIND}" == "stage1-training" ]]; then
            printf '  if [[ ${training_phase_completed} -eq 1 ]]; then lifecycle_state=ready; fi\n'
        fi
        printf '  if [[ ${finalization_exit_code} -ne 0 ]]; then lifecycle_state=failed; fi\n'
        printf '  if ! finalize_lifecycle %q %q "${lifecycle_state}" workflow; then :; fi\n' \
            "${FAILURE_LIFECYCLE_MARKER}" "${FAILURE_LIFECYCLE_KIND}"
        printf 'fi\n'
    fi
    printf 'if [[ ${finalization_exit_code} -ne 0 ]]; then\n'
    printf '  if [[ ${job_exit_code} -eq 0 ]]; then job_exit_code=${finalization_exit_code}; fi\n'
    printf 'fi\n'
    if [[ "${JOB_ROLE}" == "gpu-probe" ]]; then
        printf 'if [[ ${diagnostic_lease_acquired} -ne 1 ]]; then\n'
        printf '  printf '\''Diagnostic did not acquire the GPU lease; preserving the Pod and existing work\\n'\''\n'
        printf '  exit "${job_exit_code}"\n'
        printf 'fi\n'
        printf 'probe_terminal_state=failed\n'
        printf 'if [[ ${job_exit_code} -eq 0 ]]; then probe_terminal_state=succeeded; fi\n'
        printf 'if [[ ${job_exit_code} -eq 124 ]]; then probe_terminal_state=timed_out; fi\n'
        printf 'if ! publish_probe_state "${probe_terminal_state}" "${job_exit_code}"; then\n'
        printf '  printf '\''Diagnostic terminal signal could not be saved; local guard hard limit remains the fallback\\n'\'' >&2\n'
        printf '  if [[ ${job_exit_code} -eq 0 ]]; then job_exit_code=74; fi\n'
        printf 'fi\n'
        printf 'printf '\''Diagnostic finished; awaiting Pod termination by the local guard\\n'\''\n'
        # Retain the GPU lease until the local guard removes the Pod; another
        # workload must not start between terminal publication and local polling.
        printf 'while sleep 60; do :; done\n'
    else
        printf 'export RUNPOD_SHUTDOWN_DIR=%q\n' "${JOB_DIR}/pod-shutdown"
        printf 'export RUNPOD_SHUTDOWN_MARKER=%q\n' "${JOB_DIR}/pod-shutdown/shutdown.json"
        printf 'if ! bash %q; then\n' "${SELF_TERMINATE_SCRIPT}"
        printf '  printf '\''Pod self-termination failed; external lifecycle guard remains armed\\n'\'' >&2\n'
        printf 'fi\n'
    fi
    printf 'exit "${job_exit_code}"\n'
} > "${RUNNER_TMP}"
mv "${RUNNER_TMP}" "${RUNNER}"
chmod 700 "${RUNNER}"

tmux -L "${TMUX_SOCKET}" new-session -d -s "${SESSION_NAME}" -c "${PROJECT_ROOT}"
tmux -L "${TMUX_SOCKET}" set-option -t "${SESSION_NAME}" remain-on-exit on
tmux -L "${TMUX_SOCKET}" send-keys -t "${SESSION_NAME}" -l "exec bash ${RUNNER}"
tmux -L "${TMUX_SOCKET}" send-keys -t "${SESSION_NAME}" Enter
tmux -L "${TMUX_SOCKET}" has-session -t "${SESSION_NAME}"
trap - EXIT

printf 'Detached tmux session started: %s\n' "${SESSION_NAME}"
printf 'Attach: tmux -L %s attach -t %s\n' "${TMUX_SOCKET}" "${SESSION_NAME}"
printf 'Detach without stopping the job: Ctrl-b d\n'
printf 'Persistent log: %s\n' "${JOB_LOG}"
printf 'Job status: %s\n' "${JOB_STATUS}"
