#!/usr/bin/env bash

# Launch a detached local lifecycle guard and fail closed if it is not armed.
set -Eeuo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
GUARD_SCRIPT="${SCRIPT_DIR}/terminate_runpod_after.sh"
RUNPODCTL_WRAPPER="${SCRIPT_DIR}/runpodctl_project.sh"
# shellcheck source=lib/runpod_project_env.sh
source "${SCRIPT_DIR}/lib/runpod_project_env.sh"

if [[ $# -ne 4 ]]; then
    echo "Usage: launch_runpod_guard.sh POD_ID DELAY_SECONDS LIFECYCLE_KEY LOG_FILE" >&2
    exit 2
fi

POD_ID="$1"
DELAY_SECONDS="$2"
LIFECYCLE_KEY="$3"
LOG_FILE="$4"
STARTUP_TIMEOUT_SECONDS="${RUNPOD_GUARD_STARTUP_TIMEOUT_SECONDS:-15}"
RUNPOD_GUARD_VOLUME_ROOT="${RUNPOD_GUARD_VOLUME_ROOT:-/runpod-volume}"
RUNPOD_GUARD_RUN_ID="${RUNPOD_GUARD_RUN_ID:-}"
RUNPOD_GUARD_KEEP_AWAKE="${RUNPOD_GUARD_KEEP_AWAKE:-auto}"
RUNPOD_GUARD_EMERGENCY_TERMINATE_ON_STARTUP_FAILURE="${RUNPOD_GUARD_EMERGENCY_TERMINATE_ON_STARTUP_FAILURE:-1}"

if [[ ! "${POD_ID}" =~ ^[A-Za-z0-9_-]+$ \
    || ! "${DELAY_SECONDS}" =~ ^[1-9][0-9]*$ \
    || ! "${STARTUP_TIMEOUT_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Guard launch identifiers and time limits are invalid" >&2
    exit 2
fi
if [[ "${RUNPOD_GUARD_KEEP_AWAKE}" != "auto" \
    && "${RUNPOD_GUARD_KEEP_AWAKE}" != "0" \
    && "${RUNPOD_GUARD_KEEP_AWAKE}" != "1" ]]; then
    echo "RUNPOD_GUARD_KEEP_AWAKE must be auto, 0, or 1" >&2
    exit 2
fi
if [[ "${RUNPOD_GUARD_EMERGENCY_TERMINATE_ON_STARTUP_FAILURE}" != "0" \
    && "${RUNPOD_GUARD_EMERGENCY_TERMINATE_ON_STARTUP_FAILURE}" != "1" ]]; then
    echo "RUNPOD_GUARD_EMERGENCY_TERMINATE_ON_STARTUP_FAILURE must be 0 or 1" >&2
    exit 2
fi
if [[ ! "${RUNPOD_GUARD_VOLUME_ROOT}" =~ ^/[A-Za-z0-9._/-]+$ \
    || ( "${RUNPOD_GUARD_VOLUME_ROOT}" != "/" \
        && "${RUNPOD_GUARD_VOLUME_ROOT}" == */ ) ]]; then
    echo "RUNPOD_GUARD_VOLUME_ROOT must be a canonical absolute path" >&2
    exit 2
fi
if [[ -n "${RUNPOD_GUARD_RUN_ID}" \
    && ( ! "${RUNPOD_GUARD_RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,119}$ \
        || "${RUNPOD_GUARD_RUN_ID}" == *--* ) ]]; then
    echo "RUNPOD_GUARD_RUN_ID must be a safe 1-120 character run ID" >&2
    exit 2
fi
case "${LIFECYCLE_KEY}" in
    lifecycle/stage1/cpu-preparation.json|\
        lifecycle/stage1/mixed-finalization.json|\
        lifecycle/stage1/training.json|lifecycle/stage1/validation.json|lifecycle/stage1/baseline.json) ;;
    *)
        echo "Unsupported lifecycle marker key" >&2
        exit 2
        ;;
esac
if [[ ( "${LIFECYCLE_KEY}" == "lifecycle/stage1/training.json" \
        || "${LIFECYCLE_KEY}" == "lifecycle/stage1/validation.json" \
        || "${LIFECYCLE_KEY}" == "lifecycle/stage1/baseline.json" ) \
    && -z "${RUNPOD_GUARD_RUN_ID}" ]]; then
    echo "GPU lifecycle guards require RUNPOD_GUARD_RUN_ID" >&2
    exit 2
fi
if [[ "${LOG_FILE}" != /* ]]; then
    echo "Guard log path must be absolute" >&2
    exit 2
fi
if [[ ! -r "${GUARD_SCRIPT}" || ! -r "${RUNPODCTL_WRAPPER}" ]]; then
    echo "RunPod guard scripts are unavailable" >&2
    exit 127
fi
runpod_assert_project_env_file "${LOCAL_PROJECT_ROOT}"
RUNPOD_ENV_FILE="$(runpod_project_env_file "${LOCAL_PROJECT_ROOT}")"
export RUNPOD_ENV_FILE

HOST_BOOT_ID=""
if [[ -r /proc/sys/kernel/random/boot_id ]]; then
    IFS= read -r linux_boot_id < /proc/sys/kernel/random/boot_id || true
    if [[ "${linux_boot_id:-}" =~ ^[A-Za-z0-9._-]+$ ]]; then
        HOST_BOOT_ID="linux-${linux_boot_id}"
    fi
elif [[ -x /usr/sbin/sysctl ]] || command -v sysctl >/dev/null 2>&1; then
    sysctl_command="$(command -v sysctl 2>/dev/null || true)"
    sysctl_command="${sysctl_command:-/usr/sbin/sysctl}"
    boot_info="$("${sysctl_command}" -n kern.boottime 2>/dev/null || true)"
    if [[ "${boot_info}" =~ sec[[:space:]]*=[[:space:]]*([0-9]+) ]]; then
        HOST_BOOT_ID="darwin-${BASH_REMATCH[1]}"
    fi
fi
if [[ "${RUNPOD_GUARD_EMERGENCY_TERMINATE_ON_STARTUP_FAILURE}" == "0" \
    && -z "${HOST_BOOT_ID}" ]]; then
    echo "Guard recovery cannot establish the current host boot identity" >&2
    exit 2
fi

mkdir -p "$(dirname "${LOG_FILE}")"
READY_FILE="${LOG_FILE%.log}.ready.json"
PID_FILE="${LOG_FILE%.log}.pid"
CAFFEINATE_PID_FILE="${LOG_FILE%.log}.caffeinate.pid"
KEEP_AWAKE_FILE="${LOG_FILE%.log}.keep-awake.json"
ready_tmp="${READY_FILE}.tmp.$$"
printf '{"state":"launching","pod_id":"%s","delay_seconds":%d,"host_boot_id":"%s"}\n' \
    "${POD_ID}" "${DELAY_SECONDS}" "${HOST_BOOT_ID}" > "${ready_tmp}"
mv "${ready_tmp}" "${READY_FILE}"

nohup env -i \
    "HOME=${HOME:-/tmp}" \
    "PATH=${PATH}" \
    "RUNPOD_ENV_FILE=${RUNPOD_ENV_FILE}" \
    "RUNPOD_GUARD_MAX_ATTEMPTS=${RUNPOD_GUARD_MAX_ATTEMPTS:-3}" \
    "RUNPOD_GUARD_RETRY_SECONDS=${RUNPOD_GUARD_RETRY_SECONDS:-30}" \
    "RUNPOD_GUARD_POLL_SECONDS=${RUNPOD_GUARD_POLL_SECONDS:-30}" \
    "RUNPOD_GUARD_S3_CONNECT_TIMEOUT=${RUNPOD_GUARD_S3_CONNECT_TIMEOUT:-5}" \
    "RUNPOD_GUARD_S3_READ_TIMEOUT=${RUNPOD_GUARD_S3_READ_TIMEOUT:-15}" \
    "RUNPOD_GUARD_S3_MAX_ATTEMPTS=${RUNPOD_GUARD_S3_MAX_ATTEMPTS:-1}" \
    "RUNPOD_GUARD_LIFECYCLE_KEY=${LIFECYCLE_KEY}" \
    "RUNPOD_GUARD_RUN_ID=${RUNPOD_GUARD_RUN_ID}" \
    "RUNPOD_GUARD_VOLUME_ROOT=${RUNPOD_GUARD_VOLUME_ROOT}" \
    "RUNPOD_GUARD_HOST_BOOT_ID=${HOST_BOOT_ID}" \
    "RUNPOD_GUARD_READY_FILE=${READY_FILE}" \
    "RUNPOD_GUARD_REQUIRE_LIFECYCLE=1" \
    "RUNPOD_TEST_MODE=${RUNPOD_TEST_MODE:-0}" \
    bash "${GUARD_SCRIPT}" "${POD_ID}" "${DELAY_SECONDS}" "${LOG_FILE}" \
    </dev/null >/dev/null 2>&1 &
GUARD_PID=$!
pid_tmp="${PID_FILE}.tmp.$$"
printf '%s\n' "${GUARD_PID}" > "${pid_tmp}"
mv "${pid_tmp}" "${PID_FILE}"

deadline=$((SECONDS + STARTUP_TIMEOUT_SECONDS))
while [[ ${SECONDS} -lt ${deadline} ]]; do
    if python3 -c \
        'import json, sys
with open(sys.argv[1], "r") as stream:
    payload = json.load(stream)
raise SystemExit(0 if payload.get("state") == "armed" and payload.get("pod_id") == sys.argv[2] and payload.get("pid") == int(sys.argv[3]) else 1)' \
        "${READY_FILE}" "${POD_ID}" "${GUARD_PID}" 2>/dev/null; then
        keep_awake_required=0
        if [[ "${RUNPOD_GUARD_KEEP_AWAKE}" == "1" \
            || ( "${RUNPOD_GUARD_KEEP_AWAKE}" == "auto" \
                && "$(uname -s)" == "Darwin" ) ]]; then
            keep_awake_required=1
        fi
        if [[ ${keep_awake_required} -eq 1 ]]; then
            if ! command -v caffeinate >/dev/null 2>&1; then
                echo "caffeinate is required to keep the local lifecycle guard awake" >&2
                break
            fi
            nohup caffeinate -is -w "${GUARD_PID}" </dev/null >/dev/null 2>&1 &
            CAFFEINATE_PID=$!
            sleep 0.2
            if ! kill -0 "${CAFFEINATE_PID}" 2>/dev/null; then
                echo "caffeinate exited before the lifecycle guard was protected" >&2
                break
            fi
            caffeinate_pid_tmp="${CAFFEINATE_PID_FILE}.tmp.$$"
            printf '%s\n' "${CAFFEINATE_PID}" > "${caffeinate_pid_tmp}"
            mv "${caffeinate_pid_tmp}" "${CAFFEINATE_PID_FILE}"
            keep_awake_tmp="${KEEP_AWAKE_FILE}.tmp.$$"
            printf '{"state":"armed","mechanism":"caffeinate","guard_pid":%d,"pid":%d}\n' \
                "${GUARD_PID}" "${CAFFEINATE_PID}" > "${keep_awake_tmp}"
            mv "${keep_awake_tmp}" "${KEEP_AWAKE_FILE}"
        else
            keep_awake_tmp="${KEEP_AWAKE_FILE}.tmp.$$"
            printf '{"state":"not-required","platform":"%s","guard_pid":%d}\n' \
                "$(uname -s)" "${GUARD_PID}" > "${keep_awake_tmp}"
            mv "${keep_awake_tmp}" "${KEEP_AWAKE_FILE}"
        fi
        printf '%s\n' "${GUARD_PID}"
        exit 0
    fi
    if ! kill -0 "${GUARD_PID}" 2>/dev/null; then
        break
    fi
    sleep 0.2
done

if kill -0 "${GUARD_PID}" 2>/dev/null; then
    kill "${GUARD_PID}" 2>/dev/null || true
    wait "${GUARD_PID}" 2>/dev/null || true
fi
if [[ "${RUNPOD_GUARD_EMERGENCY_TERMINATE_ON_STARTUP_FAILURE}" == "1" ]]; then
    printf '[%s] guard startup handshake failed; requesting emergency Pod termination\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${LOG_FILE}"
    if bash "${RUNPODCTL_WRAPPER}" pod delete "${POD_ID}" >> "${LOG_FILE}" 2>&1; then
        printf '[%s] emergency termination request succeeded\n' \
            "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${LOG_FILE}"
    else
        printf '[%s] emergency termination failed; manual intervention is required\n' \
            "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${LOG_FILE}"
    fi
    echo "External guard failed to arm; the newly created Pod was sent an emergency delete request" >&2
else
    printf '[%s] guard startup handshake failed; emergency termination disabled; Pod left running\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${LOG_FILE}"
    echo "External guard failed to arm; the existing Pod was left running" >&2
fi
exit 4
