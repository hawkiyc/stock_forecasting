#!/usr/bin/env bash

# Reconcile only a confirmed orphaned GPU lifecycle before creating another paid Pod.
set -Eeuo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
S3_WRAPPER="${RUNPOD_S3_WRAPPER:-${SCRIPT_DIR}/runpod_s3_project.sh}"
RUNPODCTL_WRAPPER="${RUNPODCTL_WRAPPER:-${SCRIPT_DIR}/runpodctl_project.sh}"
READINESS_HELPER="${RUNPOD_READINESS_HELPER:-${SCRIPT_DIR}/runpod_readiness.py}"
RUNPOD_VOLUME_MOUNT_PATH="${RUNPOD_VOLUME_MOUNT_PATH:-/runpod-volume}"
ORPHAN_CONFIRMATIONS="${RUNPOD_ORPHAN_CONFIRMATIONS:-2}"
ORPHAN_CONFIRMATION_DELAY_SECONDS="${RUNPOD_ORPHAN_CONFIRMATION_DELAY_SECONDS:-2}"
ORPHAN_MINIMUM_AGE_SECONDS="${RUNPOD_ORPHAN_MINIMUM_AGE_SECONDS:-60}"

# shellcheck source=lib/runpod_project_env.sh
source "${SCRIPT_DIR}/lib/runpod_project_env.sh"

if [[ $# -ne 2 ]]; then
    echo "Usage: ensure_runpod_gpu_workflow_available.sh LIFECYCLE_KEY LIFECYCLE_KIND" >&2
    exit 2
fi
LIFECYCLE_KEY="$1"
LIFECYCLE_KIND="$2"

case "${LIFECYCLE_KEY}:${LIFECYCLE_KIND}" in
    lifecycle/stage1/training.json:stage1-training|\
        lifecycle/stage1/validation.json:stage1-validation) ;;
    *)
        echo "GPU lifecycle key and kind do not form an approved pair" >&2
        exit 2
        ;;
esac
if [[ -z "${RUNPOD_NETWORK_VOLUME_ID:-}" ]]; then
    runpod_load_create_env "${LOCAL_PROJECT_ROOT}"
fi
if [[ ! "${RUNPOD_NETWORK_VOLUME_ID:-}" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo "RUNPOD_NETWORK_VOLUME_ID is required for GPU lifecycle reconciliation" >&2
    exit 2
fi
if [[ ! "${RUNPOD_VOLUME_MOUNT_PATH}" =~ ^/[A-Za-z0-9._/-]+$ ]]; then
    echo "RUNPOD_VOLUME_MOUNT_PATH is invalid" >&2
    exit 2
fi
if [[ ! "${ORPHAN_CONFIRMATIONS}" =~ ^[1-9][0-9]*$ \
    || "${ORPHAN_CONFIRMATIONS}" -lt 2 \
    || ! "${ORPHAN_CONFIRMATION_DELAY_SECONDS}" =~ ^[0-9]+$ \
    || ! "${ORPHAN_MINIMUM_AGE_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "GPU lifecycle orphan confirmation settings are invalid" >&2
    exit 2
fi
if [[ ! -r "${S3_WRAPPER}" || ! -r "${RUNPODCTL_WRAPPER}" \
    || ! -r "${READINESS_HELPER}" ]]; then
    echo "GPU lifecycle reconciliation helpers are unavailable" >&2
    exit 127
fi
if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 is required for GPU lifecycle reconciliation" >&2
    exit 127
fi

load_lifecycle() {
    bash "${S3_WRAPPER}" s3 cp \
        "s3://${RUNPOD_NETWORK_VOLUME_ID}/${LIFECYCLE_KEY}" - \
        --only-show-errors
}

probe_pod() {
    local pod_id="$1"
    local probe_json=""
    local probe_exit_code=0
    local probe_state=""

    if probe_json="$(bash "${RUNPODCTL_WRAPPER}" pod get "${pod_id}" 2>&1)"; then
        probe_exit_code=0
    else
        probe_exit_code=$?
    fi
    if ! probe_state="$(printf '%s\n' "${probe_json}" \
        | python3 "${READINESS_HELPER}" runpod-pod-probe-state \
            --response - \
            --expected-pod-id "${pod_id}" \
            --command-exit-code "${probe_exit_code}")"; then
        printf 'Unable to prove whether RunPod Pod %s still exists; refusing to create a competing paid Pod\n' \
            "${pod_id}" >&2
        return 2
    fi
    printf '%s\n' "${probe_state}"
}

lifecycle_json="$(load_lifecycle)"
for marker_attempt in 1 2 3; do
    lifecycle_status="$(printf '%s\n' "${lifecycle_json}" \
        | python3 "${READINESS_HELPER}" gpu-workflow-status \
            --marker - \
            --network-volume-root "${RUNPOD_VOLUME_MOUNT_PATH}" \
            --kind "${LIFECYCLE_KIND}")"
    IFS=$'\t' read -r lifecycle_state lifecycle_pod_id <<< "${lifecycle_status}"
    case "${lifecycle_state}" in
        ready|failed|timed_out)
            exit 0
            ;;
        preparing|finalizing) ;;
        *)
            echo "GPU lifecycle state is unsupported after validation" >&2
            exit 2
            ;;
    esac

    marker_changed=0
    for ((confirmation = 1; confirmation <= ORPHAN_CONFIRMATIONS; confirmation++)); do
        pod_state="$(probe_pod "${lifecycle_pod_id}")" || exit $?
        if [[ "${pod_state}" == "present" ]]; then
            printf 'RunPod readiness check failed: Another GPU workflow is active; Pod %s still exists\n' \
                "${lifecycle_pod_id}" >&2
            exit 2
        fi
        if [[ "${pod_state}" != "absent" ]]; then
            echo "RunPod Pod probe returned an unsupported state" >&2
            exit 2
        fi

        latest_lifecycle_json="$(load_lifecycle)"
        if [[ "${latest_lifecycle_json}" != "${lifecycle_json}" ]]; then
            lifecycle_json="${latest_lifecycle_json}"
            marker_changed=1
            break
        fi
        if [[ ${confirmation} -lt ${ORPHAN_CONFIRMATIONS} \
            && ${ORPHAN_CONFIRMATION_DELAY_SECONDS} -gt 0 ]]; then
            sleep "${ORPHAN_CONFIRMATION_DELAY_SECONDS}"
        fi
    done
    if [[ ${marker_changed} -eq 1 ]]; then
        continue
    fi

    latest_lifecycle_json="$(load_lifecycle)"
    if [[ "${latest_lifecycle_json}" != "${lifecycle_json}" ]]; then
        lifecycle_json="${latest_lifecycle_json}"
        continue
    fi
    reconciled_json="$(printf '%s\n' "${lifecycle_json}" \
        | python3 "${READINESS_HELPER}" reconcile-orphaned-gpu-workflow \
            --marker - \
            --network-volume-root "${RUNPOD_VOLUME_MOUNT_PATH}" \
            --kind "${LIFECYCLE_KIND}" \
            --expected-pod-id "${lifecycle_pod_id}" \
            --minimum-age-seconds "${ORPHAN_MINIMUM_AGE_SECONDS}")"
    printf '%s\n' "${reconciled_json}" \
        | bash "${S3_WRAPPER}" s3 cp - \
            "s3://${RUNPOD_NETWORK_VOLUME_ID}/${LIFECYCLE_KEY}" \
            --only-show-errors
    published_json="$(load_lifecycle)"
    if [[ "${published_json}" != "${reconciled_json}" ]]; then
        printf '%s\n' \
            "GPU lifecycle changed while publishing orphan reconciliation; refusing Pod creation" \
            >&2
        exit 2
    fi
    printf '%s\n' "${published_json}" \
        | python3 "${READINESS_HELPER}" gpu-workflow-available \
            --marker - \
            --network-volume-root "${RUNPOD_VOLUME_MOUNT_PATH}" \
            --kind "${LIFECYCLE_KIND}" >/dev/null
    printf 'Reconciled orphaned GPU lifecycle: key=%s pod_id=%s state=failed reason=%s\n' \
        "${LIFECYCLE_KEY}" "${lifecycle_pod_id}" runpod_pod_not_found >&2
    exit 0
done

echo "GPU lifecycle changed repeatedly during reconciliation; refusing Pod creation" >&2
exit 2
