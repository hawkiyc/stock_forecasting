#!/usr/bin/env bash

# Validate untrusted RunPod volume paths lexically before any filesystem access.

runpod_validate_absolute_path() {
    local path_value="${1:-}"
    local path_label="${2:-PATH}"

    if [[ -z "${path_value}" || "${path_value}" != /* ]]; then
        echo "${path_label} must be an absolute path" >&2
        return 2
    fi
    if [[ ! "${path_value}" =~ ^/[A-Za-z0-9._/-]*$ ]]; then
        echo "${path_label} contains unsupported characters" >&2
        return 2
    fi
    if [[ "${path_value}" == *"//"* ]]; then
        echo "${path_label} must not contain //" >&2
        return 2
    fi
    if [[ "${path_value}/" == *"/./"* ]]; then
        echo "${path_label} must not contain a dot path component" >&2
        return 2
    fi
    if [[ "${path_value}/" == *"/../"* ]]; then
        echo "${path_label} must not contain a parent path component" >&2
        return 2
    fi
    if [[ "${path_value}" != "/" && "${path_value}" == */ ]]; then
        echo "${path_label} must not end with a slash" >&2
        return 2
    fi
}

runpod_validate_path_in_root() {
    local path_value="${1:-}"
    local root_value="${2:-}"
    local path_label="${3:-PATH}"
    local root_label="${4:-NETWORK_VOLUME_ROOT}"

    runpod_validate_absolute_path "${root_value}" "${root_label}" || return
    runpod_validate_absolute_path "${path_value}" "${path_label}" || return

    if [[ "${root_value}" == "/" ]]; then
        return 0
    fi
    case "${path_value}" in
        "${root_value}"|"${root_value}"/*) return 0 ;;
        *)
            echo "${path_label} must be inside ${root_label} (${root_value})" >&2
            return 2
            ;;
    esac
}

runpod_acquire_gpu_workflow_lease() {
    local volume_root="${1:-}"
    local lease_root lease_path

    if [[ "${RUNPOD_GPU_WORKFLOW_LEASE_HELD:-0}" == "1" ]]; then
        return 0
    fi
    runpod_validate_absolute_path "${volume_root}" NETWORK_VOLUME_ROOT || return
    lease_root="${volume_root}/lifecycle/stage1"
    lease_path="${lease_root}/gpu-workflow.lock"
    runpod_validate_path_in_root \
        "${lease_path}" "${volume_root}" GPU_WORKFLOW_LEASE NETWORK_VOLUME_ROOT || return
    if ! command -v flock >/dev/null 2>&1; then
        if [[ "${RUNPOD_TEST_MODE:-0}" == "1" ]]; then
            export RUNPOD_GPU_WORKFLOW_LEASE_HELD=1
            return 0
        fi
        echo "flock is required for the shared GPU workflow lease" >&2
        return 127
    fi
    mkdir -p "${lease_root}"
    exec 9>"${lease_path}"
    if ! flock -n 9; then
        echo "Another training or validation workflow already holds the GPU workflow lease" >&2
        return 75
    fi
    export RUNPOD_GPU_WORKFLOW_LEASE_HELD=1
}
