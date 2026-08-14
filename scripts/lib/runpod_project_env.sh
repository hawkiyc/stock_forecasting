#!/usr/bin/env bash

# Read allowlisted values from a project-local dotenv file without executing it.

runpod_project_env_file() {
    local project_root="$1"
    printf '%s\n' "${RUNPOD_ENV_FILE:-${project_root}/.env}"
}

runpod_validate_project_env_file() {
    local project_root="$1"
    local env_file=""
    local file_mode=""

    env_file="$(runpod_project_env_file "${project_root}")"
    if [[ ! -f "${env_file}" ]]; then
        echo "RunPod project environment file not found: ${env_file}" >&2
        return 2
    fi

    if file_mode="$(stat -f '%Lp' "${env_file}" 2>/dev/null)"; then
        :
    elif file_mode="$(stat -c '%a' "${env_file}" 2>/dev/null)"; then
        :
    else
        echo "Unable to verify permissions for RunPod environment file: ${env_file}" >&2
        return 2
    fi

    case "${file_mode}" in
        400|600) ;;
        *)
            echo "RunPod environment file must have mode 600 or 400: ${env_file}" >&2
            return 2
            ;;
    esac
}

runpod_read_project_env_value() {
    local env_file="$1"
    local key="$2"
    local line=""
    local value=""
    local matches=0
    local first_character=""
    local last_character=""

    while IFS= read -r line || [[ -n "${line}" ]]; do
        line="${line%$'\r'}"
        case "${line}" in
            "${key}="*)
                value="${line#*=}"
                matches=$((matches + 1))
                ;;
            "export ${key}="*)
                value="${line#*=}"
                matches=$((matches + 1))
                ;;
        esac
    done < "${env_file}"

    if [[ ${matches} -eq 0 ]]; then
        return 1
    fi
    if [[ ${matches} -ne 1 ]]; then
        echo "Duplicate ${key} entries in ${env_file}" >&2
        return 2
    fi

    if [[ -n "${value}" ]]; then
        first_character="${value:0:1}"
        last_character="${value: -1}"
        case "${first_character}" in
            '"'|"'")
                if [[ "${last_character}" != "${first_character}" || ${#value} -lt 2 ]]; then
                    echo "Malformed quoted value for ${key} in ${env_file}" >&2
                    return 2
                fi
                value="${value:1:${#value}-2}"
                ;;
            *)
                if [[ "${value}" == [[:space:]]* || "${value}" == *[[:space:]] ]]; then
                    echo "Unquoted ${key} must not start or end with whitespace" >&2
                    return 2
                fi
                ;;
        esac
    fi

    printf '%s' "${value}"
}

runpod_load_project_env_key() {
    local project_root="$1"
    local key="$2"
    local required="$3"
    local env_file=""
    local value=""
    local status=0

    env_file="$(runpod_project_env_file "${project_root}")"
    if value="$(runpod_read_project_env_value "${env_file}" "${key}")"; then
        printf -v "${key}" '%s' "${value}"
        export "${key}"
        if [[ "${required}" == "required" && -z "${value}" ]]; then
            echo "${key} is required in ${env_file}" >&2
            return 2
        fi
        return 0
    else
        status=$?
    fi

    if [[ ${status} -eq 1 && "${required}" != "required" ]]; then
        return 0
    fi
    if [[ ${status} -eq 1 ]]; then
        echo "${key} is required in ${env_file}" >&2
        return 2
    fi
    return "${status}"
}

runpod_load_runpodctl_env() {
    local project_root="$1"

    set +x
    runpod_validate_project_env_file "${project_root}"
    runpod_load_project_env_key "${project_root}" RUNPOD_API_KEY required
    export RUNPOD_ENV_FILE="$(runpod_project_env_file "${project_root}")"
}

runpod_load_create_env() {
    local project_root="$1"
    local key=""

    set +x
    runpod_validate_project_env_file "${project_root}"
    for key in \
        RUNPOD_NETWORK_VOLUME_ID RUNPOD_VOLUME_ROOT NETWORK_VOLUME_ROOT PROJECT_ROOT \
        DATA_ROOT MODEL_ROOT CACHE_ROOT SAVED_MODEL_ROOT LOG_ROOT \
        WANDB_DIR HF_HOME TORCH_HOME RUNPOD_IMAGE RUNPOD_PYTHON_BIN \
        RUNPOD_API_BASE_URL RUNPOD_CPU_FLAVOR_ID RUNPOD_CPU_VCPU_COUNT \
        RUNPOD_EXPECTED_TORCH_VERSION RUNPOD_EXPECTED_CUDA_PREFIX \
        RUNPOD_EXPECTED_UBUNTU_VERSION RUNPOD_CONFIG MAX_RUNTIME_SECONDS \
        RUNPOD_HARD_LIMIT_SECONDS RUNPOD_TERMINATE_AFTER RUNPOD_DATACENTER_ID \
        RUNPOD_HF_SECRET_NAME RUNPOD_WANDB_SECRET_NAME WANDB_ENTITY WANDB_PROJECT \
        RUNPOD_CPU_POD_NAME RUNPOD_CPU_CONTAINER_DISK_GB \
        RUNPOD_CPU_MAX_RUNTIME_SECONDS RUNPOD_CPU_HARD_LIMIT_SECONDS \
        RUNPOD_EODHD_SECRET_NAME FIN_TS_DATASET_PROFILE \
        STAGE1_US_SYMBOLS STAGE1_US_ETF_SYMBOLS STAGE1_SYMBOL_LIMIT \
        STAGE1_DATA_START STAGE1_DATA_END STAGE1_MAX_API_CALLS \
        STAGE1_EODHD_QPS STAGE1_TAIWAN_QPS; do
        runpod_load_project_env_key "${project_root}" "${key}" optional
    done
    export RUNPOD_ENV_FILE="$(runpod_project_env_file "${project_root}")"
}

runpod_load_s3_env() {
    local project_root="$1"

    set +x
    runpod_validate_project_env_file "${project_root}"
    runpod_load_project_env_key "${project_root}" RUNPOD_NETWORK_VOLUME_ID required
    runpod_load_project_env_key "${project_root}" RUNPOD_S3_ACCESS_KEY_ID required
    runpod_load_project_env_key "${project_root}" RUNPOD_S3_SECRET_ACCESS_KEY required
    runpod_load_project_env_key "${project_root}" RUNPOD_S3_REGION required
    runpod_load_project_env_key "${project_root}" RUNPOD_S3_ENDPOINT optional
    runpod_load_project_env_key "${project_root}" RUNPOD_REMOTE_PROJECT_DIR optional
    runpod_load_project_env_key "${project_root}" RUNPOD_S3_CONNECT_TIMEOUT optional
    runpod_load_project_env_key "${project_root}" RUNPOD_S3_READ_TIMEOUT optional
    export RUNPOD_ENV_FILE="$(runpod_project_env_file "${project_root}")"
}

runpod_assert_project_env_file() {
    local project_root="$1"
    runpod_validate_project_env_file "${project_root}"
}

runpod_clear_exported_environment() {
    local variable_name=""

    while IFS= read -r variable_name; do
        if [[ "${variable_name}" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
            unset "${variable_name}" 2>/dev/null || true
        fi
    done < <(compgen -e)
}
