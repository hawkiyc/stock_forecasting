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
        RUNPOD_NETWORK_VOLUME_ID RUNPOD_DATACENTER_ID \
        TPEX_PROXY_URL RUNPOD_TPEX_PROXY_SECRET_NAME; do
        runpod_load_project_env_key "${project_root}" "${key}" optional
    done
    export RUNPOD_ENV_FILE="$(runpod_project_env_file "${project_root}")"
}

runpod_load_tpex_proxy_deploy_env() {
    local project_root="$1"
    local key=""

    set +x
    runpod_validate_project_env_file "${project_root}"
    for key in \
        CLOUDFLARE_API_TOKEN CLOUDFLARE_ACCOUNT_ID TPEX_PROXY_SHARED_SECRET \
        RUNPOD_API_KEY; do
        runpod_load_project_env_key "${project_root}" "${key}" required
    done
    runpod_load_project_env_key "${project_root}" CLOUDFLARE_WORKERS_SUBDOMAIN optional
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
