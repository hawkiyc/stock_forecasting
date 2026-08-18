#!/usr/bin/env bash

# Load only validated, allowlisted values from an immutable selection contract.

runpod_load_active_selection() {
    local project_root="$1"
    local helper="${project_root}/scripts/runpod_selection.py"
    local key=""
    local value=""

    if [[ ! -f "${helper}" || -L "${helper}" ]]; then
        echo "RunPod selection helper is unavailable: ${helper}" >&2
        return 2
    fi
    if ! command -v python3 >/dev/null 2>&1; then
        echo "python3 is required to load the RunPod selection" >&2
        return 127
    fi

    unset RUNPOD_SELECTION_ID RUNPOD_SELECTION_SHA256 \
        RUNPOD_DATASET_REQUEST_SHA256 RUNPOD_SELECTION_FILE \
        RUNPOD_REMOTE_SELECTION_RELATIVE_PATH RUNPOD_REMOTE_SELECTION_PATH \
        RUNPOD_STAGE RUNPOD_CONFIG RUNPOD_STAGE_CONFIG_SHA256 DATA_ROOT \
        FIN_TS_DATASET_PROFILE STAGE1_US_SYMBOLS STAGE1_US_ETF_SYMBOLS \
        STAGE1_SYMBOL_LIMIT STAGE1_DATA_START STAGE1_DATA_END \
        STAGE1_MAX_API_CALLS STAGE1_EODHD_QPS STAGE1_TAIWAN_QPS \
        MAX_RUNTIME_SECONDS RUNPOD_HARD_LIMIT_SECONDS RUNPOD_TERMINATE_AFTER

    while IFS= read -r -d '' key && IFS= read -r -d '' value; do
        case "${key}" in
            RUNPOD_SELECTION_ID|RUNPOD_SELECTION_SHA256|RUNPOD_DATASET_REQUEST_SHA256|\
            RUNPOD_SELECTION_FILE|RUNPOD_REMOTE_SELECTION_RELATIVE_PATH|\
            RUNPOD_REMOTE_SELECTION_PATH|RUNPOD_STAGE|RUNPOD_CONFIG|\
            RUNPOD_STAGE_CONFIG_SHA256|DATA_ROOT|FIN_TS_DATASET_PROFILE|\
            STAGE1_US_SYMBOLS|STAGE1_US_ETF_SYMBOLS|STAGE1_SYMBOL_LIMIT|\
            STAGE1_DATA_START|STAGE1_DATA_END|MAX_RUNTIME_SECONDS|\
            RUNPOD_HARD_LIMIT_SECONDS|RUNPOD_TERMINATE_AFTER)
                printf -v "${key}" '%s' "${value}"
                export "${key}"
                ;;
            *)
                echo "Selection helper emitted an unsupported variable: ${key}" >&2
                return 2
                ;;
        esac
    done < <(python3 "${helper}" export --project-root "${project_root}" --null)

    if [[ -z "${RUNPOD_SELECTION_ID:-}" \
        || -z "${RUNPOD_SELECTION_SHA256:-}" \
        || -z "${RUNPOD_DATASET_REQUEST_SHA256:-}" ]]; then
        echo "No valid active RunPod selection was loaded" >&2
        return 2
    fi
}
