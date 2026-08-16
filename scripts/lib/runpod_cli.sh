#!/usr/bin/env bash

# Parse user-facing RunPod workflow options into canonical runtime values.

runpod_duration_seconds() {
    local value="$1"
    local label="${2:-duration}"
    local amount unit

    if [[ ! "${value}" =~ ^[1-9][0-9]*[mhd]$ ]]; then
        printf '%s must use a positive duration such as 30m, 12h, or 2d\n' \
            "${label}" >&2
        return 2
    fi
    amount="${value%?}"
    unit="${value: -1}"
    case "${unit}" in
        m) printf '%d\n' "$((amount * 60))" ;;
        h) printf '%d\n' "$((amount * 3600))" ;;
        d) printf '%d\n' "$((amount * 86400))" ;;
    esac
}

runpod_validate_gpu_id() {
    local value="$1"
    if [[ -z "${value}" || "${value}" == *$'\n'* || "${value}" == *$'\r'* \
        || ! "${value}" =~ ^[A-Za-z0-9][A-Za-z0-9._+():/\ -]{0,127}$ ]]; then
        echo "--gpuId must be a non-empty RunPod GPU identifier" >&2
        return 2
    fi
}

runpod_validate_cpu_flavor() {
    local value="$1"
    case "${value}" in
        cpu3c|cpu3g|cpu3m|cpu5c|cpu5g|cpu5m) return 0 ;;
        *)
            echo "--cpuFlavor must be one of: cpu3c, cpu3g, cpu3m, cpu5c, cpu5g, cpu5m" >&2
            return 2
            ;;
    esac
}

runpod_validate_cpu_number() {
    local value="$1"
    if [[ ! "${value}" =~ ^[1-9][0-9]*$ || "${value}" -gt 32 ]]; then
        echo "--cpuNumber must be an integer from 1 through 32" >&2
        return 2
    fi
}
