#!/usr/bin/env bash

# Provide bounded retries around idempotent RunPod S3 synchronization operations.

_runpod_s3_retry() {
    local mode="$1"
    local wrapper="$2"
    local description="$3"
    local payload="$4"
    shift 4

    local max_attempts="${RUNPOD_S3_RETRY_MAX_ATTEMPTS:-8}"
    local delay_seconds="${RUNPOD_S3_RETRY_INITIAL_BACKOFF_SECONDS:-1}"
    local max_delay_seconds="${RUNPOD_S3_RETRY_MAX_BACKOFF_SECONDS:-15}"
    local attempt=1
    local operation_status=0
    local captured_output=""
    local next_delay=0

    if [[ ! "${max_attempts}" =~ ^[1-9][0-9]*$ \
        || ! "${delay_seconds}" =~ ^[0-9]+$ \
        || ! "${max_delay_seconds}" =~ ^[0-9]+$ ]]; then
        echo "RunPod S3 retry settings must be non-negative integers with positive attempts" >&2
        return 2
    fi
    if [[ ! -f "${wrapper}" || ! -r "${wrapper}" ]]; then
        echo "RunPod S3 retry wrapper is unavailable: ${wrapper}" >&2
        return 127
    fi
    if [[ -z "${description}" || "${description}" == *$'\n'* ]]; then
        echo "RunPod S3 retry description must be a non-empty single line" >&2
        return 2
    fi
    case "${mode}" in
        command|capture|stdin) ;;
        *)
            echo "Unsupported RunPod S3 retry mode: ${mode}" >&2
            return 2
            ;;
    esac

    while [[ ${attempt} -le ${max_attempts} ]]; do
        operation_status=0
        case "${mode}" in
            command)
                if bash "${wrapper}" "$@"; then
                    return 0
                else
                    operation_status=$?
                fi
                ;;
            capture)
                captured_output=""
                if captured_output="$(bash "${wrapper}" "$@")"; then
                    printf '%s' "${captured_output}"
                    return 0
                else
                    operation_status=$?
                fi
                ;;
            stdin)
                if printf '%s\n' "${payload}" | bash "${wrapper}" "$@"; then
                    return 0
                else
                    operation_status=$?
                fi
                ;;
        esac

        if [[ ${attempt} -ge ${max_attempts} ]]; then
            printf 'RunPod S3 operation failed after %d attempts: %s\n' \
                "${max_attempts}" "${description}" >&2
            return "${operation_status}"
        fi

        printf 'RunPod S3 operation failed with exit code %d; retrying in %ss (%d/%d): %s\n' \
            "${operation_status}" "${delay_seconds}" "${attempt}" \
            "${max_attempts}" "${description}" >&2
        if [[ ${delay_seconds} -gt 0 ]]; then
            sleep "${delay_seconds}"
        fi
        next_delay=$((delay_seconds * 2))
        if [[ ${next_delay} -gt ${max_delay_seconds} ]]; then
            next_delay="${max_delay_seconds}"
        fi
        delay_seconds="${next_delay}"
        attempt=$((attempt + 1))
    done
}

runpod_s3_retry_command() {
    local wrapper="$1"
    local description="$2"
    shift 2
    _runpod_s3_retry command "${wrapper}" "${description}" "" "$@"
}

runpod_s3_retry_capture() {
    local wrapper="$1"
    local description="$2"
    shift 2
    _runpod_s3_retry capture "${wrapper}" "${description}" "" "$@"
}

runpod_s3_retry_stdin() {
    local wrapper="$1"
    local description="$2"
    local payload="$3"
    shift 3
    _runpod_s3_retry stdin "${wrapper}" "${description}" "${payload}" "$@"
}
