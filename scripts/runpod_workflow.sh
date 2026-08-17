#!/usr/bin/env bash

# Provide one script entry point for the complete RunPod lifecycle.
set -Eeuo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/runpod_cli.sh
source "${SCRIPT_DIR}/lib/runpod_cli.sh"

usage() {
    cat >&2 <<'EOF'
Usage:
  bash scripts/runpod_workflow.sh credentials
  bash scripts/runpod_workflow.sh volume deploy [OPTIONS]
  bash scripts/runpod_workflow.sh configure [SELECTION OPTIONS]
  bash scripts/runpod_workflow.sh selection show
  bash scripts/runpod_workflow.sh sync [--dry-run|--apply]
  bash scripts/runpod_workflow.sh cpu prepare [--interactive] [--maxRuntime DURATION] [--cpuNumber N] [--cpuFlavor FLAVOR]
  bash scripts/runpod_workflow.sh readiness [--code-only|--gpu]
  bash scripts/runpod_workflow.sh train [--maxRuntime DURATION] [--gpuId GPU_ID]
  bash scripts/runpod_workflow.sh resume [--maxRuntime DURATION] [--gpuId GPU_ID] [RUN_ID]
  bash scripts/runpod_workflow.sh validate [VALIDATION OPTIONS]
  bash scripts/runpod_workflow.sh status
  bash scripts/runpod_workflow.sh download [--resume] [--checkpointScope all|best] [RUN_ID]
  bash scripts/runpod_workflow.sh cpu-logs
EOF
}

if [[ $# -lt 1 ]]; then
    usage
    exit 2
fi

COMMAND="$1"
shift
case "${COMMAND}" in
    help|-h|--help)
        [[ $# -eq 0 ]] || { usage; exit 2; }
        usage
        exit 0
        ;;
    credentials)
        [[ $# -eq 0 ]] || { usage; exit 2; }
        exec bash "${SCRIPT_DIR}/configure_runpod_credentials.sh"
        ;;
    volume)
        [[ "${1:-}" == "deploy" ]] || { usage; exit 2; }
        shift
        exec bash "${SCRIPT_DIR}/deploy_runpod_network_volume.sh" "$@"
        ;;
    configure)
        exec bash "${SCRIPT_DIR}/configure_runpod_training.sh" "$@"
        ;;
    selection)
        [[ "${1:-}" == "show" && $# -eq 1 ]] || { usage; exit 2; }
        exec python3 "${SCRIPT_DIR}/runpod_selection.py" show \
            --project-root "${SCRIPT_DIR}/.."
        ;;
    sync)
        exec bash "${SCRIPT_DIR}/sync_project_to_runpod_volume.sh" "$@"
        ;;
    cpu)
        [[ "${1:-}" == "prepare" ]] || { usage; exit 2; }
        shift
        cpu_max_runtime=6h
        cpu_number=8
        cpu_flavor=cpu3g
        cpu_interactive=0
        if [[ $# -eq 0 ]]; then
            cpu_interactive=1
        fi
        while [[ $# -gt 0 ]]; do
            case "$1" in
                --interactive)
                    cpu_interactive=1
                    shift
                    ;;
                --maxRuntime|--max-runtime)
                    [[ $# -ge 2 ]] || { echo "$1 requires a value" >&2; exit 2; }
                    cpu_max_runtime="$2"
                    shift 2
                    ;;
                --cpuNumber|--cpu-number)
                    [[ $# -ge 2 ]] || { echo "$1 requires a value" >&2; exit 2; }
                    cpu_number="$2"
                    shift 2
                    ;;
                --cpuFlavor|--cpu-flavor)
                    [[ $# -ge 2 ]] || { echo "$1 requires a value" >&2; exit 2; }
                    cpu_flavor="$2"
                    shift 2
                    ;;
                *)
                    echo "Unknown CPU prepare option: $1" >&2
                    usage
                    exit 2
                    ;;
            esac
        done
        if [[ ${cpu_interactive} -eq 1 ]]; then
            cat >&2 <<'EOF'
CPU preparation resource configuration
Available flavors: cpu3c, cpu3g, cpu3m, cpu5c, cpu5g, cpu5m
Press Enter to accept the value shown in brackets.
EOF
            while true; do
                printf 'Maximum runtime [%s]: ' "${cpu_max_runtime}" >&2
                if ! IFS= read -r cpu_entered_max_runtime; then
                    echo "Input closed before CPU Pod creation" >&2
                    exit 2
                fi
                cpu_candidate_max_runtime="${cpu_entered_max_runtime:-${cpu_max_runtime}}"
                if cpu_max_seconds="$(runpod_duration_seconds \
                    "${cpu_candidate_max_runtime}" --maxRuntime)"; then
                    cpu_max_runtime="${cpu_candidate_max_runtime}"
                    break
                fi
            done
            while true; do
                printf 'vCPU count [%s]: ' "${cpu_number}" >&2
                if ! IFS= read -r cpu_entered_number; then
                    echo "Input closed before CPU Pod creation" >&2
                    exit 2
                fi
                cpu_candidate_number="${cpu_entered_number:-${cpu_number}}"
                if runpod_validate_cpu_number "${cpu_candidate_number}"; then
                    cpu_number="${cpu_candidate_number}"
                    break
                fi
            done
            while true; do
                printf 'CPU flavor [%s]: ' "${cpu_flavor}" >&2
                if ! IFS= read -r cpu_entered_flavor; then
                    echo "Input closed before CPU Pod creation" >&2
                    exit 2
                fi
                cpu_candidate_flavor="${cpu_entered_flavor:-${cpu_flavor}}"
                if runpod_validate_cpu_flavor "${cpu_candidate_flavor}"; then
                    cpu_flavor="${cpu_candidate_flavor}"
                    break
                fi
            done
            printf '\nCPU preparation Pod request:\n' >&2
            printf '  Maximum runtime: %s\n' "${cpu_max_runtime}" >&2
            printf '  External guard grace: 1h\n' >&2
            printf '  vCPU count: %s\n' "${cpu_number}" >&2
            printf '  CPU flavor: %s\n' "${cpu_flavor}" >&2
            while true; do
                printf 'Create this CPU preparation Pod? [y/N]: ' >&2
                if ! IFS= read -r cpu_confirmation; then
                    echo "Input closed; no CPU Pod was created" >&2
                    exit 2
                fi
                case "${cpu_confirmation}" in
                    [Yy]|[Yy][Ee][Ss]) break ;;
                    ""|[Nn]|[Nn][Oo])
                        echo "CPU preparation Pod creation cancelled; no Pod was created"
                        exit 0
                        ;;
                    *) echo "Please answer y or n" >&2 ;;
                esac
            done
        else
            runpod_validate_cpu_number "${cpu_number}"
            runpod_validate_cpu_flavor "${cpu_flavor}"
            cpu_max_seconds="$(runpod_duration_seconds \
                "${cpu_max_runtime}" --maxRuntime)"
        fi
        export RUNPOD_CPU_MAX_RUNTIME_SECONDS="${cpu_max_seconds}"
        export RUNPOD_CPU_HARD_LIMIT_SECONDS="$((cpu_max_seconds + 3600))"
        export RUNPOD_CPU_VCPU_COUNT="${cpu_number}"
        export RUNPOD_CPU_FLAVOR_ID="${cpu_flavor}"
        exec bash "${SCRIPT_DIR}/create_runpod_cpu_pod.sh"
        ;;
    readiness)
        exec bash "${SCRIPT_DIR}/verify_runpod_stage_readiness.sh" "$@"
        ;;
    train)
        train_max_runtime=12h
        train_gpu_id="NVIDIA GeForce RTX 5090"
        while [[ $# -gt 0 ]]; do
            case "$1" in
                --maxRuntime|--max-runtime)
                    [[ $# -ge 2 ]] || { echo "$1 requires a value" >&2; exit 2; }
                    train_max_runtime="$2"
                    shift 2
                    ;;
                --gpuId|--gpu-id)
                    [[ $# -ge 2 ]] || { echo "$1 requires a value" >&2; exit 2; }
                    train_gpu_id="$2"
                    shift 2
                    ;;
                *)
                    echo "Unknown training option: $1" >&2
                    usage
                    exit 2
                    ;;
            esac
        done
        runpod_validate_gpu_id "${train_gpu_id}"
        train_max_seconds="$(runpod_duration_seconds "${train_max_runtime}" --maxRuntime)"
        export RUNPOD_CLI_MAX_RUNTIME_SECONDS="${train_max_seconds}"
        export RUNPOD_CLI_HARD_LIMIT_SECONDS="$((train_max_seconds + 3600))"
        export RUNPOD_CLI_TERMINATE_AFTER="$(((train_max_seconds + 3659) / 60))m"
        export RUNPOD_CLI_GPU_ID="${train_gpu_id}"
        exec env RUNPOD_GPU_WORKFLOW=train bash "${SCRIPT_DIR}/create_runpod_pod.sh"
        ;;
    resume)
        exec bash "${SCRIPT_DIR}/create_runpod_resume_pod.sh" "$@"
        ;;
    validate)
        exec bash "${SCRIPT_DIR}/create_runpod_validation_pod.sh" "$@"
        ;;
    status)
        [[ $# -eq 0 ]] || { usage; exit 2; }
        exec bash "${SCRIPT_DIR}/show_runpod_status.sh"
        ;;
    download)
        exec bash "${SCRIPT_DIR}/download_runpod_results.sh" "$@"
        ;;
    cpu-logs)
        [[ $# -eq 0 ]] || { usage; exit 2; }
        exec bash "${SCRIPT_DIR}/download_runpod_cpu_logs.sh"
        ;;
    *)
        usage
        exit 2
        ;;
esac
