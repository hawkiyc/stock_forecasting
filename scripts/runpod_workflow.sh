#!/usr/bin/env bash

# Provide one script entry point for the complete RunPod lifecycle.
set -Eeuo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    cat >&2 <<'EOF'
Usage:
  bash scripts/runpod_workflow.sh credentials
  bash scripts/runpod_workflow.sh volume deploy [OPTIONS]
  bash scripts/runpod_workflow.sh configure [SELECTION OPTIONS]
  bash scripts/runpod_workflow.sh selection show
  bash scripts/runpod_workflow.sh sync [--dry-run|--apply]
  bash scripts/runpod_workflow.sh cpu prepare
  bash scripts/runpod_workflow.sh readiness [--code-only|--gpu]
  bash scripts/runpod_workflow.sh train
  bash scripts/runpod_workflow.sh validate [VALIDATION OPTIONS]
  bash scripts/runpod_workflow.sh status
  bash scripts/runpod_workflow.sh download [--resume] [RUN_ID]
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
        [[ "${1:-}" == "prepare" && $# -eq 1 ]] || { usage; exit 2; }
        exec bash "${SCRIPT_DIR}/create_runpod_cpu_pod.sh"
        ;;
    readiness)
        exec bash "${SCRIPT_DIR}/verify_runpod_stage_readiness.sh" "$@"
        ;;
    train)
        [[ $# -eq 0 ]] || { usage; exit 2; }
        exec env RUNPOD_GPU_WORKFLOW=train bash "${SCRIPT_DIR}/create_runpod_pod.sh"
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
