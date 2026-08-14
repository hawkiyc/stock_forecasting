#!/usr/bin/env bash

# Install tmux into the current Pod container and publish a persistent status file.
set -Eeuo pipefail
umask 077

if [[ $# -ne 1 ]]; then
    echo "Usage: install_runpod_tmux_worker.sh STATUS_FILE" >&2
    exit 2
fi
STATUS_FILE="$1"
STATUS_TMP="${STATUS_FILE}.tmp"

case "${STATUS_FILE}" in
    /workspace|/workspace/*)
        echo "tmux installation status must never use /workspace" >&2
        exit 2
        ;;
    /*) ;;
    *)
        echo "tmux installation status must be an absolute path" >&2
        exit 2
        ;;
esac

write_status() {
    local state="$1"
    local exit_code="$2"
    printf '{"state":"%s","exit_code":%d,"updated_at":"%s"}\n' \
        "${state}" "${exit_code}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${STATUS_TMP}"
    mv "${STATUS_TMP}" "${STATUS_FILE}"
}

install_failed() {
    local exit_code=$?
    trap - EXIT
    write_status failed "${exit_code}"
    exit "${exit_code}"
}
trap install_failed EXIT

mkdir -p "$(dirname "${STATUS_FILE}")"
write_status installing 0
if [[ "$(id -u)" -ne 0 ]]; then
    echo "tmux installation requires the root user in the RunPod container" >&2
    exit 3
fi
if ! command -v apt-get >/dev/null 2>&1; then
    echo "apt-get is unavailable in the selected RunPod image" >&2
    exit 127
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends tmux
tmux -V
write_status ready 0
trap - EXIT
