#!/usr/bin/env python3
"""Terminate the current network-volume Pod with its PID 1 scoped API key."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

SAFE_ID = re.compile(r"[A-Za-z0-9_-]+")
SAFE_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,119}")


def _pid1_environment(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for entry in path.read_bytes().split(b"\0"):
        if not entry or b"=" not in entry:
            continue
        raw_name, raw_value = entry.split(b"=", maxsplit=1)
        name = raw_name.decode("utf-8", errors="strict")
        if name in {"RUNPOD_API_KEY", "RUNPOD_POD_ID"}:
            values[name] = raw_value.decode("utf-8", errors="strict")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stop-script", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=Path("/proc/1/environ"))
    arguments = parser.parse_args()
    if arguments.source != Path("/proc/1/environ") and os.environ.get("RUNPOD_TEST_MODE") != "1":
        parser.error("A custom PID 1 environment is allowed only in test mode")
    stop_script = arguments.stop_script.resolve()
    if not stop_script.is_file() or stop_script.is_symlink():
        raise ValueError(f"Pod shutdown script is unavailable: {stop_script}")

    pid1 = _pid1_environment(arguments.source)
    pod_id = pid1.get("RUNPOD_POD_ID", "")
    api_key = pid1.get("RUNPOD_API_KEY", "")
    if SAFE_ID.fullmatch(pod_id) is None:
        raise ValueError("PID 1 did not provide a valid RUNPOD_POD_ID")
    if not api_key or "\n" in api_key or "\r" in api_key:
        raise ValueError("PID 1 did not provide a usable pod-scoped RUNPOD_API_KEY")

    network_root = os.environ.get("NETWORK_VOLUME_ROOT", "/runpod-volume")
    run_key = os.environ.get("RUNPOD_RUN_KEY", "")
    if run_key and (SAFE_RUN_ID.fullmatch(run_key) is None or "--" in run_key):
        raise ValueError("RUNPOD_RUN_KEY is invalid")
    shutdown_dir = os.environ.get(
        "RUNPOD_SHUTDOWN_DIR",
        f"{network_root}/logs/{run_key or 'bootstrap'}/pod-shutdown",
    )
    shutdown_marker = os.environ.get(
        "RUNPOD_SHUTDOWN_MARKER",
        f"{shutdown_dir}/shutdown.json",
    )
    environment = {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "NETWORK_VOLUME_ROOT": network_root,
        "LOG_ROOT": os.environ.get("LOG_ROOT", f"{network_root}/logs"),
        "RUNPOD_API_KEY": api_key,
        "RUNPOD_POD_ID": pod_id,
        "RUNPOD_SHUTDOWN_ACTION": "terminate",
        "RUNPOD_SHUTDOWN_DIR": shutdown_dir,
        "RUNPOD_SHUTDOWN_MARKER": shutdown_marker,
        "RUNPOD_TEST_MODE": os.environ.get("RUNPOD_TEST_MODE", "0"),
    }
    if run_key:
        environment["RUNPOD_RUN_KEY"] = run_key
    completed = subprocess.run(
        ["bash", str(stop_script)],
        check=False,
        env=environment,
    )
    return completed.returncode


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, UnicodeError, ValueError) as error:
        print(f"Unable to terminate the current RunPod Pod: {error}", file=sys.stderr)
        raise SystemExit(2) from error
