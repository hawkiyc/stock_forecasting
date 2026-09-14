#!/usr/bin/env python3
"""Publish and validate the independent diagnostic signal for the local Pod guard."""

# Keep the local control-plane helper compatible with pre-3.11 system Python.
# ruff: noqa: UP017

import argparse
import json
import os
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

MAX_MARKER_BYTES = 65536


def identity(volume_root, pod_id, owner_run_id, launch_id):
    if (
        not re.fullmatch(r"/[A-Za-z0-9._/-]+", volume_root)
        or volume_root == "/"
        or "//" in volume_root
        or volume_root.endswith("/")
        or any(part in {".", ".."} for part in volume_root.split("/"))
    ):
        raise ValueError("Invalid network volume root")
    for value, pattern, label in (
        (pod_id, r"[A-Za-z0-9_-]+", "Pod ID"),
        (owner_run_id, r"[A-Za-z0-9][A-Za-z0-9._-]{0,119}", "owner run ID"),
        (launch_id, r"launch-[A-Za-z0-9_-]{1,150}", "launch ID"),
    ):
        if not isinstance(value, str) or re.fullmatch(pattern, value) is None or "--" in value:
            raise ValueError("Invalid " + label)
    root = Path(volume_root)
    job_dir = root / "logs/tmux/fin-ts-probe-scales" / launch_id
    return {
        "marker": root / "lifecycle/diagnostics/representation-scales" / (pod_id + ".json"),
        "log_path": job_dir / "combined.log",
        "status_path": job_dir / "status.json",
    }


def load_bounded(stream):
    content = stream.read(MAX_MARKER_BYTES + 1)
    if len(content) > MAX_MARKER_BYTES:
        raise ValueError("Diagnostic marker exceeds the bounded read limit")
    return json.loads(content)


def validate(payload, volume_root, pod_id, owner_run_id):
    if not isinstance(payload, dict):
        raise ValueError("Diagnostic marker must be an object")
    paths = identity(volume_root, pod_id, owner_run_id, payload.get("launch_id"))
    if (
        type(payload.get("schema_version")) is not int
        or payload["schema_version"] != 1
        or payload.get("kind") != "representation-scale-probe"
        or payload.get("pod_id") != pod_id
        or payload.get("owner_run_id") != owner_run_id
        or payload.get("gpu_lease_acquired") is not True
        or payload.get("log_path") != str(paths["log_path"])
        or payload.get("status_path") != str(paths["status_path"])
    ):
        raise ValueError("Diagnostic marker identity or paths do not match this guard")
    state, code = payload.get("state"), payload.get("exit_code")
    if type(code) is not int or not 0 <= code <= 255:
        raise ValueError("Invalid diagnostic exit code")
    valid_state = (
        (state in {"running", "succeeded"} and code == 0)
        or (state == "failed" and code not in {0, 124})
        or (state == "timed_out" and code == 124)
    )
    if not valid_state:
        raise ValueError("Diagnostic state and exit code are inconsistent")
    generated_at = payload.get("generated_at")
    if not isinstance(generated_at, str):
        raise ValueError("Diagnostic timestamp is missing")
    if datetime.fromisoformat(generated_at.replace("Z", "+00:00")).tzinfo is None:
        raise ValueError("Diagnostic timestamp must include its timezone")
    return state


def publish(arguments):
    paths = identity(
        arguments.network_volume_root, arguments.pod_id, arguments.owner_run_id, arguments.launch_id
    )
    if os.environ.get("RUNPOD_GPU_WORKFLOW_LEASE_HELD") != "1":
        raise ValueError("Diagnostic publication requires the GPU lease")
    os.fstat(9)
    payload = {
        "schema_version": 1,
        "kind": "representation-scale-probe",
        "pod_id": arguments.pod_id,
        "owner_run_id": arguments.owner_run_id,
        "launch_id": arguments.launch_id,
        "gpu_lease_acquired": True,
        "state": arguments.state,
        "exit_code": arguments.exit_code,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "log_path": str(paths["log_path"]),
        "status_path": str(paths["status_path"]),
    }
    validate(payload, arguments.network_volume_root, arguments.pod_id, arguments.owner_run_id)
    marker = paths["marker"]
    if marker.resolve() != marker or paths["status_path"].resolve() != paths["status_path"]:
        raise ValueError("Diagnostic lifecycle paths must not traverse symlinks")
    if arguments.state != "running":
        with paths["status_path"].open("rb") as stream:
            status = load_bounded(stream)
        if (
            status.get("state") != arguments.state
            or status.get("exit_code") != arguments.exit_code
            or status.get("log_path") != str(paths["log_path"])
        ):
            raise ValueError("Terminal diagnostic signal requires matching persisted tmux status")
    marker.parent.mkdir(parents=True, exist_ok=True)
    temporary = marker.with_name(marker.name + ".tmp." + uuid.uuid4().hex)
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, sort_keys=True)
        stream.write("\n")
    temporary.replace(marker)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("write-state", "guard-state"))
    parser.add_argument("--network-volume-root", required=True)
    parser.add_argument("--pod-id", required=True)
    parser.add_argument("--owner-run-id", required=True)
    parser.add_argument("--launch-id")
    parser.add_argument("--state")
    parser.add_argument("--exit-code", type=int)
    arguments = parser.parse_args()
    if arguments.command == "write-state":
        publish(arguments)
    else:
        print(
            validate(
                load_bounded(sys.stdin.buffer), arguments.network_volume_root,
                arguments.pod_id, arguments.owner_run_id,
            )
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, TypeError, ValueError, AttributeError) as error:
        print("Invalid diagnostic lifecycle: " + str(error), file=sys.stderr)
        raise SystemExit(2) from error
