#!/usr/bin/env python3
"""Check complete baseline artifacts on the control host before creating a paid Pod."""

from __future__ import annotations

import argparse
import json
import os
import runpy
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "require", "identity"))
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    project = args.project_root.resolve()
    selection_tools = runpy.run_path(str(project / "scripts/runpod_selection.py"))
    _, selection = selection_tools["_resolve_selection_path"](
        project, os.environ.get("RUNPOD_SELECTION_FILE")
    )
    contract_tools = runpy.run_path(str(project / "src/stock_forecasting/baseline_contract.py"))
    identity = contract_tools["baseline_contract"](project, selection)
    contract_tools["validate_local_configuration"](
        project, selection, identity["contract"]["parameters"]
    )
    if args.command == "identity":
        print(json.dumps(identity))
        return 0
    if os.environ.get("RUNPOD_POD_ID") and os.environ.get("RUNPOD_TEST_MODE") != "1":
        raise ValueError("The baseline cache gate must run on the local control host")
    bucket = os.environ.get("RUNPOD_NETWORK_VOLUME_ID", "")
    import re

    if not re.fullmatch(r"[A-Za-z0-9_-]+", bucket):
        raise ValueError("Network volume ID must be loaded from the project .env")
    wrapper = str(project / "scripts/runpod_s3_project.sh")
    prefix = "baselines/" + identity["baseline_id"] + "/"
    command = [
        "bash",
        wrapper,
        "s3",
        "cp",
        f"s3://{bucket}/{prefix}complete.json",
        "-",
        "--only-show-errors",
    ]
    response = subprocess.run(command, capture_output=True, text=True, timeout=180)
    if response.returncode:
        if not any(code in response.stderr for code in ("NoSuchKey", "404", "Not Found")):
            raise RuntimeError(
                "Unable to verify the baseline cache; no Pod will be created. "
                "Check S3 credentials/connectivity."
            )
        if args.command == "require":
            raise ValueError(
                "Matching full-data baselines are missing. "
                "Run: bash scripts/runpod_workflow.sh baseline"
            )
        print(json.dumps({"complete": False, **identity}))
        return 0
    payload = json.loads(response.stdout)
    contract_tools["validate_complete"](payload, identity)
    # Verify the active immutable manifest locally as well, before any paid Pod.
    manifest_key = (
        f"datasets/{selection['dataset_request_sha256']}/prepared/bar-store/bar-store.json"
    )
    manifest_response = subprocess.run(
        ["bash", wrapper, "s3", "cp", f"s3://{bucket}/{manifest_key}", "-", "--only-show-errors"],
        capture_output=True,
        timeout=180,
    )
    import hashlib

    if manifest_response.returncode or hashlib.sha256(
        manifest_response.stdout
    ).hexdigest() != payload.get("data_identity", {}).get("manifest_sha256"):
        raise ValueError(
            "The active prepared-data manifest differs from the saved baseline; no Pod was created"
        )
    if json.loads(manifest_response.stdout)["split_counts"] != payload["sample_counts"]:
        raise ValueError("Saved baseline counts do not cover the complete prepared splits")

    def check_artifact(item):
        relative, metadata = item
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Unsafe baseline artifact path")
        result = subprocess.run(
            [
                "bash",
                wrapper,
                "s3api",
                "head-object",
                "--bucket",
                bucket,
                "--key",
                prefix + relative,
                "--query",
                "ContentLength",
                "--output",
                "text",
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )
        if result.returncode or result.stdout.strip() != str(metadata["bytes"]):
            raise ValueError(f"Baseline artifact is missing or truncated: {relative}")

    # The manifest contains a bounded model suite, not one future per data sample.
    workers = max(
        1,
        min(int(os.environ.get("RUNPOD_BASELINE_PREFLIGHT_WORKERS", "4")), os.cpu_count() or 1, 8),
    )
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(check_artifact, payload["artifacts"].items()))
    print(json.dumps({"complete": True, **identity}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError, subprocess.TimeoutExpired) as error:
        print(f"Baseline preflight failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
