#!/usr/bin/env python3
"""Verify that a Python process is using the approved RunPod PyTorch runtime."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _public_version(version: str) -> str:
    return version.split("+", maxsplit=1)[0]


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _ubuntu_version() -> str:
    os_release = Path("/etc/os-release")
    if not os_release.exists():
        return ""
    for line in os_release.read_text(encoding="utf-8").splitlines():
        if line.startswith("VERSION_ID="):
            return line.partition("=")[2].strip().strip('"')
    return ""


def _collect_runtime(*, require_cuda_device: bool) -> dict[str, Any]:
    import torch

    cuda_available = torch.cuda.is_available()
    if require_cuda_device and not cuda_available:
        raise RuntimeError("CUDA is unavailable in the selected RunPod runtime.")

    gpu_name = ""
    compute_capability = ""
    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0)
        major, minor = torch.cuda.get_device_capability(0)
        compute_capability = f"{major}.{minor}"
        # Allocate a tensor to catch driver or architecture incompatibility before training.
        probe = torch.zeros(1, device="cuda")
        del probe
        torch.cuda.synchronize()

    return {
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": platform.python_version(),
        "python_prefix": str(Path(sys.prefix).resolve()),
        "python_base_prefix": str(Path(sys.base_prefix).resolve()),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "ubuntu_version": _ubuntu_version(),
        "torch_version": str(torch.__version__),
        "torch_public_version": _public_version(str(torch.__version__)),
        "torch_cuda_version": str(torch.version.cuda or ""),
        "torch_path": str(Path(torch.__file__).resolve()),
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
        "compute_capability": compute_capability,
        "runpod_image": os.environ.get("RUNPOD_IMAGE", ""),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-torch", default="2.9.1+cu128")
    parser.add_argument("--expected-cuda-prefix", default="12.8")
    parser.add_argument("--expected-ubuntu", default="24.04")
    parser.add_argument("--network-volume-root", type=Path, default=Path("/runpod-volume"))
    parser.add_argument("--role", choices=("image", "venv"), required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-cuda-device", action="store_true")
    return parser.parse_args()


def _validate(
    metadata: dict[str, Any],
    *,
    args: argparse.Namespace,
) -> list[str]:
    errors: list[str] = []
    python_pair = tuple(int(part) for part in metadata["python_version"].split(".")[:2])
    if python_pair != (3, 12):
        errors.append("The approved RunPod runtime requires Python 3.12.")
    if metadata["platform_system"] != "Linux" or metadata["platform_machine"] != "x86_64":
        errors.append("The approved runtime requires Linux x86_64.")
    if args.expected_ubuntu and metadata["ubuntu_version"] != args.expected_ubuntu:
        errors.append(
            f"Ubuntu {args.expected_ubuntu} is required; "
            f"found {metadata['ubuntu_version'] or 'unknown'}."
        )
    if metadata["torch_version"] != args.expected_torch:
        errors.append(
            f"PyTorch {args.expected_torch} is required; found {metadata['torch_version']}."
        )
    if not metadata["torch_cuda_version"].startswith(args.expected_cuda_prefix):
        errors.append(
            f"A CUDA {args.expected_cuda_prefix}.x PyTorch build is required; "
            f"found {metadata['torch_cuda_version'] or 'CPU-only'}."
        )

    network_root = args.network_volume_root.resolve()
    torch_path = Path(metadata["torch_path"]).resolve()
    if _inside(torch_path, network_root):
        errors.append(
            "PyTorch is shadowed by a network-volume package; it must come from the RunPod image."
        )

    in_venv = metadata["python_prefix"] != metadata["python_base_prefix"]
    if args.role == "image" and in_venv:
        errors.append(
            "The image check must use the RunPod image Python, not a virtual environment."
        )
    if args.role == "venv" and not in_venv:
        errors.append("The project check must use the persistent project virtual environment.")

    if args.reference is not None:
        reference = json.loads(args.reference.read_text(encoding="utf-8"))
        for key in (
            "python_version",
            "torch_version",
            "torch_cuda_version",
            "torch_path",
            "runpod_image",
        ):
            if metadata[key] != reference.get(key):
                errors.append(
                    f"Project environment {key} does not match the RunPod image: "
                    f"{metadata[key]!r} != {reference.get(key)!r}."
                )
    return errors


def main() -> None:
    args = _parse_args()
    try:
        # The image may intentionally omit NumPy because Poetry installs it in the
        # persistent project environment.  Torch emits an optional-import warning
        # during this image probe; the project venv dependency gate checks NumPy
        # explicitly after Poetry installation.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Failed to initialize NumPy")
            metadata = _collect_runtime(require_cuda_device=args.require_cuda_device)
        errors = _validate(metadata, args=args)
    except (ImportError, OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"RunPod runtime verification failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error

    if errors:
        for error in errors:
            print(f"RunPod runtime verification failed: {error}", file=sys.stderr)
        raise SystemExit(2)

    metadata["verified_at"] = os.environ.get(
        "RUNPOD_RUNTIME_VERIFIED_AT", datetime.now(UTC).isoformat()
    )
    serialized = json.dumps(metadata, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary_output = args.output.with_suffix(f"{args.output.suffix}.tmp")
        temporary_output.write_text(f"{serialized}\n", encoding="utf-8")
        temporary_output.replace(args.output)
    print(serialized)


if __name__ == "__main__":
    main()
