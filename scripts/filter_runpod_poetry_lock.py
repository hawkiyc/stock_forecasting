#!/usr/bin/env python3
"""Create a Poetry install view that leaves RunPod image packages untouched."""

from __future__ import annotations

import argparse
import re
import tomllib
from pathlib import Path

_IMAGE_RUNTIME_NAMES = {"torch", "torchaudio", "torchvision", "triton"}
_IMAGE_RUNTIME_PREFIXES = ("cuda-", "nvidia-")
_PACKAGE_START = "[[package]]"
_METADATA_START = "[metadata]"
_PACKAGE_NAME = re.compile(r'^name\s*=\s*"([^"]+)"\s*$')


def _is_image_runtime_package(name: str) -> bool:
    normalized = name.lower().replace("_", "-")
    return normalized in _IMAGE_RUNTIME_NAMES or normalized.startswith(
        _IMAGE_RUNTIME_PREFIXES
    )


def _package_name(segment: list[str]) -> str:
    for line in segment[1:]:
        match = _PACKAGE_NAME.match(line.strip())
        if match:
            return match.group(1)
    raise ValueError("Poetry lock contains a package table without a package name")


def filter_lock_text(lock_text: str) -> tuple[str, list[str]]:
    """Remove only image-owned package tables while preserving all other lock text."""
    try:
        parsed = tomllib.loads(lock_text)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"Poetry lock is not valid TOML: {exc}") from exc

    packages = parsed.get("package", [])
    if not isinstance(packages, list):
        raise ValueError("Poetry lock does not contain a package array")

    lines = lock_text.splitlines(keepends=True)
    boundaries = [
        index
        for index, line in enumerate(lines)
        if line.strip() in {_PACKAGE_START, _METADATA_START}
    ]
    if not boundaries:
        raise ValueError("Poetry lock does not contain package or metadata sections")

    output = lines[: boundaries[0]]
    removed: list[str] = []
    for boundary_index, start in enumerate(boundaries):
        end = boundaries[boundary_index + 1] if boundary_index + 1 < len(boundaries) else len(lines)
        segment = lines[start:end]
        if lines[start].strip() == _PACKAGE_START:
            name = _package_name(segment)
            if _is_image_runtime_package(name):
                removed.append(name)
                continue
        output.extend(segment)

    filtered_text = "".join(output)
    try:
        filtered = tomllib.loads(filtered_text)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"Filtered Poetry lock is not valid TOML: {exc}") from exc

    remaining_image_packages = sorted(
        str(package.get("name", ""))
        for package in filtered.get("package", [])
        if isinstance(package, dict)
        and _is_image_runtime_package(str(package.get("name", "")))
    )
    if remaining_image_packages:
        raise ValueError(
            "Filtered Poetry lock still contains image packages: "
            + ", ".join(remaining_image_packages)
        )
    return filtered_text, sorted(set(removed))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        source_text = args.input.read_text(encoding="utf-8")
        filtered_text, removed = filter_lock_text(source_text)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(filtered_text, encoding="utf-8")
    except (OSError, ValueError) as exc:
        print(str(exc))
        return 2

    print(
        "Prepared Poetry install view without RunPod image packages: "
        + (", ".join(removed) if removed else "none")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
