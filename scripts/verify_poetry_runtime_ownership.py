#!/usr/bin/env python3
"""Ensure Poetry never owns packages supplied by the RunPod image."""

from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path
from typing import Any

_FORBIDDEN_NAMES = {"torch", "torchaudio", "torchvision", "triton"}
_FORBIDDEN_PREFIXES = ("cuda-", "nvidia-")


def _load_toml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required Poetry file is missing: {path}")
    with path.open("rb") as stream:
        payload = tomllib.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a TOML object in {path}")
    return payload


def _is_image_runtime_package(name: str) -> bool:
    normalized = name.lower().replace("_", "-")
    return normalized in _FORBIDDEN_NAMES or normalized.startswith(_FORBIDDEN_PREFIXES)


def _normalize_package_name(name: str) -> str:
    return name.lower().replace("_", "-")


def _requirement_name(requirement: str) -> str:
    name = requirement.split(";", maxsplit=1)[0].strip()
    for separator in (" ", "<", ">", "=", "!", "~", "@", "["):
        name = name.split(separator, maxsplit=1)[0]
    return _normalize_package_name(name)


def _project_dependency_names(pyproject: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    project = pyproject.get("project", {})
    if isinstance(project, dict):
        for requirement in project.get("dependencies", []):
            if isinstance(requirement, str):
                names.add(_requirement_name(requirement))
        optional_dependencies = project.get("optional-dependencies", {})
        if isinstance(optional_dependencies, dict):
            for requirements in optional_dependencies.values():
                if isinstance(requirements, list):
                    names.update(
                        _requirement_name(requirement)
                        for requirement in requirements
                        if isinstance(requirement, str)
                    )

    tool = pyproject.get("tool", {})
    poetry = tool.get("poetry", {}) if isinstance(tool, dict) else {}
    if isinstance(poetry, dict):
        dependencies = poetry.get("dependencies", {})
        if isinstance(dependencies, dict):
            names.update(
                _normalize_package_name(name)
                for name in dependencies
                if name != "python"
            )
        groups = poetry.get("group", {})
        if isinstance(groups, dict):
            for group in groups.values():
                if not isinstance(group, dict):
                    continue
                dependencies = group.get("dependencies", {})
                if isinstance(dependencies, dict):
                    names.update(_normalize_package_name(name) for name in dependencies)
    return names


def _transitive_locked_packages(
    pyproject: dict[str, Any], lock: dict[str, Any]
) -> set[str]:
    packages = lock.get("package", [])
    if not isinstance(packages, list):
        return set()
    dependency_graph: dict[str, set[str]] = {}
    for package in packages:
        if not isinstance(package, dict):
            continue
        name = _normalize_package_name(str(package.get("name", "")))
        if not name:
            continue
        dependencies = package.get("dependencies", {})
        dependency_graph[name] = {
            _normalize_package_name(str(dependency_name))
            for dependency_name in dependencies
        } if isinstance(dependencies, dict) else set()

    reachable: set[str] = set()
    pending = list(_project_dependency_names(pyproject))
    while pending:
        name = pending.pop()
        if name in reachable:
            continue
        reachable.add(name)
        pending.extend(dependency_graph.get(name, ()))
    return reachable


def find_violations(
    project_root: Path, *, allow_transitive_image_packages: bool = False
) -> list[str]:
    pyproject = _load_toml(project_root / "pyproject.toml")
    lock = _load_toml(project_root / "poetry.lock")
    violations: list[str] = []

    project = pyproject.get("project", {})
    if isinstance(project, dict):
        for requirement in project.get("dependencies", []):
            if not isinstance(requirement, str):
                continue
            name = _requirement_name(requirement)
            if _is_image_runtime_package(name):
                violations.append(f"project dependency must come from the image: {name}")

    tool = pyproject.get("tool", {})
    poetry = tool.get("poetry", {}) if isinstance(tool, dict) else {}
    if isinstance(poetry, dict):
        dependencies = poetry.get("dependencies", {})
        if isinstance(dependencies, dict):
            for name in dependencies:
                if _is_image_runtime_package(name):
                    violations.append(f"Poetry dependency must come from the image: {name}")
        for source in poetry.get("source", []):
            if isinstance(source, dict) and "download.pytorch.org" in str(source.get("url", "")):
                violations.append("Poetry must not configure a PyTorch wheel source")

    locked_runtime_packages = sorted(
        {
            str(package.get("name", ""))
            for package in lock.get("package", [])
            if isinstance(package, dict) and _is_image_runtime_package(str(package.get("name", "")))
        }
    )
    transitive_locked_packages = _transitive_locked_packages(pyproject, lock)
    unapproved_locked_runtime_packages = sorted(
        package
        for package in locked_runtime_packages
        if not allow_transitive_image_packages or package not in transitive_locked_packages
    )
    if unapproved_locked_runtime_packages:
        violations.append(
            "Poetry lock contains unapproved RunPod image packages: "
            + ", ".join(unapproved_locked_runtime_packages)
        )
    return violations


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument(
        "--allow-transitive-image-packages",
        action="store_true",
        help="Allow image packages pulled transitively by a project dependency.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        violations = find_violations(
            args.project_root.resolve(),
            allow_transitive_image_packages=args.allow_transitive_image_packages,
        )
    except (FileNotFoundError, ValueError, tomllib.TOMLDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if violations:
        for violation in violations:
            print(violation, file=sys.stderr)
        return 3
    print("Poetry ownership check passed: PyTorch and CUDA remain image-managed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
