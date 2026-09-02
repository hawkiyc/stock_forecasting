"""Dependency-light runtime CPU and memory resource detection."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CGROUP_V1_UNLIMITED_THRESHOLD_BYTES = 1 << 60


@dataclass(frozen=True)
class AvailableMemoryEstimate:
    """Available memory selected from scope-compatible runtime observations."""

    available_bytes: int
    source: str
    observations: tuple[tuple[str, int], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "available_bytes": self.available_bytes,
            "source": self.source,
            "observations_bytes": dict(self.observations),
        }


def select_visible_cpu_count(
    *,
    reported_cpu_count: int | None,
    affinity_cpu_count: int | None = None,
) -> int:
    """Select the process-visible CPU count from host and affinity limits."""

    reported = 1 if reported_cpu_count is None else reported_cpu_count
    if isinstance(reported, bool) or reported < 1:
        raise ValueError("reported_cpu_count must be positive when provided")
    if affinity_cpu_count is not None and (
        isinstance(affinity_cpu_count, bool) or affinity_cpu_count < 1
    ):
        raise ValueError("affinity_cpu_count must be positive when provided")
    return max(1, min(reported, affinity_cpu_count or reported))


def detect_visible_cpu_count() -> int:
    """Return the CPU count visible to this process, including affinity limits."""

    affinity_count: int | None = None
    if hasattr(os, "sched_getaffinity"):
        try:
            affinity_count = len(os.sched_getaffinity(0))
        except OSError:
            affinity_count = None
    return select_visible_cpu_count(
        reported_cpu_count=os.cpu_count(),
        affinity_cpu_count=affinity_count,
    )


def select_available_memory_estimate(
    *,
    cgroup_headrooms: Sequence[tuple[str, int]] = (),
    linux_mem_available_bytes: int | None = None,
    posix_free_pages_bytes: int | None = None,
) -> AvailableMemoryEstimate:
    """Select safe capacity without treating POSIX free pages as Linux availability."""

    observations: list[tuple[str, int]] = []
    eligible: list[tuple[str, int]] = []
    for name, value in cgroup_headrooms:
        if isinstance(value, bool) or value < 1:
            raise ValueError("cgroup headroom observations must be positive integers")
        observation = (name, value)
        observations.append(observation)
        eligible.append(observation)

    if linux_mem_available_bytes is not None:
        if (
            isinstance(linux_mem_available_bytes, bool)
            or linux_mem_available_bytes < 1
        ):
            raise ValueError("linux_mem_available_bytes must be a positive integer")
        observation = ("linux_mem_available", linux_mem_available_bytes)
        observations.append(observation)
        eligible.append(observation)

    if posix_free_pages_bytes is not None:
        if isinstance(posix_free_pages_bytes, bool) or posix_free_pages_bytes < 1:
            raise ValueError("posix_free_pages_bytes must be a positive integer")
        observations.append(("posix_free_pages_fallback", posix_free_pages_bytes))

    # SC_AVPHYS_PAGES reports immediately free pages and excludes reclaimable
    # Linux page cache. It is only a fallback when cgroup headroom and the
    # kernel's MemAvailable estimate are both unavailable.
    candidates = eligible or [
        observation
        for observation in observations
        if observation[0] == "posix_free_pages_fallback"
    ]
    if not candidates:
        raise RuntimeError("Unable to determine available runtime memory")
    source, available_bytes = min(candidates, key=lambda observation: observation[1])
    return AvailableMemoryEstimate(
        available_bytes=available_bytes,
        source=source,
        observations=tuple(observations),
    )


def _read_nonnegative_integer(path: Path) -> int | None:
    try:
        value = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return None
    if not value or value == "max":
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed >= 0 else None


def _read_linux_mem_available_bytes() -> int | None:
    try:
        meminfo = Path("/proc/meminfo").read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None
    for line in meminfo.splitlines():
        fields = line.split()
        if line.startswith("MemAvailable:") and len(fields) >= 2 and fields[1].isdigit():
            available_bytes = int(fields[1]) * 1024
            return available_bytes if available_bytes > 0 else None
    return None


def _read_posix_free_pages_bytes() -> int | None:
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        available_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
    except (OSError, ValueError, TypeError, AttributeError):
        return None
    available_bytes = page_size * available_pages
    return available_bytes if available_bytes > 0 else None


def detect_available_memory() -> AvailableMemoryEstimate:
    """Detect cgroup headroom and Linux reclaimable memory at runtime."""

    cgroup_headrooms: list[tuple[str, int]] = []
    cgroup_pairs = (
        (
            "cgroup_v2_headroom",
            Path("/sys/fs/cgroup/memory.max"),
            Path("/sys/fs/cgroup/memory.current"),
            False,
        ),
        (
            "cgroup_v1_headroom",
            Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
            Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"),
            True,
        ),
    )
    for source, limit_path, usage_path, has_unlimited_sentinel in cgroup_pairs:
        limit = _read_nonnegative_integer(limit_path)
        usage = _read_nonnegative_integer(usage_path)
        if (
            limit is None
            or usage is None
            or limit <= usage
            or (has_unlimited_sentinel and limit >= CGROUP_V1_UNLIMITED_THRESHOLD_BYTES)
        ):
            continue
        cgroup_headrooms.append((source, limit - usage))

    return select_available_memory_estimate(
        cgroup_headrooms=cgroup_headrooms,
        linux_mem_available_bytes=_read_linux_mem_available_bytes(),
        posix_free_pages_bytes=_read_posix_free_pages_bytes(),
    )
