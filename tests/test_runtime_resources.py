"""Runtime CPU and memory resource detection contracts."""

from __future__ import annotations

import pytest

from stock_forecasting import training
from stock_forecasting.runtime_resources import (
    AvailableMemoryEstimate,
    select_available_memory_estimate,
    select_visible_cpu_count,
)


def test_linux_available_memory_ignores_low_posix_free_page_fallback() -> None:
    cgroup_headroom = 59_999_997_952 - 7_939_940_352
    linux_mem_available = 73_760_624 * 1024
    posix_free_pages = 4_067_721_216

    estimate = select_available_memory_estimate(
        cgroup_headrooms=(("cgroup_v2_headroom", cgroup_headroom),),
        linux_mem_available_bytes=linux_mem_available,
        posix_free_pages_bytes=posix_free_pages,
    )

    assert estimate.available_bytes == cgroup_headroom
    assert estimate.source == "cgroup_v2_headroom"
    assert dict(estimate.observations)["posix_free_pages_fallback"] == posix_free_pages


def test_linux_mem_available_still_limits_larger_cgroup_headroom() -> None:
    estimate = select_available_memory_estimate(
        cgroup_headrooms=(("cgroup_v2_headroom", 48 * 1024**3),),
        linux_mem_available_bytes=32 * 1024**3,
        posix_free_pages_bytes=4 * 1024**3,
    )

    assert estimate.available_bytes == 32 * 1024**3
    assert estimate.source == "linux_mem_available"


def test_posix_free_pages_are_used_only_as_a_fallback() -> None:
    estimate = select_available_memory_estimate(posix_free_pages_bytes=4 * 1024**3)

    assert estimate.available_bytes == 4 * 1024**3
    assert estimate.source == "posix_free_pages_fallback"

    with pytest.raises(RuntimeError, match="Unable to determine available runtime memory"):
        select_available_memory_estimate()


def test_runpod_memory_observations_allow_requested_dataloader_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cgroup_headroom = 59_999_997_952 - 7_939_940_352
    estimate = AvailableMemoryEstimate(
        available_bytes=cgroup_headroom,
        source="cgroup_v2_headroom",
        observations=(
            ("cgroup_v2_headroom", cgroup_headroom),
            ("linux_mem_available", 73_760_624 * 1024),
            ("posix_free_pages_fallback", 4_067_721_216),
        ),
    )
    monkeypatch.setattr(training, "detect_available_memory", lambda: estimate)

    plan = training.plan_dataloader_workers(
        8,
        source="runpod_auto",
        visible_cpu_count=32,
    )

    assert plan.effective_workers == 8
    assert plan.available_memory_bytes == cgroup_headroom
    assert plan.available_memory_source == "cgroup_v2_headroom"
    assert plan.as_dict()["available_memory_observations_bytes"] == dict(
        estimate.observations
    )


def test_visible_cpu_count_respects_affinity_limit() -> None:
    assert select_visible_cpu_count(
        reported_cpu_count=32,
        affinity_cpu_count=16,
    ) == 16
    assert select_visible_cpu_count(
        reported_cpu_count=8,
        affinity_cpu_count=16,
    ) == 8
    assert select_visible_cpu_count(
        reported_cpu_count=None,
        affinity_cpu_count=None,
    ) == 1

    with pytest.raises(ValueError, match="reported_cpu_count must be positive"):
        select_visible_cpu_count(reported_cpu_count=0)
    with pytest.raises(ValueError, match="reported_cpu_count must be positive"):
        select_visible_cpu_count(reported_cpu_count=False)
