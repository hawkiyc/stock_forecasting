"""Fail-closed lifecycle contracts for the local RunPod termination guard."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = ROOT / "scripts/runpod_readiness.py"
SPEC = importlib.util.spec_from_file_location("runpod_guard_readiness", HELPER_PATH)
assert SPEC is not None and SPEC.loader is not None
READINESS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(READINESS)


def test_guard_accepts_matching_cpu_preparation_schema_v1() -> None:
    marker = {
        "schema_version": 1,
        "kind": "stage1-cpu-preparation",
        "state": "ready",
        "pod_id": "test-pod",
    }

    assert (
        READINESS._guard_lifecycle_state(
            marker,
            expected_kind="stage1-cpu-preparation",
            expected_pod_id="test-pod",
        )
        == "ready"
    )


@pytest.mark.parametrize(
    "marker",
    (
        {
            "schema_version": 2,
            "kind": "stage1-cpu-preparation",
            "state": "ready",
            "pod_id": "test-pod",
        },
        {
            "schema_version": 2,
            "kind": "stage1-cpu-preparation",
            "state": "failed",
            "pod_id": "test-pod",
        },
    ),
)
def test_guard_rejects_schema_versions_outside_the_state_contract(
    marker: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="schema"):
        READINESS._guard_lifecycle_state(
            marker,
            expected_kind="stage1-cpu-preparation",
            expected_pod_id="test-pod",
        )


def test_guard_rejects_cpu_preparation_from_another_pod() -> None:
    marker = {
        "schema_version": 1,
        "kind": "stage1-cpu-preparation",
        "state": "ready",
        "pod_id": "stale-pod",
    }

    with pytest.raises(ValueError, match="monitored Pod"):
        READINESS._guard_lifecycle_state(
            marker,
            expected_kind="stage1-cpu-preparation",
            expected_pod_id="current-pod",
        )


def test_guard_preserves_schema_v1_failure_and_resumable_semantics() -> None:
    failed = {
        "schema_version": 1,
        "kind": "stage1-cpu-preparation",
        "state": "failed",
        "pod_id": "test-pod",
    }
    downloaded_active = {
        "schema_version": 1,
        "kind": "stage1-cpu-preparation",
        "state": "downloaded",
        "pod_id": "test-pod",
    }
    downloaded_terminal = {**downloaded_active, "exit_code": 75}

    assert (
        READINESS._guard_lifecycle_state(
            failed,
            expected_kind="stage1-cpu-preparation",
            expected_pod_id="test-pod",
        )
        == "failed"
    )
    assert (
        READINESS._guard_lifecycle_state(
            downloaded_active,
            expected_kind="stage1-cpu-preparation",
            expected_pod_id="test-pod",
        )
        == "downloaded_active"
    )
    assert (
        READINESS._guard_lifecycle_state(
            downloaded_terminal,
            expected_kind="stage1-cpu-preparation",
            expected_pod_id="test-pod",
        )
        == "downloaded"
    )


def test_guard_preserves_gpu_run_identity_and_completion_checks() -> None:
    marker = {
        "schema_version": 1,
        "kind": "stage1-training",
        "state": "ready",
        "pod_id": "gpu-pod",
        "wandb_run_id": "run-123",
        "training_completed": True,
    }

    assert (
        READINESS._guard_lifecycle_state(
            marker,
            expected_kind="stage1-training",
            expected_pod_id="gpu-pod",
            active_run_id="run-123",
        )
        == "ready"
    )
    with pytest.raises(ValueError, match="active run"):
        READINESS._guard_lifecycle_state(
            marker,
            expected_kind="stage1-training",
            expected_pod_id="gpu-pod",
            active_run_id="other-run",
        )

    incomplete = {**marker, "training_completed": False}
    with pytest.raises(ValueError, match="incomplete"):
        READINESS._guard_lifecycle_state(
            incomplete,
            expected_kind="stage1-training",
            expected_pod_id="gpu-pod",
            active_run_id="run-123",
        )
