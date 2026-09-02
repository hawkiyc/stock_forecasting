"""Fail-closed tests for stale RunPod GPU lifecycle reconciliation."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from stock_forecasting.data.content_identity import (
    code_content_identity,
    dataset_content_identity,
    semantic_source_paths,
)

ROOT = Path(__file__).resolve().parents[1]
READINESS_PATH = ROOT / "scripts/runpod_readiness.py"
RECONCILER_PATH = ROOT / "scripts/ensure_runpod_gpu_workflow_available.sh"
SPEC = importlib.util.spec_from_file_location("runpod_gpu_readiness", READINESS_PATH)
assert SPEC is not None and SPEC.loader is not None
READINESS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(READINESS)


def _active_training_marker() -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": "stage1-training",
        "state": "preparing",
        "generated_at": "2026-09-02T12:25:56+00:00",
        "pod_id": "stale-gpu-pod",
        "launch_id": "launch-20260902T122555Z-91844384",
        "wandb_run_id": "run-20260902T122329Z-2542323195",
        "log_path": (
            "/runpod-volume/logs/run-20260902T122329Z-2542323195/launcher/"
            "launch-20260902T122555Z-91844384"
        ),
        "max_runtime_seconds": 86400,
    }


@pytest.mark.parametrize(
    ("payload", "exit_code", "expected"),
    (
        ({"id": "stale-gpu-pod"}, 0, "present"),
        (
            {
                "error": "failed to get pod: pod not found",
                "code": "not_found",
                "status": 404,
            },
            1,
            "absent",
        ),
    ),
)
def test_runpod_pod_probe_accepts_only_explicit_present_or_not_found(
    payload: dict[str, object],
    exit_code: int,
    expected: str,
) -> None:
    assert (
        READINESS._runpod_pod_probe_state(
            payload,
            expected_pod_id="stale-gpu-pod",
            command_exit_code=exit_code,
        )
        == expected
    )


@pytest.mark.parametrize(
    "payload",
    (
        {"error": "request timed out", "code": "network_error"},
        {"error": "forbidden", "code": "forbidden", "status": 403},
        {"id": "different-pod"},
    ),
)
def test_runpod_pod_probe_fails_closed_for_indeterminate_responses(
    payload: dict[str, object],
) -> None:
    exit_code = 0 if "id" in payload else 1
    with pytest.raises(ValueError, match=r"indeterminate|different Pod"):
        READINESS._runpod_pod_probe_state(
            payload,
            expected_pod_id="stale-gpu-pod",
            command_exit_code=exit_code,
        )


def test_orphan_reconciliation_preserves_run_identity_and_resume_metadata() -> None:
    marker = _active_training_marker()
    observed_at = datetime(2026, 9, 2, 12, 30, tzinfo=UTC)

    reconciled = READINESS._orphaned_gpu_workflow_payload(
        marker,
        expected_kind="stage1-training",
        expected_pod_id="stale-gpu-pod",
        volume_root=Path("/runpod-volume"),
        minimum_age_seconds=60,
        now=observed_at,
    )

    assert reconciled["state"] == "failed"
    assert reconciled["reason"] == "runpod_pod_not_found"
    assert reconciled["pod_id"] == marker["pod_id"]
    assert reconciled["launch_id"] == marker["launch_id"]
    assert reconciled["wandb_run_id"] == marker["wandb_run_id"]
    assert reconciled["log_path"] == marker["log_path"]
    assert reconciled["training_completed"] is False
    assert reconciled["resume_discovery_required"] is True
    assert reconciled["timed_out"] is False
    assert reconciled["reconciliation"] == {
        "previous_state": "preparing",
        "pod_lookup": "not_found",
        "reconciled_at": observed_at.isoformat(),
    }


def test_recent_active_lifecycle_cannot_be_reconciled_from_a_transient_404() -> None:
    marker = _active_training_marker()
    generated_at = datetime.fromisoformat(str(marker["generated_at"]))

    with pytest.raises(ValueError, match="too recent"):
        READINESS._orphaned_gpu_workflow_payload(
            marker,
            expected_kind="stage1-training",
            expected_pod_id="stale-gpu-pod",
            volume_root=Path("/runpod-volume"),
            minimum_age_seconds=60,
            now=generated_at + timedelta(seconds=30),
        )


def test_orphaned_finalization_preserves_completed_training() -> None:
    marker = {
        **_active_training_marker(),
        "state": "finalizing",
        "training_completed": True,
    }

    reconciled = READINESS._orphaned_gpu_workflow_payload(
        marker,
        expected_kind="stage1-training",
        expected_pod_id="stale-gpu-pod",
        volume_root=Path("/runpod-volume"),
        minimum_age_seconds=60,
        now=datetime(2026, 9, 2, 12, 30, tzinfo=UTC),
    )

    assert reconciled["state"] == "failed"
    assert reconciled["training_completed"] is True
    assert reconciled["resume_discovery_required"] is False


def test_dataset_compatibility_ignores_release_only_changes() -> None:
    selected_datasets = ["eodhd_us"]
    code_identity = code_content_identity()
    content_identity = dataset_content_identity(
        selected_datasets,
        provider_digests=code_identity["provider_materialization_digests"],
    )
    pipeline_digest = READINESS._payload_sha256(content_identity)
    dataset = {
        "selected_datasets": selected_datasets,
        "data_content_identity": content_identity,
        "data_pipeline_digest": pipeline_digest,
        "code_release_digest": "b" * 64,
    }

    READINESS._validate_dataset_code_compatibility(
        dataset,
        expected_numerical_pipeline_digest=pipeline_digest,
    )
    dataset["code_release_digest"] = "c" * 64
    READINESS._validate_dataset_code_compatibility(
        dataset,
        expected_numerical_pipeline_digest=pipeline_digest,
    )

    with pytest.raises(ValueError, match="different numerical pipeline"):
        READINESS._validate_dataset_code_compatibility(
            dataset,
            expected_numerical_pipeline_digest="d" * 64,
        )


def test_control_plane_uses_the_dataset_builders_exact_numerical_scope() -> None:
    identity, paths = READINESS._project_content_identity(ROOT)

    assert identity == code_content_identity()
    assert paths == semantic_source_paths()
    assert "src/stock_forecasting/data/providers/http.py" not in paths
    assert not any(path.startswith("scripts/") for path in paths)
    code_payload = {"data_content_identity": identity}
    assert READINESS._numerical_pipeline_digest_from_code_payload(
        code_payload
    ) == READINESS._payload_sha256(identity)
    selected_identity = dataset_content_identity(
        ["eodhd_us"],
        provider_digests=identity["provider_materialization_digests"],
    )
    assert READINESS._numerical_pipeline_digest_from_code_payload(
        code_payload,
        ["eodhd_us"],
    ) == READINESS._payload_sha256(selected_identity)


def _write_mock_wrappers(tmp_path: Path) -> tuple[Path, Path]:
    s3_wrapper = tmp_path / "mock-s3.sh"
    s3_wrapper.write_text(
        """#!/usr/bin/env bash
set -eu
if [[ "$1" != "s3" || "$2" != "cp" ]]; then
    exit 64
fi
if [[ "$3" == s3://* && "$4" == "-" ]]; then
    /bin/cat "${MOCK_MARKER_PATH}"
    exit 0
fi
if [[ "$3" == "-" && "$4" == s3://* ]]; then
    /bin/cat > "${MOCK_MARKER_PATH}"
    exit 0
fi
exit 64
""",
        encoding="utf-8",
    )
    pod_wrapper = tmp_path / "mock-runpodctl.sh"
    pod_wrapper.write_text(
        """#!/usr/bin/env bash
set -eu
printf 'probe\n' >> "${MOCK_PROBE_COUNT_PATH}"
printf '%s\n' "${MOCK_POD_RESPONSE}"
exit "${MOCK_POD_EXIT_CODE}"
""",
        encoding="utf-8",
    )
    s3_wrapper.chmod(0o700)
    pod_wrapper.chmod(0o700)
    return s3_wrapper, pod_wrapper


def _run_reconciler(
    tmp_path: Path,
    marker: dict[str, object],
    *,
    pod_response: dict[str, object],
    pod_exit_code: int,
) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
    marker_path = tmp_path / "training.json"
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    probe_count_path = tmp_path / "probe-count.txt"
    s3_wrapper, pod_wrapper = _write_mock_wrappers(tmp_path)
    environment = {
        **os.environ,
        "RUNPOD_NETWORK_VOLUME_ID": "test-volume",
        "RUNPOD_VOLUME_MOUNT_PATH": "/runpod-volume",
        "RUNPOD_S3_WRAPPER": str(s3_wrapper),
        "RUNPODCTL_WRAPPER": str(pod_wrapper),
        "RUNPOD_READINESS_HELPER": str(READINESS_PATH),
        "RUNPOD_ORPHAN_CONFIRMATIONS": "2",
        "RUNPOD_ORPHAN_CONFIRMATION_DELAY_SECONDS": "0",
        "RUNPOD_ORPHAN_MINIMUM_AGE_SECONDS": "60",
        "MOCK_MARKER_PATH": str(marker_path),
        "MOCK_PROBE_COUNT_PATH": str(probe_count_path),
        "MOCK_POD_RESPONSE": json.dumps(pod_response),
        "MOCK_POD_EXIT_CODE": str(pod_exit_code),
    }
    result = subprocess.run(
        [
            "bash",
            str(RECONCILER_PATH),
            "lifecycle/stage1/training.json",
            "stage1-training",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    return result, marker_path, probe_count_path


def test_reconciler_requires_two_404s_before_publishing_failed_marker(
    tmp_path: Path,
) -> None:
    result, marker_path, probe_count_path = _run_reconciler(
        tmp_path,
        _active_training_marker(),
        pod_response={
            "error": "failed to get pod: pod not found",
            "code": "not_found",
            "status": 404,
        },
        pod_exit_code=1,
    )

    assert result.returncode == 0, result.stderr
    assert probe_count_path.read_text(encoding="utf-8").splitlines() == [
        "probe",
        "probe",
    ]
    reconciled = json.loads(marker_path.read_text(encoding="utf-8"))
    assert reconciled["state"] == "failed"
    assert reconciled["reason"] == "runpod_pod_not_found"
    assert "Reconciled orphaned GPU lifecycle" in result.stderr


def test_reconciler_blocks_when_marker_pod_still_exists(tmp_path: Path) -> None:
    marker = _active_training_marker()
    result, marker_path, probe_count_path = _run_reconciler(
        tmp_path,
        marker,
        pod_response={"id": "stale-gpu-pod", "desiredStatus": "RUNNING"},
        pod_exit_code=0,
    )

    assert result.returncode == 2
    assert "still exists" in result.stderr
    assert probe_count_path.read_text(encoding="utf-8").splitlines() == ["probe"]
    assert json.loads(marker_path.read_text(encoding="utf-8")) == marker


def test_reconciler_blocks_on_runpod_network_error(tmp_path: Path) -> None:
    marker = _active_training_marker()
    result, marker_path, probe_count_path = _run_reconciler(
        tmp_path,
        marker,
        pod_response={"error": "request timed out", "code": "network_error"},
        pod_exit_code=1,
    )

    assert result.returncode == 2
    assert "refusing to create a competing paid Pod" in result.stderr
    assert probe_count_path.read_text(encoding="utf-8").splitlines() == ["probe"]
    assert json.loads(marker_path.read_text(encoding="utf-8")) == marker


def test_gpu_gate_uses_pipeline_digest_and_orphan_reconciler() -> None:
    gate = (ROOT / "scripts/verify_runpod_stage_readiness.sh").read_text(
        encoding="utf-8"
    )
    creator = (ROOT / "scripts/create_runpod_pod.sh").read_text(encoding="utf-8")

    assert "code-numerical-pipeline-digest" in gate
    assert "--expected-numerical-pipeline-digest" in gate
    assert "--expected-code-release-digest" not in gate
    assert "ensure_runpod_gpu_workflow_available.sh" in creator
