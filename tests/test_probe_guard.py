"""Exercise the diagnostic publisher and real local guard with fake cloud transports."""

from __future__ import annotations

import io
import json
import runpy
import shlex
import shutil
import subprocess
import sys
import unittest
from argparse import Namespace
from unittest.mock import patch

from test_probe_tmux import ROOT, ProbeHarness

CONTRACT = runpy.run_path(str(ROOT / "scripts/runpod_probe_lifecycle.py"))


class ProbeGuardTests(unittest.TestCase):
    def harness(self) -> ProbeHarness:
        harness = ProbeHarness()
        self.addCleanup(harness.temporary.cleanup)
        for relative in (
            "scripts/terminate_runpod_after.sh",
            "scripts/runpodctl_project.sh",
            "scripts/lib/runpod_project_env.sh",
            "scripts/runpod_readiness.py",
            "src/stock_forecasting/data/content_identity.py",
            "src/stock_forecasting/dataset_identity.py",
            "src/stock_forecasting/dataset_profiles.py",
            "src/stock_forecasting/training_stage_contract.py",
        ):
            target = harness.project / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, target)
        env_file = harness.project / ".env"
        env_file.write_text(
            "RUNPOD_API_KEY=local-test-only\nRUNPOD_NETWORK_VOLUME_ID=fixture-volume\n",
            encoding="utf-8",
        )
        env_file.chmod(0o600)
        harness.write_script(
            harness.bin / "python3", "exec " + shlex.quote(sys.executable) + ' "$@"\n'
        )
        harness.write_script(harness.bin / "aws", "exit 90\n")
        # Use the real project credential wrapper; only the final CLI is fake.
        harness.write_script(
            harness.bin / "runpodctl",
            r"""
            root="$(cd "$(dirname "$0")/.." && pwd)"
            [[ "${RUNPOD_API_KEY:-}" == local-test-only ]] || exit 91
            [[ "$*" == 'pod delete probe-fixture' ]] || exit 92
            printf '%s\n' "$*" >> "${root}/local-delete-calls"
            """,
        )
        harness.write_script(
            harness.scripts / "runpod_s3_project.sh",
            r"""
            [[ "$1" == s3 && "$2" == cp ]] || exit 93
            if [[ "$3" == - ]]; then
                printf '%s\n' "$4" >> "${HARNESS_ROOT}/s3-write-calls"
                cat >/dev/null
                exit 0
            fi
            key="${3#s3://fixture-volume/}"
            [[ "${key}" != "$3" ]] || exit 94
            if [[ "${HARNESS_DIAGNOSTIC_TRANSIENT:-0}" == 1 \
                && "${key}" == lifecycle/diagnostics/* ]]; then
                [[ ! -f "${HARNESS_ROOT}/diagnostic-read-once" ]] || exit 5
                : > "${HARNESS_ROOT}/diagnostic-read-once"
            fi
            cat "${HARNESS_ROOT}/volume/${key}"
            exit "${HARNESS_S3_READ_EXIT:-0}"
            """,
        )
        return harness

    def payload(self, harness: ProbeHarness, state: str = "succeeded", code: int = 0) -> dict:
        job_dir = harness.volume / "logs/tmux/fin-ts-probe-scales/launch-fixture"
        return {
            "schema_version": 1,
            "kind": "representation-scale-probe",
            "pod_id": "probe-fixture",
            "owner_run_id": "run-pod-owner",
            "launch_id": "launch-fixture",
            "gpu_lease_acquired": True,
            "state": state,
            "exit_code": code,
            "generated_at": "2026-09-14T08:00:00+00:00",
            "log_path": str(job_dir / "combined.log"),
            "status_path": str(job_dir / "status.json"),
        }

    def run_guard(
        self, harness: ProbeHarness, seconds: int = 5, lifecycle: str = "training"
    ) -> subprocess.CompletedProcess:
        environment = {
            **harness.environment,
            "RUNPOD_TEST_MODE": "1",
            "RUNPOD_GUARD_LIFECYCLE_KEY": f"lifecycle/stage1/{lifecycle}.json",
            "RUNPOD_GUARD_RUN_ID": (
                "run-pod-owner" if lifecycle in {"training", "validation"} else ""
            ),
            "RUNPOD_GUARD_VOLUME_ROOT": str(harness.volume),
            "RUNPOD_GUARD_POLL_SECONDS": "1",
            "RUNPOD_GUARD_MAX_ATTEMPTS": "1",
        }
        environment.pop("RUNPOD_POD_ID")
        return subprocess.run(
            [
                "bash", str(harness.scripts / "terminate_runpod_after.sh"),
                "probe-fixture", str(seconds), str(harness.root / "local-guard.log"),
            ],
            env=environment, capture_output=True, text=True, check=False, timeout=12,
        )

    def test_complete_runner_to_local_guard_to_project_cli_chain(self) -> None:
        for environment, state, code in (
            ({}, "succeeded", 0),
            ({"HARNESS_JOB_EXIT": "7"}, "failed", 7),
            ({"HARNESS_TIMEOUT": "1"}, "timed_out", 124),
        ):
            with self.subTest(state=state):
                harness = self.harness()
                harness.environment.update(environment)
                launch = harness.command("runpod_tmux_launch.sh", "probe-scales")
                self.assertEqual(launch.returncode, 0, launch.stderr)
                runner = harness.run_worker()
                self.assertEqual(runner.returncode, code, runner.stdout + runner.stderr)
                self.assertFalse((harness.root / "shutdown-called").exists())
                result = self.run_guard(harness)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(harness.read("local-delete-calls"), "pod delete probe-fixture\n")
                log = harness.read("local-guard.log")
                self.assertIn("termination triggered by lifecycle-" + state, log)
                self.assertIn("lifecycle/diagnostics/representation-scales/probe-fixture.json", log)
                self.assertFalse((harness.root / "s3-write-calls").exists())
                self.assertEqual(list((harness.volume / "lifecycle/stage1").glob("*.json")), [])

    def test_running_diagnostic_waits_for_hard_limit_without_training_timeout_writes(self) -> None:
        harness = self.harness()
        marker = harness.diagnostic_marker()
        marker.parent.mkdir(parents=True)
        marker.write_text(json.dumps(self.payload(harness, "running")), encoding="utf-8")
        result = self.run_guard(harness, seconds=1)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        log = harness.read("local-guard.log")
        self.assertIn("termination triggered by hard-limit", log)
        self.assertNotIn("lifecycle-running", log)
        self.assertFalse((harness.root / "s3-write-calls").exists())

    def test_s3_failure_cannot_authorize_termination_even_after_valid_json(self) -> None:
        harness = self.harness()
        harness.environment["HARNESS_S3_READ_EXIT"] = "5"
        marker = harness.diagnostic_marker()
        marker.parent.mkdir(parents=True)
        marker.write_text(json.dumps(self.payload(harness)), encoding="utf-8")
        result = self.run_guard(harness, seconds=1)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("termination triggered by hard-limit", harness.read("local-guard.log"))

    def test_guard_keeps_diagnostic_ownership_across_transient_read_failures(self) -> None:
        harness = self.harness()
        harness.environment["HARNESS_DIAGNOSTIC_TRANSIENT"] = "1"
        marker = harness.diagnostic_marker()
        marker.parent.mkdir(parents=True)
        marker.write_text(json.dumps(self.payload(harness, "running")), encoding="utf-8")
        training = harness.volume / "lifecycle/stage1/training.json"
        training.parent.mkdir(parents=True)
        training.write_text(json.dumps({
            "schema_version": 1, "kind": "stage1-training", "state": "failed",
            "pod_id": "probe-fixture", "wandb_run_id": "run-pod-owner",
            "launch_id": "launch-training", "exit_code": 7,
        }))
        result = self.run_guard(harness, seconds=2)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        log = harness.read("local-guard.log")
        self.assertIn("termination triggered by hard-limit", log)
        self.assertNotIn("termination triggered by lifecycle-failed", log)
        self.assertFalse((harness.root / "s3-write-calls").exists())

    def test_wrong_pod_or_owner_cannot_trigger_diagnostic_termination(self) -> None:
        for field, value in (("pod_id", "other-pod"), ("owner_run_id", "historical-model-run")):
            with self.subTest(field=field):
                harness = self.harness()
                marker = harness.diagnostic_marker()
                marker.parent.mkdir(parents=True)
                marker.write_text(json.dumps({**self.payload(harness), field: value}))
                result = self.run_guard(harness, seconds=1)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                log = harness.read("local-guard.log")
                self.assertIn("termination triggered by hard-limit", log)
                self.assertNotIn("termination triggered by lifecycle-succeeded", log)

    def test_existing_workflow_markers_still_trigger_local_termination(self) -> None:
        for lifecycle in ("cpu-preparation", "mixed-finalization", "training", "validation"):
            with self.subTest(lifecycle=lifecycle):
                harness = self.harness()
                marker = harness.volume / "lifecycle/stage1" / (lifecycle + ".json")
                marker.parent.mkdir(parents=True)
                marker.write_text(json.dumps({
                    "schema_version": 1, "kind": "stage1-" + lifecycle, "state": "failed",
                    "pod_id": "probe-fixture", "wandb_run_id": "run-pod-owner",
                    "launch_id": "launch-existing", "exit_code": 7,
                }))
                result = self.run_guard(harness, lifecycle=lifecycle)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn(
                    "termination triggered by lifecycle-failed", harness.read("local-guard.log")
                )

    def test_marker_validation_rejects_untrusted_fields_and_inconsistent_states(self) -> None:
        harness = self.harness()
        base = self.payload(harness)
        for field, value in (
            ("schema_version", True), ("kind", "stage1-training"),
            ("pod_id", "another-pod"), ("owner_run_id", "run-checkpoint"),
            ("launch_id", "../unsafe"), ("gpu_lease_acquired", False),
            ("log_path", "/tmp/other.log"), ("status_path", "/tmp/other.json"),
            ("state", "complete"), ("state", "failed"), ("state", "timed_out"),
            ("exit_code", True), ("exit_code", -1), ("exit_code", 256),
            ("generated_at", "2026-09-14T08:00:00"), ("generated_at", "invalid"),
        ):
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                CONTRACT["validate"](
                    {**base, field: value}, str(harness.volume), "probe-fixture", "run-pod-owner"
                )

    def test_json_reads_are_bounded(self) -> None:
        with self.assertRaisesRegex(ValueError, "bounded read limit"):
            CONTRACT["load_bounded"](io.BytesIO(b" " * 65537))

    def test_terminal_publication_requires_persisted_matching_status(self) -> None:
        harness = self.harness()
        arguments = Namespace(
            network_volume_root=str(harness.volume), pod_id="probe-fixture",
            owner_run_id="run-pod-owner", launch_id="launch-fixture",
            state="succeeded", exit_code=0,
        )
        with (
            patch.dict("os.environ", {"RUNPOD_GPU_WORKFLOW_LEASE_HELD": "1"}),
            patch("os.fstat"),
        ):
            with self.assertRaises(FileNotFoundError):
                CONTRACT["publish"](arguments)
            paths = CONTRACT["identity"](
                str(harness.volume), "probe-fixture", "run-pod-owner", "launch-fixture"
            )
            paths["status_path"].parent.mkdir(parents=True)
            paths["status_path"].write_text(json.dumps({
                "state": "failed", "exit_code": 7, "log_path": str(paths["log_path"]),
            }))
            with self.assertRaisesRegex(ValueError, "matching persisted tmux status"):
                CONTRACT["publish"](arguments)
            self.assertFalse(harness.diagnostic_marker().exists())


if __name__ == "__main__":
    unittest.main()
