"""Dependency-free tests for read-only scale-probe run discovery."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import stock_forecasting.probe_selection as selection
from stock_forecasting.runtime_resources import AvailableMemoryEstimate


class ProbeSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="scale-probe-selection-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve() / "savedModel"
        self.root.mkdir()
        memory = AvailableMemoryEstimate(8 * 1024**3, "fixture", ())
        self.cpu_patch = patch.object(selection, "detect_visible_cpu_count", return_value=4)
        self.memory_patch = patch.object(selection, "detect_available_memory", return_value=memory)
        self.cpu_patch.start()
        self.memory_patch.start()
        self.addCleanup(self.cpu_patch.stop)
        self.addCleanup(self.memory_patch.stop)

    def completed_run(
        self,
        name: str,
        timestamp: str = "2026-09-01T00:00:00+00:00",
        stop_reason: str = "epochs_completed",
    ) -> Path:
        run = self.root / name
        marker = run / selection.COMPLETION_METADATA
        marker.parent.mkdir(parents=True)
        marker.write_text(
            json.dumps(
                {
                    "schema_version": "4.0",
                    "kind": "training-completion-result",
                    "run_id": name,
                    "run_key": name,
                    "created_at": timestamp,
                    "stop_reason": stop_reason,
                }
            ),
            encoding="utf-8",
        )
        return run

    def latest(self, workers: int | None = None) -> Path:
        return selection.latest_completed_probe_run(
            self.root, schema_version="4.0", selection_workers=workers
        )

    def test_run_id_is_resolved_under_the_volume_without_discovery(self) -> None:
        run = self.root / "run-selected"
        run.mkdir()
        with patch.object(selection, "latest_completed_probe_run") as discovery:
            self.assertEqual(
                selection.probe_source_path(
                    run.name, saved_model_root=self.root, schema_version="4.0"
                ),
                run,
            )
        discovery.assert_not_called()

    def test_omitted_selector_uses_latest_completion_not_id_mtime_or_active_run(self) -> None:
        older = self.completed_run("run-z-new-name", "2026-09-02T00:00:00+00:00")
        latest = self.completed_run("run-a-old-name", "2026-09-03T00:00:00Z", "early_stopping")
        unfinished = self.root / "run-newest-in-progress"
        (unfinished / "checkpoint-999999").mkdir(parents=True)
        os.utime(older, (2_000_000_000, 2_000_000_000))
        with patch.dict(os.environ, {"WANDB_RUN_ID": unfinished.name}):
            selected = selection.probe_source_path(
                None, saved_model_root=self.root, schema_version="4.0"
            )
        self.assertEqual(selected, latest)

    def test_timestamp_offsets_and_ties_are_deterministic(self) -> None:
        self.completed_run("run-a", "2026-09-02T09:00:00+09:00")
        self.completed_run("run-b", "2026-09-02T00:00:00+00:00")
        expected = self.completed_run("run-c", "2026-09-02T01:00:00+01:00")
        self.assertEqual(self.latest(), expected)
        self.assertEqual(self.latest(workers=2), expected)

    def test_no_completed_run_does_not_choose_an_intermediate_checkpoint(self) -> None:
        (self.root / "run-in-progress" / "checkpoint-000001").mkdir(parents=True)
        with self.assertRaisesRegex(FileNotFoundError, "No completed training run"):
            self.latest()

    def test_legacy_absolute_run_and_checkpoint_paths_are_supported(self) -> None:
        run = self.root / "run-legacy"
        checkpoint = run / "checkpoint-000001"
        checkpoint.mkdir(parents=True)
        for source in (run, checkpoint):
            with self.subTest(source=source):
                self.assertEqual(
                    selection.probe_source_path(
                        source, saved_model_root=self.root, schema_version="4.0"
                    ),
                    source,
                )

    def test_unsafe_ids_relative_paths_and_external_paths_are_rejected(self) -> None:
        for value in ("", "..", "../run-other", "run-a/checkpoint-000001", "run--a", "/tmp/x"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                selection.probe_source_path(
                    value, saved_model_root=self.root, schema_version="4.0"
                )

    def test_symlinked_runs_and_completion_markers_are_rejected(self) -> None:
        run = self.completed_run("run-original")
        alias = self.root / "run-alias"
        alias.symlink_to(run, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlinks"):
            self.latest()
        with self.assertRaisesRegex(ValueError, "symlinks"):
            selection.probe_source_path(
                alias.name, saved_model_root=self.root, schema_version="4.0"
            )

    def test_symlinked_completion_directory_is_rejected_without_reading_target(self) -> None:
        run = self.root / "run-symlink"
        run.mkdir()
        (run / "completion-result").symlink_to(self.root / "outside", target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlinks"):
            self.latest()

    def test_malformed_marker_fails_instead_of_falling_back(self) -> None:
        self.completed_run("run-valid")
        broken = self.completed_run("run-broken") / selection.COMPLETION_METADATA
        for content in ("{", "[]", "null", "\ufffd"):
            broken.write_text(content, encoding="utf-8")
            with self.subTest(content=content), self.assertRaises(ValueError):
                self.latest()

    def test_completion_identity_schema_stop_reason_and_timestamp_are_checked(self) -> None:
        marker = self.completed_run("run-invalid") / selection.COMPLETION_METADATA
        original = json.loads(marker.read_text(encoding="utf-8"))
        changes = (
            ("run_id", "another-run"),
            ("run_key", "another-run"),
            ("schema_version", "old"),
            ("kind", "checkpoint"),
            ("stop_reason", "timed_out"),
            ("stop_reason", {}),
            ("created_at", "not-a-date"),
            ("created_at", "2026-09-01T00:00:00"),
            ("created_at", None),
        )
        for key, value in changes:
            marker.write_text(json.dumps({**original, key: value}), encoding="utf-8")
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.latest()

    def test_metadata_read_size_is_bounded(self) -> None:
        self.completed_run("run-too-large")
        with (
            patch.object(selection, "MAX_COMPLETION_METADATA_BYTES", 32),
            self.assertRaisesRegex(ValueError, "bounded read limit"),
        ):
            self.latest()

    def test_worker_plan_respects_cpu_memory_request_and_storage_limits(self) -> None:
        self.assertEqual(selection._selection_worker_count(None), 4)
        self.assertEqual(selection._selection_worker_count(2), 2)
        low_memory = AvailableMemoryEstimate(512 * 1024**2, "fixture-cgroup", ())
        with (
            patch.object(selection, "detect_visible_cpu_count", return_value=64),
            patch.object(selection, "detect_available_memory", return_value=low_memory),
            self.assertLogs(selection.LOGGER, level="WARNING") as logs,
        ):
            self.assertEqual(selection._selection_worker_count(8), 1)
        self.assertIn("memory limit=1", logs.output[0])
        with patch.object(selection, "detect_visible_cpu_count", return_value=64):
            self.assertEqual(
                selection._selection_worker_count(None), selection.MAX_SELECTION_WORKERS
            )

    def test_unsafe_worker_count_or_insufficient_memory_fails_before_scanning(self) -> None:
        for value in (0, -1, 9, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.latest(workers=value)
        with (
            patch.object(
                selection,
                "detect_available_memory",
                return_value=AvailableMemoryEstimate(128 * 1024**2, "fixture", ()),
            ),
            self.assertRaises(MemoryError),
        ):
            self.latest()

    def test_scan_keeps_only_one_bounded_batch_in_flight(self) -> None:
        for index in range(23):
            self.completed_run(f"run-{index:02d}")
        outstanding = 0
        peak = 0

        class ObservedFuture:
            def __init__(self, future: Future) -> None:
                self.future = future

            def result(self, *, timeout: float) -> object:
                nonlocal outstanding
                try:
                    return self.future.result(timeout=timeout)
                finally:
                    outstanding -= 1

        class ObservedExecutor(ThreadPoolExecutor):
            def submit(self, *args: object, **kwargs: object) -> ObservedFuture:
                nonlocal outstanding, peak
                outstanding += 1
                peak = max(peak, outstanding)
                return ObservedFuture(super().submit(*args, **kwargs))

        with patch.object(selection, "ThreadPoolExecutor", ObservedExecutor):
            self.assertEqual(self.latest(workers=2).name, "run-22")
        self.assertEqual(peak, 2)
        self.assertEqual(outstanding, 0)

    def test_worker_io_failures_and_timeouts_are_not_silenced(self) -> None:
        self.completed_run("run-error")
        for error in (OSError("storage read failed"), TimeoutError("metadata read timed out")):
            with (
                self.subTest(error=error),
                patch.object(selection, "_completion_time", side_effect=error),
                self.assertRaises(type(error)),
            ):
                self.latest()

    def test_discovery_does_not_modify_artifacts_or_open_model_weights(self) -> None:
        run = self.completed_run("run-read-only")
        (run / "adapter.safetensors").write_bytes(b"not-a-model")
        before = {path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        self.assertEqual(self.latest(), run)
        after = {path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        self.assertEqual(before, after)


class ProbeShellTests(unittest.TestCase):
    def setUp(self) -> None:
        self.command = [
            "bash",
            str(Path(__file__).resolve().parents[1] / "scripts/runpod_workflow.sh"),
            "probe-scales",
        ]
        self.environment = {**os.environ, "RUNPOD_TEST_MODE": "1"}
        self.environment.pop("RUNPOD_POD_ID", None)

    def test_help_describes_optional_run_id_and_existing_pod_requirement(self) -> None:
        result = subprocess.run(
            [*self.command, "--help"],
            env=self.environment,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("[--checkpoint RUN_ID]", result.stdout)
        self.assertIn("latest completed training run", result.stdout)
        self.assertIn("does not create a Pod", result.stdout)
        self.assertIn("local guard terminates the Pod", result.stdout)
        self.assertIn("SSH may disconnect", result.stdout)

    def test_local_execution_is_blocked_with_or_without_a_selector(self) -> None:
        for arguments in ([], ["--checkpoint", "run-test"]):
            result = subprocess.run(
                [*self.command, *arguments],
                env=self.environment,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("no local model execution", result.stderr)

    def test_optional_selector_reaches_mount_guard_without_loading_python(self) -> None:
        with tempfile.TemporaryDirectory(prefix="probe-shell-") as directory:
            root = str(Path(directory).resolve())
            environment = {
                **self.environment,
                "RUNPOD_POD_ID": "probe-shell-fixture",
                "RUNPOD_EXPECTED_VOLUME_ID": "",
                "RUNPOD_VOLUME_ID": "",
                "NETWORK_VOLUME_ROOT": root,
                "RUNPOD_VOLUME_ROOT": root,
                "PROJECT_ROOT": root + "/project",
                "CACHE_ROOT": root + "/cache",
                "SAVED_MODEL_ROOT": root + "/savedModel",
            }
            cases = (
                [],
                ["--checkpoint", "run-test"],
                ["--batch-size", "8", "--checkpoint=run-test"],
                ["--checkpoint", root + "/savedModel/run-test/checkpoint-000001"],
            )
            for arguments in cases:
                with self.subTest(arguments=arguments):
                    result = subprocess.run(
                        [*self.command, *arguments],
                        env=environment,
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    self.assertEqual(result.returncode, 2, result.stderr)
                    self.assertIn("RUNPOD_EXPECTED_VOLUME_ID is missing", result.stderr)
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
