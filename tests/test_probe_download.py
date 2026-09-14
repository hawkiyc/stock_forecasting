"""Exercise the actual local download entrypoint with an isolated fake S3 service."""

from __future__ import annotations

import hashlib
import json
import os
import runpy
import shlex
import shutil
import subprocess
import sys
import tempfile
import textwrap
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
HELPER = runpy.run_path(str(ROOT / "scripts/download_runpod_probes.py"))


class DownloadFixture:
    def __init__(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="probe-download-test-")
        self.root = Path(self.temporary.name).resolve()
        self.cloud = self.root / "cloud"
        self.cloud.mkdir()
        self.bin = self.root / "bin"
        self.bin.mkdir()
        for relative in (
            "scripts/runpod_workflow.sh", "scripts/download_runpod_probes.sh",
            "scripts/download_runpod_probes.py", "scripts/runpod_s3_project.sh",
            "scripts/lib/runpod_project_env.sh", "scripts/lib/runpod_cli.sh",
            "src/stock_forecasting/runtime_resources.py",
        ):
            target = self.root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, target)
        (self.root / ".env").write_text(
            "RUNPOD_NETWORK_VOLUME_ID=fixture-volume\n"
            "RUNPOD_S3_REGION=EU-RO-1\n"
            "RUNPOD_S3_ACCESS_KEY_ID=fixture-access-only\n"
            "RUNPOD_S3_SECRET_ACCESS_KEY=fixture-secret-only\n",
            encoding="utf-8",
        )
        (self.root / ".env").chmod(0o600)
        self.write_executable(
            self.bin / "python3", "#!/bin/sh\nexec " + shlex.quote(sys.executable) + ' "$@"\n'
        )
        self.write_executable(
            self.bin / "aws",
            "#!" + sys.executable + "\n" + textwrap.dedent('''
            import json
            import os
            import sys
            from pathlib import Path

            root = Path(__file__).resolve().parents[1]
            arguments = sys.argv[1:]
            assert os.environ['AWS_ACCESS_KEY_ID'] == 'fixture-access-only'
            assert os.environ['AWS_SECRET_ACCESS_KEY'] == 'fixture-secret-only'
            assert 'RUNPOD_API_KEY' not in os.environ
            with (root / 'requests.jsonl').open('a') as log:
                log.write(json.dumps(arguments) + '\\n')
            if arguments[:2] == ['s3api', 'list-objects-v2']:
                assert arguments[arguments.index('--bucket') + 1] == 'fixture-volume'
                assert '--no-paginate' in arguments
                prefix = arguments[arguments.index('--prefix') + 1]
                keys = sorted(str(p.relative_to(root / 'cloud'))
                              for p in (root / 'cloud').rglob('*') if p.is_file())
                keys = [key for key in keys if key.startswith(prefix)]
                offset = int(arguments[arguments.index('--continuation-token') + 1]) \\
                         if '--continuation-token' in arguments else 0
                page = keys[offset:offset + 5]
                truncated = offset + 5 < len(keys)
                result = {'Contents': [{'Key': key} for key in page],
                          'IsTruncated': truncated}
                if truncated:
                    result['NextContinuationToken'] = str(offset + 5)
                print(json.dumps(result))
            elif arguments[:2] == ['s3', 'cp']:
                prefix = 's3://fixture-volume/'
                assert arguments[2].startswith(prefix)
                assert arguments[3] == '-'
                key = arguments[2][len(prefix):]
                source = root / 'cloud' / key
                if not source.is_file():
                    raise SystemExit(9)
                with source.open('rb') as stream:
                    for block in iter(lambda: stream.read(65536), b''):
                        sys.stdout.buffer.write(block)
            else:
                raise SystemExit(90)
            '''),
        )
        self.environment = {
            "PATH": str(self.bin) + os.pathsep + os.defpath,
            "HOME": str(self.root), "LC_ALL": "C", "PYTHONDONTWRITEBYTECODE": "1",
            "RUNPOD_PROBE_DOWNLOAD_WORKERS": "2",
            "RUNPOD_NETWORK_VOLUME_ID": "incorrect-ambient-volume",
            "RUNPOD_API_KEY": "must-not-reach-aws",
        }

    @staticmethod
    def write_executable(path, content):
        path.write_text(content, encoding="utf-8")
        path.chmod(0o700)

    def probe(self, run, probe, stamp, *, state="complete", checkpoint="checkpoint-000001"):
        folder = self.cloud / HELPER["REMOTE_ROOT"] / run / checkpoint / probe
        folder.mkdir(parents=True)
        checkpoint_path = f"/runpod-volume/savedModel/{run}/{checkpoint}"
        (folder / "status.json").write_text(json.dumps({
            "state": state, "checkpoint": checkpoint_path,
        }))
        if state != "complete":
            return folder
        for name in ("probe.log", "summary.md", *HELPER["HASHED_ARTIFACTS"]):
            (folder / name).write_bytes((name + ":" + probe).encode())
        (folder / "report.json").write_text(json.dumps({
            "kind": "historical-scale-representation-probe", "schema_version": "1.0",
            "created_at": stamp, "checkpoint": {"path": checkpoint_path},
            "artifacts_sha256": {
                name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
                for name in HELPER["HASHED_ARTIFACTS"]
            },
        }))
        return folder

    def command(self, *arguments):
        return subprocess.run(
            ["bash", str(self.root / "scripts/runpod_workflow.sh"), "download-probes", *arguments],
            cwd=self.root, env=self.environment, capture_output=True, text=True, timeout=60,
        )

    def target(self, source):
        return self.root / HELPER["LOCAL_ROOT"] / source.relative_to(
            self.cloud / HELPER["REMOTE_ROOT"]
        )


class ProbeDownloadTests(unittest.TestCase):
    def fixture(self):
        fixture = DownloadFixture()
        self.addCleanup(fixture.temporary.cleanup)
        return fixture

    def test_help_and_invalid_arguments_need_neither_credentials_nor_network(self):
        fixture = self.fixture()
        (fixture.root / ".env").rename(fixture.root / ".env-unused")
        help_result = fixture.command("--help")
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn("download-probes [PROBE_RUN_ID]", help_result.stdout)
        for arguments in (
            ("../run-a",), ("/tmp/run-a",), ("s3://bucket/key",), ("--resume",),
            ("--checkpoint", "checkpoint-000001"), ("run-a", "extra"), ("",),
        ):
            with self.subTest(arguments=arguments):
                self.assertNotEqual(fixture.command(*arguments).returncode, 0)
        self.assertFalse((fixture.root / "requests.jsonl").exists())

    def test_omitted_run_selects_latest_complete_report_across_paginated_volume(self):
        fixture = self.fixture()
        old = fixture.probe("run-z", "probe-20260914T200000Z-aaaaaaaa", "2026-09-14T08:00:00Z")
        newest = fixture.probe("run-a", "probe-20260914T070000Z-bbbbbbbb", "2026-09-14T10:00:00Z")
        fixture.probe("run-new", "probe-20260914T210000Z-cccccccc", "unused", state="running")
        fixture.probe("run-new", "probe-20260914T220000Z-dddddddd", "unused", state="failed")
        result = fixture.command()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("run_id=run-a state=complete", result.stdout)
        self.assertEqual(sorted(p.name for p in fixture.target(newest).iterdir()),
                         sorted(HELPER["ARTIFACTS"]))
        self.assertFalse(fixture.target(old).exists())
        requests = [
            json.loads(line) for line in (fixture.root / "requests.jsonl").read_text().splitlines()
        ]
        self.assertTrue(any("--continuation-token" in request for request in requests))
        self.assertTrue(all("savedModel" not in str(request) for request in requests))

    def test_explicit_model_run_resolves_latest_checkpoint_and_probe_automatically(self):
        fixture = self.fixture()
        old = fixture.probe("run-a", "probe-20260914T070000Z-aaaaaaaa", "2026-09-14T08:00:00Z",
                            checkpoint="checkpoint-999999")
        chosen = fixture.probe("run-a", "probe-20260914T080000Z-bbbbbbbb", "2026-09-14T09:00:00Z")
        other = fixture.probe("run-b", "probe-20260914T090000Z-cccccccc", "2026-09-14T10:00:00Z")
        result = fixture.command("run-a")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(fixture.target(chosen).is_dir())
        self.assertFalse(fixture.target(old).exists())
        self.assertFalse(fixture.target(other).exists())
        self.assertNotIn("run-b", (fixture.root / "requests.jsonl").read_text())

    def test_no_completed_result_is_an_explicit_failure(self):
        fixture = self.fixture()
        fixture.probe("run-a", "probe-20260914T070000Z-aaaaaaaa", "unused", state="failed")
        for arguments in ((), ("run-missing",)):
            with self.subTest(arguments=arguments):
                result = fixture.command(*arguments)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("No completed scale diagnostic", result.stderr)
                self.assertNotIn("Downloaded scale diagnostic", result.stdout)

    def test_hash_failure_does_not_publish_and_same_command_repairs_without_resume(self):
        fixture = self.fixture()
        source = fixture.probe("run-a", "probe-20260914T070000Z-aaaaaaaa", "2026-09-14T08:00:00Z")
        artifact = source / "samples.jsonl"
        original = artifact.read_bytes()
        artifact.write_bytes(b"corrupt")
        failed = fixture.command()
        self.assertNotEqual(failed.returncode, 0)
        self.assertIn("SHA-256 mismatch", failed.stderr)
        self.assertFalse(fixture.target(source).exists())
        artifact.write_bytes(original)
        completed = fixture.command()
        self.assertEqual(completed.returncode, 0, completed.stderr)
        target = fixture.target(source)
        (target / "personal-notes.md").write_text("preserve this")
        (target / "samples.jsonl").write_bytes(b"interrupted local copy")
        repeated = fixture.command()
        self.assertEqual(repeated.returncode, 0, repeated.stderr)
        self.assertEqual((target / "samples.jsonl").read_bytes(), original)
        self.assertEqual((target / "personal-notes.md").read_text(), "preserve this")

    def test_missing_artifact_fails_without_falling_back_to_an_older_result(self):
        fixture = self.fixture()
        old = fixture.probe("run-a", "probe-20260914T070000Z-aaaaaaaa", "2026-09-14T08:00:00Z")
        newer = fixture.probe("run-a", "probe-20260914T080000Z-bbbbbbbb", "2026-09-14T09:00:00Z")
        (newer / "summary.md").rename(newer / "missing-summary.fixture")
        result = fixture.command()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("RunPod S3 read failed", result.stderr)
        self.assertFalse(fixture.target(newer).exists())
        self.assertFalse(fixture.target(old).exists())

    def test_report_identity_and_timestamp_are_not_inferred_from_directory_names(self):
        fixture = self.fixture()
        source = fixture.probe("run-a", "probe-20260914T070000Z-aaaaaaaa", "2026-09-14T08:00:00Z")
        report = json.loads((source / "report.json").read_text())
        for field, value, message in (
            ("checkpoint", {"path": "/runpod-volume/savedModel/run-other/checkpoint-000001"},
             "identity"),
            ("created_at", "2026-09-14T08:00:00", "timezone"),
            ("artifacts_sha256", {"../escape": "a" * 64}, "artifact manifest"),
        ):
            with self.subTest(field=field):
                (source / "report.json").write_text(json.dumps(dict(report, **{field: value})))
                result = fixture.command()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)
                self.assertFalse(fixture.target(source).exists())

    def test_local_symlinks_are_rejected_without_writing_through_them(self):
        fixture = self.fixture()
        fixture.probe("run-a", "probe-20260914T070000Z-aaaaaaaa", "2026-09-14T08:00:00Z")
        outside = fixture.root / "outside"
        outside.mkdir()
        (fixture.root / "artifacts").symlink_to(outside, target_is_directory=True)
        result = fixture.command()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("canonical directory", result.stderr)
        self.assertEqual(list(outside.iterdir()), [])

    def test_worker_budget_respects_memory_cpu_service_and_requested_limit(self):
        resources = {
            "detect_visible_cpu_count": lambda: 64,
            "detect_available_memory": lambda: SimpleNamespace(
                available_bytes=64 * 1024**3, source="fixture"
            ),
        }
        with (
            patch.object(HELPER["runpy"], "run_path", return_value=resources),
            patch.dict(os.environ, {"RUNPOD_PROBE_DOWNLOAD_WORKERS": "3"}),
        ):
            self.assertEqual(HELPER["worker_count"](ROOT), 3)
            resources["detect_visible_cpu_count"] = lambda: 2
            self.assertEqual(HELPER["worker_count"](ROOT), 2)
            resources["detect_available_memory"] = lambda: SimpleNamespace(
                available_bytes=1024**3, source="fixture"
            )
            self.assertEqual(HELPER["worker_count"](ROOT), 1)
            resources["detect_available_memory"] = lambda: SimpleNamespace(
                available_bytes=1024**2, source="fixture"
            )
            with self.assertRaises(MemoryError):
                HELPER["worker_count"](ROOT)

    def test_failed_batch_waits_for_active_writers_without_submitting_more_work(self):
        started = threading.Event()
        finished = threading.Event()
        submitted = []

        def items():
            for index in range(100):
                submitted.append(index)
                yield index

        def worker(index):
            if index == 0:
                self.assertTrue(started.wait(2))
                raise RuntimeError("fixture worker failure")
            started.set()
            threading.Event().wait(0.02)
            finished.set()

        with (
            HELPER["ThreadPoolExecutor"](max_workers=2) as executor,
            self.assertRaisesRegex(RuntimeError, "fixture worker failure"),
        ):
            list(HELPER["parallel_results"](executor, worker, items(), 2))
        self.assertEqual(submitted, [0, 1])
        self.assertTrue(finished.is_set())

    def test_remote_metadata_read_is_bounded(self):
        reader = HELPER["S3Reader"](ROOT, "fixture-volume")
        with (
            patch.object(
                reader, "command", side_effect=lambda args, stream: stream.write(b"x" * 65)
            ),
            self.assertRaisesRegex(ValueError, "bounded read limit"),
        ):
            reader.json_command(["fixture"], limit=64)

    def test_transport_timeout_terminates_and_reaps_its_process_group(self):
        reader = HELPER["S3Reader"](ROOT, "fixture-volume")
        process = Mock(pid=123456)
        process.wait.side_effect = [subprocess.TimeoutExpired(["fixture"], 1), 0]
        with (
            patch.object(HELPER["subprocess"], "Popen", return_value=process),
            patch.object(HELPER["os"], "killpg") as kill_group,
        ):
            with tempfile.TemporaryFile() as output, self.assertRaises(subprocess.TimeoutExpired):
                reader.command(["fixture"], output)
            kill_group.assert_called_once_with(process.pid, HELPER["signal"].SIGKILL)
        self.assertEqual(process.wait.call_count, 2)

    def test_transport_timeout_reaps_an_already_exited_process_group(self):
        reader = HELPER["S3Reader"](ROOT, "fixture-volume")
        process = Mock(pid=123456)
        process.wait.side_effect = [subprocess.TimeoutExpired(["fixture"], 1), 0]
        with (
            patch.object(HELPER["subprocess"], "Popen", return_value=process),
            patch.object(HELPER["os"], "killpg", side_effect=ProcessLookupError) as kill_group,
        ):
            with tempfile.TemporaryFile() as output, self.assertRaises(subprocess.TimeoutExpired):
                reader.command(["fixture"], output)
            kill_group.assert_called_once_with(process.pid, HELPER["signal"].SIGKILL)
        self.assertEqual(process.wait.call_count, 2)


if __name__ == "__main__":
    unittest.main()
