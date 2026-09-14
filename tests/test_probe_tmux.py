"""Dependency-free shell integration tests; no real tmux, model, or cloud calls."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ProbeHarness:
    """Execute the real launcher and generated runner with bounded local stubs."""

    def __init__(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="probe-tmux-")
        self.root = Path(self.temporary.name).resolve()
        self.volume = self.root / "volume"
        self.project = self.volume / "stock_forecasting"
        self.scripts = self.project / "scripts"
        self.bin = self.root / "bin"
        (self.scripts / "lib").mkdir(parents=True)
        self.bin.mkdir()
        for name in (
            "runpod_tmux_launch.sh",
            "runpod_workflow.sh",
            "lib/runpod_paths.sh",
            "lib/runpod_cli.sh",
        ):
            shutil.copyfile(ROOT / "scripts" / name, self.scripts / name)
        self.environment = {
            "PATH": f"{self.bin}{os.pathsep}{os.defpath}",
            "LC_ALL": "C",
            "HARNESS_ROOT": str(self.root),
            "RUNPOD_SSH_ENV_IMPORTED": "1",
            "RUNPOD_POD_ID": "probe-fixture",
            "NETWORK_VOLUME_ROOT": str(self.volume),
            "RUNPOD_VOLUME_ROOT": str(self.volume),
            "PROJECT_ROOT": str(self.project),
            "LOG_ROOT": str(self.volume / "logs"),
            "RUNPOD_PYTHON_BIN": str(self.bin / "readiness-stub"),
            "MAX_RUNTIME_SECONDS": "120",
            "RUNPOD_CPU_MAX_RUNTIME_SECONDS": "120",
            "RUNPOD_CPU_FINALIZE_MAX_RUNTIME_SECONDS": "120",
            "RUNPOD_VALIDATION_MAX_RUNTIME_SECONDS": "120",
        }
        self.write_script(
            self.scripts / "verify_runpod_mounted_readiness.sh",
            r"""
            printf '%s\n' "$*" >> "${HARNESS_ROOT}/mount-checks"
            exit "${HARNESS_MOUNT_EXIT:-0}"
            """,
        )
        self.write_script(self.scripts / "ensure_runpod_tmux.sh", "exit 0\n")
        self.write_script(
            self.bin / "tmux",
            r"""
            printf '%s\n' "$*" >> "${HARNESS_ROOT}/tmux-calls"
            [[ "$1" == -L ]] || exit 91
            socket="$2"
            shift 2
            case "$1" in
                has-session)
                    [[ "${HARNESS_DUPLICATE:-0}" == 1 \
                        || -f "${HARNESS_ROOT}/session-${socket}" ]]
                    ;;
                new-session)
                    [[ "$2" == -d ]] || exit 92
                    : > "${HARNESS_ROOT}/session-${socket}"
                    ;;
                set-option|send-keys) exit 0 ;;
                *) exit 93 ;;
            esac
            """,
        )
        self.write_script(
            self.bin / "timeout",
            r"""
            printf '%s\n' "$1" "$2" "$3" > "${HARNESS_ROOT}/timeout-options"
            [[ "$1" == --signal=TERM && "$2" == --kill-after=60 ]] || exit 94
            shift 3
            if [[ "${HARNESS_TIMEOUT:-0}" == 1 ]]; then exit 124; fi
            exec "$@"
            """,
        )
        self.write_script(
            self.bin / "flock",
            r"""
            [[ "$1" == -n && "$2" == 9 ]] || exit 95
            [[ -e /dev/fd/9 ]] || exit 96
            printf 'acquire\n' >> "${HARNESS_ROOT}/lease-calls"
            exit "${HARNESS_LEASE_EXIT:-0}"
            """,
        )
        self.write_script(
            self.bin / "readiness-stub",
            r"""
            printf '%s\n' "$*" >> "${HARNESS_ROOT}/lifecycle-calls"
            if [[ "$2" == training-phase-completed ]]; then exit 1; fi
            exit 0
            """,
        )
        for name in (
            "runpod_probe_scales.sh",
            "runpod_cpu_prepare.sh",
            "runpod_cpu_finalize.sh",
            "runpod_entrypoint.sh",
            "runpod_validation.sh",
        ):
            self.write_script(
                self.scripts / name,
                r"""
                : > "${HARNESS_ROOT}/worker-args"
                if [[ $# -gt 0 ]]; then
                    printf '%s\0' "$@" > "${HARNESS_ROOT}/worker-args"
                fi
                printf '%s\n' "${RUNPOD_ROLE}" "${RUNPOD_SCALE_PROBE_TMUX_WORKER:-0}" \
                    "${RUNPOD_GPU_WORKFLOW_LEASE_HELD:-0}" > "${HARNESS_ROOT}/worker-env"
                if [[ "${RUNPOD_ROLE}" == gpu-probe ]]; then
                    [[ -e /dev/fd/9 ]] || exit 97
                fi
                printf 'fixture worker finished\n'
                exit "${HARNESS_JOB_EXIT:-0}"
                """,
            )
        self.write_script(
            self.scripts / "runpod_self_terminate.sh",
            r"""
            # Verify terminal status and lease ownership before the shutdown request.
            status="${RUNPOD_SHUTDOWN_DIR%/pod-shutdown}/status.json"
            [[ -s "${status}" ]] || exit 98
            if [[ "${RUNPOD_ROLE}" == gpu-probe ]]; then
                [[ "${RUNPOD_GPU_WORKFLOW_LEASE_HELD:-0}" == 1 \
                    && -e /dev/fd/9 ]] || exit 99
            fi
            cp "${status}" "${HARNESS_ROOT}/status-at-shutdown.json"
            printf '%s\n' "${RUNPOD_ROLE}" > "${HARNESS_ROOT}/shutdown-called"
            mkdir -p "${RUNPOD_SHUTDOWN_DIR}"
            printf '{"fixture":true}\n' > "${RUNPOD_SHUTDOWN_MARKER}"
            exit "${HARNESS_SHUTDOWN_EXIT:-0}"
            """,
        )

    @staticmethod
    def write_script(path: Path, body: str) -> None:
        path.write_text("#!/usr/bin/env bash\nset -eu\n" + textwrap.dedent(body), encoding="utf-8")
        path.chmod(0o700)

    def command(self, name: str, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(self.scripts / name), *arguments],
            env=self.environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )

    def runners(self) -> list[Path]:
        return sorted((self.volume / "logs").rglob("runner.sh"))

    def run_worker(self) -> subprocess.CompletedProcess[str]:
        runners = self.runners()
        if len(runners) != 1:
            raise AssertionError(f"Expected one generated runner, found {len(runners)}")
        # Execute the generated script, not a second model or a real tmux server.
        return self.command(str(runners[0]))

    def read(self, name: str) -> str:
        return (self.root / name).read_text(encoding="utf-8")

    def status(self) -> dict:
        return json.loads((self.runners()[0].parent / "status.json").read_text(encoding="utf-8"))


class ProbeTmuxTests(unittest.TestCase):
    def harness(self) -> ProbeHarness:
        harness = ProbeHarness()
        self.addCleanup(harness.temporary.cleanup)
        return harness

    def assert_no_training_markers(self, harness: ProbeHarness) -> None:
        self.assertFalse((harness.root / "lifecycle-calls").exists())
        self.assertEqual(list((harness.volume / "lifecycle").rglob("*.json")), [])

    def test_legacy_workflow_alias_delegates_once_with_unchanged_arguments(self) -> None:
        for arguments in ((), ("--checkpoint", "run-selected", "--batch-size", "8")):
            with self.subTest(arguments=arguments):
                harness = self.harness()
                shutil.copyfile(
                    ROOT / "scripts/runpod_probe_scales.sh",
                    harness.scripts / "runpod_probe_scales.sh",
                )
                harness.write_script(
                    harness.scripts / "runpod_tmux_launch.sh",
                    "printf '%s\\0' \"$@\" > \"${HARNESS_ROOT}/launcher-args\"\n",
                )
                result = harness.command("runpod_workflow.sh", "probe-scales", *arguments)
                self.assertEqual(result.returncode, 0, result.stderr)
                actual = (harness.root / "launcher-args").read_bytes().split(b"\0")[:-1]
                self.assertEqual(actual, [item.encode() for item in ("probe-scales", *arguments)])
                self.assertFalse((harness.root / "lease-calls").exists())
                self.assertFalse((harness.root / "shutdown-called").exists())

    def test_launcher_help_needs_no_pod_tmux_or_python(self) -> None:
        harness = self.harness()
        shutil.copyfile(
            ROOT / "scripts/runpod_probe_scales.sh", harness.scripts / "runpod_probe_scales.sh"
        )
        harness.environment.pop("RUNPOD_POD_ID")
        harness.environment.pop("RUNPOD_SSH_ENV_IMPORTED")
        result = harness.command("runpod_tmux_launch.sh", "probe-scales", "--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("automatically terminates the Pod", result.stdout)
        self.assertIn("bash scripts/runpod_tmux_launch.sh probe-scales", result.stdout)
        self.assertNotIn("runpod_workflow.sh probe-scales", result.stdout)
        legacy_help = harness.command("runpod_workflow.sh", "probe-scales", "--help")
        self.assertEqual(legacy_help.returncode, 0, legacy_help.stderr)
        self.assertEqual(legacy_help.stdout, result.stdout)
        workflow_help = harness.command("runpod_workflow.sh", "--help")
        self.assertEqual(workflow_help.returncode, 0, workflow_help.stderr)
        self.assertNotIn("probe-scales", workflow_help.stdout + workflow_help.stderr)
        self.assertFalse((harness.root / "mount-checks").exists())
        self.assertFalse((harness.root / "tmux-calls").exists())

    def test_readme_only_documents_tmux_for_remote_probe_execution(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        for heading in (
            "### 歷史尺度表徵診斷",
            "### Historical scale representation diagnostics",
        ):
            with self.subTest(heading=heading):
                section = readme.split(heading, 1)[1].split("\n### ", 1)[0]
                self.assertIn("bash scripts/runpod_workflow.sh train --maxRuntime 2h", section)
                self.assertIn("bash scripts/runpod_tmux_launch.sh probe-scales\n", section)
                self.assertIn("bash scripts/runpod_tmux_launch.sh probe-scales --help", section)
                self.assertNotIn("runpod_workflow.sh probe-scales", section)
        diagnostic_source = (ROOT / "src/stock_forecasting/cli/probe_scales.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("Use bash scripts/runpod_tmux_launch.sh probe-scales", diagnostic_source)
        self.assertNotIn("runpod_workflow.sh probe-scales", diagnostic_source)

    def test_success_persists_status_then_terminates_while_owning_lease(self) -> None:
        harness = self.harness()
        # An inherited hint must not bypass real lease acquisition in the runner.
        harness.environment["RUNPOD_GPU_WORKFLOW_LEASE_HELD"] = "1"
        arguments = ("--checkpoint", "run-selected", "--batch-size", "8")
        result = harness.command("runpod_tmux_launch.sh", "probe-scales", *arguments)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Detached tmux session started: fin-ts-probe-scales", result.stdout)
        self.assertIn("new-session -d -s fin-ts-probe-scales", harness.read("tmux-calls"))
        self.assertIn("remain-on-exit on", harness.read("tmux-calls"))
        self.assertFalse((harness.root / "worker-args").exists())
        self.assertFalse((harness.root / "lease-calls").exists())
        self.assertFalse((harness.root / "shutdown-called").exists())
        self.assertFalse((harness.runners()[0].parent / "status.json").exists())
        runner = harness.run_worker()
        self.assertEqual(runner.returncode, 0, runner.stdout + runner.stderr)
        self.assertEqual(harness.read("lease-calls"), "acquire\n")
        self.assertEqual(harness.read("worker-env").splitlines(), ["gpu-probe", "1", "1"])
        self.assertEqual(harness.status()["state"], "succeeded")
        self.assertEqual(harness.status()["exit_code"], 0)
        self.assertEqual(harness.status()["finalization_exit_code"], 0)
        self.assertEqual(json.loads(harness.read("status-at-shutdown.json")), harness.status())
        self.assertEqual(harness.read("shutdown-called"), "gpu-probe\n")
        self.assertEqual(harness.read("timeout-options").splitlines()[-1], "120s")
        self.assertIn("fixture worker finished", Path(harness.status()["log_path"]).read_text())
        self.assertTrue((harness.runners()[0].parent / "pod-shutdown/shutdown.json").is_file())
        self.assert_no_training_markers(harness)

    def test_probe_arguments_are_quoted_not_executed_as_shell_code(self) -> None:
        harness = self.harness()
        sentinel = harness.root / "injected-command"
        arguments = ("--seed", f"x y; touch {sentinel}; $(touch {sentinel})")
        result = harness.command("runpod_tmux_launch.sh", "probe-scales", *arguments)
        self.assertEqual(result.returncode, 0, result.stderr)
        runner = harness.run_worker()
        self.assertEqual(runner.returncode, 0, runner.stderr)
        self.assertEqual(
            (harness.root / "worker-args").read_bytes().split(b"\0")[:-1],
            [item.encode() for item in arguments],
        )
        self.assertFalse(sentinel.exists())

    def test_failure_and_timeout_publish_terminal_status_before_termination(self) -> None:
        for environment, state, exit_code in (
            ({"HARNESS_JOB_EXIT": "7"}, "failed", 7),
            ({"HARNESS_TIMEOUT": "1"}, "timed_out", 124),
        ):
            with self.subTest(state=state):
                harness = self.harness()
                harness.environment.update(environment)
                result = harness.command("runpod_tmux_launch.sh", "probe-scales")
                self.assertEqual(result.returncode, 0, result.stderr)
                runner = harness.run_worker()
                self.assertEqual(runner.returncode, exit_code, runner.stdout + runner.stderr)
                self.assertEqual(harness.status()["state"], state)
                self.assertEqual(harness.status()["exit_code"], exit_code)
                self.assertEqual(
                    json.loads(harness.read("status-at-shutdown.json")), harness.status()
                )
                self.assertTrue((harness.root / "shutdown-called").is_file())
                self.assert_no_training_markers(harness)

    def test_contended_lease_preserves_pod_without_running_probe(self) -> None:
        harness = self.harness()
        harness.environment.update(
            {"HARNESS_LEASE_EXIT": "1", "RUNPOD_GPU_WORKFLOW_LEASE_HELD": "1"}
        )
        result = harness.command("runpod_tmux_launch.sh", "probe-scales")
        self.assertEqual(result.returncode, 0, result.stderr)
        runner = harness.run_worker()
        self.assertEqual(runner.returncode, 75, runner.stdout + runner.stderr)
        self.assertEqual(harness.status()["state"], "failed")
        self.assertEqual(harness.status()["exit_code"], 75)
        self.assertIn("preserving the Pod", runner.stdout)
        self.assertFalse((harness.root / "worker-args").exists())
        self.assertFalse((harness.root / "timeout-options").exists())
        self.assertFalse((harness.root / "shutdown-called").exists())
        self.assert_no_training_markers(harness)

    def test_duplicate_session_or_rejected_preflight_never_terminates_pod(self) -> None:
        for environment, exit_code in (
            ({"HARNESS_DUPLICATE": "1"}, 3),
            ({"HARNESS_MOUNT_EXIT": "2"}, 2),
            ({"MAX_RUNTIME_SECONDS": "invalid"}, 2),
        ):
            with self.subTest(environment=environment):
                harness = self.harness()
                harness.environment.update(environment)
                # This records even an early call, unlike the normal shutdown stub.
                harness.write_script(
                    harness.scripts / "runpod_self_terminate.sh",
                    ': > "${HARNESS_ROOT}/shutdown-called"\n',
                )
                result = harness.command("runpod_tmux_launch.sh", "probe-scales")
                self.assertEqual(result.returncode, exit_code, result.stdout + result.stderr)
                self.assertFalse((harness.root / "shutdown-called").exists())
                self.assertFalse((harness.root / "worker-args").exists())
                self.assertEqual(harness.runners(), [])
                self.assert_no_training_markers(harness)

    def test_shutdown_failure_is_not_confused_with_failed_diagnostic(self) -> None:
        harness = self.harness()
        harness.environment["HARNESS_SHUTDOWN_EXIT"] = "4"
        result = harness.command("runpod_tmux_launch.sh", "probe-scales")
        self.assertEqual(result.returncode, 0, result.stderr)
        runner = harness.run_worker()
        self.assertEqual(runner.returncode, 0, runner.stdout + runner.stderr)
        self.assertEqual(harness.status()["state"], "succeeded")
        self.assertIn(
            "Pod self-termination failed; external lifecycle guard remains armed", runner.stdout
        )
        self.assertTrue((harness.root / "shutdown-called").is_file())
        self.assert_no_training_markers(harness)

    def test_existing_workflows_keep_timeout_roles_and_lifecycle_finalization(self) -> None:
        for workflow, role, duration in (
            ("cpu-prepare", "cpu-prep", "120s"),
            ("cpu-finalize", "cpu-prep", "120s"),
            ("stage1-train", "gpu-train", "420s"),
            ("stage1-validate", "gpu-validation", "420s"),
        ):
            with self.subTest(workflow=workflow):
                harness = self.harness()
                harness.environment.update(
                    {"WANDB_RUN_ID": "run-legacy", "HARNESS_JOB_EXIT": "7"}
                )
                result = harness.command("runpod_tmux_launch.sh", workflow)
                self.assertEqual(result.returncode, 0, result.stderr)
                runner = harness.run_worker()
                self.assertEqual(runner.returncode, 7, runner.stdout + runner.stderr)
                self.assertEqual(harness.read("worker-env").splitlines(), [role, "0", "0"])
                self.assertEqual(harness.read("timeout-options").splitlines()[-1], duration)
                self.assertEqual(harness.read("shutdown-called"), f"{role}\n")
                self.assertEqual(harness.status()["state"], "failed")
                self.assertIn("write-state", harness.read("lifecycle-calls"))
                self.assertFalse((harness.root / "lease-calls").exists())

    def test_existing_workflows_do_not_accept_probe_arguments(self) -> None:
        harness = self.harness()
        harness.environment["RUNPOD_TEST_MODE"] = "1"
        for workflow in ("cpu-prepare", "cpu-finalize", "stage1-train", "stage1-validate"):
            with self.subTest(workflow=workflow):
                result = harness.command("runpod_tmux_launch.sh", workflow, "--batch-size", "8")
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("Usage:", result.stderr)
        result = harness.command("runpod_tmux_launch.sh")
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("Usage:", result.stderr)
        self.assertEqual(harness.runners(), [])


if __name__ == "__main__":
    unittest.main()
