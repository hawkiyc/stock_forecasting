"""Supervise one training process and persist its complete RunPod run record."""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import signal
import subprocess
import sys
import threading
import time
import traceback
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import FrameType
from typing import IO, Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from stock_forecasting.checkpointing import latest_resume_checkpoint
from stock_forecasting.run_paths import (
    canonical_network_volume_root,
    checkpoint_run_directory,
    log_run_directory,
    validate_run_environment_ids,
    validate_run_id,
)

DEFAULT_MAX_RUNTIME_SECONDS = 6 * 60 * 60
DEFAULT_SHUTDOWN_TIMEOUT_SECONDS = 30.0
SECRET_ARGUMENT_PATTERN = re.compile(r"(?i)(api[_-]?key|token|password|secret)")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _validated_volume_path(path: Path, volume_root: Path, label: str) -> Path:
    resolved = path.expanduser().resolve(strict=False)
    resolved_volume_root = volume_root.expanduser().resolve(strict=False)
    workspace = Path("/workspace")
    if _is_relative_to(resolved, workspace):
        raise ValueError(f"{label} must never use ephemeral /workspace: {resolved}")
    if not _is_relative_to(resolved, resolved_volume_root):
        raise ValueError(f"{label} must be inside NETWORK_VOLUME_ROOT: {resolved}")
    return resolved


@dataclass(frozen=True)
class VolumeLayout:
    """All paths that must survive Pod termination."""

    network_volume_root: Path
    project_root: Path
    data_root: Path
    cache_root: Path
    log_root: Path
    saved_model_root: Path
    wandb_root: Path

    @classmethod
    def from_environment(cls) -> VolumeLayout:
        network_root = canonical_network_volume_root()

        project_default = network_root / "stock_forecasting"
        project_root = _validated_volume_path(
            Path(os.environ.get("PROJECT_ROOT", str(project_default))),
            network_root,
            "PROJECT_ROOT",
        )
        layout = cls(
            network_volume_root=network_root,
            project_root=project_root,
            data_root=_validated_volume_path(
                Path(os.environ.get("DATA_ROOT", str(network_root / "data"))),
                network_root,
                "DATA_ROOT",
            ),
            cache_root=_validated_volume_path(
                Path(os.environ.get("CACHE_ROOT", str(network_root / "cache"))),
                network_root,
                "CACHE_ROOT",
            ),
            log_root=_validated_volume_path(
                Path(os.environ.get("LOG_ROOT", str(network_root / "logs"))),
                network_root,
                "LOG_ROOT",
            ),
            saved_model_root=_validated_volume_path(
                Path(os.environ.get("SAVED_MODEL_ROOT", str(network_root / "savedModel"))),
                network_root,
                "SAVED_MODEL_ROOT",
            ),
            wandb_root=_validated_volume_path(
                Path(os.environ.get("WANDB_DIR", str(network_root))),
                network_root,
                "WANDB_DIR",
            ),
        )
        canonical_roots = {
            "LOG_ROOT": (layout.log_root, network_root / "logs"),
            "SAVED_MODEL_ROOT": (layout.saved_model_root, network_root / "savedModel"),
            "WANDB_DIR": (layout.wandb_root, network_root),
        }
        for label, (actual, expected) in canonical_roots.items():
            if actual != expected:
                raise ValueError(f"{label} must equal the canonical path: {expected}")
        return layout

    def create(self) -> None:
        for path in (
            self.network_volume_root,
            self.project_root,
            self.data_root,
            self.cache_root,
            self.log_root,
            self.saved_model_root,
            self.wandb_root / "wandb",
        ):
            path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class RunResult:
    """Stable process result returned by the supervisor."""

    exit_code: int
    child_exit_code: int | None
    timed_out: bool
    signal_number: int | None
    status: str
    run_key: str
    run_directory: Path
    checkpoint_directory: Path


@dataclass(frozen=True)
class ShutdownResult:
    attempted: bool
    skipped_reason: str | None
    action: str
    success: bool | None
    status_code: int | None
    error: str | None


def _make_run_key(environment: dict[str, str]) -> str:
    candidate = validate_run_environment_ids(
        environment.get("WANDB_RUN_ID"),
        environment.get("RUNPOD_RUN_KEY"),
    )
    if not candidate:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        candidate = f"run-{timestamp}-{secrets.token_hex(4)}"
    return validate_run_id(candidate)


def _redact_command(command: Sequence[str]) -> list[str]:
    redacted: list[str] = []
    redact_next = False
    for argument in command:
        if redact_next:
            redacted.append("<redacted>")
            redact_next = False
            continue
        if argument.startswith("-") and SECRET_ARGUMENT_PATTERN.search(argument):
            if "=" in argument:
                redacted.append(f"{argument.split('=', maxsplit=1)[0]}=<redacted>")
            else:
                redacted.append(argument)
                redact_next = True
            continue
        redacted.append(argument)
    return redacted


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(path)


def _safe_environment_snapshot(environment: dict[str, str]) -> dict[str, str]:
    allowed = {
        "CACHE_ROOT",
        "CUDA_VISIBLE_DEVICES",
        "DATA_ROOT",
        "HF_HOME",
        "LOG_ROOT",
        "NETWORK_VOLUME_ROOT",
        "POETRY_CACHE_DIR",
        "PROJECT_ROOT",
        "RUNPOD_POD_ID",
        "RUNPOD_RUN_KEY",
        "RUNPOD_VOLUME_ROOT",
        "SAVED_MODEL_ROOT",
        "TMPDIR",
        "TORCH_HOME",
        "WANDB_ARTIFACT_DIR",
        "WANDB_CACHE_DIR",
        "WANDB_DIR",
        "WANDB_MODE",
        "WANDB_PROJECT",
        "WANDB_RUN_ID",
        "WANDB_RUN_NAME",
        "XDG_CACHE_HOME",
    }
    return {key: environment[key] for key in sorted(allowed) if key in environment}


def _latest_resume_checkpoint(checkpoint_root: Path, run_key: str) -> Path | None:
    """Return the most advanced complete checkpoint for this durable W&B run ID."""

    run_directory = checkpoint_run_directory(checkpoint_root, run_key)
    if not run_directory.is_dir():
        return None
    try:
        return latest_resume_checkpoint(run_directory)
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _training_completed_marker(
    layout: VolumeLayout,
    environment: dict[str, str],
    run_key: str,
) -> tuple[bool, str | None]:
    """Verify that the train-then-validate wrapper completed the training phase."""

    marker_value = environment.get("RUNPOD_TRAINING_COMPLETED_MARKER", "").strip()
    if not marker_value:
        return False, None
    marker = _validated_volume_path(
        Path(marker_value),
        layout.network_volume_root,
        "training completed marker",
    )
    expected_log_root = log_run_directory(layout.log_root, run_key).resolve(strict=False)
    if not _is_relative_to(marker, expected_log_root):
        raise ValueError("Training completed marker must be stored below LOG_ROOT/<WANDB_RUN_ID>")
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return False, str(marker)
    except (json.JSONDecodeError, OSError) as error:
        raise ValueError(f"Training completed marker is unreadable: {marker}") from error
    if not isinstance(payload, dict):
        raise ValueError("Training completed marker must contain a JSON object")
    if payload.get("training_completed") is not True:
        raise ValueError("Training completed marker does not confirm training completion")
    if payload.get("wandb_run_id") != run_key:
        raise ValueError("Training completed marker belongs to a different run")
    return True, str(marker)


class _StreamTee(threading.Thread):
    def __init__(self, source: IO[str], destination: IO[str], mirror: IO[str]) -> None:
        super().__init__(daemon=True)
        self._source = source
        self._destination = destination
        self._mirror = mirror

    def run(self) -> None:
        for line in iter(self._source.readline, ""):
            self._destination.write(line)
            self._destination.flush()
            self._mirror.write(line)
            self._mirror.flush()


def _terminate_process_group(process: subprocess.Popen[str], grace_seconds: float) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace_seconds)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    process.wait()


def _normalise_signal_exit_code(child_exit_code: int | None) -> int:
    if child_exit_code is None:
        return 70
    if child_exit_code < 0:
        return 128 + abs(child_exit_code)
    return child_exit_code


def _shutdown_action() -> str:
    # RunPod requires termination instead of stop for Pods with a network volume.
    action = os.environ.get("RUNPOD_SHUTDOWN_ACTION", "terminate").strip().lower()
    if action not in {"stop", "terminate"}:
        raise ValueError("RUNPOD_SHUTDOWN_ACTION must be 'stop' or 'terminate'")
    return action


def shutdown_runpod(
    *,
    marker_path: Path,
    dry_run: bool = False,
    timeout_seconds: float = DEFAULT_SHUTDOWN_TIMEOUT_SECONDS,
) -> ShutdownResult:
    """Stop or terminate the current Pod once, without exposing its API key."""

    action = _shutdown_action()
    if marker_path.exists():
        return ShutdownResult(False, "shutdown-already-attempted", action, None, None, None)

    api_key = os.environ.get("RUNPOD_API_KEY")
    pod_id = os.environ.get("RUNPOD_POD_ID")
    test_mode = _env_flag("RUNPOD_TEST_MODE") or "PYTEST_CURRENT_TEST" in os.environ
    if dry_run or _env_flag("RUNPOD_DRY_RUN") or test_mode:
        return ShutdownResult(False, "dry-run-or-test-mode", action, None, None, None)
    if not api_key or not pod_id:
        return ShutdownResult(False, "missing-api-key-or-pod-id", action, None, None, None)

    marker_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(
        marker_path,
        {"action": action, "attempted_at": _utc_now(), "pod_id": pod_id, "state": "started"},
    )
    api_base = os.environ.get("RUNPOD_API_BASE_URL", "https://rest.runpod.io/v1").rstrip("/")
    encoded_pod_id = quote(pod_id, safe="")
    if action == "terminate":
        method = "DELETE"
        url = f"{api_base}/pods/{encoded_pod_id}"
    else:
        method = "POST"
        url = f"{api_base}/pods/{encoded_pod_id}/stop"

    api_request = Request(
        url,
        method=method,
        headers={"Authorization": f"Bearer {api_key}"},
    )
    max_attempts = max(1, int(os.environ.get("RUNPOD_SHUTDOWN_MAX_ATTEMPTS", "5")))
    retry_seconds = max(0.1, float(os.environ.get("RUNPOD_SHUTDOWN_RETRY_SECONDS", "5")))
    result = ShutdownResult(True, None, action, False, None, "shutdown was not attempted")
    for attempt in range(1, max_attempts + 1):
        try:
            with urlopen(api_request, timeout=timeout_seconds) as response:
                status_code = response.status
                response_text = response.read(1000).decode("utf-8", errors="replace")
            success = 200 <= status_code < 300
            result = ShutdownResult(
                attempted=True,
                skipped_reason=None,
                action=action,
                success=success,
                status_code=status_code,
                error=None if success else response_text,
            )
        except HTTPError as exc:
            response_text = exc.read(1000).decode("utf-8", errors="replace")
            result = ShutdownResult(True, None, action, False, exc.code, response_text)
        except (URLError, TimeoutError, OSError) as exc:
            result = ShutdownResult(True, None, action, False, None, str(exc))
        if result.success or attempt == max_attempts:
            break
        time.sleep(min(retry_seconds * (2 ** (attempt - 1)), 60.0))

    _atomic_write_json(
        marker_path,
        {
            **asdict(result),
            "attempts": attempt,
            "completed_at": _utc_now(),
            "pod_id": pod_id,
        },
    )
    return result


def supervise(
    command: Sequence[str],
    *,
    max_runtime_seconds: int,
    termination_grace_seconds: float = 30.0,
    dry_run_shutdown: bool = False,
    auto_shutdown: bool = True,
    poll_interval_seconds: float = 0.2,
) -> RunResult:
    """Run a command with durable logging, a hard deadline, and Pod cleanup."""

    if not command:
        raise ValueError("A training command is required")
    if max_runtime_seconds <= 0:
        raise ValueError("max_runtime_seconds must be positive")

    layout = VolumeLayout.from_environment()
    layout.create()
    environment = dict(os.environ)
    run_key = _make_run_key(environment)
    # The W&B display name is intentionally independent from the durable run ID.
    environment["WANDB_RUN_ID"] = run_key
    environment["RUNPOD_RUN_KEY"] = run_key
    environment["NETWORK_VOLUME_ROOT"] = str(layout.network_volume_root)
    environment["RUNPOD_VOLUME_ROOT"] = str(layout.network_volume_root)
    environment["PROJECT_ROOT"] = str(layout.project_root)
    environment["DATA_ROOT"] = str(layout.data_root)
    environment["CACHE_ROOT"] = str(layout.cache_root)
    environment["LOG_ROOT"] = str(layout.log_root)
    environment["SAVED_MODEL_ROOT"] = str(layout.saved_model_root)
    environment["WANDB_DIR"] = str(layout.wandb_root)
    persistent_environment = {
        "HF_HOME": layout.cache_root / "huggingface",
        "POETRY_CACHE_DIR": layout.cache_root / "pypoetry",
        "TMPDIR": layout.network_volume_root / "tmp",
        "TORCH_HOME": layout.cache_root / "torch",
        "WANDB_ARTIFACT_DIR": layout.network_volume_root / "wandb-artifacts",
        "WANDB_CACHE_DIR": layout.cache_root / "wandb",
        "XDG_CACHE_HOME": layout.cache_root / "xdg",
    }
    for name, path in persistent_environment.items():
        path.mkdir(parents=True, exist_ok=True)
        environment[name] = str(path)

    checkpoint_directory = layout.saved_model_root
    checkpoint_directory.mkdir(parents=True, exist_ok=True)

    invocation_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    run_directory = log_run_directory(layout.log_root, run_key) / invocation_id
    run_directory.mkdir(parents=True, exist_ok=False)
    marker_path = Path(
        os.environ.get("RUNPOD_SHUTDOWN_MARKER", str(run_directory / "shutdown.json"))
    )
    marker_path = _validated_volume_path(marker_path, layout.network_volume_root, "shutdown marker")
    if not _is_relative_to(
        marker_path,
        log_run_directory(layout.log_root, run_key).resolve(strict=False),
    ):
        raise ValueError("Shutdown marker must be stored below LOG_ROOT/<WANDB_RUN_ID>")

    stdout_path = run_directory / "stdout.log"
    stderr_path = run_directory / "stderr.log"
    traceback_path = run_directory / "supervisor_traceback.log"
    metadata_path = run_directory / "metadata.json"
    start_monotonic = time.monotonic()
    started_at = _utc_now()
    child_exit_code: int | None = None
    timed_out = False
    received_signal: int | None = None
    process: subprocess.Popen[str] | None = None
    status = "starting"
    supervisor_error: str | None = None

    metadata: dict[str, Any] = {
        "checkpoint_directory": str(checkpoint_directory),
        "checkpoint_pattern": str(
            checkpoint_run_directory(checkpoint_directory, run_key) / "checkpoint-*"
        ),
        "command": _redact_command(command),
        "environment": _safe_environment_snapshot(environment),
        "max_runtime_seconds": max_runtime_seconds,
        "project_root": str(layout.project_root),
        "run_key": run_key,
        "started_at": started_at,
        "status": status,
        "wandb_run_id": run_key,
        "wandb_run_name": environment.get("WANDB_RUN_NAME"),
    }
    _atomic_write_json(metadata_path, metadata)

    signal_event = threading.Event()

    def handle_signal(signum: int, _frame: FrameType | None) -> None:
        nonlocal received_signal
        received_signal = signum
        signal_event.set()

    original_handlers: dict[int, Any] = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        original_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, handle_signal)

    try:
        with (
            stdout_path.open("a", encoding="utf-8", buffering=1) as stdout_log,
            stderr_path.open("a", encoding="utf-8", buffering=1) as stderr_log,
        ):
            process = subprocess.Popen(
                list(command),
                cwd=layout.project_root,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
            if process.stdout is None or process.stderr is None:
                raise RuntimeError("Unable to capture training stdout/stderr")
            stdout_tee = _StreamTee(process.stdout, stdout_log, sys.stdout)
            stderr_tee = _StreamTee(process.stderr, stderr_log, sys.stderr)
            stdout_tee.start()
            stderr_tee.start()
            status = "running"

            while process.poll() is None:
                if signal_event.is_set():
                    status = "interrupted"
                    _terminate_process_group(process, termination_grace_seconds)
                    break
                if time.monotonic() - start_monotonic >= max_runtime_seconds:
                    timed_out = True
                    status = "timed_out"
                    _terminate_process_group(process, termination_grace_seconds)
                    break
                time.sleep(poll_interval_seconds)

            child_exit_code = process.wait()
            stdout_tee.join(timeout=5.0)
            stderr_tee.join(timeout=5.0)
            if status == "running":
                status = "succeeded" if child_exit_code == 0 else "failed"
    except BaseException as exc:
        status = "supervisor_error"
        supervisor_error = f"{type(exc).__name__}: {exc}"
        with traceback_path.open("w", encoding="utf-8") as stream:
            traceback.print_exc(file=stream)
        if process is not None:
            _terminate_process_group(process, termination_grace_seconds)
            child_exit_code = process.poll()
    finally:
        for signal_number, handler in original_handlers.items():
            signal.signal(signal_number, handler)

    if timed_out:
        exit_code = 124
    elif received_signal is not None:
        exit_code = 128 + received_signal
    elif status == "supervisor_error":
        exit_code = 70
    else:
        exit_code = _normalise_signal_exit_code(child_exit_code)

    training_marker_configured = bool(
        environment.get("RUNPOD_TRAINING_COMPLETED_MARKER", "").strip()
    )
    training_marker_error: str | None = None
    try:
        marker_training_completed, training_marker_path = _training_completed_marker(
            layout,
            environment,
            run_key,
        )
    except ValueError as error:
        marker_training_completed = False
        training_marker_path = environment.get("RUNPOD_TRAINING_COMPLETED_MARKER") or None
        training_marker_error = f"{type(error).__name__}: {error}"
    if training_marker_error is not None or (
        training_marker_configured and exit_code == 0 and not marker_training_completed
    ):
        status = "supervisor_error"
        supervisor_error = training_marker_error or (
            "Training command succeeded without a valid training-completed marker"
        )
        exit_code = 70
    training_completed = marker_training_completed if training_marker_configured else exit_code == 0

    duration_seconds = round(time.monotonic() - start_monotonic, 3)
    resume_checkpoint = (
        None if training_completed else _latest_resume_checkpoint(checkpoint_directory, run_key)
    )
    if training_completed and exit_code != 0:
        next_action = "rerun_validation"
    elif resume_checkpoint is not None:
        next_action = "resume_training"
    else:
        next_action = "fix_failure_and_start_new_run"
    recovery = {
        "schema_version": "training-recovery-v1",
        "status": status,
        "training_completed": training_completed,
        "training_completed_marker": training_marker_path,
        "training_completed_marker_error": training_marker_error,
        "next_action": next_action,
        "timed_out": timed_out,
        "max_runtime_seconds": max_runtime_seconds,
        "duration_seconds": duration_seconds,
        "wandb_run_id": run_key,
        "runpod_pod_id": environment.get("RUNPOD_POD_ID"),
        "runpod_config": environment.get("RUNPOD_CONFIG"),
        "checkpoint_search_pattern": str(
            checkpoint_run_directory(checkpoint_directory, run_key) / "checkpoint-*"
        ),
        "resume_available": resume_checkpoint is not None,
        "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint is not None else None,
        "suggested_next_max_runtime_seconds": (
            max(max_runtime_seconds * 2, max_runtime_seconds + 3600) if timed_out else None
        ),
    }
    if resume_checkpoint is not None:
        recovery["resume_environment"] = {
            "RESUME_CHECKPOINT": str(resume_checkpoint),
            "WANDB_RUN_ID": run_key,
            "RUNPOD_RUN_KEY": run_key,
        }

    if exit_code != 0:
        stderr_tail = ""
        if stderr_path.exists():
            stderr_tail = stderr_path.read_text(encoding="utf-8", errors="replace")[-8000:]
        _atomic_write_json(
            run_directory / "failure.json",
            {
                "child_exit_code": child_exit_code,
                "exit_code": exit_code,
                "signal_number": received_signal,
                "status": status,
                "stderr_tail": stderr_tail,
                "supervisor_error": supervisor_error,
                "training_completed": training_completed,
                "training_completed_marker": training_marker_path,
                "training_completed_marker_error": training_marker_error,
                "timed_out": timed_out,
            },
        )
        _atomic_write_json(run_directory / "recovery.json", recovery)

    metadata.update(
        {
            "child_exit_code": child_exit_code,
            "duration_seconds": duration_seconds,
            "ended_at": _utc_now(),
            "exit_code": exit_code,
            "signal_number": received_signal,
            "status": status,
            "supervisor_error": supervisor_error,
            "timed_out": timed_out,
            "recovery": recovery,
        }
    )
    _atomic_write_json(metadata_path, metadata)

    configured_action = os.environ.get("RUNPOD_SHUTDOWN_ACTION", "terminate")
    shutdown_result = ShutdownResult(
        False, "automatic-shutdown-disabled", configured_action, None, None, None
    )
    if auto_shutdown:
        try:
            shutdown_result = shutdown_runpod(
                marker_path=marker_path,
                dry_run=dry_run_shutdown,
            )
        except Exception as exc:
            # Lifecycle cleanup must never replace the training process exit code.
            shutdown_result = ShutdownResult(
                True,
                None,
                configured_action,
                False,
                None,
                f"{type(exc).__name__}: {exc}",
            )
    metadata["shutdown"] = asdict(shutdown_result)
    _atomic_write_json(metadata_path, metadata)

    return RunResult(
        exit_code=exit_code,
        child_exit_code=child_exit_code,
        timed_out=timed_out,
        signal_number=received_signal,
        status=status,
        run_key=run_key,
        run_directory=run_directory,
        checkpoint_directory=checkpoint_directory,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run training with persistent logs, a deadline, and RunPod shutdown."
    )
    parser.add_argument(
        "--max-runtime-seconds",
        type=int,
        default=int(os.environ.get("MAX_RUNTIME_SECONDS", DEFAULT_MAX_RUNTIME_SECONDS)),
    )
    parser.add_argument(
        "--termination-grace-seconds",
        type=float,
        default=float(os.environ.get("TERMINATION_GRACE_SECONDS", "30")),
    )
    parser.add_argument("--dry-run-shutdown", action="store_true")
    parser.add_argument("--no-auto-shutdown", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    command = list(arguments.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("a training command is required after '--'")
    result = supervise(
        command,
        max_runtime_seconds=arguments.max_runtime_seconds,
        termination_grace_seconds=arguments.termination_grace_seconds,
        dry_run_shutdown=arguments.dry_run_shutdown,
        auto_shutdown=not arguments.no_auto_shutdown,
    )
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
