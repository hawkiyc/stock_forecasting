#!/usr/bin/env python3
"""Recover local RunPod guards after the control machine wakes or reboots."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RUNPODCTL = ROOT / "scripts" / "runpodctl_project.sh"
S3 = ROOT / "scripts" / "runpod_s3_project.sh"
GUARD_LAUNCHER = ROOT / "scripts" / "launch_runpod_guard.sh"
READINESS = ROOT / "scripts" / "runpod_readiness.py"
POD_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
PROJECT_ROLES = {
    "cpu-prep": ("stage1-cpu-preparation", "lifecycle/stage1/cpu-preparation.json"),
}
GPU_ROLES = {
    "gpu-train": ("stage1-training", "lifecycle/stage1/training.json"),
    "gpu-validation": ("stage1-validation", "lifecycle/stage1/validation.json"),
}
ACTIVE_RUNTIME_STATES = {"running", "initializing", "unknown"}
ACTIVE_LIFECYCLE_STATES = {"preparing", "finalizing", "downloaded_active"}
TERMINAL_LIFECYCLE_STATES = {
    "ready",
    "failed",
    "timed_out",
    "waiting_for_provider",
    "waiting_for_budget",
    "waiting_for_resume",
    "waiting_for_preparation",
    "downloaded",
}


class CommandError(RuntimeError):
    """Raised when a control-plane command cannot provide trustworthy output."""

    def __init__(self, command: list[str], result: subprocess.CompletedProcess[str]) -> None:
        detail = result.stderr.strip() or result.stdout.strip() or "no command output"
        super().__init__(f"{' '.join(command)} failed: {detail}")


def run_command(command: list[str]) -> str:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise CommandError(command, result)
    return result.stdout


def run_json(command: list[str]) -> Any:
    output = run_command(command)
    try:
        return json.loads(output)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"{' '.join(command)} returned invalid JSON") from error


def dotenv_value(path: Path, key: str) -> str:
    """Read one simple project dotenv value without executing the file."""

    matches: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip("\r")
        if line.startswith(f"{key}=") or line.startswith(f"export {key}="):
            matches.append(line.split("=", 1)[1])
    if len(matches) != 1:
        return ""
    value = matches[0]
    if len(value) >= 2 and value[0] in "\"'" and value[-1] == value[0]:
        return value[1:-1]
    return value


def parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(UTC)


def host_boot_identity() -> tuple[str, int | None]:
    """Return a stable boot identifier and boot epoch when the host exposes them."""

    boot_id_path = Path("/proc/sys/kernel/random/boot_id")
    boot_epoch: int | None = None
    try:
        boot_id = boot_id_path.read_text(encoding="utf-8").strip()
    except OSError:
        boot_id = ""
    if boot_id and re.fullmatch(r"[A-Za-z0-9._-]+", boot_id):
        try:
            for line in Path("/proc/stat").read_text(encoding="utf-8").splitlines():
                if line.startswith("btime "):
                    boot_epoch = int(line.split()[1])
                    break
        except (OSError, ValueError, IndexError):
            boot_epoch = None
        return f"linux-{boot_id}", boot_epoch

    sysctl_path = shutil.which("sysctl")
    if sysctl_path is None and Path("/usr/sbin/sysctl").is_file():
        sysctl_path = "/usr/sbin/sysctl"
    if sysctl_path is None:
        return "", None
    try:
        result = subprocess.run(
            [sysctl_path, "-n", "kern.boottime"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return "", None
    if result.returncode != 0:
        return "", None
    match = re.search(r"\bsec\s*=\s*([0-9]+)", result.stdout)
    if match is None:
        return "", None
    boot_epoch = int(match.group(1))
    return f"darwin-{boot_epoch}", boot_epoch


def load_readiness_module() -> Any:
    spec = importlib.util.spec_from_file_location("runpod_recovery_readiness", READINESS)
    if spec is None or spec.loader is None:
        raise RuntimeError("RunPod readiness helper cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def list_lifecycle_keys(volume_id: str) -> set[str]:
    payload = run_json(
        [
            "bash",
            str(S3),
            "s3api",
            "list-objects-v2",
            "--bucket",
            volume_id,
            "--prefix",
            "lifecycle/stage1/",
            "--output",
            "json",
        ]
    )
    contents = payload.get("Contents", []) if isinstance(payload, dict) else []
    return {
        str(item["Key"])
        for item in contents
        if isinstance(item, dict) and isinstance(item.get("Key"), str)
    }


def load_marker(volume_id: str, key: str) -> dict[str, Any] | None:
    try:
        payload = run_json(
            [
                "bash",
                str(S3),
                "s3",
                "cp",
                f"s3://{volume_id}/{key}",
                "-",
                "--only-show-errors",
            ]
        )
    except (CommandError, RuntimeError):
        return None
    return payload if isinstance(payload, dict) else None


def pod_volume_id(pod: dict[str, Any]) -> str:
    value = pod.get("networkVolumeId")
    if isinstance(value, str):
        return value
    network_volume = pod.get("networkVolume")
    if isinstance(network_volume, dict) and isinstance(network_volume.get("id"), str):
        return network_volume["id"]
    return ""


def project_pod(pod: dict[str, Any], volume_id: str) -> bool:
    pod_id = str(pod.get("id", ""))
    if not POD_ID_PATTERN.fullmatch(pod_id) or pod_volume_id(pod) != volume_id:
        return False
    env = pod.get("env")
    if not isinstance(env, dict):
        return False
    role = env.get("RUNPOD_ROLE")
    if role not in PROJECT_ROLES and role not in GPU_ROLES:
        return False
    project_root = str(env.get("PROJECT_ROOT", ""))
    mount_root = str(env.get("NETWORK_VOLUME_ROOT", env.get("RUNPOD_VOLUME_ROOT", "")))
    return bool(project_root and mount_root and project_root == f"{mount_root}/stock_forecasting")


def pod_records(volume_id: str, requested_pod_id: str | None) -> list[dict[str, Any]]:
    payload = run_json(["bash", str(RUNPODCTL), "pod", "list", "--output", "json"])
    if not isinstance(payload, list):
        raise RuntimeError("RunPod Pod list did not return an array")
    records: list[dict[str, Any]] = []
    for listed in payload:
        if not isinstance(listed, dict):
            continue
        pod_id = str(listed.get("id", ""))
        if requested_pod_id and pod_id != requested_pod_id:
            continue
        if not POD_ID_PATTERN.fullmatch(pod_id):
            continue
        try:
            pod = run_json(
                [
                    "bash",
                    str(RUNPODCTL),
                    "pod",
                    "get",
                    pod_id,
                    "--include-network-volume",
                    "--output",
                    "json",
                ]
            )
        except (CommandError, RuntimeError) as error:
            print(f"pod={pod_id} action=unknown reason=pod_get_failed detail={error}")
            continue
        if isinstance(pod, dict) and project_pod(pod, volume_id):
            records.append(pod)
    return records


def expected_lifecycle(pod: dict[str, Any]) -> tuple[str, str] | None:
    env = pod.get("env", {})
    role = env.get("RUNPOD_ROLE") if isinstance(env, dict) else None
    if role in PROJECT_ROLES:
        kind, key = PROJECT_ROLES[role]
        if env.get("RUNPOD_STAGE") == "stage2":
            return "stage1-mixed-finalization", "lifecycle/stage1/mixed-finalization.json"
        return kind, key
    if role in GPU_ROLES:
        return GPU_ROLES[role]
    return None


def lifecycle_state(
    pod: dict[str, Any],
    volume_id: str,
    lifecycle_keys: set[str],
    readiness: Any,
) -> tuple[str, str]:
    expected = expected_lifecycle(pod)
    if expected is None:
        return "unknown", "unsupported_role"
    kind, key = expected
    if key not in lifecycle_keys:
        return "missing", key
    marker = load_marker(volume_id, key)
    if marker is None:
        return "unknown", f"invalid_or_unreadable:{key}"
    env = pod.get("env", {})
    active_run_id = str(env.get("WANDB_RUN_ID", "")) if isinstance(env, dict) else ""
    try:
        state = readiness._guard_lifecycle_state(
            marker,
            expected_kind=kind,
            expected_pod_id=str(pod["id"]),
            active_run_id=active_run_id,
        )
    except (TypeError, ValueError) as error:
        return "unknown", f"invalid_marker:{error}"
    if state in ACTIVE_LIFECYCLE_STATES:
        return "active", f"{key}:{state}"
    if state in TERMINAL_LIFECYCLE_STATES:
        return "terminal", f"{key}:{state}"
    return "unknown", f"unsupported_marker_state:{state}"


def guard_deadline(pod_id: str, guard_dir: Path) -> tuple[int, int] | None:
    ready_path = guard_dir / f"{pod_id}.ready.json"
    log_path = guard_dir / f"{pod_id}.log"
    try:
        ready = json.loads(ready_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        ready = {}
    armed_at = parse_timestamp(ready.get("armed_at"))
    delay_value = ready.get("delay_seconds")
    if not isinstance(delay_value, int) or delay_value < 1:
        try:
            log_text = log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            log_text = ""
        match = re.search(r"terminate after ([1-9][0-9]*) seconds", log_text)
        if match:
            delay_value = int(match.group(1))
    if armed_at is None or not isinstance(delay_value, int) or delay_value < 1:
        return None
    return int(armed_at.timestamp()) + delay_value, delay_value


def guard_alive(pod_id: str, guard_dir: Path) -> bool:
    try:
        pid = int((guard_dir / f"{pod_id}.pid").read_text(encoding="utf-8").strip())
        ready = json.loads(
            (guard_dir / f"{pod_id}.ready.json").read_text(encoding="utf-8")
        )
        if (
            ready.get("state") != "armed"
            or ready.get("pod_id") != pod_id
            or ready.get("pid") != pid
        ):
            return False
        current_boot_id, current_boot_epoch = host_boot_identity()
        recorded_boot_id = ready.get("host_boot_id")
        if isinstance(recorded_boot_id, str) and recorded_boot_id:
            if current_boot_id and recorded_boot_id != current_boot_id:
                return False
        elif current_boot_epoch is not None:
            armed_at = parse_timestamp(ready.get("armed_at"))
            if armed_at is not None and armed_at.timestamp() < current_boot_epoch:
                return False
        os.kill(pid, 0)
    except PermissionError:
        # A permission error means the process exists but cannot be probed here.
        # Keep the Pod and avoid creating a duplicate guard in that case.
        return True
    except ProcessLookupError:
        return False
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    try:
        process = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return True
    if process.returncode == 0:
        command = process.stdout.strip()
        return "terminate_runpod_after.sh" in command and pod_id in command
    if "operation not permitted" in process.stderr.lower():
        return True
    if not process.stdout.strip() and not process.stderr.strip():
        return False
    return True


def safe_delete(pod_id: str) -> None:
    run_command(["bash", str(RUNPODCTL), "pod", "delete", pod_id])


def rearm_guard(pod: dict[str, Any], remaining: int, guard_dir: Path) -> None:
    expected = expected_lifecycle(pod)
    if expected is None:
        raise RuntimeError("Pod role has no approved lifecycle mapping")
    _, lifecycle_key = expected
    required_commands = ["bash", "python3", "runpodctl", "aws"]
    if sys.platform == "darwin":
        required_commands.append("caffeinate")
    missing_commands = [name for name in required_commands if shutil.which(name) is None]
    if missing_commands:
        raise RuntimeError(
            "guard re-arm prerequisites are missing: " + ", ".join(missing_commands)
        )
    required_files = [GUARD_LAUNCHER, RUNPODCTL, S3, READINESS]
    unavailable_files = [str(path) for path in required_files if not os.access(path, os.R_OK)]
    if unavailable_files:
        raise RuntimeError(
            "guard re-arm project files are unavailable: " + ", ".join(unavailable_files)
        )
    current_boot_id, _ = host_boot_identity()
    if not current_boot_id:
        raise RuntimeError("guard re-arm cannot establish the current host boot identity")
    env = pod.get("env", {})
    run_id = str(env.get("WANDB_RUN_ID", "")) if isinstance(env, dict) else ""
    guard_log = guard_dir / f"{pod['id']}.log"
    command_env = os.environ.copy()
    command_env["RUNPOD_GUARD_VOLUME_ROOT"] = str(
        env.get("NETWORK_VOLUME_ROOT", env.get("RUNPOD_VOLUME_ROOT", "/runpod-volume"))
    )
    command_env["RUNPOD_GUARD_RUN_ID"] = run_id
    command_env["RUNPOD_GUARD_EMERGENCY_TERMINATE_ON_STARTUP_FAILURE"] = "0"
    result = subprocess.run(
        [
            "bash",
            str(GUARD_LAUNCHER),
            str(pod["id"]),
            str(max(1, remaining)),
            lifecycle_key,
            str(guard_log),
        ],
        capture_output=True,
        text=True,
        check=False,
        env=command_env,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "no launcher output"
        raise RuntimeError(f"guard re-arm failed: {detail}")


def observe(
    volume_id: str,
    requested_pod_id: str | None,
    guard_dir: Path,
    readiness: Any,
) -> list[dict[str, Any]]:
    lifecycle_keys = list_lifecycle_keys(volume_id)
    observations: list[dict[str, Any]] = []
    for pod in pod_records(volume_id, requested_pod_id):
        pod_id = str(pod["id"])
        runtime = str(pod.get("runtimeStatus", "unknown"))
        if runtime not in ACTIVE_RUNTIME_STATES:
            observations.append({"pod": pod, "action": "skip", "reason": f"runtime={runtime}"})
            continue
        state, evidence = lifecycle_state(pod, volume_id, lifecycle_keys, readiness)
        if runtime in {"initializing", "unknown"}:
            action, reason = "keep", f"runtime={runtime}; lifecycle={state}:{evidence}"
        elif state == "active":
            action, reason = "keep", evidence
        elif state == "terminal":
            action, reason = "terminate", evidence
        else:
            action, reason = "keep", f"cannot_prove_idle; lifecycle={state}:{evidence}"
        deadline_info = guard_deadline(pod_id, guard_dir)
        if action == "keep" and runtime == "running" and deadline_info is not None:
            deadline, _ = deadline_info
            remaining = deadline - int(time.time())
            if remaining <= 0:
                action, reason = "terminate", "external_guard_deadline_expired"
            elif not guard_alive(pod_id, guard_dir):
                reason += f"; guard_missing; guard_remaining_seconds={remaining}"
                observations.append(
                    {
                        "pod": pod,
                        "action": action,
                        "reason": reason,
                        "rearm_remaining": remaining,
                    }
                )
                continue
        observations.append({"pod": pod, "action": action, "reason": reason})
    return observations


def print_observation(observation: dict[str, Any], prefix: str = "") -> None:
    pod = observation["pod"]
    print(
        f"{prefix}pod={pod['id']} name={pod.get('name', 'unknown')} "
        f"runtime={pod.get('runtimeStatus', 'unknown')} action={observation['action']} "
        f"reason={observation['reason']}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect project RunPod workloads and restore their local guards after wake."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="terminate confirmed idle Pods and re-arm active guards",
    )
    parser.add_argument("--pod-id", help="inspect only one exact Pod ID")
    parser.add_argument("--confirmations", type=int, default=2)
    parser.add_argument("--confirmation-delay-seconds", type=int, default=5)
    parser.add_argument(
        "--guard-dir",
        type=Path,
        default=Path(
            os.environ.get(
                "RUNPOD_GUARD_LOG_DIR", Path.home() / ".local/state/runpod-guards"
            )
        ),
    )
    arguments = parser.parse_args()
    if arguments.pod_id and not POD_ID_PATTERN.fullmatch(arguments.pod_id):
        parser.error("--pod-id is invalid")
    if arguments.confirmations < 2 or arguments.confirmation_delay_seconds < 0:
        parser.error("--confirmations must be at least 2 and delay must not be negative")
    env_file = Path(os.environ.get("RUNPOD_ENV_FILE", ROOT / ".env"))
    if not env_file.is_file():
        print(f"RunPod project environment file not found: {env_file}", file=sys.stderr)
        return 2
    volume_id = dotenv_value(env_file, "RUNPOD_NETWORK_VOLUME_ID")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", volume_id):
        print("RUNPOD_NETWORK_VOLUME_ID is required and must use safe characters", file=sys.stderr)
        return 2
    readiness = load_readiness_module()
    try:
        first = observe(
            volume_id,
            arguments.pod_id,
            arguments.guard_dir,
            readiness,
        )
    except (CommandError, RuntimeError) as error:
        print(f"Recovery check failed closed: {error}", file=sys.stderr)
        return 2
    for observation in first:
        print_observation(observation)
    if not arguments.apply:
        print("DRY RUN: no Pod was terminated and no guard was re-armed.")
        return 0

    confirmed = first
    first_actions = {str(item["pod"]["id"]): item["action"] for item in first}
    for confirmation_number in range(1, arguments.confirmations):
        if arguments.confirmation_delay_seconds:
            time.sleep(arguments.confirmation_delay_seconds)
        try:
            confirmed = observe(
                volume_id,
                arguments.pod_id,
                arguments.guard_dir,
                readiness,
            )
        except (CommandError, RuntimeError) as error:
            print(f"Recovery confirmation failed closed: {error}", file=sys.stderr)
            return 2
        confirmed_actions = {
            str(item["pod"]["id"]): item["action"] for item in confirmed
        }
        if first_actions != confirmed_actions:
            print(
                f"Recovery observations changed at confirmation {confirmation_number + 1}; "
                "no action was taken.",
                file=sys.stderr,
            )
            return 2

    # Re-check immediately before any mutation so a terminal observation that
    # changed back to active cannot be acted on after the confirmation delay.
    try:
        latest = observe(
            volume_id,
            arguments.pod_id,
            arguments.guard_dir,
            readiness,
        )
    except (CommandError, RuntimeError) as error:
        print(f"Final recovery check failed closed: {error}", file=sys.stderr)
        return 2
    latest_actions = {str(item["pod"]["id"]): item["action"] for item in latest}
    confirmed_actions = {str(item["pod"]["id"]): item["action"] for item in confirmed}
    if latest_actions != confirmed_actions:
        print(
            "Recovery observations changed before mutation; no action was taken.",
            file=sys.stderr,
        )
        return 2
    confirmed = latest

    failures = 0
    for observation in confirmed:
        print_observation(observation, prefix="confirmed ")
        pod = observation["pod"]
        pod_id = str(pod["id"])
        if observation["action"] == "terminate":
            try:
                safe_delete(pod_id)
                print(f"terminated pod={pod_id}")
            except (CommandError, RuntimeError) as error:
                failures += 1
                print(f"pod={pod_id} terminate_failed detail={error}", file=sys.stderr)
        elif observation["action"] == "keep" and "rearm_remaining" in observation:
            try:
                rearm_guard(pod, int(observation["rearm_remaining"]), arguments.guard_dir)
                print(
                    f"re-armed guard pod={pod_id} "
                    f"remaining_seconds={observation['rearm_remaining']}"
                )
            except (OSError, RuntimeError) as error:
                failures += 1
                print(f"pod={pod_id} guard_rearm_failed detail={error}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
