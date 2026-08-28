"""Static guardrails around the stable RunPod lifecycle shell."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from stock_forecasting.config import ExperimentConfig

ROOT = Path(__file__).resolve().parents[1]


def test_all_shell_scripts_remain_syntax_valid() -> None:
    scripts = sorted((ROOT / "scripts").rglob("*.sh"))
    assert scripts
    result = subprocess.run(
        ["bash", "-n", *[str(path) for path in scripts]],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_runpod_entrypoint_scripts_declare_explicit_interpreters() -> None:
    entrypoints = (
        ROOT / "scripts/create_runpod_cpu_pod.sh",
        ROOT / "scripts/create_runpod_pod.sh",
        ROOT / "scripts/create_runpod_resume_pod.sh",
        ROOT / "scripts/create_runpod_validation_pod.sh",
        ROOT / "scripts/runpod_cpu_prepare.sh",
        ROOT / "scripts/runpod_cpu_finalize.sh",
        ROOT / "scripts/runpod_entrypoint.sh",
        ROOT / "scripts/runpod_self_terminate.py",
        ROOT / "scripts/runpod_self_terminate.sh",
        ROOT / "scripts/runpod_train_then_validate.sh",
        ROOT / "scripts/runpod_validation.sh",
        ROOT / "scripts/runpod_wandb_sync.sh",
        ROOT / "scripts/runpod_workflow.sh",
    )
    for path in entrypoints:
        expected_shebang = (
            "#!/usr/bin/env python3" if path.suffix == ".py" else "#!/usr/bin/env bash"
        )
        assert path.read_text(encoding="utf-8").splitlines()[0] == expected_shebang, path


def test_dataset_lifecycle_accepts_matching_resumable_progress(
    tmp_path: Path,
) -> None:
    volume_root = tmp_path / "runpod-volume"
    digest = "a" * 64
    progress_path = volume_root / "datasets" / digest / "download-progress.json"
    progress_path.parent.mkdir(parents=True)
    identity = {"dataset_request_sha256": digest}
    identity_sha256 = hashlib.sha256(
        json.dumps(
            identity,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    marker = volume_root / "lifecycle" / "stage1" / "dataset.json"
    for state in (
        "waiting_for_provider",
        "waiting_for_budget",
        "waiting_for_resume",
        "downloaded",
    ):
        progress_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "ohlcv-download-progress",
                    "state": state,
                    "identity": identity,
                    "identity_sha256": identity_sha256,
                }
            ),
            encoding="utf-8",
        )
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/runpod_readiness.py"),
                "write-state",
                "--output",
                str(marker),
                "--network-volume-root",
                str(volume_root),
                "--kind",
                "stage1-dataset",
                "--state",
                state,
                "--launch-id",
                "test-launch",
                "--exit-code",
                "75",
                "--progress-path",
                str(progress_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, result.stderr
        payload = json.loads(marker.read_text(encoding="utf-8"))
        assert payload["state"] == state
        assert payload["progress_path"] == str(progress_path.resolve())


def test_downloaded_lifecycle_can_be_an_active_intermediate_checkpoint(
    tmp_path: Path,
) -> None:
    volume_root = tmp_path / "runpod-volume"
    digest = "b" * 64
    progress_path = volume_root / "datasets" / digest / "download-progress.json"
    progress_path.parent.mkdir(parents=True)
    identity = {"dataset_request_sha256": digest}
    progress_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "ohlcv-download-progress",
                "state": "downloaded",
                "identity": identity,
                "identity_sha256": hashlib.sha256(
                    json.dumps(
                        identity,
                        ensure_ascii=True,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    marker = volume_root / "lifecycle" / "stage1" / "dataset.json"

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/runpod_readiness.py"),
            "write-state",
            "--output",
            str(marker),
            "--network-volume-root",
            str(volume_root),
            "--kind",
            "stage1-dataset",
            "--state",
            "downloaded",
            "--launch-id",
            "test-launch",
            "--progress-path",
            str(progress_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert payload["state"] == "downloaded"
    assert "exit_code" not in payload


def test_same_launch_terminal_finalizer_preserves_download_progress(
    tmp_path: Path,
) -> None:
    volume_root = tmp_path / "runpod-volume"
    digest = "c" * 64
    progress_path = volume_root / "datasets" / digest / "download-progress.json"
    progress_path.parent.mkdir(parents=True)
    identity = {"dataset_request_sha256": digest}
    progress_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "ohlcv-download-progress",
                "state": "failed",
                "identity": identity,
                "identity_sha256": hashlib.sha256(
                    json.dumps(
                        identity,
                        ensure_ascii=True,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    marker = volume_root / "lifecycle" / "stage1" / "dataset.json"
    environment = {**os.environ, "RUNPOD_POD_ID": "test-pod"}
    base_command = [
        sys.executable,
        str(ROOT / "scripts/runpod_readiness.py"),
        "write-state",
        "--output",
        str(marker),
        "--network-volume-root",
        str(volume_root),
        "--kind",
        "stage1-dataset",
        "--state",
        "failed",
        "--launch-id",
        "test-launch",
        "--exit-code",
        "1",
    ]
    first = subprocess.run(
        [*base_command, "--progress-path", str(progress_path)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    second = subprocess.run(
        [*base_command, "--inherit-existing"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert payload["state"] == "failed"
    assert payload["pod_id"] == "test-pod"
    assert payload["progress_path"] == str(progress_path.resolve())


def test_tmux_resumable_dataset_validator_requires_matching_progress(
    tmp_path: Path,
) -> None:
    volume_root = tmp_path / "runpod-volume"
    digest = "d" * 64
    progress_path = volume_root / "datasets" / digest / "download-progress.json"
    progress_path.parent.mkdir(parents=True)
    identity = {"dataset_request_sha256": digest}
    progress_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "ohlcv-download-progress",
                "state": "waiting_for_provider",
                "identity": identity,
                "identity_sha256": hashlib.sha256(
                    json.dumps(
                        identity,
                        ensure_ascii=True,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    marker = volume_root / "lifecycle" / "stage1" / "dataset.json"
    environment = {**os.environ, "RUNPOD_POD_ID": "test-pod"}
    write_result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/runpod_readiness.py"),
            "write-state",
            "--output",
            str(marker),
            "--network-volume-root",
            str(volume_root),
            "--kind",
            "stage1-dataset",
            "--state",
            "waiting_for_provider",
            "--launch-id",
            "test-launch",
            "--exit-code",
            "75",
            "--progress-path",
            str(progress_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    validate_result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/runpod_readiness.py"),
            "resumable-dataset-lifecycle",
            "--marker",
            str(marker),
            "--network-volume-root",
            str(volume_root),
            "--launch-id",
            "test-launch",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert write_result.returncode == 0, write_result.stderr
    assert validate_result.returncode == 0, validate_result.stderr
    assert validate_result.stdout.strip() == "waiting_for_provider"


def test_runpod_entrypoints_use_only_two_quant_stages() -> None:
    relevant = (
        ROOT / "scripts/create_runpod_pod.sh",
        ROOT / "scripts/runpod_entrypoint.sh",
        ROOT / "scripts/runpod_train_then_validate.sh",
        ROOT / "scripts/runpod_validation.sh",
        ROOT / "scripts/runpod_selection.py",
        ROOT / "scripts/verify_runpod_stage_readiness.sh",
    )
    combined = "\n".join(path.read_text(encoding="utf-8") for path in relevant)

    assert "stage1_kronos_base_lora.yaml" in combined
    assert "stage2_kronos_base_lora.yaml" in combined
    for forbidden in (
        "stage1_gemma",
        "stage2_gemma",
        "stage3_gpt",
        "stage4_gpt",
        "fact_head",
        "text_generation",
        "llm_tokenizer",
    ):
        assert forbidden not in combined.lower()


def test_stable_runpod_lifecycle_and_synchronization_boundaries_remain() -> None:
    create_gpu = (ROOT / "scripts/create_runpod_pod.sh").read_text(encoding="utf-8")
    create_cpu = (ROOT / "scripts/create_runpod_cpu_pod.sh").read_text(encoding="utf-8")
    sync = (ROOT / "scripts/sync_project_to_runpod_volume.sh").read_text(encoding="utf-8")
    entrypoint = (ROOT / "scripts/runpod_entrypoint.sh").read_text(encoding="utf-8")
    tmux = (ROOT / "scripts/runpod_tmux_launch.sh").read_text(encoding="utf-8")
    validation = (ROOT / "scripts/runpod_validation.sh").read_text(encoding="utf-8")
    guard = (ROOT / "scripts/launch_runpod_guard.sh").read_text(encoding="utf-8")
    selection_loader = (ROOT / "scripts/lib/runpod_selection.sh").read_text(
        encoding="utf-8"
    )
    pid1_environment = (ROOT / "scripts/runpod_reexec_with_pid1_env.py").read_text(
        encoding="utf-8"
    )

    for script in (create_gpu, create_cpu):
        assert "RUNPOD_NETWORK_VOLUME_ID" in script
        assert "RUNPOD_DATACENTER_ID" in script
    assert "rest.runpod.io" in create_cpu
    assert "runpodctl_project.sh" in create_gpu
    assert "pod create" in create_gpu
    assert "--dry-run" in sync
    assert "--apply" in sync
    assert 'CODE_MARKER_KEY="lifecycle/stage1/code.json"' in sync
    syncing_index = sync.index("render_code_manifest syncing")
    ready_index = sync.rindex("render_code_manifest ready")
    verification_index = sync.rindex('python3 "${READINESS_HELPER}" check-code')
    assert syncing_index < ready_index < verification_index
    assert "runpod_train_then_validate.sh" in entrypoint
    assert "runpod_self_terminate.sh" in entrypoint
    assert "runpod_self_terminate.sh" in tmux
    assert "launch_runpod_guard.sh" in create_gpu
    assert "launch_runpod_guard.sh" in create_cpu
    assert '"RUNPOD_DATASET_REVISION":"%s"' in create_gpu
    assert '"RUNPOD_DATASET_REVISION":"%s"' in create_cpu
    assert "RUNPOD_DATASET_REVISION" in selection_loader
    assert '"RUNPOD_DATASET_REVISION"' in pid1_environment
    assert 'caffeinate -is -w "${GUARD_PID}"' in guard
    assert (
        'RUNPOD_VALIDATION_LIFECYCLE_MARKER="${NETWORK_VOLUME_ROOT}'
        '/lifecycle/stage1/validation.json"' in entrypoint
    )
    assert "stock_forecasting.cli.validate_benchmarks" in validation


def test_cpu_log_lifecycle_and_downloader_share_canonical_paths() -> None:
    tmux = (ROOT / "scripts/runpod_tmux_launch.sh").read_text(encoding="utf-8")
    download = (ROOT / "scripts/download_runpod_cpu_logs.sh").read_text(encoding="utf-8")

    lifecycle_writer = tmux.split("publish_lifecycle() {", maxsplit=1)[1]
    assert '"${NETWORK_VOLUME_ROOT}" "${LAUNCH_ID}" "${JOB_LOG}"' in lifecycle_writer
    assert "expected_log_dir = expected_prefix + launch_id" in download
    assert 'expected_log_file = expected_log_dir + "/combined.log"' in download
    assert "log_path == expected_log_dir" in download
    assert "log_path == expected_log_file" in download


def test_cpu_acquisition_budget_is_resumable_and_reserves_preparation_time() -> None:
    ingestion = (ROOT / "src/stock_forecasting/data/ingestion.py").read_text(encoding="utf-8")
    progress = (ROOT / "src/stock_forecasting/data/download_progress.py").read_text(
        encoding="utf-8"
    )
    prepare = (ROOT / "scripts/runpod_cpu_prepare.sh").read_text(encoding="utf-8")
    guard = (ROOT / "scripts/terminate_runpod_after.sh").read_text(encoding="utf-8")
    status = (ROOT / "scripts/show_runpod_status.sh").read_text(encoding="utf-8")
    tmux = (ROOT / "scripts/runpod_tmux_launch.sh").read_text(encoding="utf-8")
    readiness = (ROOT / "scripts/runpod_readiness.py").read_text(encoding="utf-8")

    assert "Estimated HTTP requests" not in ingestion
    assert "exceed max_api_calls" not in ingestion
    assert "plan_with_official_trading_sessions" in ingestion
    assert "taiwan_pre_calendar_weekday_upper_bound_calls" in ingestion
    assert 'limited_providers={"eodhd"}' in ingestion
    assert "_run_parallel_provider_loops(runners)" in ingestion
    assert ingestion.count("max_backoff_seconds=options.max_backoff_seconds") == 2
    assert (
        '"provider_max_backoff_scope": ["eodhd", "tpex_official", "twse_official"]'
        in ingestion
    )
    assert "parallel_independent_loops_joined_before_process_exit" in ingestion
    assert '"waiting_for_budget"' in progress
    assert '"waiting_for_resume"' in progress
    assert "RUNPOD_CPU_PREPARE_RESERVE_SECONDS" in prepare
    assert "ACQUISITION_DEADLINE_EPOCH" in prepare
    assert "RUNPOD_PROVIDER_MAX_BACKOFF_SECONDS" in prepare
    assert 'error.get("provider_outcomes")' in status
    assert 'for key in ("category", "error_type", "operation", "item", "status_code")' in status
    assert 'ln "${RAW_STAGING}" "${RAW_FINAL}"' in prepare
    assert "fin-ts-verify-download" in prepare
    assert "obsolete-security-scope" in prepare
    assert "rebuilding from verified provider cache entries" in prepare
    assert "resumable-dataset-lifecycle" in tmux
    assert "${cpu_resumable_lifecycle_valid} -ne 1" in tmux
    assert "The CPU worker publishes the precise waiting state" in tmux
    assert "same_launch" in readiness
    for state in ("waiting_for_budget", "waiting_for_resume", "downloaded"):
        assert state in guard
    assert 'print("downloaded_active")' in guard
    assert 'payload.get("exit_code") != 75' in guard


def test_network_volume_identity_and_exact_mount_are_fail_closed() -> None:
    verifier = (ROOT / "scripts/verify_runpod_mounted_readiness.sh").read_text(encoding="utf-8")
    create_gpu = (ROOT / "scripts/create_runpod_pod.sh").read_text(encoding="utf-8")
    create_cpu = (ROOT / "scripts/create_runpod_cpu_pod.sh").read_text(encoding="utf-8")
    reexec = (ROOT / "scripts/runpod_reexec_with_pid1_env.py").read_text(encoding="utf-8")
    volume_writers = (
        ROOT / "scripts/runpod_tmux_launch.sh",
        ROOT / "scripts/runpod_cpu_prepare.sh",
        ROOT / "scripts/runpod_cpu_finalize.sh",
        ROOT / "scripts/runpod_entrypoint.sh",
        ROOT / "scripts/runpod_validation.sh",
    )

    assert 'RUNPOD_EXPECTED_VOLUME_ID":"%s"' in create_gpu
    assert 'RUNPOD_EXPECTED_VOLUME_ID":"%s"' in create_cpu
    assert '"RUNPOD_EXPECTED_VOLUME_ID"' in reexec
    assert '"RUNPOD_VOLUME_ID"' in reexec
    assert '"RUNPOD_API_KEY"' in reexec
    assert '"RUNPOD_API_KEY",' not in reexec.split("FORBIDDEN_NAMES", maxsplit=1)[0]
    assert '"RUNPOD_VOLUME_ID" != "${RUNPOD_EXPECTED_VOLUME_ID}"' not in verifier
    assert '"${RUNPOD_VOLUME_ID}" != "${RUNPOD_EXPECTED_VOLUME_ID}"' in verifier
    assert "mountpoint -q" in verifier
    assert "findmnt -rn -M" in verifier
    assert "/proc/self/mountinfo" in verifier
    for path in volume_writers:
        script = path.read_text(encoding="utf-8")
        assert 'verify_runpod_mounted_readiness.sh" --mount-only' in script, path


def test_workflow_exposes_bounded_cpu_gpu_resume_and_validation_options() -> None:
    workflow = (ROOT / "scripts/runpod_workflow.sh").read_text(encoding="utf-8")
    cpu_creator = (ROOT / "scripts/create_runpod_cpu_pod.sh").read_text(encoding="utf-8")
    gpu_creator = (ROOT / "scripts/create_runpod_pod.sh").read_text(encoding="utf-8")
    reexec = (ROOT / "scripts/runpod_reexec_with_pid1_env.py").read_text(encoding="utf-8")
    resume = (ROOT / "scripts/create_runpod_resume_pod.sh").read_text(encoding="utf-8")
    validation = (ROOT / "scripts/create_runpod_validation_pod.sh").read_text(encoding="utf-8")

    for option in (
        "--interactive",
        "--maxRuntime",
        "--prepareReserve",
        "--maxApiCalls",
        "--eodhdQps",
        "--taiwanQps",
        "--maxBackoff",
        "--gpuId",
        "--cpuNumber",
        "--cpuFlavor",
    ):
        assert option in workflow
    assert "cpu_max_runtime=6h" in workflow
    assert "cpu_prepare_reserve=auto" in workflow
    assert 'cpu_max_api_calls=""' in workflow
    assert "cpu_eodhd_qps=16" in workflow
    assert "cpu_taiwan_qps=0.5" in workflow
    assert "cpu_max_backoff=1m" in workflow
    assert "cpu_number=8" in workflow
    assert "cpu_flavor=cpu3g" in workflow
    assert "train_max_runtime=12h" in workflow
    assert 'train_gpu_id="NVIDIA GeForce RTX 5090"' in workflow
    assert "cpu3c|cpu3g|cpu3m|cpu5c|cpu5g|cpu5m" in cpu_creator
    assert '"${RUNPOD_CPU_VCPU_COUNT}" -gt 32' in cpu_creator
    assert "CONTAINER_DISK_GB_PER_VCPU=10" in cpu_creator
    assert "CONTAINER_DISK_GB_PER_VCPU=15" in cpu_creator
    assert "MAX_CONTAINER_DISK_GB" in cpu_creator
    assert "RUNPOD_CPU_PREPARE_RESERVE_SECONDS" in cpu_creator
    assert "RUNPOD_CPU_MAX_API_CALLS" in cpu_creator
    assert "RUNPOD_CPU_EODHD_QPS" in cpu_creator
    assert "RUNPOD_CPU_TAIWAN_QPS" in cpu_creator
    for setting in (
        "RUNPOD_CPU_MAX_API_CALLS",
        "RUNPOD_CPU_EODHD_QPS",
        "RUNPOD_CPU_TAIWAN_QPS",
    ):
        assert setting not in gpu_creator
    assert "RUNPOD_PROVIDER_MAX_BACKOFF_SECONDS" in cpu_creator
    assert '"RUNPOD_CPU_PREPARE_RESERVE_SECONDS"' in reexec
    assert '"RUNPOD_CPU_MAX_API_CALLS"' in reexec
    assert '"RUNPOD_CPU_EODHD_QPS"' in reexec
    assert '"RUNPOD_CPU_TAIWAN_QPS"' in reexec
    assert '"RUNPOD_PROVIDER_MAX_BACKOFF_SECONDS"' in reexec
    assert "resumable-training-run" in resume
    assert "training-completed.json" in resume
    assert "--maxRuntime" in validation
    assert "--gpuId" in validation


def test_cpu_prepare_without_options_is_interactive_and_cancel_safe() -> None:
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/runpod_workflow.sh"), "cpu", "prepare"],
        input="\n100000\n\n\n\n\n\n\n\n",
        check=False,
        capture_output=True,
        text=True,
    )

    combined = result.stdout + result.stderr
    assert result.returncode == 0, combined
    assert "CPU preparation acquisition and resource configuration" in combined
    assert "Maximum runtime [6h]:" in combined
    assert "Maximum additional EODHD network attempts for this CPU Pod (required):" in combined
    assert "EODHD requests per second [16]:" in combined
    assert "TWSE/TPEx requests per second per provider [0.5]:" in combined
    assert "Time reserved for data cleaning/window construction [auto]:" in combined
    assert "Maximum provider retry backoff [1m]:" in combined
    assert "vCPU count [8]:" in combined
    assert "CPU flavor [cpu3g]:" in combined
    assert "Maximum runtime: 6h" in combined
    assert "Cleaning/window reserve: automatic" in combined
    assert "Maximum additional EODHD network attempts: 100000" in combined
    assert "EODHD requests per second: 16" in combined
    assert "TWSE/TPEx requests per second per provider: 0.5" in combined
    assert "Maximum provider retry backoff: 1m" in combined
    assert "vCPU count: 8" in combined
    assert "CPU flavor: cpu3g" in combined
    assert "Create this CPU preparation Pod? [y/N]:" in combined
    assert "CPU preparation Pod creation cancelled; no Pod was created" in combined


def test_cpu_prepare_rejects_a_reserve_that_consumes_the_complete_runtime() -> None:
    result = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts/runpod_workflow.sh"),
            "cpu",
            "prepare",
            "--maxRuntime",
            "6h",
            "--prepareReserve",
            "6h",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "--prepareReserve must be shorter than --maxRuntime" in result.stderr


def test_noninteractive_cpu_prepare_requires_an_explicit_api_budget() -> None:
    result = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts/runpod_workflow.sh"),
            "cpu",
            "prepare",
            "--maxRuntime",
            "6h",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "--max-api-calls must be a positive integer" in result.stderr


@pytest.mark.parametrize(
    ("option", "value", "message"),
    (
        ("--max-api-calls", "0", "--max-api-calls must be a positive integer"),
        ("--eodhd-qps", "0", "--eodhd-qps must be a positive number"),
        ("--taiwan-qps", "invalid", "--taiwan-qps must be a positive number"),
    ),
)
def test_cpu_prepare_rejects_invalid_acquisition_settings(
    option: str,
    value: str,
    message: str,
) -> None:
    arguments = [
        "bash",
        str(ROOT / "scripts/runpod_workflow.sh"),
        "cpu",
        "prepare",
        "--max-api-calls",
        "1",
    ]
    arguments.extend((option, value))
    result = subprocess.run(
        arguments,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert message in result.stderr


def test_cpu_prepare_rejects_an_invalid_maximum_provider_backoff() -> None:
    result = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts/runpod_workflow.sh"),
            "cpu",
            "prepare",
            "--maxBackoff",
            "60",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "--maxBackoff must use a positive duration" in result.stderr


def test_download_defaults_to_all_retained_checkpoints_and_can_select_best() -> None:
    workflow = (ROOT / "scripts/runpod_workflow.sh").read_text(encoding="utf-8")
    download = (ROOT / "scripts/download_runpod_results.sh").read_text(encoding="utf-8")

    assert "download [--resume] [--checkpointScope all|best] [RUN_ID]" in workflow
    assert 'CHECKPOINT_SCOPE="all"' in download
    assert "--checkpointScope|--checkpoint-scope" in download
    assert "checkpoint-download-names" in download
    assert "BEST_CHECKPOINT=\"${CHECKPOINT_NAMES%%$'\\n'*}\"" in download
    assert "checkpoint_scope=%s checkpoint_count=%s" in download
    assert "list-objects-v2" not in download


def test_checkpoint_download_names_follow_validated_retention_manifest(
    tmp_path: Path,
) -> None:
    run_id = "download-contract-run"
    contract_digest = "a" * 64
    transaction_id = "b" * 32
    checkpoints = [
        {
            "path": "checkpoint-000020",
            "global_step": 20,
            "value": 0.4,
            "rank": 1,
        },
        {
            "path": "checkpoint-000010",
            "global_step": 10,
            "value": 0.5,
            "rank": 2,
        },
    ]
    leaderboard = {
        "schema_version": "3.0",
        "run_id": run_id,
        "run_key": run_id,
        "training_resume_contract_sha256": contract_digest,
        "selection_source": "validation",
        "monitor": "primary_5d/selection_score",
        "mode": "min",
        "save_top_k": 2,
        "transaction_id": transaction_id,
        "best_checkpoint": "checkpoint-000020",
        "checkpoints": checkpoints,
    }
    pointer = {
        "schema_version": "3.0",
        "run_id": run_id,
        "run_key": run_id,
        "training_resume_contract_sha256": contract_digest,
        "selection_source": "validation",
        "monitor": "primary_5d/selection_score",
        "mode": "min",
        "value": 0.4,
        "transaction_id": transaction_id,
        "path": "checkpoint-000020",
    }
    leaderboard_path = tmp_path / "checkpoint-leaderboard.json"
    pointer_path = tmp_path / "best-checkpoint.json"
    leaderboard_path.write_text(json.dumps(leaderboard), encoding="utf-8")
    pointer_path.write_text(json.dumps(pointer), encoding="utf-8")

    for scope, expected in (
        ("all", ["checkpoint-000020", "checkpoint-000010"]),
        ("best", ["checkpoint-000020"]),
    ):
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/runpod_readiness.py"),
                "checkpoint-download-names",
                "--leaderboard",
                str(leaderboard_path),
                "--pointer",
                str(pointer_path),
                "--run-id",
                run_id,
                "--scope",
                scope,
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.splitlines() == expected

    pointer["transaction_id"] = "c" * 32
    pointer_path.write_text(json.dumps(pointer), encoding="utf-8")
    mismatch = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/runpod_readiness.py"),
            "checkpoint-download-names",
            "--leaderboard",
            str(leaderboard_path),
            "--pointer",
            str(pointer_path),
            "--run-id",
            run_id,
            "--scope",
            "all",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert mismatch.returncode == 2
    assert "transactions disagree" in mismatch.stderr


def test_wandb_contract_logs_every_optimizer_step_and_validation_metrics() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    config = (ROOT / "src/stock_forecasting/config.py").read_text(encoding="utf-8")
    training = (ROOT / "src/stock_forecasting/training.py").read_text(encoding="utf-8")
    validation = (ROOT / "src/stock_forecasting/validation_benchmark.py").read_text(
        encoding="utf-8"
    )
    tracking = (ROOT / "src/stock_forecasting/tracking.py").read_text(encoding="utf-8")
    wandb_sync = (ROOT / "src/stock_forecasting/cli/sync_wandb.py").read_text(encoding="utf-8")
    wandb_sync_script = (ROOT / "scripts/runpod_wandb_sync.sh").read_text(encoding="utf-8")

    assert '"wandb==0.28.0"' in pyproject
    assert "log_every_steps: Literal[1] = 1" in config
    assert '"train/loss": optimizer_step_loss' in training
    assert "step=global_step" in training
    assert '_flatten_metrics(validation_metrics, "validation")' in training
    assert 'run.define_metric("trainer/global_step")' in tracking
    assert 'run.define_metric("train/*", step_metric="trainer/global_step")' in tracking
    assert 'run.define_metric("validation/*", step_metric="trainer/global_step")' in tracking
    assert 'backend_payload["trainer/global_step"] = step' in tracking
    assert '"benchmark_validation"' in validation
    assert 'validation_step_key = "benchmark_validation/global_step"' in validation
    assert 'run.define_metric("benchmark_validation/*"' in validation
    assert "validation_history[validation_step_key] = validation_step" in validation
    assert "config.as_dict()" in tracking
    assert '"offline_pending"' in tracking
    assert '"sync_failed"' in wandb_sync
    assert '"online_running"' in wandb_sync
    assert '"sync"' in wandb_sync
    assert '"--include-offline"' in wandb_sync
    assert '"--legacy"' in wandb_sync
    assert "trap terminate_sync_pod EXIT" in wandb_sync_script


def test_eodhd_secret_is_cpu_only() -> None:
    reexec = (ROOT / "scripts/runpod_reexec_with_pid1_env.py").read_text(encoding="utf-8")
    create_cpu = (ROOT / "scripts/create_runpod_cpu_pod.sh").read_text(encoding="utf-8")
    create_gpu = (ROOT / "scripts/create_runpod_pod.sh").read_text(encoding="utf-8")

    assert "EODHD_API_TOKEN" in create_cpu
    assert "EODHD_API_TOKEN" not in create_gpu
    assert 'imported.pop("EODHD_API_TOKEN", None)' in reexec


def test_stage_configs_have_identical_architecture_digest() -> None:
    stage1 = ExperimentConfig.from_yaml(ROOT / "configs/stage1_kronos_base_lora.yaml")
    stage2 = ExperimentConfig.from_yaml(ROOT / "configs/stage2_kronos_base_lora.yaml")
    assert stage1.model.architecture_digest() == stage2.model.architecture_digest()


def test_runpod_setup_and_stage_configs_pin_the_same_kronos_revision() -> None:
    stage1 = ExperimentConfig.from_yaml(ROOT / "configs/stage1_kronos_base_lora.yaml")
    stage2 = ExperimentConfig.from_yaml(ROOT / "configs/stage2_kronos_base_lora.yaml")
    setup = (ROOT / "scripts/setup_runpod_environment.sh").read_text(encoding="utf-8")
    revision = stage1.model.kronos_source_revision

    assert revision is not None
    assert stage2.model.kronos_source_revision == revision
    assert f'KRONOS_COMMIT="{revision}"' in setup


def test_stage_configs_pin_identical_hugging_face_weight_revisions() -> None:
    stage1 = ExperimentConfig.from_yaml(ROOT / "configs/stage1_kronos_base_lora.yaml")
    stage2 = ExperimentConfig.from_yaml(ROOT / "configs/stage2_kronos_base_lora.yaml")

    assert stage1.model.time_series_model_revision == ("2b554741eca47781b64468546e77fef3e85130e6")
    assert stage1.model.time_series_tokenizer_revision == (
        "0e0117387f39004a9016484a186a908917e22426"
    )
    assert stage2.model.time_series_model_revision == stage1.model.time_series_model_revision
    assert (
        stage2.model.time_series_tokenizer_revision == stage1.model.time_series_tokenizer_revision
    )


def test_stage_configs_preserve_stable_runpod_path_overrides() -> None:
    for name in ("stage1_kronos_base_lora.yaml", "stage2_kronos_base_lora.yaml"):
        config_text = (ROOT / "configs" / name).read_text(encoding="utf-8")

        assert "${DATA_ROOT:-/runpod-volume/data}" in config_text
        assert "${SAVED_MODEL_ROOT:-/runpod-volume/savedModel}" in config_text
        assert "${WANDB_DIR:-/runpod-volume}" in config_text
        assert "${LOG_ROOT:-/runpod-volume/logs}" in config_text


def test_training_code_has_no_network_provider_imports() -> None:
    training_files = (
        ROOT / "src/stock_forecasting/training.py",
        ROOT / "src/stock_forecasting/cli/train.py",
        ROOT / "src/stock_forecasting/validation_benchmark.py",
    )
    combined = "\n".join(path.read_text(encoding="utf-8") for path in training_files)

    assert "requests." not in combined
    assert "EODHDProvider" not in combined
    assert "TWSEProvider" not in combined
    assert "TPExProvider" not in combined
    assert "ingest_daily_ohlcv" not in combined
