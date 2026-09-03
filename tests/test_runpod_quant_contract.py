"""Static guardrails around the stable RunPod lifecycle shell."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from stock_forecasting.config import ExperimentConfig

ROOT = Path(__file__).resolve().parents[1]


def test_ruff_preserves_default_virtualenv_exclusions() -> None:
    with (ROOT / "pyproject.toml").open("rb") as stream:
        ruff_config = tomllib.load(stream)["tool"]["ruff"]

    assert "exclude" not in ruff_config
    assert ruff_config["extend-exclude"] == [
        "src/stock_forecasting/_vendor/kronos"
    ]


def test_runpod_stage_contract_reports_stage1_five_percent_cap() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/runpod_readiness.py"),
            "stage-contract",
            "--config",
            str(ROOT / "configs/stage1_kronos_base_lora.yaml"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["training_stage"] == "stage1"
    assert payload["train_fraction"] == pytest.approx(0.05)
    assert payload["max_samples"] == 500_000


def test_runpod_s3_capture_retry_discards_failed_partial_output(tmp_path: Path) -> None:
    counter_path = tmp_path / "capture-attempts.txt"
    fake_wrapper = tmp_path / "fake-s3-capture.sh"
    fake_wrapper.write_text(
        """#!/usr/bin/env bash
set -eu
attempt=0
if [[ -f \"${COUNTER_PATH}\" ]]; then
    attempt=\"$(cat \"${COUNTER_PATH}\")\"
fi
attempt=$((attempt + 1))
printf '%s\\n' \"${attempt}\" > \"${COUNTER_PATH}\"
if [[ ${attempt} -lt 3 ]]; then
    printf 'partial-%s' \"${attempt}\"
    exit 41
fi
printf '{\"state\":\"ready\"}'
""",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; runpod_s3_retry_capture "$2" "read marker" ignored',
            "bash",
            str(ROOT / "scripts/lib/runpod_s3_retry.sh"),
            str(fake_wrapper),
        ],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "COUNTER_PATH": str(counter_path),
            "RUNPOD_S3_RETRY_MAX_ATTEMPTS": "3",
            "RUNPOD_S3_RETRY_INITIAL_BACKOFF_SECONDS": "0",
            "RUNPOD_S3_RETRY_MAX_BACKOFF_SECONDS": "0",
        },
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == '{"state":"ready"}'
    assert counter_path.read_text(encoding="utf-8").strip() == "3"
    assert result.stderr.count("retrying in 0s") == 2


def test_runpod_s3_stdin_retry_replays_the_complete_payload(tmp_path: Path) -> None:
    counter_path = tmp_path / "stdin-attempts.txt"
    payloads_path = tmp_path / "payloads.txt"
    fake_wrapper = tmp_path / "fake-s3-stdin.sh"
    fake_wrapper.write_text(
        """#!/usr/bin/env bash
set -eu
payload=\"$(cat)\"
attempt=0
if [[ -f \"${COUNTER_PATH}\" ]]; then
    attempt=\"$(cat \"${COUNTER_PATH}\")\"
fi
attempt=$((attempt + 1))
printf '%s\\n' \"${attempt}\" > \"${COUNTER_PATH}\"
printf '%s\\n' \"${payload}\" >> \"${PAYLOADS_PATH}\"
if [[ ${attempt} -lt 2 ]]; then
    exit 42
fi
""",
        encoding="utf-8",
    )
    payload = '{"state":"ready"}'

    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; runpod_s3_retry_stdin "$2" "publish marker" "$3" ignored',
            "bash",
            str(ROOT / "scripts/lib/runpod_s3_retry.sh"),
            str(fake_wrapper),
            payload,
        ],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "COUNTER_PATH": str(counter_path),
            "PAYLOADS_PATH": str(payloads_path),
            "RUNPOD_S3_RETRY_MAX_ATTEMPTS": "2",
            "RUNPOD_S3_RETRY_INITIAL_BACKOFF_SECONDS": "0",
            "RUNPOD_S3_RETRY_MAX_BACKOFF_SECONDS": "0",
        },
    )

    assert result.returncode == 0, result.stderr
    assert payloads_path.read_text(encoding="utf-8").splitlines() == [payload, payload]
    assert result.stderr.count("retrying in 0s") == 1


def test_runpod_stage_contract_rejects_wrong_stage1_sample_cap(
    tmp_path: Path,
) -> None:
    config = tmp_path / "stage1.yaml"
    config.write_text(
        (ROOT / "configs/stage1_kronos_base_lora.yaml")
        .read_text(encoding="utf-8")
        .replace("max_samples: 500000", "max_samples: null"),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/runpod_readiness.py"),
            "stage-contract",
            "--config",
            str(config),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "stage and max_samples disagree" in result.stderr


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


def test_runpod_runtime_never_executes_git() -> None:
    shell_runtime = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((ROOT / "scripts").rglob("*.sh"))
    )
    python_runtime = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((ROOT / "src/stock_forecasting").rglob("*.py"))
        if "_vendor/kronos" not in path.as_posix()
    )

    for forbidden in (
        "git clone",
        "git checkout",
        "git rev-parse",
        "git remote",
        '["git",',
        "['git',",
    ):
        assert forbidden not in shell_runtime
        assert forbidden not in python_runtime


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


def test_cpu_preparation_lifecycle_accepts_matching_resumable_progress(
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
    marker = volume_root / "lifecycle" / "stage1" / "cpu-preparation.json"
    for state in (
        "waiting_for_provider",
        "waiting_for_budget",
        "waiting_for_resume",
        "waiting_for_preparation",
        "downloaded",
    ):
        progress_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "ohlcv-download-progress",
                    "state": "downloaded" if state == "waiting_for_preparation" else state,
                    "identity": identity,
                    "identity_sha256": identity_sha256,
                    "context": {"dataset_request_sha256": digest},
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
                "stage1-cpu-preparation",
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
                "context": {"dataset_request_sha256": digest},
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
    marker = volume_root / "lifecycle" / "stage1" / "cpu-preparation.json"

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
            "stage1-cpu-preparation",
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
                "context": {"dataset_request_sha256": digest},
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
    marker = volume_root / "lifecycle" / "stage1" / "cpu-preparation.json"
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
        "stage1-cpu-preparation",
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


@pytest.mark.parametrize(
    ("lifecycle_state", "progress_state"),
    (
        ("waiting_for_provider", "waiting_for_provider"),
        ("waiting_for_preparation", "downloaded"),
    ),
)
def test_tmux_resumable_cpu_preparation_validator_requires_matching_progress(
    tmp_path: Path,
    lifecycle_state: str,
    progress_state: str,
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
                "state": progress_state,
                "identity": identity,
                "context": {"dataset_request_sha256": digest},
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
    marker = volume_root / "lifecycle" / "stage1" / "cpu-preparation.json"
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
            "stage1-cpu-preparation",
            "--state",
            lifecycle_state,
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
            "resumable-cpu-preparation-lifecycle",
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
    assert validate_result.stdout.strip() == lifecycle_state


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
    assert 'source "${SCRIPT_DIR}/lib/runpod_s3_retry.sh"' in sync
    assert 'bash "${S3_WRAPPER}"' not in sync
    for retry_helper in (
        "runpod_s3_retry_command",
        "runpod_s3_retry_capture",
        "runpod_s3_retry_stdin",
    ):
        assert retry_helper in sync
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
    assert 'CPU_PREPARATION_KEY="lifecycle/stage1/cpu-preparation.json"' in download
    assert '"${LOCAL_CPU_ROOT}/cpu-preparation.json"' in download
    assert 'LIFECYCLE_FIELDS="$(python3 - "${LOCAL_CPU_ROOT}/cpu-preparation.json"' in download
    assert 'DATASET_KEY="lifecycle/stage1/dataset.json"' in download
    assert "The immutable dataset marker is useful context" in download
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
    assert '"automatic_provider_checkpoint_reuse": True' in progress
    assert "RUNPOD_CPU_PREPARE_RESERVE_SECONDS" in prepare
    assert "ACQUISITION_DEADLINE_EPOCH" in prepare
    assert "RUNPOD_PROVIDER_MAX_BACKOFF_SECONDS" in prepare
    assert '--h-start "${FIN_TS_H_START}"' in prepare
    download_arguments = prepare.split("DOWNLOAD_ARGUMENTS=(", maxsplit=1)[1].split(
        ")", maxsplit=1
    )[0]
    assert "FIN_TS_H_START" not in download_arguments
    assert '--provider-checkpoint-root "${PROVIDER_CHECKPOINT_ROOT}"' in download_arguments
    assert 'PROVIDER_CHECKPOINT_ROOT="${DATA_ROOT}/provider-checkpoints"' in prepare
    assert 'error.get("provider_outcomes")' in status
    assert 'outcome.get("materialization_checkpoint")' in status
    assert '"checkpoint="' in status
    assert 'for key in ("category", "error_type", "operation", "item", "status_code")' in status
    assert 'ln "${RAW_STAGING}" "${RAW_FINAL}"' in prepare
    assert "fin-ts-verify-download" in prepare
    assert '--output "${BAR_STORE_FINAL}"' in prepare
    assert '--deadline-epoch-seconds "$((WORKFLOW_DEADLINE_EPOCH - 120))"' in prepare
    assert 'PREP_RESUMABLE_STATE=waiting_for_preparation' in prepare
    assert 'BAR_STORE_SUCCESS="${BAR_STORE_FINAL}/_SUCCESS.json"' in prepare
    assert (
        'DATASET_MANIFEST_STAGING="${DATA_ROOT}/.dataset-manifest-'
        '${LAUNCH_ID}.staging.json"' in prepare
    )
    assert 'DATASET_MANIFEST_STAGING="${DATA_STAGING_ROOT}' not in prepare
    assert 'mv "${DATASET_MANIFEST_STAGING}" "${DATASET_MANIFEST_FINAL}"' in prepare
    assert "PROCESSED_FINAL" not in prepare
    acquisition_branch = prepare.index("if [[ ${REUSE_DOWNLOADED_DATASET} -eq 0 ]]")
    eodhd_secret_check = prepare.index(
        "EODHD_API_TOKEN RunPod Secret is missing or was not resolved"
    )
    assert acquisition_branch < eodhd_secret_check < prepare.index("DOWNLOAD_ARGUMENTS=(")
    assert "incompatible-acquisition" in prepare
    assert "rebuilding from verified provider cache entries" in prepare
    assert "resumable-cpu-preparation-lifecycle" in tmux
    assert "${cpu_resumable_lifecycle_valid} -ne 1" in tmux
    assert "The CPU worker publishes the precise waiting state" in tmux
    assert "same_launch" in readiness
    for state in (
        "waiting_for_budget",
        "waiting_for_resume",
        "waiting_for_preparation",
        "downloaded",
    ):
        assert state in guard
    assert 'return "downloaded_active"' in readiness
    assert 'payload.get("exit_code") != 75' in readiness
    assert 'python3 "${RUNPOD_READINESS_HELPER}" guard-lifecycle-state' in guard
    assert 'expected_kind == "stage1-dataset" and state == "ready"' not in readiness


def test_cpu_execution_lifecycle_cannot_overwrite_immutable_dataset_readiness() -> None:
    prepare = (ROOT / "scripts/runpod_cpu_prepare.sh").read_text(encoding="utf-8")
    finalize = (ROOT / "scripts/runpod_cpu_finalize.sh").read_text(encoding="utf-8")
    tmux = (ROOT / "scripts/runpod_tmux_launch.sh").read_text(encoding="utf-8")
    create_cpu = (ROOT / "scripts/create_runpod_cpu_pod.sh").read_text(
        encoding="utf-8"
    )
    status = (ROOT / "scripts/show_runpod_status.sh").read_text(encoding="utf-8")

    assert (
        'CPU_PREPARATION_MARKER="${LIFECYCLE_ROOT}/stage1/cpu-preparation.json"'
        in prepare
    )
    lifecycle_writer = prepare.split(
        "write_lifecycle_state() {", maxsplit=1
    )[1].split("\n}", maxsplit=1)[0]
    assert '--output "${CPU_PREPARATION_MARKER}"' in lifecycle_writer
    assert "--kind stage1-cpu-preparation" in lifecycle_writer
    assert "DATASET_MARKER" not in lifecycle_writer
    assert 'archive_dataset_readiness "${reason}"' in prepare
    assert "archive_stale_dataset_readiness" in prepare
    assert "archive_dataset_readiness stale-selection" in prepare
    for writer in (prepare, finalize):
        assert 'DATASET_MARKER_STAGING=' in writer
        assert 'fin-ts-verify-stage1-data' in writer
        assert '--verify-only > "${DATASET_MARKER_STAGING}"' in writer
        assert '--marker "${DATASET_MARKER_STAGING}"' in writer
        assert 'mv "${DATASET_MARKER_STAGING}" "${DATASET_MARKER}"' in writer
        assert '--output "${DATASET_MARKER}"' not in writer

    prepare_readiness_writer = prepare.split(
        "publish_dataset_readiness() {", maxsplit=1
    )[1].split("\n}", maxsplit=1)[0]
    prepare_verify_index = prepare_readiness_writer.index(
        '--verify-only > "${DATASET_MARKER_STAGING}"'
    )
    prepare_bind_index = prepare_readiness_writer.index(
        '--marker "${DATASET_MARKER_STAGING}"'
    )
    prepare_publish_index = prepare_readiness_writer.index(
        'mv "${DATASET_MARKER_STAGING}" "${DATASET_MARKER}"'
    )
    assert prepare_verify_index < prepare_bind_index < prepare_publish_index

    finalizer_contract_validator = finalize.split(
        "validate_existing_dataset_contract() {", maxsplit=1
    )[1].split("\n}", maxsplit=1)[0]
    assert "check-dataset" in finalizer_contract_validator
    assert '--marker "${DATASET_MARKER}"' in finalizer_contract_validator
    assert "--network-volume-root" not in finalizer_contract_validator

    finalizer_stager = finalize.split(
        "stage_dataset_readiness() {", maxsplit=1
    )[1].split("\n}", maxsplit=1)[0]
    assert 'fin-ts-verify-stage1-data' in finalizer_stager
    assert '--dataset-manifest "${DATASET_MANIFEST}"' in finalizer_stager
    assert '--model-manifest "${MODEL_MANIFEST}"' in finalizer_stager
    assert '--verify-only > "${DATASET_MARKER_STAGING}"' in finalizer_stager

    finalizer_publisher = finalize.split(
        "publish_dataset_readiness() {", maxsplit=1
    )[1].split("\n}", maxsplit=1)[0]
    finalizer_bind_index = finalizer_publisher.index(
        '--marker "${DATASET_MARKER_STAGING}"'
    )
    finalizer_strict_check_index = finalizer_publisher.index("check-dataset")
    finalizer_publish_index = finalizer_publisher.index(
        'mv "${DATASET_MARKER_STAGING}" "${DATASET_MARKER}"'
    )
    assert '--network-volume-root "${NETWORK_VOLUME_ROOT}"' in finalizer_publisher
    assert (
        finalizer_bind_index
        < finalizer_strict_check_index
        < finalizer_publish_index
    )

    finalizer_main = finalize.split(
        "write_finalization_state finalizing", maxsplit=1
    )[1]
    assert finalizer_main.index("validate_existing_dataset_contract") < (
        finalizer_main.index("prefetch_hf_models.sh")
    )
    assert finalizer_main.index("prefetch_hf_models.sh") < finalizer_main.index(
        "stage_dataset_readiness"
    )
    assert finalizer_main.index("stage_dataset_readiness") < finalizer_main.index(
        '"${POETRY_BIN}" run pytest'
    )
    assert finalizer_main.index('"${POETRY_BIN}" run pytest') < (
        finalizer_main.index("publish_dataset_readiness")
    )
    assert "lifecycle/stage1/cpu-preparation.json" in tmux
    assert "stage1-cpu-preparation" in tmux
    assert (
        "RUNPOD_CPU_GUARD_LIFECYCLE_KEY=lifecycle/stage1/cpu-preparation.json"
        in create_cpu
    )
    assert "lifecycle/stage1/cpu-preparation.json cpu_prepare" in status
    assert 'verify-marker' in status
    assert 'reported_state=ready' in status
    assert 'reason=selection_contract' in status


def test_gpu_gate_verifies_lazy_bar_store_artifacts() -> None:
    gate = (ROOT / "scripts/verify_runpod_stage_readiness.sh").read_text(
        encoding="utf-8"
    )

    for artifact in ("bar_store_manifest", "symbol_index", "cutoff_ranges"):
        assert f"verify_remote_size {artifact}" in gate
    assert "verify_remote_size processed" not in gate


def test_checkpoint_contract_persists_runtime_label_scales() -> None:
    checkpointing = (ROOT / "src/stock_forecasting/checkpointing.py").read_text(
        encoding="utf-8"
    )
    contract = (ROOT / "src/stock_forecasting/run_contract.py").read_text(
        encoding="utf-8"
    )
    readiness = (ROOT / "scripts/runpod_readiness.py").read_text(encoding="utf-8")

    assert 'CHECKPOINT_ARTIFACT_SCHEMA_VERSION = "4.0"' in contract
    assert '"runtime_robust_scales": _model_runtime_robust_scales(model)' in checkpointing
    assert "_restore_runtime_robust_scales(" in checkpointing
    assert "_validate_runtime_robust_scales(trainer_state)" in readiness


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
    assert "Time reserved for data cleaning/bar-store construction [auto]:" in combined
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
    assert "savedModel/${RUN_ID}/completion-result/" in download
    assert "training-result.json" in download
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
        "schema_version": "4.0",
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
        "schema_version": "4.0",
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


def test_wandb_contract_logs_dynamic_epoch_points_and_validation_metrics() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    config = (ROOT / "src/stock_forecasting/config.py").read_text(encoding="utf-8")
    local_config = (ROOT / "configs/local_mock.yaml").read_text(encoding="utf-8")
    stage1_config = (ROOT / "configs/stage1_kronos_base_lora.yaml").read_text(
        encoding="utf-8"
    )
    stage2_config = (ROOT / "configs/stage2_kronos_base_lora.yaml").read_text(
        encoding="utf-8"
    )
    create_pod = (ROOT / "scripts/create_runpod_pod.sh").read_text(encoding="utf-8")
    training = (ROOT / "src/stock_forecasting/training.py").read_text(encoding="utf-8")
    validation = (ROOT / "src/stock_forecasting/validation_benchmark.py").read_text(
        encoding="utf-8"
    )
    tracking = (ROOT / "src/stock_forecasting/tracking.py").read_text(encoding="utf-8")
    wandb_sync = (ROOT / "src/stock_forecasting/cli/sync_wandb.py").read_text(encoding="utf-8")
    wandb_sync_script = (ROOT / "scripts/runpod_wandb_sync.sh").read_text(encoding="utf-8")

    assert '"wandb==0.28.0"' in pyproject
    assert 'project: str = "stock_forecasting"' in config
    assert "project: stock_forecasting" in local_config
    assert "project: ${WANDB_PROJECT:-stock_forecasting}" in stage1_config
    assert "project: ${WANDB_PROJECT:-stock_forecasting}" in stage2_config
    assert 'WANDB_PROJECT="${WANDB_PROJECT:-stock_forecasting}"' in create_pod
    assert "loss_log_points_per_epoch" in config
    assert "epoch_loss_logging_steps(" in training
    assert '"train/loss": logged_loss' in training
    assert "step=global_step" in training
    assert "last_validation_flat_metrics = _flatten_metrics(validation_metrics)" in training
    assert 'f"validation/{key}": value' in training
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


def test_runpod_training_auto_tunes_multiprocess_data_loading() -> None:
    entrypoint = (ROOT / "scripts/runpod_entrypoint.sh").read_text(encoding="utf-8")
    reexec = (ROOT / "scripts/runpod_reexec_with_pid1_env.py").read_text(encoding="utf-8")
    training = (ROOT / "src/stock_forecasting/training.py").read_text(encoding="utf-8")

    assert 'FIN_TS_DATALOADER_WORKERS="${FIN_TS_DATALOADER_WORKERS:-auto}"' in entrypoint
    assert '"FIN_TS_DATALOADER_WORKERS"' in reexec
    assert "DATALOADER_AUTO_MAX_WORKERS = 32" in training
    assert "resolve_runtime_batch_plan(" in training
    assert "plan_runtime_prefetch(" in training
    assert "iter_device_batches(" in training
    assert "DATALOADER_SELECTION_BLOCK_SIZE = 128" in training
    assert '"dataloader_worker_plan": worker_plan.as_dict()' in training
    assert "resolve_runtime_robust_scales(" in training
    assert '"parallel_backend": "pytorch_dataloader_processes"' in training
    assert "window_materialized" not in training


def test_eodhd_secret_is_cpu_only() -> None:
    reexec = (ROOT / "scripts/runpod_reexec_with_pid1_env.py").read_text(encoding="utf-8")
    create_cpu = (ROOT / "scripts/create_runpod_cpu_pod.sh").read_text(encoding="utf-8")
    create_gpu = (ROOT / "scripts/create_runpod_pod.sh").read_text(encoding="utf-8")

    assert "EODHD_API_TOKEN" in create_cpu
    assert "EODHD_API_TOKEN" not in create_gpu
    assert 'imported.pop("EODHD_API_TOKEN", None)' in reexec


def test_tpex_cloud_run_relay_is_closed_cpu_only_and_warmed_before_download() -> None:
    relay = (ROOT / "cloudrun/tpex-relay/src/relay.mjs").read_text(encoding="utf-8")
    server = (ROOT / "cloudrun/tpex-relay/src/server.mjs").read_text(encoding="utf-8")
    package = (ROOT / "cloudrun/tpex-relay/package.json").read_text(encoding="utf-8")
    deploy_script = (ROOT / "scripts/deploy_tpex_cloud_run_relay.sh").read_text(
        encoding="utf-8"
    )
    secret_creator = (
        ROOT / "scripts/create_runpod_tpex_proxy_secret.py"
    ).read_text(encoding="utf-8")
    workflow = (ROOT / "scripts/runpod_workflow.sh").read_text(encoding="utf-8")
    sync = (ROOT / "scripts/sync_project_to_runpod_volume.sh").read_text(
        encoding="utf-8"
    )
    create_cpu = (ROOT / "scripts/create_runpod_cpu_pod.sh").read_text(
        encoding="utf-8"
    )
    create_gpu = (ROOT / "scripts/create_runpod_pod.sh").read_text(encoding="utf-8")
    reexec = (ROOT / "scripts/runpod_reexec_with_pid1_env.py").read_text(
        encoding="utf-8"
    )
    verifier = (ROOT / "scripts/verify_tpex_cloud_run_relay.sh").read_text(
        encoding="utf-8"
    )
    warmup = (ROOT / "scripts/warm_tpex_cloud_run_relay.sh").read_text(
        encoding="utf-8"
    )
    cpu_prepare = (ROOT / "scripts/runpod_cpu_prepare.sh").read_text(encoding="utf-8")
    dotenv_helper = (ROOT / "scripts/update_runpod_env.py").read_text(encoding="utf-8")
    http_client = (
        ROOT / "src/stock_forecasting/data/providers/http.py"
    ).read_text(encoding="utf-8")

    assert 'const UPSTREAM_ORIGIN = "https://www.tpex.org.tw"' in relay
    for path in (
        "/www/zh-tw/afterTrading/dailyQuotes",
        "/www/zh-tw/bulletin/exDailyQ",
        "/www/zh-tw/indexInfo/ROE",
        "/www/zh-tw/indexInfo/inx",
    ):
        assert path in relay
        assert path in verifier
    assert 'request.method !== "GET"' in relay
    assert 'AUTHENTICATION_HEADER = "X-TPEX-Relay-Token"' in relay
    assert 'requestUrl.pathname === "/_internal/warmup"' in relay
    assert "timeoutSignalFactory" in relay
    assert "UPSTREAM_TIMEOUT_MS = 30_000" in relay
    assert "MAX_UPSTREAM_BODY_BYTES = 16 * 1024 * 1024" in relay
    assert 'redirect: "manual"' in relay
    assert 'cache: "no-store"' in relay
    assert "MAX_UPSTREAM_REDIRECTS = 3" in relay
    assert 'redirected.origin !== UPSTREAM_ORIGIN' in relay
    assert 'error: "tpex_upstream_redirect_limit_exceeded"' in relay
    assert 'error: "tpex_upstream_redirect_loop"' in relay
    assert "visitedStates.has(redirectedState)" in relay
    assert 'const RELAY_VERSION = "1"' in relay
    assert 'typeof headers.getSetCookie === "function"' in relay
    assert 'headers.set("Cookie"' in relay
    assert "MAX_UPSTREAM_REDIRECT_COOKIES = 8" in relay
    assert '"X-TPEX-Relay-Region"' in relay
    assert '"X-TPEX-Upstream-Redirect-Cookies"' in relay
    assert 'process.env.TPEX_PROXY_SHARED_SECRET' in server
    assert '"node": "22.x.x"' in package
    assert '"gcp-build": "node --test test/*.test.mjs"' in package
    for boundary in (
        '--region="${GCP_CLOUD_RUN_REGION}"',
        "--execution-environment=gen2",
        "--allow-unauthenticated",
        "--cpu=1",
        "--memory=512Mi",
        "--concurrency=1",
        "--min=0",
        "--max=1",
        "--timeout=60s",
        "--cpu-throttling",
        "--cpu-boost",
    ):
        assert boundary in deploy_script
    assert 'GCP_CLOUD_RUN_REGION}" != "asia-east1"' in deploy_script
    assert "roles/run.builder" in deploy_script
    assert "roles/secretmanager.secretAccessor" in deploy_script
    assert 'secret_version}" =~ ^[1-9][0-9]*$' in deploy_script
    assert '--set-secrets="TPEX_PROXY_SHARED_SECRET=' in deploy_script
    assert "Cloud Run service deployment completed" in deploy_script
    assert "did not create/update the RunPod TPEx Secret" in deploy_script
    preflight_index = deploy_script.index("--check-access")
    relay_deploy_index = deploy_script.index('gcloud run deploy "${GCP_TPEX_RELAY_SERVICE}"')
    secret_create_index = deploy_script.rindex("create_runpod_tpex_proxy_secret.py")
    assert preflight_index < relay_deploy_index < secret_create_index
    assert 'query="query { myself { id } }"' in secret_creator
    assert 'RUNPOD_GRAPHQL_USER_AGENT = "stock-forecasting-runpod-control/0.1"' in (
        secret_creator
    )
    assert '"User-Agent": RUNPOD_GRAPHQL_USER_AGENT' in secret_creator
    assert '"error_code", "error_name", "error_category", "detail"' in secret_creator
    for action in ("configure", "deploy", "verify", "status"):
        assert action in workflow
    assert "tpex-relay|tpex-proxy" in workflow
    assert "cloudrun/tpex-relay/src/relay.mjs" in sync
    assert "cloudrun/tpex-relay/src/server.mjs" in sync
    assert (
        "GCP_TPEX_RELAY_SECRET|RUNPOD_EODHD_SECRET_NAME|RUNPOD_HF_SECRET_NAME|"
        "RUNPOD_TPEX_PROXY_SECRET_NAME|RUNPOD_WANDB_SECRET_NAME" in sync
    )
    assert '*API_KEY|*TOKEN|*SECRET*|*PASSWORD*' in sync
    assert '"TPEX_PROXY_TOKEN":"%s"' in create_cpu
    assert "workers.dev" not in create_cpu
    assert "run.app" in create_cpu
    assert "legacy workers.dev origin" in dotenv_helper
    assert 'GCP_CLOUD_RUN_REGION must be asia-east1' in dotenv_helper
    assert "TPEX_PROXY_TOKEN" not in create_gpu
    assert 'imported.pop("TPEX_PROXY_TOKEN", None)' in reexec
    assert "request_sha256, _identity = self._identity(endpoint=endpoint" in http_client
    assert "request_endpoint = self.transport.request_url(endpoint)" in http_client
    assert 'return {"X-TPEX-Relay-Token": self.token}' in http_client
    assert "VERIFY_MAX_ATTEMPTS=8" in verifier
    assert "__FIN_TS_HTTP_STATUS__" in verifier
    assert "__FIN_TS_RELAY_REGION__" in verifier
    assert "__FIN_TS_UPSTREAM_REDIRECT_COOKIES__" in verifier
    assert "retrying in %ss" in verifier
    assert '"${TPEX_PROXY_URL%/}/_internal/warmup"' in warmup
    assert "no TPEx upstream request was sent" in warmup
    warmup_index = cpu_prepare.index('warm_tpex_cloud_run_relay.sh"')
    download_index = cpu_prepare.index('run fin-ts-download "${DOWNLOAD_ARGUMENTS[@]}"')
    assert warmup_index < download_index


def test_stage_configs_have_identical_architecture_digest() -> None:
    stage1 = ExperimentConfig.from_yaml(ROOT / "configs/stage1_kronos_base_lora.yaml")
    stage2 = ExperimentConfig.from_yaml(ROOT / "configs/stage2_kronos_base_lora.yaml")
    assert stage1.model_architecture_digest() == stage2.model_architecture_digest()


def test_runpod_setup_uses_bundled_kronos_source_without_remote_git() -> None:
    stage1 = ExperimentConfig.from_yaml(ROOT / "configs/stage1_kronos_base_lora.yaml")
    stage2 = ExperimentConfig.from_yaml(ROOT / "configs/stage2_kronos_base_lora.yaml")
    setup = (ROOT / "scripts/setup_runpod_environment.sh").read_text(encoding="utf-8")
    sync = (ROOT / "scripts/sync_project_to_runpod_volume.sh").read_text(
        encoding="utf-8"
    )
    factory = (ROOT / "src/stock_forecasting/factory.py").read_text(encoding="utf-8")
    revision = stage1.model.kronos_source_revision

    assert revision is not None
    assert stage2.model.kronos_source_revision == revision
    assert stage1.model.kronos_source_root == stage2.model.kronos_source_root
    assert stage1.model.kronos_source_root.as_posix().endswith(
        "/third_party/Kronos"
    )
    assert "BUNDLED_KRONOS_ROOT" in setup
    assert "Would install the pinned bundled Kronos source" in setup
    assert "model/kronos.py" in setup
    assert 'append_manifest_file "src/stock_forecasting/_vendor/kronos/LICENSE"' in sync
    assert "git clone" not in setup
    assert "git -C" not in setup
    assert "command -v git" not in setup
    assert '["git",' not in factory


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
