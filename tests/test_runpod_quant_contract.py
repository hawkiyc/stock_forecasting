"""Static guardrails around the stable RunPod lifecycle shell."""

from __future__ import annotations

import hashlib
import json
import stat
import subprocess
import sys
from pathlib import Path

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


def test_runpod_entrypoint_scripts_remain_executable() -> None:
    entrypoints = (
        ROOT / "scripts/create_runpod_cpu_pod.sh",
        ROOT / "scripts/create_runpod_pod.sh",
        ROOT / "scripts/runpod_cpu_prepare.sh",
        ROOT / "scripts/runpod_cpu_finalize.sh",
        ROOT / "scripts/runpod_entrypoint.sh",
        ROOT / "scripts/runpod_train_then_validate.sh",
        ROOT / "scripts/runpod_validation.sh",
    )
    for path in entrypoints:
        assert path.stat().st_mode & stat.S_IXUSR, path


def test_dataset_lifecycle_accepts_matching_provider_wait_progress(
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
    progress_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "ohlcv-download-progress",
                "state": "waiting_for_provider",
                "identity": identity,
                "identity_sha256": identity_sha256,
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
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert payload["state"] == "waiting_for_provider"
    assert payload["progress_path"] == str(progress_path.resolve())


def test_runpod_entrypoints_use_only_two_quant_stages() -> None:
    relevant = (
        ROOT / "scripts/create_runpod_pod.sh",
        ROOT / "scripts/runpod_entrypoint.sh",
        ROOT / "scripts/runpod_train_then_validate.sh",
        ROOT / "scripts/runpod_validation.sh",
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
    validation = (ROOT / "scripts/runpod_validation.sh").read_text(encoding="utf-8")

    for script in (create_gpu, create_cpu):
        assert "RUNPOD_NETWORK_VOLUME_ID" in script
        assert "RUNPOD_DATACENTER_ID" in script
        assert "rest.runpod.io" in script
    assert "--dry-run" in sync
    assert "--apply" in sync
    assert "runpod_project_s3_ready" in sync
    assert "runpod_train_then_validate.sh" in entrypoint
    assert "terminate_runpod_after.sh" in entrypoint
    assert "lifecycle/stage1/validation.json" in validation


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
