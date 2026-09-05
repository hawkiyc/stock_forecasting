"""Contracts for reusing numerical validation metrics from checkpoints."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from stock_forecasting.checkpoint_resume_migrations import CHECKPOINT_RETENTION_MIGRATIONS
from stock_forecasting.config import ExperimentConfig
from stock_forecasting.run_contract import (
    CHECKPOINT_ARTIFACT_SCHEMA_VERSION,
    training_resume_contract,
)
from stock_forecasting.training import _flatten_metrics
from stock_forecasting.validation_benchmark import (
    VALIDATION_BENCHMARK_SCHEMA_VERSION,
    ValidationBenchmark,
    _unflatten_validation_metrics,
    build_evaluation_contract,
    checkpoint_validation_snapshot,
)

ROOT = Path(__file__).resolve().parents[1]


def _canonical_digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validation_metrics() -> dict[str, object]:
    return {
        "loss": 0.25,
        "primary_5d": {
            "selection_score": 0.20,
            "normalized_pinball": 0.20,
        },
        "cross_sectional_5d": {
            "rank_ic_mean": 0.12,
            "net_long_short_sharpe": 0.75,
        },
        "per_horizon": {"5d": {"median_mae": 0.10}},
    }


def test_checkpoint_snapshot_round_trips_trainer_metric_namespace(
    tmp_path: Path,
) -> None:
    metrics = _validation_metrics()
    state = {
        "global_step": 200,
        "created_at": "2026-09-02T08:18:00+00:00",
        "metrics": _flatten_metrics(metrics),
    }
    (tmp_path / "trainer-state.json").write_text(
        json.dumps(state),
        encoding="utf-8",
    )

    snapshot = checkpoint_validation_snapshot(tmp_path)

    assert snapshot == {
        "metrics": metrics,
        "source": "checkpoint_validation_snapshot",
        "global_step": 200,
        "created_at": "2026-09-02T08:18:00+00:00",
    }


def test_checkpoint_snapshot_accepts_legacy_validation_prefix() -> None:
    metrics = _validation_metrics()

    assert _unflatten_validation_metrics(_flatten_metrics(metrics, "validation")) == metrics


def test_checkpoint_snapshot_rejects_mixed_metric_namespaces() -> None:
    metrics = _validation_metrics()
    mixed = {
        **_flatten_metrics(metrics),
        **_flatten_metrics(metrics, "validation"),
    }

    with pytest.raises(ValueError, match="mixes validation metric namespaces"):
        _unflatten_validation_metrics(mixed)


def test_checkpoint_snapshot_requires_primary_and_cross_sectional_metrics() -> None:
    with pytest.raises(ValueError, match="complete numerical validation metric snapshot"):
        _unflatten_validation_metrics({"primary_5d/selection_score": 0.2})


def _evaluation_contract(
    config: ExperimentConfig,
    checkpoint: Path,
) -> dict[str, Any]:
    for name in ("adapter.safetensors", "resolved-config.yaml", "trainer-state.json"):
        path = checkpoint / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name, encoding="utf-8")
    return build_evaluation_contract(
        config,
        run_id="run-validation-contract",
        checkpoint=checkpoint,
        training_resume_contract_sha256="a" * 64,
        dataset_artifacts={"sha256": "b" * 64},
        models=["always_buy", "kronos_full"],
        seeds=[42],
    )


def test_evaluation_contract_ignores_pod_runtime_and_tracking_settings(
    tmp_path: Path,
) -> None:
    first = ExperimentConfig.from_yaml(ROOT / "configs/local_mock.yaml")
    second = first.model_copy(deep=True)
    second.runtime.max_runtime_seconds = first.runtime.max_runtime_seconds * 2
    second.runtime.termination_max_attempts += 1
    second.wandb.project = "different-tracking-project"
    second.wandb.mode = "offline"

    first_contract = _evaluation_contract(first, tmp_path / "checkpoint")
    second_contract = _evaluation_contract(second, tmp_path / "checkpoint")

    assert first_contract == second_contract


def test_evaluation_contract_keeps_numerical_validation_settings(
    tmp_path: Path,
) -> None:
    first = ExperimentConfig.from_yaml(ROOT / "configs/local_mock.yaml")
    second = first.model_copy(deep=True)
    second.validation.neural_epochs += 1

    first_contract = _evaluation_contract(first, tmp_path / "checkpoint")
    second_contract = _evaluation_contract(second, tmp_path / "checkpoint")

    assert first_contract != second_contract


def test_validation_setup_accepts_an_allowlisted_training_code_migration(
    tmp_path: Path,
) -> None:
    config = ExperimentConfig.from_yaml(ROOT / "configs/local_mock.yaml")
    run_id = "run-validation-migrated-code"
    config.training.output_root = tmp_path / "savedModel"
    config.validation.output_root = tmp_path / "evaluations"
    run_directory = config.training.output_root / run_id
    checkpoint = config.training.output_root / run_id / "checkpoint-000001"
    checkpoint.mkdir(parents=True)
    current_contract = training_resume_contract(config)
    migration = CHECKPOINT_RETENTION_MIGRATIONS[0]
    assert {
        path: current_contract["training_implementation"]["files"][path]
        for path in migration["to_files"]
    } == migration["to_files"]
    stored_contract = copy.deepcopy(current_contract)
    stored_files = stored_contract["training_implementation"]["files"]
    stored_files.update(migration["from_files"])
    stored_contract["training_implementation"]["sha256"] = _canonical_digest(stored_files)
    stored_training_digest = _canonical_digest(stored_contract)
    (run_directory / "run-manifest.json").write_text(
        json.dumps(
            {
                "schema_version": CHECKPOINT_ARTIFACT_SCHEMA_VERSION,
                "run_id": run_id,
                "run_key": run_id,
                "training_resume_contract": stored_contract,
                "training_resume_contract_sha256": stored_training_digest,
            }
        ),
        encoding="utf-8",
    )
    config.save_resolved(run_directory / "resolved-config.yaml")
    config.save_resolved(checkpoint / "resolved-config.yaml")
    (checkpoint / "adapter.safetensors").write_bytes(b"adapter")
    (checkpoint / "trainer-state.json").write_text(
        json.dumps(
            {
                "schema_version": CHECKPOINT_ARTIFACT_SCHEMA_VERSION,
                "run_id": run_id,
                "run_key": run_id,
                "training_resume_contract_sha256": stored_training_digest,
            }
        ),
        encoding="utf-8",
    )

    benchmark = ValidationBenchmark(
        config,
        run_id=run_id,
        checkpoint=checkpoint,
        output=config.validation.output_root / run_id / "validation-benchmark.json",
        lifecycle=tmp_path / "lifecycle" / "stage1" / "validation.json",
        models=["always_buy"],
        seeds=[42],
        resume=True,
        recompute_full_model=False,
    )

    inputs = benchmark.evaluation_contract["inputs"]
    assert inputs["training_resume_contract_sha256"] == stored_training_digest
    assert inputs["dataset_artifacts"] == current_contract["dataset_artifacts"]


def test_resume_normalizes_legacy_contract_without_rerunning_completed_results(
    tmp_path: Path,
) -> None:
    config = ExperimentConfig.from_yaml(ROOT / "configs/local_mock.yaml")
    checkpoint = tmp_path / "checkpoint"
    current_contract = _evaluation_contract(config, checkpoint)
    legacy_contract = json.loads(json.dumps(current_contract))
    legacy_contract["version"] = "5.0"
    legacy_contract["inputs"]["config"] = config.as_dict()
    del legacy_contract["inputs"]["validation_numerical_config"]
    legacy_contract["digest"] = "legacy-full-config-digest"
    output = tmp_path / "validation-benchmark.json"
    output.write_text(
        json.dumps(
            {
                "schema_version": VALIDATION_BENCHMARK_SCHEMA_VERSION,
                "evaluation_contract": legacy_contract,
                "state": "ready",
                "run_id": "run-validation-contract",
                "checkpoint": str(checkpoint),
                "models": {
                    "always_buy": {"state": "complete"},
                    "kronos_full": {"state": "complete"},
                },
            }
        ),
        encoding="utf-8",
    )
    benchmark = object.__new__(ValidationBenchmark)
    benchmark.resume = True
    benchmark.output = output
    benchmark.evaluation_contract = current_contract
    benchmark.run_id = "run-validation-contract"
    benchmark.checkpoint = checkpoint

    resumed = benchmark._initial_payload()

    assert resumed["state"] == "running"
    assert resumed["evaluation_contract"] == current_contract
    assert "resume_contract_normalized_at" in resumed
    assert resumed["models"]["always_buy"]["state"] == "complete"
