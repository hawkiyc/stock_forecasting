"""Evaluation must consume the same runtime-plan format that training saves."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import stock_forecasting.cli.evaluate as evaluation_module
from stock_forecasting.config import ExperimentConfig
from stock_forecasting.training import (
    CanonicalTrainingSchedule,
    EarlyStoppingState,
    RuntimeBatchPlan,
    RuntimeExecutionPlan,
    RuntimeHardwareSnapshot,
    _training_progress,
    plan_dataloader_workers,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def evaluation_case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    config = ExperimentConfig.from_yaml(ROOT / "configs/local_mock.yaml")
    config.training.output_root = tmp_path
    checkpoint = tmp_path / "run-evaluation-contract" / "checkpoint-000005"
    gib = 1024**3
    batch_plan = RuntimeBatchPlan(
        source="checkpoint_fixture",
        training_batch_size=16,
        evaluation_batch_size=32,
        gradient_accumulation_steps=2,
        effective_batch_size=32,
        target_effective_batch_size=32,
        device_name="cpu",
        device_total_memory_bytes=0,
        optimizer_state_reserve_bytes=0,
        seconds_per_training_batch=None,
        training_probe=(),
        evaluation_probe=(),
    )
    execution_plan = RuntimeExecutionPlan(
        source="runtime_probe",
        replan_reason="fresh_training",
        hardware=RuntimeHardwareSnapshot(
            requested_gpu_id="",
            device_type="cpu",
            device_name="cpu",
            device_total_memory_bytes=0,
            compute_capability="",
            host_memory_capacity_bytes=64 * gib,
            host_memory_capacity_source="fixture",
            visible_cpu_count=8,
        ),
        batch_plan=batch_plan,
        worker_plan=plan_dataloader_workers(
            2,
            source="fixture",
            visible_cpu_count=8,
            available_memory_bytes=56 * gib,
        ),
        optimizer_steps_per_epoch=10,
        configured_optimizer_steps=10,
        resume_canonical_global_step=0,
        resume_runtime_global_step=0,
    )
    schedule = CanonicalTrainingSchedule(
        epochs=1,
        evaluations_per_epoch=1,
        optimizer_steps_per_epoch=10,
        configured_optimizer_steps=10,
    )
    # Use the real training serializer so a future format change reaches this test.
    state = json.loads(
        json.dumps(
            {
                "global_step": 5,
                "epoch": 0,
                "created_at": "2026-09-16T00:00:00+00:00",
                "training_progress": _training_progress(
                    processed_train_samples=160,
                    completed_epochs=0,
                    early_stopping=EarlyStoppingState(),
                    canonical_schedule=schedule,
                    runtime_execution_plan=execution_plan,
                ),
            }
        )
    )
    bundle = SimpleNamespace(model=Mock())
    loaders = (
        SimpleNamespace(sampler=range(10)),
        SimpleNamespace(sampler=range(3)),
        SimpleNamespace(sampler=range(5)),
    )
    stubs = {
        "run_preflight": Mock(),
        "set_global_seed": Mock(),
        "build_model_bundle": Mock(return_value=bundle),
        "resolve_checkpoint": Mock(return_value=checkpoint),
        "load_checkpoint": Mock(return_value=state),
        "build_evaluation_loader": Mock(
            side_effect=lambda *args, **kwargs: loaders[1 if kwargs["split"] == "validation" else 2]
        ),
        "evaluate_loader": Mock(return_value={"loss": 0.125}),
        "provenance_summary": Mock(return_value={"dataset_profile": "fixture"}),
    }
    for name, stub in stubs.items():
        monkeypatch.setattr(evaluation_module, name, stub)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    return SimpleNamespace(
        config=config,
        checkpoint=checkpoint,
        state=state,
        batch_plan=batch_plan,
        bundle=bundle,
        loaders=loaders,
        stubs=stubs,
    )


@pytest.mark.parametrize("split", ["validation", "test"])
@pytest.mark.parametrize("plan_format", ["current", "legacy"])
def test_evaluation_reads_saved_runtime_plan_without_rewriting_checkpoint(
    evaluation_case: SimpleNamespace,
    split: str,
    plan_format: str,
) -> None:
    case = evaluation_case
    progress = case.state["training_progress"]
    if plan_format == "legacy":
        case.state["training_progress"] = {
            "processed_train_samples": progress["processed_train_samples"],
            "completed_epochs": progress["completed_epochs"],
            "early_stopping": progress["early_stopping"],
            "configured_optimizer_steps": 10,
            "runtime_batch_plan": case.batch_plan.as_dict(),
        }
    else:
        assert "runtime_batch_plan" not in progress
        assert progress["runtime_execution_plan"]["batch_plan"] == case.batch_plan.as_dict()
    original = copy.deepcopy(case.state)

    result = evaluation_module.evaluate_checkpoint(case.config, case.checkpoint, split=split)

    selected_loader = case.loaders[1 if split == "validation" else 2]
    case.stubs["build_evaluation_loader"].assert_called_once_with(
        case.config,
        bundle=case.bundle,
        device=torch.device("cpu"),
        split=split,
    )
    case.stubs["evaluate_loader"].assert_called_once_with(
        case.bundle,
        selected_loader,
        case.config,
        torch.device("cpu"),
    )
    case.stubs["load_checkpoint"].assert_called_once_with(
        case.checkpoint,
        case.bundle.model,
        config=case.config,
    )
    case.bundle.model.eval.assert_called_once_with()
    assert result["run_id"] == "run-evaluation-contract"
    assert result["split"] == split
    assert result["samples"] == len(selected_loader.sampler)
    assert result["checkpoint"]["global_step"] == 5
    assert result["metrics"] == {"loss": 0.125}
    assert case.state == original


@pytest.mark.parametrize("missing_plan", [None, "absent"])
def test_current_progress_cannot_fall_back_to_a_legacy_batch_plan(
    evaluation_case: SimpleNamespace,
    missing_plan: str | None,
) -> None:
    case = evaluation_case
    progress = case.state["training_progress"]
    progress["runtime_batch_plan"] = case.batch_plan.as_dict()
    if missing_plan == "absent":
        del progress["runtime_execution_plan"]
    else:
        progress["runtime_execution_plan"] = None

    with pytest.raises(ValueError, match="no runtime execution plan"):
        evaluation_module.evaluate_checkpoint(case.config, case.checkpoint)
    case.stubs["build_evaluation_loader"].assert_not_called()
    case.stubs["evaluate_loader"].assert_not_called()


@pytest.mark.parametrize("invalid_plan", [None, {}, {"evaluation_batch_size": 0}])
def test_evaluation_rejects_invalid_nested_batch_plans(
    evaluation_case: SimpleNamespace,
    invalid_plan: object,
) -> None:
    case = evaluation_case
    case.state["training_progress"]["runtime_execution_plan"]["batch_plan"] = invalid_plan

    with pytest.raises(ValueError, match="runtime batch plan"):
        evaluation_module.evaluate_checkpoint(case.config, case.checkpoint)
    case.stubs["build_evaluation_loader"].assert_not_called()
    case.stubs["evaluate_loader"].assert_not_called()
