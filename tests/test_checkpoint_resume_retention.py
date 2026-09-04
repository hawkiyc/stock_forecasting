"""Checkpoint retention and interrupted-training resume contracts."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR

from stock_forecasting.checkpoint_resume_migrations import CHECKPOINT_RETENTION_MIGRATIONS
from stock_forecasting.checkpointing import (
    CHECKPOINT_ARTIFACT_SCHEMA_VERSION,
    TEMP_CHECKPOINT_POINTER,
    latest_resume_checkpoint,
    save_ranked_checkpoint,
    save_training_completion_result,
    validate_checkpoint_selection,
)
from stock_forecasting.config import ExperimentConfig
from stock_forecasting.run_contract import (
    compatible_training_resume_contract_digest,
    training_resume_contract,
    training_resume_contract_fingerprint,
)

ROOT = Path(__file__).resolve().parents[1]


def _load_remote_preflight() -> ModuleType:
    scripts = ROOT / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location(
            "checkpoint_resume_remote_preflight",
            scripts / "runpod_remote_checkpoint_preflight.py",
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(scripts))


REMOTE_PREFLIGHT = _load_remote_preflight()


class _CheckpointModel(nn.Module):
    def __init__(self, horizon_count: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.alpha_head = nn.Module()
        self.alpha_head.register_buffer("robust_scales", torch.ones(horizon_count))


def _digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _checkpoint_run(
    tmp_path: Path, name: str = "checkpoint-retention-run"
) -> tuple[
    ExperimentConfig,
    Path,
    nn.Module,
    torch.optim.Optimizer,
    LambdaLR,
]:
    config = ExperimentConfig.from_yaml(ROOT / "configs/local_mock.yaml")
    run_directory = tmp_path / name
    run_directory.mkdir()
    contract, contract_digest = training_resume_contract_fingerprint(config)
    (run_directory / "run-manifest.json").write_text(
        json.dumps(
            {
                "schema_version": CHECKPOINT_ARTIFACT_SCHEMA_VERSION,
                "run_id": name,
                "run_key": name,
                "training_resume_contract": contract,
                "training_resume_contract_sha256": contract_digest,
            }
        ),
        encoding="utf-8",
    )
    config.save_resolved(run_directory / "resolved-config.yaml")
    model = _CheckpointModel(len(config.data.alpha_horizons))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = LambdaLR(optimizer, lambda _step: 1.0)
    return config, run_directory, model, optimizer, scheduler


def _save(
    config: ExperimentConfig,
    run_directory: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    *,
    step: int,
    value: float,
    save_top_k: int = 5,
) -> Path:
    checkpoint, _ranking = save_ranked_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        run_directory=run_directory,
        global_step=step,
        epoch=0,
        batch_index=step,
        selection_metric_name="primary_5d/selection_score",
        selection_metric_value=value,
        selection_metric_mode="min",
        save_top_k=save_top_k,
        metrics={"primary_5d/selection_score": value},
    )
    assert checkpoint is not None
    return checkpoint


def test_latest_non_top_five_checkpoint_is_kept_only_for_resume(tmp_path: Path) -> None:
    config, run_directory, model, optimizer, scheduler = _checkpoint_run(tmp_path)
    latest_ranked: Path | None = None
    for step, value in ((10, 1.0), (20, 2.0), (30, 3.0), (40, 4.0), (50, 5.0)):
        latest_ranked = _save(
            config,
            run_directory,
            model,
            optimizer,
            scheduler,
            step=step,
            value=value,
        )
    assert latest_ranked is not None
    assert latest_resume_checkpoint(run_directory) == latest_ranked

    temporary = _save(
        config,
        run_directory,
        model,
        optimizer,
        scheduler,
        step=60,
        value=6.0,
    )
    pointer = json.loads((run_directory / TEMP_CHECKPOINT_POINTER).read_text())
    leaderboard = json.loads((run_directory / "checkpoint-leaderboard.json").read_text())
    assert pointer["path"] == temporary.name == "checkpoint-000060"
    assert [row["path"] for row in leaderboard["checkpoints"]] == [
        "checkpoint-000010",
        "checkpoint-000020",
        "checkpoint-000030",
        "checkpoint-000040",
        "checkpoint-000050",
    ]
    assert latest_resume_checkpoint(run_directory) == temporary
    validate_checkpoint_selection(
        run_directory,
        retained_checkpoint=temporary.name,
        require_latest_step=True,
    )

    newer_temporary = _save(
        config,
        run_directory,
        model,
        optimizer,
        scheduler,
        step=70,
        value=7.0,
    )
    assert not temporary.exists()
    assert latest_resume_checkpoint(run_directory) == newer_temporary

    ranked = _save(
        config,
        run_directory,
        model,
        optimizer,
        scheduler,
        step=80,
        value=0.5,
    )
    leaderboard = json.loads((run_directory / "checkpoint-leaderboard.json").read_text())
    assert [row["path"] for row in leaderboard["checkpoints"]] == [
        "checkpoint-000080",
        "checkpoint-000010",
        "checkpoint-000020",
        "checkpoint-000030",
        "checkpoint-000040",
    ]
    assert not (run_directory / TEMP_CHECKPOINT_POINTER).exists()
    assert not newer_temporary.exists()
    assert not (run_directory / "checkpoint-000050").exists()
    assert latest_resume_checkpoint(run_directory) == ranked


@pytest.mark.parametrize("stop_reason", ["epochs_completed", "early_stopping"])
def test_training_completion_removes_temporary_checkpoint(
    tmp_path: Path,
    stop_reason: str,
) -> None:
    config, run_directory, model, optimizer, scheduler = _checkpoint_run(
        tmp_path,
        name=f"completion-{stop_reason}",
    )
    retained = _save(
        config,
        run_directory,
        model,
        optimizer,
        scheduler,
        step=10,
        value=1.0,
        save_top_k=1,
    )
    temporary = _save(
        config,
        run_directory,
        model,
        optimizer,
        scheduler,
        step=20,
        value=2.0,
        save_top_k=1,
    )

    save_training_completion_result(
        model=model,
        config=config,
        run_directory=run_directory,
        global_step=20,
        completed_epochs=1,
        processed_train_samples=20,
        planned_train_samples=20,
        validation_evaluations=2,
        stop_reason=stop_reason,
        early_stopping_state={"triggered": stop_reason == "early_stopping"},
        metrics={"primary_5d/selection_score": 2.0},
    )

    assert retained.is_dir()
    assert not temporary.exists()
    assert not (run_directory / TEMP_CHECKPOINT_POINTER).exists()


def test_remote_resume_prefers_only_a_newer_temporary_checkpoint() -> None:
    retained = [
        {"path": "checkpoint-000080", "global_step": 80},
        {"path": "checkpoint-000100", "global_step": 100},
    ]
    newer = {"path": "checkpoint-000120", "global_step": 120}
    stale = {"path": "checkpoint-000090", "global_step": 90}

    assert REMOTE_PREFLIGHT._latest_resume_checkpoint_row(retained, newer) is newer
    assert REMOTE_PREFLIGHT._latest_resume_checkpoint_row(retained, stale) is retained[1]
    assert REMOTE_PREFLIGHT._latest_resume_checkpoint_row(retained, None) is retained[1]


def test_current_interrupted_run_contract_allows_only_the_retention_migration() -> None:
    config = ExperimentConfig.from_yaml(ROOT / "configs/local_mock.yaml")
    current = training_resume_contract(config)
    migration = CHECKPOINT_RETENTION_MIGRATIONS[0]
    current_files = current["training_implementation"]["files"]
    assert {path: current_files[path] for path in migration["to_files"]} == migration["to_files"]

    stored = copy.deepcopy(current)
    stored_files = stored["training_implementation"]["files"]
    stored_files.update(migration["from_files"])
    stored["training_implementation"]["sha256"] = _digest(stored_files)
    stored_digest = _digest(stored)
    manifest = {
        "training_resume_contract": stored,
        "training_resume_contract_sha256": stored_digest,
    }
    assert compatible_training_resume_contract_digest(config, manifest) == stored_digest

    incompatible = copy.deepcopy(stored)
    incompatible["training"]["epochs"] += 1
    incompatible_manifest = {
        "training_resume_contract": incompatible,
        "training_resume_contract_sha256": _digest(incompatible),
    }
    with pytest.raises(ValueError, match="does not match"):
        compatible_training_resume_contract_digest(config, incompatible_manifest)
