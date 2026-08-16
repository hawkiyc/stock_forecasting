"""Durable delivery-state contracts for online and offline W&B runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from stock_forecasting.tracking import TrackingRun
from stock_forecasting.wandb_status import read_wandb_status, update_wandb_status


class _FailingLogBackend:
    def __init__(self, directory: Path) -> None:
        self.dir = str(directory / "files")
        self.summary: dict[str, Any] = {}

    def log(self, payload: dict[str, Any]) -> None:
        raise RuntimeError(
            f"simulated upload failure at step {payload.get('trainer/global_step')}"
        )

    def finish(self, *, exit_code: int = 0) -> None:
        assert exit_code == 1


def test_component_states_merge_into_one_workflow_status(tmp_path: Path) -> None:
    transaction = tmp_path / "wandb" / "offline-run"
    transaction.mkdir(parents=True)
    status_path = update_wandb_status(
        run_id="run-test",
        component="training",
        state="offline_pending",
        mode="offline",
        project="project",
        entity=None,
        wandb_directory=tmp_path,
        transaction_directory=transaction,
    )
    update_wandb_status(
        run_id="run-test",
        component="validation",
        state="online_finished",
        mode="online",
        project="project",
        entity=None,
        wandb_directory=tmp_path,
    )
    assert read_wandb_status(status_path)["state"] == "offline_pending"

    update_wandb_status(
        run_id="run-test",
        component="training",
        state="synced",
        mode="offline",
        project="project",
        entity=None,
        wandb_directory=tmp_path,
        transaction_directory=transaction,
    )
    assert read_wandb_status(status_path)["state"] == "ready"


def test_online_log_failure_remains_pending_after_finish(tmp_path: Path) -> None:
    run_directory = tmp_path / "savedModel" / "run-test"
    transaction = tmp_path / "wandb" / "run-test"
    run_directory.mkdir(parents=True)
    transaction.mkdir(parents=True)
    backend = _FailingLogBackend(transaction)
    tracking = TrackingRun(
        id="run-test",
        name="test",
        directory=run_directory,
        backend=backend,
        mode="online",
        tracking_enabled=True,
        project="project",
        entity=None,
        wandb_directory=tmp_path,
    )

    with pytest.raises(RuntimeError, match="simulated upload failure"):
        tracking.log({"train/loss": 1.0}, step=1)
    tracking.finish(exit_code=1)

    payload = read_wandb_status(
        tmp_path / "lifecycle" / "runs" / "run-test" / "wandb.json"
    )
    assert payload["state"] == "offline_pending"
    assert payload["components"]["training"]["state"] == "offline_pending"
    metric = json.loads((run_directory / "metrics.jsonl").read_text(encoding="utf-8"))
    assert metric["step"] == 1
    assert metric["train/loss"] == 1.0


def test_offline_transaction_write_failure_is_not_marked_recoverable(tmp_path: Path) -> None:
    run_directory = tmp_path / "savedModel" / "run-test"
    transaction = tmp_path / "wandb" / "offline-run"
    run_directory.mkdir(parents=True)
    transaction.mkdir(parents=True)
    tracking = TrackingRun(
        id="run-test",
        name="test",
        directory=run_directory,
        backend=_FailingLogBackend(transaction),
        mode="offline",
        tracking_enabled=True,
        project="project",
        entity=None,
        wandb_directory=tmp_path,
    )

    with pytest.raises(RuntimeError, match="simulated upload failure"):
        tracking.log({"train/loss": 1.0}, step=1)
    tracking.finish(exit_code=1)

    payload = read_wandb_status(
        tmp_path / "lifecycle" / "runs" / "run-test" / "wandb.json"
    )
    assert payload["state"] == "failed"
    assert payload["components"]["training"]["state"] == "failed"


def test_resume_preserves_every_pending_transaction(tmp_path: Path) -> None:
    first = tmp_path / "wandb" / "offline-first"
    second = tmp_path / "wandb" / "online-resume"
    first.mkdir(parents=True)
    second.mkdir(parents=True)

    status_path = update_wandb_status(
        run_id="run-test",
        component="training",
        state="offline_pending",
        mode="offline",
        project="project",
        entity=None,
        wandb_directory=tmp_path,
        transaction_directory=first,
    )
    update_wandb_status(
        run_id="run-test",
        component="training",
        state="online_running",
        mode="online",
        project="project",
        entity=None,
        wandb_directory=tmp_path,
        transaction_directory=second,
    )
    update_wandb_status(
        run_id="run-test",
        component="training",
        state="online_finished",
        mode="online",
        project="project",
        entity=None,
        wandb_directory=tmp_path,
        transaction_directory=second,
    )

    payload = read_wandb_status(status_path)
    training = payload["components"]["training"]
    assert training["state"] == "offline_pending"
    assert {
        transaction["directory"]: transaction["state"]
        for transaction in training["transactions"]
    } == {
        str(first): "offline_pending",
        str(second): "online_finished",
    }

    update_wandb_status(
        run_id="run-test",
        component="training",
        state="synced",
        mode="offline",
        project="project",
        entity=None,
        wandb_directory=tmp_path,
        transaction_directory=first,
    )
    assert read_wandb_status(status_path)["state"] == "ready"
