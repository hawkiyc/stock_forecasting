"""Durable W&B delivery status stored on the RunPod network volume."""

from __future__ import annotations

import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from stock_forecasting.run_paths import canonical_network_volume_root, validate_run_id

WandbComponent = Literal["training", "validation"]
COMPONENT_STATES = frozenset(
    {
        "online_running",
        "online_finished",
        "offline_pending",
        "sync_failed",
        "synced",
        "failed",
    }
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _status_root(wandb_directory: Path | None = None) -> Path:
    if os.environ.get("RUNPOD_POD_ID"):
        return canonical_network_volume_root()
    if wandb_directory is not None:
        return wandb_directory.expanduser().resolve(strict=False)
    return canonical_network_volume_root()


def wandb_status_path(run_id: str, *, wandb_directory: Path | None = None) -> Path:
    root = _status_root(wandb_directory)
    return root / "lifecycle" / "runs" / validate_run_id(run_id) / "wandb.json"


def _overall_state(components: dict[str, Any]) -> str:
    states = {
        str(value.get("state"))
        for value in components.values()
        if isinstance(value, dict) and value.get("state")
    }
    if "failed" in states:
        return "failed"
    if "sync_failed" in states:
        return "sync_failed"
    if "offline_pending" in states:
        return "offline_pending"
    if "online_running" in states:
        return "running"
    if states and states <= {"online_finished", "synced"}:
        return "ready"
    return "unknown"


def _transaction_records(previous: dict[str, Any] | None) -> list[dict[str, Any]]:
    if previous is None:
        return []
    records = previous.get("transactions")
    if isinstance(records, list):
        return [dict(record) for record in records if isinstance(record, dict)]
    legacy_directory = previous.get("transaction_directory")
    if not isinstance(legacy_directory, str) or not legacy_directory:
        return []
    return [
        {
            "directory": legacy_directory,
            "state": str(previous.get("state", "offline_pending")),
            "mode": str(previous.get("mode", "unknown")),
            "updated_at": str(previous.get("updated_at", _utc_now())),
        }
    ]


def _component_state(requested_state: str, transactions: list[dict[str, Any]]) -> str:
    if requested_state in {"failed", "sync_failed"}:
        return requested_state
    states = {
        str(transaction.get("state"))
        for transaction in transactions
        if transaction.get("state")
    }
    if "failed" in states:
        return "failed"
    if "sync_failed" in states:
        return "sync_failed"
    if "offline_pending" in states:
        return "offline_pending"
    if "online_running" in states:
        return "online_running"
    return requested_state


def update_wandb_status(
    *,
    run_id: str,
    component: WandbComponent,
    state: str,
    mode: str,
    project: str,
    entity: str | None,
    wandb_directory: Path,
    transaction_directory: Path | None = None,
    error: str | None = None,
) -> Path:
    """Atomically merge one W&B component's delivery state."""

    if state not in COMPONENT_STATES:
        raise ValueError(f"Unsupported W&B component state: {state}")
    canonical_run_id = validate_run_id(run_id)
    path = wandb_status_path(canonical_run_id, wandb_directory=wandb_directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {}
    if path.exists():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError(f"W&B status is not a JSON object: {path}")
        payload = loaded
    components = payload.get("components")
    if not isinstance(components, dict):
        components = {}

    previous_value = components.get(component)
    previous = previous_value if isinstance(previous_value, dict) else None
    transactions = _transaction_records(previous)
    now = _utc_now()
    transaction_text: str | None = None
    if transaction_directory is not None:
        transaction = transaction_directory.expanduser().resolve(strict=False)
        if os.environ.get("RUNPOD_POD_ID"):
            root = canonical_network_volume_root()
            if transaction != root and root not in transaction.parents:
                raise ValueError("W&B transaction directory must be on the network volume")
        transaction_text = str(transaction)
        transaction_record: dict[str, Any] = {
            "directory": transaction_text,
            "state": state,
            "mode": mode,
            "updated_at": now,
        }
        if error:
            transaction_record["error"] = error[:2000]
        transactions = [
            record
            for record in transactions
            if record.get("directory") != transaction_text
        ]
        transactions.append(transaction_record)

    component_payload: dict[str, Any] = {
        "state": _component_state(state, transactions),
        "latest_state": state,
        "mode": mode,
        "updated_at": now,
    }
    if transactions:
        component_payload["transactions"] = transactions
        latest_directory = transaction_text or transactions[-1].get("directory")
        if isinstance(latest_directory, str) and latest_directory:
            component_payload["transaction_directory"] = latest_directory
    if error:
        component_payload["error"] = error[:2000]
    components[component] = component_payload

    next_payload = {
        "schema_version": 1,
        "kind": "wandb-delivery-status",
        "run_id": canonical_run_id,
        "project": project,
        "entity": entity or None,
        "state": _overall_state(components),
        "components": components,
        "created_at": payload.get("created_at", now),
        "updated_at": now,
    }
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        temporary.write_text(
            json.dumps(next_payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def read_wandb_status(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 1
        or payload.get("kind") != "wandb-delivery-status"
    ):
        raise ValueError(f"Unsupported W&B status schema: {path}")
    validate_run_id(str(payload.get("run_id", "")))
    return payload
