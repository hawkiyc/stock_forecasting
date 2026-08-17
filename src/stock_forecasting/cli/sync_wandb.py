"""Upload pending W&B transactions preserved on the network volume."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from stock_forecasting.run_paths import canonical_network_volume_root, validate_run_id
from stock_forecasting.wandb_status import (
    WandbComponent,
    read_wandb_status,
    update_wandb_status,
)

PENDING_STATES = frozenset({"online_running", "offline_pending", "sync_failed"})
COMPONENTS: tuple[WandbComponent, ...] = ("training", "validation")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id", nargs="?")
    return parser


def _status_paths(root: Path, run_id: str | None) -> list[Path]:
    if run_id:
        return [root / "lifecycle" / "runs" / validate_run_id(run_id) / "wandb.json"]
    return sorted((root / "lifecycle" / "runs").glob("*/wandb.json"))


def _wandb_cli() -> Path:
    candidate = Path(sys.executable).resolve().parent / "wandb"
    if not candidate.is_file() or not os.access(candidate, os.X_OK):
        raise FileNotFoundError(f"W&B CLI is unavailable beside the project Python: {candidate}")
    return candidate


def _sync_component(
    *,
    root: Path,
    status: dict[str, Any],
    component: WandbComponent,
    transaction: Path,
    transaction_mode: str,
    wandb_cli: Path,
) -> bool:
    run_id = validate_run_id(str(status["run_id"]))
    if transaction != root and root not in transaction.parents:
        raise ValueError("W&B transaction directory escapes the network volume")
    if not transaction.is_dir() or transaction.is_symlink():
        raise FileNotFoundError(f"W&B transaction directory is unavailable: {transaction}")
    completed = subprocess.run(
        [
            str(wandb_cli),
            "sync",
            "--legacy",
            "--include-offline",
            "--include-online",
            "--append",
            "--id",
            run_id,
            str(transaction),
        ],
        check=False,
    )
    next_state = "synced" if completed.returncode == 0 else "sync_failed"
    update_wandb_status(
        run_id=run_id,
        component=component,
        state=next_state,
        mode=transaction_mode,
        project=str(status.get("project", "")),
        entity=status.get("entity") if isinstance(status.get("entity"), str) else None,
        wandb_directory=root,
        transaction_directory=transaction,
        error=(None if completed.returncode == 0 else f"wandb sync exited {completed.returncode}"),
    )
    return completed.returncode == 0


def _record_sync_failure(
    *,
    root: Path,
    status: dict[str, Any],
    component: WandbComponent,
    details: dict[str, Any],
    error: BaseException | str,
    transaction: Path | None = None,
    transaction_mode: str | None = None,
) -> None:
    message = str(error)
    if isinstance(error, BaseException):
        message = f"{type(error).__name__}: {error}"
    update_wandb_status(
        run_id=validate_run_id(str(status["run_id"])),
        component=component,
        state="sync_failed",
        mode=transaction_mode or str(details.get("mode", "unknown")),
        project=str(status.get("project", "")),
        entity=status.get("entity") if isinstance(status.get("entity"), str) else None,
        wandb_directory=root,
        transaction_directory=transaction,
        error=message,
    )


def _pending_transactions(root: Path, details: dict[str, Any]) -> list[tuple[Path, str]]:
    records = details.get("transactions")
    values: list[tuple[str, str]] = []
    if isinstance(records, list):
        for record in records:
            if not isinstance(record, dict) or record.get("state") not in PENDING_STATES:
                continue
            value = record.get("directory")
            if not isinstance(value, str) or not value:
                raise ValueError("Pending W&B transaction has no directory")
            values.append((value, str(record.get("mode", details.get("mode", "unknown")))))
    elif details.get("state") in PENDING_STATES:
        value = details.get("transaction_directory")
        if isinstance(value, str) and value:
            values.append((value, str(details.get("mode", "unknown"))))
    transactions: list[tuple[Path, str]] = []
    for value, mode in dict.fromkeys(values):
        transaction = Path(value).expanduser().resolve(strict=False)
        if transaction != root and root not in transaction.parents:
            raise ValueError("W&B transaction directory escapes the network volume")
        transactions.append((transaction, mode))
    return transactions


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    root = canonical_network_volume_root()
    paths = _status_paths(root, arguments.run_id)
    if not paths:
        print("No W&B delivery status files were found.")
        return 0
    setup_error: BaseException | None = None
    cli: Path | None = None
    if not os.environ.get("WANDB_API_KEY"):
        setup_error = RuntimeError(
            "WANDB_API_KEY is required to sync pending W&B transactions"
        )
    else:
        try:
            cli = _wandb_cli()
        except (OSError, ValueError) as error:
            setup_error = error
    attempted = 0
    failures = 0
    for path in paths:
        if not path.is_file() or path.is_symlink():
            if arguments.run_id:
                raise FileNotFoundError(f"W&B status is unavailable: {path}")
            continue
        status = read_wandb_status(path)
        components = status.get("components")
        if not isinstance(components, dict):
            raise ValueError(f"W&B status components are invalid: {path}")
        for component in COMPONENTS:
            details = components.get(component)
            if not isinstance(details, dict) or details.get("state") not in PENDING_STATES:
                continue
            try:
                transactions = _pending_transactions(root, details)
            except (OSError, ValueError) as error:
                _record_sync_failure(
                    root=root,
                    status=status,
                    component=component,
                    details=details,
                    error=error,
                )
                attempted += 1
                failures += 1
                continue
            if not transactions:
                _record_sync_failure(
                    root=root,
                    status=status,
                    component=component,
                    details=details,
                    error="Pending W&B component has no transaction directory",
                )
                attempted += 1
                failures += 1
                continue
            for transaction, transaction_mode in transactions:
                attempted += 1
                if setup_error is not None:
                    _record_sync_failure(
                        root=root,
                        status=status,
                        component=component,
                        details=details,
                        error=setup_error,
                        transaction=transaction,
                        transaction_mode=transaction_mode,
                    )
                    failures += 1
                    continue
                if cli is None:
                    raise RuntimeError("W&B CLI setup produced no executable")
                try:
                    succeeded = _sync_component(
                        root=root,
                        status=status,
                        component=component,
                        transaction=transaction,
                        transaction_mode=transaction_mode,
                        wandb_cli=cli,
                    )
                except (OSError, ValueError) as error:
                    _record_sync_failure(
                        root=root,
                        status=status,
                        component=component,
                        details=details,
                        error=error,
                        transaction=transaction,
                        transaction_mode=transaction_mode,
                    )
                    succeeded = False
                failures += int(not succeeded)
    print(f"W&B sync transactions attempted={attempted} failed={failures}")
    return 0 if failures == 0 else 4


if __name__ == "__main__":
    raise SystemExit(main())
