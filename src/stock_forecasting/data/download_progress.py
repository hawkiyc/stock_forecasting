"""Persistent, quota-aware progress for resumable provider downloads."""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from stock_forecasting.data.manifest import atomic_write_json, canonical_json_sha256
from stock_forecasting.data.providers.http import (
    AcquisitionDeadlineExceeded,
    NetworkRequestBudget,
    NetworkRequestBudgetExceeded,
    ProviderAcquisitionError,
    ProviderRequestError,
)

DOWNLOAD_PROGRESS_SCHEMA_VERSION = 1


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _positive_attempt_number(payload: dict[str, Any] | None) -> int:
    if payload is None:
        return 1
    attempt = payload.get("attempt")
    if not isinstance(attempt, dict):
        return 1
    number = attempt.get("number")
    if not isinstance(number, int) or isinstance(number, bool) or number < 1:
        return 1
    return number + 1


def _cache_inventory(raw_cache_roots: Iterable[Path]) -> dict[str, Any]:
    providers: dict[str, dict[str, int]] = {}
    response_count = 0
    size_bytes = 0
    seen_responses: set[tuple[str, str]] = set()
    roots = tuple(raw_cache_roots)
    for raw_cache_root in roots:
        if not raw_cache_root.is_dir() or raw_cache_root.is_symlink():
            continue
        for response_path in sorted(raw_cache_root.glob("*/*/*.json")):
            if not response_path.is_file() or response_path.is_symlink():
                continue
            relative = response_path.relative_to(raw_cache_root)
            provider = relative.parts[0]
            identity = (provider, response_path.stem)
            if identity in seen_responses:
                continue
            seen_responses.add(identity)
            response_size = response_path.stat().st_size
            response_count += 1
            size_bytes += response_size
            provider_inventory = providers.setdefault(
                provider,
                {"responses": 0, "size_bytes": 0},
            )
            provider_inventory["responses"] += 1
            provider_inventory["size_bytes"] += response_size
    return {
        "responses": response_count,
        "size_bytes": size_bytes,
        "by_provider": dict(sorted(providers.items())),
        "read_roots": len(roots),
    }


def _safe_error_metadata(error: BaseException) -> tuple[str, dict[str, Any]]:
    if isinstance(error, ProviderAcquisitionError):
        return error.state, error.metadata()
    if isinstance(error, ProviderRequestError):
        state = "waiting_for_provider" if error.retryable else "failed"
        return state, error.metadata()
    if isinstance(error, NetworkRequestBudgetExceeded):
        payload: dict[str, Any] = {
            "category": "network_request_safety_budget_exhausted",
            "retryable": True,
            "consumed": error.consumed,
            "maximum": error.maximum,
        }
        if error.provider is not None:
            payload["provider"] = error.provider
        return "waiting_for_budget", payload
    if isinstance(error, AcquisitionDeadlineExceeded):
        payload = {
            "category": "acquisition_time_budget_exhausted",
            "retryable": True,
            "deadline_epoch_seconds": error.deadline_epoch_seconds,
            "observed_epoch_seconds": error.observed_epoch_seconds,
        }
        if error.required_wait_seconds is not None:
            payload["required_wait_seconds"] = error.required_wait_seconds
        return "waiting_for_resume", payload
    return (
        "failed",
        {
            "category": "non_provider_failure",
            "retryable": False,
            "error_type": type(error).__name__,
        },
    )


class DownloadProgress:
    """Persist cache-backed progress without treating partial data as ready."""

    def __init__(
        self,
        *,
        path: Path,
        raw_cache_root: Path,
        read_cache_roots: Iterable[Path] = (),
        identity: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> None:
        if path.is_symlink():
            raise ValueError("Download progress path must not be a symlink")
        self.path = path
        self.raw_cache_root = raw_cache_root
        self.read_cache_roots = tuple(
            root
            for root in dict.fromkeys(Path(value) for value in read_cache_roots)
            if root != raw_cache_root
        )
        self.identity = identity
        self.identity_sha256 = canonical_json_sha256(identity)
        self.context = context or {}
        self.existing = self._load_existing()
        self.attempt_number = _positive_attempt_number(self.existing)
        self.started_at = _utc_now()

    def _load_existing(self) -> dict[str, Any] | None:
        if not self.path.exists():
            return None
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Download progress must contain a JSON object")
        if (
            payload.get("schema_version") != DOWNLOAD_PROGRESS_SCHEMA_VERSION
            or payload.get("kind") != "ohlcv-download-progress"
        ):
            raise ValueError("Download progress schema or kind is unsupported")
        if (
            payload.get("identity") != self.identity
            or payload.get("identity_sha256") != self.identity_sha256
        ):
            raise ValueError(
                "Download progress belongs to a different dataset request; "
                "use its own immutable dataset namespace"
            )
        return payload

    def _base_payload(
        self,
        *,
        state: str,
        request_budget: NetworkRequestBudget,
        estimated_http_requests: int | None = None,
    ) -> dict[str, Any]:
        first_started_at = self.started_at
        if self.existing is not None and isinstance(self.existing.get("first_started_at"), str):
            first_started_at = self.existing["first_started_at"]
        payload: dict[str, Any] = {
            "schema_version": DOWNLOAD_PROGRESS_SCHEMA_VERSION,
            "kind": "ohlcv-download-progress",
            "state": state,
            "first_started_at": first_started_at,
            "updated_at": _utc_now(),
            "identity": self.identity,
            "identity_sha256": self.identity_sha256,
            "context": self.context,
            "attempt": {
                "number": self.attempt_number,
                "started_at": self.started_at,
                "network_requests": request_budget.network_requests,
                "network_requests_by_provider": request_budget.provider_counts,
                "limited_network_requests": request_budget.limited_network_requests,
                "limited_providers": (
                    None
                    if request_budget.limited_providers is None
                    else sorted(request_budget.limited_providers)
                ),
                "max_network_requests": request_budget.max_network_requests,
                "max_network_requests_semantics": "limited_providers_only",
                "acquisition_deadline_epoch_seconds": (request_budget.deadline_epoch_seconds),
            },
            "cache": _cache_inventory((self.raw_cache_root, *self.read_cache_roots)),
            "resume": {
                "scope": "successful_raw_json_responses",
                "automatic_cache_reuse": True,
                "next_action": "rerun_the_same_dataset_request",
                "workflow_command": "bash scripts/runpod_workflow.sh cpu prepare",
            },
        }
        if estimated_http_requests is not None and estimated_http_requests > 0:
            payload["plan"] = {"estimated_http_requests": estimated_http_requests}
        return payload

    def start(self, request_budget: NetworkRequestBudget) -> None:
        """Mark an acquisition attempt without discarding prior cache entries."""

        atomic_write_json(
            self.path,
            self._base_payload(state="acquiring", request_budget=request_budget),
        )

    def fail(
        self,
        error: BaseException,
        request_budget: NetworkRequestBudget,
        *,
        estimated_http_requests: int | None = None,
    ) -> str:
        """Record a safe terminal state and return it to the CLI boundary."""

        state, error_payload = _safe_error_metadata(error)
        payload = self._base_payload(
            state=state,
            request_budget=request_budget,
            estimated_http_requests=estimated_http_requests,
        )
        payload["last_error"] = error_payload
        atomic_write_json(self.path, payload)
        return state

    def complete(
        self,
        request_budget: NetworkRequestBudget,
        *,
        estimated_http_requests: int | None = None,
    ) -> None:
        """Mark the raw download complete only after its immutable manifest exists."""

        payload = self._base_payload(
            state="downloaded",
            request_budget=request_budget,
            estimated_http_requests=estimated_http_requests,
        )
        payload["completed_at"] = _utc_now()
        atomic_write_json(self.path, payload)
