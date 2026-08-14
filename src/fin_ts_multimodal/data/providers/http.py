"""Rate-limited JSON HTTP client with immutable response caching."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

from fin_ts_multimodal.data.manifest import canonical_json_sha256

from .base import RequestRecord

_SECRET_PARAMETER_NAMES = frozenset({"api_token", "api_key", "token", "password", "secret"})


class NetworkRequestBudget:
    """Share one fail-closed network-attempt budget across provider clients."""

    def __init__(self, max_network_requests: int) -> None:
        if max_network_requests < 1:
            raise ValueError("max_network_requests must be positive")
        self.max_network_requests = max_network_requests
        self._network_requests = 0
        self._provider_counts: dict[str, int] = {}
        self._lock = threading.Lock()

    def consume(self, provider: str) -> None:
        """Reserve one attempt before sending it so retries cannot exceed the cap."""

        with self._lock:
            if self._network_requests >= self.max_network_requests:
                raise RuntimeError(
                    "External API network-attempt budget exhausted: "
                    f"{self._network_requests}/{self.max_network_requests}"
                )
            self._network_requests += 1
            self._provider_counts[provider] = self._provider_counts.get(provider, 0) + 1

    @property
    def network_requests(self) -> int:
        with self._lock:
            return self._network_requests

    @property
    def provider_counts(self) -> dict[str, int]:
        with self._lock:
            return dict(sorted(self._provider_counts.items()))


class CachedJsonClient:
    """Share one HTTP session, throttle requests, and never overwrite raw responses."""

    def __init__(
        self,
        *,
        provider: str,
        raw_cache_root: str | Path,
        max_requests_per_second: float,
        timeout_seconds: float = 30.0,
        max_attempts: int = 3,
        session: requests.Session | None = None,
        request_budget: NetworkRequestBudget | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if max_requests_per_second <= 0.0:
            raise ValueError("max_requests_per_second must be positive")
        if timeout_seconds <= 0.0 or max_attempts < 1:
            raise ValueError("HTTP timeout and attempts must be positive")
        self.provider = provider
        self.raw_cache_root = Path(raw_cache_root)
        self.minimum_interval = 1.0 / max_requests_per_second
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max_attempts
        self.session = session or requests.Session()
        self.request_budget = request_budget
        self.clock = clock
        self.sleeper = sleeper
        self._last_request_at: float | None = None

    def _identity(
        self,
        *,
        endpoint: str,
        params: Mapping[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        public_params = {
            key: value
            for key, value in sorted(params.items())
            if key.lower() not in _SECRET_PARAMETER_NAMES
        }
        identity = {
            "provider": self.provider,
            "endpoint": endpoint,
            "params": public_params,
        }
        return canonical_json_sha256(identity), identity

    def _cache_path(self, request_sha256: str) -> Path:
        return self.raw_cache_root / self.provider / request_sha256[:2] / f"{request_sha256}.json"

    def _record(
        self,
        *,
        request_sha256: str,
        cache_path: Path,
        body: bytes,
        cache_hit: bool,
        requested_at: str,
    ) -> RequestRecord:
        response_digest = hashlib.sha256(body).hexdigest()
        relative = cache_path.relative_to(self.raw_cache_root)
        return RequestRecord(
            provider=self.provider,
            request_sha256=request_sha256,
            cache_relative_path=relative.as_posix(),
            response_sha256=response_digest,
            response_size_bytes=len(body),
            cache_hit=cache_hit,
            requested_at=requested_at,
        )

    def get_json(
        self,
        endpoint: str,
        *,
        params: Mapping[str, Any],
    ) -> tuple[Any, RequestRecord]:
        request_sha256, _identity = self._identity(endpoint=endpoint, params=params)
        cache_path = self._cache_path(request_sha256)
        if cache_path.is_file():
            body = cache_path.read_bytes()
            return (
                json.loads(body),
                self._record(
                    request_sha256=request_sha256,
                    cache_path=cache_path,
                    body=body,
                    cache_hit=True,
                    requested_at=datetime.fromtimestamp(
                        cache_path.stat().st_mtime, tz=UTC
                    ).isoformat(),
                ),
            )

        last_error_name = "unknown"
        for attempt in range(1, self.max_attempts + 1):
            now = self.clock()
            if self._last_request_at is not None:
                remaining = self.minimum_interval - (now - self._last_request_at)
                if remaining > 0.0:
                    self.sleeper(remaining)
            requested_at = datetime.now(UTC).isoformat()
            try:
                self._last_request_at = self.clock()
                if self.request_budget is not None:
                    self.request_budget.consume(self.provider)
                response = self.session.get(
                    endpoint,
                    params=dict(params),
                    timeout=self.timeout_seconds,
                    headers={"User-Agent": "fin-ts-quant-research/0.2"},
                )
                if response.status_code == 429 or response.status_code >= 500:
                    raise requests.HTTPError(
                        f"Retryable HTTP {response.status_code}",
                        response=response,
                    )
                response.raise_for_status()
                body = response.content
                payload = json.loads(body)
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    with cache_path.open("xb") as stream:
                        stream.write(body)
                except FileExistsError:
                    body = cache_path.read_bytes()
                    payload = json.loads(body)
                return (
                    payload,
                    self._record(
                        request_sha256=request_sha256,
                        cache_path=cache_path,
                        body=body,
                        cache_hit=False,
                        requested_at=requested_at,
                    ),
                )
            except (json.JSONDecodeError, requests.RequestException) as error:
                last_error_name = type(error).__name__
                response = error.response if isinstance(error, requests.HTTPError) else None
                retryable_http = (
                    response is None or response.status_code == 429 or response.status_code >= 500
                )
                if not retryable_http:
                    break
                if attempt < self.max_attempts:
                    delay = min(float(2 ** (attempt - 1)), 30.0)
                    if response is not None:
                        retry_after = response.headers.get("Retry-After")
                        if retry_after and retry_after.replace(".", "", 1).isdigit():
                            delay = min(float(retry_after), 60.0)
                    self.sleeper(delay)
        suffix = "" if attempt == 1 else "s"
        raise RuntimeError(
            f"{self.provider} request failed after {attempt} attempt{suffix} ({last_error_name})"
        ) from None
