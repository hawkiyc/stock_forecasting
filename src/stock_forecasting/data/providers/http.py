"""Rate-limited JSON HTTP client with immutable response caching."""

from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

import requests

from stock_forecasting.data.manifest import canonical_json_sha256

from .base import RequestRecord

_SECRET_PARAMETER_NAMES = frozenset({"api_token", "api_key", "token", "password", "secret"})


class NetworkRequestBudgetExceeded(RuntimeError):
    """Signal that the project-side network-attempt safety ceiling was reached."""

    def __init__(self, *, consumed: int, maximum: int) -> None:
        self.consumed = consumed
        self.maximum = maximum
        super().__init__(
            "External API network-attempt budget exhausted: "
            f"{self.consumed}/{self.maximum}"
        )


class ProviderRequestError(RuntimeError):
    """Expose safe, structured provider failure metadata without request secrets."""

    def __init__(
        self,
        *,
        provider: str,
        category: str,
        attempts: int,
        retryable: bool,
        status_code: int | None = None,
        retry_after_seconds: float | None = None,
        rate_limit_limit: int | None = None,
        rate_limit_remaining: int | None = None,
    ) -> None:
        self.provider = provider
        self.category = category
        self.attempts = attempts
        self.retryable = retryable
        self.status_code = status_code
        self.retry_after_seconds = retry_after_seconds
        self.rate_limit_limit = rate_limit_limit
        self.rate_limit_remaining = rate_limit_remaining
        suffix = "" if attempts == 1 else "s"
        status = "" if status_code is None else f", HTTP {status_code}"
        super().__init__(
            f"{provider} request failed after {attempts} attempt{suffix} "
            f"(category={category}{status})"
        )

    def metadata(self) -> dict[str, Any]:
        """Return fields that are safe to persist in logs and progress manifests."""

        payload: dict[str, Any] = {
            "category": self.category,
            "provider": self.provider,
            "attempts": self.attempts,
            "retryable": self.retryable,
        }
        optional = {
            "status_code": self.status_code,
            "retry_after_seconds": self.retry_after_seconds,
            "rate_limit_limit": self.rate_limit_limit,
            "rate_limit_remaining": self.rate_limit_remaining,
        }
        payload.update({key: value for key, value in optional.items() if value is not None})
        return payload


def _nonnegative_integer_header(response: requests.Response, name: str) -> int | None:
    value = response.headers.get(name)
    if value is None or not value.strip().isdigit():
        return None
    parsed = int(value)
    return parsed if parsed >= 0 else None


def _retry_after_seconds(response: requests.Response) -> float | None:
    value = response.headers.get("Retry-After")
    if value is None:
        return None
    normalized = value.strip()
    if normalized.replace(".", "", 1).isdigit():
        parsed_seconds = float(normalized)
        return max(0.0, parsed_seconds) if math.isfinite(parsed_seconds) else None
    try:
        reset_at = parsedate_to_datetime(normalized)
    except (TypeError, ValueError, OverflowError):
        return None
    if reset_at.tzinfo is None:
        reset_at = reset_at.replace(tzinfo=UTC)
    return max(0.0, (reset_at.astimezone(UTC) - datetime.now(UTC)).total_seconds())


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
                raise NetworkRequestBudgetExceeded(
                    consumed=self._network_requests,
                    maximum=self.max_network_requests,
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

        last_category = "unknown"
        last_retryable = False
        last_status_code: int | None = None
        last_retry_after_seconds: float | None = None
        last_rate_limit_limit: int | None = None
        last_rate_limit_remaining: int | None = None
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
                response = error.response if isinstance(error, requests.HTTPError) else None
                last_status_code = response.status_code if response is not None else None
                last_retry_after_seconds = (
                    _retry_after_seconds(response) if response is not None else None
                )
                last_rate_limit_limit = (
                    _nonnegative_integer_header(response, "X-RateLimit-Limit")
                    if response is not None
                    else None
                )
                last_rate_limit_remaining = (
                    _nonnegative_integer_header(response, "X-RateLimit-Remaining")
                    if response is not None
                    else None
                )
                if isinstance(error, json.JSONDecodeError):
                    last_category = "invalid_json"
                    last_retryable = True
                elif response is None:
                    last_category = "network_error"
                    last_retryable = True
                elif response.status_code == 429:
                    last_category = "rate_limited"
                    last_retryable = True
                elif response.status_code >= 500:
                    last_category = "provider_unavailable"
                    last_retryable = True
                else:
                    last_category = "provider_rejected"
                    last_retryable = False
                if not last_retryable:
                    break
                if attempt < self.max_attempts:
                    delay = min(float(2 ** (attempt - 1)), 30.0)
                    if last_retry_after_seconds is not None:
                        delay = min(last_retry_after_seconds, 60.0)
                    self.sleeper(delay)
        raise ProviderRequestError(
            provider=self.provider,
            category=last_category,
            attempts=attempt,
            retryable=last_retryable,
            status_code=last_status_code,
            retry_after_seconds=last_retry_after_seconds,
            rate_limit_limit=last_rate_limit_limit,
            rate_limit_remaining=last_rate_limit_remaining,
        ) from None
