"""Rate-limited JSON HTTP client with immutable response caching."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import requests

from stock_forecasting.data.manifest import canonical_json_sha256

from .base import RequestRecord

_SECRET_PARAMETER_NAMES = frozenset({"api_token", "api_key", "token", "password", "secret"})
_CACHE_REVISION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_RUN_APP_HOST_PATTERN = re.compile(
    r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?"
    r"(?:\.[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?)*\.run\.app"
)
_TPEX_UPSTREAM_HOST = "www.tpex.org.tw"
_TPEX_RELAY_PATHS = frozenset(
    {
        "/www/zh-tw/afterTrading/dailyQuotes",
        "/www/zh-tw/bulletin/exDailyQ",
        "/www/zh-tw/indexInfo/ROE",
        "/www/zh-tw/indexInfo/inx",
    }
)


@dataclass(frozen=True)
class TpexRelayTransport:
    """Rewrite approved TPEx requests without changing their cache identity."""

    origin: str
    token: str = field(repr=False)

    def __post_init__(self) -> None:
        parsed = urlsplit(self.origin)
        try:
            port = parsed.port
        except ValueError as error:
            raise ValueError("TPEx relay origin has an invalid port") from error
        if (
            parsed.scheme != "https"
            or parsed.hostname is None
            or parsed.username is not None
            or parsed.password is not None
            or port is not None
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
            or _RUN_APP_HOST_PATTERN.fullmatch(parsed.hostname) is None
        ):
            raise ValueError("TPEx relay origin must be a root run.app HTTPS origin")
        if not 32 <= len(self.token) <= 512 or any(
            character in self.token for character in ("\r", "\n", "\0")
        ):
            raise ValueError("TPEx relay token has an invalid format")
        object.__setattr__(self, "origin", f"https://{parsed.hostname}")

    def request_url(self, endpoint: str) -> str:
        parsed = urlsplit(endpoint)
        try:
            port = parsed.port
        except ValueError as error:
            raise ValueError("TPEx endpoint has an invalid port") from error
        if (
            parsed.scheme != "https"
            or parsed.hostname != _TPEX_UPSTREAM_HOST
            or parsed.username is not None
            or parsed.password is not None
            or port is not None
            or parsed.path not in _TPEX_RELAY_PATHS
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("TPEx relay transport rejected an unsupported upstream endpoint")
        return f"{self.origin}{parsed.path}"

    def request_headers(self) -> dict[str, str]:
        return {"X-TPEX-Relay-Token": self.token}


# Preserve the former import name while callers migrate to the provider-neutral name.
TpexWorkerTransport = TpexRelayTransport


class NetworkRequestBudgetExceeded(RuntimeError):
    """Signal that the project-side network-attempt safety ceiling was reached."""

    def __init__(
        self,
        *,
        consumed: int,
        maximum: int,
        provider: str | None = None,
    ) -> None:
        self.consumed = consumed
        self.maximum = maximum
        self.provider = provider
        scope = "External API" if provider is None else provider
        super().__init__(
            f"{scope} network-attempt budget exhausted: {self.consumed}/{self.maximum}"
        )


class AcquisitionDeadlineExceeded(RuntimeError):
    """Signal that acquisition stopped to preserve CPU preparation time."""

    def __init__(
        self,
        *,
        deadline_epoch_seconds: float,
        observed_epoch_seconds: float,
        required_wait_seconds: float | None = None,
    ) -> None:
        self.deadline_epoch_seconds = deadline_epoch_seconds
        self.observed_epoch_seconds = observed_epoch_seconds
        self.required_wait_seconds = required_wait_seconds
        wait_context = (
            "" if required_wait_seconds is None else f", required_wait={required_wait_seconds:.3f}s"
        )
        super().__init__(
            "Acquisition time budget exhausted before data preparation: "
            f"deadline={deadline_epoch_seconds:.3f}, observed={observed_epoch_seconds:.3f}"
            f"{wait_context}"
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
        backoff_wait_count: int = 0,
        total_backoff_seconds: float = 0.0,
        last_backoff_seconds: float | None = None,
        proposed_backoff_seconds: float | None = None,
        max_backoff_seconds: float | None = None,
        exit_reason: str | None = None,
    ) -> None:
        self.provider = provider
        self.category = category
        self.attempts = attempts
        self.retryable = retryable
        self.status_code = status_code
        self.retry_after_seconds = retry_after_seconds
        self.rate_limit_limit = rate_limit_limit
        self.rate_limit_remaining = rate_limit_remaining
        self.backoff_wait_count = backoff_wait_count
        self.total_backoff_seconds = total_backoff_seconds
        self.last_backoff_seconds = last_backoff_seconds
        self.proposed_backoff_seconds = proposed_backoff_seconds
        self.max_backoff_seconds = max_backoff_seconds
        self.exit_reason = exit_reason
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
        if self.backoff_wait_count or self.proposed_backoff_seconds is not None:
            backoff: dict[str, Any] = {
                "wait_count": self.backoff_wait_count,
                "total_wait_seconds": self.total_backoff_seconds,
            }
            backoff_optional = {
                "last_wait_seconds": self.last_backoff_seconds,
                "proposed_wait_seconds": self.proposed_backoff_seconds,
                "maximum_seconds": self.max_backoff_seconds,
                "exit_reason": self.exit_reason,
            }
            backoff.update(
                {key: value for key, value in backoff_optional.items() if value is not None}
            )
            payload["backoff"] = backoff
        return payload

    def _clone(self) -> ProviderRequestError:
        """Copy structured provider state without sharing traceback objects across threads."""

        return ProviderRequestError(
            provider=self.provider,
            category=self.category,
            attempts=self.attempts,
            retryable=self.retryable,
            status_code=self.status_code,
            retry_after_seconds=self.retry_after_seconds,
            rate_limit_limit=self.rate_limit_limit,
            rate_limit_remaining=self.rate_limit_remaining,
            backoff_wait_count=self.backoff_wait_count,
            total_backoff_seconds=self.total_backoff_seconds,
            last_backoff_seconds=self.last_backoff_seconds,
            proposed_backoff_seconds=self.proposed_backoff_seconds,
            max_backoff_seconds=self.max_backoff_seconds,
            exit_reason=self.exit_reason,
        )


class ProviderAcquisitionError(RuntimeError):
    """Summarize provider loops only after every parallel loop has exited."""

    def __init__(
        self,
        *,
        state: str,
        provider_outcomes: Mapping[str, Mapping[str, Any]],
        retryable: bool,
    ) -> None:
        self.state = state
        self.provider_outcomes = {
            provider: dict(outcome) for provider, outcome in sorted(provider_outcomes.items())
        }
        self.retryable = retryable
        super().__init__(
            "Parallel provider acquisition stopped after every provider loop exited "
            f"(state={state})"
        )

    def metadata(self) -> dict[str, Any]:
        """Return the aggregate state without retaining underlying exceptions."""

        return {
            "category": "parallel_provider_loops_exited",
            "retryable": self.retryable,
            "state": self.state,
            "provider_outcomes": self.provider_outcomes,
        }


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
    """Track every request while limiting only explicitly scoped providers."""

    def __init__(
        self,
        max_network_requests: int,
        *,
        deadline_epoch_seconds: float | None = None,
        limited_providers: frozenset[str] | set[str] | None = None,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        if max_network_requests < 1:
            raise ValueError("max_network_requests must be positive")
        if deadline_epoch_seconds is not None and (
            not math.isfinite(deadline_epoch_seconds) or deadline_epoch_seconds <= 0.0
        ):
            raise ValueError("deadline_epoch_seconds must be finite and positive")
        self.max_network_requests = max_network_requests
        self.deadline_epoch_seconds = deadline_epoch_seconds
        self.limited_providers = None if limited_providers is None else frozenset(limited_providers)
        self.wall_clock = wall_clock
        self._network_requests = 0
        self._limited_network_requests = 0
        self._provider_counts: dict[str, int] = {}
        self._lock = threading.Lock()

    def _check_time_locked(self) -> None:
        if self.deadline_epoch_seconds is None:
            return
        observed = self.wall_clock()
        if observed >= self.deadline_epoch_seconds:
            raise AcquisitionDeadlineExceeded(
                deadline_epoch_seconds=self.deadline_epoch_seconds,
                observed_epoch_seconds=observed,
            )

    def check_time(self) -> None:
        """Stop cache replay and network work before the preparation reserve."""

        with self._lock:
            self._check_time_locked()

    def check_wait(self, wait_seconds: float) -> None:
        """Refuse a retry sleep that would consume the preparation reserve."""

        if not math.isfinite(wait_seconds) or wait_seconds < 0.0:
            raise ValueError("wait_seconds must be finite and non-negative")
        with self._lock:
            self._check_time_locked()
            if self.deadline_epoch_seconds is None:
                return
            observed = self.wall_clock()
            if observed + wait_seconds >= self.deadline_epoch_seconds:
                raise AcquisitionDeadlineExceeded(
                    deadline_epoch_seconds=self.deadline_epoch_seconds,
                    observed_epoch_seconds=observed,
                    required_wait_seconds=wait_seconds,
                )

    def consume(self, provider: str) -> None:
        """Reserve one attempt before sending it so retries cannot exceed the cap."""

        with self._lock:
            self._check_time_locked()
            is_limited = self.limited_providers is None or provider in self.limited_providers
            if is_limited and self._limited_network_requests >= self.max_network_requests:
                raise NetworkRequestBudgetExceeded(
                    consumed=self._limited_network_requests,
                    maximum=self.max_network_requests,
                    provider=provider,
                )
            self._network_requests += 1
            if is_limited:
                self._limited_network_requests += 1
            self._provider_counts[provider] = self._provider_counts.get(provider, 0) + 1

    @property
    def network_requests(self) -> int:
        with self._lock:
            return self._network_requests

    @property
    def limited_network_requests(self) -> int:
        with self._lock:
            return self._limited_network_requests

    @property
    def provider_counts(self) -> dict[str, int]:
        with self._lock:
            return dict(sorted(self._provider_counts.items()))


class CachedJsonClient:
    """Share one throttle across thread-local sessions and immutable responses."""

    def __init__(
        self,
        *,
        provider: str,
        raw_cache_root: str | Path,
        read_cache_roots: Iterable[str | Path] = (),
        cache_revision: str = "v1",
        max_requests_per_second: float,
        timeout_seconds: float = 30.0,
        max_attempts: int | None = None,
        max_backoff_seconds: float | None = None,
        headers: Mapping[str, str] | None = None,
        retryable_status_codes: frozenset[int] | set[int] | None = None,
        transport: TpexRelayTransport | None = None,
        session: requests.Session | None = None,
        request_budget: NetworkRequestBudget | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if max_requests_per_second <= 0.0:
            raise ValueError("max_requests_per_second must be positive")
        if _CACHE_REVISION_PATTERN.fullmatch(cache_revision) is None:
            raise ValueError("cache_revision must be a safe 1-64 character label")
        if timeout_seconds <= 0.0:
            raise ValueError("HTTP timeout must be positive")
        if max_attempts is not None and max_attempts < 1:
            raise ValueError("HTTP attempts must be positive when bounded")
        if max_backoff_seconds is not None and (
            not math.isfinite(max_backoff_seconds) or max_backoff_seconds <= 0.0
        ):
            raise ValueError("Maximum backoff must be finite and positive when bounded")
        self.provider = provider
        self.raw_cache_root = Path(raw_cache_root)
        self.cache_revision = cache_revision
        self.read_cache_roots = tuple(
            path
            for path in dict.fromkeys(Path(value) for value in read_cache_roots)
            if path != self.raw_cache_root
        )
        self.minimum_interval = 1.0 / max_requests_per_second
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max_attempts
        self.max_backoff_seconds = max_backoff_seconds
        self.headers = {
            "User-Agent": "fin-ts-quant-research/0.2",
            **({} if headers is None else dict(headers)),
        }
        self.retryable_status_codes = frozenset(
            {429} if retryable_status_codes is None else retryable_status_codes
        )
        if any(
            not isinstance(status_code, int)
            or isinstance(status_code, bool)
            or not 400 <= status_code <= 599
            for status_code in self.retryable_status_codes
        ):
            raise ValueError("Retryable HTTP status codes must be integers from 400 to 599")
        self.transport = transport
        self.session = session
        self.request_budget = request_budget
        self.clock = clock
        self.sleeper = sleeper
        self._last_request_at: float | None = None
        self._rate_lock = threading.Lock()
        self._provided_session_lock = threading.Lock()
        self._provider_stop_lock = threading.Lock()
        self._provider_stop_error: ProviderRequestError | None = None
        self._thread_local = threading.local()

    def _raise_if_provider_stopped(self) -> None:
        with self._provider_stop_lock:
            error = (
                None
                if self._provider_stop_error is None
                else self._provider_stop_error._clone()
            )
        if error is not None:
            raise error from None

    def _stop_provider(self, error: ProviderRequestError) -> None:
        with self._provider_stop_lock:
            if self._provider_stop_error is None:
                self._provider_stop_error = error._clone()

    def _session(self) -> requests.Session:
        if self.session is not None:
            return self.session
        existing = getattr(self._thread_local, "session", None)
        if isinstance(existing, requests.Session):
            return existing
        created = requests.Session()
        self._thread_local.session = created
        return created

    def _reserve_request_slot(self) -> None:
        """Serialize request starts so all workers obey one provider QPS contract."""

        with self._rate_lock:
            now = self.clock()
            if self._last_request_at is not None:
                remaining = self.minimum_interval - (now - self._last_request_at)
                if remaining > 0.0:
                    if self.request_budget is not None:
                        self.request_budget.check_wait(remaining)
                    self.sleeper(remaining)
            self._last_request_at = self.clock()

    def _send(self, endpoint: str, params: Mapping[str, Any]) -> requests.Response:
        session = self._session()
        request_endpoint = endpoint
        request_headers = dict(self.headers)
        if self.transport is not None:
            request_endpoint = self.transport.request_url(endpoint)
            request_headers.update(self.transport.request_headers())
        if self.session is None:
            return session.get(
                request_endpoint,
                params=dict(params),
                timeout=self.timeout_seconds,
                headers=request_headers,
            )
        # Injected sessions are primarily deterministic test doubles and are not
        # assumed to be thread-safe.
        with self._provided_session_lock:
            return session.get(
                request_endpoint,
                params=dict(params),
                timeout=self.timeout_seconds,
                headers=request_headers,
            )

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
        # Preserve the historical v1 key so existing immutable responses remain
        # reusable. Later provider revisions are isolated by the request digest.
        if self.cache_revision != "v1":
            identity["cache_revision"] = self.cache_revision
        return canonical_json_sha256(identity), identity

    def _cache_path(self, request_sha256: str) -> Path:
        return self.raw_cache_root / self.provider / request_sha256[:2] / f"{request_sha256}.json"

    def _cache_candidates(self, request_sha256: str) -> tuple[Path, ...]:
        relative = Path(self.provider) / request_sha256[:2] / f"{request_sha256}.json"
        return tuple(root / relative for root in (self.raw_cache_root, *self.read_cache_roots))

    @staticmethod
    def _read_cache_entry(path: Path) -> tuple[Any, bytes]:
        if path.is_symlink() or not path.is_file():
            raise ValueError("Provider cache entry must be a regular file")
        body = path.read_bytes()
        return json.loads(body), body

    def _quarantine_corrupt_primary_cache(self, path: Path) -> None:
        """Move a corrupt writable entry aside without touching fallback caches."""

        if path.parent.parent.parent != self.raw_cache_root:
            return
        quarantine = path.with_name(f"{path.name}.corrupt-{uuid.uuid4().hex}")
        try:
            path.replace(quarantine)
        except FileNotFoundError:
            # Another process may already have repaired the same immutable key.
            return

    def _load_cached_response(
        self,
        request_sha256: str,
    ) -> tuple[Any, bytes, Path] | None:
        for index, candidate in enumerate(self._cache_candidates(request_sha256)):
            if not candidate.exists() and not candidate.is_symlink():
                continue
            try:
                payload, body = self._read_cache_entry(candidate)
            except (OSError, ValueError):
                if index == 0 and not candidate.is_symlink():
                    self._quarantine_corrupt_primary_cache(candidate)
                # Read-only fallback namespaces are never mutated. A later valid
                # fallback or a fresh network response may repair the primary.
                continue
            return payload, body, candidate
        return None

    def _publish_cache_entry(
        self,
        path: Path,
        *,
        body: bytes,
        payload: Any,
    ) -> tuple[Any, bytes]:
        """Publish validated JSON with create-if-absent atomicity across processes."""

        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
        try:
            with temporary.open("xb") as stream:
                stream.write(body)
                stream.flush()
                os.fsync(stream.fileno())
            while True:
                try:
                    os.link(temporary, path, follow_symlinks=False)
                    return payload, body
                except FileExistsError as error:
                    if path.is_symlink():
                        raise ValueError(
                            "Provider cache entry must not be a symbolic link"
                        ) from error
                    try:
                        existing_payload, existing_body = self._read_cache_entry(path)
                    except (OSError, ValueError):
                        self._quarantine_corrupt_primary_cache(path)
                        continue
                    return existing_payload, existing_body
        finally:
            temporary.unlink(missing_ok=True)

    def _record(
        self,
        *,
        request_sha256: str,
        body: bytes,
        cache_hit: bool,
        requested_at: str,
    ) -> RequestRecord:
        response_digest = hashlib.sha256(body).hexdigest()
        relative = Path(self.provider) / request_sha256[:2] / f"{request_sha256}.json"
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
        self._raise_if_provider_stopped()
        if self.request_budget is not None:
            self.request_budget.check_time()
        request_sha256, _identity = self._identity(endpoint=endpoint, params=params)
        cache_path = self._cache_path(request_sha256)
        cached = self._load_cached_response(request_sha256)
        if cached is not None:
            payload, body, cached_path = cached
            return (
                payload,
                self._record(
                    request_sha256=request_sha256,
                    body=body,
                    cache_hit=True,
                    requested_at=datetime.fromtimestamp(
                        cached_path.stat().st_mtime, tz=UTC
                    ).isoformat(),
                ),
            )

        last_category = "unknown"
        last_retryable = False
        last_status_code: int | None = None
        last_retry_after_seconds: float | None = None
        last_rate_limit_limit: int | None = None
        last_rate_limit_remaining: int | None = None
        attempt = 0
        backoff_wait_count = 0
        total_backoff_seconds = 0.0
        last_backoff_seconds: float | None = None
        proposed_backoff_seconds: float | None = None
        exit_reason: str | None = None
        while True:
            self._raise_if_provider_stopped()
            attempt += 1
            self._reserve_request_slot()
            self._raise_if_provider_stopped()
            requested_at = datetime.now(UTC).isoformat()
            try:
                if self.request_budget is not None:
                    self.request_budget.consume(self.provider)
                response = self._send(endpoint, params)
                if (
                    response.status_code in self.retryable_status_codes
                    or response.status_code >= 500
                ):
                    raise requests.HTTPError(
                        f"Retryable HTTP {response.status_code}",
                        response=response,
                    )
                response.raise_for_status()
                body = response.content
                payload = json.loads(body)
                payload, body = self._publish_cache_entry(
                    cache_path,
                    body=body,
                    payload=payload,
                )
                return (
                    payload,
                    self._record(
                        request_sha256=request_sha256,
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
                elif (
                    response.status_code in self.retryable_status_codes
                    or response.status_code >= 500
                ):
                    last_category = (
                        "access_temporarily_denied"
                        if response.status_code == 403
                        else "provider_unavailable"
                    )
                    last_retryable = True
                else:
                    last_category = "provider_rejected"
                    last_retryable = False
                if not last_retryable:
                    exit_reason = "non_retryable_provider_response"
                    break
                proposed_backoff_seconds = min(
                    float(2 ** min(attempt - 1, 12)),
                    3600.0,
                )
                if last_retry_after_seconds is not None:
                    proposed_backoff_seconds = max(
                        proposed_backoff_seconds,
                        last_retry_after_seconds,
                    )
                if self.max_attempts is not None and attempt >= self.max_attempts:
                    exit_reason = "maximum_attempts_reached"
                    break
                if (
                    self.max_backoff_seconds is not None
                    and proposed_backoff_seconds > self.max_backoff_seconds
                ):
                    exit_reason = "proposed_backoff_exceeds_maximum"
                    break
                if self.request_budget is not None:
                    self.request_budget.check_wait(proposed_backoff_seconds)
                self.sleeper(proposed_backoff_seconds)
                backoff_wait_count += 1
                total_backoff_seconds += proposed_backoff_seconds
                last_backoff_seconds = proposed_backoff_seconds
        request_error = ProviderRequestError(
            provider=self.provider,
            category=last_category,
            attempts=attempt,
            retryable=last_retryable,
            status_code=last_status_code,
            retry_after_seconds=last_retry_after_seconds,
            rate_limit_limit=last_rate_limit_limit,
            rate_limit_remaining=last_rate_limit_remaining,
            backoff_wait_count=backoff_wait_count,
            total_backoff_seconds=total_backoff_seconds,
            last_backoff_seconds=last_backoff_seconds,
            proposed_backoff_seconds=proposed_backoff_seconds,
            max_backoff_seconds=self.max_backoff_seconds,
            exit_reason=exit_reason,
        )
        if exit_reason == "proposed_backoff_exceeds_maximum":
            self._stop_provider(request_error)
        raise request_error from None
