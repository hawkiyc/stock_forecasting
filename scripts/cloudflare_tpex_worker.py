#!/usr/bin/env python3
"""Deploy the restricted TPEx relay through the Cloudflare Workers API."""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, NoReturn

CLOUDFLARE_API_BASE = "https://api.cloudflare.com/client/v4"
DEFAULT_WORKER_NAME = "stock-forecasting-tpex-proxy"
COMPATIBILITY_DATE = "2026-08-28"
PLACEMENT_REGION = "gcp:asia-east1"
MAX_SOURCE_BYTES = 1_000_000
DNS_LABEL_PATTERN = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?")


class DeploymentError(RuntimeError):
    """Raised when a safe Cloudflare deployment step fails."""


class CloudflareAPIError(DeploymentError):
    """Preserve an HTTP status so missing resources can be handled explicitly."""

    def __init__(self, message: str, *, status_code: int) -> None:
        self.status_code = status_code
        super().__init__(message)


def _fail(message: str) -> NoReturn:
    raise DeploymentError(message)


def _required_environment(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        _fail(f"{name} is required")
    return value


def _safe_error_text(raw: bytes, *secret_values: str) -> str:
    text = raw[:65536].decode("utf-8", errors="replace")
    for value in secret_values:
        if value:
            text = text.replace(value, "<redacted>")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return "Cloudflare returned a non-JSON error response"
    errors = payload.get("errors") if isinstance(payload, dict) else None
    if not isinstance(errors, list):
        return "Cloudflare rejected the request without a structured error"
    messages = [
        str(item.get("message", "unknown error"))
        for item in errors
        if isinstance(item, dict)
    ]
    return "; ".join(messages) or "Cloudflare rejected the request"


class CloudflareClient:
    def __init__(
        self,
        *,
        account_id: str,
        api_token: str,
        additional_redactions: tuple[str, ...] = (),
    ) -> None:
        if re.fullmatch(r"[0-9a-fA-F]{32}", account_id) is None:
            _fail("CLOUDFLARE_ACCOUNT_ID must contain exactly 32 hexadecimal characters")
        if len(api_token) < 20 or len(api_token) > 512:
            _fail("CLOUDFLARE_API_TOKEN has an invalid length")
        self.account_id = account_id
        self.api_token = api_token
        self.redactions = (api_token, *additional_redactions)

    def request(
        self,
        method: str,
        path: str,
        *,
        body: bytes | None = None,
        content_type: str | None = None,
    ) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Accept": "application/json",
        }
        if content_type is not None:
            headers["Content-Type"] = content_type
        request = urllib.request.Request(
            f"{CLOUDFLARE_API_BASE}{path}",
            data=body,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                raw = response.read()
        except urllib.error.HTTPError as error:
            raw = error.read()
            raise CloudflareAPIError(
                f"Cloudflare API {method} {path} failed with HTTP {error.code}: "
                f"{_safe_error_text(raw, *self.redactions)}",
                status_code=error.code,
            ) from None
        except urllib.error.URLError as error:
            reason = str(error.reason)
            for value in self.redactions:
                if value:
                    reason = reason.replace(value, "<redacted>")
            raise DeploymentError(
                f"Cloudflare API {method} {path} could not be reached: {reason}"
            ) from None
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as error:
            raise DeploymentError("Cloudflare API returned invalid JSON") from error
        if not isinstance(payload, dict) or payload.get("success") is not True:
            raise DeploymentError(_safe_error_text(raw, *self.redactions))
        return payload


def _json_body(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("utf-8")


def _multipart_upload(source: bytes, shared_secret: str) -> tuple[bytes, str]:
    boundary = f"----stock-forecasting-{secrets.token_hex(16)}"
    metadata = {
        "main_module": "index.mjs",
        "compatibility_date": COMPATIBILITY_DATE,
        "placement": {"region": PLACEMENT_REGION},
        "bindings": [
            {
                "type": "secret_text",
                "name": "TPEX_PROXY_SHARED_SECRET",
                "text": shared_secret,
            }
        ],
        "annotations": {
            "workers/message": "Deploy restricted TPEx relay for stock_forecasting"
        },
    }
    chunks = [
        f"--{boundary}\r\n".encode("ascii"),
        b'Content-Disposition: form-data; name="metadata"\r\n',
        b"Content-Type: application/json\r\n\r\n",
        _json_body(metadata),
        b"\r\n",
        f"--{boundary}\r\n".encode("ascii"),
        b'Content-Disposition: form-data; name="index.mjs"; filename="index.mjs"\r\n',
        b"Content-Type: application/javascript+module\r\n\r\n",
        source,
        b"\r\n",
        f"--{boundary}--\r\n".encode("ascii"),
    ]
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def _account_subdomain(client: CloudflareClient, desired: str) -> str:
    path = f"/accounts/{client.account_id}/workers/subdomain"
    payload: dict[str, Any] | None
    try:
        payload = client.request("GET", path)
    except CloudflareAPIError as error:
        if error.status_code != 404:
            raise
        payload = None
    result = payload.get("result") if payload is not None else None
    actual = result.get("subdomain") if isinstance(result, dict) else None
    if not isinstance(actual, str) or DNS_LABEL_PATTERN.fullmatch(actual) is None:
        if not desired:
            raise DeploymentError(
                "The Cloudflare account has no workers.dev subdomain; rerun "
                "tpex-proxy configure and provide a globally unique subdomain"
            ) from None
        payload = client.request(
            "PUT",
            path,
            body=_json_body({"subdomain": desired}),
            content_type="application/json",
        )
        result = payload.get("result")
        actual = result.get("subdomain") if isinstance(result, dict) else None
    if not isinstance(actual, str) or DNS_LABEL_PATTERN.fullmatch(actual) is None:
        _fail("Cloudflare did not return a valid workers.dev account subdomain")
    if desired and actual != desired:
        _fail(
            "CLOUDFLARE_WORKERS_SUBDOMAIN does not match the existing account subdomain"
        )
    return actual


def deploy(arguments: argparse.Namespace) -> int:
    source_path = Path(arguments.source).expanduser().resolve()
    if not source_path.is_file() or source_path.is_symlink():
        _fail(f"Worker source is unavailable: {source_path}")
    source = source_path.read_bytes()
    if not source or len(source) > MAX_SOURCE_BYTES:
        _fail("Worker source must be non-empty and no larger than 1 MB")
    if re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", arguments.worker_name) is None:
        _fail("Worker name contains unsupported characters")

    api_token = _required_environment("CLOUDFLARE_API_TOKEN")
    account_id = _required_environment("CLOUDFLARE_ACCOUNT_ID")
    shared_secret = _required_environment("TPEX_PROXY_SHARED_SECRET")
    desired_subdomain = os.environ.get("CLOUDFLARE_WORKERS_SUBDOMAIN", "")
    if len(shared_secret) < 32 or len(shared_secret) > 512:
        _fail("TPEX_PROXY_SHARED_SECRET must contain 32 through 512 characters")
    if desired_subdomain and DNS_LABEL_PATTERN.fullmatch(desired_subdomain) is None:
        _fail("CLOUDFLARE_WORKERS_SUBDOMAIN has an invalid format")
    for secret_value in (api_token, shared_secret):
        if secret_value.encode("utf-8") in source:
            _fail("Worker source contains a configured credential value")
    if re.search(
        rb"-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----|"
        rb"(?:hf_|sk-proj-|gh[pousr]_)[A-Za-z0-9_-]{20,}",
        source,
    ):
        _fail("Worker source contains a high-confidence credential pattern")

    client = CloudflareClient(
        account_id=account_id,
        api_token=api_token,
        additional_redactions=(shared_secret,),
    )
    account_subdomain = _account_subdomain(client, desired_subdomain)
    upload_body, content_type = _multipart_upload(source, shared_secret)
    script_path = (
        f"/accounts/{client.account_id}/workers/scripts/{arguments.worker_name}"
    )
    client.request("PUT", script_path, body=upload_body, content_type=content_type)
    client.request(
        "POST",
        f"{script_path}/subdomain",
        body=_json_body({"enabled": True, "previews_enabled": False}),
        content_type="application/json",
    )
    print(f"https://{arguments.worker_name}.{account_subdomain}.workers.dev")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    deploy_parser = subparsers.add_parser("deploy")
    deploy_parser.add_argument("--source", required=True)
    deploy_parser.add_argument("--worker-name", default=DEFAULT_WORKER_NAME)
    deploy_parser.set_defaults(handler=deploy)
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    try:
        return int(arguments.handler(arguments))
    except (DeploymentError, OSError, ValueError) as error:
        print(f"Cloudflare TPEx Worker deployment failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
