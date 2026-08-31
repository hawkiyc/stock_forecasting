#!/usr/bin/env python3
"""Create a non-destructive, uniquely named RunPod Secret for the TPEx relay."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime

RUNPOD_GRAPHQL_ENDPOINT = "https://api.runpod.io/graphql"
RUNPOD_GRAPHQL_USER_AGENT = "stock-forecasting-runpod-control/0.1"


def _redacted_error(raw: bytes, *secrets_to_hide: str) -> str:
    text = raw[:65536].decode("utf-8", errors="replace")
    for value in secrets_to_hide:
        if value:
            text = text.replace(value, "<redacted>")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return "RunPod returned a non-JSON error response"
    errors = payload.get("errors") if isinstance(payload, dict) else None
    if isinstance(errors, list):
        messages = [
            str(item.get("message", "unknown error"))
            for item in errors
            if isinstance(item, dict)
        ]
        if messages:
            return "; ".join(messages)
    gateway_fields = []
    for name in ("error_code", "error_name", "error_category", "detail"):
        value = payload.get(name) if isinstance(payload, dict) else None
        if isinstance(value, (str, int, float, bool)):
            rendered = " ".join(str(value).split())[:500]
            gateway_fields.append(f"{name}={rendered}")
    if gateway_fields:
        return "; ".join(gateway_fields)
    return "RunPod rejected the GraphQL request"


def _required_api_key() -> str:
    api_key = os.environ.get("RUNPOD_API_KEY", "")
    if not api_key:
        raise ValueError("RUNPOD_API_KEY is required")
    return api_key


def _graphql_request(
    *,
    api_key: str,
    query: str,
    additional_redactions: tuple[str, ...] = (),
) -> dict[str, object]:
    endpoint = RUNPOD_GRAPHQL_ENDPOINT + "?" + urllib.parse.urlencode(
        {"api_key": api_key}
    )
    request = urllib.request.Request(
        endpoint,
        data=json.dumps({"query": query}, separators=(",", ":")).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": RUNPOD_GRAPHQL_USER_AGENT,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read()
    except urllib.error.HTTPError as error:
        raw = error.read()
        raise RuntimeError(
            f"RunPod GraphQL request failed with HTTP {error.code}: "
            f"{_redacted_error(raw, api_key, *additional_redactions)}"
        ) from None
    except urllib.error.URLError as error:
        reason = str(error.reason)
        for value in (api_key, *additional_redactions):
            if value:
                reason = reason.replace(value, "<redacted>")
        raise RuntimeError(f"RunPod GraphQL API could not be reached: {reason}") from None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise RuntimeError("RunPod GraphQL API returned invalid JSON") from error
    if not isinstance(payload, dict):
        raise RuntimeError("RunPod GraphQL API returned an invalid response object")
    return payload


def _check_access(api_key: str) -> None:
    payload = _graphql_request(
        api_key=api_key,
        query="query { myself { id } }",
    )
    data = payload.get("data")
    myself = data.get("myself") if isinstance(data, dict) else None
    if not isinstance(myself, dict):
        raise RuntimeError(_redacted_error(json.dumps(payload).encode("utf-8"), api_key))


def _create_secret(api_key: str) -> str:
    shared_secret = os.environ.get("TPEX_PROXY_SHARED_SECRET", "")
    if len(shared_secret) < 32 or len(shared_secret) > 512:
        raise ValueError("TPEX_PROXY_SHARED_SECRET has an invalid length")

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ").lower()
    secret_name = f"tpex_relay_token_{stamp}_{secrets.token_hex(3)}"
    query = (
        "mutation { secretCreate(input: { value: "
        f"{json.dumps(shared_secret)}, name: {json.dumps(secret_name)}"
        " }) { id name } }"
    )
    payload = _graphql_request(
        api_key=api_key,
        query=query,
        additional_redactions=(shared_secret,),
    )
    data = payload.get("data")
    created = data.get("secretCreate") if isinstance(data, dict) else None
    if not isinstance(created, dict) or created.get("name") != secret_name:
        raise RuntimeError(
            _redacted_error(json.dumps(payload).encode("utf-8"), api_key, shared_secret)
        )
    return secret_name


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-access",
        action="store_true",
        help="Verify read-only RunPod GraphQL access without creating a Secret.",
    )
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    api_key = _required_api_key()
    if arguments.check_access:
        _check_access(api_key)
        print("RunPod GraphQL access preflight passed.")
        return 0
    print(_create_secret(api_key))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"RunPod TPEx relay Secret setup failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
