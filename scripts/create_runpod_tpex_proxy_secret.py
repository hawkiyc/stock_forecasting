#!/usr/bin/env python3
"""Create a non-destructive, uniquely named RunPod Secret for the TPEx relay."""

from __future__ import annotations

import json
import os
import secrets
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime


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
    return "RunPod rejected the secret creation request"


def main() -> int:
    api_key = os.environ.get("RUNPOD_API_KEY", "")
    shared_secret = os.environ.get("TPEX_PROXY_SHARED_SECRET", "")
    if not api_key:
        raise ValueError("RUNPOD_API_KEY is required")
    if len(shared_secret) < 32 or len(shared_secret) > 512:
        raise ValueError("TPEX_PROXY_SHARED_SECRET has an invalid length")

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ").lower()
    secret_name = f"tpex_proxy_token_{stamp}_{secrets.token_hex(3)}"
    query = (
        "mutation { secretCreate(input: { value: "
        f"{json.dumps(shared_secret)}, name: {json.dumps(secret_name)}"
        " }) { id name } }"
    )
    endpoint = "https://api.runpod.io/graphql?" + urllib.parse.urlencode(
        {"api_key": api_key}
    )
    request = urllib.request.Request(
        endpoint,
        data=json.dumps({"query": query}, separators=(",", ":")).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read()
    except urllib.error.HTTPError as error:
        raise RuntimeError(
            f"RunPod secret creation failed with HTTP {error.code}: "
            f"{_redacted_error(error.read(), api_key, shared_secret)}"
        ) from None
    except urllib.error.URLError as error:
        reason = str(error.reason).replace(api_key, "<redacted>").replace(
            shared_secret, "<redacted>"
        )
        raise RuntimeError(f"RunPod secret API could not be reached: {reason}") from None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise RuntimeError("RunPod secret API returned invalid JSON") from error
    created = payload.get("data", {}).get("secretCreate") if isinstance(payload, dict) else None
    if not isinstance(created, dict) or created.get("name") != secret_name:
        raise RuntimeError(_redacted_error(raw, api_key, shared_secret))
    print(secret_name)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Unable to create RunPod TPEx proxy Secret: {error}", file=sys.stderr)
        raise SystemExit(2) from error
