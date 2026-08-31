#!/usr/bin/env python3
"""Validate a bounded Cloud Run relay response without printing response bodies."""

from __future__ import annotations

import argparse
import json
import re
import sys
from typing import NoReturn

STATUS_MARKER = b"\n__FIN_TS_HTTP_STATUS__:"
TRANSIENT_STATUSES = {0, 404, 408, 425, 429, 500, 502, 503, 504}
FATAL_RELAY_ERRORS = {
    "method_not_allowed",
    "unauthorized",
    "unsupported_tpex_request",
    "unsupported_warmup_request",
    "tpex_upstream_body_too_large",
    "tpex_upstream_invalid_json",
    "tpex_upstream_redirect_cookie_limit_exceeded",
    "tpex_upstream_redirect_rejected",
}


def _retry(message: str) -> NoReturn:
    print(message)
    raise SystemExit(75)


def _fail(message: str) -> NoReturn:
    print(message)
    raise SystemExit(2)


def _safe_metadata(values: dict[str, str], name: str, pattern: str) -> str:
    value = values.get(name, "")
    return value if re.fullmatch(pattern, value) else ""


def _official_table(payload: dict[str, object]) -> bool:
    if isinstance(payload.get("tables"), list):
        return True
    for key, fields in payload.items():
        if (
            key.startswith("fields")
            and isinstance(fields, list)
            and isinstance(payload.get(f"data{key.removeprefix('fields')}"), list)
        ):
            return True
    return False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("warmup", "official"), required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--expected-region", required=True)
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    raw = sys.stdin.buffer.read()
    body, separator, metadata_raw = raw.rpartition(STATUS_MARKER)
    if not separator:
        _retry(f"{arguments.label}: relay did not return HTTP metadata")
    metadata_lines = metadata_raw.splitlines()
    try:
        status = int(metadata_lines[0].decode("ascii"))
    except (IndexError, UnicodeDecodeError, ValueError):
        _retry(f"{arguments.label}: relay returned invalid HTTP metadata")

    raw_metadata: dict[str, str] = {}
    for line in metadata_lines[1:]:
        name, metadata_separator, value = line.partition(b":")
        if metadata_separator:
            raw_metadata[name.decode("ascii", errors="ignore")] = value.decode(
                "ascii", errors="ignore"
            ).strip()
    diagnostics = {
        "relay_version": _safe_metadata(
            raw_metadata, "__FIN_TS_RELAY_VERSION__", r"[0-9]{1,6}"
        ),
        "relay_region": _safe_metadata(
            raw_metadata, "__FIN_TS_RELAY_REGION__", r"[a-z0-9-]{1,63}"
        ),
        "relay_revision": _safe_metadata(
            raw_metadata, "__FIN_TS_RELAY_REVISION__", r"[A-Za-z0-9._/-]{1,128}"
        ),
        "upstream_redirects": _safe_metadata(
            raw_metadata, "__FIN_TS_UPSTREAM_REDIRECTS__", r"[0-9]{1,3}"
        ),
        "redirect_cookies": _safe_metadata(
            raw_metadata, "__FIN_TS_UPSTREAM_REDIRECT_COOKIES__", r"[0-9]{1,3}"
        ),
    }
    diagnostic_suffix = "".join(
        f", {name}={value}" for name, value in diagnostics.items() if value
    )
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        payload = None

    if arguments.mode == "warmup":
        ready = (
            status == 200
            and isinstance(payload, dict)
            and payload.get("ready") is True
            and payload.get("version") == "1"
            and payload.get("region") == arguments.expected_region
        )
        if ready:
            return 0
    elif status == 200 and isinstance(payload, dict) and _official_table(payload):
        return 0

    relay_error = payload.get("error") if isinstance(payload, dict) else None
    error_label = relay_error if isinstance(relay_error, str) else "none"
    message = (
        f"{arguments.label}: Cloud Run TPEx relay verification failed: "
        f"HTTP {status}, relay_error={error_label}{diagnostic_suffix}"
    )
    if status in TRANSIENT_STATUSES and error_label not in FATAL_RELAY_ERRORS:
        _retry(message)
    _fail(message)


if __name__ == "__main__":
    raise SystemExit(main())
