#!/usr/bin/env python3
"""Atomically manage the local control-plane credential and metadata dotenv."""

import argparse
import os
import re
import shutil
import sys
import tempfile
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, NoReturn, Optional, Set

ALLOWED_KEYS = {
    "CLOUDFLARE_ACCOUNT_ID",
    "CLOUDFLARE_API_TOKEN",
    "CLOUDFLARE_WORKERS_SUBDOMAIN",
    "GCP_CLOUD_RUN_REGION",
    "GCP_PROJECT_ID",
    "GCP_TPEX_RELAY_SECRET",
    "GCP_TPEX_RELAY_SERVICE",
    "RUNPOD_API_KEY",
    "RUNPOD_NETWORK_VOLUME_ID",
    "RUNPOD_TPEX_PROXY_SECRET_NAME",
    "RUNPOD_DATACENTER_ID",
    "RUNPOD_S3_ACCESS_KEY_ID",
    "RUNPOD_S3_SECRET_ACCESS_KEY",
    "RUNPOD_S3_REGION",
    "RUNPOD_S3_ENDPOINT",
    "TPEX_PROXY_SHARED_SECRET",
    "TPEX_PROXY_URL",
}
SECRET_KEYS = {
    "CLOUDFLARE_API_TOKEN",
    "RUNPOD_API_KEY",
    "RUNPOD_S3_ACCESS_KEY_ID",
    "RUNPOD_S3_SECRET_ACCESS_KEY",
    "TPEX_PROXY_SHARED_SECRET",
}
ASSIGNMENT_PATTERN = re.compile(r"^(?:export[ \t]+)?([A-Z][A-Z0-9_]*)=(.*)$")


class DotenvError(ValueError):
    """Raised when a dotenv update is unsafe or malformed."""


def _fail(message: str) -> NoReturn:
    raise DotenvError(message)


def _decode_value(value: str, key: str) -> str:
    if value.startswith(('"', "'")):
        quote = value[0]
        if len(value) < 2 or value[-1] != quote:
            _fail(f"Malformed quoted value for {key}")
        value = value[1:-1]
    if any(character in value for character in ("\n", "\r", "\0")):
        _fail(f"{key} contains a control character")
    return value


def _parse_values(text: str, label: str) -> Dict[str, str]:
    values = {}  # type: Dict[str, str]
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = ASSIGNMENT_PATTERN.fullmatch(raw_line)
        if match is None:
            _fail(f"Unsupported line in {label}:{line_number}")
        key, raw_value = match.groups()
        if key not in ALLOWED_KEYS:
            continue
        if key in values:
            _fail(f"Duplicate {key} in {label}")
        values[key] = _decode_value(raw_value, key)
    return values


def _read_null_updates() -> Dict[str, str]:
    raw = sys.stdin.buffer.read()
    if not raw:
        return {}
    parts = raw.split(b"\0")
    if parts[-1] != b"":
        _fail("NUL-delimited update input is truncated")
    parts.pop()
    if len(parts) % 2:
        _fail("NUL-delimited update input has an unmatched key")
    updates = {}  # type: Dict[str, str]
    for index in range(0, len(parts), 2):
        try:
            key = parts[index].decode("ascii")
            value = parts[index + 1].decode("utf-8")
        except UnicodeDecodeError as error:
            raise DotenvError("Dotenv update input is not valid text") from error
        if key not in ALLOWED_KEYS:
            _fail(f"Unsupported dotenv key: {key}")
        if key in updates:
            _fail(f"Duplicate dotenv update: {key}")
        updates[key] = value
    return updates


def _validate_value(key: str, value: str) -> str:
    if any(character.isspace() or character in {'"', "'", "#", "\0"} for character in value):
        _fail(f"{key} contains characters that cannot be stored safely")
    if key in SECRET_KEYS:
        if not value or len(value) > 512:
            _fail(f"{key} must be a non-empty credential")
    elif key == "RUNPOD_NETWORK_VOLUME_ID":
        if value and re.fullmatch(r"[A-Za-z0-9_-]+", value) is None:
            _fail("RUNPOD_NETWORK_VOLUME_ID contains unsupported characters")
    elif key == "CLOUDFLARE_ACCOUNT_ID":
        if value and re.fullmatch(r"[0-9a-fA-F]{32}", value) is None:
            _fail("CLOUDFLARE_ACCOUNT_ID must contain 32 hexadecimal characters")
    elif key == "CLOUDFLARE_WORKERS_SUBDOMAIN":
        if value and re.fullmatch(
            r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", value
        ) is None:
            _fail("CLOUDFLARE_WORKERS_SUBDOMAIN has an invalid format")
    elif key == "GCP_PROJECT_ID":
        if value and re.fullmatch(r"[a-z][a-z0-9-]{4,28}[a-z0-9]", value) is None:
            _fail("GCP_PROJECT_ID has an invalid format")
    elif key == "GCP_CLOUD_RUN_REGION":
        if value != "asia-east1":
            _fail("GCP_CLOUD_RUN_REGION must be asia-east1")
    elif key == "GCP_TPEX_RELAY_SERVICE":
        if value and (
            len(value) > 49
            or re.fullmatch(r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?", value) is None
        ):
            _fail("GCP_TPEX_RELAY_SERVICE has an invalid Cloud Run service name")
    elif key == "GCP_TPEX_RELAY_SECRET":
        if value and (
            len(value) > 255
            or re.fullmatch(r"[A-Za-z0-9_-]+", value) is None
        ):
            _fail("GCP_TPEX_RELAY_SECRET has an invalid Secret Manager name")
    elif key == "RUNPOD_TPEX_PROXY_SECRET_NAME":
        if value and re.fullmatch(r"[A-Za-z0-9_-]+", value) is None:
            _fail("RUNPOD_TPEX_PROXY_SECRET_NAME contains unsupported characters")
    elif key in {"RUNPOD_DATACENTER_ID", "RUNPOD_S3_REGION"}:
        if re.fullmatch(r"[A-Z0-9]+(?:-[A-Z0-9]+)+", value) is None:
            _fail(f"{key} has an invalid format")
    elif key == "RUNPOD_S3_ENDPOINT":
        prefix = "https://s3api-"
        suffix = ".runpod.io/"
        region = value[len(prefix) : -len(suffix)] if value.startswith(prefix) else ""
        if value != f"https://s3api-{region}.runpod.io/" or re.fullmatch(
            r"[a-z0-9]+(?:-[a-z0-9]+)+", region
        ) is None:
            _fail("RUNPOD_S3_ENDPOINT is not an approved RunPod endpoint")
    elif key == "TPEX_PROXY_URL":
        if value and re.fullmatch(
            r"https://(?:"
            r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?"
            r"\.[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.workers\.dev"
            r"|"
            r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?"
            r"(?:\.[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?)*\.run\.app"
            r")/?",
            value,
        ) is None:
            _fail(
                "TPEX_PROXY_URL must be an approved run.app origin or a preserved "
                "legacy workers.dev origin"
            )
    return value


def _render_template(template_text: str, values: Dict[str, str]) -> str:
    rendered: List[str] = []
    seen: Set[str] = set()
    for line_number, raw_line in enumerate(template_text.splitlines(), start=1):
        match = ASSIGNMENT_PATTERN.fullmatch(raw_line)
        if match is None:
            rendered.append(raw_line)
            continue
        key = match.group(1)
        if key not in ALLOWED_KEYS:
            _fail(f"Unsupported key in dotenv template:{line_number}: {key}")
        if key in seen:
            _fail(f"Duplicate key in dotenv template: {key}")
        seen.add(key)
        rendered.append(f"{key}={values.get(key, '')}")
    missing = ALLOWED_KEYS - seen
    if missing:
        _fail(f"Dotenv template is missing keys: {', '.join(sorted(missing))}")
    return "\n".join(rendered).rstrip() + "\n"


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    except BaseException:
        with suppress(OSError):
            if temporary.exists():
                temporary.unlink()
        raise


def command_apply(arguments: argparse.Namespace) -> int:
    env_path = Path(arguments.env_file).expanduser().resolve()
    template_path = Path(arguments.template).expanduser().resolve()
    if not template_path.is_file() or template_path.is_symlink():
        _fail(f"Dotenv template is unavailable: {template_path}")
    template_text = template_path.read_text(encoding="utf-8")
    current_text = ""
    if env_path.exists():
        if not env_path.is_file() or env_path.is_symlink():
            _fail(f"Refusing to update a non-regular dotenv path: {env_path}")
        current_text = env_path.read_text(encoding="utf-8")
    current_values = _parse_values(current_text, str(env_path))
    updates = {key: _validate_value(key, value) for key, value in _read_null_updates().items()}
    merged = dict(current_values)
    merged.update(updates)
    for key, value in tuple(merged.items()):
        if value:
            merged[key] = _validate_value(key, value)
    datacenter = merged.get("RUNPOD_DATACENTER_ID", "")
    region = merged.get("RUNPOD_S3_REGION", "")
    endpoint = merged.get("RUNPOD_S3_ENDPOINT", "")
    if datacenter and region and datacenter != region:
        _fail("RUNPOD_DATACENTER_ID and RUNPOD_S3_REGION must match")
    if region and endpoint:
        expected_endpoint = f"https://s3api-{region.lower()}.runpod.io/"
        if endpoint != expected_endpoint:
            _fail("RUNPOD_S3_ENDPOINT does not match RUNPOD_S3_REGION")
    for required_key in arguments.require_key:
        if required_key not in ALLOWED_KEYS:
            _fail(f"Unsupported required dotenv key: {required_key}")
        if not merged.get(required_key):
            _fail(f"{required_key} is required; enter it instead of leaving it blank")
    rendered = _render_template(template_text, merged)
    if current_text == rendered and env_path.exists():
        os.chmod(env_path, 0o600)
        print(f"Control-plane dotenv is already current: {env_path}")
        return 0

    backup_path: Optional[Path] = None
    if env_path.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        backup_path = env_path.with_name(f"{env_path.name}.backup-{stamp}")
        shutil.copy2(env_path, backup_path)
        os.chmod(backup_path, 0o600)
    _atomic_write(env_path, rendered)
    print(f"Updated control-plane dotenv: {env_path}")
    if backup_path is not None:
        print(f"Previous dotenv backup: {backup_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True
    apply = subparsers.add_parser("apply-null")
    apply.add_argument("--env-file", required=True)
    apply.add_argument("--template", required=True)
    apply.add_argument("--require-key", action="append", default=[])
    apply.set_defaults(handler=command_apply)
    return parser


def main() -> int:
    parser = build_parser()
    arguments = parser.parse_args()
    try:
        return int(arguments.handler(arguments))
    except (DotenvError, OSError, ValueError) as error:
        print(f"Control-plane dotenv update failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
