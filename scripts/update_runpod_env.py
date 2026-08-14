#!/usr/bin/env python3
"""Atomically manage the small, credential-only RunPod dotenv file."""

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
    "RUNPOD_API_KEY",
    "RUNPOD_NETWORK_VOLUME_ID",
    "RUNPOD_DATACENTER_ID",
    "RUNPOD_S3_ACCESS_KEY_ID",
    "RUNPOD_S3_SECRET_ACCESS_KEY",
    "RUNPOD_S3_REGION",
    "RUNPOD_S3_ENDPOINT",
}
SECRET_KEYS = {
    "RUNPOD_API_KEY",
    "RUNPOD_S3_ACCESS_KEY_ID",
    "RUNPOD_S3_SECRET_ACCESS_KEY",
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
    return value


def _render_template(template_text: str, values: Dict[str, str]) -> str:
    rendered = []  # type: List[str]
    seen = set()  # type: Set[str]
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
        expected_endpoint = "https://s3api-{}.runpod.io/".format(region.lower())
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
        print(f"RunPod dotenv is already current: {env_path}")
        return 0

    backup_path = None  # type: Optional[Path]
    if env_path.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        backup_path = env_path.with_name(f"{env_path.name}.backup-{stamp}")
        shutil.copy2(env_path, backup_path)
        os.chmod(backup_path, 0o600)
    _atomic_write(env_path, rendered)
    print(f"Updated credential-only RunPod dotenv: {env_path}")
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
        print(f"RunPod dotenv update failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
