#!/usr/bin/env python3
"""Create and validate immutable RunPod training-selection contracts."""

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from contextlib import suppress
from datetime import date, datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, NoReturn, Optional, Set, Tuple, Union


SCHEMA_VERSION = 1
SELECTION_ID_PATTERN = re.compile(r"selection-[0-9a-f]{16}")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
SYMBOL_PATTERN = re.compile(r"[A-Z0-9.^_-]+")
SAFE_RELATIVE_PATH_PATTERN = re.compile(r"[A-Za-z0-9._/-]+")
REVISION_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
EODHD_DEFAULT_DAILY_API_CALL_LIMIT = 100_000
EODHD_DEFAULT_REQUESTS_PER_MINUTE = 1_000
EODHD_DEFAULT_QPS = str(EODHD_DEFAULT_REQUESTS_PER_MINUTE // 60)

PROFILE_DATASETS = {
    "tw_only": ["tpex_official", "twse_official"],
    "us_only_eodhd": ["eodhd_us"],
    "us_tw_eodhd": ["eodhd_us", "tpex_official", "twse_official"],
}
STAGE_CONFIGS = {
    "stage1": "configs/stage1_kronos_base_lora.yaml",
    "stage2": "configs/stage2_kronos_base_lora.yaml",
}
STAGE_RUNTIME = {
    "stage1": {
        "max_runtime_seconds": 21600,
        "hard_limit_seconds": 25200,
        "terminate_after": "7h",
    },
    "stage2": {
        "max_runtime_seconds": 21600,
        "hard_limit_seconds": 25200,
        "terminate_after": "7h",
    },
}

# These settings define dataset semantics and must change the dataset request digest.
PREPARATION_CONTRACT = {
    "schema_version": 1,
    "processed_schema_version": "3.0",
    "window_size": 128,
    "stride": 5,
    "effective_sample_stride": 1,
    "alpha_horizons": [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    "label_kind": "benchmark_relative_adjusted_log_return",
    "signal_timing": "after_close_t",
    "entry_timing": "regular_session_open_t_plus_1",
    "entry_day_counts_as_holding_day_one": True,
    "exit_timing": "regular_session_close_t_plus_h",
    "input_adjustment": "point_in_time_total_return_ohlc_split_adjusted_volume",
    "eodhd_split_policy": "per_symbol_historical_splits_all_ranges",
    "us_symbol_limit_policy": (
        "up_to_n_etfs_and_n_stocks_active_then_delisted_ticker_plus_vti"
    ),
    "benchmark_mapping_sha256": (
        "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"
    ),
    "target_horizon": 5,
    "diagnostic_horizons": [1, 20],
    "flat_volatility_multiplier": 0.25,
    "max_abs_log_return": 0.5,
    "train_fraction": 0.70,
    "validation_fraction": 0.15,
    "purge_bars": 20,
    "embargo_bars": 5,
    "effective_embargo_bars": 14,
}

EXPORT_KEYS = (
    "RUNPOD_SELECTION_ID",
    "RUNPOD_SELECTION_SHA256",
    "RUNPOD_DATASET_REQUEST_SHA256",
    "RUNPOD_SELECTION_FILE",
    "RUNPOD_REMOTE_SELECTION_RELATIVE_PATH",
    "RUNPOD_REMOTE_SELECTION_PATH",
    "RUNPOD_STAGE",
    "RUNPOD_CONFIG",
    "RUNPOD_STAGE_CONFIG_SHA256",
    "DATA_ROOT",
    "FIN_TS_DATASET_PROFILE",
    "STAGE1_US_SYMBOLS",
    "STAGE1_US_ETF_SYMBOLS",
    "STAGE1_SYMBOL_LIMIT",
    "STAGE1_DATA_START",
    "STAGE1_DATA_END",
    "STAGE1_MAX_API_CALLS",
    "STAGE1_EODHD_QPS",
    "STAGE1_TAIWAN_QPS",
    "MAX_RUNTIME_SECONDS",
    "RUNPOD_HARD_LIMIT_SECONDS",
    "RUNPOD_TERMINATE_AFTER",
)


class SelectionError(ValueError):
    """Raised when a selection or readiness marker violates its contract."""


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _payload_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fail(message: str) -> NoReturn:
    raise SelectionError(message)


def _validate_project_root(value: Union[str, Path]) -> Path:
    root = Path(value).expanduser().resolve()
    if not root.is_dir():
        _fail(f"Project root does not exist: {root}")
    return root


def _validate_relative_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or not SAFE_RELATIVE_PATH_PATTERN.fullmatch(value):
        _fail(f"{label} must be a safe relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        _fail(f"{label} must be a safe relative path")
    return path.as_posix()


def _parse_date(value: str, label: str) -> str:
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as error:
        raise SelectionError(f"{label} must use YYYY-MM-DD") from error
    return parsed.isoformat()


def _positive_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        _fail(f"{label} must be a positive integer")
    return value


def _positive_number_string(value: Any, label: str) -> str:
    text = str(value).strip()
    if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", text):
        _fail(f"{label} must be a positive number")
    if float(text) <= 0:
        _fail(f"{label} must be greater than zero")
    return text


def _normalize_symbols(
    values: Optional[Union[List[str], Tuple[str, ...]]], label: str
) -> List[str]:
    normalized = set()  # type: Set[str]
    for value in values or []:
        for candidate in re.split(r"[\s,]+", value.strip()):
            if not candidate:
                continue
            symbol = candidate.upper()
            if not SYMBOL_PATTERN.fullmatch(symbol):
                _fail(f"{label} contains an unsupported symbol: {candidate}")
            normalized.add(symbol)
    return sorted(normalized)


def _selection_core(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "schema_version": payload.get("schema_version"),
        "stage": payload.get("stage"),
        "dataset_request": payload.get("dataset_request"),
        "acquisition_policy": payload.get("acquisition_policy"),
        "runtime": payload.get("runtime"),
    }


def _dataset_request_core(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "schema_version": payload.get("schema_version"),
        "profile": payload.get("profile"),
        "revision": payload.get("revision"),
        "selected_datasets": payload.get("selected_datasets"),
        "date_range": payload.get("date_range"),
        "universe": payload.get("universe"),
        "preparation": payload.get("preparation"),
    }


def _selection_dir(project_root: Path) -> Path:
    return project_root / ".runpod" / "selections"


def _active_pointer_path(project_root: Path) -> Path:
    return project_root / ".runpod" / "active-selection.json"


def _atomic_write_json(path: Path, payload: Dict[str, Any], mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        with suppress(OSError):
            if temporary.exists():
                temporary.unlink()
        raise


def _load_json_path(path: Path, label: str) -> Dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        _fail(f"{label} is missing or is a symlink: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SelectionError(f"{label} is not valid JSON: {path}") from error
    if not isinstance(payload, dict):
        _fail(f"{label} must contain a JSON object")
    return payload


def _resolve_selection_path(
    project_root: Path,
    selection: Optional[Union[str, Path]],
    *,
    validate_local_config: bool = True,
) -> Tuple[Path, Dict[str, Any]]:
    if selection is None:
        pointer_path = _active_pointer_path(project_root)
        pointer = _load_json_path(pointer_path, "Active selection pointer")
        if pointer.get("schema_version") != SCHEMA_VERSION:
            _fail("Active selection pointer schema is unsupported")
        relative_path = _validate_relative_path(pointer.get("selection_path"), "selection_path")
        candidate = (project_root / relative_path).resolve()
        try:
            candidate.relative_to(project_root)
        except ValueError as error:
            raise SelectionError("Active selection path escapes the project root") from error
        selection_path = candidate
    else:
        selection_path = Path(selection)
        if not selection_path.is_absolute():
            selection_path = project_root / selection_path
        selection_path = selection_path.resolve()

    payload = _load_json_path(selection_path, "Training selection")
    validated = _validate_selection(
        payload,
        project_root=project_root if validate_local_config else None,
    )
    if selection is None:
        if pointer.get("selection_id") != validated["selection_id"]:
            _fail("Active selection pointer has a mismatched selection_id")
        if pointer.get("selection_sha256") != validated["selection_sha256"]:
            _fail("Active selection pointer has a mismatched selection_sha256")
    return selection_path, validated


def _validate_selection(
    payload: Dict[str, Any], *, project_root: Optional[Path] = None
) -> Dict[str, Any]:
    expected_selection_keys = {
        "schema_version",
        "created_at",
        "selection_id",
        "selection_sha256",
        "stage",
        "dataset_request",
        "dataset_request_sha256",
        "acquisition_policy",
        "runtime",
    }
    if set(payload) != expected_selection_keys:
        _fail("Training selection fields are incomplete or unsupported")
    if payload.get("schema_version") != SCHEMA_VERSION:
        _fail("Training selection schema is unsupported")
    selection_id = payload.get("selection_id")
    selection_sha256 = payload.get("selection_sha256")
    dataset_request_sha256 = payload.get("dataset_request_sha256")
    if not isinstance(selection_id, str) or not SELECTION_ID_PATTERN.fullmatch(selection_id):
        _fail("Training selection_id is invalid")
    if not isinstance(selection_sha256, str) or not SHA256_PATTERN.fullmatch(selection_sha256):
        _fail("Training selection_sha256 is invalid")
    if not isinstance(dataset_request_sha256, str) or not SHA256_PATTERN.fullmatch(
        dataset_request_sha256
    ):
        _fail("Dataset request digest is invalid")

    stage = payload.get("stage")
    if not isinstance(stage, dict) or stage.get("name") not in STAGE_CONFIGS:
        _fail("Training selection stage is unsupported")
    expected_config_path = STAGE_CONFIGS[stage["name"]]
    if stage.get("config_path") != expected_config_path:
        _fail("Training stage does not map to the approved config")
    config_sha256 = stage.get("config_sha256")
    if not isinstance(config_sha256, str) or not SHA256_PATTERN.fullmatch(config_sha256):
        _fail("Training config digest is invalid")
    if project_root is not None:
        config_path = (project_root / expected_config_path).resolve()
        try:
            config_path.relative_to(project_root)
        except ValueError as error:
            raise SelectionError("Training config escapes the project root") from error
        if not config_path.is_file() or config_path.is_symlink():
            _fail(f"Approved training config is unavailable: {config_path}")
        if _file_sha256(config_path) != config_sha256:
            _fail(
                "Active selection was created for a different config revision; "
                "run configure_runpod_training.sh again"
            )

    request = payload.get("dataset_request")
    if not isinstance(request, dict) or request.get("schema_version") != SCHEMA_VERSION:
        _fail("Dataset request is invalid")
    if set(request) != {
        "schema_version",
        "profile",
        "revision",
        "selected_datasets",
        "date_range",
        "universe",
        "preparation",
    }:
        _fail("Dataset request fields are incomplete or unsupported")
    profile = request.get("profile")
    if profile not in PROFILE_DATASETS:
        _fail("Dataset profile is unsupported")
    if request.get("selected_datasets") != PROFILE_DATASETS[profile]:
        _fail("Dataset providers disagree with the selected profile")
    revision = request.get("revision")
    if not isinstance(revision, str) or not REVISION_PATTERN.fullmatch(revision):
        _fail("Dataset revision label is invalid")
    date_range = request.get("date_range")
    if not isinstance(date_range, dict):
        _fail("Dataset date range is invalid")
    start = _parse_date(str(date_range.get("start_inclusive", "")), "start_inclusive")
    end = _parse_date(str(date_range.get("end_exclusive", "")), "end_exclusive")
    if start >= end:
        _fail("Dataset start date must precede its exclusive end date")

    universe = request.get("universe")
    if not isinstance(universe, dict) or universe.get("mode") not in {"all", "explicit"}:
        _fail("Dataset universe mode is invalid")
    if set(universe) != {
        "mode",
        "us_stocks",
        "us_etfs",
        "symbol_limit",
        "include_delisted_us",
    }:
        _fail("Dataset universe fields are incomplete or unsupported")
    if universe.get("include_delisted_us") is not True:
        _fail("The point-in-time US universe must include delisted instruments")
    stocks = universe.get("us_stocks")
    etfs = universe.get("us_etfs")
    if (
        not isinstance(stocks, list)
        or stocks != sorted(set(stocks))
        or any(
            not isinstance(value, str) or not SYMBOL_PATTERN.fullmatch(value)
            for value in stocks
        )
        or not isinstance(etfs, list)
        or etfs != sorted(set(etfs))
        or any(not isinstance(value, str) or not SYMBOL_PATTERN.fullmatch(value) for value in etfs)
    ):
        _fail("Dataset universe contains invalid symbols")
    symbol_limit = universe.get("symbol_limit")
    if symbol_limit is not None:
        _positive_int(symbol_limit, "symbol_limit")
    if profile == "tw_only":
        if universe.get("mode") != "all" or stocks or etfs or symbol_limit is not None:
            _fail("tw_only must use the complete official Taiwan universe")
    elif universe.get("mode") == "explicit":
        if not stocks and not etfs:
            _fail("An explicit US universe must include at least one stock or ETF")
        if symbol_limit is not None:
            _fail("symbol_limit cannot be combined with an explicit US universe")
    elif stocks or etfs:
        _fail("US symbol lists require universe mode explicit")

    if request.get("preparation") != PREPARATION_CONTRACT:
        _fail("Dataset preparation contract is unsupported")
    if _payload_sha256(_dataset_request_core(request)) != dataset_request_sha256:
        _fail("Dataset request digest is inconsistent")

    acquisition = payload.get("acquisition_policy")
    if not isinstance(acquisition, dict):
        _fail("Dataset acquisition policy is invalid")
    if set(acquisition) != {"max_api_calls", "eodhd_qps", "taiwan_qps"}:
        _fail("Dataset acquisition policy fields are incomplete or unsupported")
    _positive_int(acquisition.get("max_api_calls"), "max_api_calls")
    _positive_number_string(acquisition.get("eodhd_qps"), "eodhd_qps")
    _positive_number_string(acquisition.get("taiwan_qps"), "taiwan_qps")

    if payload.get("runtime") != STAGE_RUNTIME[stage["name"]]:
        _fail("Training runtime contract is unsupported")
    expected_selection_sha256 = _payload_sha256(_selection_core(payload))
    if selection_sha256 != expected_selection_sha256:
        _fail("Training selection digest is inconsistent")
    if selection_id != f"selection-{selection_sha256[:16]}":
        _fail("Training selection_id does not match its digest")
    return payload


def _build_selection(arguments: argparse.Namespace, project_root: Path) -> Dict[str, Any]:
    stage_name = arguments.stage
    profile = arguments.data_profile
    if stage_name not in STAGE_CONFIGS:
        _fail("stage must be stage1 or stage2")
    if profile not in PROFILE_DATASETS:
        _fail("data profile must be tw_only, us_only_eodhd, or us_tw_eodhd")

    start = _parse_date(arguments.start, "start")
    end = _parse_date(arguments.end, "end")
    if start >= end:
        _fail("Dataset start date must precede its exclusive end date")
    stocks = _normalize_symbols(arguments.stocks, "US stock list")
    etfs = _normalize_symbols(arguments.etfs, "US ETF list")
    universe_mode = arguments.universe
    symbol_limit = arguments.symbol_limit
    if profile == "tw_only":
        if universe_mode != "all" or stocks or etfs or symbol_limit is not None:
            _fail("tw_only cannot accept US symbols or a symbol limit")
    elif universe_mode == "explicit":
        if not stocks and not etfs:
            _fail("An explicit US universe must include at least one stock or ETF")
        if symbol_limit is not None:
            _fail("symbol_limit cannot be combined with an explicit US universe")
    elif stocks or etfs:
        _fail("US symbol lists require --universe explicit")
    if symbol_limit is not None:
        _positive_int(symbol_limit, "symbol_limit")

    config_path = project_root / STAGE_CONFIGS[stage_name]
    if not config_path.is_file() or config_path.is_symlink():
        _fail(f"Approved stage config is unavailable: {config_path}")
    revision = getattr(arguments, "dataset_revision", "v1")
    if not isinstance(revision, str) or not REVISION_PATTERN.fullmatch(revision):
        _fail("dataset revision must be a safe 1-64 character label")
    request = {
        "schema_version": SCHEMA_VERSION,
        "profile": profile,
        "revision": revision,
        "selected_datasets": PROFILE_DATASETS[profile],
        "date_range": {
            "start_inclusive": start,
            "end_exclusive": end,
        },
        "universe": {
            "mode": universe_mode,
            "us_stocks": stocks,
            "us_etfs": etfs,
            "symbol_limit": symbol_limit,
            "include_delisted_us": True,
        },
        "preparation": PREPARATION_CONTRACT,
    }
    payload = {  # type: Dict[str, Any]
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage": {
            "name": stage_name,
            "config_path": STAGE_CONFIGS[stage_name],
            "config_sha256": _file_sha256(config_path),
        },
        "dataset_request": request,
        "dataset_request_sha256": _payload_sha256(_dataset_request_core(request)),
        "acquisition_policy": {
            "max_api_calls": _positive_int(arguments.max_api_calls, "max_api_calls"),
            "eodhd_qps": _positive_number_string(arguments.eodhd_qps, "eodhd_qps"),
            "taiwan_qps": _positive_number_string(arguments.taiwan_qps, "taiwan_qps"),
        },
        "runtime": STAGE_RUNTIME[stage_name],
    }
    payload["selection_sha256"] = _payload_sha256(_selection_core(payload))
    payload["selection_id"] = f"selection-{payload['selection_sha256'][:16]}"
    return _validate_selection(payload, project_root=project_root)


def _activate_selection(project_root: Path, payload: Dict[str, Any]) -> Path:
    selection_path = _selection_dir(project_root) / f"{payload['selection_id']}.json"
    if selection_path.exists():
        existing = _load_json_path(selection_path, "Existing immutable selection")
        _validate_selection(existing, project_root=project_root)
        if existing["selection_sha256"] != payload["selection_sha256"]:
            _fail("Immutable selection ID collision")
        payload = existing
    else:
        _atomic_write_json(selection_path, payload)
    relative_selection_path = selection_path.relative_to(project_root).as_posix()
    _atomic_write_json(
        _active_pointer_path(project_root),
        {
            "schema_version": SCHEMA_VERSION,
            "selection_id": payload["selection_id"],
            "selection_sha256": payload["selection_sha256"],
            "selection_path": relative_selection_path,
        },
    )
    return selection_path


def _prompt_choice(label: str, choices: Tuple[str, ...], default: str) -> str:
    joined = "/".join(choices)
    while True:
        answer = input(f"{label} [{joined}] (default: {default}): ").strip()
        value = answer or default
        if value in choices:
            return value
        print(f"Unsupported value: {value}", file=sys.stderr)


def _prompt_text(label: str, default: str = "") -> str:
    suffix = f" (default: {default})" if default else ""
    answer = input(f"{label}{suffix}: ").strip()
    return answer or default


def _populate_interactive(arguments: argparse.Namespace) -> None:
    arguments.stage = _prompt_choice("Training stage", ("stage1", "stage2"), "stage1")
    arguments.data_profile = _prompt_choice(
        "Dataset profile",
        ("tw_only", "us_only_eodhd", "us_tw_eodhd"),
        "us_tw_eodhd",
    )
    arguments.dataset_revision = _prompt_text("Dataset revision label", "v1")
    arguments.start = _prompt_text("Dataset start date (inclusive)", "2010-01-01")
    arguments.end = _prompt_text("Dataset end date (exclusive)", date.today().isoformat())
    if arguments.data_profile == "tw_only":
        arguments.universe = "all"
        arguments.stocks = []
        arguments.etfs = []
        arguments.symbol_limit = None
    else:
        arguments.universe = _prompt_choice("US universe", ("all", "explicit"), "all")
        if arguments.universe == "explicit":
            arguments.stocks = [_prompt_text("US stocks (comma or space separated)")]
            arguments.etfs = [_prompt_text("US ETFs (comma or space separated)")]
            arguments.symbol_limit = None
        else:
            arguments.stocks = []
            arguments.etfs = []
            limit = _prompt_text(
                "Optional per-type US discovery limit: up to N ETFs and N stocks "
                "(blank means all)"
            )
            arguments.symbol_limit = int(limit) if limit else None
    arguments.max_api_calls = int(
        _prompt_text(
            "HTTP request safety ceiling (official paid-plan default is 100000)",
            str(EODHD_DEFAULT_DAILY_API_CALL_LIMIT),
        )
    )
    arguments.eodhd_qps = _prompt_text(
        "EODHD requests per second (default 16 = 960 requests/minute)",
        EODHD_DEFAULT_QPS,
    )
    arguments.taiwan_qps = _prompt_text("Taiwan requests per second", "0.5")


def _selection_exports(selection_path: Path, payload: Dict[str, Any]) -> Dict[str, str]:
    request = payload["dataset_request"]
    universe = request["universe"]
    acquisition = payload["acquisition_policy"]
    runtime = payload["runtime"]
    digest = payload["dataset_request_sha256"]
    selection_id = payload["selection_id"]
    exports = {
        "RUNPOD_SELECTION_ID": selection_id,
        "RUNPOD_SELECTION_SHA256": payload["selection_sha256"],
        "RUNPOD_DATASET_REQUEST_SHA256": digest,
        "RUNPOD_SELECTION_FILE": str(selection_path),
        "RUNPOD_REMOTE_SELECTION_RELATIVE_PATH": f"lifecycle/selections/{selection_id}.json",
        "RUNPOD_REMOTE_SELECTION_PATH": f"/runpod-volume/lifecycle/selections/{selection_id}.json",
        "RUNPOD_STAGE": payload["stage"]["name"],
        "RUNPOD_CONFIG": payload["stage"]["config_path"],
        "RUNPOD_STAGE_CONFIG_SHA256": payload["stage"]["config_sha256"],
        "DATA_ROOT": f"/runpod-volume/datasets/{digest}",
        "FIN_TS_DATASET_PROFILE": request["profile"],
        "STAGE1_US_SYMBOLS": " ".join(universe["us_stocks"]),
        "STAGE1_US_ETF_SYMBOLS": " ".join(universe["us_etfs"]),
        "STAGE1_SYMBOL_LIMIT": (
            "" if universe["symbol_limit"] is None else str(universe["symbol_limit"])
        ),
        "STAGE1_DATA_START": request["date_range"]["start_inclusive"],
        "STAGE1_DATA_END": request["date_range"]["end_exclusive"],
        "STAGE1_MAX_API_CALLS": str(acquisition["max_api_calls"]),
        "STAGE1_EODHD_QPS": str(acquisition["eodhd_qps"]),
        "STAGE1_TAIWAN_QPS": str(acquisition["taiwan_qps"]),
        "MAX_RUNTIME_SECONDS": str(runtime["max_runtime_seconds"]),
        "RUNPOD_HARD_LIMIT_SECONDS": str(runtime["hard_limit_seconds"]),
        "RUNPOD_TERMINATE_AFTER": runtime["terminate_after"],
    }
    if tuple(exports) != EXPORT_KEYS:
        _fail("Internal selection export order is inconsistent")
    return exports


def _load_marker(value: str) -> Dict[str, Any]:
    if value == "-":
        try:
            payload = json.load(sys.stdin)
        except json.JSONDecodeError as error:
            raise SelectionError("Readiness marker from stdin is invalid JSON") from error
        if not isinstance(payload, dict):
            _fail("Readiness marker must contain a JSON object")
        return payload
    return _load_json_path(Path(value), "Readiness marker")


def _assert_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        _fail(f"{label} mismatch: prepared={actual!r}, selected={expected!r}")


def _validate_marker_artifact_paths(
    marker: Dict[str, Any], dataset_request_sha256: str
) -> None:
    dataset_prefix = f"datasets/{dataset_request_sha256}/"
    for name in ("raw", "processed", "dataset_manifest", "download_manifest", "request_log"):
        artifact = marker.get(name)
        if not isinstance(artifact, dict):
            _fail(f"Readiness marker artifact is missing: {name}")
        relative_path = _validate_relative_path(
            artifact.get("relative_path"), f"{name}.relative_path"
        )
        if not relative_path.startswith(dataset_prefix):
            _fail(
                f"Readiness marker {name} is outside the selected dataset namespace: "
                f"{relative_path}"
            )
    model = marker.get("model_manifest")
    if not isinstance(model, dict) or model.get("relative_path") != "cache/hf-models.json":
        _fail("Readiness marker model manifest path is not approved")


def _verify_marker(marker: Dict[str, Any], selection: Dict[str, Any]) -> None:
    if marker.get("schema_version") != 2 or marker.get("kind") != "stage1-dataset":
        _fail("Readiness marker schema is unsupported")
    if marker.get("state") != "ready":
        _fail(f"Readiness marker is not ready: {marker.get('state', 'missing')}")
    request = selection["dataset_request"]
    _assert_equal(marker.get("dataset_profile"), request["profile"], "dataset profile")
    _assert_equal(
        marker.get("selected_datasets"), request["selected_datasets"], "dataset providers"
    )
    _assert_equal(marker.get("date_range"), request["date_range"], "dataset date range")
    _assert_equal(marker.get("requested_dataset"), request, "requested dataset contract")
    _assert_equal(marker.get("selected_stage"), selection["stage"]["name"], "stage")
    _assert_equal(
        marker.get("stage_config_path"), selection["stage"]["config_path"], "stage config path"
    )
    _assert_equal(
        marker.get("stage_config_sha256"),
        selection["stage"]["config_sha256"],
        "stage config digest",
    )
    _assert_equal(marker.get("selection_id"), selection["selection_id"], "selection_id")
    _assert_equal(
        marker.get("selection_sha256"),
        selection["selection_sha256"],
        "selection_sha256",
    )
    _assert_equal(
        marker.get("dataset_request_sha256"),
        selection["dataset_request_sha256"],
        "dataset_request_sha256",
    )
    expected_data_root = f"datasets/{selection['dataset_request_sha256']}"
    _assert_equal(marker.get("data_root_relative"), expected_data_root, "dataset namespace")
    _validate_marker_artifact_paths(marker, selection["dataset_request_sha256"])


def _bind_marker(
    marker: Dict[str, Any],
    selection: Dict[str, Any],
    *,
    volume_root: Path,
) -> Dict[str, Any]:
    if marker.get("schema_version") != 2 or marker.get("kind") != "stage1-dataset":
        _fail("Readiness marker schema is unsupported")
    if marker.get("state") != "ready":
        _fail("Only a ready dataset marker can be bound to a training selection")
    request = selection["dataset_request"]
    _assert_equal(marker.get("dataset_profile"), request["profile"], "dataset profile")
    _assert_equal(
        marker.get("selected_datasets"), request["selected_datasets"], "dataset providers"
    )
    _assert_equal(marker.get("date_range"), request["date_range"], "dataset date range")
    _validate_marker_artifact_paths(marker, selection["dataset_request_sha256"])

    download_artifact = marker.get("download_manifest")
    relative_download_path = _validate_relative_path(
        download_artifact.get("relative_path"), "download_manifest.relative_path"
    )
    download_path = (volume_root / relative_download_path).resolve()
    resolved_volume_root = volume_root.resolve()
    try:
        download_path.relative_to(resolved_volume_root)
    except ValueError as error:
        raise SelectionError("Download manifest escapes the network volume") from error
    download = _load_json_path(download_path, "Download manifest")
    _assert_equal(download.get("dataset_profile"), request["profile"], "download profile")
    _assert_equal(
        download.get("selected_datasets"), request["selected_datasets"], "download providers"
    )
    _assert_equal(download.get("date_range"), request["date_range"], "download date range")
    api_policy = download.get("api_policy")
    if not isinstance(api_policy, dict):
        _fail("Download manifest API policy is invalid")
    _assert_equal(
        api_policy.get("symbol_limit"),
        request["universe"]["symbol_limit"],
        "download symbol limit",
    )
    _assert_equal(
        api_policy.get("include_delisted"),
        request["universe"]["include_delisted_us"],
        "download delisted-universe policy",
    )
    if request["universe"]["mode"] == "explicit":
        symbols = marker.get("symbols", {}).get("values")
        if not isinstance(symbols, list):
            _fail("Readiness marker resolved universe is invalid")
        missing = sorted(
            (set(request["universe"]["us_stocks"]) | set(request["universe"]["us_etfs"]))
            - set(symbols)
        )
        if missing:
            _fail(f"Downloaded data is missing explicitly selected symbols: {', '.join(missing)}")

    bound = dict(marker)
    bound.update(
        {
            "selection_id": selection["selection_id"],
            "selection_sha256": selection["selection_sha256"],
            "dataset_request_sha256": selection["dataset_request_sha256"],
            "selected_stage": selection["stage"]["name"],
            "stage_config_path": selection["stage"]["config_path"],
            "stage_config_sha256": selection["stage"]["config_sha256"],
            "requested_dataset": request,
            "data_root_relative": f"datasets/{selection['dataset_request_sha256']}",
        }
    )
    _verify_marker(bound, selection)
    return bound


def _verify_environment(
    selection_path: Path,
    selection: Dict[str, Any],
    environment: Dict[str, str],
) -> None:
    exports = _selection_exports(selection_path, selection)
    network_volume_root = environment.get("NETWORK_VOLUME_ROOT") or environment.get(
        "RUNPOD_VOLUME_ROOT"
    )
    if not network_volume_root or not network_volume_root.startswith("/"):
        _fail("NETWORK_VOLUME_ROOT is required to verify the Pod selection environment")
    network_volume_root = network_volume_root.rstrip("/")
    expected = {
        key: exports[key]
        for key in (
            "RUNPOD_SELECTION_ID",
            "RUNPOD_SELECTION_SHA256",
            "RUNPOD_DATASET_REQUEST_SHA256",
            "RUNPOD_STAGE",
            "RUNPOD_CONFIG",
            "RUNPOD_STAGE_CONFIG_SHA256",
            "FIN_TS_DATASET_PROFILE",
            "STAGE1_US_SYMBOLS",
            "STAGE1_US_ETF_SYMBOLS",
            "STAGE1_SYMBOL_LIMIT",
            "STAGE1_DATA_START",
            "STAGE1_DATA_END",
            "STAGE1_MAX_API_CALLS",
            "STAGE1_EODHD_QPS",
            "STAGE1_TAIWAN_QPS",
        )
    }
    expected["DATA_ROOT"] = (
        f"{network_volume_root}/datasets/{selection['dataset_request_sha256']}"
    )
    expected["RUNPOD_REMOTE_SELECTION_PATH"] = str(selection_path)
    for key, expected_value in expected.items():
        _assert_equal(environment.get(key), expected_value, f"Pod environment {key}")


def command_create(arguments: argparse.Namespace) -> int:
    project_root = _validate_project_root(arguments.project_root)
    if arguments.interactive:
        _populate_interactive(arguments)
    required = ("stage", "data_profile", "start", "end", "universe")
    missing = [name for name in required if getattr(arguments, name) is None]
    if missing:
        _fail(f"Missing required selection options: {', '.join(missing)}")
    payload = _build_selection(arguments, project_root)
    path = _activate_selection(project_root, payload)
    request = payload["dataset_request"]
    print(f"Active selection: {payload['selection_id']}")
    print(f"Stage: {payload['stage']['name']} ({payload['stage']['config_path']})")
    print(
        "Dataset: {} {} to {}".format(
            request["profile"],
            request["date_range"]["start_inclusive"],
            request["date_range"]["end_exclusive"],
        )
    )
    print(f"Dataset request SHA-256: {payload['dataset_request_sha256']}")
    print(f"Selection file: {path}")
    return 0


def command_show(arguments: argparse.Namespace) -> int:
    project_root = _validate_project_root(arguments.project_root)
    _, payload = _resolve_selection_path(project_root, arguments.selection)
    json.dump(payload, sys.stdout, ensure_ascii=False, sort_keys=True, indent=2)
    sys.stdout.write("\n")
    return 0


def command_export(arguments: argparse.Namespace) -> int:
    project_root = _validate_project_root(arguments.project_root)
    selection_path, payload = _resolve_selection_path(project_root, arguments.selection)
    exports = _selection_exports(selection_path, payload)
    if arguments.null:
        output = sys.stdout.buffer
        for key, value in exports.items():
            output.write(key.encode("ascii") + b"\0" + value.encode("utf-8") + b"\0")
        output.flush()
    else:
        json.dump(exports, sys.stdout, ensure_ascii=False, sort_keys=True, indent=2)
        sys.stdout.write("\n")
    return 0


def command_verify_marker(arguments: argparse.Namespace) -> int:
    project_root = _validate_project_root(arguments.project_root)
    _, selection = _resolve_selection_path(project_root, arguments.selection)
    marker = _load_marker(arguments.marker)
    _verify_marker(marker, selection)
    print(
        f"Dataset selection ready: {selection['selection_id']} "
        f"({selection['dataset_request_sha256']})"
    )
    return 0


def command_verify_selection_copy(arguments: argparse.Namespace) -> int:
    project_root = _validate_project_root(arguments.project_root)
    _, selected = _resolve_selection_path(project_root, arguments.selection)
    candidate = _load_marker(arguments.candidate)
    candidate = _validate_selection(candidate, project_root=project_root)
    _assert_equal(candidate["selection_id"], selected["selection_id"], "selection_id")
    _assert_equal(
        candidate["selection_sha256"],
        selected["selection_sha256"],
        "selection_sha256",
    )
    _assert_equal(
        candidate["dataset_request_sha256"],
        selected["dataset_request_sha256"],
        "dataset_request_sha256",
    )
    print(f"Remote selection matches active selection: {selected['selection_id']}")
    return 0


def command_bind_marker(arguments: argparse.Namespace) -> int:
    project_root = _validate_project_root(arguments.project_root)
    selection_path, selection = _resolve_selection_path(
        project_root,
        arguments.selection,
        validate_local_config=True,
    )
    _verify_environment(selection_path, selection, dict(os.environ))
    marker_path = Path(arguments.marker).resolve()
    marker = _load_json_path(marker_path, "Readiness marker")
    bound = _bind_marker(marker, selection, volume_root=Path(arguments.volume_root))
    _atomic_write_json(marker_path, bound)
    print(f"Bound dataset marker to selection: {selection['selection_id']}")
    return 0


def command_verify_environment(arguments: argparse.Namespace) -> int:
    project_root = _validate_project_root(arguments.project_root)
    selection_path, selection = _resolve_selection_path(
        project_root,
        arguments.selection,
        validate_local_config=True,
    )
    _verify_environment(selection_path, selection, dict(os.environ))
    print(f"Pod selection environment verified: {selection['selection_id']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create and verify immutable RunPod training selections."
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True

    create = subparsers.add_parser("create")
    create.add_argument("--project-root", required=True)
    create.add_argument("--interactive", action="store_true")
    create.add_argument("--stage", choices=tuple(STAGE_CONFIGS))
    create.add_argument("--data-profile", choices=tuple(PROFILE_DATASETS))
    create.add_argument(
        "--dataset-revision",
        default="v1",
        help="Explicit revision label used to create a new immutable data namespace.",
    )
    create.add_argument("--start", help="Inclusive market-data start date (YYYY-MM-DD).")
    create.add_argument("--end", help="Exclusive market-data end date (YYYY-MM-DD).")
    create.add_argument("--universe", choices=("all", "explicit"))
    create.add_argument(
        "--stocks",
        action="append",
        default=[],
        help="Comma- or space-separated US stocks; repeatable in explicit mode.",
    )
    create.add_argument(
        "--etfs",
        action="append",
        default=[],
        help="Comma- or space-separated US ETFs; repeatable in explicit mode.",
    )
    create.add_argument(
        "--symbol-limit",
        type=int,
        help=(
            "Deterministic bounded US-discovery check in all mode: keep up to N "
            "ETFs and N stocks, then add VTI if required."
        ),
    )
    create.add_argument(
        "--max-api-calls",
        type=int,
        default=EODHD_DEFAULT_DAILY_API_CALL_LIMIT,
        help=(
            "Project-side HTTP request safety ceiling; default 100000 matches the "
            "official paid-plan daily API-call limit."
        ),
    )
    create.add_argument(
        "--eodhd-qps",
        default=EODHD_DEFAULT_QPS,
        help=(
            "Even requests/second pacing; default floors the official 1000/minute "
            "limit to 16 requests/second (960/minute)."
        ),
    )
    create.add_argument("--taiwan-qps", default="0.5")
    create.set_defaults(handler=command_create)

    show = subparsers.add_parser("show")
    show.add_argument("--project-root", required=True)
    show.add_argument("--selection")
    show.set_defaults(handler=command_show)

    export = subparsers.add_parser("export")
    export.add_argument("--project-root", required=True)
    export.add_argument("--selection")
    export.add_argument("--null", action="store_true")
    export.set_defaults(handler=command_export)

    verify_marker = subparsers.add_parser("verify-marker")
    verify_marker.add_argument("--project-root", required=True)
    verify_marker.add_argument("--selection")
    verify_marker.add_argument("--marker", required=True)
    verify_marker.set_defaults(handler=command_verify_marker)

    verify_selection_copy = subparsers.add_parser("verify-selection-copy")
    verify_selection_copy.add_argument("--project-root", required=True)
    verify_selection_copy.add_argument("--selection")
    verify_selection_copy.add_argument("--candidate", required=True)
    verify_selection_copy.set_defaults(handler=command_verify_selection_copy)

    bind_marker = subparsers.add_parser("bind-marker")
    bind_marker.add_argument("--project-root", required=True)
    bind_marker.add_argument("--selection", required=True)
    bind_marker.add_argument("--marker", required=True)
    bind_marker.add_argument("--volume-root", required=True)
    bind_marker.set_defaults(handler=command_bind_marker)

    verify_environment = subparsers.add_parser("verify-environment")
    verify_environment.add_argument("--project-root", required=True)
    verify_environment.add_argument("--selection", required=True)
    verify_environment.set_defaults(handler=command_verify_environment)
    return parser


def main() -> int:
    parser = build_parser()
    arguments = parser.parse_args()
    try:
        return int(arguments.handler(arguments))
    except (OSError, SelectionError, ValueError) as error:
        print(f"Selection contract error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
