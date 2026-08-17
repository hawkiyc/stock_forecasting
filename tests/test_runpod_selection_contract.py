from __future__ import annotations

import argparse
import importlib.util
import shutil
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = ROOT / "scripts/runpod_selection.py"
SPEC = importlib.util.spec_from_file_location("runpod_selection_contract", HELPER_PATH)
assert SPEC is not None and SPEC.loader is not None
SELECTION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SELECTION)


def test_declared_python_namespace_matches_source_directory() -> None:
    project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert 'packages = [{ include = "stock_forecasting", from = "src" }]' in project
    assert "fin_ts_multimodal" not in project


def _project_root(tmp_path: Path) -> Path:
    config_root = tmp_path / "configs"
    config_root.mkdir()
    for name in ("stage1_kronos_base_lora.yaml", "stage2_kronos_base_lora.yaml"):
        shutil.copy2(ROOT / "configs" / name, config_root / name)
    return tmp_path


def _arguments(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "stage": "stage1",
        "data_profile": "tw_only",
        "dataset_revision": "v1",
        "start": "2020-01-01",
        "end": "2024-01-01",
        "universe": "all",
        "stocks": [],
        "etfs": [],
        "symbol_limit": None,
        "max_api_calls": 100000,
        "eodhd_qps": "16",
        "taiwan_qps": "0.5",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _marker(selection: dict[str, object]) -> dict[str, object]:
    digest = str(selection["dataset_request_sha256"])
    request = selection["dataset_request"]
    assert isinstance(request, dict)
    stage = selection["stage"]
    assert isinstance(stage, dict)
    root = f"datasets/{digest}"
    return {
        "schema_version": 2,
        "kind": "stage1-dataset",
        "state": "ready",
        "selection_id": selection["selection_id"],
        "selection_sha256": selection["selection_sha256"],
        "dataset_request_sha256": digest,
        "selected_stage": stage["name"],
        "stage_config_path": stage["config_path"],
        "stage_config_sha256": stage["config_sha256"],
        "dataset_profile": request["profile"],
        "selected_datasets": request["selected_datasets"],
        "date_range": request["date_range"],
        "requested_dataset": request,
        "data_root_relative": root,
        "raw": {"relative_path": f"{root}/raw/market.parquet"},
        "processed": {"relative_path": f"{root}/processed/windows.parquet"},
        "dataset_manifest": {"relative_path": f"{root}/dataset-manifest.json"},
        "download_manifest": {"relative_path": f"{root}/download-manifest.json"},
        "request_log": {"relative_path": f"{root}/manifests/api-request-log.jsonl"},
        "model_manifest": {"relative_path": "cache/hf-models.json"},
    }


def _selection_environment(
    selection_path: Path,
    selection: dict[str, object],
) -> dict[str, str]:
    environment = SELECTION._selection_exports(selection_path, selection)
    environment["NETWORK_VOLUME_ROOT"] = "/runpod-volume"
    environment["RUNPOD_REMOTE_SELECTION_PATH"] = str(selection_path)
    return environment


def test_exact_cpu_marker_matches_active_training_selection(tmp_path: Path) -> None:
    project_root = _project_root(tmp_path)
    selection = SELECTION._build_selection(_arguments(), project_root)

    SELECTION._verify_marker(_marker(selection), selection)


def test_cpu_tw_only_marker_rejects_us_tw_training_selection(tmp_path: Path) -> None:
    project_root = _project_root(tmp_path)
    prepared = SELECTION._build_selection(_arguments(), project_root)
    requested = SELECTION._build_selection(
        _arguments(data_profile="us_tw_eodhd", universe="all"),
        project_root,
    )

    with pytest.raises(SELECTION.SelectionError, match="dataset profile mismatch"):
        SELECTION._verify_marker(_marker(prepared), requested)


def test_missing_optional_empty_pod_environment_values_are_normalized(
    tmp_path: Path,
) -> None:
    project_root = _project_root(tmp_path)
    selection = SELECTION._build_selection(
        _arguments(data_profile="us_tw_eodhd", universe="all"),
        project_root,
    )
    selection_path = tmp_path / "selection.json"
    environment = _selection_environment(selection_path, selection)
    for key in SELECTION.OPTIONAL_EMPTY_ENVIRONMENT_KEYS:
        assert environment[key] == ""
        environment.pop(key)

    SELECTION._verify_environment(selection_path, selection, environment)


def test_missing_nonempty_optional_pod_environment_value_is_rejected(
    tmp_path: Path,
) -> None:
    project_root = _project_root(tmp_path)
    selection = SELECTION._build_selection(
        _arguments(
            data_profile="us_tw_eodhd",
            universe="explicit",
            stocks=["AAPL"],
            etfs=[],
        ),
        project_root,
    )
    selection_path = tmp_path / "selection.json"
    environment = _selection_environment(selection_path, selection)
    assert environment.pop("STAGE1_US_SYMBOLS") == "AAPL.US"

    with pytest.raises(SELECTION.SelectionError, match="STAGE1_US_SYMBOLS mismatch"):
        SELECTION._verify_environment(selection_path, selection, environment)


def test_missing_required_pod_environment_value_remains_fail_closed(
    tmp_path: Path,
) -> None:
    project_root = _project_root(tmp_path)
    selection = SELECTION._build_selection(_arguments(), project_root)
    selection_path = tmp_path / "selection.json"
    environment = _selection_environment(selection_path, selection)
    environment.pop("RUNPOD_STAGE")

    with pytest.raises(SELECTION.SelectionError, match="RUNPOD_STAGE mismatch"):
        SELECTION._verify_environment(selection_path, selection, environment)


def test_date_or_universe_changes_dataset_request_identity(tmp_path: Path) -> None:
    project_root = _project_root(tmp_path)
    baseline = SELECTION._build_selection(
        _arguments(data_profile="us_tw_eodhd", universe="all"),
        project_root,
    )
    different_dates = SELECTION._build_selection(
        _arguments(data_profile="us_tw_eodhd", universe="all", end="2023-01-01"),
        project_root,
    )
    explicit_universe = SELECTION._build_selection(
        _arguments(
            data_profile="us_tw_eodhd",
            universe="explicit",
            stocks=["AAPL,MSFT"],
            etfs=["VTI"],
        ),
        project_root,
    )

    assert baseline["dataset_request_sha256"] != different_dates["dataset_request_sha256"]
    assert baseline["dataset_request_sha256"] != explicit_universe["dataset_request_sha256"]


def test_acquisition_rate_changes_selection_but_not_dataset_identity(tmp_path: Path) -> None:
    project_root = _project_root(tmp_path)
    baseline = SELECTION._build_selection(
        _arguments(data_profile="us_only_eodhd", universe="all"),
        project_root,
    )
    slower = SELECTION._build_selection(
        _arguments(data_profile="us_only_eodhd", universe="all", eodhd_qps="1"),
        project_root,
    )

    assert baseline["dataset_request_sha256"] == slower["dataset_request_sha256"]
    assert baseline["selection_sha256"] != slower["selection_sha256"]


def test_official_eodhd_limits_are_selection_defaults(tmp_path: Path) -> None:
    arguments = SELECTION.build_parser().parse_args(
        [
            "create",
            "--project-root",
            str(tmp_path),
        ]
    )

    assert arguments.max_api_calls == 100000
    assert arguments.start == "2005-01-01"
    assert arguments.end is None
    assert float(arguments.eodhd_qps) == pytest.approx(16.0)
    assert float(arguments.eodhd_qps) * 60.0 == pytest.approx(960.0)
    assert float(arguments.eodhd_qps) * 60.0 < 1000.0


def test_selection_requires_an_explicit_end_date(tmp_path: Path) -> None:
    project_root = _project_root(tmp_path)

    with pytest.raises(SELECTION.SelectionError, match="end is required"):
        SELECTION._build_selection(_arguments(end=None), project_root)


def test_explicit_revision_creates_a_new_dataset_namespace(tmp_path: Path) -> None:
    project_root = _project_root(tmp_path)
    first = SELECTION._build_selection(_arguments(dataset_revision="v1"), project_root)
    revised = SELECTION._build_selection(
        _arguments(dataset_revision="provider-refresh-20260814"),
        project_root,
    )

    assert first["dataset_request_sha256"] != revised["dataset_request_sha256"]


def test_tw_only_rejects_us_symbol_limit(tmp_path: Path) -> None:
    project_root = _project_root(tmp_path)

    with pytest.raises(SELECTION.SelectionError, match="tw_only"):
        SELECTION._build_selection(_arguments(symbol_limit=10), project_root)
