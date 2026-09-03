from __future__ import annotations

import argparse
import copy
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
        "h_start": 3,
        "universe": "all",
        "stocks": [],
        "etfs": [],
        "symbol_limit": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _marker(selection: dict[str, object]) -> dict[str, object]:
    digest = str(selection["dataset_request_sha256"])
    request = selection["dataset_request"]
    assert isinstance(request, dict)
    stage = selection["stage"]
    assert isinstance(stage, dict)
    storage_preparation = SELECTION.dataset_request_identity_payload(request)[
        "storage_preparation"
    ]
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
        "storage_preparation_spec": storage_preparation,
        "storage_preparation_spec_sha256": SELECTION._payload_sha256(
            storage_preparation
        ),
        "data_root_relative": root,
        "raw": {"relative_path": f"{root}/raw/market.parquet"},
        "bar_store_manifest": {
            "relative_path": f"{root}/prepared/bar-store/bar-store.json"
        },
        "symbol_index": {
            "relative_path": f"{root}/prepared/bar-store/symbol-index.parquet"
        },
        "cutoff_ranges": {
            "relative_path": f"{root}/prepared/bar-store/cutoff-ranges.parquet"
        },
        "dataset_manifest": {"relative_path": f"{root}/dataset-manifest.json"},
        "download_manifest": {"relative_path": f"{root}/download-manifest.json"},
        "request_log": {"relative_path": f"{root}/manifests/api-request-log.jsonl"},
        "model_manifest": {"relative_path": "cache/hf-models.json"},
    }


def _unbound_marker(selection: dict[str, object]) -> dict[str, object]:
    marker = _marker(selection)
    for field in (
        "selection_id",
        "selection_sha256",
        "dataset_request_sha256",
        "selected_stage",
        "stage_config_path",
        "stage_config_sha256",
        "requested_dataset",
        "data_root_relative",
    ):
        marker.pop(field)
    return marker


def _write_download_manifest(
    volume_root: Path,
    marker: dict[str, object],
    selection: dict[str, object],
) -> None:
    request = selection["dataset_request"]
    assert isinstance(request, dict)
    universe = request["universe"]
    assert isinstance(universe, dict)
    artifact = marker["download_manifest"]
    assert isinstance(artifact, dict)
    path = volume_root / str(artifact["relative_path"])
    path.parent.mkdir(parents=True)
    SELECTION._atomic_write_json(
        path,
        {
            "dataset_profile": request["profile"],
            "selected_datasets": request["selected_datasets"],
            "date_range": request["date_range"],
            "training_security_scope": (
                "common_stock_adr_tdr_and_allowlisted_unleveraged_equity_etf_v1"
            ),
            "api_policy": {
                "symbol_limit": universe["symbol_limit"],
                "include_delisted": universe["include_delisted_us"],
                "cache_revision": request["revision"],
            },
        },
    )


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


def test_unbound_cpu_marker_is_validated_then_bound_to_the_selection(
    tmp_path: Path,
) -> None:
    project_root = _project_root(tmp_path / "project")
    selection = SELECTION._build_selection(_arguments(), project_root)
    marker = _unbound_marker(selection)
    volume_root = tmp_path / "volume"
    _write_download_manifest(volume_root, marker, selection)

    with pytest.raises(SELECTION.SelectionError, match="requested dataset contract"):
        SELECTION._verify_marker(marker, selection)

    bound = SELECTION._bind_marker(marker, selection, volume_root=volume_root)

    assert "requested_dataset" not in marker
    assert bound["requested_dataset"] == selection["dataset_request"]
    assert bound["dataset_request_sha256"] == selection["dataset_request_sha256"]
    assert bound["selection_id"] == selection["selection_id"]
    assert bound["selection_sha256"] == selection["selection_sha256"]
    SELECTION._verify_marker(bound, selection)


def test_binding_rejects_tampered_unbound_dataset_content(tmp_path: Path) -> None:
    project_root = _project_root(tmp_path / "project")
    selection = SELECTION._build_selection(_arguments(), project_root)
    marker = _unbound_marker(selection)
    storage = marker["storage_preparation_spec"]
    assert isinstance(storage, dict)
    storage["window_size"] = 256
    marker["storage_preparation_spec_sha256"] = SELECTION._payload_sha256(storage)
    volume_root = tmp_path / "volume"
    _write_download_manifest(volume_root, marker, selection)

    with pytest.raises(SELECTION.SelectionError, match="dataset storage contract"):
        SELECTION._bind_marker(marker, selection, volume_root=volume_root)


def test_training_only_selection_revision_reuses_identical_dataset_marker(
    tmp_path: Path,
) -> None:
    project_root = _project_root(tmp_path)
    prepared = SELECTION._build_selection(_arguments(), project_root)
    marker = _marker(prepared)
    config = project_root / "configs" / "stage1_kronos_base_lora.yaml"
    config.write_text(
        config.read_text(encoding="utf-8") + "\n# Training-only fixture revision.\n",
        encoding="utf-8",
    )
    current = SELECTION._build_selection(_arguments(), project_root)

    assert current["dataset_request_sha256"] == prepared["dataset_request_sha256"]
    assert current["selection_sha256"] != prepared["selection_sha256"]
    SELECTION._verify_marker(marker, current)


def test_dataset_request_envelope_schema_is_not_a_data_identity_input(
    tmp_path: Path,
) -> None:
    request = SELECTION._build_selection(_arguments(), _project_root(tmp_path))[
        "dataset_request"
    ]
    changed = copy.deepcopy(request)
    changed["schema_version"] = 999

    assert SELECTION._payload_sha256(SELECTION._dataset_request_core(request)) == (
        SELECTION._payload_sha256(SELECTION._dataset_request_core(changed))
    )


def test_date_range_change_creates_a_new_dataset_namespace(tmp_path: Path) -> None:
    project_root = _project_root(tmp_path)
    baseline = SELECTION._build_selection(_arguments(), project_root)
    extended = SELECTION._build_selection(
        _arguments(end="2026-08-01"),
        project_root,
    )

    assert baseline["dataset_request_sha256"] != extended[
        "dataset_request_sha256"
    ]


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
    assert environment.pop("STAGE1_US_SYMBOLS") == "AAPL"

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


def test_h_start_changes_training_selection_but_reuses_the_same_bar_store_namespace(
    tmp_path: Path,
) -> None:
    project_root = _project_root(tmp_path)
    first_day = SELECTION._build_selection(_arguments(h_start=1), project_root)
    third_day = SELECTION._build_selection(_arguments(h_start=3), project_root)

    assert first_day["dataset_request_sha256"] == third_day["dataset_request_sha256"]
    assert first_day["selection_sha256"] != third_day["selection_sha256"]
    assert first_day["dataset_request"]["revision"] == "v1"
    assert third_day["dataset_request"]["revision"] == "v1"
    assert first_day["dataset_request"]["preparation"]["alpha_horizons"] == list(
        range(1, 15)
    )
    exports = SELECTION._selection_exports(tmp_path / "selection.json", first_day)
    assert exports["FIN_TS_H_START"] == "1"
    SELECTION._verify_marker(_marker(first_day), third_day)


def test_marker_rejects_a_tampered_storage_contract(tmp_path: Path) -> None:
    selection = SELECTION._build_selection(_arguments(), _project_root(tmp_path))
    marker = _marker(selection)
    storage = marker["storage_preparation_spec"]
    assert isinstance(storage, dict)
    storage["window_size"] = 256
    marker["storage_preparation_spec_sha256"] = SELECTION._payload_sha256(storage)

    with pytest.raises(SELECTION.SelectionError, match="dataset storage contract"):
        SELECTION._verify_marker(marker, selection)


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("schema_version", 999),
        ("processed_schema_version", "renamed-only"),
        ("stride", 999),
        ("effective_sample_stride", 999),
        ("label_kind", "renamed-label"),
        ("signal_timing", "renamed-timing"),
        ("entry_timing", "renamed-entry"),
        ("entry_day_counts_as_holding_day_one", False),
        ("exit_timing", "renamed-exit"),
        ("input_adjustment", "renamed-adjustment"),
        ("training_security_scope", "renamed-audit-scope"),
        ("split_policy", "renamed-split-policy"),
        ("eodhd_split_policy", "renamed-provider-policy"),
        ("us_symbol_limit_policy", "renamed-symbol-policy"),
        ("target_horizon", 13),
        ("diagnostic_horizons", [2, 7]),
        ("flat_volatility_multiplier", 9.0),
        ("embargo_bars", 999),
        ("h_start", 2),
        ("alpha_horizons", [2, 3]),
    ),
)
def test_non_storage_preparation_values_do_not_change_dataset_namespace(
    tmp_path: Path,
    field: str,
    replacement: object,
) -> None:
    request = SELECTION._build_selection(_arguments(), _project_root(tmp_path))[
        "dataset_request"
    ]
    changed = copy.deepcopy(request)
    changed["preparation"][field] = replacement

    assert SELECTION._payload_sha256(SELECTION._dataset_request_core(request)) == (
        SELECTION._payload_sha256(SELECTION._dataset_request_core(changed))
    )


def test_provider_order_does_not_change_dataset_namespace(tmp_path: Path) -> None:
    request = SELECTION._build_selection(
        _arguments(data_profile="us_tw_eodhd"),
        _project_root(tmp_path),
    )["dataset_request"]
    reordered = copy.deepcopy(request)
    reordered["selected_datasets"] = list(reversed(request["selected_datasets"]))

    assert SELECTION._payload_sha256(SELECTION._dataset_request_core(request)) == (
        SELECTION._payload_sha256(SELECTION._dataset_request_core(reordered))
    )


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("bar_store_schema_version", "2.0"),
        ("storage_kind", "different-storage-layout"),
        ("window_size", 256),
        ("max_horizon", 15),
        ("window_materialized", True),
        ("labels_materialized", True),
        ("benchmark_mapping_sha256", "f" * 64),
        ("max_abs_log_return", 0.25),
        ("train_fraction", 0.60),
        ("validation_fraction", 0.20),
        ("purge_bars", 30),
        ("effective_embargo_bars", 20),
    ),
)
def test_storage_preparation_values_change_dataset_namespace(
    tmp_path: Path,
    field: str,
    replacement: object,
) -> None:
    request = SELECTION._build_selection(_arguments(), _project_root(tmp_path))[
        "dataset_request"
    ]
    changed = copy.deepcopy(request)
    changed["preparation"][field] = replacement

    assert SELECTION._payload_sha256(SELECTION._dataset_request_core(request)) != (
        SELECTION._payload_sha256(SELECTION._dataset_request_core(changed))
    )


def test_eodhd_volume_semantics_are_part_of_the_immutable_dataset_contract() -> None:
    assert SELECTION.PREPARATION_CONTRACT["schema_version"] == 5
    assert SELECTION.PREPARATION_CONTRACT["eodhd_split_policy"] == (
        "per_symbol_full_historical_splits_reconstruct_unadjusted_volume_v2"
    )
    assert SELECTION.PREPARATION_CONTRACT["training_security_scope"] == (
        "common_stock_adr_tdr_and_allowlisted_unleveraged_equity_etf_v1"
    )
    assert SELECTION.PREPARATION_CONTRACT["split_policy"] == (
        "global_chronological_cutoff_with_purge_embargo_and_label_end_guard_v1"
    )


def test_provider_acquisition_values_cannot_change_selection_identity(tmp_path: Path) -> None:
    project_root = _project_root(tmp_path)
    baseline = SELECTION._build_selection(
        _arguments(data_profile="us_only_eodhd", universe="all"),
        project_root,
    )
    with_transient_values = SELECTION._build_selection(
        _arguments(
            data_profile="us_only_eodhd",
            universe="all",
            max_api_calls=1,
            eodhd_qps="1",
            taiwan_qps="1",
            max_backoff_seconds=1,
        ),
        project_root,
    )

    assert (
        baseline["dataset_request_sha256"]
        == with_transient_values["dataset_request_sha256"]
    )
    assert baseline["selection_sha256"] == with_transient_values["selection_sha256"]
    assert baseline["schema_version"] == SELECTION.SELECTION_SCHEMA_VERSION == 3
    assert baseline["dataset_request"]["schema_version"] == 3
    assert "acquisition_policy" not in baseline
    exports = SELECTION._selection_exports(tmp_path / "selection.json", baseline)
    for key in (
        "STAGE1_MAX_API_CALLS",
        "STAGE1_EODHD_QPS",
        "STAGE1_TAIWAN_QPS",
    ):
        assert key not in exports


@pytest.mark.parametrize(
    "option",
    ("--max-api-calls", "--eodhd-qps", "--taiwan-qps", "--maxBackoff"),
)
def test_provider_acquisition_options_are_rejected_by_configure(
    tmp_path: Path,
    option: str,
) -> None:
    with pytest.raises(SystemExit):
        SELECTION.build_parser().parse_args(
            [
                "create",
                "--project-root",
                str(tmp_path),
                option,
                "1",
            ]
        )


def test_selection_defaults_only_cover_dataset_semantics(tmp_path: Path) -> None:
    arguments = SELECTION.build_parser().parse_args(
        ["create", "--project-root", str(tmp_path)]
    )

    assert arguments.start == "2005-01-01"
    assert arguments.end is None
    assert arguments.h_start == 3
    for name in ("max_api_calls", "eodhd_qps", "taiwan_qps", "max_backoff_seconds"):
        assert not hasattr(arguments, name)


def test_selection_storage_defaults_have_one_dependency_free_source() -> None:
    assert {
        field: SELECTION._BASE_PREPARATION_CONTRACT[field]
        for field in SELECTION.DATASET_STORAGE_PREPARATION_FIELDS
    } == SELECTION.DEFAULT_DATASET_STORAGE_PREPARATION

    cpu_prepare = (ROOT / "scripts/runpod_cpu_prepare.sh").read_text(encoding="utf-8")
    for duplicated_option in (
        "--window-size",
        "--max-abs-log-return",
        "--train-fraction",
        "--validation-fraction",
        "--purge-bars",
        "--effective-embargo-bars",
    ):
        assert duplicated_option not in cpu_prepare


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
    exports = SELECTION._selection_exports(tmp_path / "selection.json", revised)
    assert exports["RUNPOD_DATASET_REVISION"] == "provider-refresh-20260814"


def test_tw_only_rejects_us_symbol_limit(tmp_path: Path) -> None:
    project_root = _project_root(tmp_path)

    with pytest.raises(SELECTION.SelectionError, match="tw_only"):
        SELECTION._build_selection(_arguments(symbol_limit=10), project_root)
