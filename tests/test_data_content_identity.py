"""Content-only identity boundaries for durable data artifacts."""

from __future__ import annotations

import ast
import shutil
from pathlib import Path

import pytest

from stock_forecasting.data.content_identity import (
    _canonical_ast_dump,
    bar_store_materialization_digest,
    code_content_identity,
    content_identity_digest,
    dataset_content_identity,
    provider_materialization_digest,
    raw_dataset_materialization_digest,
    semantic_definitions_digest,
    semantic_source_paths,
)

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "src" / "stock_forecasting"


def _copy_package(tmp_path: Path) -> Path:
    destination = tmp_path / "stock_forecasting"
    shutil.copytree(PACKAGE_ROOT, destination)
    return destination


def test_canonical_ast_dump_omits_runtime_dependent_empty_fields() -> None:
    syntax = ast.parse(
        "def clean():\n    return None\n",
        feature_version=(3, 12),
    )

    assert _canonical_ast_dump(syntax) == (
        "Module(body=[FunctionDef(name='clean', args=arguments(), "
        "body=[Return(value=Constant(value=None))])])"
    )


def test_semantic_digest_ignores_docs_types_and_error_messages(tmp_path: Path) -> None:
    first = tmp_path / "first.py"
    second = tmp_path / "second.py"
    first.write_text(
        '''def clean(value: int) -> int:
    """First documentation."""
    if value < 0:
        raise ValueError("first error")
    return value + 1
''',
        encoding="utf-8",
    )
    second.write_text(
        '''# Formatting and comments are not data semantics.
def clean(value: float) -> float:
    """Different documentation."""

    if value < 0:
        raise ValueError("different error")
    return value + 1
''',
        encoding="utf-8",
    )

    assert semantic_definitions_digest(first, ("clean",)) == semantic_definitions_digest(
        second,
        ("clean",),
    )

    second.write_text(
        second.read_text(encoding="utf-8").replace("return value + 1", "return value + 2"),
        encoding="utf-8",
    )
    assert semantic_definitions_digest(first, ("clean",)) != semantic_definitions_digest(
        second,
        ("clean",),
    )

    first.write_text(
        "def clean(value):\n    raise ControlSignal(1)\n",
        encoding="utf-8",
    )
    second.write_text(
        "def clean(value):\n    raise ControlSignal(2)\n",
        encoding="utf-8",
    )
    assert semantic_definitions_digest(first, ("clean",)) != semantic_definitions_digest(
        second,
        ("clean",),
    )


def test_semantic_digest_requires_local_dependencies_to_be_classified(
    tmp_path: Path,
) -> None:
    source = tmp_path / "pipeline.py"
    source.write_text(
        "def helper(value):\n"
        "    return value + 1\n\n"
        "def clean(value):\n"
        "    return helper(value)\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unclassified local dependencies: helper"):
        semantic_definitions_digest(source, ("clean",))
    semantic_definitions_digest(source, ("helper", "clean"))
    semantic_definitions_digest(
        source,
        ("clean",),
        ignored_local_dependencies=frozenset({"helper"}),
    )


def test_runtime_and_lifecycle_code_are_outside_dataset_identity(tmp_path: Path) -> None:
    copied = _copy_package(tmp_path)
    before = code_content_identity(package_root=copied)

    http_path = copied / "data" / "providers" / "http.py"
    http_path.write_text(
        http_path.read_text(encoding="utf-8")
        + "\n\ndef runtime_only_diagnostic() -> str:\n    return 'changed'\n",
        encoding="utf-8",
    )
    eodhd_path = copied / "data" / "providers" / "eodhd.py"
    eodhd_source = eodhd_path.read_text(encoding="utf-8")
    eodhd_runtime_changed = eodhd_source.replace(
        "        self.client = client\n        self._api_token = api_token\n",
        "        self.client = client\n        self._api_token = api_token\n"
        "        self.runtime_diagnostic = 'changed'\n",
        1,
    ).replace(
        "    def _asset_type(value: Any) -> str | None:\n",
        "    def runtime_only_diagnostic(self) -> str:\n"
        "        return 'changed'\n\n"
        "    @staticmethod\n"
        "    def _asset_type(value: Any) -> str | None:\n",
        1,
    )
    assert eodhd_runtime_changed != eodhd_source
    eodhd_path.write_text(eodhd_runtime_changed, encoding="utf-8")
    runtime_resources_path = copied / "runtime_resources.py"
    runtime_resources_path.write_text(
        runtime_resources_path.read_text(encoding="utf-8")
        + "\n\ndef runtime_only_resource_log() -> str:\n    return 'changed'\n",
        encoding="utf-8",
    )
    cli_wrapper_path = copied / "cli" / "prepare_data_runtime.py"
    cli_wrapper_path.write_text(
        cli_wrapper_path.read_text(encoding="utf-8")
        + "\n\ndef runtime_only_cli_log() -> str:\n    return 'changed'\n",
        encoding="utf-8",
    )
    training_path = copied / "training.py"
    training_path.write_text(
        training_path.read_text(encoding="utf-8")
        + "\n\ndef runtime_only_training_log() -> str:\n    return 'changed'\n",
        encoding="utf-8",
    )
    integrity_path = copied / "bar_store_integrity.py"
    integrity_path.write_text(
        integrity_path.read_text(encoding="utf-8")
        + "\n\ndef runtime_only_integrity_log() -> str:\n    return 'changed'\n",
        encoding="utf-8",
    )
    bar_store_path = copied / "data" / "bar_store.py"
    original = bar_store_path.read_text(encoding="utf-8")
    runtime_changed = (
        original.replace(
            "SAFE_MEMORY_FRACTION = 0.60",
            "SAFE_MEMORY_FRACTION = 0.55",
            1,
        )
        .replace(
            'RAW_SCAN_ALGORITHM = "partitioned-bucket-row-groups-v3"',
            'RAW_SCAN_ALGORITHM = "runtime-layout-change"',
            1,
        )
        .replace(
            'return int.from_bytes(digest, byteorder="big", signed=False) % bucket_count',
            'return (int.from_bytes(digest, byteorder="big", signed=False) + 1) % bucket_count',
            1,
        )
    )
    assert runtime_changed != original
    bar_store_path.write_text(runtime_changed, encoding="utf-8")

    assert code_content_identity(package_root=copied) == before
    paths = semantic_source_paths()
    assert "src/stock_forecasting/data/providers/http.py" not in paths
    assert "src/stock_forecasting/bar_store_integrity.py" not in paths
    assert "src/stock_forecasting/dataset_profiles.py" in paths
    assert not any(path.startswith("scripts/") for path in paths)
    assert not any("cli/" in path for path in paths)


def test_provenance_label_changes_do_not_change_data_identity(tmp_path: Path) -> None:
    copied = _copy_package(tmp_path)
    before = code_content_identity(package_root=copied)
    schema_path = copied / "data" / "schema.py"
    source = schema_path.read_text(encoding="utf-8")
    changed = source.replace(
        '    "common_stock_adr_tdr_and_allowlisted_unleveraged_equity_etf_v1"\n',
        '    "renamed_provenance_label_without_a_data_logic_change"\n',
        1,
    )
    assert changed != source
    schema_path.write_text(changed, encoding="utf-8")

    splits_path = copied / "data" / "splits.py"
    split_source = splits_path.read_text(encoding="utf-8")
    split_changed = split_source.replace(
        'SPLIT_POLICY = "global_chronological_cutoff_with_purge_embargo_and_label_end_guard_v1"',
        'SPLIT_POLICY = "renamed_split_audit_label"',
        1,
    )
    assert split_changed != split_source
    splits_path.write_text(split_changed, encoding="utf-8")

    assert code_content_identity(package_root=copied) == before


def test_diagnostics_do_not_change_a_selected_definition_digest(tmp_path: Path) -> None:
    first = tmp_path / "first.py"
    second = tmp_path / "second.py"
    first.write_text(
        "def clean(value):\n    return value + 1\n",
        encoding="utf-8",
    )
    second.write_text(
        "def clean(value):\n"
        "    print('diagnostic')\n"
        "    logger.info('another diagnostic')\n"
        "    return value + 1\n",
        encoding="utf-8",
    )

    assert semantic_definitions_digest(first, ("clean",)) == semantic_definitions_digest(
        second,
        ("clean",),
    )


def test_audit_metadata_does_not_change_provider_or_bar_store_identity(
    tmp_path: Path,
) -> None:
    copied = _copy_package(tmp_path)
    provider_before = provider_materialization_digest("eodhd", package_root=copied)
    bar_store_before = bar_store_materialization_digest(package_root=copied)

    eodhd_path = copied / "data" / "providers" / "eodhd.py"
    source = eodhd_path.read_text(encoding="utf-8")
    changed = source.replace(
        '"coverage": "provider_full_historical_splits_response",',
        '"coverage": "renamed_audit_coverage",',
        1,
    ).replace(
        'f"eodhd_adjusted_close+{split_adjustment_source}+reconstructed_raw_volume"',
        'f"renamed_audit_source+{split_adjustment_source}"',
        1,
    )
    assert changed != source
    eodhd_path.write_text(changed, encoding="utf-8")

    benchmark_path = copied / "data" / "benchmarks.py"
    benchmark_source = benchmark_path.read_text(encoding="utf-8")
    benchmark_changed = benchmark_source.replace(
        '"us_domestic_equity", "us_vti"',
        '"renamed_reason", "renamed_policy"',
        1,
    )
    assert benchmark_changed != benchmark_source
    benchmark_path.write_text(benchmark_changed, encoding="utf-8")

    bar_store_path = copied / "data" / "bar_store.py"
    bar_store_source = bar_store_path.read_text(encoding="utf-8")
    bar_store_changed = bar_store_source.replace(
        'resolved.loc[missing, "eligibility_reason"] = "missing_benchmark"',
        'resolved.loc[missing, "eligibility_reason"] = "renamed_audit_reason"',
        1,
    )
    assert bar_store_changed != bar_store_source
    bar_store_path.write_text(bar_store_changed, encoding="utf-8")

    assert provider_materialization_digest("eodhd", package_root=copied) == provider_before
    assert bar_store_materialization_digest(package_root=copied) == bar_store_before

    taiwan_before = {
        provider: provider_materialization_digest(provider, package_root=copied)
        for provider in ("tpex_official", "twse_official")
    }
    taiwan_path = copied / "data" / "providers" / "taiwan.py"
    taiwan_source = taiwan_path.read_text(encoding="utf-8")
    taiwan_changed = taiwan_source.replace(
        'return "preferred_stock_code"',
        'return "renamed_preferred_stock_audit_label"',
        1,
    ).replace(
        'action_adjustment_source = "twse_twt49u"',
        'action_adjustment_source = "renamed_twse_audit_source"',
        1,
    ).replace(
        'action_adjustment_source = "tpex_exdailyq"',
        'action_adjustment_source = "renamed_tpex_audit_source"',
        1,
    )
    assert taiwan_changed != taiwan_source
    taiwan_path.write_text(taiwan_changed, encoding="utf-8")
    for provider, expected in taiwan_before.items():
        assert provider_materialization_digest(provider, package_root=copied) == expected

    identity_before_schema_change = code_content_identity(package_root=copied)
    schema_path = copied / "data" / "schema.py"
    schema_source = schema_path.read_text(encoding="utf-8")
    schema_changed = schema_source.replace(
        "class MarketDataValidationError(ValueError):",
        "class MarketDataValidationError(RuntimeError):",
        1,
    )
    assert schema_changed != schema_source
    schema_path.write_text(schema_changed, encoding="utf-8")
    assert code_content_identity(package_root=copied) == identity_before_schema_change


def test_benchmark_eligibility_changes_bar_store_identity(tmp_path: Path) -> None:
    copied = _copy_package(tmp_path)
    before = bar_store_materialization_digest(package_root=copied)
    benchmark_path = copied / "data" / "benchmarks.py"
    source = benchmark_path.read_text(encoding="utf-8")
    changed = source.replace(
        'return BenchmarkDecision(benchmark, True, "us_domestic_equity", "us_vti")',
        'return BenchmarkDecision(benchmark, False, "us_domestic_equity", "us_vti")',
        1,
    )
    assert changed != source
    benchmark_path.write_text(changed, encoding="utf-8")

    assert bar_store_materialization_digest(package_root=copied) != before


def test_provider_identities_are_isolated_and_bar_store_is_independent(
    tmp_path: Path,
) -> None:
    copied = _copy_package(tmp_path)
    before = code_content_identity(package_root=copied)
    taiwan_path = copied / "data" / "providers" / "taiwan.py"
    source = taiwan_path.read_text(encoding="utf-8")
    taiwan_path.write_text(
        source.replace(
            '"date": date.replace("-", "/"),',
            '"date": date.replace("-", "/") + "?",',
            1,
        ),
        encoding="utf-8",
    )
    after = code_content_identity(package_root=copied)

    assert (
        after["provider_materialization_digests"]["tpex_official"]
        != before["provider_materialization_digests"]["tpex_official"]
    )
    assert (
        after["provider_materialization_digests"]["twse_official"]
        == before["provider_materialization_digests"]["twse_official"]
    )
    assert (
        after["provider_materialization_digests"]["eodhd_us"]
        == before["provider_materialization_digests"]["eodhd_us"]
    )
    assert (
        after["bar_store_materialization_digest"]
        == before["bar_store_materialization_digest"]
    )


def test_shared_date_range_semantics_invalidate_only_provider_materializations(
    tmp_path: Path,
) -> None:
    copied = _copy_package(tmp_path)
    before = code_content_identity(package_root=copied)
    ingestion_path = copied / "data" / "ingestion.py"
    source = ingestion_path.read_text(encoding="utf-8")
    changed = source.replace(
        "    return date.fromisoformat(value)\n",
        "    return date.fromisoformat(value.strip())\n",
        1,
    )
    assert changed != source
    ingestion_path.write_text(changed, encoding="utf-8")
    after = code_content_identity(package_root=copied)

    for provider in ("eodhd_us", "tpex_official", "twse_official"):
        assert (
            after["provider_materialization_digests"][provider]
            != before["provider_materialization_digests"][provider]
        )
    assert (
        after["provider_materialization_digests"]["massive_us"]
        == before["provider_materialization_digests"]["massive_us"]
    )
    assert after["raw_materialization_digest"] == before["raw_materialization_digest"]
    assert (
        after["bar_store_materialization_digest"]
        == before["bar_store_materialization_digest"]
    )


def test_provider_call_wiring_is_a_provider_content_identity_input(
    tmp_path: Path,
) -> None:
    copied = _copy_package(tmp_path)
    before = code_content_identity(package_root=copied)
    ingestion_path = copied / "data" / "ingestion.py"
    source = ingestion_path.read_text(encoding="utf-8")
    changed = source.replace(
        "        end=_inclusive_end(exclusive_end),\n"
        "        dataset_profile=dataset_profile,\n",
        "        end=exclusive_end,\n"
        "        dataset_profile=dataset_profile,\n",
        1,
    )
    assert changed != source
    ingestion_path.write_text(changed, encoding="utf-8")
    after = code_content_identity(package_root=copied)

    assert (
        after["provider_materialization_digests"]["eodhd_us"]
        != before["provider_materialization_digests"]["eodhd_us"]
    )
    for provider in ("massive_us", "tpex_official", "twse_official"):
        assert (
            after["provider_materialization_digests"][provider]
            == before["provider_materialization_digests"][provider]
        )
    assert after["raw_materialization_digest"] == before[
        "raw_materialization_digest"
    ]
    assert after["bar_store_materialization_digest"] == before[
        "bar_store_materialization_digest"
    ]


def test_provider_methods_are_reached_only_through_hashed_data_plane_helpers() -> None:
    source = (PACKAGE_ROOT / "data" / "ingestion.py").read_text(encoding="utf-8")

    assert source.count("provider.discover(") == 1
    assert source.count("provider.fetch_materialized_instrument(") == 1
    assert source.count("provider.fetch_actions(") == 1
    assert source.count("provider.fetch_benchmark_month(") == 1
    assert source.count("provider.fetch_adjusted_date(") == 1
    for helper in (
        "_discover_eodhd_materialization_universe",
        "_fetch_eodhd_materialized_instrument",
        "_taiwan_materialization_months",
        "_fetch_taiwan_actions",
        "_fetch_taiwan_benchmark_month",
        "_taiwan_materialization_dates",
        "_fetch_taiwan_adjusted_date",
    ):
        assert helper in source


def test_market_specific_allowlists_do_not_invalidate_other_provider(
    tmp_path: Path,
) -> None:
    copied = _copy_package(tmp_path)
    before = code_content_identity(package_root=copied)
    benchmark_path = copied / "data" / "benchmarks.py"
    source = benchmark_path.read_text(encoding="utf-8")
    changed = source.replace(
        '        "XLY.US",\n',
        '        "XLY.US",\n        "FIXTURE.US",\n',
        1,
    )
    assert changed != source
    benchmark_path.write_text(changed, encoding="utf-8")
    after = code_content_identity(package_root=copied)

    assert (
        after["provider_materialization_digests"]["eodhd_us"]
        != before["provider_materialization_digests"]["eodhd_us"]
    )
    assert (
        after["provider_materialization_digests"]["twse_official"]
        == before["provider_materialization_digests"]["twse_official"]
    )
    assert (
        after["provider_materialization_digests"]["tpex_official"]
        == before["provider_materialization_digests"]["tpex_official"]
    )


def test_selected_dataset_digest_excludes_unselected_provider_changes(
    tmp_path: Path,
) -> None:
    copied = _copy_package(tmp_path)
    before = dataset_content_identity(["eodhd_us"], package_root=copied)
    taiwan_path = copied / "data" / "providers" / "taiwan.py"
    taiwan_path.write_text(
        taiwan_path.read_text(encoding="utf-8").replace(
            '"date": date.replace("-", "/"),',
            '"date": date.replace("-", "/") + "?",',
            1,
        ),
        encoding="utf-8",
    )
    after = dataset_content_identity(["eodhd_us"], package_root=copied)

    assert after == before
    assert content_identity_digest(after) == content_identity_digest(before)


def test_cleaning_logic_changes_bar_store_identity(tmp_path: Path) -> None:
    copied = _copy_package(tmp_path)
    before = bar_store_materialization_digest(package_root=copied)
    provider_before = provider_materialization_digest("eodhd", package_root=copied)
    bar_store_path = copied / "data" / "bar_store.py"
    source = bar_store_path.read_text(encoding="utf-8")
    changed = source.replace(
        "(np.abs(log_returns) > max_abs_log_return)",
        "(np.abs(log_returns) >= max_abs_log_return)",
        1,
    )
    assert changed != source
    bar_store_path.write_text(changed, encoding="utf-8")

    assert bar_store_materialization_digest(package_root=copied) != before
    assert (
        provider_materialization_digest("eodhd", package_root=copied)
        == provider_before
    )


def test_symbol_grouping_order_does_not_change_bar_store_identity(
    tmp_path: Path,
) -> None:
    copied = _copy_package(tmp_path)
    before = bar_store_materialization_digest(package_root=copied)
    bar_store_path = copied / "data" / "bar_store.py"
    source = bar_store_path.read_text(encoding="utf-8")
    changed = source.replace(
        '    for symbol, rows in frame.groupby("symbol", sort=True):\n'
        '        yield str(symbol), _ordered_symbol_rows(rows)\n',
        '    for symbol, rows in frame.groupby("symbol", sort=False):\n'
        '        yield str(symbol), _ordered_symbol_rows(rows)\n',
        1,
    )
    assert changed != source
    bar_store_path.write_text(changed, encoding="utf-8")

    assert bar_store_materialization_digest(package_root=copied) == before


def test_symbol_index_file_order_does_not_change_bar_store_identity(
    tmp_path: Path,
) -> None:
    copied = _copy_package(tmp_path)
    before = bar_store_materialization_digest(package_root=copied)
    bar_store_path = copied / "data" / "bar_store.py"
    source = bar_store_path.read_text(encoding="utf-8")
    changed = source.replace(
        '    return index.sort_values("symbol", kind="stable").reset_index(drop=True)\n',
        '    return index.sort_values(\n'
        '        "symbol", ascending=False, kind="stable"\n'
        '    ).reset_index(drop=True)\n',
        1,
    )
    assert changed != source
    bar_store_path.write_text(changed, encoding="utf-8")

    assert bar_store_materialization_digest(package_root=copied) == before


def test_raw_composition_logic_invalidates_only_raw_and_downstream_identity(
    tmp_path: Path,
) -> None:
    copied = _copy_package(tmp_path)
    before = code_content_identity(package_root=copied)
    provider_before = provider_materialization_digest(
        "eodhd",
        package_root=copied,
    )
    ingestion_path = copied / "data" / "ingestion.py"
    source = ingestion_path.read_text(encoding="utf-8")
    changed = source.replace(
        "    return normalize_ohlcv_frame(frame)\n",
        "    return normalize_ohlcv_frame(frame).reset_index(drop=True)\n",
        1,
    )
    assert changed != source
    ingestion_path.write_text(changed, encoding="utf-8")
    after = code_content_identity(package_root=copied)

    assert raw_dataset_materialization_digest(package_root=copied) != before[
        "raw_materialization_digest"
    ]
    assert after["raw_materialization_digest"] != before[
        "raw_materialization_digest"
    ]
    assert (
        provider_materialization_digest("eodhd", package_root=copied)
        == provider_before
    )
    assert (
        after["bar_store_materialization_digest"]
        == before["bar_store_materialization_digest"]
    )


def test_raw_policy_description_is_not_a_data_identity_input(tmp_path: Path) -> None:
    copied = _copy_package(tmp_path)
    before = raw_dataset_materialization_digest(package_root=copied)
    ingestion_path = copied / "data" / "ingestion.py"
    source = ingestion_path.read_text(encoding="utf-8")
    changed = source.replace(
        'RAW_DATASET_ASSEMBLY_POLICY = "complete_provider_checkpoint_concat_v1"',
        'RAW_DATASET_ASSEMBLY_POLICY = "renamed_audit_description"',
        1,
    )
    assert changed != source
    ingestion_path.write_text(changed, encoding="utf-8")

    assert raw_dataset_materialization_digest(package_root=copied) == before


def test_raw_provider_concatenation_order_is_not_a_data_identity_input(
    tmp_path: Path,
) -> None:
    copied = _copy_package(tmp_path)
    before = raw_dataset_materialization_digest(package_root=copied)
    ingestion_path = copied / "data" / "ingestion.py"
    source = ingestion_path.read_text(encoding="utf-8")
    changed = source.replace(
        "    for provider_name in sorted(outcomes):\n",
        "    for provider_name in sorted(outcomes, reverse=True):\n",
        1,
    )
    assert changed != source
    ingestion_path.write_text(changed, encoding="utf-8")

    assert raw_dataset_materialization_digest(package_root=copied) == before


def test_unrelated_profile_membership_does_not_change_data_code_identity(
    tmp_path: Path,
) -> None:
    copied = _copy_package(tmp_path)
    before = code_content_identity(package_root=copied)
    profiles_path = copied / "dataset_profiles.py"
    source = profiles_path.read_text(encoding="utf-8")
    changed = source.replace(
        '"tw_only": ("tpex_official", "twse_official"),',
        '"tw_only": ("twse_official", "tpex_official"),',
        1,
    )
    assert changed != source
    profiles_path.write_text(changed, encoding="utf-8")

    assert code_content_identity(package_root=copied) == before


def test_provider_route_changes_only_that_provider_identity(tmp_path: Path) -> None:
    copied = _copy_package(tmp_path)
    before = code_content_identity(package_root=copied)
    profiles_path = copied / "dataset_profiles.py"
    source = profiles_path.read_text(encoding="utf-8")
    changed = source.replace(
        '"eodhd_us": "eodhd",',
        '"eodhd_us": "fixture_eodhd",',
        1,
    )
    assert changed != source
    profiles_path.write_text(changed, encoding="utf-8")
    after = code_content_identity(package_root=copied)

    assert (
        after["provider_materialization_digests"]["eodhd_us"]
        != before["provider_materialization_digests"]["eodhd_us"]
    )
    for dataset in ("massive_us", "tpex_official", "twse_official"):
        assert (
            after["provider_materialization_digests"][dataset]
            == before["provider_materialization_digests"][dataset]
        )
    assert after["raw_materialization_digest"] == before[
        "raw_materialization_digest"
    ]
    assert after["bar_store_materialization_digest"] == before[
        "bar_store_materialization_digest"
    ]
