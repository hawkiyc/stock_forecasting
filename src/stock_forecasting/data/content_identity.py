"""Content-only identities for durable market-data artifacts.

This module deliberately excludes acquisition pacing, process counts, memory
planning, logging, lifecycle scripts, CLI wrappers, and error-message text.
Only selected definitions that can change numerical rows or lazy cutoffs are
part of a durable dataset identity.
"""

from __future__ import annotations

import ast
import hashlib
import json
import runpy
from pathlib import Path
from typing import Any

CONTENT_IDENTITY_SCHEMA_VERSION = "data-content-identity-v2"

_SHARED_PROVIDER_COMPONENTS = {
    "data/schema.py": (
        "REQUIRED_COLUMNS",
        "OPTIONAL_COLUMNS",
        "PRICE_COLUMNS",
        "SUPPORTED_ASSET_TYPES",
        "_ALIASES",
        "_canonical_column_name",
        "normalize_ohlcv_frame",
        "_validate_ohlcv_frame",
    ),
    "data/adjustments.py": (
        "PRICE_FIELDS",
        "ADJUSTED_CLOSE_FIELD",
        "ADJUSTED_VOLUME_FIELD",
        "ensure_adjustment_columns",
        "normalize_action_frame",
        "apply_cumulative_adjustments",
    ),
}

_RAW_DATASET_COMPONENTS = {
    "data/ingestion.py": (
        "_canonical_raw_batch",
    ),
    "data/schema.py": _SHARED_PROVIDER_COMPONENTS["data/schema.py"],
}

_PROVIDER_COMPONENTS = {
    "eodhd_us": {
        "data/schema.py": (
            "TRAINING_TARGET_ASSET_TYPES",
        ),
        "data/providers/eodhd.py": (
            "_EODHD_DAILY_COLUMNS",
            "_clean_daily_rows",
            "EODHDProvider.name",
            "EODHDProvider.base_url",
            "EODHDProvider._asset_type",
            "EODHDProvider._discover_one",
            "EODHDProvider.discover",
            "EODHDProvider.fetch_instrument",
            "EODHDProvider.fetch_materialized_instrument",
            "EODHDProvider.fetch_historical_split_events",
        ),
        "data/ingestion.py": (
            "_parse_date",
            "_inclusive_end",
            "_explicit_instruments",
            "_validate_explicit_instruments",
            "_limit_instruments",
            "_ensure_us_benchmark",
            "_select_eodhd_materialization_universe",
            "_discover_eodhd_materialization_universe",
            "_fetch_eodhd_materialized_instrument",
        ),
        "data/benchmarks.py": (
            "US_BENCHMARK",
            "US_DOMESTIC_EQUITY_ETFS",
            "is_allowlisted_us_equity_etf",
        ),
    },
    "massive_us": {
        "data/providers/massive.py": (
            "MassiveProvider.name",
            "MassiveProvider.reference_tickers_endpoint",
            "MassiveProvider.daily_aggregates_template",
            "MassiveProvider.discover",
            "MassiveProvider.fetch_instrument",
        ),
    },
    "twse_official": {
        "data/providers/taiwan.py": (
            "_ETF_CODE",
            "_STOCK_CODE",
            "_DEPOSITARY_RECEIPT_CODE",
            "_clean_field",
            "_number",
            "_gregorian_date",
            "_field_index",
            "_first_table",
            "_is_no_data_payload",
            "_index_frame",
            "_tables",
            "_find_quote_table",
            "_asset_type",
            "_parse_quotes",
            "_TaiwanMarketProvider.fetch_date",
            "_TaiwanMarketProvider.fetch_adjusted_date",
            "TWSEProvider.name",
            "TWSEProvider.market",
            "TWSEProvider.suffix",
            "TWSEProvider.endpoint",
            "TWSEProvider.aliases",
            "TWSEProvider._params",
            "TWSEProvider.fetch_actions",
            "TWSEProvider.fetch_benchmark_month",
        ),
        "data/ingestion.py": (
            "_parse_date",
            "_inclusive_end",
            "_months",
            "_benchmark_trading_dates",
            "_taiwan_materialization_months",
            "_fetch_taiwan_actions",
            "_fetch_taiwan_benchmark_month",
            "_taiwan_materialization_dates",
            "_fetch_taiwan_adjusted_date",
        ),
        "data/benchmarks.py": (
            "TWSE_DOMESTIC_EQUITY_ETFS",
            "TPEX_EXPOSURE_ETFS",
            "is_allowlisted_taiwan_equity_etf",
        ),
    },
    "tpex_official": {
        "data/providers/taiwan.py": (
            "_ETF_CODE",
            "_STOCK_CODE",
            "_DEPOSITARY_RECEIPT_CODE",
            "_clean_field",
            "_number",
            "_gregorian_date",
            "_field_index",
            "_first_table",
            "_is_no_data_payload",
            "_index_frame",
            "_tables",
            "_find_quote_table",
            "_asset_type",
            "_parse_quotes",
            "_TaiwanMarketProvider.fetch_date",
            "_TaiwanMarketProvider.fetch_adjusted_date",
            "TPExProvider.name",
            "TPExProvider.market",
            "TPExProvider.suffix",
            "TPExProvider.endpoint",
            "TPExProvider.aliases",
            "TPExProvider._params",
            "TPExProvider.fetch_actions",
            "TPExProvider.fetch_benchmark_month",
        ),
        "data/ingestion.py": (
            "_parse_date",
            "_inclusive_end",
            "_months",
            "_benchmark_trading_dates",
            "_taiwan_materialization_months",
            "_fetch_taiwan_actions",
            "_fetch_taiwan_benchmark_month",
            "_taiwan_materialization_dates",
            "_fetch_taiwan_adjusted_date",
        ),
        "data/benchmarks.py": (
            "TWSE_DOMESTIC_EQUITY_ETFS",
            "TPEX_EXPOSURE_ETFS",
            "is_allowlisted_taiwan_equity_etf",
        ),
    },
}

_BAR_STORE_COMPONENTS = {
    "dataset_identity.py": (
        "FIXED_SPLIT_POLICY", "FIXED_EVALUATION_SPLIT", "validated_fixed_split",
    ),
    "data/schema.py": (
        "REQUIRED_COLUMNS",
        "OPTIONAL_COLUMNS",
        "PRICE_COLUMNS",
        "SUPPORTED_ASSET_TYPES",
        "TRAINING_TARGET_ASSET_TYPES",
        "_ALIASES",
        "_canonical_column_name",
        "normalize_ohlcv_frame",
        "_validate_ohlcv_frame",
    ),
    "data/adjustments.py": (
        "PRICE_FIELDS",
        "ADJUSTED_CLOSE_FIELD",
        "ADJUSTED_VOLUME_FIELD",
        "ensure_adjustment_columns",
    ),
    "data/benchmarks.py": (
        "US_BENCHMARK",
        "TWSE_BENCHMARK",
        "TPEX_BENCHMARK",
        "US_DOMESTIC_EQUITY_ETFS",
        "TWSE_DOMESTIC_EQUITY_ETFS",
        "TPEX_EXPOSURE_ETFS",
        "_TWSE_COMMON_OR_TDR_SYMBOL",
        "_TPEX_COMMON_STOCK_SYMBOL",
        "is_allowlisted_us_equity_etf",
        "is_allowlisted_taiwan_equity_etf",
        "is_allowlisted_unleveraged_equity_etf",
        "is_training_target_security",
        "BenchmarkDecision",
        "_market",
        "resolve_benchmark",
    ),
    "data/bar_store.py": (
        "DEFAULT_MAX_HORIZON",
        "_metadata_value",
        "_ordered_symbol_rows",
        "_true_ranges",
        "_enrich_compacted_frame",
        "_symbol_index_metadata",
        "_exclude_symbols_with_missing_benchmarks",
        "_candidate_mask",
        "_candidate_range_records",
        "_eligible_bucket_rows",
        "_candidate_bucket_records",
        "_global_calendars",
        "_ordered_unique_timestamps",
        "_split_boundaries",
        "_observed_timestamp_array",
        "_split_codes",
        "_split_range_records",
        "_split_bucket_records",
        "_ordered_split_ranges",
    ),
}

# Every same-module dependency reached from a selected semantic definition must
# either be selected or be explicitly classified here. This keeps future data
# helpers from silently bypassing the identity while preserving known
# execution/audit-only helpers outside it.
_NON_CONTENT_LOCAL_DEPENDENCIES = {
    "data/schema.py": frozenset({"MarketDataValidationError"}),
    "data/providers/taiwan.py": frozenset({"_unsupported_action_type"}),
}

class _SemanticAstNormalizer(ast.NodeTransformer):
    """Remove source details that cannot change generated numerical data."""

    _AUDIT_ONLY_KEYS = frozenset(
        {
            "adjustment_source",
            "benchmark_policy",
            "eligibility_reason",
            "source",
        }
    )

    @staticmethod
    def _without_docstring(body: list[ast.stmt]) -> list[ast.stmt]:
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            return body[1:]
        return body

    @staticmethod
    def _strip_arguments(arguments: ast.arguments) -> None:
        for argument in (
            *arguments.posonlyargs,
            *arguments.args,
            *arguments.kwonlyargs,
        ):
            argument.annotation = None
            argument.type_comment = None
        if arguments.vararg is not None:
            arguments.vararg.annotation = None
            arguments.vararg.type_comment = None
        if arguments.kwarg is not None:
            arguments.kwarg.annotation = None
            arguments.kwarg.type_comment = None

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        node.body = self._without_docstring(node.body)
        self._strip_arguments(node.args)
        node.returns = None
        node.type_comment = None
        if hasattr(node, "type_params"):
            node.type_params = []
        return self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        node.body = self._without_docstring(node.body)
        self._strip_arguments(node.args)
        node.returns = None
        node.type_comment = None
        if hasattr(node, "type_params"):
            node.type_params = []
        return self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        node.body = self._without_docstring(node.body)
        if hasattr(node, "type_params"):
            node.type_params = []
        return self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AST:
        node.annotation = ast.Constant(value=None)
        return self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        node = self.generic_visit(node)
        if any(
            isinstance(value, ast.Constant)
            and value.value in self._AUDIT_ONLY_KEYS
            for target in node.targets
            for value in ast.walk(target)
        ):
            node.value = ast.Constant(value="")
        return node

    def visit_Dict(self, node: ast.Dict) -> ast.AST:
        node = self.generic_visit(node)
        for index, key in enumerate(node.keys):
            if (
                isinstance(key, ast.Constant)
                and key.value in self._AUDIT_ONLY_KEYS
            ):
                node.values[index] = ast.Constant(value="")
        return node

    def visit_Call(self, node: ast.Call) -> ast.AST:
        node = self.generic_visit(node)
        function = node.func
        function_name = (
            function.id
            if isinstance(function, ast.Name)
            else function.attr
            if isinstance(function, ast.Attribute)
            else ""
        )
        if function_name == "ProviderFetch":
            for keyword in node.keywords:
                if keyword.arg == "metadata":
                    keyword.value = ast.Dict(keys=[], values=[])
        elif function_name == "BenchmarkDecision":
            for index in range(2, len(node.args)):
                node.args[index] = ast.Constant(value="")
            for keyword in node.keywords:
                if keyword.arg in {"reason", "policy"}:
                    keyword.value = ast.Constant(value="")
        return node

    def visit_Expr(self, node: ast.Expr) -> ast.AST | None:
        """Remove standalone diagnostics that cannot alter returned data."""

        node = self.generic_visit(node)
        if not isinstance(node.value, ast.Call):
            return node
        function = node.value.func
        if isinstance(function, ast.Name) and function.id == "print":
            return None
        if not isinstance(function, ast.Attribute) or function.attr not in {
            "critical",
            "debug",
            "error",
            "exception",
            "info",
            "log",
            "warning",
        }:
            return node
        receiver = function.value
        if isinstance(receiver, ast.Name) and receiver.id.lower().strip("_") in {
            "log",
            "logger",
        }:
            return None
        return node

    def visit_Raise(self, node: ast.Raise) -> ast.AST:
        node = self.generic_visit(node)
        if isinstance(node.exc, ast.Call):
            exception = node.exc.func
            exception_name = (
                exception.id
                if isinstance(exception, ast.Name)
                else exception.attr
                if isinstance(exception, ast.Attribute)
                else ""
            )
            if (
                exception_name.endswith(("Error", "Exception"))
                and node.exc.args
                and isinstance(node.exc.args[0], (ast.Constant, ast.JoinedStr))
                and (
                    isinstance(node.exc.args[0], ast.JoinedStr)
                    or isinstance(node.exc.args[0].value, str)
                )
            ):
                # Conventional first positional arguments are human-facing
                # messages. Structured positional and keyword arguments remain
                # semantic because callers may use them for control flow.
                node.exc.args[0] = ast.Constant(value="")
        return node


def _canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_ast_dump(node: ast.AST) -> str:
    """Serialize an AST without runtime-dependent empty sequence fields."""

    if not isinstance(node, ast.AST):
        raise TypeError(f"expected AST, got {node.__class__.__name__!r}")

    def _format(value: Any) -> str:
        if isinstance(value, ast.AST):
            node_type = type(value)
            fields = []
            for name in value._fields:
                try:
                    field_value = getattr(value, name)
                except AttributeError:
                    continue
                if field_value is None and getattr(node_type, name, ...) is None:
                    continue
                if isinstance(field_value, list) and not field_value:
                    continue
                fields.append(f"{name}={_format(field_value)}")
            return f"{value.__class__.__name__}({', '.join(fields)})"
        if isinstance(value, list):
            return f"[{', '.join(_format(item) for item in value)}]"
        return repr(value)

    return _format(node)


def _definition_name(node: ast.AST) -> str | None:
    if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
        return node.name
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        names = [target.id for target in targets if isinstance(target, ast.Name)]
        return names[0] if len(names) == 1 else None
    return None


def _definition_nodes(syntax: ast.Module) -> dict[str, ast.AST]:
    """Index explicitly addressable definitions without hashing whole classes."""

    definitions: dict[str, ast.AST] = {}
    for node in syntax.body:
        name = _definition_name(node)
        if name is None:
            continue
        definitions[name] = node
        if not isinstance(node, ast.ClassDef):
            continue
        for member in node.body:
            member_name = _definition_name(member)
            if member_name is not None:
                definitions[f"{name}.{member_name}"] = member
    return definitions


def _unselected_local_dependencies(
    definitions: dict[str, ast.AST],
    names: tuple[str, ...],
    normalized_nodes: list[ast.AST],
) -> set[str]:
    """Return same-module dependencies not covered by the semantic selection."""

    selected = set(names)
    top_level = {name for name in definitions if "." not in name}
    dependencies: set[str] = set()
    for selected_name, node in zip(names, normalized_nodes, strict=True):
        dependencies.update(
            child.id
            for child in ast.walk(node)
            if isinstance(child, ast.Name)
            and isinstance(child.ctx, ast.Load)
            and child.id in top_level
        )
        if "." not in selected_name:
            continue
        for child in ast.walk(node):
            if (
                not isinstance(child, ast.Attribute)
                or not isinstance(child.value, ast.Name)
                or child.value.id not in {"self", "cls"}
            ):
                continue
            candidates = {
                name
                for name in definitions
                if "." in name and name.rsplit(".", maxsplit=1)[1] == child.attr
            }
            if candidates and not candidates.intersection(selected):
                dependencies.update(candidates)
    return dependencies.difference(selected)


def semantic_definitions_digest(
    path: str | Path,
    names: tuple[str, ...],
    *,
    ignored_local_dependencies: frozenset[str] = frozenset(),
) -> str:
    """Hash normalized AST for explicitly selected top-level definitions."""

    source = Path(path)
    syntax = ast.parse(
        source.read_text(encoding="utf-8"),
        filename=str(source),
        feature_version=(3, 12),
    )
    by_name = _definition_nodes(syntax)
    missing = sorted(set(names).difference(by_name))
    if missing:
        raise ValueError(
            f"Semantic identity source {source} is missing definitions: "
            + ", ".join(missing)
        )
    normalizer = _SemanticAstNormalizer()
    normalized_nodes = [
        normalizer.visit(ast.fix_missing_locations(by_name[name])) for name in names
    ]
    unresolved = sorted(
        _unselected_local_dependencies(by_name, names, normalized_nodes).difference(
            ignored_local_dependencies
        )
    )
    if unresolved:
        raise ValueError(
            f"Semantic identity source {source} has unclassified local dependencies: "
            + ", ".join(unresolved)
        )
    normalized = [
        {
            "name": name,
            "ast": _canonical_ast_dump(node),
        }
        for name, node in zip(names, normalized_nodes, strict=True)
    ]
    return _canonical_json_sha256(normalized)


def _package_root(package_root: str | Path | None) -> Path:
    return (
        Path(package_root).resolve(strict=True)
        if package_root is not None
        else Path(__file__).resolve().parents[1]
    )


def _merged_components(*groups: dict[str, tuple[str, ...]]) -> dict[str, tuple[str, ...]]:
    merged: dict[str, list[str]] = {}
    for group in groups:
        for relative, names in group.items():
            selected = merged.setdefault(relative, [])
            selected.extend(name for name in names if name not in selected)
    return {relative: tuple(names) for relative, names in sorted(merged.items())}


def _component_digest(
    components: dict[str, tuple[str, ...]],
    *,
    package_root: str | Path | None = None,
) -> str:
    root = _package_root(package_root)
    payload = {
        relative: {
            "definitions": list(names),
            "digest": semantic_definitions_digest(
                root / relative,
                names,
                ignored_local_dependencies=_NON_CONTENT_LOCAL_DEPENDENCIES.get(
                    relative,
                    frozenset(),
                ),
            ),
        }
        for relative, names in sorted(components.items())
    }
    return _canonical_json_sha256(payload)


def _provider_routing(
    *,
    package_root: str | Path | None = None,
) -> tuple[dict[str, str], dict[str, str]]:
    """Load the dependency-free provider route from the selected source tree."""

    root = _package_root(package_root)
    namespace = runpy.run_path(str(root / "dataset_profiles.py"))
    dataset_to_provider = namespace.get("DATASET_RUNTIME_PROVIDER")
    provider_to_dataset = namespace.get("RUNTIME_PROVIDER_DATASET")
    if not isinstance(dataset_to_provider, dict) or not isinstance(
        provider_to_dataset,
        dict,
    ):
        raise ValueError("Dataset provider routing contract is invalid")
    if any(
        not isinstance(dataset, str)
        or not isinstance(provider, str)
        or provider_to_dataset.get(provider) != dataset
        for dataset, provider in dataset_to_provider.items()
    ):
        raise ValueError("Dataset provider routing contract is inconsistent")
    return dataset_to_provider, provider_to_dataset


def provider_materialization_digest(
    provider: str,
    *,
    package_root: str | Path | None = None,
) -> str:
    """Return the content identity for one durable provider Parquet."""

    dataset_to_provider, provider_to_dataset = _provider_routing(
        package_root=package_root
    )
    dataset = provider_to_dataset.get(provider, provider)
    try:
        runtime_provider = dataset_to_provider[dataset]
        components = _merged_components(
            _SHARED_PROVIDER_COMPONENTS,
            _PROVIDER_COMPONENTS[dataset],
        )
    except KeyError as error:
        raise ValueError(f"Unsupported provider content identity: {provider}") from error
    return _canonical_json_sha256(
        {
            "dataset": dataset,
            "runtime_provider": runtime_provider,
            "definitions_digest": _component_digest(
                components,
                package_root=package_root,
            ),
        }
    )


def bar_store_materialization_digest(
    *,
    package_root: str | Path | None = None,
) -> str:
    """Return the content identity for cleaned bars and lazy cutoff ranges."""

    return _component_digest(_BAR_STORE_COMPONENTS, package_root=package_root)


def raw_dataset_materialization_digest(
    *,
    package_root: str | Path | None = None,
) -> str:
    """Return the logical identity for composing provider parts into raw bars."""

    return _component_digest(_RAW_DATASET_COMPONENTS, package_root=package_root)


def semantic_source_paths() -> list[str]:
    """List project-relative source paths used by any content identity."""

    paths = set(_BAR_STORE_COMPONENTS)
    paths.update(_RAW_DATASET_COMPONENTS)
    paths.update(_SHARED_PROVIDER_COMPONENTS)
    for components in _PROVIDER_COMPONENTS.values():
        paths.update(components)
    project_paths = [
        f"src/stock_forecasting/{relative}" for relative in sorted(paths)
    ]
    project_paths.append("src/stock_forecasting/dataset_profiles.py")
    return sorted(project_paths)


def code_content_identity(
    *,
    package_root: str | Path | None = None,
) -> dict[str, Any]:
    """Describe all provider and bar-store semantics in a source release."""

    return {
        "schema_version": CONTENT_IDENTITY_SCHEMA_VERSION,
        "provider_materialization_digests": {
            provider: provider_materialization_digest(
                provider,
                package_root=package_root,
            )
            for provider in sorted(_PROVIDER_COMPONENTS)
        },
        "raw_materialization_digest": raw_dataset_materialization_digest(
            package_root=package_root
        ),
        "bar_store_materialization_digest": bar_store_materialization_digest(
            package_root=package_root
        ),
    }


def dataset_content_identity(
    selected_datasets: list[str] | tuple[str, ...],
    *,
    package_root: str | Path | None = None,
    provider_digests: dict[str, str] | None = None,
    raw_digest: str | None = None,
) -> dict[str, Any]:
    """Select only semantics that can affect the requested dataset profile."""

    selected = sorted(set(selected_datasets))
    unknown = sorted(set(selected).difference(_PROVIDER_COMPONENTS))
    if unknown:
        raise ValueError("Unsupported selected datasets: " + ", ".join(unknown))
    available = (
        code_content_identity(package_root=package_root)["provider_materialization_digests"]
        if provider_digests is None
        else provider_digests
    )
    missing = sorted(set(selected).difference(available))
    if missing:
        raise ValueError("Missing provider content identities: " + ", ".join(missing))
    resolved_raw_digest = (
        raw_dataset_materialization_digest(package_root=package_root)
        if raw_digest is None
        else raw_digest
    )
    if (
        not isinstance(resolved_raw_digest, str)
        or len(resolved_raw_digest) != 64
        or any(
            character not in "0123456789abcdef"
            for character in resolved_raw_digest
        )
    ):
        raise ValueError("Raw materialization identity must be a lowercase SHA-256 digest")
    return {
        "schema_version": CONTENT_IDENTITY_SCHEMA_VERSION,
        "selected_datasets": selected,
        "provider_materialization_digests": {
            provider: available[provider] for provider in selected
        },
        "raw_materialization_digest": resolved_raw_digest,
        "bar_store_materialization_digest": bar_store_materialization_digest(
            package_root=package_root
        ),
    }


def content_identity_digest(identity: dict[str, Any]) -> str:
    """Hash a validated content-identity payload."""

    if identity.get("schema_version") != CONTENT_IDENTITY_SCHEMA_VERSION:
        raise ValueError("Data content identity schema is unsupported")
    return _canonical_json_sha256(identity)
