"""Durable provider materialization checkpoint contracts."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest

from stock_forecasting.data.ingestion import (
    IngestionOptions,
    _checkpointed_provider_runner,
    _provider_checkpoint_identity,
    _ProviderArtifacts,
    _ProviderStats,
    _run_parallel_provider_loops,
)
from stock_forecasting.data.provider_checkpoint import load_provider_checkpoint


def _options(tmp_path: Path, *, end: str = "2026-05-01") -> IngestionOptions:
    return IngestionOptions(
        profile="us_tw_eodhd",
        start="2005-01-01",
        end=end,
        output=tmp_path / "raw" / "market.parquet",
        manifest_root=tmp_path / "launch",
        raw_cache_root=tmp_path / "api-cache",
        provider_checkpoint_root=tmp_path / "provider-checkpoints",
        dataset_request_sha256="a" * 64,
    )


def _local_artifacts(
    tmp_path: Path,
    *,
    provider: str,
    attempt: str,
    cache_hits: int = 1,
) -> _ProviderArtifacts:
    root = tmp_path / "parts" / attempt / provider
    root.mkdir(parents=True)
    parquet = root / "market.parquet"
    request_log = root / "request-log.jsonl"
    parquet.write_bytes(f"{provider}-{attempt}-parquet".encode())
    request_log.write_text(
        json.dumps({"cache_hit": cache_hits == 1, "provider": provider}) + "\n",
        encoding="utf-8",
    )
    return _ProviderArtifacts(
        provider=provider,
        parquet_path=parquet,
        request_log_path=request_log,
        row_count=3,
        request_count=1,
        cache_hits=cache_hits,
    )


def _checkpointed(
    *,
    options: IngestionOptions,
    provider: str,
    runner: Callable[[], _ProviderArtifacts],
    stats: dict[str, _ProviderStats],
) -> Callable[[], _ProviderArtifacts]:
    assert options.provider_checkpoint_root is not None
    return _checkpointed_provider_runner(
        provider=provider,
        runner=runner,
        checkpoint_root=options.provider_checkpoint_root,
        checkpoint_identity=_provider_checkpoint_identity(options, provider=provider),
        stats=stats,
    )


def test_successful_provider_checkpoint_survives_another_provider_failure(
    tmp_path: Path,
) -> None:
    options = _options(tmp_path)
    first_stats = {
        "eodhd": _ProviderStats(),
        "tpex_official": _ProviderStats(),
    }
    calls = {"eodhd": 0, "tpex_official": 0}

    def run_eodhd() -> _ProviderArtifacts:
        calls["eodhd"] += 1
        first_stats["eodhd"].estimated_calls = 42
        first_stats["eodhd"].dropped_source_rows = 2
        return _local_artifacts(
            tmp_path,
            provider="eodhd",
            attempt="first",
        )

    def fail_tpex() -> _ProviderArtifacts:
        calls["tpex_official"] += 1
        raise ValueError("fixture provider contract failure")

    first_outcomes = _run_parallel_provider_loops(
        {
            "eodhd": _checkpointed(
                options=options,
                provider="eodhd",
                runner=run_eodhd,
                stats=first_stats,
            ),
            "tpex_official": _checkpointed(
                options=options,
                provider="tpex_official",
                runner=fail_tpex,
                stats=first_stats,
            ),
        }
    )

    first_eodhd = first_outcomes["eodhd"].result
    assert isinstance(first_eodhd, _ProviderArtifacts)
    assert first_eodhd.checkpoint_identity_sha256 is not None
    assert first_eodhd.checkpoint_reused is False
    assert first_outcomes["tpex_official"].error is not None
    assert calls == {"eodhd": 1, "tpex_official": 1}

    second_stats = {
        "eodhd": _ProviderStats(),
        "tpex_official": _ProviderStats(),
    }

    def rerun_eodhd() -> _ProviderArtifacts:
        pytest.fail("A completed EODHD materialization must be reused")

    def run_tpex() -> _ProviderArtifacts:
        calls["tpex_official"] += 1
        second_stats["tpex_official"].estimated_calls = 7
        return _local_artifacts(
            tmp_path,
            provider="tpex_official",
            attempt="second",
        )

    second_outcomes = _run_parallel_provider_loops(
        {
            "eodhd": _checkpointed(
                options=options,
                provider="eodhd",
                runner=rerun_eodhd,
                stats=second_stats,
            ),
            "tpex_official": _checkpointed(
                options=options,
                provider="tpex_official",
                runner=run_tpex,
                stats=second_stats,
            ),
        }
    )

    reused_eodhd = second_outcomes["eodhd"].result
    completed_tpex = second_outcomes["tpex_official"].result
    assert isinstance(reused_eodhd, _ProviderArtifacts)
    assert isinstance(completed_tpex, _ProviderArtifacts)
    assert reused_eodhd.checkpoint_reused is True
    assert completed_tpex.checkpoint_reused is False
    assert second_stats["eodhd"].estimated_calls == 42
    assert second_stats["eodhd"].dropped_source_rows == 2
    assert calls == {"eodhd": 1, "tpex_official": 2}


def test_provider_checkpoint_identity_ignores_attempt_pacing_but_binds_dataset(
    tmp_path: Path,
) -> None:
    options = _options(tmp_path)
    changed_pacing = replace(
        options,
        max_api_calls=25,
        eodhd_requests_per_second=8.0,
        taiwan_requests_per_second=0.25,
        max_backoff_seconds=300.0,
    )
    changed_end = replace(options, end="2026-06-01", dataset_request_sha256="b" * 64)

    original = _provider_checkpoint_identity(options, provider="eodhd")
    assert _provider_checkpoint_identity(changed_pacing, provider="eodhd") == original
    assert _provider_checkpoint_identity(changed_end, provider="eodhd") != original
    assert _provider_checkpoint_identity(options, provider="twse_official") != original


def test_invalid_provider_checkpoint_is_quarantined_and_rebuilt(tmp_path: Path) -> None:
    options = _options(tmp_path)
    first_stats = {"eodhd": _ProviderStats(estimated_calls=4)}
    first = _checkpointed(
        options=options,
        provider="eodhd",
        runner=lambda: _local_artifacts(
            tmp_path,
            provider="eodhd",
            attempt="before-corruption",
        ),
        stats=first_stats,
    )()
    first.parquet_path.write_bytes(b"tampered")

    replacement_calls = 0
    second_stats = {"eodhd": _ProviderStats(estimated_calls=5)}

    def rebuild() -> _ProviderArtifacts:
        nonlocal replacement_calls
        replacement_calls += 1
        return _local_artifacts(
            tmp_path,
            provider="eodhd",
            attempt="after-corruption",
        )

    rebuilt = _checkpointed(
        options=options,
        provider="eodhd",
        runner=rebuild,
        stats=second_stats,
    )()

    assert replacement_calls == 1
    assert rebuilt.checkpoint_reused is False
    assert options.provider_checkpoint_root is not None
    quarantined = list(
        (options.provider_checkpoint_root / "quarantine" / "eodhd").iterdir()
    )
    assert len(quarantined) == 1
    loaded = load_provider_checkpoint(
        options.provider_checkpoint_root,
        provider="eodhd",
        identity=_provider_checkpoint_identity(options, provider="eodhd"),
    )
    assert loaded is not None
    assert loaded.parquet_path.read_bytes() == b"eodhd-after-corruption-parquet"
