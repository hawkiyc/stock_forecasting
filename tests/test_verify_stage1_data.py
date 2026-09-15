"""Exercise the CPU preparation-to-readiness handoff with fixed-date artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from stock_forecasting.cli.prepare_data import main as prepare_main
from stock_forecasting.cli.verify_stage1_data import build_readiness_manifest
from stock_forecasting.config import ExperimentConfig
from stock_forecasting.data.content_identity import (
    code_content_identity,
    content_identity_digest,
    semantic_source_paths,
)
from stock_forecasting.data.manifest import artifact_metadata, atomic_write_json, sha256_file
from stock_forecasting.data.schema import TRAINING_SECURITY_SCOPE
from stock_forecasting.dataset_identity import FIXED_EVALUATION_SPLIT, MIN_FIXED_EVALUATION_DATES

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(params=["stage1", "stage2"])
def readiness_inputs(
    request: pytest.FixtureRequest,
    tmp_path: Path,
    market_frame: pd.DataFrame,
) -> dict[str, Any]:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        (ROOT / "configs" / f"{request.param}_kronos_base_lora.yaml")
        .read_text(encoding="utf-8")
        .replace("input_length: 128", "input_length: 64"),
        encoding="utf-8",
    )
    config = ExperimentConfig.from_yaml(config_path)
    days = sorted(market_frame["timestamp"].unique())
    replacement = pd.bdate_range("2025-01-01", periods=len(days), tz="UTC")
    frame = market_frame.copy()
    frame["timestamp"] = frame["timestamp"].map(dict(zip(days, replacement, strict=True)))
    raw = tmp_path / "raw" / "market.parquet"
    raw.parent.mkdir()
    frame.to_parquet(raw, compression="zstd", index=False)
    request_log = tmp_path / "api-request-log.jsonl"
    request_log.write_text('{"provider":"fixture"}\n', encoding="utf-8")

    code_identity = code_content_identity()
    providers = ["eodhd", "tpex_official", "twse_official"]
    symbols = sorted(frame["symbol"].unique().tolist())
    download_path = tmp_path / "download-manifest.json"
    atomic_write_json(
        download_path,
        {
            "schema_version": "2.0",
            "kind": "ohlcv-dataset",
            "state": "downloaded",
            "training_security_scope": TRAINING_SECURITY_SCOPE,
            "dataset_profile": config.data.dataset_profile,
            "selected_datasets": config.data.selected_datasets,
            "providers": providers,
            "markets": sorted(frame["market"].unique().tolist()),
            "date_range": {"start_inclusive": "2025-01-01", "end_exclusive": "2026-06-01"},
            "symbols": {"count": len(symbols), "values": symbols},
            "quality": {},
            "api_policy": {
                "provider_materialization_checkpoints": {provider: {} for provider in providers},
            },
            "data_content_identity": {
                "schema_version": code_identity["schema_version"],
                "selected_datasets": config.data.selected_datasets,
                "provider_materialization_digests": {
                    name: code_identity["provider_materialization_digests"][name]
                    for name in config.data.selected_datasets
                },
                "raw_materialization_digest": code_identity["raw_materialization_digest"],
            },
            "artifacts": {
                name: artifact_metadata(path, root=tmp_path, row_count=rows)
                for name, path, rows in (
                    ("raw", raw, len(frame)), ("request_log", request_log, 1),
                )
            },
        },
    )
    dataset_path = tmp_path / ".dataset-manifest-launch-fixture.staging.json"
    assert prepare_main(
        [
            "--input", str(raw),
            "--output", str(tmp_path / "prepared" / "bar-store"),
            "--dataset-manifest", str(dataset_path),
            "--fixed-evaluation", "--window-size", "64",
            "--bucket-count", "2", "--batch-rows", "200", "--workers", "2",
        ]
    ) == 0

    code_path = tmp_path / "code.json"
    pipeline_paths = list(semantic_source_paths())
    atomic_write_json(
        code_path,
        {
            "kind": "code", "state": "ready",
            "data_content_identity": code_identity,
            "data_pipeline_digest": content_identity_digest(code_identity),
            "data_pipeline_paths": pipeline_paths,
            "files": [
                {"path": relative, "sha256": sha256_file(ROOT / relative)}
                for relative in pipeline_paths
            ],
        },
    )
    model_path = tmp_path / "hf-models.json"
    revisions = {
        config.model.time_series_model_id: config.model.time_series_model_revision,
        config.model.time_series_tokenizer_id: config.model.time_series_tokenizer_revision,
    }
    repositories = {}
    for index, repository in enumerate(revisions):
        snapshot = tmp_path / "cache" / str(index)
        snapshot.mkdir(parents=True)
        repositories[repository] = str(snapshot)
    # Supply offline cache metadata only; this test never downloads or loads a model.
    atomic_write_json(
        model_path,
        {
            "local_files_only_verified": True,
            "kronos_source_revision": config.model.kronos_source_revision,
            "repositories": repositories,
            "repository_revisions": revisions,
            "time_series_smoke_test": {
                "passed": True, "local_files_only": True, "backend": "kronos",
                "kronos_source_revision": config.model.kronos_source_revision,
                "time_series_model_revision": config.model.time_series_model_revision,
                "time_series_tokenizer_revision": config.model.time_series_tokenizer_revision,
            },
        },
    )
    return {
        "dataset_manifest_path": dataset_path,
        "code_manifest_path": code_path,
        "model_manifest_path": model_path,
        "config_path": config_path,
        "volume_root": tmp_path,
        "launch_id": "launch-fixture",
    }


def test_fixed_preparation_passes_readiness_without_changing_data(
    readiness_inputs: dict[str, Any],
) -> None:
    paths = [path for path in readiness_inputs["volume_root"].rglob("*") if path.is_file()]
    before = {path: sha256_file(path) for path in paths}
    marker = build_readiness_manifest(**readiness_inputs)

    assert marker["state"] == "ready"
    assert marker["storage_preparation_spec"]["fixed_split"] == FIXED_EVALUATION_SPLIT
    assert marker["storage_preparation_spec"]["window_materialized"] is False
    assert marker["storage_preparation_spec"]["labels_materialized"] is False
    assert {path: sha256_file(path) for path in paths} == before


def test_readiness_still_rejects_different_config_dates(
    readiness_inputs: dict[str, Any],
) -> None:
    config_path = readiness_inputs["config_path"]
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace("2025-06-01", "2025-07-01"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Dataset fixed dates differ from config"):
        build_readiness_manifest(**readiness_inputs)


@pytest.mark.parametrize("split", ["validation", "test"])
def test_readiness_rejects_insufficient_effective_dates(
    readiness_inputs: dict[str, Any], split: str,
) -> None:
    dataset_path = readiness_inputs["dataset_manifest_path"]
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    for market in payload["split_audit"]["dates_by_market"].values():
        coverage = market[split]
        coverage["cutoff_dates"] = coverage["cutoff_dates"][:MIN_FIXED_EVALUATION_DATES - 1]
        coverage["unique_cutoff_count"] = len(coverage["cutoff_dates"])
    atomic_write_json(dataset_path, payload)

    with pytest.raises(
        ValueError, match=rf"{split} has only {MIN_FIXED_EVALUATION_DATES - 1} effective dates",
    ):
        build_readiness_manifest(**readiness_inputs)
