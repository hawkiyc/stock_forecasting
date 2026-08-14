"""Prefetch configured Hugging Face repositories into persistent storage."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from stock_forecasting.config import ExperimentConfig
from stock_forecasting.run_paths import canonical_network_volume_root

SnapshotDownloader = Callable[..., str]
DEFAULT_HF_HUB_ETAG_TIMEOUT = 60
DEFAULT_HF_HUB_DOWNLOAD_TIMEOUT = 300
DEFAULT_PREFETCH_MAX_WORKERS = 4
DEFAULT_PREFETCH_MAX_ATTEMPTS = 3
DEFAULT_PREFETCH_RETRY_BACKOFF_SECONDS = 15.0
_RETRYABLE_DOWNLOAD_ERROR_NAMES = frozenset(
    {
        "CloseError",
        "ConnectError",
        "ConnectTimeout",
        "ConnectionError",
        "LocalProtocolError",
        "NetworkError",
        "PoolTimeout",
        "ReadError",
        "ReadTimeout",
        "RemoteProtocolError",
        "TimeoutError",
        "WriteError",
        "WriteTimeout",
    }
)


def repositories_from_config(config: Any) -> dict[str, str]:
    """Return exact non-mock Kronos repository revisions used by an experiment."""

    if config.model.time_series_backend == "mock":
        return {}
    model_revision = config.model.time_series_model_revision
    tokenizer_revision = config.model.time_series_tokenizer_revision
    assert model_revision is not None
    assert tokenizer_revision is not None
    return {
        config.model.time_series_model_id: model_revision,
        config.model.time_series_tokenizer_id: tokenizer_revision,
    }


def prefetch_repositories(
    repositories: Mapping[str, str],
    *,
    cache_directory: Path,
    verify_only: bool,
    downloader: SnapshotDownloader,
    max_workers: int = DEFAULT_PREFETCH_MAX_WORKERS,
    max_attempts: int = DEFAULT_PREFETCH_MAX_ATTEMPTS,
    retry_backoff_seconds: float = DEFAULT_PREFETCH_RETRY_BACKOFF_SECONDS,
) -> dict[str, str]:
    """Download snapshots and always finish with an offline-only verification."""

    if max_workers < 1:
        raise ValueError("max_workers must be positive")
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    if retry_backoff_seconds < 0:
        raise ValueError("retry_backoff_seconds must be non-negative")

    def download_with_retry(repository: str, revision: str) -> str:
        for attempt in range(1, max_attempts + 1):
            try:
                return downloader(
                    repo_id=repository,
                    revision=revision,
                    cache_dir=str(cache_directory),
                    local_files_only=False,
                    max_workers=max_workers,
                )
            except Exception as error:
                error_name = type(error).__name__
                if error_name not in _RETRYABLE_DOWNLOAD_ERROR_NAMES or attempt == max_attempts:
                    raise
                delay = retry_backoff_seconds * (2 ** (attempt - 1))
                print(
                    f"Transient Hugging Face download error for {repository} "
                    f"({error_name}); retry {attempt + 1}/{max_attempts} "
                    f"after {delay:.1f}s",
                    flush=True,
                )
                time.sleep(delay)
        raise AssertionError("unreachable")

    resolved: dict[str, str] = {}
    cache_directory.mkdir(parents=True, exist_ok=True)
    for repository, revision in repositories.items():
        snapshot_path = (
            downloader(
                repo_id=repository,
                revision=revision,
                cache_dir=str(cache_directory),
                local_files_only=True,
                max_workers=max_workers,
            )
            if verify_only
            else download_with_retry(repository, revision)
        )
        if not verify_only:
            snapshot_path = downloader(
                repo_id=repository,
                revision=revision,
                cache_dir=str(cache_directory),
                local_files_only=True,
                max_workers=max_workers,
            )
        resolved[repository] = str(snapshot_path)
    return resolved


def _configure_huggingface_download_environment() -> None:
    """Set slow-network-safe defaults before importing huggingface_hub."""

    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", str(DEFAULT_HF_HUB_ETAG_TIMEOUT))
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", str(DEFAULT_HF_HUB_DOWNLOAD_TIMEOUT))


def _positive_int_environment(name: str, default: int) -> int:
    value = os.environ.get(name, str(default))
    try:
        parsed = int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if parsed < 1:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


def _non_negative_float_environment(name: str, default: float) -> float:
    value = os.environ.get(name, str(default))
    try:
        parsed = float(value)
    except ValueError as error:
        raise ValueError(f"{name} must be non-negative") from error
    if parsed < 0:
        raise ValueError(f"{name} must be non-negative")
    return parsed


def smoke_test_time_series_backbone(
    config: Any,
    *,
    backbone_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Instantiate the cached Kronos adapter and verify its hidden-state contract on CPU."""

    if config.model.time_series_backend == "mock":
        return {"backend": "mock", "skipped": True}

    import torch

    from stock_forecasting.factory import _prepare_kronos_import
    from stock_forecasting.models import KronosBackbone

    revision = config.model.kronos_source_revision
    assert revision is not None
    _prepare_kronos_import(config.model.kronos_source_root, revision)
    factory = backbone_factory or KronosBackbone.from_pretrained
    backbone = factory(
        model_name_or_path=config.model.time_series_model_id,
        tokenizer_name_or_path=config.model.time_series_tokenizer_id,
        model_revision=config.model.time_series_model_revision,
        tokenizer_revision=config.model.time_series_tokenizer_revision,
        local_files_only=True,
        max_context=config.data.input_length,
    )
    backbone.to(torch.device("cpu")).eval()

    bars = config.data.input_length
    steps = torch.arange(bars, dtype=torch.float32)
    close = 100.0 + 0.02 * steps
    series = torch.stack(
        [
            close - 0.05,
            close + 0.20,
            close - 0.20,
            close,
            1_000_000.0 + 100.0 * steps,
        ],
        dim=-1,
    ).unsqueeze(0)
    timestamps = torch.stack(
        [
            torch.zeros_like(steps),
            torch.full_like(steps, 16.0),
            steps.remainder(5.0),
            steps.remainder(28.0) + 1.0,
            torch.ones_like(steps),
        ],
        dim=-1,
    ).unsqueeze(0)
    attention_mask = torch.ones(1, bars, dtype=torch.bool)
    with torch.no_grad():
        output = backbone(
            series,
            attention_mask=attention_mask,
            timestamps=timestamps,
        )
    hidden = output.last_hidden_state
    expected = (1, bars, int(backbone.hidden_size))
    if tuple(hidden.shape) != expected:
        raise RuntimeError(f"Kronos smoke test returned {tuple(hidden.shape)}, expected {expected}")
    if not torch.isfinite(hidden).all():
        raise RuntimeError("Kronos smoke test returned non-finite hidden states")
    if hidden.requires_grad or any(parameter.requires_grad for parameter in backbone.parameters()):
        raise RuntimeError("Kronos smoke test found an unfrozen backbone")
    return {
        "backend": str(config.model.time_series_backend),
        "hidden_size": int(backbone.hidden_size),
        "input_bars": bars,
        "kronos_source_revision": revision,
        "time_series_model_revision": config.model.time_series_model_revision,
        "time_series_tokenizer_revision": config.model.time_series_tokenizer_revision,
        "local_files_only": True,
        "passed": True,
    }


def _validated_manifest_path(path: Path, volume_root: Path) -> Path:
    resolved = path.expanduser().resolve(strict=False)
    resolved_volume = volume_root.expanduser().resolve(strict=False)
    try:
        resolved.relative_to(resolved_volume)
    except ValueError as exc:
        raise ValueError("MODEL_CACHE_MANIFEST must be on NETWORK_VOLUME_ROOT") from exc
    if resolved == Path("/workspace") or Path("/workspace") in resolved.parents:
        raise ValueError("MODEL_CACHE_MANIFEST must never use /workspace")
    return resolved


def _validated_cache_path(path: Path, volume_root: Path) -> Path:
    resolved = path.expanduser().resolve(strict=False)
    resolved_volume = volume_root.expanduser().resolve(strict=False)
    try:
        resolved.relative_to(resolved_volume)
    except ValueError as exc:
        raise ValueError("HF_HOME must be on NETWORK_VOLUME_ROOT") from exc
    if resolved == Path("/workspace") or Path("/workspace") in resolved.parents:
        raise ValueError("HF_HOME must never use /workspace")
    return resolved


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--smoke-time-series-backbone", action="store_true")
    parser.add_argument("--max-workers", type=int, default=DEFAULT_PREFETCH_MAX_WORKERS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    config = ExperimentConfig.from_yaml(arguments.config)
    _configure_huggingface_download_environment()
    max_attempts = _positive_int_environment(
        "HF_PREFETCH_MAX_ATTEMPTS", DEFAULT_PREFETCH_MAX_ATTEMPTS
    )
    retry_backoff_seconds = _non_negative_float_environment(
        "HF_PREFETCH_RETRY_BACKOFF_SECONDS", DEFAULT_PREFETCH_RETRY_BACKOFF_SECONDS
    )
    volume_root = canonical_network_volume_root()
    if volume_root == Path("/workspace") or Path("/workspace") in volume_root.parents:
        raise ValueError("NETWORK_VOLUME_ROOT must never use /workspace")
    hf_home = _validated_cache_path(
        Path(os.environ.get("HF_HOME", str(volume_root / "cache/huggingface"))),
        volume_root,
    )
    cache_directory = hf_home / "hub"
    manifest_path = _validated_manifest_path(
        Path(
            os.environ.get(
                "MODEL_CACHE_MANIFEST",
                str(volume_root / "cache/hf-models.json"),
            )
        ),
        volume_root,
    )

    from huggingface_hub import snapshot_download

    repositories = repositories_from_config(config)
    resolved = prefetch_repositories(
        repositories,
        cache_directory=cache_directory,
        verify_only=arguments.verify_only,
        downloader=snapshot_download,
        max_workers=arguments.max_workers,
        max_attempts=max_attempts,
        retry_backoff_seconds=retry_backoff_seconds,
    )
    smoke_result = (
        smoke_test_time_series_backbone(config) if arguments.smoke_time_series_backbone else None
    )
    manifest: dict[str, Any] = {
        "config": str(arguments.config),
        "created_at": datetime.now(UTC).isoformat(),
        "download_policy": {
            "hf_hub_disable_xet": os.environ.get("HF_HUB_DISABLE_XET", "0"),
            "hf_xet_num_concurrent_range_gets": os.environ.get(
                "HF_XET_NUM_CONCURRENT_RANGE_GETS", ""
            ),
            "hf_hub_etag_timeout": os.environ["HF_HUB_ETAG_TIMEOUT"],
            "hf_hub_download_timeout": os.environ["HF_HUB_DOWNLOAD_TIMEOUT"],
            "max_attempts": max_attempts,
            "max_workers": arguments.max_workers,
            "retry_backoff_seconds": retry_backoff_seconds,
        },
        "local_files_only_verified": True,
        "kronos_source_revision": config.model.kronos_source_revision,
        "repository_revisions": repositories,
        "repositories": resolved,
        "time_series_smoke_test": smoke_result,
        "verify_only": arguments.verify_only,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Verified {len(repositories)} repositories for local_files_only loading")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
