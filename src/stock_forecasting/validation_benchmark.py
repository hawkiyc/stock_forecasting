"""Resumable numerical validation across rules, ML, DL, and the trained Kronos model."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
import uuid
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch.utils.data import DataLoader

from stock_forecasting.baselines import (
    RULE_BASELINE_NAMES,
    BaselineArrays,
    BaselineRecordDataset,
    GradientBoostingBaseline,
    baseline_arrays,
    concatenate_baseline_batches,
    fit_causal_gru,
    fit_dlinear,
    fit_patchtst,
    rule_baseline_suite,
)
from stock_forecasting.cli.evaluate import evaluate_checkpoint, resolve_checkpoint
from stock_forecasting.config import ExperimentConfig
from stock_forecasting.data import (
    BlockwisePermutationSampler,
    LazyFinancialWindowDataset,
)
from stock_forecasting.evaluation_protocol import (
    EVALUATION_PROTOCOL_VERSION,
    evaluation_sampler,
    paired_block_comparison,
    sample_membership,
)
from stock_forecasting.evaluation_resume_migrations import EVALUATION_RESUME_MIGRATIONS
from stock_forecasting.run_contract import (
    training_resume_contract,
    validate_training_resume_contract,
)
from stock_forecasting.run_paths import (
    LIFECYCLE_SCHEMA_VERSION,
    checkpoint_run_directory,
    validate_checkpoint_path,
    validate_evaluation_path,
    validate_run_id,
    validate_run_lifecycle_payload,
    validate_selected_run_environment,
    validate_training_lifecycle_path,
    validate_validation_lifecycle_path,
    validate_wandb_directory,
)
from stock_forecasting.runtime_resources import detect_available_memory
from stock_forecasting.training import (
    _loader_process_options,
    _requested_dataloader_workers,
    plan_dataloader_workers,
    resolve_runtime_robust_scales,
)
from stock_forecasting.training_paths import resolve_bar_store_path
from stock_forecasting.wandb_status import update_wandb_status

LEARNED_BASELINES = ("gbdt", "gru", "dlinear", "patchtst")
FULL_MODEL_NAME = "kronos_full"
ALL_VALIDATION_MODELS = (*RULE_BASELINE_NAMES, *LEARNED_BASELINES, FULL_MODEL_NAME)
EVALUATION_CONTRACT_VERSION = "6.0"
VALIDATION_BENCHMARK_SCHEMA_VERSION = "6.0"
LEGACY_EVALUATION_CONTRACT_VERSIONS: tuple[str, ...] = ()
VALIDATION_NON_NUMERICAL_CONFIG_FIELDS = (
    "enabled",
    "auto_run_after_training",
    "output_root",
    "models",
    "seeds",
    "recompute_full_model",
    "resume_completed_models",
)
MODEL_DEFINITIONS = {
    "always_buy": "Constant positive-alpha distribution calibrated only on train labels.",
    "zero_return": "Constant zero-alpha distribution calibrated only on train labels.",
    "momentum_5d": "Past-only asset-minus-benchmark five-day momentum comparator.",
    "reversal_5d": "Negative past-only relative five-day momentum comparator.",
    "ma_crossover": "Past-only relative 20/50-day moving-average comparator.",
    "rsi": "Past-only relative RSI(14) comparator.",
    "macd": "Past-only relative MACD(12,26,9) comparator.",
    "volatility_scaled": "Relative momentum scaled by past relative-return volatility.",
    "gbdt": "Per-horizon histogram gradient-boosted alpha quantile regressors.",
    "gru": "Shared causal GRU over paired asset and benchmark OHLCV histories.",
    "dlinear": "DLinear-style decomposition of relative historical OHLCV.",
    "patchtst": "Compact causal PatchTST over paired historical OHLCV.",
    "kronos_full": (
        "Shared Kronos-base LoRA with gated conditioning "
        "and configurable scale/benchmark residuals."
    ),
}


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    try:
        temporary.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _file_fingerprint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation contract requires checkpoint file: {path}")
    digest = hashlib.sha256()
    size_bytes = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            size_bytes += len(chunk)
    return {"sha256": digest.hexdigest(), "size_bytes": size_bytes}


def _validation_numerical_config(config: ExperimentConfig) -> dict[str, Any]:
    """Return only settings that can change numerical benchmark results."""

    payload = config.validation.model_dump(mode="json")
    return {
        name: value
        for name, value in payload.items()
        if name not in VALIDATION_NON_NUMERICAL_CONFIG_FIELDS
    }


def _stored_validation_numerical_config(
    inputs: dict[str, Any],
) -> dict[str, Any] | None:
    current = inputs.get("validation_numerical_config")
    if isinstance(current, dict):
        source = current
    else:
        legacy_config = inputs.get("config")
        if not isinstance(legacy_config, dict):
            return None
        legacy_validation = legacy_config.get("validation")
        if not isinstance(legacy_validation, dict):
            return None
        source = legacy_validation
    return {
        name: value
        for name, value in source.items()
        if name not in VALIDATION_NON_NUMERICAL_CONFIG_FIELDS
    }


def _evaluation_contract_resume_view(
    contract: Any,
) -> dict[str, Any] | None:
    """Normalize current and legacy contracts to reusable numerical inputs."""

    if not isinstance(contract, dict):
        return None
    if contract.get("version") not in (
        *LEGACY_EVALUATION_CONTRACT_VERSIONS,
        EVALUATION_CONTRACT_VERSION,
    ):
        return None
    inputs = contract.get("inputs")
    if not isinstance(inputs, dict):
        return None
    required_inputs = (
        "run_id",
        "training_resume_contract_sha256",
        "dataset_artifacts",
        "model_architecture_sha256",
        "models",
        "seeds",
        "checkpoint",
        "evaluation_schema",
        "evaluation_protocol",
        "evaluation_implementation",
    )
    if any(name not in inputs for name in required_inputs):
        return None
    numerical_config = _stored_validation_numerical_config(inputs)
    if numerical_config is None:
        return None
    return {
        **{name: inputs[name] for name in required_inputs},
        "validation_numerical_config": numerical_config,
    }


def _evaluation_contracts_are_resume_compatible(
    stored: Any,
    current: dict[str, Any],
) -> bool:
    """Compare only inputs that can change persisted numerical results."""

    stored_view = _evaluation_contract_resume_view(stored)
    current_view = _evaluation_contract_resume_view(current)
    if stored_view is None or current_view is None:
        return False
    if stored_view == current_view:
        return True
    stored_files = stored_view.pop("evaluation_implementation")
    current_files = current_view.pop("evaluation_implementation")
    if (
        stored_view != current_view
        or not isinstance(stored_files, dict)
        or not isinstance(current_files, dict)
        or set(stored_files) != set(current_files)
    ):
        return False
    changed_files = {name for name in stored_files if stored_files[name] != current_files[name]}
    return any(
        changed_files == set(migration["from_files"]) == set(migration["to_files"])
        and all(stored_files[name] == value for name, value in migration["from_files"].items())
        and all(current_files[name] == value for name, value in migration["to_files"].items())
        for migration in EVALUATION_RESUME_MIGRATIONS
    )


def build_evaluation_contract(
    config: ExperimentConfig,
    *,
    run_id: str,
    checkpoint: Path,
    training_resume_contract_sha256: str,
    dataset_artifacts: dict[str, Any],
    models: list[str],
    seeds: list[int],
) -> dict[str, Any]:
    """Fingerprint every semantic input that makes persisted validation reusable."""

    checkpoint_files = {
        name: _file_fingerprint(checkpoint / name)
        for name in ("adapter.safetensors", "resolved-config.yaml", "trainer-state.json")
    }
    body = {
        "version": EVALUATION_CONTRACT_VERSION,
        "inputs": {
            "run_id": run_id,
            "training_resume_contract_sha256": training_resume_contract_sha256,
            "dataset_artifacts": dataset_artifacts,
            "model_architecture_sha256": config.model_architecture_digest(),
            "validation_numerical_config": _validation_numerical_config(config),
            "models": list(models),
            "seeds": list(seeds),
            "checkpoint": {
                "path": str(checkpoint.resolve(strict=False)),
                "files": checkpoint_files,
            },
            "evaluation_schema": VALIDATION_BENCHMARK_SCHEMA_VERSION,
            "evaluation_protocol": {
                "version": EVALUATION_PROTOCOL_VERSION,
                "split": "test" if config.data.fixed_split else "validation",
                "fixed_split": config.data.fixed_split,
                "max_samples": config.training.evaluation_max_samples,
            },
            "evaluation_implementation": {
                name: _file_fingerprint(Path(__file__).parent / name)
                for name in (
                    "validation_benchmark.py",
                    "baselines.py",
                    "evaluation_protocol.py",
                    "metrics.py",
                    "cli/evaluate.py",
                    "evaluation_store.py",
                    "baseline_contract.py",
                )
            },
        },
    }
    return {
        **body,
        "digest_algorithm": "sha256",
        "digest": hashlib.sha256(_canonical_json(body).encode("utf-8")).hexdigest(),
    }


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def validated_persistent_path(path: Path, volume_root: Path, label: str) -> Path:
    resolved = path.expanduser().resolve(strict=False)
    root = volume_root.expanduser().resolve(strict=False)
    if resolved == Path("/workspace") or Path("/workspace") in resolved.parents:
        raise ValueError(f"{label} must never use ephemeral /workspace")
    if not _is_relative_to(resolved, root):
        raise ValueError(f"{label} must be inside NETWORK_VOLUME_ROOT: {resolved}")
    return resolved


def discover_run_id(
    explicit_run_id: str | None,
    *,
    training_lifecycle: Path,
) -> str:
    candidate = explicit_run_id or os.environ.get("WANDB_RUN_ID", "")
    if not candidate:
        resolved_lifecycle = training_lifecycle.resolve(strict=False)
        if len(resolved_lifecycle.parents) < 3:
            raise ValueError("Training lifecycle path is not below a network volume root")
        volume_root = resolved_lifecycle.parents[2]
        validate_training_lifecycle_path(
            training_lifecycle,
            network_volume_root=volume_root,
        )
        lifecycle = _read_json(training_lifecycle)
        if lifecycle.get("state") not in {"ready", "failed", "timed_out"}:
            raise ValueError("Latest training lifecycle is not terminal")
        if lifecycle.get("training_completed") is not True:
            raise ValueError("Latest lifecycle does not identify a completed training run")
        candidate = validate_run_lifecycle_payload(
            lifecycle,
            network_volume_root=volume_root,
            expected_kind="stage1-training",
        )
    return validate_selected_run_environment(
        candidate,
        os.environ.get("WANDB_RUN_ID"),
        os.environ.get("RUNPOD_RUN_KEY"),
    )


def discover_checkpoint(
    checkpoint: Path | None,
    *,
    saved_model_root: Path,
    run_id: str,
) -> Path:
    if checkpoint is not None:
        resolved = resolve_checkpoint(checkpoint, saved_model_root=saved_model_root).resolve()
    else:
        run_directory = checkpoint_run_directory(saved_model_root, run_id)
        if not (run_directory / "best-checkpoint.json").is_file():
            raise ValueError(f"Validation-ranked checkpoint is unavailable for run {run_id}")
        resolved = resolve_checkpoint(
            run_directory,
            saved_model_root=saved_model_root,
        ).resolve()
    return validate_checkpoint_path(
        resolved,
        saved_model_root=saved_model_root,
        run_id=run_id,
    )


def _unflatten_validation_metrics(flattened: dict[str, Any]) -> dict[str, Any]:
    prefix = "validation/"
    prefixed: dict[str, Any] = {}
    unprefixed: dict[str, Any] = {}
    for name, value in flattened.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Checkpoint validation metric name is invalid")
        if name.startswith(prefix):
            relative_name = name[len(prefix) :]
            if not relative_name:
                raise ValueError("Checkpoint validation metric name is invalid")
            prefixed[relative_name] = value
        else:
            unprefixed[name] = value
    if prefixed and unprefixed:
        raise ValueError("Checkpoint mixes validation metric namespaces")

    metrics = prefixed or unprefixed
    nested: dict[str, Any] = {}
    for name, value in sorted(metrics.items()):
        components = name.split("/")
        if any(not component for component in components):
            raise ValueError("Checkpoint validation metric name is invalid")
        cursor = nested
        for component in components[:-1]:
            child = cursor.setdefault(component, {})
            if not isinstance(child, dict):
                raise ValueError("Checkpoint validation metric paths collide")
            cursor = child
        leaf = components[-1]
        if leaf in cursor:
            raise ValueError("Checkpoint validation metric paths collide")
        cursor[leaf] = value
    if not isinstance(nested.get("primary_5d"), dict) or not isinstance(
        nested.get("cross_sectional_5d"), dict
    ):
        raise ValueError("Checkpoint has no complete numerical validation metric snapshot")
    return nested


def checkpoint_validation_snapshot(checkpoint: Path) -> dict[str, Any]:
    state = _read_json(checkpoint / "trainer-state.json")
    metrics = state.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("Checkpoint trainer state has no validation metrics")
    return {
        "metrics": _unflatten_validation_metrics(metrics),
        "source": "checkpoint_validation_snapshot",
        "global_step": state.get("global_step"),
        "created_at": state.get("created_at"),
    }


def _aggregate_numeric(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    keys = sorted(set().union(*(payload.keys() for payload in payloads)))
    output: dict[str, Any] = {}
    for key in keys:
        values = [payload[key] for payload in payloads if key in payload]
        if values and all(isinstance(value, dict) for value in values):
            output[key] = _aggregate_numeric([value for value in values if isinstance(value, dict)])
        elif values and all(
            isinstance(value, (int, float)) and math.isfinite(float(value)) for value in values
        ):
            array = np.asarray(values, dtype=np.float64)
            output[key] = {
                "count": int(array.size),
                "median": float(np.median(array)),
                "min": float(array.min()),
                "max": float(array.max()),
            }
    return output


def _flatten_numeric(payload: dict[str, Any], prefix: str = "") -> dict[str, float]:
    output: dict[str, float] = {}
    for key, value in payload.items():
        if key in ("sample_membership", "daily_normalized_pinball"):
            continue
        name = f"{prefix}/{key}" if prefix else key
        if isinstance(value, dict):
            output.update(_flatten_numeric(value, name))
        elif isinstance(value, (int, float)) and math.isfinite(float(value)):
            output[name] = float(value)
    return output


def _metric_path(payload: dict[str, Any], path: str) -> float:
    value: Any = payload
    for component in path.split("/"):
        if not isinstance(value, dict) or component not in value:
            return math.nan
        value = value[component]
    return float(value) if isinstance(value, (int, float)) else math.nan


def _median_metric(payloads: list[dict[str, Any]], path: str) -> float | None:
    values = [_metric_path(payload, path) for payload in payloads]
    finite = [value for value in values if math.isfinite(value)]
    return float(np.median(finite)) if finite else None


def _lazy_baseline_arrays(
    config: ExperimentConfig,
    *,
    split: Literal["train", "validation", "test"],
    seed: int,
    build_arrays: bool = True,
    loader_options: dict[str, Any] | None = None,
) -> tuple[int, int, BaselineArrays | None]:
    """Stream a bounded lazy sample into numerical arrays, never window artifacts."""

    dataset = LazyFinancialWindowDataset(
        resolve_bar_store_path(config.data.bar_store_path),
        split=split,
        window_size=config.data.input_length,
        h_start=config.data.h_start,
    )
    maximum = config.validation.baseline_max_samples_per_split
    if split == "train" and config.data.max_samples is not None:
        maximum = min(maximum or len(dataset), config.data.max_samples)
    sampler = (
        evaluation_sampler(len(dataset), config, split)
        if split != "train"
        else BlockwisePermutationSampler(
            len(dataset),
            fraction=config.data.train_fraction,
            max_samples=maximum,
            seed=seed,
            block_size=128,
        )
    )
    if build_arrays:
        # Include concatenation copies, labels/features, metadata, and loader buffers.
        bytes_per_sample = config.data.input_length * 2 * 5 * 4 + 2048
        estimated_bytes = len(sampler) * bytes_per_sample * 3
        available_bytes = detect_available_memory().available_bytes
        if estimated_bytes > available_bytes // 2:
            raise MemoryError(
                "Bounded baseline arrays exceed half of available host memory; "
                "use a larger-memory evaluation Pod without changing sample membership"
            )
    arrays = (
        concatenate_baseline_batches(
            DataLoader(
                BaselineRecordDataset(dataset),
                batch_size=128,
                sampler=sampler,
                collate_fn=baseline_arrays,
                **(loader_options or {}),
            )
        )
        if build_arrays
        else None
    )
    return len(dataset), len(sampler), arrays


def _comparison_summary(models: dict[str, Any]) -> dict[str, Any]:
    paths = {
        "selection_score": "primary_5d/selection_score",
        "normalized_pinball": "aggregate/normalized_pinball",
        "median_correlation": "primary_5d/median_correlation",
        "interval_coverage": "primary_5d/interval_coverage",
        "rank_ic_mean": "cross_sectional_5d/rank_ic_mean",
        "net_long_short_sharpe": "cross_sectional_5d/net_long_short_sharpe",
    }
    rows: list[dict[str, Any]] = []
    for name, result in models.items():
        if result.get("state") != "complete":
            continue
        payloads = (
            list(result.get("seed_results", {}).values())
            if result.get("seed_results")
            else [result.get("metrics", {})]
        )
        row: dict[str, Any] = {"model": name}
        row.update(
            {
                metric_name: _median_metric(payloads, metric_path)
                for metric_name, metric_path in paths.items()
            }
        )
        rows.append(row)
    return {
        "rows": sorted(rows, key=lambda row: row["model"]),
        "selection_score_ranking": [
            row["model"]
            for row in sorted(
                rows,
                key=lambda row: (
                    math.inf if row["selection_score"] is None else row["selection_score"],
                    row["model"],
                ),
            )
        ],
    }


class ValidationBenchmark:
    """Persist each numerical comparator so interrupted GPU validation can resume."""

    def __init__(
        self,
        config: ExperimentConfig,
        *,
        run_id: str,
        checkpoint: Path,
        output: Path,
        lifecycle: Path,
        models: list[str],
        seeds: list[int],
        resume: bool,
        recompute_full_model: bool,
        defer_terminal_lifecycle: bool = False,
    ) -> None:
        run_id = validate_run_id(run_id)
        checkpoint = validate_checkpoint_path(
            checkpoint,
            saved_model_root=config.training.output_root,
            run_id=run_id,
        )
        lifecycle = validate_validation_lifecycle_path(
            lifecycle,
            network_volume_root=config.validation.output_root.parent,
        )
        validate_evaluation_path(
            output,
            evaluation_root=config.validation.output_root,
            run_id=run_id,
            filename="validation-benchmark.json",
        )
        unknown = sorted(set(models) - set(ALL_VALIDATION_MODELS))
        if unknown:
            raise ValueError(f"Unknown validation models: {unknown}")
        if not models or not seeds:
            raise ValueError("Validation requires at least one model and one seed")
        training_digest = validate_training_resume_contract(
            config,
            run_directory=checkpoint.parent,
            checkpoint_directory=checkpoint,
        )
        training_contract = training_resume_contract(config)
        self.config = config
        self.run_id = run_id
        self.checkpoint = checkpoint
        self.output = output
        self.lifecycle = lifecycle
        self.models = models
        self.seeds = seeds
        self.resume = resume
        self.evaluation_split: Literal["validation", "test"] = (
            "test" if config.data.fixed_split else "validation"
        )
        self.recompute_full_model = recompute_full_model or self.evaluation_split == "test"
        requested_workers, source = _requested_dataloader_workers(config)
        self.worker_plan = plan_dataloader_workers(requested_workers, source=source)
        self.loader_options = _loader_process_options(self.worker_plan, persistent=False)
        if self.worker_plan.effective_workers > 0:
            self.loader_options["multiprocessing_context"] = "spawn"
        if config.model.time_series_backend == "kronos" and self.worker_plan.effective_workers == 0:
            raise RuntimeError(
                "No safe baseline DataLoader worker fits the CPU/memory budget; "
                "increase Pod resources or the configured worker limit"
            )
        self.defer_terminal_lifecycle = defer_terminal_lifecycle
        self.evaluation_contract = build_evaluation_contract(
            config,
            run_id=run_id,
            checkpoint=checkpoint,
            training_resume_contract_sha256=training_digest,
            dataset_artifacts=training_contract["dataset_artifacts"],
            models=models,
            seeds=seeds,
        )
        self.payload = self._initial_payload()

    def _initial_payload(self) -> dict[str, Any]:
        if self.resume and self.output.is_file():
            payload = _read_json(self.output)
            stored_contract = payload.get("evaluation_contract")
            if (
                payload.get("schema_version") != VALIDATION_BENCHMARK_SCHEMA_VERSION
                or not _evaluation_contracts_are_resume_compatible(
                    stored_contract,
                    self.evaluation_contract,
                )
                or payload.get("run_id") != self.run_id
                or Path(str(payload.get("checkpoint", ""))).resolve() != self.checkpoint.resolve()
            ):
                raise ValueError(
                    "Existing numerical validation output does not match this run contract"
                )
            if stored_contract != self.evaluation_contract:
                payload["evaluation_contract"] = self.evaluation_contract
                payload["resume_contract_normalized_at"] = _utc_now()
            payload.pop("error", None)
            payload["state"] = "running"
            payload["resumed_at"] = _utc_now()
            return payload
        return {
            "schema_version": VALIDATION_BENCHMARK_SCHEMA_VERSION,
            "evaluation_contract": self.evaluation_contract,
            "state": "running",
            "selection_split": "validation",
            "evaluation_split": self.evaluation_split,
            "test_unlocked": False,
            "protocol": {
                "causality": "All inputs end at cutoff_at; forward labels are never features.",
                "model_selection": (
                    "Validation primary_5d/selection_score compatibility path; "
                    "the value aggregates normalized pinball over horizons "
                    f"{self.config.data.h_start} through {self.config.data.max_horizon}."
                ),
                "training_subset": (
                    "Matches the trainer's deterministic blockwise target-count policy."
                ),
                "test_policy": (
                    "Holdout is scored only after validation selects checkpoints; "
                    "it never drives training, early stopping, or checkpoint selection."
                    if self.evaluation_split == "test"
                    else "Legacy fraction-split report: validation only, not a holdout result."
                ),
                "classification": (
                    "No classifier is trained; signals are distribution post-processing."
                ),
            },
            "model_definitions": {name: MODEL_DEFINITIONS[name] for name in self.models},
            "run_id": self.run_id,
            "checkpoint": str(self.checkpoint),
            "dataset_profile": self.config.data.dataset_profile,
            "selected_datasets": self.config.data.selected_datasets,
            "model_architecture_sha256": self.config.model_architecture_digest(),
            "config": self.config.as_dict(),
            "requested_models": self.models,
            "seeds": self.seeds,
            "started_at": _utc_now(),
            "models": {},
        }

    def _publish(self) -> None:
        self.payload["updated_at"] = _utc_now()
        _atomic_json(self.output, self.payload)

    def _publish_lifecycle(self, state: str, *, error: str | None = None) -> None:
        payload: dict[str, Any] = {
            "schema_version": LIFECYCLE_SCHEMA_VERSION,
            "kind": "stage1-validation",
            "state": state,
            "generated_at": _utc_now(),
            "pod_id": os.environ.get("RUNPOD_POD_ID", ""),
            "launch_id": os.environ.get("RUNPOD_LAUNCH_ID", "manual-validation"),
            "wandb_run_id": self.run_id,
            "checkpoint": str(self.checkpoint),
            "result_path": str(self.output),
            "validation_completed": state == "ready",
        }
        if error:
            payload["error"] = error
        _atomic_json(self.lifecycle, payload)

    def _completed(self, model_name: str) -> bool:
        result = self.payload.get("models", {}).get(model_name, {})
        if not self.resume or result.get("state") != "complete":
            return False
        return not (
            model_name == FULL_MODEL_NAME
            and self.recompute_full_model
            and result.get("source") != "checkpoint_recomputed"
        )

    def _record_single(
        self,
        name: str,
        metrics: dict[str, Any],
        source: str,
        duration_seconds: float,
    ) -> None:
        self.payload["models"][name] = {
            "state": "complete",
            "source": source,
            "metrics": metrics,
            "duration_seconds": duration_seconds,
            "completed_at": _utc_now(),
        }
        self._publish()

    def _record_seed(
        self,
        name: str,
        seed: int,
        metrics: dict[str, Any],
        duration_seconds: float,
    ) -> None:
        result = self.payload["models"].setdefault(
            name,
            {"state": "running", "seed_results": {}, "seed_durations_seconds": {}},
        )
        result["seed_results"][str(seed)] = metrics
        result["seed_durations_seconds"][str(seed)] = duration_seconds
        self._publish()

    def _run_learned_baseline(
        self,
        name: str,
        train: BaselineArrays,
        validation: BaselineArrays,
        evaluation: BaselineArrays,
        robust_scales: list[float] | None,
    ) -> None:
        existing = self.payload["models"].get(name, {}).get("seed_results", {})
        for seed in self.seeds:
            if self.resume and str(seed) in existing:
                continue
            started_at = time.perf_counter()
            if name == "gbdt":
                metrics = GradientBoostingBaseline.fit(
                    train,
                    seed=seed,
                    robust_scales=robust_scales,
                ).evaluate(evaluation)
            elif name == "gru":
                model, metrics = fit_causal_gru(
                    train,
                    validation,
                    seed=seed,
                    epochs=self.config.validation.neural_epochs,
                    patience=self.config.validation.neural_patience,
                    batch_size=self.config.validation.neural_batch_size,
                    learning_rate=self.config.validation.neural_learning_rate,
                    robust_scales=robust_scales,
                    evaluation=evaluation,
                    loader_options=self.loader_options,
                )
                del model
            elif name == "dlinear":
                model, metrics = fit_dlinear(
                    train,
                    validation,
                    seed=seed,
                    epochs=self.config.validation.neural_epochs,
                    patience=self.config.validation.neural_patience,
                    batch_size=self.config.validation.neural_batch_size,
                    learning_rate=self.config.validation.neural_learning_rate,
                    robust_scales=robust_scales,
                    evaluation=evaluation,
                    loader_options=self.loader_options,
                )
                del model
            elif name == "patchtst":
                model, metrics = fit_patchtst(
                    train,
                    validation,
                    seed=seed,
                    epochs=self.config.validation.neural_epochs,
                    patience=self.config.validation.neural_patience,
                    batch_size=self.config.validation.neural_batch_size,
                    learning_rate=self.config.validation.neural_learning_rate,
                    robust_scales=robust_scales,
                    evaluation=evaluation,
                    loader_options=self.loader_options,
                )
                del model
            else:
                raise AssertionError(f"Unhandled learned baseline: {name}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._record_seed(name, seed, metrics, time.perf_counter() - started_at)
            existing = self.payload["models"][name]["seed_results"]
        result = self.payload["models"][name]
        if all(str(seed) in result["seed_results"] for seed in self.seeds):
            result["aggregate"] = _aggregate_numeric(
                [result["seed_results"][str(seed)] for seed in self.seeds]
            )
            result["state"] = "complete"
            result["completed_at"] = _utc_now()
            self._publish()

    def run(self) -> dict[str, Any]:
        self._publish_lifecycle("preparing")
        self._publish()
        try:
            if self.config.validation.require_prebuilt_baselines:
                return self._run_prebuilt()
            robust_scales = None
            if self.config.data.fixed_split:
                calibration_dataset = LazyFinancialWindowDataset(
                    resolve_bar_store_path(self.config.data.bar_store_path),
                    split="train",
                    window_size=self.config.data.input_length,
                    h_start=self.config.data.h_start,
                )
                calibration = resolve_runtime_robust_scales(
                    calibration_dataset,
                    sample_count=self.config.data.label_scale_calibration_samples,
                    seed=self.config.data.calibration_seed,
                    worker_plan=self.worker_plan,
                )
                robust_scales = np.asarray(calibration.scales, dtype=np.float32).tolist()
                stored_scales = _read_json(self.checkpoint / "trainer-state.json").get(
                    "runtime_robust_scales"
                )
                if stored_scales != robust_scales:
                    raise ValueError("Checkpoint and benchmark train-calibration scales differ")
                self.payload["label_scale_calibration"] = {
                    "split": "train",
                    "identity_sha256": calibration.identity_sha256,
                    "sample_count": calibration.sample_count,
                    "seed": self.config.data.calibration_seed,
                    "scales": robust_scales,
                }
            arrays: dict[str, BaselineArrays] = {}
            raw_counts: dict[str, int] = {}
            sample_counts: dict[str, int] = {}
            for position, split in enumerate(("train", "validation", "test")):
                raw_count, sample_count, split_arrays = _lazy_baseline_arrays(
                    self.config,
                    split=split,
                    seed=self.config.training.seed + position,
                    build_arrays=split != "test" or self.evaluation_split == "test",
                    loader_options=self.loader_options,
                )
                raw_counts[split] = raw_count
                sample_counts[split] = sample_count
                if split_arrays is not None:
                    arrays[split] = split_arrays
            self.payload["raw_sample_counts"] = raw_counts
            self.payload["sample_counts"] = sample_counts
            self.payload["runtime"] = {
                "cuda_available": torch.cuda.is_available(),
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
                "dataloader_worker_plan": self.worker_plan.as_dict(),
            }
            evaluation = arrays[self.evaluation_split]
            self.payload["evaluation_membership"] = sample_membership(
                evaluation.symbols,
                evaluation.dates,
            )
            requested_rules = [name for name in RULE_BASELINE_NAMES if name in self.models]
            if requested_rules and any(not self._completed(name) for name in requested_rules):
                started_at = time.perf_counter()
                rule_results = rule_baseline_suite(
                    arrays["train"],
                    evaluation,
                    robust_scales=robust_scales,
                )
                duration = time.perf_counter() - started_at
                for name in requested_rules:
                    if not self._completed(name):
                        self._record_single(name, rule_results[name], "past_only_rule", duration)
            for name in LEARNED_BASELINES:
                if name in self.models and not self._completed(name):
                    self._run_learned_baseline(
                        name,
                        arrays["train"],
                        arrays["validation"],
                        evaluation,
                        robust_scales,
                    )
            if FULL_MODEL_NAME in self.models and not self._completed(FULL_MODEL_NAME):
                started_at = time.perf_counter()
                if self.recompute_full_model or self.evaluation_split == "test":
                    result = evaluate_checkpoint(
                        self.config,
                        self.checkpoint,
                        split=self.evaluation_split,
                    )
                    metrics = result["metrics"]
                    source = "checkpoint_recomputed"
                else:
                    snapshot = checkpoint_validation_snapshot(self.checkpoint)
                    metrics = snapshot["metrics"]
                    source = snapshot["source"]
                self._record_single(
                    FULL_MODEL_NAME,
                    metrics,
                    source,
                    time.perf_counter() - started_at,
                )
            incomplete = [
                name
                for name in self.models
                if self.payload["models"].get(name, {}).get("state") != "complete"
            ]
            if incomplete:
                raise RuntimeError(f"Validation did not complete models: {incomplete}")
            self.payload["comparison"] = _comparison_summary(self.payload["models"])
            if self.evaluation_split == "test":
                self._verify_paired_results(robust_scales)
                self.payload["test_unlocked"] = True
                self.payload["comparison"]["ranking_role"] = (
                    "Descriptive holdout comparison only; not model or checkpoint selection."
                )
            self.payload.pop("error", None)
            self.payload["state"] = "ready"
            self.payload["completed_at"] = _utc_now()
            self._publish()
            self._publish_lifecycle("finalizing" if self.defer_terminal_lifecycle else "ready")
            return self.payload
        except BaseException as error:
            self.payload["state"] = "failed"
            self.payload["error"] = f"{type(error).__name__}: {error}"
            self._publish()
            self._publish_lifecycle(
                "finalizing" if self.defer_terminal_lifecycle else "failed",
                error=self.payload["error"],
            )
            raise

    def _run_prebuilt(self) -> dict[str, Any]:
        from stock_forecasting.baseline_contract import require_baselines

        cached = require_baselines(self.config)
        scales = cached["robust_scales"]
        stored = _read_json(self.checkpoint / "trainer-state.json").get("runtime_robust_scales")
        if stored != scales:
            raise ValueError("Checkpoint and prebuilt baseline use different train scales")
        self.payload["baseline_identity"] = cached["identity"]
        self.payload["baseline_reused"] = True
        self.payload["sample_counts"] = cached["sample_counts"]
        self.payload["raw_sample_counts"] = cached["sample_counts"]
        self.payload["evaluation_membership"] = cached["evaluation_membership"]
        for name in self.models:
            if name != FULL_MODEL_NAME:
                if name not in cached["models"]:
                    raise ValueError(f"Required baseline is missing: {name}")
                self.payload["models"][name] = cached["models"][name]
        if FULL_MODEL_NAME in self.models and not self._completed(FULL_MODEL_NAME):
            started = time.perf_counter()
            result = evaluate_checkpoint(self.config, self.checkpoint, split="test")
            self._record_single(
                FULL_MODEL_NAME,
                result["metrics"],
                "checkpoint_recomputed",
                time.perf_counter() - started,
            )
        self.payload["comparison"] = _comparison_summary(self.payload["models"])
        self._verify_paired_results(scales)
        self.payload["test_unlocked"] = True
        self.payload["comparison"]["ranking_role"] = (
            "Descriptive full-holdout comparison only; never checkpoint selection."
        )
        self.payload.pop("error", None)
        self.payload["state"] = "ready"
        self.payload["completed_at"] = _utc_now()
        self._publish()
        self._publish_lifecycle("finalizing" if self.defer_terminal_lifecycle else "ready")
        return self.payload

    def _verify_paired_results(self, robust_scales: list[float] | None) -> None:
        """Fail closed on stale scores or mismatched sample membership/scaling."""

        results = self.payload["models"]
        for name, result in results.items():
            scores = result.get("seed_results") or {"single": result.get("metrics", {})}
            for metrics in scores.values():
                if metrics.get("sample_membership") != self.payload["evaluation_membership"]:
                    raise ValueError(f"{name} used different holdout samples")
                if metrics.get("evaluation_robust_scales") != robust_scales:
                    raise ValueError(f"{name} used different train-calibrated scales")
        full = results.get(FULL_MODEL_NAME, {}).get("metrics")
        comparisons = {}
        if full:
            for name, result in results.items():
                if name == FULL_MODEL_NAME:
                    continue
                scores = result.get("seed_results") or {"single": result["metrics"]}
                comparisons[name] = {
                    seed: paired_block_comparison(
                        full["daily_normalized_pinball"],
                        metrics["daily_normalized_pinball"],
                    )
                    for seed, metrics in scores.items()
                }
        self.payload["comparison"]["full_minus_baseline_date_paired"] = comparisons


def log_validation_to_wandb(
    config: ExperimentConfig,
    payload: dict[str, Any],
    output: Path,
) -> None:
    if not config.wandb.enabled or config.wandb.mode == "disabled":
        return
    run_id = validate_selected_run_environment(
        str(payload.get("run_id", "")),
        os.environ.get("WANDB_RUN_ID"),
        os.environ.get("RUNPOD_RUN_KEY"),
    )
    validate_wandb_directory(config.wandb.directory)
    import wandb

    selected_mode = config.wandb.mode
    run: Any | None = None
    fallback_error: BaseException | None = None
    init_arguments: dict[str, Any] = {
        "id": run_id,
        "project": config.wandb.project,
        "entity": config.wandb.entity or None,
        "job_type": "post-training-validation",
        "dir": str(config.wandb.directory),
        "config": {"post_training_validation": config.validation.model_dump(mode="json")},
    }
    try:
        run = wandb.init(
            **init_arguments,
            resume=None if selected_mode == "offline" else "must",
            mode=selected_mode,
        )
    except Exception as error:
        fallback_error = error
        if not config.wandb.allow_offline_fallback:
            update_wandb_status(
                run_id=run_id,
                component="validation",
                state="failed",
                mode=selected_mode,
                project=config.wandb.project,
                entity=config.wandb.entity,
                wandb_directory=config.wandb.directory,
                error=f"{type(error).__name__}: {error}",
            )
            raise
        selected_mode = "offline"
        try:
            run = wandb.init(
                **init_arguments,
                resume=None,
                mode="offline",
                tags=[*config.wandb.tags, "offline-validation-fallback"],
            )
        except BaseException as offline_error:
            update_wandb_status(
                run_id=run_id,
                component="validation",
                state="failed",
                mode="offline",
                project=config.wandb.project,
                entity=config.wandb.entity,
                wandb_directory=config.wandb.directory,
                error=f"{type(offline_error).__name__}: {offline_error}",
            )
            raise

    if run is None:
        raise RuntimeError("W&B validation run was not initialized")
    transaction_directory = Path(str(run.dir)).resolve(strict=False)
    if transaction_directory.name == "files":
        transaction_directory = transaction_directory.parent
    update_wandb_status(
        run_id=run_id,
        component="validation",
        state="offline_pending" if selected_mode == "offline" else "online_running",
        mode=selected_mode,
        project=config.wandb.project,
        entity=config.wandb.entity,
        wandb_directory=config.wandb.directory,
        transaction_directory=transaction_directory,
        error=(
            f"{type(fallback_error).__name__}: {fallback_error}"
            if fallback_error is not None
            else None
        ),
    )
    try:
        if str(run.id) != run_id:
            raise RuntimeError("W&B returned a run ID different from validation run_id")
        trainer_state = _read_json(Path(str(payload["checkpoint"])) / "trainer-state.json")
        validation_step = int(trainer_state.get("global_step", 0))
        if validation_step < 0:
            raise ValueError("Validation checkpoint global_step must be non-negative")
        validation_step_key = "benchmark_validation/global_step"
        run.define_metric("benchmark_validation/*", step_metric=validation_step_key)
        validation_history = _flatten_numeric(
            payload.get("models", {}),
            "benchmark_validation",
        )
        validation_history[validation_step_key] = validation_step
        # The training run has already committed its final internal W&B step.
        # Append a new history row and use the checkpoint step as its chart axis.
        run.log(validation_history)
        artifact = wandb.Artifact(f"validation-benchmark-{run_id}", type="evaluation")
        artifact.add_file(str(output))
        run.log_artifact(artifact, aliases=["validation", "latest"])
        run.summary["benchmark_validation_completed"] = payload.get("state") == "ready"
        run.summary["benchmark_validation_path"] = str(output)
        run.summary["benchmark_validation_global_step"] = validation_step
        run.finish()
    except BaseException as error:
        update_wandb_status(
            run_id=run_id,
            component="validation",
            state=("failed" if selected_mode == "offline" else "offline_pending"),
            mode=selected_mode,
            project=config.wandb.project,
            entity=config.wandb.entity,
            wandb_directory=config.wandb.directory,
            transaction_directory=transaction_directory,
            error=f"{type(error).__name__}: {error}",
        )
        with suppress(BaseException):
            run.finish(exit_code=1)
        raise
    update_wandb_status(
        run_id=run_id,
        component="validation",
        state="offline_pending" if selected_mode == "offline" else "online_finished",
        mode=selected_mode,
        project=config.wandb.project,
        entity=config.wandb.entity,
        wandb_directory=config.wandb.directory,
        transaction_directory=transaction_directory,
    )
