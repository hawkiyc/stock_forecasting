"""Resumable numerical validation across rules, ML, DL, and the trained Kronos model."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fin_ts_multimodal.baselines import (
    RULE_BASELINE_NAMES,
    BaselineArrays,
    GradientBoostingBaseline,
    baseline_arrays,
    fit_causal_gru,
    fit_dlinear,
    fit_patchtst,
    rule_baseline_suite,
)
from fin_ts_multimodal.cli.evaluate import evaluate_checkpoint, resolve_checkpoint
from fin_ts_multimodal.config import ExperimentConfig
from fin_ts_multimodal.data import FinancialWindowDataset
from fin_ts_multimodal.data.io import read_processed_records
from fin_ts_multimodal.run_contract import (
    training_resume_contract_fingerprint,
    validate_training_resume_contract,
)
from fin_ts_multimodal.run_paths import (
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
from fin_ts_multimodal.training import (
    deterministic_stratified_indices,
    resolve_processed_dataset,
)

LEARNED_BASELINES = ("gbdt", "gru", "dlinear", "patchtst")
FULL_MODEL_NAME = "kronos_full"
ALL_VALIDATION_MODELS = (*RULE_BASELINE_NAMES, *LEARNED_BASELINES, FULL_MODEL_NAME)
EVALUATION_CONTRACT_VERSION = "4.0"
VALIDATION_BENCHMARK_SCHEMA_VERSION = "4.0"
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
    "kronos_full": "Shared Kronos-base LoRA encoder with dynamic benchmark conditioning.",
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
            "model_architecture_sha256": config.model.architecture_digest(),
            "config": config.as_dict(),
            "models": list(models),
            "seeds": list(seeds),
            "checkpoint": {
                "path": str(checkpoint.resolve(strict=False)),
                "files": checkpoint_files,
            },
            "evaluation_schema": VALIDATION_BENCHMARK_SCHEMA_VERSION,
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
    nested: dict[str, Any] = {}
    prefix = "validation/"
    for name, value in flattened.items():
        if not isinstance(name, str) or not name.startswith(prefix):
            continue
        components = name[len(prefix) :].split("/")
        cursor = nested
        for component in components[:-1]:
            cursor = cursor.setdefault(component, {})
        cursor[components[-1]] = value
    if "primary_5d" not in nested or "cross_sectional_5d" not in nested:
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


def select_training_records(
    records: list[dict[str, Any]],
    fraction: float,
    max_samples: int | None,
) -> list[dict[str, Any]]:
    """Use the exact deterministic market/asset stratification used by training."""

    dataset = FinancialWindowDataset(records, split=None)
    indices = deterministic_stratified_indices(
        dataset,
        fraction=fraction,
        max_samples=max_samples,
    )
    return [dataset.records[index] for index in indices]


def _comparison_summary(models: dict[str, Any]) -> dict[str, Any]:
    paths = {
        "selection_score": "primary_5d/selection_score",
        "normalized_pinball": "primary_5d/normalized_pinball",
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
        training_contract, current_digest = training_resume_contract_fingerprint(config)
        if current_digest != training_digest:
            raise RuntimeError("Training resume contract changed during validation setup")
        self.config = config
        self.run_id = run_id
        self.checkpoint = checkpoint
        self.output = output
        self.lifecycle = lifecycle
        self.models = models
        self.seeds = seeds
        self.resume = resume
        self.recompute_full_model = recompute_full_model
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
            if (
                payload.get("schema_version") != VALIDATION_BENCHMARK_SCHEMA_VERSION
                or payload.get("evaluation_contract") != self.evaluation_contract
                or payload.get("run_id") != self.run_id
                or Path(str(payload.get("checkpoint", ""))).resolve() != self.checkpoint.resolve()
            ):
                raise ValueError(
                    "Existing numerical validation output does not match this run contract"
                )
            payload.pop("error", None)
            payload["state"] = "running"
            payload["resumed_at"] = _utc_now()
            return payload
        return {
            "schema_version": VALIDATION_BENCHMARK_SCHEMA_VERSION,
            "evaluation_contract": self.evaluation_contract,
            "state": "running",
            "selection_split": "validation",
            "test_unlocked": False,
            "protocol": {
                "causality": "All inputs end at cutoff_at; forward labels are never features.",
                "model_selection": (
                    "Validation primary_5d/selection_score compatibility path; "
                    "the value aggregates normalized pinball over horizons 3 through 14."
                ),
                "training_subset": "Matches the trainer's exact deterministic stratification.",
                "test_policy": "The test split is counted but never evaluated.",
                "classification": (
                    "No classifier is trained; signals are distribution post-processing."
                ),
            },
            "model_definitions": {name: MODEL_DEFINITIONS[name] for name in self.models},
            "run_id": self.run_id,
            "checkpoint": str(self.checkpoint),
            "dataset_profile": self.config.data.dataset_profile,
            "selected_datasets": self.config.data.selected_datasets,
            "model_architecture_sha256": self.config.model.architecture_digest(),
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
    ) -> None:
        existing = self.payload["models"].get(name, {}).get("seed_results", {})
        for seed in self.seeds:
            if self.resume and str(seed) in existing:
                continue
            started_at = time.perf_counter()
            if name == "gbdt":
                metrics = GradientBoostingBaseline.fit(train, seed=seed).evaluate(validation)
            elif name == "gru":
                model, metrics = fit_causal_gru(
                    train,
                    validation,
                    seed=seed,
                    epochs=self.config.validation.neural_epochs,
                    patience=self.config.validation.neural_patience,
                    batch_size=self.config.validation.neural_batch_size,
                    learning_rate=self.config.validation.neural_learning_rate,
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
            records = read_processed_records(
                resolve_processed_dataset(self.config.data.processed_path)
            )
            split_records = {
                split: [record for record in records if record.get("split") == split]
                for split in ("train", "validation", "test")
            }
            if any(not split_records[split] for split in split_records):
                raise ValueError("Validation requires non-empty chronological splits")
            self.payload["raw_sample_counts"] = {
                split: len(values) for split, values in split_records.items()
            }
            split_records["train"] = select_training_records(
                split_records["train"],
                self.config.data.train_fraction,
                self.config.data.max_samples,
            )
            self.payload["sample_counts"] = {
                split: len(values) for split, values in split_records.items()
            }
            self.payload["runtime"] = {
                "cuda_available": torch.cuda.is_available(),
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
            }
            arrays = {
                split: baseline_arrays(split_records[split]) for split in ("train", "validation")
            }
            requested_rules = [name for name in RULE_BASELINE_NAMES if name in self.models]
            if requested_rules and any(not self._completed(name) for name in requested_rules):
                started_at = time.perf_counter()
                rule_results = rule_baseline_suite(arrays["train"], arrays["validation"])
                duration = time.perf_counter() - started_at
                for name in requested_rules:
                    if not self._completed(name):
                        self._record_single(name, rule_results[name], "past_only_rule", duration)
            for name in LEARNED_BASELINES:
                if name in self.models and not self._completed(name):
                    self._run_learned_baseline(name, arrays["train"], arrays["validation"])
            if FULL_MODEL_NAME in self.models and not self._completed(FULL_MODEL_NAME):
                started_at = time.perf_counter()
                if self.recompute_full_model:
                    result = evaluate_checkpoint(
                        self.config,
                        self.checkpoint,
                        split="validation",
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

    run = wandb.init(
        id=run_id,
        project=config.wandb.project,
        entity=config.wandb.entity or None,
        resume="must",
        job_type="post-training-validation",
        dir=str(config.wandb.directory),
        mode=config.wandb.mode,
    )
    if str(run.id) != run_id:
        run.finish(exit_code=1)
        raise RuntimeError("W&B returned a run ID different from validation run_id")
    run.log(_flatten_numeric(payload.get("models", {}), "benchmark_validation"))
    artifact = wandb.Artifact(f"validation-benchmark-{run_id}", type="evaluation")
    artifact.add_file(str(output))
    run.log_artifact(artifact, aliases=["validation", "latest"])
    run.summary["benchmark_validation_completed"] = payload.get("state") == "ready"
    run.summary["benchmark_validation_path"] = str(output)
    run.finish()
