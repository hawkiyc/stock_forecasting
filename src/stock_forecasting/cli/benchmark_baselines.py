"""Benchmark rule, traditional ML, and causal DL models on identical splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from stock_forecasting.baselines import (
    BaselineArrays,
    GradientBoostingBaseline,
    baseline_arrays,
    evaluate_causal_gru,
    fit_causal_gru,
    rule_baseline_suite,
)
from stock_forecasting.config import ExperimentConfig
from stock_forecasting.data import BlockwisePermutationSampler, LazyFinancialWindowDataset
from stock_forecasting.evaluation_paths import validate_standalone_baseline_output
from stock_forecasting.run_paths import validate_wandb_directory
from stock_forecasting.training_paths import resolve_bar_store_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("rule", "gbdt", "gru"),
        default=("rule", "gbdt", "gru"),
    )
    parser.add_argument("--gru-epochs", type=int, default=30)
    parser.add_argument("--gru-patience", type=int, default=5)
    parser.add_argument("--unlock-test", action="store_true")
    parser.add_argument("--disable-wandb", action="store_true")
    return parser


def _flatten(payload: dict[str, Any], prefix: str = "") -> dict[str, float]:
    output: dict[str, float] = {}
    for key, value in payload.items():
        name = f"{prefix}/{key}" if prefix else key
        if isinstance(value, dict):
            output.update(_flatten(value, name))
        elif isinstance(value, (int, float)):
            output[name] = float(value)
    return output


def benchmark_baselines(
    config: ExperimentConfig,
    *,
    models: list[str],
    gru_epochs: int,
    gru_patience: int,
    unlock_test: bool,
) -> dict[str, Any]:
    bar_store = resolve_bar_store_path(config.data.bar_store_path)
    arrays: dict[str, BaselineArrays] = {}
    raw_counts: dict[str, int] = {}
    sample_counts: dict[str, int] = {}
    for position, split in enumerate(("train", "validation", "test")):
        dataset = LazyFinancialWindowDataset(
            bar_store,
            split=split,
            window_size=config.data.input_length,
            h_start=config.data.h_start,
        )
        raw_counts[split] = len(dataset)
        sampler = BlockwisePermutationSampler(
            len(dataset),
            fraction=config.data.train_fraction if split == "train" else 1.0,
            max_samples=config.validation.baseline_max_samples_per_split,
            seed=config.training.seed + position,
        )
        sample_counts[split] = len(sampler)
        if split != "test" or unlock_test:
            arrays[split] = baseline_arrays(
                dataset.record_at(index) for index in sampler
            )
    results: dict[str, Any] = {}
    if "rule" in models:
        rule_results = rule_baseline_suite(arrays["train"], arrays["validation"])
        results["rule"] = {"validation": rule_results["momentum_5d"]}
        if unlock_test:
            test_rules = rule_baseline_suite(arrays["train"], arrays["test"])
            results["rule"]["test"] = test_rules["momentum_5d"]
    if "gbdt" in models:
        model = GradientBoostingBaseline.fit(arrays["train"], seed=config.training.seed)
        results["gbdt"] = {"validation": model.evaluate(arrays["validation"])}
        if unlock_test:
            results["gbdt"]["test"] = model.evaluate(arrays["test"])
    if "gru" in models:
        model, validation = fit_causal_gru(
            arrays["train"],
            arrays["validation"],
            seed=config.training.seed,
            epochs=gru_epochs,
            patience=gru_patience,
        )
        results["gru"] = {"validation": validation}
        if unlock_test:
            results["gru"]["test"] = evaluate_causal_gru(model, arrays["test"])
    return {
        "schema_version": "1.0",
        "result_scope": "standalone_baseline_without_training_run_identity",
        "official_per_run_result": False,
        "run_id": None,
        "selection_split": "validation",
        "test_unlocked": unlock_test,
        "raw_sample_counts": raw_counts,
        "sample_counts": sample_counts,
        "models": results,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = ExperimentConfig.from_yaml(args.config)
    use_wandb = config.wandb.enabled and config.wandb.mode != "disabled" and not args.disable_wandb
    if use_wandb:
        validate_wandb_directory(config.wandb.directory)
    output = validate_standalone_baseline_output(
        args.output,
        validation_output_root=config.validation.output_root,
    )
    payload = benchmark_baselines(
        config,
        models=list(args.models),
        gru_epochs=args.gru_epochs,
        gru_patience=args.gru_patience,
        unlock_test=args.unlock_test,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if use_wandb:
        import wandb

        run = wandb.init(
            project=config.wandb.project,
            entity=config.wandb.entity or None,
            group="baselines",
            job_type="causal-baseline-benchmark",
            tags=[*config.wandb.tags, "baseline"],
            config={
                "experiment": config.as_dict(),
                "models": list(args.models),
                "gru_epochs": args.gru_epochs,
                "gru_patience": args.gru_patience,
                "test_unlocked": args.unlock_test,
            },
            dir=str(config.wandb.directory),
            mode=config.wandb.mode,
        )
        run.log(_flatten(payload))
        artifact = wandb.Artifact(f"baseline-results-{run.id}", type="evaluation")
        artifact.add_file(str(output))
        run.log_artifact(artifact, aliases=["validation" if not args.unlock_test else "final-test"])
        run.finish()
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
