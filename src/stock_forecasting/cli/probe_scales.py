"""Probe historical scale information in an existing frozen RunPod checkpoint."""

# Bilingual report text intentionally preserves Chinese punctuation.
# ruff: noqa: RUF001

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import sklearn
import torch

from stock_forecasting.checkpointing import CHECKPOINT_TRANSACTION, load_checkpoint
from stock_forecasting.cli.evaluate import resolve_checkpoint
from stock_forecasting.config import ExperimentConfig
from stock_forecasting.data.dataset import LazyFinancialWindowDataset
from stock_forecasting.data.manifest import provenance_summary, sha256_file
from stock_forecasting.factory import build_model_bundle
from stock_forecasting.preflight import run_preflight
from stock_forecasting.representation_scale_probe import (
    READOUT_DESCRIPTIONS,
    TARGET_NAMES,
    HistoricalScaleDataset,
    ScaleProbeSettings,
    extract_probe_split,
    fit_scale_probes,
)
from stock_forecasting.run_paths import canonical_network_volume_root, validate_training_output_root
from stock_forecasting.training import set_global_seed

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    defaults = ScaleProbeSettings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="Canonical retained checkpoint or run directory with a validated best pointer.",
    )
    parser.add_argument("--train-samples", type=int, default=defaults.train_samples)
    parser.add_argument("--validation-samples", type=int, default=defaults.validation_samples)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--num-workers", type=int, default=defaults.num_workers)
    parser.add_argument("--ridge-alpha", type=float, default=defaults.ridge_alpha)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    return parser


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def resolve_probe_checkpoint(path: Path, *, saved_model_root: Path) -> Path:
    """Reject pending transactions before shared selection can reconcile them."""

    source = path.expanduser().resolve(strict=False)
    root = saved_model_root.expanduser().resolve(strict=False)
    run = source.parent if (source / "adapter.safetensors").is_file() else source
    if run.parent != root:
        raise ValueError("Probe checkpoints must belong to a canonical saved-model run")
    transaction = run / CHECKPOINT_TRANSACTION
    if transaction.exists() or transaction.is_symlink():
        raise ValueError(
            "Checkpoint has a pending selection transaction; recover it through the original "
            "training workflow before running this read-only diagnostic"
        )
    return resolve_checkpoint(source, saved_model_root=root)


def render_summary(report: dict[str, Any]) -> str:
    """Keep the complete Chinese section before the complete English section."""

    def number(value: float | None) -> str:
        return "N/A" if value is None else f"{value:.6g}"

    rows = []
    for readout, payload in report["metrics"]["readouts"].items():
        for target in TARGET_NAMES:
            score = payload["validation"][target]
            control = payload["shuffled_label_validation"][target]
            rows.append(
                f"| {readout} | {target} | {number(payload['train'][target]['r2'])} | "
                f"{number(score['r2'])} | {number(score['mae'])} | "
                f"{number(score['rmse'])} | {number(score['pearson_r'])} | "
                f"{number(score['mse_skill_vs_train_mean'])} | {number(control['r2'])} |"
            )
    table_rows = "\n".join(rows)
    checkpoint = report["checkpoint"]["path"]
    train = report["splits"]["train"]
    validation = report["splits"]["validation"]
    return (
        "# 歷史尺度表徵診斷 / Historical scale representation probe\n\n"
        "## 中文\n\n"
        f"Checkpoint：`{checkpoint}`。模型權重固定，只擬合獨立 ridge 探針。\n\n"
        f"Train：{train['samples']} 筆，{train['cutoff_min']} 至 {train['cutoff_max']}。\n\n"
        f"Validation：{validation['samples']} 筆，{validation['cutoff_min']} 至 "
        f"{validation['cutoff_max']}。日期為實際抽樣 cutoff，不是重新定義的切分邊界。\n\n"
        "使用 checkpoint 原有 train / validation 切分；未建立 test loader、未計算未來 alpha "
        "標籤。探針從完整 train split 抽樣，不保證等於 Stage 1 訓練曾用的 5% 子集。\n\n"
        "前六個目標為個股、benchmark、兩者差值的最近 20 / 60 個交易日對數報酬標準差；"
        "最後兩個為完整輸入視窗收盤價的標準差 / 平均值。使用 as-of 調整、母體標準差"
        "（ddof=0），不年化；0.01 表示日對數報酬標準差 1%。這不是未來持有期 alpha。\n\n"
        "縮放器、ridge 與打亂標籤對照都只在 train 擬合；ridge alpha 固定，不用 validation "
        "選參。MAE / RMSE 越低越好；R² 越接近 1 越好，負值表示劣於 validation 自身平均值。"
        "Skill = 1 − MSE / train-mean-baseline MSE；正值才代表勝過 train 平均值對照。"
        "常數目標的 R²、常數向量的相關係數與零分母 skill 記為 N/A，不強填 0。\n\n"
        "| 讀取位置 | 歷史目標 | Train R² | Val R² | Val MAE | Val RMSE | Val Pearson r | "
        "Val skill | 打亂標籤 Val R² |\n"
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
        f"{table_rows}\n\n"
        "解讀限制：低分只表示這種 pooling + 線性讀取未成功，不證明資訊不存在；"
        "各層維度不同，不能把分數差直接等同於資訊損失。重疊視窗與共用 benchmark "
        "不是獨立樣本；本報告不提供顯著性或自動架構裁決。Train 高、Val 低應先排查"
        "探針過擬合或分布差異。Val 必須同時優於平均值與打亂標籤對照，才支持"
        "可讀出歷史尺度；仍不代表能預測未來 alpha 或改善交易績效。\n\n"
        "詳見 report.json 的各市場指標、表徵維度、樣本數與 SHA-256；"
        "samples.jsonl 與 validation_predictions.npz 按同一列順序保存樣本與預測。\n\n"
        "## English\n\n"
        f"Checkpoint: `{checkpoint}`. Model weights are frozen; only separate ridge probes fit.\n\n"
        f"Train: {train['samples']} samples, {train['cutoff_min']} to {train['cutoff_max']}.\n\n"
        f"Validation: {validation['samples']} samples, {validation['cutoff_min']} to "
        f"{validation['cutoff_max']}. These are sampled cutoffs, not new split boundaries.\n\n"
        "The checkpoint's train/validation membership is preserved. No test loader or future "
        "alpha labels are constructed. Probe training samples the full train split, not "
        "necessarily the 5% subset previously seen during Stage 1 training.\n\n"
        "Targets are asset, benchmark and asset-minus-benchmark daily log-return standard "
        "deviations over the last 20/60 trading returns, plus each stream's close-price "
        "standard deviation / mean over the full context. As-of adjusted inputs, population "
        "std (ddof=0), no annualization; 0.01 means 1% daily log-return std. These are not "
        "future holding-period alpha labels.\n\n"
        "All scalers, ridge weights and shuffled-label controls fit only on train. Ridge alpha "
        "is fixed without validation tuning. Lower MAE/RMSE is better. R² approaches 1 for "
        "better fits; negative R² underperforms the validation mean. Skill is 1 minus MSE "
        "divided by train-mean-baseline MSE; positive skill beats that deployable baseline. "
        "Constant-target R², constant-vector correlations and zero-denominator skill are N/A.\n\n"
        "| Readout | Historical target | Train R² | Val R² | Val MAE | Val RMSE | Val Pearson r | "
        "Val skill | Shuffled Val R² |\n"
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
        f"{table_rows}\n\n"
        "Low scores demonstrate failure of this pooling + linear readout, not absence of "
        "information. Readout dimensions differ, so score gaps do not establish information "
        "loss. Overlapping windows/shared benchmarks are not independent samples; no "
        "significance claim or automatic architecture decision is made. High train and low "
        "validation scores can indicate probe overfitting or distribution shift. Validation "
        "should beat both controls to support scale decodability; even success does not "
        "establish future-alpha predictability or better trading performance.\n\n"
        "See report.json for per-market metrics, feature dimensions, sample counts and "
        "SHA-256 provenance. samples.jsonl and validation_predictions.npz share row order.\n"
    )


def run_probe(checkpoint_path: Path, settings: ScaleProbeSettings) -> Path:
    """Run on an existing GPU Pod, without mutating training/evaluation lifecycle state."""

    if not os.environ.get("RUNPOD_POD_ID") or not torch.cuda.is_available():
        raise RuntimeError("Checkpoint scale probing must run inside an existing CUDA RunPod Pod")
    if os.environ.get("RUNPOD_GPU_WORKFLOW_LEASE_HELD") != "1":
        raise RuntimeError(
            "Use bash scripts/runpod_workflow.sh probe-scales to verify the mount and GPU lease"
        )
    volume = canonical_network_volume_root()
    checkpoint = resolve_probe_checkpoint(checkpoint_path, saved_model_root=volume / "savedModel")
    config = ExperimentConfig.from_yaml(checkpoint / "resolved-config.yaml")
    validate_training_output_root(config.training.output_root)
    if config.model.time_series_backend != "kronos":
        raise ValueError("The production scale probe requires a Kronos checkpoint")
    run_preflight(config, require_data=True, enforce_runtime_limit=False).require_success()
    diagnostic_root = volume / "diagnostics" / "representation-scales"
    if diagnostic_root.resolve() != diagnostic_root:
        raise ValueError("Diagnostic output must not traverse symlinked directories")
    output = (
        diagnostic_root
        / checkpoint.parent.name
        / checkpoint.name
        / (f"probe-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}")
    )
    if output.resolve() != output:
        raise ValueError("Diagnostic output must not traverse symlinked directories")
    output.mkdir(parents=True, exist_ok=False)
    log_handler = logging.FileHandler(output / "probe.log", encoding="utf-8")
    log_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(log_handler)
    LOGGER.info("Independent diagnostic output: %s", output)
    _write_json(output / "status.json", {"state": "running", "checkpoint": str(checkpoint)})
    try:
        set_global_seed(settings.seed)
        device = torch.device("cuda")
        bundle = build_model_bundle(config, device)
        state = load_checkpoint(checkpoint, bundle.model, config=config)
        # Load the trainable-state union before disabling gradients on this private instance.
        bundle.model.requires_grad_(False)
        bundle.model.eval()
        extracted = {}
        for split in ("train", "validation"):
            dataset = HistoricalScaleDataset(
                LazyFinancialWindowDataset(
                    config.data.bar_store_path,
                    split=split,
                    window_size=config.data.input_length,
                    h_start=config.data.h_start,
                )
            )
            extracted[split] = extract_probe_split(
                bundle.model,
                dataset,
                device=device,
                settings=settings,
                mixed_precision=config.training.mixed_precision,
            )
        result = fit_scale_probes(extracted["train"], extracted["validation"], settings)
        np.savez(output / "probes.npz", **result.fitted_arrays)
        np.savez(output / "validation_predictions.npz", **result.validation_predictions)
        with (output / "samples.jsonl").open("x", encoding="utf-8") as stream:
            for split, values in extracted.items():
                for row_index, sample in enumerate(values.samples):
                    stream.write(
                        json.dumps(
                            {"split": split, "row_index": row_index, **sample},
                            ensure_ascii=False,
                            allow_nan=False,
                        )
                        + "\n"
                    )
        provenance = provenance_summary(config.data.resolved_manifest_path)
        report = {
            "schema_version": "1.0",
            "kind": "historical-scale-representation-probe",
            "created_at": datetime.now(UTC).isoformat(),
            "checkpoint": {
                "path": str(checkpoint),
                "global_step": state.get("global_step"),
                "epoch": state.get("epoch"),
                "training_stage": config.training.stage,
                "model_architecture_sha256": config.model_architecture_digest(),
                "training_resume_contract_sha256": state.get("training_resume_contract_sha256"),
                "files_sha256": {
                    name: sha256_file(checkpoint / name)
                    for name in (
                        "adapter.safetensors",
                        "resolved-config.yaml",
                        "trainer-state.json",
                    )
                },
                "run_manifest_sha256": sha256_file(checkpoint.parent / "run-manifest.json"),
                "model_id": config.model.time_series_model_id,
                "model_revision": config.model.time_series_model_revision,
                "tokenizer_id": config.model.time_series_tokenizer_id,
                "tokenizer_revision": config.model.time_series_tokenizer_revision,
                "kronos_source_revision": config.model.kronos_source_revision,
            },
            "data_provenance": {
                key: provenance[key]
                for key in (
                    "manifest_path",
                    "manifest_sha256",
                    "dataset_profile",
                    "date_range",
                    "split_counts",
                    "split_audit",
                    "storage_preparation_spec_sha256",
                )
            },
            "settings": settings.as_dict(),
            "sampling": {
                "method": "uniform_cutoff_without_replacement_sorted_for_io",
                "scope": "full_checkpoint_train_and_validation_splits_not_stage_fraction_subset",
                "checkpoint_train_fraction": config.data.train_fraction,
                "checkpoint_max_samples": config.data.max_samples,
                "train_seed": settings.seed,
                "validation_seed": settings.seed + 1,
                "shuffled_label_seed": settings.seed + 2,
            },
            "target_contract": {
                "names": TARGET_NAMES,
                "std_ddof": 0,
                "annualized": False,
                "input_length": config.data.input_length,
                "minimum_valid_bars": 61,
                "prices": "asof_adjusted_input_closes_before_kronos_normalization",
                "daily_return": "diff(log(close))",
                "relative_return": "asset_daily_log_return - benchmark_daily_log_return",
                "close_cv": "population_std(close) / mean(close) over full input context",
                "future_alpha_labels_used": False,
            },
            "readouts": READOUT_DESCRIPTIONS,
            "splits": {name: values.summary() for name, values in extracted.items()},
            "metrics": result.metrics,
            "runtime": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "numpy": np.__version__,
                "sklearn": sklearn.__version__,
                "gpu": torch.cuda.get_device_name(device),
                "mixed_precision": config.training.mixed_precision,
            },
            "diagnostic_source_sha256": {
                name: sha256_file(Path(__file__).resolve().parents[1] / name)
                for name in ("representation_scale_probe.py", "cli/probe_scales.py")
            },
            "artifacts_sha256": {
                name: sha256_file(output / name)
                for name in ("samples.jsonl", "probes.npz", "validation_predictions.npz")
            },
        }
        _write_json(output / "report.json", report)
        (output / "summary.md").write_text(render_summary(report), encoding="utf-8")
        _write_json(output / "status.json", {"state": "complete", "checkpoint": str(checkpoint)})
        LOGGER.info("Scale probe complete: %s", output)
        return output
    except BaseException as error:
        _write_json(output / "status.json", {"state": "failed", "error_type": type(error).__name__})
        LOGGER.exception("Scale probe failed; partial artifacts are not completed results")
        raise
    finally:
        logging.getLogger().removeHandler(log_handler)
        log_handler.close()


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = ScaleProbeSettings(
        train_samples=args.train_samples,
        validation_samples=args.validation_samples,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        ridge_alpha=args.ridge_alpha,
        seed=args.seed,
    )
    output = run_probe(args.checkpoint, settings)
    print(json.dumps({"state": "complete", "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
