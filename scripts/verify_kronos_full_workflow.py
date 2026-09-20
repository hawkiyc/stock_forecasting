#!/usr/bin/env python3
"""Authorized cloud-only, offline four-window training integration smoke test."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from stock_forecasting.baseline_storage import lazy_dataset, loader_options
from stock_forecasting.config import ExperimentConfig
from stock_forecasting.data.dataset import FinancialBatchCollator
from stock_forecasting.factory import build_model_bundle
from stock_forecasting.models.scale_features import (
    fit_scale_feature_statistics,
    historical_scale_features,
)
from stock_forecasting.training import (
    _move_batch_to_device,
    build_evaluation_loader,
    forward_batch,
    iter_device_batches,
)


def main():
    if not os.environ.get("RUNPOD_POD_ID") or not torch.cuda.is_available():
        raise ValueError("This smoke test may only run on the authorized CUDA Pod")
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        raise ValueError("Offline model loading is mandatory")
    torch.set_num_threads(1)
    torch.manual_seed(43)
    config = ExperimentConfig.from_yaml("configs/stage2_kronos_base_lora.yaml")
    source = lazy_dataset(config, "train")
    loader = DataLoader(
        Subset(source, list(range(4))),
        batch_size=4,
        collate_fn=FinancialBatchCollator(),
        **loader_options(2),
    )
    batch = next(iter(loader))
    statistics = fit_scale_feature_statistics(
        historical_scale_features(
            batch["asset_series"], batch["benchmark_series"], extended=True
        ).numpy(),
        {"split": "train", "purpose": "smoke-only-not-production-calibration"},
    )
    device = torch.device("cuda")
    bundle = build_model_bundle(
        config,
        device,
        robust_scales=[0.03] * len(config.data.alpha_horizons),
        scale_feature_statistics=statistics,
    )
    batch = _move_batch_to_device(batch, device)
    parameters = [p for p in bundle.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=1e-5)
    bundle.model.train()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = forward_batch(bundle, batch, config, device)
    assert output.loss is not None and torch.isfinite(output.loss)
    output.loss.backward()
    gradients = {
        name: float(p.grad.detach().abs().sum())
        for name, p in bundle.model.named_parameters()
        if p.requires_grad and p.grad is not None
    }
    assert gradients and all(torch.isfinite(p.grad).all() for p in parameters if p.grad is not None)
    assert any("lora" in name and value > 0 for name, value in gradients.items())
    assert any("market_embedding" in name and value > 0 for name, value in gradients.items())
    assert any("scale_gate" in name and value > 0 for name, value in gradients.items())
    optimizer.step()
    summary = {
        "samples": 4,
        "split": "train",
        "lora_rank": config.model.lora.rank,
        "trainable_parameters": sum(p.numel() for p in parameters),
        "loss": float(output.loss.detach()),
        "pinball_loss": float(output.pinball_loss.detach()),
        "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
        "gradient_checks": "finite_and_nonzero",
        "offline": True,
        "production_weights_saved": False,
    }
    del optimizer, output, batch, loader
    bundle.model.zero_grad(set_to_none=True)
    bundle.model.eval()
    evaluation = build_evaluation_loader(config, bundle=bundle, device=device, split="test")
    started, profiled = time.monotonic(), 0
    with torch.inference_mode():
        for batch in iter_device_batches(evaluation, device):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = forward_batch(bundle, batch, config, device)
            assert torch.isfinite(output.alpha_quantiles).all()
            profiled += len(batch["symbols"])
            if profiled >= 2048:
                break
    torch.cuda.synchronize()
    summary["evaluation_profile"] = {
        "full_split_size": len(evaluation.dataset),
        "profiled_samples": profiled,
        "batch_size": evaluation.batch_size,
        "workers": evaluation.num_workers,
        "prefetch_factor": evaluation.prefetch_factor,
        "pin_memory": evaluation.pin_memory,
        "drop_last": evaluation.drop_last,
        "seconds": time.monotonic() - started,
        "samples_per_second": profiled / (time.monotonic() - started),
        "purpose": "bounded throughput profile, not full holdout scoring",
        "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
    }
    root = (
        Path(os.environ["NETWORK_VOLUME_ROOT"])
        / "diagnostics/full-workflow"
        / os.environ["WANDB_RUN_ID"]
    )
    root.mkdir(parents=True, exist_ok=True)
    (root / "kronos-smoke.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
