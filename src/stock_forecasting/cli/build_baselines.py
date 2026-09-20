"""Build and persist matching full-data baselines on a dedicated GPU Pod."""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import torch

from stock_forecasting.baseline_build import build_baselines
from stock_forecasting.config import ExperimentConfig
from stock_forecasting.data.manifest import atomic_write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    if not torch.cuda.is_available() or os.environ.get("RUNPOD_ROLE") != "gpu-baseline":
        raise ValueError("Full baseline building requires the dedicated CUDA Pod workflow")
    config = ExperimentConfig.from_yaml(args.config)
    root = Path(os.environ.get("NETWORK_VOLUME_ROOT", "/runpod-volume"))
    lifecycle = root / "lifecycle/stage1/baseline.json"

    def publish(state, error=None):
        atomic_write_json(
            lifecycle,
            {
                "schema_version": 1,
                "kind": "stage1-baseline",
                "state": state,
                "pod_id": os.environ["RUNPOD_POD_ID"],
                "wandb_run_id": os.environ["WANDB_RUN_ID"],
                "launch_id": os.environ.get("RUNPOD_LAUNCH_ID", "manual-baseline"),
                "generated_at": datetime.now(UTC).isoformat(),
                "baseline_completed": state == "ready",
                "error": error,
            },
        )

    publish("preparing")
    try:
        payload = build_baselines(config)
        publish("finalizing" if os.environ.get("RUNPOD_TMUX_LOG_FILE") else "ready")
        print(json.dumps({"state": "complete", "baseline_id": payload["identity"]["baseline_id"]}))
    except BaseException as error:
        publish(
            "finalizing" if os.environ.get("RUNPOD_TMUX_LOG_FILE") else "failed",
            f"{type(error).__name__}: {error}",
        )
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
