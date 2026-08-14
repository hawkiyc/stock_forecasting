"""Train the quant-only Kronos LoRA model for Stage 1 or Stage 2."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

from fin_ts_multimodal.checkpointing import validate_checkpoint_selection
from fin_ts_multimodal.config import ExperimentConfig
from fin_ts_multimodal.run_contract import validate_training_resume_contract
from fin_ts_multimodal.run_paths import (
    canonical_network_volume_root,
    validate_run_environment_ids,
    validate_training_resume_path,
)
from fin_ts_multimodal.training import train, write_training_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    source_config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
    config = ExperimentConfig.from_yaml(args.config)
    if config.training.resume_checkpoint is not None:
        raise ValueError(
            "training.resume_checkpoint must not be set in YAML; use the RunPod resume contract"
        )
    resume_checkpoint = os.environ.get("RESUME_CHECKPOINT", "").strip()
    if resume_checkpoint:
        wandb_run_id = os.environ.get("WANDB_RUN_ID")
        runpod_run_key = os.environ.get("RUNPOD_RUN_KEY")
        if not wandb_run_id or not runpod_run_key:
            raise ValueError("Training resume requires WANDB_RUN_ID and RUNPOD_RUN_KEY")
        run_id = validate_run_environment_ids(
            wandb_run_id,
            runpod_run_key,
        )
        assert run_id is not None
        volume_root = canonical_network_volume_root()
        saved_model_root = Path(
            os.environ.get("SAVED_MODEL_ROOT", str(volume_root / "savedModel"))
        ).resolve(strict=False)
        checkpoint_path = validate_training_resume_path(
            resume_checkpoint,
            saved_model_root=saved_model_root,
            run_id=run_id,
        )
        try:
            checkpoint_path.relative_to(volume_root)
        except ValueError as error:
            raise ValueError("RESUME_CHECKPOINT must be stored on NETWORK_VOLUME_ROOT") from error
        validate_checkpoint_selection(
            checkpoint_path.parent,
            retained_checkpoint=checkpoint_path.name,
            require_latest_step=True,
        )
        validate_training_resume_contract(
            config,
            run_directory=checkpoint_path.parent,
            checkpoint_directory=checkpoint_path,
        )
        config.training.resume_checkpoint = checkpoint_path
    previous_source_config_sha256 = os.environ.get("RUNPOD_SOURCE_CONFIG_SHA256")
    os.environ["RUNPOD_SOURCE_CONFIG_SHA256"] = source_config_sha256
    try:
        result = train(config)
    finally:
        if previous_source_config_sha256 is None:
            os.environ.pop("RUNPOD_SOURCE_CONFIG_SHA256", None)
        else:
            os.environ["RUNPOD_SOURCE_CONFIG_SHA256"] = previous_source_config_sha256
    write_training_result(result)


if __name__ == "__main__":
    main()
