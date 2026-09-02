"""Tensor and trainability contracts for the numerical model."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from stock_forecasting.checkpointing import (
    CHECKPOINT_ARTIFACT_SCHEMA_VERSION,
    _load_trainable_model_state,
    _restore_runtime_robust_scales,
    _validate_loaded_model_contract,
    save_training_completion_result,
    trainable_state_dict,
)
from stock_forecasting.cli.prefetch_models import prefetch_repositories
from stock_forecasting.config import ExperimentConfig
from stock_forecasting.data import FinancialBatchCollator, FinancialWindowDataset
from stock_forecasting.factory import (
    PINNED_KRONOS_SOURCE_REVISION,
    _promote_trainable_parameters_to_fp32,
    build_model_bundle,
    verify_kronos_source_revision,
)
from stock_forecasting.models.backbones import (
    DeterministicTimeSeriesBackbone,
    KronosBackbone,
)
from stock_forecasting.models.lora import LoRALinear, inject_lora, lora_parameter_names
from stock_forecasting.models.outputs import MODEL_OUTPUT_SCHEMA_VERSION
from stock_forecasting.run_contract import training_resume_contract_digest
from stock_forecasting.training import (
    EarlyStoppingState,
    epoch_evaluation_steps,
    evaluate_loader,
    plan_dataloader_workers,
)

ROOT = Path(__file__).resolve().parents[1]


def test_bundled_kronos_source_matches_pinned_revision(tmp_path: Path) -> None:
    source_root = ROOT / "src/stock_forecasting/_vendor/kronos"

    assert (
        verify_kronos_source_revision(source_root, PINNED_KRONOS_SOURCE_REVISION)
        == PINNED_KRONOS_SOURCE_REVISION
    )

    tampered_root = tmp_path / "kronos"
    shutil.copytree(source_root, tampered_root)
    module_path = tampered_root / "model/module.py"
    module_path.write_text(
        module_path.read_text(encoding="utf-8") + "# tampered\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="source digest mismatch"):
        verify_kronos_source_revision(tampered_root, PINNED_KRONOS_SOURCE_REVISION)


def test_dataloader_worker_plan_is_cpu_and_memory_bounded() -> None:
    gib = 1024**3
    full = plan_dataloader_workers(
        8,
        source="test",
        visible_cpu_count=16,
        available_memory_bytes=64 * gib,
    )
    memory_limited = plan_dataloader_workers(
        8,
        source="test",
        visible_cpu_count=16,
        available_memory_bytes=8 * gib,
    )
    cpu_limited = plan_dataloader_workers(
        8,
        source="test",
        visible_cpu_count=4,
        available_memory_bytes=64 * gib,
    )
    disabled = plan_dataloader_workers(
        0,
        source="test",
        visible_cpu_count=16,
        available_memory_bytes=64 * gib,
    )

    assert full.effective_workers == 8
    assert full.prefetch_factor == 2
    assert full.symbol_cache_size_per_worker == 8
    assert memory_limited.effective_workers == 2
    assert cpu_limited.effective_workers == 2
    assert disabled.effective_workers == 0
    assert disabled.prefetch_factor is None
    assert (
        full.effective_workers * full.active_persistent_pools * full.symbol_cache_size_per_worker
        <= 128
    )


def test_epoch_relative_validation_schedule_has_exactly_five_even_checkpoints() -> None:
    assert epoch_evaluation_steps(0, 13, 5) == (3, 6, 8, 11, 13)
    assert epoch_evaluation_steps(1, 13, 5) == (16, 19, 21, 24, 26)

    with pytest.raises(ValueError, match="at least one optimizer step"):
        epoch_evaluation_steps(0, 4, 5)


def test_stage1_early_stopping_cannot_trigger_before_second_epoch() -> None:
    state = EarlyStoppingState()
    values = [1.0, 1.1, 1.2, 1.3, 1.4]

    for value in values:
        assert (
            state.observe(
                value,
                mode="min",
                min_delta=0.0,
                patience=5,
                epoch_number=1,
                start_epoch=2,
                enabled=True,
            )
            is False
        )

    assert state.stale_evaluations == 4
    assert state.observe(
        1.5,
        mode="min",
        min_delta=0.0,
        patience=5,
        epoch_number=2,
        start_epoch=2,
        enabled=True,
    )
    assert state.triggered is True
    assert EarlyStoppingState.from_dict(state.as_dict()) == state


def test_validation_loss_improvement_resets_early_stopping_patience() -> None:
    state = EarlyStoppingState()
    for value in (1.0, 1.1, 1.2):
        state.observe(
            value,
            mode="min",
            min_delta=0.0,
            patience=2,
            epoch_number=1,
            start_epoch=1,
            enabled=False,
        )

    assert state.stale_evaluations == 2
    assert not state.observe(
        0.9,
        mode="min",
        min_delta=0.0,
        patience=2,
        epoch_number=2,
        start_epoch=1,
        enabled=True,
    )
    assert state.stale_evaluations == 0
    assert state.best_value == pytest.approx(0.9)


def test_training_completion_result_is_compact_atomic_and_immutable(
    tmp_path: Path,
) -> None:
    config = ExperimentConfig.from_yaml(ROOT / "configs/local_mock.yaml")
    run_directory = tmp_path / "completion-contract-run"
    run_directory.mkdir()
    contract_digest = training_resume_contract_digest(config)
    (run_directory / "run-manifest.json").write_text(
        json.dumps(
            {
                "schema_version": CHECKPOINT_ARTIFACT_SCHEMA_VERSION,
                "run_id": run_directory.name,
                "run_key": run_directory.name,
                "training_resume_contract_sha256": contract_digest,
            }
        ),
        encoding="utf-8",
    )
    model = nn.Module()
    model.register_parameter("weight", nn.Parameter(torch.ones(1)))
    alpha_head = nn.Module()
    alpha_head.register_buffer(
        "robust_scales",
        torch.ones(len(config.data.alpha_horizons)),
    )
    model.add_module("alpha_head", alpha_head)

    result_directory = save_training_completion_result(
        model=model,
        config=config,
        run_directory=run_directory,
        global_step=10,
        completed_epochs=1,
        processed_train_samples=40,
        planned_train_samples=40,
        validation_evaluations=1,
        stop_reason="epochs_completed",
        early_stopping_state=EarlyStoppingState().as_dict(),
        metrics={"primary_5d/selection_score": 0.5},
    )

    assert {path.name for path in result_directory.iterdir()} == {
        "adapter.safetensors",
        "resolved-config.yaml",
        "training-result.json",
    }
    assert not (result_directory / "optimizer.pt").exists()
    assert not (result_directory / "scheduler.pt").exists()
    payload = json.loads((result_directory / "training-result.json").read_text())
    assert payload["kind"] == "training-completion-result"
    assert payload["stop_reason"] == "epochs_completed"
    assert payload["training_resume_contract_sha256"] == contract_digest

    with pytest.raises(FileExistsError, match="already exists"):
        save_training_completion_result(
            model=model,
            config=config,
            run_directory=run_directory,
            global_step=10,
            completed_epochs=1,
            processed_train_samples=40,
            planned_train_samples=40,
            validation_evaluations=1,
            stop_reason="epochs_completed",
            early_stopping_state=EarlyStoppingState().as_dict(),
            metrics={"primary_5d/selection_score": 0.5},
        )


def test_prefetch_binds_every_repository_to_its_exact_revision(tmp_path: Path) -> None:
    revisions = {
        "owner/model": "a" * 40,
        "owner/tokenizer": "b" * 40,
    }
    calls: list[dict[str, object]] = []

    def downloader(**kwargs: object) -> str:
        calls.append(kwargs)
        snapshot = tmp_path / str(kwargs["revision"])
        snapshot.mkdir(exist_ok=True)
        return str(snapshot)

    resolved = prefetch_repositories(
        revisions,
        cache_directory=tmp_path / "cache",
        verify_only=True,
        downloader=downloader,
    )

    assert set(resolved) == set(revisions)
    assert [(call["repo_id"], call["revision"]) for call in calls] == list(revisions.items())
    assert all(call["local_files_only"] is True for call in calls)


@pytest.mark.parametrize("h_start", [1, 2, 3])
def test_quant_model_preserves_head_and_reusable_encoder_shapes(h_start: int) -> None:
    config = ExperimentConfig.from_yaml(ROOT / "configs/local_mock.yaml")
    config.data.h_start = h_start
    bundle = build_model_bundle(config, torch.device("cpu"))
    asset_series = torch.rand(2, config.data.input_length, 5)
    benchmark_series = torch.rand(2, config.data.input_length, 5)
    mask = torch.ones(2, config.data.input_length, dtype=torch.bool)
    timestamps = torch.zeros(2, config.data.input_length, 5, dtype=torch.long)

    output = bundle.model(
        asset_series,
        benchmark_series,
        asset_attention_mask=mask,
        benchmark_attention_mask=mask,
        asset_timestamps=timestamps,
        benchmark_timestamps=timestamps,
        target_alpha=torch.zeros(2, len(config.data.alpha_horizons)),
    )

    assert set(output) == {
        "loss",
        "pinball_loss",
        "alpha_quantiles",
        "asset_last_hidden_state",
        "benchmark_last_hidden_state",
        "asset_attention_mask",
        "benchmark_attention_mask",
        "asset_latent_tokens",
        "benchmark_latent_tokens",
        "conditioned_latent_tokens",
        "conditioning_gate",
    }
    assert output.alpha_quantiles.shape == (2, 15 - h_start, 3)
    assert output.asset_last_hidden_state.shape == (2, config.data.input_length, 64)
    assert output.benchmark_last_hidden_state.shape == (2, config.data.input_length, 64)
    assert output.asset_attention_mask.shape == (2, config.data.input_length)
    assert output.benchmark_attention_mask.shape == (2, config.data.input_length)
    assert output.asset_latent_tokens.shape == (2, 8, 64)
    assert output.benchmark_latent_tokens.shape == (2, 8, 64)
    assert output.conditioned_latent_tokens.shape == (2, 8, 64)
    assert output.conditioning_gate.shape == (2, 8, 64)
    assert torch.all(output.alpha_quantiles[..., 0] <= output.alpha_quantiles[..., 1])
    assert torch.all(output.alpha_quantiles[..., 1] <= output.alpha_quantiles[..., 2])
    assert output.loss is not None
    assert output.pinball_loss is output.loss

    encoded = bundle.model.encode_ohlcv(
        asset_series,
        attention_mask=mask,
        timestamps=timestamps,
    )
    assert encoded.last_hidden_state.shape == output.asset_last_hidden_state.shape
    assert encoded.latent_tokens.shape == output.asset_latent_tokens.shape


def test_mock_checkpoint_contains_only_reusable_numeric_and_alpha_modules() -> None:
    config = ExperimentConfig.from_yaml(ROOT / "configs/local_mock.yaml")
    bundle = build_model_bundle(config, torch.device("cpu"))
    state = trainable_state_dict(bundle.model)

    assert state
    assert all(
        name.startswith(("resampler.", "benchmark_conditioner.", "alpha_head."))
        for name in state
    )
    assert not any(name.startswith("backbone.") for name in state)


def test_checkpoint_restore_requires_the_exact_trainable_parameter_union() -> None:
    config = ExperimentConfig.from_yaml(ROOT / "configs/local_mock.yaml")
    model = build_model_bundle(config, torch.device("cpu")).model
    state = trainable_state_dict(model)
    missing_key = next(iter(state))
    incomplete = {name: tensor for name, tensor in state.items() if name != missing_key}

    with pytest.raises(ValueError, match="Missing trainable checkpoint keys"):
        _load_trainable_model_state(model, incomplete)

    with pytest.raises(ValueError, match="Unexpected checkpoint keys"):
        _load_trainable_model_state(
            model,
            {**state, "legacy_text_head.weight": torch.ones(1)},
        )


@pytest.mark.parametrize("h_start", [1, 2, 3])
def test_checkpoint_runtime_robust_scales_restore_exact_horizon_contract(
    h_start: int,
) -> None:
    config = ExperimentConfig.from_yaml(ROOT / "configs/local_mock.yaml")
    config.data.h_start = h_start
    model = build_model_bundle(config, torch.device("cpu")).model
    expected = [0.01 * (index + 1) for index in range(len(config.data.alpha_horizons))]

    _restore_runtime_robust_scales(
        model,
        {"runtime_robust_scales": expected},
        require_match=False,
    )

    torch.testing.assert_close(
        model.alpha_head.robust_scales,
        torch.tensor(expected, dtype=torch.float32),
    )
    with pytest.raises(ValueError, match="differ from train calibration"):
        _restore_runtime_robust_scales(
            model,
            {"runtime_robust_scales": [1.0] * len(expected)},
            require_match=True,
        )
    with pytest.raises(ValueError, match="do not match model horizons"):
        _restore_runtime_robust_scales(
            model,
            {
                "runtime_robust_scales": [1.0]
                * (13 if len(expected) == 12 else 12)
            },
            require_match=False,
        )


def test_loaded_checkpoint_must_match_model_ids_architecture_and_stage(
    tmp_path: Path,
) -> None:
    config = ExperimentConfig.from_yaml(ROOT / "configs/local_mock.yaml")
    config.save_resolved(tmp_path / "resolved-config.yaml")
    state = {
        "model_output_schema_version": MODEL_OUTPUT_SCHEMA_VERSION,
        "training_resume_contract_sha256": training_resume_contract_digest(config),
        "model_architecture_sha256": config.model_architecture_digest(),
        "training_stage": config.training.stage,
        "time_series_model_id": config.model.time_series_model_id,
        "time_series_tokenizer_id": config.model.time_series_tokenizer_id,
        "time_series_model_revision": config.model.time_series_model_revision,
        "time_series_tokenizer_revision": config.model.time_series_tokenizer_revision,
        "kronos_source_revision": config.model.kronos_source_revision,
    }

    _validate_loaded_model_contract(tmp_path, state, config)

    with pytest.raises(ValueError, match="implementation or dataset contract"):
        _validate_loaded_model_contract(
            tmp_path,
            {**state, "training_resume_contract_sha256": "0" * 64},
            config,
        )

    with pytest.raises(ValueError, match="training stage"):
        _validate_loaded_model_contract(
            tmp_path,
            {**state, "training_stage": "stage2"},
            config,
        )

    with pytest.raises(ValueError, match="predates the conditional alpha"):
        _validate_loaded_model_contract(
            tmp_path,
            {**state, "model_output_schema_version": "3.1"},
            config,
        )


def test_trainable_master_parameters_remain_fp32() -> None:
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
    model[0].requires_grad_(False)
    model.to(dtype=torch.bfloat16)

    _promote_trainable_parameters_to_fp32(model)

    assert model[0].weight.dtype == torch.bfloat16
    assert model[1].weight.dtype == torch.float32
    assert model[1].bias.dtype == torch.float32


def test_deterministic_backbone_is_causal() -> None:
    backbone = DeterministicTimeSeriesBackbone(hidden_size=16, max_context=12)
    original = torch.rand(1, 12, 5)
    changed = original.clone()
    changed[:, 8:] = 1000.0

    first = backbone(original).last_hidden_state
    second = backbone(changed).last_hidden_state

    torch.testing.assert_close(first[:, :8], second[:, :8])


class _BatchTokenizer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))
        self.encoded_shapes: list[tuple[int, ...]] = []

    def encode(
        self,
        values: torch.Tensor,
        *,
        half: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert half
        self.encoded_shapes.append(tuple(values.shape))
        return values[..., :1], values[..., 1:2]


class _BatchKronos(nn.Module):
    d_model = 3

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))

    def decode_s1(
        self,
        token_a: torch.Tensor,
        token_b: torch.Tensor,
        stamps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        stamp_feature = stamps[..., :1].to(dtype=token_a.dtype)
        context = torch.cat([token_a, token_b, stamp_feature], dim=-1) * self.scale
        return token_a, context


def test_kronos_loader_passes_independent_pinned_weight_revisions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str, dict[str, object]]] = []

    class LoaderTokenizer(_BatchTokenizer):
        @classmethod
        def from_pretrained(cls, repo_id: str, **kwargs: object) -> LoaderTokenizer:
            calls.append(("tokenizer", repo_id, kwargs))
            return cls()

    class LoaderModel(_BatchKronos):
        @classmethod
        def from_pretrained(cls, repo_id: str, **kwargs: object) -> LoaderModel:
            calls.append(("model", repo_id, kwargs))
            return cls()

    monkeypatch.setattr(
        "stock_forecasting.models.backbones.importlib.import_module",
        lambda _name: SimpleNamespace(
            Kronos=LoaderModel,
            KronosTokenizer=LoaderTokenizer,
        ),
    )

    KronosBackbone.from_pretrained(
        "owner/model",
        "owner/tokenizer",
        model_revision="a" * 40,
        tokenizer_revision="b" * 40,
        local_files_only=True,
    )

    assert calls == [
        (
            "tokenizer",
            "owner/tokenizer",
            {"local_files_only": True, "revision": "b" * 40},
        ),
        (
            "model",
            "owner/model",
            {"local_files_only": True, "revision": "a" * 40},
        ),
    ]


def test_kronos_backbone_batches_equal_length_samples_together() -> None:
    tokenizer = _BatchTokenizer()
    backbone = KronosBackbone(
        _BatchKronos(),
        tokenizer,
        max_context=4,
    )
    series = torch.rand(3, 4, 5)
    mask = torch.tensor(
        [
            [True, True, True, True],
            [True, True, False, False],
            [True, True, True, True],
        ]
    )
    timestamps = torch.zeros(3, 4, 5, dtype=torch.long)

    output = backbone(series, attention_mask=mask, timestamps=timestamps)

    assert tokenizer.encoded_shapes == [(1, 2, 6), (2, 4, 6)]
    assert output.last_hidden_state.shape == (3, 4, 3)
    assert torch.count_nonzero(output.last_hidden_state[1, 2:]) == 0
    assert torch.equal(output.attention_mask, mask)


def test_evaluation_restores_training_mode(
    window_records: list[dict[str, object]],
) -> None:
    config = ExperimentConfig.from_yaml(ROOT / "configs/local_mock.yaml")
    dataset = FinancialWindowDataset(window_records, split="validation")
    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=FinancialBatchCollator(),
    )
    bundle = build_model_bundle(config, torch.device("cpu"))
    bundle.model.train()

    metrics = evaluate_loader(bundle, loader, config, torch.device("cpu"))

    assert bundle.model.training
    assert metrics["samples"] == len(dataset)
    assert "aggregate" in metrics
    assert set(metrics["per_horizon"]) == {
        f"{horizon}d" for horizon in config.data.alpha_horizons
    }
    assert "selection_score" in metrics["primary_5d"]
    assert metrics["primary_5d"]["selection_score"] == pytest.approx(
        metrics["aggregate"]["selection_score"]
    )
    assert "postprocess_signal_distribution" in metrics
    assert "macro_f1" not in repr(metrics).lower()
    assert set(metrics["subgroups"]) == {"asset_type", "market", "provider", "year"}


class _Predictor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer = nn.ModuleDict(
            {
                name: nn.Linear(4, 4)
                for name in ("q_proj", "k_proj", "v_proj", "out_proj", "w1", "w2", "w3")
            }
        )
        self.outside_q_proj = nn.Linear(4, 4)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = inputs
        for layer in self.transformer.values():
            hidden = layer(hidden)
        return self.outside_q_proj(hidden)


def test_lora_is_scoped_to_exact_kronos_predictor_targets() -> None:
    predictor = _Predictor().requires_grad_(False)
    targets = ("q_proj", "k_proj", "v_proj", "out_proj", "w1", "w2", "w3")
    matched = inject_lora(
        predictor,
        target_modules=targets,
        rank=2,
        alpha=4.0,
        dropout=0.0,
    )

    assert set(matched) == {f"transformer.{name}" for name in targets}
    assert all(isinstance(predictor.transformer[name], LoRALinear) for name in targets)
    assert isinstance(predictor.outside_q_proj, nn.Linear)
    assert not predictor.outside_q_proj.weight.requires_grad
    assert lora_parameter_names(predictor)
    assert all(
        parameter.requires_grad == (".lora_a." in name or ".lora_b." in name)
        for name, parameter in predictor.named_parameters()
    )

    predictor(torch.rand(3, 4)).sum().backward()
    assert all(
        parameter.grad is not None
        for name, parameter in predictor.named_parameters()
        if name in lora_parameter_names(predictor)
    )
