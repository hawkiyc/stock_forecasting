"""Causal time-series backbones.

The Kronos integration is deliberately lazy. Importing this package never imports
the upstream Kronos repository and never downloads model weights.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor, nn

from .exceptions import ModelCapabilityError, OptionalDependencyError
from .lora import inject_lora
from .outputs import TimeSeriesBackboneOutput


def freeze_module(module: nn.Module) -> nn.Module:
    """Freeze parameters and switch a module to deterministic evaluation mode."""

    module.requires_grad_(False)
    module.eval()
    return module


def _valid_mask(
    attention_mask: Tensor | None,
    *,
    batch_size: int,
    sequence_length: int,
    device: torch.device,
) -> Tensor:
    if attention_mask is None:
        return torch.ones(batch_size, sequence_length, dtype=torch.bool, device=device)
    if attention_mask.shape != (batch_size, sequence_length):
        raise ValueError(
            "attention_mask must have shape "
            f"({batch_size}, {sequence_length}), got {tuple(attention_mask.shape)}"
        )
    mask = attention_mask.to(device=device, dtype=torch.bool)
    if not mask.any(dim=1).all():
        raise ValueError("Every time-series sample must contain at least one valid bar")
    # Variable-length integrations assume padding is only on the right.
    if ((~mask[:, :-1]) & mask[:, 1:]).any():
        raise ValueError("time-series attention_mask must be right padded")
    return mask


class DeterministicTimeSeriesBackbone(nn.Module):
    """Download-free causal backbone for tests and remote CPU pipeline smoke checks.

    It uses fixed projections and a causal cumulative mean. It is not intended as a
    forecasting baseline, but it preserves the production backbone's tensor contract.
    """

    def __init__(
        self,
        input_dim: int = 5,
        hidden_size: int = 32,
        max_context: int = 128,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_size <= 0 or max_context <= 0:
            raise ValueError("input_dim, hidden_size, and max_context must be positive")
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.max_context = max_context

        indices = torch.arange(input_dim * hidden_size, dtype=torch.float32)
        projection = torch.sin(indices + 1.0).reshape(input_dim, hidden_size)
        column_norm = projection.square().sum(dim=0, keepdim=True).sqrt().clamp_min(1e-6)
        projection = projection / column_norm
        self.register_buffer("projection", projection, persistent=True)

    def forward(
        self,
        ohlcv: Tensor,
        attention_mask: Tensor | None = None,
        timestamps: Tensor | None = None,
    ) -> TimeSeriesBackboneOutput:
        del timestamps
        if ohlcv.ndim != 3:
            raise ValueError("ohlcv must have shape [batch, bars, features]")
        batch_size, sequence_length, feature_count = ohlcv.shape
        if feature_count != self.input_dim:
            raise ValueError(f"Expected {self.input_dim} OHLCV features, got {feature_count}")
        if sequence_length > self.max_context:
            raise ValueError(
                f"Input contains {sequence_length} bars, exceeding max_context={self.max_context}"
            )
        if not torch.isfinite(ohlcv).all():
            raise ValueError("ohlcv contains non-finite values")

        mask = _valid_mask(
            attention_mask,
            batch_size=batch_size,
            sequence_length=sequence_length,
            device=ohlcv.device,
        )
        valid = mask.unsqueeze(-1).to(dtype=ohlcv.dtype)
        values = ohlcv * valid
        counts = valid.cumsum(dim=1).clamp_min(1.0)
        causal_mean = values.cumsum(dim=1) / counts
        local_change = values - torch.cat([values[:, :1], values[:, :-1]], dim=1)
        features = values + 0.5 * causal_mean + 0.25 * local_change
        hidden = torch.tanh(features @ self.projection.to(dtype=features.dtype))
        hidden = hidden * valid
        return TimeSeriesBackboneOutput(last_hidden_state=hidden, attention_mask=mask)


class KronosBackbone(nn.Module):
    """Adapter exposing causal hidden states from the official Kronos implementation.

    The upstream Kronos repository must be installed or placed on ``PYTHONPATH`` so
    that its top-level ``model`` package is importable. The adapter uses the public
    ``KronosTokenizer.encode`` and ``Kronos.decode_s1`` methods. It intentionally
    does not fall back to forecast values when hidden states are unavailable.
    """

    DEFAULT_FEATURES = ("open", "high", "low", "close", "volume")

    def __init__(
        self,
        model: nn.Module,
        tokenizer: nn.Module,
        *,
        hidden_size: int | None = None,
        max_context: int = 128,
        clip_value: float = 5.0,
        feature_names: Sequence[str] = DEFAULT_FEATURES,
    ) -> None:
        super().__init__()
        if max_context <= 0 or clip_value <= 0:
            raise ValueError("max_context and clip_value must be positive")
        required_methods = ((tokenizer, "encode"), (model, "decode_s1"))
        missing = [
            name for owner, name in required_methods if not callable(getattr(owner, name, None))
        ]
        if missing:
            raise ModelCapabilityError(
                "Kronos hidden-state integration requires callable methods: " + ", ".join(missing)
            )
        inferred_hidden_size = hidden_size or getattr(model, "d_model", None)
        if not isinstance(inferred_hidden_size, int) or inferred_hidden_size <= 0:
            raise ModelCapabilityError(
                "Could not infer Kronos d_model; pass hidden_size explicitly"
            )
        if tuple(feature_names) != self.DEFAULT_FEATURES:
            raise ValueError(
                "KronosBackbone currently requires feature order: "
                + ", ".join(self.DEFAULT_FEATURES)
            )

        self.model = freeze_module(model)
        self.tokenizer = freeze_module(tokenizer)
        self.hidden_size = inferred_hidden_size
        self.max_context = max_context
        self.clip_value = clip_value
        self.feature_names = tuple(feature_names)
        self._predictor_trainable = False
        self.lora_module_names: tuple[str, ...] = ()

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str = "NeoQuasar/Kronos-small",
        tokenizer_name_or_path: str = "NeoQuasar/Kronos-Tokenizer-base",
        *,
        model_revision: str | None = None,
        tokenizer_revision: str | None = None,
        local_files_only: bool = True,
        model_module: str = "model",
        **kwargs: Any,
    ) -> KronosBackbone:
        """Load official Kronos classes, defaulting to cache-only operation.

        Set ``local_files_only=False`` explicitly only in a download/bootstrap job.
        """

        try:
            upstream = importlib.import_module(model_module)
        except ImportError as error:
            raise OptionalDependencyError(
                "The official Kronos repository is not importable. Clone/install "
                "shiyu-coder/Kronos and expose its top-level 'model' package."
            ) from error

        kronos_class = getattr(upstream, "Kronos", None)
        tokenizer_class = getattr(upstream, "KronosTokenizer", None)
        if kronos_class is None or tokenizer_class is None:
            raise ModelCapabilityError(
                f"Module {model_module!r} does not export Kronos and KronosTokenizer"
            )
        model_load_kwargs = {
            "local_files_only": local_files_only,
            **({"revision": model_revision} if model_revision is not None else {}),
        }
        tokenizer_load_kwargs = {
            "local_files_only": local_files_only,
            **({"revision": tokenizer_revision} if tokenizer_revision is not None else {}),
        }
        try:
            tokenizer = tokenizer_class.from_pretrained(
                tokenizer_name_or_path,
                **tokenizer_load_kwargs,
            )
            model = kronos_class.from_pretrained(model_name_or_path, **model_load_kwargs)
        except TypeError as error:
            if local_files_only or model_revision is not None or tokenizer_revision is not None:
                raise ModelCapabilityError(
                    "Installed Kronos loader does not accept the required offline or pinned "
                    "revision arguments. Instantiate KronosBackbone(model, tokenizer) from "
                    "verified local weights instead."
                ) from error
            tokenizer = tokenizer_class.from_pretrained(tokenizer_name_or_path)
            model = kronos_class.from_pretrained(model_name_or_path)
        return cls(model=model, tokenizer=tokenizer, **kwargs)

    def enable_lora(
        self,
        *,
        target_modules: Sequence[str],
        rank: int,
        alpha: float,
        dropout: float,
    ) -> tuple[str, ...]:
        """Enable low-rank predictor updates while keeping base weights frozen."""

        if self._predictor_trainable:
            raise RuntimeError("Kronos predictor LoRA is already enabled")
        names = inject_lora(
            self.model,
            target_modules=target_modules,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            allowed_prefixes=("transformer.",),
        )
        self._predictor_trainable = True
        self.lora_module_names = names
        return names

    @property
    def predictor_trainable(self) -> bool:
        return self._predictor_trainable

    def train(self, mode: bool = True) -> KronosBackbone:
        super().train(mode)
        self.tokenizer.eval()
        if self._predictor_trainable:
            self.model.train(mode)
        else:
            self.model.eval()
        return self

    def _prepare_features(self, samples: Tensor) -> Tensor:
        if samples.ndim not in {2, 3} or samples.shape[-1] != 5:
            raise ValueError("KronosBackbone expects OHLCV input with exactly five features")
        typical_price = samples[..., :4].mean(dim=-1, keepdim=True)
        amount = samples[..., 4:5] * typical_price
        values = torch.cat([samples, amount], dim=-1).float()
        mean = values.mean(dim=-2, keepdim=True)
        std = values.std(dim=-2, keepdim=True, unbiased=False).clamp_min(1e-5)
        normalized = ((values - mean) / std).clamp(-self.clip_value, self.clip_value)
        tokenizer_parameter = next(self.tokenizer.parameters(), None)
        tokenizer_dtype = (
            tokenizer_parameter.dtype if tokenizer_parameter is not None else normalized.dtype
        )
        return normalized.to(dtype=tokenizer_dtype)

    def _encode_batch(self, samples: Tensor, timestamps: Tensor | None) -> Tensor:
        if samples.ndim != 3:
            raise ValueError("Kronos samples must have shape [batch, bars, 5]")
        group_size, valid_length, _ = samples.shape
        normalized = self._prepare_features(samples)
        if timestamps is None:
            stamps = torch.zeros(
                group_size,
                valid_length,
                5,
                device=samples.device,
                dtype=torch.long,
            )
        else:
            if timestamps.shape != (group_size, valid_length, 5):
                raise ValueError("Kronos timestamps must have shape [batch, bars, 5]")
            stamps = timestamps.to(device=samples.device, dtype=torch.long)
        with torch.no_grad():
            token_ids = self.tokenizer.encode(normalized, half=True)
        if not isinstance(token_ids, (tuple, list)) or len(token_ids) != 2:
            raise ModelCapabilityError(
                "KronosTokenizer.encode(..., half=True) did not return two token streams"
            )
        decoded = self.model.decode_s1(token_ids[0], token_ids[1], stamps)
        if not isinstance(decoded, (tuple, list)) or len(decoded) < 2:
            raise ModelCapabilityError("Kronos.decode_s1 did not return (logits, context)")
        context = decoded[1]
        expected = (group_size, valid_length, self.hidden_size)
        if not isinstance(context, Tensor) or tuple(context.shape) != expected:
            actual = getattr(context, "shape", type(context).__name__)
            raise ModelCapabilityError(
                f"Unexpected Kronos context shape {actual}; expected {expected}"
            )
        return context

    def forward(
        self,
        ohlcv: Tensor,
        attention_mask: Tensor | None = None,
        timestamps: Tensor | None = None,
    ) -> TimeSeriesBackboneOutput:
        if ohlcv.ndim != 3:
            raise ValueError("ohlcv must have shape [batch, bars, features]")
        batch_size, sequence_length, feature_count = ohlcv.shape
        if feature_count != 5:
            raise ValueError("KronosBackbone expects [open, high, low, close, volume]")
        if sequence_length > self.max_context:
            raise ValueError(
                f"Input contains {sequence_length} bars, exceeding max_context={self.max_context}"
            )
        if not torch.isfinite(ohlcv).all():
            raise ValueError("ohlcv contains non-finite values")
        if timestamps is not None and timestamps.shape != (batch_size, sequence_length, 5):
            raise ValueError("timestamps must have shape [batch, bars, 5]")
        mask = _valid_mask(
            attention_mask,
            batch_size=batch_size,
            sequence_length=sequence_length,
            device=ohlcv.device,
        )

        self.tokenizer.eval()
        lengths = mask.sum(dim=1).detach().cpu().tolist()
        groups: dict[int, list[int]] = {}
        for index, valid_length in enumerate(lengths):
            groups.setdefault(int(valid_length), []).append(index)

        contexts: list[Tensor | None] = [None] * batch_size
        for valid_length in sorted(groups):
            group_indices = groups[valid_length]
            index_tensor = torch.tensor(group_indices, device=ohlcv.device, dtype=torch.long)
            group_ohlcv = ohlcv.index_select(0, index_tensor)[:, :valid_length]
            group_timestamps = (
                None
                if timestamps is None
                else timestamps.index_select(0, index_tensor)[:, :valid_length]
            )
            group_context = self._encode_batch(group_ohlcv, group_timestamps)
            padded_group = torch.nn.functional.pad(
                group_context,
                (0, 0, 0, sequence_length - valid_length),
            )
            for offset, original_index in enumerate(group_indices):
                contexts[original_index] = padded_group[offset]
        if any(context is None for context in contexts):
            raise RuntimeError("Kronos batch grouping did not encode every sample")
        hidden = torch.stack(
            [context for context in contexts if context is not None],
            dim=0,
        )
        return TimeSeriesBackboneOutput(last_hidden_state=hidden, attention_mask=mask)
