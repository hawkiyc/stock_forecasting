"""Mapping-compatible outputs for the quant-only model."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from torch import Tensor

MODEL_OUTPUT_SCHEMA_VERSION = "4.0"


class OutputMapping(Mapping[str, Any]):
    """Provide the small mapping interface used by the training loop."""

    def __getitem__(self, key: str) -> Any:
        if key not in vars(self):
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        return iter(vars(self))

    def __len__(self) -> int:
        return len(vars(self))

    def to_dict(self) -> dict[str, Any]:
        return {key: self[key] for key in self}


@dataclass(frozen=True)
class TimeSeriesBackboneOutput(OutputMapping):
    """Causal per-bar representations produced by a time-series backbone."""

    last_hidden_state: Tensor
    attention_mask: Tensor


@dataclass(frozen=True)
class QuantEncoderOutput(OutputMapping):
    """Reusable numerical representations before the fixed quant head."""

    last_hidden_state: Tensor
    attention_mask: Tensor
    latent_tokens: Tensor


@dataclass(frozen=True)
class QuantForecastOutput(OutputMapping):
    """Conditional multi-horizon alpha distribution and reusable numeric latents."""

    loss: Tensor | None
    pinball_loss: Tensor | None
    alpha_quantiles: Tensor
    asset_last_hidden_state: Tensor
    benchmark_last_hidden_state: Tensor
    asset_attention_mask: Tensor
    benchmark_attention_mask: Tensor
    asset_latent_tokens: Tensor
    benchmark_latent_tokens: Tensor
    conditioned_latent_tokens: Tensor
    conditioning_gate: Tensor
