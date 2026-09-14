"""Dynamic benchmark conditioning and multi-horizon alpha quantile head."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from stock_forecasting.data.horizons import (
    DEFAULT_ALPHA_HORIZONS,
    validate_alpha_horizons,
)
from stock_forecasting.models.scale_features import NumericalResidualBranch


class GatedBenchmarkConditioner(nn.Module):
    """Condition asset latents on historical benchmark latents with a learned gate."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        num_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be positive and divisible by num_heads")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.hidden_dim = hidden_dim
        self.asset_norm = nn.LayerNorm(hidden_dim)
        self.benchmark_norm = nn.LayerNorm(hidden_dim)
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        gate_linear = cast(nn.Linear, self.gate[1])
        nn.init.zeros_(gate_linear.weight)
        nn.init.constant_(gate_linear.bias, -2.0)

    def forward(self, asset_tokens: Tensor, benchmark_tokens: Tensor) -> tuple[Tensor, Tensor]:
        expected_rank = asset_tokens.ndim == benchmark_tokens.ndim == 3
        if (
            not expected_rank
            or asset_tokens.shape[0] != benchmark_tokens.shape[0]
            or asset_tokens.shape[-1] != self.hidden_dim
            or benchmark_tokens.shape[-1] != self.hidden_dim
        ):
            raise ValueError("Conditioner inputs must be paired [batch, tokens, hidden] tensors")
        attended, _ = self.cross_attention(
            query=self.asset_norm(asset_tokens),
            key=self.benchmark_norm(benchmark_tokens),
            value=self.benchmark_norm(benchmark_tokens),
            need_weights=False,
        )
        gate = self.gate(torch.cat([asset_tokens, attended], dim=-1))
        conditioned = self.output_norm(asset_tokens + gate * self.dropout(attended))
        return cast(Tensor, conditioned), cast(Tensor, gate)


class MultiHorizonAlphaHead(nn.Module):
    """Predict ordered q10/q50/q90 benchmark-relative log-return quantiles."""

    def __init__(
        self,
        input_dim: int,
        *,
        horizons: Sequence[int] = DEFAULT_ALPHA_HORIZONS,
        hidden_dim: int | None = None,
        quantiles: Sequence[float] = (0.1, 0.5, 0.9),
        robust_scales: Sequence[float] | None = None,
        dropout: float = 0.0,
        feature_mode: str = "baseline",
        fp32_head: bool = False,
    ) -> None:
        super().__init__()
        ordered_horizons = validate_alpha_horizons(horizons)
        ordered_quantiles = tuple(float(value) for value in quantiles)
        if ordered_quantiles != (0.1, 0.5, 0.9):
            raise ValueError("MultiHorizonAlphaHead requires quantiles (0.1, 0.5, 0.9)")
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        hidden_dim = hidden_dim or max(32, input_dim // 2)
        scales = tuple(float(value) for value in (robust_scales or [1.0] * len(ordered_horizons)))
        if len(scales) != len(ordered_horizons) or any(
            not math.isfinite(value) or value <= 0.0 for value in scales
        ):
            raise ValueError("robust_scales must contain one finite positive value per horizon")

        self.input_dim = input_dim
        self.horizons = ordered_horizons
        self.quantiles = ordered_quantiles
        self.horizon_embeddings = nn.Embedding(len(ordered_horizons), input_dim)
        self.trunk = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.quantile_parameters = nn.Linear(hidden_dim, 3)
        self.fp32_head = fp32_head or feature_mode != "baseline"
        self.numeric_branch = (
            None if feature_mode == "baseline"
            else NumericalResidualBranch(hidden_dim, input_dim, feature_mode)
        )
        self.register_buffer(
            "robust_scales",
            torch.tensor(scales, dtype=torch.float32),
            persistent=False,
        )

    def forward(
        self, conditioned_tokens: Tensor, *,
        scale_features: Tensor | None = None, benchmark_tokens: Tensor | None = None,
    ) -> Tensor:
        if self.fp32_head:
            with torch.autocast(device_type=conditioned_tokens.device.type, enabled=False):
                return self._forward(conditioned_tokens.float(), scale_features, benchmark_tokens)
        return self._forward(conditioned_tokens, scale_features, benchmark_tokens)

    def _forward(
        self, conditioned_tokens: Tensor, scale_features: Tensor | None,
        benchmark_tokens: Tensor | None,
    ) -> Tensor:
        if conditioned_tokens.ndim != 3 or conditioned_tokens.shape[-1] != self.input_dim:
            raise ValueError(
                f"conditioned_tokens must have shape [batch, tokens, {self.input_dim}]"
            )
        pooled = conditioned_tokens.mean(dim=1)
        horizon_ids = torch.arange(
            len(self.horizons),
            device=conditioned_tokens.device,
            dtype=torch.long,
        )
        horizon_inputs = pooled[:, None, :] + self.horizon_embeddings(horizon_ids)[None, :, :]
        hidden = self.trunk(horizon_inputs)
        raw = self.quantile_parameters(hidden)
        if self.numeric_branch is not None:
            raw = raw + self.numeric_branch(hidden, scale_features, benchmark_tokens)
        median = raw[..., 1]
        lower = median - F.softplus(raw[..., 0])
        upper = median + F.softplus(raw[..., 2])
        return torch.stack([lower, median, upper], dim=-1)

    def pinball_loss(self, predictions: Tensor, target: Tensor) -> Tensor:
        predictions = predictions.float()
        expected = (predictions.shape[0], len(self.horizons), len(self.quantiles))
        if tuple(predictions.shape) != expected:
            raise ValueError(f"alpha_quantiles must have shape {expected}")
        target = target.to(device=predictions.device, dtype=predictions.dtype)
        if tuple(target.shape) != (predictions.shape[0], len(self.horizons)):
            raise ValueError("target_alpha must have shape [batch, horizons]")
        valid = torch.isfinite(target)
        safe_target = torch.where(valid, target, torch.zeros_like(target))
        scales = self.robust_scales.to(device=predictions.device, dtype=predictions.dtype)
        errors = (safe_target[..., None] - predictions) / scales[None, :, None]
        levels = predictions.new_tensor(self.quantiles)
        losses = torch.maximum(levels * errors, (levels - 1.0) * errors)
        weights = valid[..., None].expand_as(losses).to(dtype=losses.dtype)
        return (losses * weights).sum() / weights.sum().clamp_min(1.0)
