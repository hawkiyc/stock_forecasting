"""Trainable causal-context resampling modules."""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn


class _PerceiverLayer(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        num_heads: int,
        ff_multiplier: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(latent_dim)
        self.context_norm = nn.LayerNorm(latent_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ff_norm = nn.LayerNorm(latent_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * ff_multiplier),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * ff_multiplier, latent_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, latents: Tensor, context: Tensor, context_mask: Tensor) -> Tensor:
        query = self.query_norm(latents)
        key_value = self.context_norm(context)
        attended, _ = self.cross_attention(
            query=query,
            key=key_value,
            value=key_value,
            key_padding_mask=~context_mask,
            need_weights=False,
        )
        latents = latents + self.dropout(attended)
        return cast(Tensor, latents + self.dropout(self.feed_forward(self.ff_norm(latents))))


class _GatedProjectionMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_multiplier: int,
        dropout: float,
    ) -> None:
        super().__init__()
        intermediate_dim = input_dim * hidden_multiplier
        self.input_norm = nn.LayerNorm(input_dim)
        self.hidden_projection = nn.Linear(input_dim, intermediate_dim * 2)
        self.output_projection = nn.Linear(intermediate_dim, output_dim)
        self.output_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: Tensor) -> Tensor:
        value, gate = self.hidden_projection(self.input_norm(inputs)).chunk(2, dim=-1)
        hidden = torch.nn.functional.gelu(value) * torch.sigmoid(gate)
        return cast(Tensor, self.output_norm(self.output_projection(self.dropout(hidden))))


class CausalPerceiverResampler(nn.Module):
    """Compress a fully observed historical context into fixed soft tokens.

    Cross-attention is unrestricted within the supplied context because every input
    bar must precede the prediction cutoff. Causality across cutoffs is guaranteed by
    the causal backbone and the caller's historical-only window construction.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        num_soft_tokens: int = 16,
        latent_dim: int | None = None,
        depth: int = 2,
        num_heads: int = 4,
        ff_multiplier: int = 4,
        projector_hidden_multiplier: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        latent_dim = latent_dim or input_dim
        positive_values = (input_dim, output_dim, num_soft_tokens, latent_dim, depth)
        if any(value <= 0 for value in (*positive_values, projector_hidden_multiplier)):
            raise ValueError("Projection dimensions, token count, and depth must be positive")
        if latent_dim % num_heads != 0:
            raise ValueError("latent_dim must be divisible by num_heads")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_soft_tokens = num_soft_tokens
        self.latent_dim = latent_dim
        self.projector_hidden_multiplier = projector_hidden_multiplier
        self.context_projection = nn.Linear(input_dim, latent_dim)
        self.latents = nn.Parameter(torch.empty(num_soft_tokens, latent_dim))
        self.layers = nn.ModuleList(
            [_PerceiverLayer(latent_dim, num_heads, ff_multiplier, dropout) for _ in range(depth)]
        )
        self.output_projection = _GatedProjectionMLP(
            input_dim=latent_dim,
            output_dim=output_dim,
            hidden_multiplier=projector_hidden_multiplier,
            dropout=dropout,
        )
        nn.init.normal_(self.latents, std=latent_dim**-0.5)

    def encode_latents(self, hidden_states: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        """Return reusable causal latents before the output projection."""

        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape [batch, bars, hidden]")
        batch_size, sequence_length, hidden_size = hidden_states.shape
        if hidden_size != self.input_dim:
            raise ValueError(f"Expected hidden size {self.input_dim}, got {hidden_size}")
        if attention_mask is None:
            mask = torch.ones(
                batch_size,
                sequence_length,
                dtype=torch.bool,
                device=hidden_states.device,
            )
        else:
            if attention_mask.shape != (batch_size, sequence_length):
                raise ValueError("attention_mask does not match hidden_states")
            mask = attention_mask.to(device=hidden_states.device, dtype=torch.bool)
        if not mask.any(dim=1).all():
            raise ValueError("Every sample must provide at least one valid backbone token")

        context = self.context_projection(hidden_states)
        latents = self.latents.unsqueeze(0).expand(batch_size, -1, -1)
        for layer in self.layers:
            latents = layer(latents, context, mask)
        return cast(Tensor, latents)

    def project_latents(self, latents: Tensor) -> Tensor:
        if latents.ndim != 3 or latents.shape[-1] != self.latent_dim:
            raise ValueError(f"latents must have shape [batch, tokens, {self.latent_dim}]")
        return cast(Tensor, self.output_projection(latents))

    def forward(self, hidden_states: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        return self.project_latents(self.encode_latents(hidden_states, attention_mask))


class CausalTokenProjector(CausalPerceiverResampler):
    """Compatibility name emphasizing historical-only soft-token projection."""
