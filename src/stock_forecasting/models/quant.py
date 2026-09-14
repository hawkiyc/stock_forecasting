"""Quant-only conditional alpha model built on reusable OHLCV representations."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .forecast import GatedBenchmarkConditioner, MultiHorizonAlphaHead
from .outputs import QuantEncoderOutput, QuantForecastOutput
from .projector import CausalPerceiverResampler
from .scale_features import historical_scale_features


class QuantForecastModel(nn.Module):
    """Encode paired OHLCV histories and emit only continuous alpha quantiles."""

    def __init__(
        self,
        backbone: nn.Module,
        resampler: CausalPerceiverResampler,
        benchmark_conditioner: GatedBenchmarkConditioner,
        alpha_head: MultiHorizonAlphaHead,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.resampler = resampler
        self.benchmark_conditioner = benchmark_conditioner
        self.alpha_head = alpha_head

    def encode_ohlcv(
        self,
        ohlcv: Tensor,
        *,
        attention_mask: Tensor | None = None,
        timestamps: Tensor | None = None,
    ) -> QuantEncoderOutput:
        """Expose reusable representations for a possible future multimodal model."""

        backbone_output = self.backbone(
            ohlcv,
            attention_mask=attention_mask,
            timestamps=timestamps,
        )
        resampler_dtype = self.resampler.context_projection.weight.dtype
        latent_tokens = self.resampler(
            backbone_output.last_hidden_state.to(dtype=resampler_dtype),
            None if attention_mask is None else backbone_output.attention_mask,
        )
        return QuantEncoderOutput(
            last_hidden_state=backbone_output.last_hidden_state,
            attention_mask=backbone_output.attention_mask,
            latent_tokens=latent_tokens,
        )

    def forward(
        self,
        asset_ohlcv: Tensor,
        benchmark_ohlcv: Tensor,
        *,
        asset_attention_mask: Tensor | None = None,
        benchmark_attention_mask: Tensor | None = None,
        asset_timestamps: Tensor | None = None,
        benchmark_timestamps: Tensor | None = None,
        target_alpha: Tensor | None = None,
    ) -> QuantForecastOutput:
        scales = None
        branch = self.alpha_head.numeric_branch
        if branch is not None and branch.scale_projection is not None:
            if (asset_attention_mask is None) != (benchmark_attention_mask is None):
                raise ValueError("Scale branch requires aligned asset and benchmark masks")
            if asset_attention_mask is not None and not torch.equal(
                asset_attention_mask, benchmark_attention_mask
            ):
                raise ValueError("Scale branch requires aligned asset and benchmark masks")
            with torch.autocast(device_type=asset_ohlcv.device.type, enabled=False):
                scales = historical_scale_features(
                    asset_ohlcv, benchmark_ohlcv, mask=asset_attention_mask,
                )
        can_fuse_pair = (
            asset_ohlcv.shape == benchmark_ohlcv.shape
            and (asset_attention_mask is None) == (benchmark_attention_mask is None)
            and (asset_timestamps is None) == (benchmark_timestamps is None)
        )
        if can_fuse_pair:
            asset_batch_size = asset_ohlcv.shape[0]
            if asset_attention_mask is None:
                combined_mask = None
            else:
                assert benchmark_attention_mask is not None
                combined_mask = torch.cat(
                    [asset_attention_mask, benchmark_attention_mask],
                    dim=0,
                )
            if asset_timestamps is None:
                combined_timestamps = None
            else:
                assert benchmark_timestamps is not None
                combined_timestamps = torch.cat(
                    [asset_timestamps, benchmark_timestamps],
                    dim=0,
                )
            combined = self.encode_ohlcv(
                torch.cat([asset_ohlcv, benchmark_ohlcv], dim=0),
                attention_mask=combined_mask,
                timestamps=combined_timestamps,
            )
            asset_encoded = QuantEncoderOutput(
                last_hidden_state=combined.last_hidden_state[:asset_batch_size],
                attention_mask=combined.attention_mask[:asset_batch_size],
                latent_tokens=combined.latent_tokens[:asset_batch_size],
            )
            benchmark_encoded = QuantEncoderOutput(
                last_hidden_state=combined.last_hidden_state[asset_batch_size:],
                attention_mask=combined.attention_mask[asset_batch_size:],
                latent_tokens=combined.latent_tokens[asset_batch_size:],
            )
        else:
            asset_encoded = self.encode_ohlcv(
                asset_ohlcv,
                attention_mask=asset_attention_mask,
                timestamps=asset_timestamps,
            )
            benchmark_encoded = self.encode_ohlcv(
                benchmark_ohlcv,
                attention_mask=benchmark_attention_mask,
                timestamps=benchmark_timestamps,
            )
        conditioned, gate = self.benchmark_conditioner(
            asset_encoded.latent_tokens,
            benchmark_encoded.latent_tokens,
        )
        alpha_quantiles = self.alpha_head(
            conditioned, scale_features=scales,
            benchmark_tokens=benchmark_encoded.latent_tokens,
        ).float()
        pinball_loss = (
            None
            if target_alpha is None
            else self.alpha_head.pinball_loss(alpha_quantiles, target_alpha)
        )
        return QuantForecastOutput(
            loss=pinball_loss,
            pinball_loss=pinball_loss,
            alpha_quantiles=alpha_quantiles,
            asset_last_hidden_state=asset_encoded.last_hidden_state,
            benchmark_last_hidden_state=benchmark_encoded.last_hidden_state,
            asset_attention_mask=asset_encoded.attention_mask,
            benchmark_attention_mask=benchmark_encoded.attention_mask,
            asset_latent_tokens=asset_encoded.latent_tokens,
            benchmark_latent_tokens=benchmark_encoded.latent_tokens,
            conditioned_latent_tokens=conditioned,
            conditioning_gate=gate,
        )
