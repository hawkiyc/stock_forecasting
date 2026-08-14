"""Quant-only conditional alpha model built on reusable OHLCV representations."""

from __future__ import annotations

from torch import Tensor, nn

from .forecast import GatedBenchmarkConditioner, MultiHorizonAlphaHead
from .outputs import QuantEncoderOutput, QuantForecastOutput
from .projector import CausalPerceiverResampler


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
            backbone_output.attention_mask,
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
        alpha_quantiles = self.alpha_head(conditioned).float()
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
