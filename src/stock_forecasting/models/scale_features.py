"""Past-only scale statistics and a checkpoint-bound numerical residual branch."""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor, nn

SCALE_FEATURE_VERSION = "historical-scales-log-iqr-v1"
EXTENDED_FEATURE_VERSION = "historical-market-scales-v2"
SCALE_FEATURE_NAMES = (
    *(
        f"{stream}_log_return_std_{window}d"
        for window in (20, 60)
        for stream in ("asset", "benchmark", "relative")
    ),
    "asset_close_cv_context",
    "benchmark_close_cv_context",
)
SCALE_LOG_EPSILON = 1e-8
SCALE_IQR_FLOOR = 1e-3
SCALE_STANDARDIZED_CLIP = 10.0
EXTENDED_FEATURE_NAMES = (
    *SCALE_FEATURE_NAMES,
    *(
        f"{stream}_return_mean_{window}d"
        for window in (5, 20)
        for stream in ("asset", "benchmark", "relative")
    ),
    "correlation_60d",
    "beta_60d",
    "relative_downside_std_20d",
    "asset_volume_ratio_20d",
    "benchmark_volume_ratio_20d",
    "benchmark_drawdown_60d",
)


def historical_scale_features(
    asset: Tensor,
    benchmark: Tensor,
    *,
    mask: Tensor | None = None,
    extended: bool = False,
) -> Tensor:
    """Vectorize eight scale statistics and optionally twelve market descriptors.

    Both inputs are already point-in-time adjusted, before Kronos normalization.
    Native tensor kernels process the whole batch; no per-sample Python loop or
    extra model/worker copy is needed.
    """

    if asset.ndim != 3 or asset.shape != benchmark.shape or asset.shape[-1] != 5:
        raise ValueError("Scale inputs must be aligned [batch, time, 5] OHLCV tensors")
    if mask is None:
        mask = torch.ones(asset.shape[:2], dtype=torch.bool, device=asset.device)
    if mask.shape != asset.shape[:2] or mask.dtype != torch.bool:
        raise ValueError("Scale mask must be an aligned boolean tensor")
    lengths = mask.sum(dim=1)
    if not bool((lengths >= 61).all()):
        raise ValueError("Historical scales require at least 61 valid observed bars")
    closes = torch.stack((asset[..., 3].float(), benchmark[..., 3].float()), dim=-1)
    valid_closes = closes.masked_fill(~mask[..., None], 1.0)
    if not bool((torch.isfinite(valid_closes) & (valid_closes > 0)).all()):
        raise ValueError("Historical adjusted closes must be finite and positive")
    positions = torch.arange(asset.shape[1], device=asset.device).expand_as(mask)
    order = positions.masked_fill(~mask, asset.shape[1]).sort(dim=1).values
    ordered = valid_closes.gather(
        1,
        order.clamp_max(asset.shape[1] - 1)[..., None].expand(-1, -1, 2),
    )
    returns = ordered[:, 1:].log() - ordered[:, :-1].log()
    returns = torch.cat((returns, returns[..., :1] - returns[..., 1:2]), dim=-1)
    return_positions = positions[:, :-1]
    features = []
    for window in (20, 60):
        selected = (return_positions >= lengths[:, None] - 1 - window) & (
            return_positions < lengths[:, None] - 1
        )
        mean = returns.masked_fill(~selected[..., None], 0).sum(dim=1) / window
        variance = (returns - mean[:, None]).square().masked_fill(~selected[..., None], 0).sum(
            dim=1
        ) / window
        features.append(variance.clamp_min(0).sqrt())
    selected_closes = positions < lengths[:, None]
    count = lengths[:, None].float()
    mean_close = ordered.masked_fill(~selected_closes[..., None], 0).sum(dim=1) / count
    variance_close = (ordered - mean_close[:, None]).square().masked_fill(
        ~selected_closes[..., None], 0
    ).sum(dim=1) / count
    features.append(variance_close.clamp_min(0).sqrt() / mean_close)
    basic = torch.cat(features, dim=-1)
    if not extended:
        return basic
    extra = []
    for window in (5, 20):
        selected = (return_positions >= lengths[:, None] - 1 - window) & (
            return_positions < lengths[:, None] - 1
        )
        extra.append(returns.masked_fill(~selected[..., None], 0).sum(1) / window)
    selected60 = (return_positions >= lengths[:, None] - 61) & (
        return_positions < lengths[:, None] - 1
    )
    mean60 = returns.masked_fill(~selected60[..., None], 0).sum(1) / 60
    centered = returns - mean60[:, None]
    covariance = (centered[..., 0] * centered[..., 1]).masked_fill(~selected60, 0).sum(1) / 60
    correlation = covariance / (basic[:, 3] * basic[:, 4]).clamp_min(1e-8)
    beta = covariance / basic[:, 4].square().clamp_min(1e-8)
    selected20 = (return_positions >= lengths[:, None] - 21) & (
        return_positions < lengths[:, None] - 1
    )
    downside = (
        returns[..., 2].clamp_max(0).square().masked_fill(~selected20, 0).sum(1) / 20
    ).sqrt()
    volume = torch.stack((asset[..., 4].float(), benchmark[..., 4].float()), -1)
    volume = volume.masked_fill(~mask[..., None], 0)
    if not bool((torch.isfinite(volume) & (volume >= 0)).all()):
        raise ValueError("Observed historical volume must be finite and nonnegative")
    volume = volume.gather(1, order.clamp_max(asset.shape[1] - 1)[..., None].expand(-1, -1, 2))
    selected_volume = (positions >= lengths[:, None] - 20) & (positions < lengths[:, None])
    mean_volume = volume.masked_fill(~selected_volume[..., None], 0).sum(1) / 20
    last_volume = volume.gather(1, (lengths - 1)[:, None, None].expand(-1, 1, 2)).squeeze(1)
    volume_ratio = ((last_volume.clamp_min(0) + 1) / (mean_volume.clamp_min(0) + 1)).log()
    recent = (positions >= lengths[:, None] - 60) & (positions < lengths[:, None])
    peak = ordered[..., 1].masked_fill(~recent, 0).amax(1)
    last = ordered[..., 1].gather(1, (lengths - 1)[:, None]).squeeze(1)
    drawdown = (last / peak.clamp_min(1e-8)).log()
    extra.extend(
        [correlation[:, None], beta[:, None], downside[:, None], volume_ratio, drawdown[:, None]]
    )
    return torch.cat((basic, *extra), -1)


def _transform(values):
    """Log positive scale features; signed asinh for returns and market descriptors."""
    if isinstance(values, Tensor):
        basic = (values[..., :8] + SCALE_LOG_EPSILON).log()
        return basic if values.shape[-1] == 8 else torch.cat((basic, values[..., 8:].asinh()), -1)
    basic = np.log(values[..., :8] + SCALE_LOG_EPSILON)
    return (
        basic if values.shape[-1] == 8 else np.concatenate((basic, np.arcsinh(values[..., 8:])), -1)
    )


def fit_scale_feature_statistics(values: np.ndarray, identity: dict[str, Any]) -> dict[str, Any]:
    """Fit per-feature log medians/IQRs on a bounded, train-only calibration sample."""

    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] not in (8, 20) or len(values) < 4:
        raise ValueError("Scale calibration requires at least four rows of 8 or 20 statistics")
    if (
        not np.isfinite(values).all()
        or (values[:, :8] < 0).any()
        or identity.get("split") != "train"
    ):
        raise ValueError("Scale calibration requires finite nonnegative train-only statistics")
    logged = _transform(values)
    q25, q75 = np.quantile(logged, [0.25, 0.75], axis=0)
    body = {
        "version": SCALE_FEATURE_VERSION if values.shape[1] == 8 else EXTENDED_FEATURE_VERSION,
        "feature_names": list(
            SCALE_FEATURE_NAMES if values.shape[1] == 8 else EXTENDED_FEATURE_NAMES
        ),
        "center": np.median(logged, axis=0).tolist(),
        "scale": np.maximum(q75 - q25, SCALE_IQR_FLOOR).tolist(),
        "identity": identity,
    }
    return {**body, "sha256": _statistics_digest(body)}


def _statistics_digest(body: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def validate_scale_feature_statistics(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != {
        "version",
        "feature_names",
        "center",
        "scale",
        "identity",
        "sha256",
    }:
        raise ValueError("Checkpoint scale-feature statistics are missing or malformed")
    names = {
        SCALE_FEATURE_VERSION: SCALE_FEATURE_NAMES,
        EXTENDED_FEATURE_VERSION: EXTENDED_FEATURE_NAMES,
    }.get(payload["version"])
    if names is None or payload["feature_names"] != list(names):
        raise ValueError("Checkpoint scale-feature schema differs from the numerical branch")
    for key in ("center", "scale"):
        values = np.asarray(payload[key], dtype=np.float64)
        if values.shape != (len(names),) or not np.isfinite(values).all():
            raise ValueError("Scale-feature statistics must match the finite feature vector")
        if key == "scale" and (values < SCALE_IQR_FLOOR).any():
            raise ValueError("Scale-feature IQRs must be positive and respect the floor")
    if not isinstance(payload["identity"], dict) or payload["identity"].get("split") != "train":
        raise ValueError("Scale-feature statistics must be calibrated exclusively on train")
    body = {key: value for key, value in payload.items() if key != "sha256"}
    if _statistics_digest(body) != payload["sha256"]:
        raise ValueError("Checkpoint scale-feature statistics digest mismatch")
    return copy.deepcopy(payload)


class NumericalResidualBranch(nn.Module):
    """Fuse scale, benchmark, and horizon-conditioned hidden features after LayerNorm."""

    def __init__(
        self, hidden_dim: int, benchmark_dim: int, mode: str, *, extended: bool = False
    ) -> None:
        super().__init__()
        if mode not in ("scales", "benchmark", "combined"):
            raise ValueError("Unknown numerical residual feature mode")
        self.mode = mode
        self.feature_count = 20 if extended else 8
        self.hidden_projection = nn.Linear(hidden_dim, 16)
        self.scale_projection = (
            nn.Sequential(nn.Linear(self.feature_count, 32), nn.GELU(), nn.Linear(32, 16))
            if mode in ("scales", "combined")
            else None
        )
        self.benchmark_projection = (
            nn.Linear(benchmark_dim, 16) if mode in ("benchmark", "combined") else None
        )
        width = 16 * (
            1 + int(self.scale_projection is not None) + int(self.benchmark_projection is not None)
        )
        self.fusion = nn.Sequential(nn.Linear(width, 32), nn.GELU(), nn.Linear(32, 3))
        # Zero only the output layer so upstream parameters can learn after the first step.
        output_layer = cast(nn.Linear, self.fusion[-1])
        nn.init.zeros_(output_layer.weight)
        nn.init.zeros_(output_layer.bias)
        self.register_buffer("feature_center", torch.zeros(self.feature_count), persistent=False)
        self.register_buffer("feature_scale", torch.ones(self.feature_count), persistent=False)
        self.statistics: dict[str, Any] | None = None

    def set_statistics(self, payload: dict[str, Any], *, require_match: bool = False) -> None:
        validated = validate_scale_feature_statistics(payload)
        if len(validated["center"]) != self.feature_count:
            raise ValueError("Scale statistics do not match this head's feature count")
        if self.scale_projection is None:
            raise ValueError("This head does not use scale features")
        if require_match and self.statistics != validated:
            raise ValueError("Checkpoint scale statistics differ from train calibration")
        self.statistics = validated
        self.feature_center.copy_(self.feature_center.new_tensor(validated["center"]))
        self.feature_scale.copy_(self.feature_scale.new_tensor(validated["scale"]))

    def forward(self, hidden: Tensor, scales: Tensor | None, benchmark: Tensor | None) -> Tensor:
        features = [self.hidden_projection(hidden.float())]
        if self.scale_projection is not None:
            if scales is None or self.statistics is None:
                raise ValueError("Scale features require checkpoint-bound train calibration")
            normalized = (_transform(scales.float()) - self.feature_center) / self.feature_scale
            encoded = self.scale_projection(
                normalized.clamp(-SCALE_STANDARDIZED_CLIP, SCALE_STANDARDIZED_CLIP)
            )
            features.append(encoded[:, None].expand(-1, hidden.shape[1], -1))
        if self.benchmark_projection is not None:
            if benchmark is None:
                raise ValueError("Benchmark-direct mode requires benchmark latents")
            encoded = self.benchmark_projection(benchmark.float().mean(dim=1))
            features.append(encoded[:, None].expand(-1, hidden.shape[1], -1))
        return self.fusion(torch.cat(features, dim=-1))
