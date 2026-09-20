"""Small same-date, same-market auxiliary ranking objective."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

MARKET_IDS = {"US": 0, "TWSE": 1, "TPEX": 2}


def market_ids(values, device) -> Tensor:
    return torch.tensor([MARKET_IDS.get(str(v).upper(), 3) for v in values], device=device)


def ranking_groups(dates, markets, symbols, device) -> tuple[Tensor, Tensor]:
    groups, securities = {}, {}
    group_ids, security_ids = [], []
    for day, market, symbol in zip(dates, markets, symbols, strict=True):
        group_ids.append(groups.setdefault((str(day), str(market)), len(groups)))
        security_ids.append(securities.setdefault(str(symbol), len(securities)))
    return torch.tensor(group_ids, device=device), torch.tensor(security_ids, device=device)


def same_date_ranking_loss(
    predictions: Tensor,
    targets: Tensor,
    scales: Tensor,
    groups: Tensor,
    securities: Tensor,
    max_pairs: int,
) -> Tensor:
    """Never rank different dates/markets or duplicate padded copies of one stock."""
    candidates = torch.arange(len(targets), device=targets.device)
    if len(candidates) > 512:
        candidates = candidates[torch.randperm(len(candidates), device=targets.device)[:512]]
    pairs = candidates[
        torch.triu_indices(len(candidates), len(candidates), offset=1, device=targets.device)
    ]
    valid = (groups[pairs[0]] == groups[pairs[1]]) & (securities[pairs[0]] != securities[pairs[1]])
    pairs = pairs[:, valid]
    if pairs.shape[1] == 0:
        return predictions.sum() * 0
    if pairs.shape[1] > max_pairs:
        pairs = pairs[:, torch.randperm(pairs.shape[1], device=pairs.device)[:max_pairs]]
    left, right = pairs
    delta = (targets[left].float() - targets[right].float()) / scales
    gap = (predictions[left, :, 1].float() - predictions[right, :, 1].float()) / scales
    valid = torch.isfinite(delta) & (delta.abs() > 0.01)
    losses = F.softplus(-delta.nan_to_num().sign() * gap)
    return losses.masked_fill(~valid, 0).sum() / valid.sum().clamp_min(1)
