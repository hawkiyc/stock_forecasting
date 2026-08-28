"""Shared holding-day horizon contract for alpha forecasting."""

from __future__ import annotations

from collections.abc import Iterable

MIN_H_START = 1
MAX_H_START = 3
DEFAULT_H_START = 3
MAX_ALPHA_HORIZON = 14


def alpha_horizons_from_start(h_start: int) -> tuple[int, ...]:
    """Return the contiguous holding-day horizons for an approved start."""

    if isinstance(h_start, bool) or not isinstance(h_start, int):
        raise ValueError("h_start must be an integer from 1 through 3")
    if not MIN_H_START <= h_start <= MAX_H_START:
        raise ValueError("h_start must be an integer from 1 through 3")
    return tuple(range(h_start, MAX_ALPHA_HORIZON + 1))


def validate_alpha_horizons(horizons: Iterable[int]) -> tuple[int, ...]:
    """Require a contiguous h_start-through-14 alpha horizon sequence."""

    ordered = tuple(int(horizon) for horizon in horizons)
    if not ordered:
        raise ValueError("alpha_horizons cannot be empty")
    try:
        expected = alpha_horizons_from_start(ordered[0])
    except ValueError as error:
        raise ValueError(
            "alpha_horizons must start at trading day 1, 2, or 3 and end at day 14"
        ) from error
    if ordered != expected:
        raise ValueError(
            "alpha_horizons must be contiguous from h_start through trading day 14"
        )
    return ordered


DEFAULT_ALPHA_HORIZONS = alpha_horizons_from_start(DEFAULT_H_START)
