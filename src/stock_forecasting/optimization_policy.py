"""Validation-driven warmup/plateau schedule shared by neural experiments."""

from __future__ import annotations

import math


class ValidationPlateauScheduler:
    """Drop LR before early stopping and preserve reductions across resumptions."""

    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        *,
        patience: int = 2,
        factor: float = 0.3,
        min_ratio: float = 0.09,
        low_lr_evaluations: int = 2,
    ):
        self.optimizer = optimizer
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        self.warmup_steps = warmup_steps
        self.patience, self.factor, self.min_ratio = patience, factor, min_ratio
        self.required_low_evaluations = low_lr_evaluations
        self.last_epoch = 0
        self.ratio = 1.0
        self.best = None
        self.stale = 0
        self.low_evaluations = 0
        self._apply()

    def _apply(self):
        warmup = min(1.0, (self.last_epoch + 1) / max(1, self.warmup_steps))
        for group, base in zip(self.optimizer.param_groups, self.base_lrs, strict=True):
            group["lr"] = base * self.ratio * warmup

    def step(self):
        self.last_epoch += 1
        self._apply()

    def observe(self, value: float, min_delta: float = 0.0):
        if not math.isfinite(value):
            raise ValueError("Plateau scheduler requires a finite validation loss")
        if self.ratio <= self.min_ratio * (1 + 1e-8) and self.last_epoch >= self.warmup_steps:
            self.low_evaluations += 1
        if self.best is None or value < self.best - min_delta:
            self.best, self.stale = value, 0
        elif self.last_epoch >= self.warmup_steps:
            self.stale += 1
            if self.stale >= self.patience:
                self.ratio = max(self.min_ratio, self.ratio * self.factor)
                self.stale = 0
                self._apply()

    @property
    def permits_early_stop(self):
        return self.low_evaluations >= self.required_low_evaluations

    def get_last_lr(self):
        return [group["lr"] for group in self.optimizer.param_groups]

    def state_dict(self):
        return {key: value for key, value in vars(self).items() if key != "optimizer"}

    def load_state_dict(self, state):
        expected = set(vars(self)) - {"optimizer"}
        if set(state) != expected:
            raise ValueError("Incompatible validation scheduler checkpoint")
        self.__dict__.update(state)
        self._apply()

    def realign(self, step: int, warmup_steps: int):
        self.last_epoch, self.warmup_steps = step, warmup_steps
        self._apply()
